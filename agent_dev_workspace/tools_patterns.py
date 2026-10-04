"""Pattern mining tool using PAMI for frequent pattern discovery per category."""

import os
import tempfile
import pandas as pd
from typing import Optional, Literal
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def make_tools(session):
    """Factory function returning pattern mining tools initialized with the shared session."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"],
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> dict:
        """Mines frequent itemsets/patterns for a specific category using PAMI algorithms.

        Filters session.dataframe to category_name, builds a fresh category-local DTM with
        CountVectorizer(stop_words='english'), applies vocabulary filtering (variance, tfidf,
        or term_frequency), converts surviving terms to a transactional database, and mines
        patterns using FPGrowth, Top-K (FAE), or MaxFPGrowth.

        Args:
            category_name: Which category/label to filter and mine.
            filtering_method: Method to filter terms ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support count (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top k patterns count (required for 'topk').

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm, patterns list.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe found in session. Load dataset first."}

        # Step 1: Filter dataframe to category_name
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        if df_cat.empty:
            return {"status": "error", "message": f"No documents found for category '{category_name}'."}

        # Remove empty or whitespace-only text rows before vectorizing
        valid_texts = df_cat["text"].dropna().astype(str)
        valid_texts = valid_texts[valid_texts.str.strip() != ""]

        if valid_texts.empty:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no valid text found in category"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Build fresh, category-local DTM with stop_words='english'
        vectorizer = CountVectorizer(stop_words="english")
        try:
            dtm_sparse = vectorizer.fit_transform(valid_texts)
        except ValueError:
            # e.g., empty vocabulary after stopword removal
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "empty vocabulary after stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = vectorizer.get_feature_names_out()
        dtm_dense = dtm_sparse.toarray()

        if len(feature_names) == 0:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Step 2: Apply chosen vocabulary filtering method
        kept_indices = []
        if filtering_method == "variance":
            variances = pd.Series(dtm_dense.var(axis=0))
            p5 = variances.quantile(0.05)
            p95 = variances.quantile(0.95)
            kept_mask = (variances > p5) & (variances < p95)
            kept_indices = kept_mask[kept_mask].index.tolist()

        elif filtering_method == "term_frequency":
            counts = pd.Series(dtm_dense.sum(axis=0))
            p5 = counts.quantile(0.05)
            p95 = counts.quantile(0.95)
            kept_mask = (counts > p5) & (counts < p95)
            kept_indices = kept_mask[kept_mask].index.tolist()

        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            tfidf_matrix = transformer.fit_transform(dtm_sparse).toarray()
            mean_tfidf = pd.Series(tfidf_matrix.mean(axis=0))
            p20 = mean_tfidf.quantile(0.20)
            kept_mask = mean_tfidf >= p20
            kept_indices = kept_mask[kept_mask].index.tolist()

        # Step 3: Handle zero surviving terms cleanly
        if not kept_indices:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        surviving_terms = [feature_names[i] for i in kept_indices]
        filtered_dtm = dtm_dense[:, kept_indices]

        # Step 4: Convert surviving terms into transactional format
        # One transaction per document, items are terms present with count >= 1
        df_transactions = pd.DataFrame(filtered_dtm, columns=surviving_terms)
        transaction_lines = []
        for _, row in df_transactions.iterrows():
            items = [col for col in surviving_terms if row[col] >= 1]
            if items:
                transaction_lines.append("\t".join(items))

        if not transaction_lines:
            result_id = session.next_result_id("mine_patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "patterns": [],
                "note": "no non-empty transactions generated"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # Write transaction database directly to file (PAMI transactional database format)
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "transactions.db")
            with open(db_file, "w", encoding="utf-8") as f:
                for line in transaction_lines:
                    f.write(line + "\n")

            # Mine patterns using selected PAMI algorithm
            patterns_list = []

            if algorithm == "fpgrowth":
                if min_sup is None:
                    return {"status": "error", "message": "min_sup argument is required for algorithm 'fpgrowth'."}
                from PAMI.frequentPattern.basic import FPGrowth
                obj = FPGrowth.FPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        p_str = row["Patterns"]
                        p_items = p_str.strip().split("\t") if isinstance(p_str, str) else list(p_str)
                        sup = int(row["Support"])
                        patterns_list.append({"pattern": p_items, "support": sup})

            elif algorithm == "topk":
                if k is None:
                    return {"status": "error", "message": "k argument is required for algorithm 'topk'."}
                from PAMI.frequentPattern.topk import FAE
                obj = FAE.FAE(iFile=db_file, k=k, sep="\t")
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        p_str = row["Patterns"]
                        p_items = p_str.strip().split("\t") if isinstance(p_str, str) else list(p_str)
                        sup = int(row["Support"])
                        patterns_list.append({"pattern": p_items, "support": sup})

            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    return {"status": "error", "message": "min_sup argument is required for algorithm 'maxfpgrowth'."}
                from PAMI.frequentPattern.maximal import MaxFPGrowth
                obj = MaxFPGrowth.MaxFPGrowth(iFile=db_file, minSup=min_sup, sep="\t")
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        p_str = row["Patterns"]
                        p_items = p_str.strip().split("\t") if isinstance(p_str, str) else list(p_str)
                        sup = int(row["Support"])
                        patterns_list.append({"pattern": p_items, "support": sup})

        result_id = session.next_result_id("mine_patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns_list),
            "patterns": patterns_list
        }

        # Store pattern list in session.artifacts
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)

        return summary

    return [mine_patterns_tool]
