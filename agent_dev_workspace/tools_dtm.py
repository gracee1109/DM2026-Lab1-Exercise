"""Document-Term Matrix (DTM) construction and exploration tools.

Provides build_dtm_tool, term_frequency_tool, and dtm_heatmap_tool.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session: Any) -> List[Any]:
    """Factory function to build and return DTM tools bound to the given session."""

    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        min_df: float = 1,
        max_df: float = 1.0,
        ngram_range: List[int] = [1, 1],
        stop_words: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build a Document-Term Matrix (DTM) from text in session.dataframe.

        Args:
            max_features: Maximum number of terms to keep.
            min_df: Ignore terms with document frequency strictly lower than threshold.
            max_df: Ignore terms with document frequency strictly higher than threshold.
            ngram_range: Lower and upper boundary of n-values for n-grams.
            stop_words: Stop words setting (e.g., 'english' or None).

        Returns:
            Dict containing result_id, n_documents, n_terms, non_zero, total_elements,
            and sparsity_pct.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {
                "status": "error",
                "message": "No dataset loaded in session. Call load_dataset_tool first.",
            }

        text_col = "text"
        if text_col not in session.dataframe.columns:
            return {
                "status": "error",
                "message": f"Column '{text_col}' not found in dataframe.",
            }

        corpus = session.dataframe[text_col].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=tuple(ngram_range),
            stop_words=stop_words,
        )

        dtm = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].tolist())
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].tolist())

        n_rows, n_cols = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = int(n_rows * n_cols)
        sparsity_pct = (
            float(100.0 * (1.0 - non_zero / total_elements))
            if total_elements > 0
            else 0.0
        )

        result_id = session.next_result_id("build_dtm")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "feature_names": feature_names,
        }

        session.store_result("build_dtm_tool", {
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "ngram_range": ngram_range,
            "stop_words": stop_words,
        }, summary)

        return summary

    @tool
    def term_frequency_tool(top_n: int = 20) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in session.feature_matrix.

        Args:
            top_n: Number of highest-frequency terms to return.

        Returns:
            Dict containing top term frequencies and total term count.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "DTM not built. Call build_dtm_tool first.",
            }

        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        terms = session.feature_names

        df_freq = pd.DataFrame({"term": terms, "frequency": counts})
        df_freq.sort_values(by=["frequency", "term"], ascending=[False, True], inplace=True)

        top_terms = df_freq.head(top_n).to_dict(orient="records")

        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_terms": len(terms),
            "top_terms": top_terms,
        }

        session.store_result(
            "term_frequency_tool",
            {"top_n": top_n},
            summary,
            full_report=df_freq,
        )

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Generate a heatmap of a positional slice of the global DTM.

        Args:
            n_terms: Number of terms (columns) to include in the slice.
            n_documents: Number of documents (rows) to include in the slice.

        Returns:
            Dict containing matrix slice, term labels, and document labels.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "DTM not built. Call build_dtm_tool first.",
            }

        rows = min(n_documents, session.feature_matrix.shape[0])
        cols = min(n_terms, session.feature_matrix.shape[1])

        sub_matrix = session.feature_matrix[:rows, :cols].toarray()
        term_labels = session.feature_names[:cols]
        doc_labels = [f"Doc_{i}" for i in range(rows)]

        fig, ax = plt.subplots(figsize=(max(6, cols * 0.5), max(4, rows * 0.3)))
        sns.heatmap(
            sub_matrix,
            annot=True,
            fmt="d",
            xticklabels=term_labels,
            yticklabels=doc_labels,
            cmap="Blues",
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap ({rows} Docs x {cols} Terms)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_documents_sliced": rows,
            "n_terms_sliced": cols,
            "matrix_slice": sub_matrix.tolist(),
            "terms": term_labels,
            "documents": doc_labels,
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
