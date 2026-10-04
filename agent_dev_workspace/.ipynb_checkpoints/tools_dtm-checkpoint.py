"""Document-Term Matrix (DTM) construction tool.

Provides build_dtm_tool to construct scipy sparse feature matrix and feature names
from session.dataframe text.
"""

from typing import Any, Dict, List, Optional
import numpy as np
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

        Fits CountVectorizer on text, storing feature_matrix, feature_names,
        vectorizer artifact, and reports matrix dimensions and sparsity.

        Args:
            max_features: Maximum number of terms to keep ordered by term frequency.
            min_df: Ignore terms with document frequency strictly lower than threshold.
            max_df: Ignore terms with document frequency strictly higher than threshold.
            ngram_range: Lower and upper boundary of range of n-values for n-grams (e.g. [1, 1]).
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

    return [build_dtm_tool]