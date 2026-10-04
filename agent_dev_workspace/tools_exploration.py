"""Exploration tools: document cosine similarity and feature-vs-feature correlation matrix."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Optional
from langchain_core.tools import tool
from sklearn.metrics.pairwise import cosine_similarity


def make_tools(session):
    """Factory function returning exploration tools initialized with the shared session."""

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int,
        doc_text_1: Optional[str] = None,
        doc_text_2: Optional[str] = None
    ) -> dict:
        """Computes cosine similarity between two documents.

        Can compute similarity using indices of existing documents in session.dataframe / feature_matrix,
        or dynamically vectorize doc_text_1 / doc_text_2 using session.artifacts["count_vectorizer"].

        Args:
            doc_index_1: Index of the first document (0-based row index in session.feature_matrix).
            doc_index_2: Index of the second document (0-based row index in session.feature_matrix).
            doc_text_1: Optional raw text for document 1.
            doc_text_2: Optional raw text for document 2.

        Returns:
            Dict containing result_id, doc_index_1, doc_index_2, and similarity_score.
        """
        if doc_text_1 is not None and doc_text_2 is not None:
            # Custom text input mode using fitted vectorizer
            vectorizer = session.artifacts.get("count_vectorizer")
            if vectorizer is None:
                return {"status": "error", "message": "count_vectorizer missing in session.artifacts. Run build_dtm_tool first."}

            vec1 = vectorizer.transform([doc_text_1])
            vec2 = vectorizer.transform([doc_text_2])
            sim = float(cosine_similarity(vec1, vec2)[0, 0])
            if np.isnan(sim):
                sim = 0.0

        else:
            # Document index mode using session.feature_matrix
            if session.feature_matrix is None:
                return {"status": "error", "message": "No feature_matrix found in session. Run build_dtm_tool first."}

            n_docs = session.feature_matrix.shape[0]
            if doc_index_1 < 0 or doc_index_1 >= n_docs or doc_index_2 < 0 or doc_index_2 >= n_docs:
                return {"status": "error", "message": f"Document index out of bounds. Valid range: 0 to {n_docs - 1}."}

            row1 = session.feature_matrix[doc_index_1]
            row2 = session.feature_matrix[doc_index_2]

            sim = float(cosine_similarity(row1, row2)[0, 0])
            if np.isnan(sim):
                sim = 0.0

        result_id = session.next_result_id("cosine_similarity")

        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "similarity_score": round(sim, 4)
        }

        session.store_result(
            tool_name="cosine_similarity_tool",
            args={"doc_index_1": doc_index_1, "doc_index_2": doc_index_2},
            summary=summary
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(top_n: int = 20) -> dict:
        """Computes Pearson correlation matrix over top-N variance terms and plots an annotated heatmap.

        Operates on the global DTM (session.feature_matrix / feature_names). Selects top_n terms
        by variance, calculates Pearson correlation matrix (np.corrcoef), renders a seaborn heatmap,
        attaches it to session.pending_figure, and returns the correlation values.

        Args:
            top_n: Number of top variance terms to include (default 20).

        Returns:
            Dict containing result_id, terms, correlation_matrix, and top_n.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "DTM or feature_names missing in session. Run build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        # 1. Compute term variances across documents
        variances = np.var(dense_matrix, axis=0)

        # Sort terms by variance descending
        sorted_indices = np.argsort(-variances)
        
        # Take top_n terms (fewer than top_n terms in vocabulary means take all)
        n_terms = min(top_n, len(feature_names))
        top_indices = sorted_indices[:n_terms]

        top_terms = feature_names[top_indices].tolist()
        top_matrix = dense_matrix[:, top_indices]

        # 2. Compute Pearson correlation matrix across top variance columns
        # top_matrix has shape (n_documents, n_terms); np.corrcoef expects variables as rows, so transpose top_matrix
        corr_matrix = np.corrcoef(top_matrix.T)

        # Handle edge case where a term has 0 variance (yielding NaN in correlation)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        corr_list = corr_matrix.tolist()

        # 3. Build annotated heatmap using seaborn
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            xticklabels=top_terms,
            yticklabels=top_terms,
            ax=ax,
            vmin=-1.0,
            vmax=1.0
        )
        ax.set_title(f"Feature Correlation Matrix (Top {n_terms} Terms by Variance)")
        fig.tight_layout()

        # Attach to session.pending_figure
        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")

        summary = {
            "result_id": result_id,
            "top_n": n_terms,
            "terms": top_terms,
            "correlation_matrix": corr_list
        }

        session.store_result(
            tool_name="feature_correlation_matrix_tool",
            args={"top_n": top_n},
            summary=summary
        )

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
