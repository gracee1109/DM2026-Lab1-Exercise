"""Filtering tools for feature selection based on variance and label correlation."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from langchain_core.tools import tool


def make_tools(session):
    """Factory function returning filtering tools initialized with the shared session."""

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Flags terms in the document-term matrix by variance for feature selection.

        Calculates variance across all documents for each term in the session feature matrix.
        Returns terms above the specified threshold in kept_terms and terms at or below
        the threshold in removed_terms. Stores a full report in session results.

        Args:
            threshold: Minimum variance threshold for keeping a feature (default 0.0).

        Returns:
            Dict containing result_id, total_terms, kept_terms, removed_terms, threshold.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix found in session. Run build_dtm_tool first."}

        # Convert sparse matrix to dense array for variance calculation
        dense_matrix = session.feature_matrix.toarray()
        
        # Calculate variance for each term across documents (axis 0)
        variances = np.var(dense_matrix, axis=0)

        # Build report dataframe
        df_report = pd.DataFrame({
            "term": session.feature_names,
            "variance": variances
        })
        
        # Sort descending by variance
        df_report = df_report.sort_values(by="variance", ascending=False).reset_index(drop=True)

        kept_terms = df_report[df_report["variance"] > threshold]["term"].tolist()
        removed_terms = df_report[df_report["variance"] <= threshold]["term"].tolist()

        result_id = session.next_result_id("variance_filter")
        
        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        # full_report DataFrame has term as 1st col, metric as 2nd col, sorted for visualize_result_tool
        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Computes Pearson linear correlation between each term count and a target binary class label.

        Calculates Pearson correlation coefficient r for each feature against a one-vs-rest binary label
        for the specified target class.

        Args:
            target_class: The target category name to correlate against (one-vs-rest).
            threshold: Minimum absolute correlation coefficient |r| to keep a term (default 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, kept_terms, removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"status": "error", "message": "DTM or labels missing in session. Run build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        
        # Ensure session.labels is a numpy array to avoid AttributeError if it is a list
        labels_arr = np.array(session.labels)
        
        # One-vs-rest binary target vector (1 for target_class, 0 for other classes)
        binary_target = (labels_arr == target_class).astype(float)

        pearson_rs = []
        for j in range(dense_matrix.shape[1]):
            term_counts = dense_matrix[:, j]
            # Calculate Pearson correlation coefficient
            std_term = np.std(term_counts)
            std_target = np.std(binary_target)
            if std_term == 0 or std_target == 0:
                r = 0.0
            else:
                r = float(np.corrcoef(term_counts, binary_target)[0, 1])
                if np.isnan(r):
                    r = 0.0
            pearson_rs.append(r)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "pearson_r": pearson_rs
        })
        
        # Sort descending by absolute Pearson r value
        df_report = df_report.sort_values(by="pearson_r", key=abs, ascending=False).reset_index(drop=True)

        kept_terms = df_report[df_report["pearson_r"].abs() >= threshold]["term"].tolist()
        removed_terms = df_report[df_report["pearson_r"].abs() < threshold]["term"].tolist()

        result_id = session.next_result_id("pearson_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(target_class: str, threshold: float = 0.0) -> dict:
        """Computes Spearman rank correlation between each term count and a target binary class label.

        Calculates Spearman rank correlation coefficient r for each feature against a one-vs-rest binary label
        for the specified target class. Useful for non-linear, monotonic relationships.

        Args:
            target_class: The target category name to correlate against (one-vs-rest).
            threshold: Minimum absolute Spearman correlation coefficient |r| to keep a term (default 0.0).

        Returns:
            Dict containing result_id, target_class, threshold, kept_terms, removed_terms.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"status": "error", "message": "DTM or labels missing in session. Run build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        
        # Ensure session.labels is a numpy array to avoid AttributeError if it is a list
        labels_arr = np.array(session.labels)
        binary_target = (labels_arr == target_class).astype(float)

        spearman_rs = []
        for j in range(dense_matrix.shape[1]):
            term_counts = dense_matrix[:, j]
            # scipy.stats.spearmanr calculation
            if np.std(term_counts) == 0 or np.std(binary_target) == 0:
                r = 0.0
            else:
                res = spearmanr(term_counts, binary_target)
                r = float(res.statistic) if hasattr(res, 'statistic') else float(res[0])
                if np.isnan(r):
                    r = 0.0
            spearman_rs.append(r)

        df_report = pd.DataFrame({
            "term": session.feature_names,
            "spearman_r": spearman_rs
        })

        df_report = df_report.sort_values(by="spearman_r", key=abs, ascending=False).reset_index(drop=True)

        kept_terms = df_report[df_report["spearman_r"].abs() >= threshold]["term"].tolist()
        removed_terms = df_report[df_report["spearman_r"].abs() < threshold]["term"].tolist()

        result_id = session.next_result_id("spearman_filter")

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(session.feature_names),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
