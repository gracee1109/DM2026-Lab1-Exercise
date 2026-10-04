"""Dimensionality reduction and label binarization tools."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Literal, Optional
from langchain_core.tools import tool
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap


def make_tools(session):
    """Factory function returning reduction tools initialized with the shared session."""

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"],
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42
    ) -> dict:
        """Reduces document-term matrix dimensions to lower dimensional coordinates and plots them.

        Performs PCA, t-SNE, or UMAP dimension reduction on session.feature_matrix.
        Generates a scatter plot colored by category/label and attaches it to session.pending_figure.

        Args:
            method: Reduction algorithm ('pca', 'tsne', or 'umap').
            n_components: Target dimensions, typically 2 for plotting (default 2).
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors parameter for UMAP (default 15).
            random_state: Random state seed for reproducibility (default 42).

        Returns:
            Dict containing result_id, method, coordinates, and explained_variance_ratio (for PCA).
        """
        if session.feature_matrix is None:
            return {"status": "error", "message": "No document-term matrix found in session. Run build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        n_samples, n_features = dense_matrix.shape

        explained_variance_ratio = None

        # Perform reduction based on selected method
        if method == "pca":
            pca = PCA(n_components=n_components, random_state=random_state)
            coords = pca.fit_transform(dense_matrix)
            explained_variance_ratio = [float(val) for val in pca.explained_variance_ratio_]

        elif method == "tsne":
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state
            )
            coords = tsne.fit_transform(dense_matrix)

        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state
            )
            coords = reducer.fit_transform(dense_matrix)

        coords_list = coords.tolist()

        # Build scatter plot colored by category / label
        fig, ax = plt.subplots(figsize=(8, 6))

        if session.labels is not None:
            labels_arr = np.array(session.labels)
            unique_labels = np.unique(labels_arr)
            for label in unique_labels:
                idx = (labels_arr == label)
                ax.scatter(coords[idx, 0], coords[idx, 1], label=str(label), alpha=0.7)
            ax.legend(title="Category")
        else:
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.7)

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        fig.tight_layout()

        # Set figure on session.pending_figure (plotting requirement)
        session.pending_figure = fig

        result_id = session.next_result_id("reduce_dimensions")
        
        # Store reduced coordinates in session.artifacts for future pipeline reference
        session.artifacts[f"reduced_coords_{method}"] = coords_list

        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "n_samples": n_samples,
            "coordinates": coords_list,
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            tool_name="reduce_dimensions_tool",
            args={
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state
            },
            summary=summary
        )

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encodes category labels in session.dataframe.

        This tool has no fixed contract or known answer in the specification.
        Dynamically checks for 'category_name' or 'label' columns in session.dataframe,
        creates one-hot dummy variables, and returns the result summary.

        Returns:
            Dict containing result_id, categories, and shape of one-hot encoded labels.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataframe found in session."}

        target_col = None
        if "category_name" in session.dataframe.columns:
            target_col = "category_name"
        elif "label" in session.dataframe.columns:
            target_col = "label"
        else:
            return {"status": "error", "message": "Neither 'category_name' nor 'label' column found in dataframe."}

        # Perform one-hot encoding on target_col
        one_hot = pd.get_dummies(session.dataframe[target_col], prefix="cat", dtype=int)
        
        result_id = session.next_result_id("binarize_labels")

        summary = {
            "result_id": result_id,
            "target_column": target_col,
            "categories": one_hot.columns.tolist(),
            "n_rows": len(one_hot),
            "n_categories": len(one_hot.columns),
            "one_hot_matrix": one_hot.values.tolist()
        }

        session.artifacts["binarized_labels"] = one_hot

        session.store_result(
            tool_name="binarize_labels_tool",
            args={},
            summary=summary,
            full_report=one_hot
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
