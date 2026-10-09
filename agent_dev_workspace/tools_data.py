"""Tools for data loading, inspection, and preparation."""

import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

# Ensure punkt is available for nltk word tokenization
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def make_tools(session):
    """Factory that creates data manipulation and inspection tools bound to session."""

    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and directories in the workspace or a specified subdirectory.

        Args:
            subdirectory: Subdirectory path relative to workspace root (default ".").

        Returns:
            Dict containing result_id, subdirectory, directories list, and files list.
        """
        target_path = subdirectory
        if not os.path.exists(target_path):
            result_id = session.next_result_id("list_files")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": f"Path '{subdirectory}' does not exist.",
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary

        directories = []
        files = []

        try:
            entries = os.listdir(target_path)
            for entry in sorted(entries):
                if entry.startswith("."):
                    continue
                full_entry = os.path.join(target_path, entry)
                if os.path.isdir(full_entry):
                    directories.append(entry)
                else:
                    files.append(entry)
        except Exception as e:
            result_id = session.next_result_id("list_files")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": str(e),
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files,
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5) -> Dict[str, Any]:
        """Inspect the top n rows of the working DataFrame alongside column names and shape.

        Args:
            n: Number of initial rows to display from the working DataFrame. Defaults to 5.

        Returns:
            Dict containing result_id, columns list, dataset shape [rows, cols],
            n_rows_shown, and the preview rows as a list of records.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("inspect_data")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No dataset currently loaded in session. Load a dataset first.",
            }
            session.store_result("inspect_data_tool", {"n": n}, summary)
            return summary

        result_id = session.next_result_id("inspect_data")
        df = session.dataframe
        head_df = df.head(n)

        rows = head_df.to_dict(orient="records")

        summary = {
            "result_id": result_id,
            "status": "success",
            "columns": list(df.columns),
            "shape": [int(df.shape[0]), int(df.shape[1])],
            "n_rows_shown": len(rows),
            "rows": rows,
        }

        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def check_missing_tool(column: str = "text") -> Dict[str, Any]:
        """Checks for missing or empty text values in the specified column of session.dataframe.

        Args:
            column: Name of the text column to inspect for missing/empty values (default "text").

        Returns:
            Dict containing result_id, total_rows, n_missing, missing_indices, and missing_pct.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("check_missing")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No dataset currently loaded in session. Load a dataset first.",
            }
            session.store_result("check_missing_tool", {"column": column}, summary)
            return summary

        if column not in session.dataframe.columns:
            result_id = session.next_result_id("check_missing")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": f"Column '{column}' not found in working DataFrame.",
            }
            session.store_result("check_missing_tool", {"column": column}, summary)
            return summary

        result_id = session.next_result_id("check_missing")
        series = session.dataframe[column]

        missing_mask = series.isna() | series.astype(str).str.strip().eq("")
        missing_indices = series[missing_mask].index.tolist()

        total_rows = len(series)
        n_missing = len(missing_indices)
        missing_pct = round(100.0 * n_missing / total_rows, 4) if total_rows > 0 else 0.0

        summary = {
            "result_id": result_id,
            "status": "success",
            "column": column,
            "total_rows": total_rows,
            "n_missing": n_missing,
            "missing_indices": missing_indices,
            "missing_pct": missing_pct,
        }

        session.store_result("check_missing_tool", {"column": column}, summary)
        return summary

    @tool
    def check_duplicates_tool(column: str = "text", drop: bool = False) -> Dict[str, Any]:
        """Checks for and optionally drops duplicate document rows in session.dataframe.

        When drop=True, uses keep=False to drop every copy of a duplicated row, matching
        X.drop_duplicates(keep=False, inplace=True).

        Args:
            column: Column name to check for duplicate content (default "text").
            drop: Whether to drop duplicate rows from session.dataframe (default False).

        Returns:
            Dict containing result_id, n_duplicates, n_rows_before, n_rows_after, and dropped.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("check_duplicates")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No dataset currently loaded in session. Load a dataset first.",
            }
            session.store_result("check_duplicates_tool", {"column": column, "drop": drop}, summary)
            return summary

        if column not in session.dataframe.columns:
            result_id = session.next_result_id("check_duplicates")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": f"Column '{column}' not found in working DataFrame.",
            }
            session.store_result("check_duplicates_tool", {"column": column, "drop": drop}, summary)
            return summary

        result_id = session.next_result_id("check_duplicates")
        df = session.dataframe
        n_rows_before = len(df)

        dup_mask_extra = df.duplicated(subset=[column], keep="first")
        n_duplicates = int(dup_mask_extra.sum())

        if drop:
            clean_df = df.drop_duplicates(subset=[column], keep=False).reset_index(drop=True)
            session.dataframe = clean_df
            n_rows_after = len(clean_df)
        else:
            n_rows_after = n_rows_before

        summary = {
            "result_id": result_id,
            "status": "success",
            "column": column,
            "n_duplicates": n_duplicates,
            "n_rows_before": n_rows_before,
            "n_rows_after": n_rows_after,
            "dropped": drop,
        }

        session.store_result("check_duplicates_tool", {"column": column, "drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: Optional[int] = 42) -> Dict[str, Any]:
        """Subsamples n rows from session.dataframe.

        Args:
            n: Number of rows to sample (default 5).
            random_state: Seed for reproducible sampling (default 42).

        Returns:
            Dict containing result_id, sampled_indices, n_sampled, and sample_rows.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("sample_data")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No dataset currently loaded in session. Load a dataset first.",
            }
            session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
            return summary

        df = session.dataframe
        sample_n = min(n, len(df))
        sampled_df = df.sample(n=sample_n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_requested": n,
            "n_sampled": len(sampled_df),
            "random_state": random_state,
            "sampled_indices": sampled_indices,
            "rows": sampled_df.to_dict(orient="records"),
        }

        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool(column: str = "text") -> Dict[str, Any]:
        """Calculates text length statistics and generates a boxplot by category.

        Computes character length df[column].apply(len), returns overall and per-category
        summary statistics, and builds a Seaborn boxplot set on session.pending_figure.

        Args:
            column: Name of the text column to analyze (default "text").

        Returns:
            Dict containing result_id, overall_stats, and category_stats.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("describe_data")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No dataset currently loaded in session. Load a dataset first.",
            }
            session.store_result("describe_data_tool", {"column": column}, summary)
            return summary

        if column not in session.dataframe.columns:
            result_id = session.next_result_id("describe_data")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": f"Column '{column}' not found in working DataFrame.",
            }
            session.store_result("describe_data_tool", {"column": column}, summary)
            return summary

        df = session.dataframe.copy()
        
        # Calculate character length
        df["text_length"] = df[column].astype(str).fillna("").apply(len)

        overall_stats = df["text_length"].describe().to_dict()

        # Categorical column identification
        cat_col = None
        if "category_name" in df.columns:
            cat_col = "category_name"
        elif "label" in df.columns:
            cat_col = "label"

        category_stats = {}
        if cat_col:
            grouped = df.groupby(cat_col)["text_length"]
            for name, group in grouped:
                category_stats[str(name)] = group.describe().to_dict()

            # Plot boxplot grouped by category
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.boxplot(data=df, x=cat_col, y="text_length", ax=ax)
            ax.set_title(f"Text Length Distribution by {cat_col}")
            ax.set_xlabel("Category")
            ax.set_ylabel("Text Length (characters)")
            fig.tight_layout()
            session.pending_figure = fig

        result_id = session.next_result_id("describe_data")

        summary = {
            "result_id": result_id,
            "status": "success",
            "column": column,
            "overall_stats": overall_stats,
            "category_stats": category_stats,
        }

        session.store_result("describe_data_tool", {"column": column}, summary)
        return summary

    @tool
    def tokenize_tool(column: str = "text") -> Dict[str, Any]:
        """Tokenize document text into unigram token lists, handling lowercasing and missing values.

        Args:
            column: Name of the text column in session.dataframe to tokenize. Defaults to "text".

        Returns:
            Dict containing result_id, total_tokens, unique_vocab_size, avg_tokens_per_doc,
            and sample token lists.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("tokenize")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": "No dataset currently loaded in session. Load a dataset first.",
            }
            session.store_result("tokenize_tool", {"column": column}, summary)
            return summary

        if column not in session.dataframe.columns:
            result_id = session.next_result_id("tokenize")
            summary = {
                "result_id": result_id,
                "status": "error",
                "message": f"Column '{column}' not found in working DataFrame.",
            }
            session.store_result("tokenize_tool", {"column": column}, summary)
            return summary

        result_id = session.next_result_id("tokenize")
        df = session.dataframe

        def safe_tokenize(val: Any) -> List[str]:
            if pd.isna(val) or val is None:
                return []
            text_str = str(val).strip().lower()
            if not text_str:
                return []
            return nltk.word_tokenize(text_str)

        tokenized_lists = df[column].apply(safe_tokenize).tolist()

        # Store in dataframe and artifacts
        df["unigrams"] = tokenized_lists
        session.artifacts["unigrams"] = tokenized_lists

        # Statistics computation
        total_tokens = sum(len(doc) for doc in tokenized_lists)
        all_vocab = {tok for doc in tokenized_lists for tok in doc}
        unique_vocab_size = len(all_vocab)
        avg_tokens = total_tokens / len(tokenized_lists) if tokenized_lists else 0.0

        summary = {
            "result_id": result_id,
            "status": "success",
            "total_documents": len(tokenized_lists),
            "total_tokens": total_tokens,
            "unique_vocab_size": unique_vocab_size,
            "avg_tokens_per_doc": round(avg_tokens, 4),
            "sample_tokens": tokenized_lists[:3],
        }

        session.store_result("tokenize_tool", {"column": column}, summary)
        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        tokenize_tool,
    ]
