"""Tools for data loading, inspection, and preparation."""

from typing import Any, Dict, List
import nltk
import pandas as pd
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

    return [inspect_data_tool, tokenize_tool]
