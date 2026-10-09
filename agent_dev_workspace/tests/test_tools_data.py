"""Tests for tools_data.py tools."""

import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


@pytest.fixture
def loaded_session():
    """Provides a fresh SessionState pre-loaded with sample_fixture.csv for each test."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )
    return session


def test_inspect_data_tool_default(loaded_session):
    data_tools = make_tools(loaded_session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    result = inspect_data_tool.invoke({})

    assert result["status"] == "success"
    assert result["shape"] == [8, 3]
    assert result["n_rows_shown"] == 5
    assert len(result["rows"]) == 5
    assert "text" in result["columns"]
    assert "category" in result["columns"]
    assert "category_name" in result["columns"]


def test_inspect_data_tool_custom_n(loaded_session):
    data_tools = make_tools(loaded_session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    result = inspect_data_tool.invoke({"n": 2})

    assert result["status"] == "success"
    assert result["n_rows_shown"] == 2
    assert len(result["rows"]) == 2


def test_inspect_data_tool_edge_cases(loaded_session):
    data_tools = make_tools(loaded_session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # n = 0 (returns empty row list, maintains column structure)
    res_zero = inspect_data_tool.invoke({"n": 0})
    assert res_zero["status"] == "success"
    assert res_zero["n_rows_shown"] == 0
    assert len(res_zero["rows"]) == 0
    assert "text" in res_zero["columns"]

    # n exceeding dataset size (returns all 8 rows without failing)
    res_large = inspect_data_tool.invoke({"n": 100})
    assert res_large["status"] == "success"
    assert res_large["n_rows_shown"] == 8
    assert len(res_large["rows"]) == 8

    # negative n (pandas df.head(-2) drops the last 2 rows from 8, returning 6)
    res_neg = inspect_data_tool.invoke({"n": -2})
    assert res_neg["status"] == "success"
    assert res_neg["n_rows_shown"] == 6
    assert len(res_neg["rows"]) == 6


def test_inspect_data_tool_no_data():
    session = SessionState()
    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    result = inspect_data_tool.invoke({})

    assert result["status"] == "error"
    assert "No dataset currently loaded" in result["message"]


def test_tokenize_tool_known_answer_10(loaded_session):
    """Tests tokenize_tool against TEST_FIXTURE.md known answer #10."""
    data_tools = make_tools(loaded_session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({"column": "text"})

    # Assert returned dict fields
    assert result["status"] == "success"
    assert result["total_documents"] == 8
    assert result["total_tokens"] == 27
    assert result["unique_vocab_size"] == 5

    # Known answer #10 token lists
    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],
    ]

    # Assert stored artifacts & dataframe column
    assert loaded_session.artifacts["unigrams"] == expected_unigrams
    assert loaded_session.dataframe["unigrams"].tolist() == expected_unigrams


def test_tokenize_tool_nonexistent_column(loaded_session):
    """Tests tokenize_tool error handling when specifying a non-existent column name."""
    data_tools = make_tools(loaded_session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    result = tokenize_tool.invoke({"column": "non_existent_col"})

    assert result["status"] == "error"
    assert "Column 'non_existent_col' not found" in result["message"]


def test_list_files_tool():
    """Tests list_files_tool against Known Answer #11 in TEST_FIXTURE.md."""
    session = SessionState()
    data_tools = make_tools(session)
    list_files_tool = next(t for t in data_tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert res["status"] == "success"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_check_missing_tool_known_answer_2(loaded_session):
    """Tests check_missing_tool against Known Answer #2 in TEST_FIXTURE.md."""
    data_tools = make_tools(loaded_session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({"column": "text"})

    assert res["status"] == "success"
    assert res["total_rows"] == 8
    assert res["n_missing"] == 1
    assert res["missing_indices"] == [7]


def test_check_duplicates_tool_known_answer_3():
    """Tests check_duplicates_tool against Known Answer #3 in TEST_FIXTURE.md."""
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke(
        {
            "file_path": "agent_dev/sample_fixture.csv",
            "text_column": "text",
            "label_column": "label",
        }
    )

    data_tools = make_tools(session)
    check_duplicates_tool = next(t for t in data_tools if t.name == "check_duplicates_tool")

    res_check = check_duplicates_tool.invoke({"column": "text", "drop": False})
    assert res_check["status"] == "success"
    assert res_check["n_duplicates"] == 1
    assert res_check["n_rows_before"] == 8
    assert res_check["n_rows_after"] == 8

    res_drop = check_duplicates_tool.invoke({"column": "text", "drop": True})
    assert res_drop["status"] == "success"
    assert res_drop["n_rows_before"] == 8
    assert res_drop["n_rows_after"] == 6
    assert len(session.dataframe) == 6


def test_sample_data_tool_known_answer_13(loaded_session):
    """Tests sample_data_tool against Known Answer #13 in TEST_FIXTURE.md."""
    data_tools = make_tools(loaded_session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert res["status"] == "success"
    assert res["n_sampled"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]


def test_describe_data_tool_known_answer_18(loaded_session):
    """Tests describe_data_tool against Known Answer #18 in TEST_FIXTURE.md."""
    data_tools = make_tools(loaded_session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_data_tool.invoke({"column": "text"})

    assert res["status"] == "success"
    
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)

    cat_stats = res["category_stats"]
    assert np.isclose(cat_stats["catA"]["mean"], 17.25, atol=1e-3)
    assert np.isclose(cat_stats["catB"]["mean"], 22.5, atol=1e-3)

    # Verify plot set on pending_figure
    assert loaded_session.pending_figure is not None
