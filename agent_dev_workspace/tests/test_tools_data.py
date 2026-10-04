"""Tests for tools_data.py tools."""

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
