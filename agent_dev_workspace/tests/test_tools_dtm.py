import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


@pytest.fixture
def fixture_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    return session


def test_build_dtm_tool_fixture(fixture_session):
    expected_vocab = ["alpha", "always", "beta", "delta", "gamma"]
    assert sorted(fixture_session.feature_names) == expected_vocab
    assert fixture_session.feature_matrix.shape == (8, 5)


def test_term_frequency_tool(fixture_session):
    dtm_tools = make_tools(fixture_session)
    term_freq_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    res = term_freq_tool.invoke({"top_n": 5})
    
    assert res["status"] == "success"
    assert res["total_terms"] == 5
    top_terms = {item["term"]: item["frequency"] for item in res["top_terms"]}
    assert "always" in top_terms
    assert top_terms["always"] >= 4


def test_dtm_heatmap_tool(fixture_session):
    dtm_tools = make_tools(fixture_session)
    heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")
    res = heatmap_tool.invoke({"n_terms": 3, "n_documents": 4})
    
    assert res["status"] == "success"
    assert res["n_documents_sliced"] == 4
    assert res["n_terms_sliced"] == 3
    assert fixture_session.pending_figure is not None
