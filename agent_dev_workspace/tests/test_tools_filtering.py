import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool():
    # Setup session and load fixture
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold=0.15 (Known Answer #5)
    res = variance_filter_tool.invoke({"threshold": 0.15})

    assert res["total_terms"] == 5
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1
    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert res["removed_terms"] == ["always"]

    # Verify full_report dataframe stored in session results
    result_entry = session.results_store[res["result_id"]]
    full_report = result_entry["full_report"]
    
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    for term, exp_var in expected_variances.items():
        actual_var = full_report[full_report["term"] == term]["variance"].values[0]
        assert np.isclose(actual_var, exp_var, atol=1e-3)


def test_pearson_filter_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    # Run pearson_filter_tool with target_class="catB" (Known Answer #6)
    res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5

    full_report = session.results_store[res["result_id"]]["full_report"]

    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, exp_r in expected_pearson.items():
        actual_r = full_report[full_report["term"] == term]["pearson_r"].values[0]
        assert np.isclose(actual_r, exp_r, atol=1e-3)


def test_spearman_filter_tool():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run spearman_filter_tool with target_class="catB" (Known Answer #6)
    res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    assert res["target_class"] == "catB"
    assert res["total_terms"] == 5

    full_report = session.results_store[res["result_id"]]["full_report"]

    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for term, exp_r in expected_spearman.items():
        actual_r = full_report[full_report["term"] == term]["spearman_r"].values[0]
        assert np.isclose(actual_r, exp_r, atol=1e-3)
