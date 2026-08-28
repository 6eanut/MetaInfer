"""Tests for the phase state machine and its WebUI graph payload."""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.phases import (
    CHEAP_PHASES,
    PHASE_ORDER,
    STRONG_PHASES,
    graph_payload,
    is_terminal,
    next_phase,
    phase_spec,
    tier_for_phase,
)


def test_phase_order_is_complete_and_terminal():
    assert PHASE_ORDER[0] == "S_baseline"
    assert PHASE_ORDER[-1] == "finished"
    assert is_terminal("finished")
    assert not is_terminal("A_plan")


def test_next_phase_chain():
    # The static machine is a linear DAG; the A_plan loop-back is a runtime
    # decision made by the pipeline (iteration cap), not a static edge.
    expected = {
        "S_baseline": "A_plan",
        "A_plan": "B_implement",
        "B_implement": "C_conformance",
        "C_conformance": "D_review",
        "D_review": "E_perf_test",
        "E_perf_test": "F_perf_plan",
        "F_perf_plan": "finished",
        "finished": None,
    }
    for phase, nxt in expected.items():
        assert next_phase(phase) == nxt
    assert next_phase("does_not_exist") is None


def test_phase_spec_unknown():
    with pytest.raises(ValueError):
        phase_spec("nope")


def test_tier_mapping():
    for p in STRONG_PHASES:
        assert tier_for_phase(p) == "strong"
    for p in CHEAP_PHASES:
        assert tier_for_phase(p) == "cheap"
    assert tier_for_phase("finished") == "cheap"


def test_tiers_are_disjoint_and_cover():
    assert not set(CHEAP_PHASES) & set(STRONG_PHASES)
    all_phases = set(PHASE_ORDER) - {"finished"}
    assert set(CHEAP_PHASES) | set(STRONG_PHASES) == all_phases


def test_graph_payload_states():
    payload = graph_payload("B_implement")
    assert payload["current"] == "B_implement"
    by_id = {n["id"]: n for n in payload["nodes"]}
    # phases before current are done
    assert by_id["S_baseline"]["state"] == "done"
    assert by_id["A_plan"]["state"] == "done"
    # current is current
    assert by_id["B_implement"]["state"] == "current"
    # everything after is pending
    assert by_id["C_conformance"]["state"] == "pending"
    assert by_id["finished"]["state"] == "pending"


def test_graph_payload_edges():
    payload = graph_payload("A_plan")
    edges = payload["edges"]
    assert {"from": "E_perf_test", "to": "F_perf_plan"} in edges
    assert {"from": "F_perf_plan", "to": "finished"} in edges
    assert len(edges) == len(PHASE_ORDER) - 1
