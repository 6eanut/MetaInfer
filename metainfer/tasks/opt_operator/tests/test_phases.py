"""Tests for the pool-evolution phase machine and its WebUI graph payload."""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.phases import (
    CHEAP_PHASES,
    EDGES,
    PHASE_ORDER,
    STRONG_PHASES,
    graph_payload,
    is_terminal,
    next_phase,
    phase_spec,
    tier_for_phase,
)


ALL_KEYS = set(PHASE_ORDER)


def test_phase_order_is_complete_and_terminal():
    assert PHASE_ORDER[0] == "harness_setup"
    assert PHASE_ORDER[-1] == "finished"
    assert is_terminal("finished")
    assert not is_terminal("verify")
    # the whole machine is present, no leftover A…F phases
    assert not (ALL_KEYS & {"S_baseline", "A_plan", "B_implement", "C_conformance",
                            "D_review", "E_perf_test", "F_perf_plan"})


def test_next_phase_primary_flow():
    # harness_setup -> select_kernel -> optimize -> verify is the primary spine.
    assert next_phase("harness_setup") == "select_kernel"
    assert next_phase("select_kernel") == "optimize"
    assert next_phase("optimize") == "verify"
    # branch/loop states decide their next step at runtime -> None from next_phase.
    for branch in ("verify", "repair", "admit_to_pool", "discarded", "finished"):
        assert next_phase(branch) is None, branch
    assert next_phase("does_not_exist") is None


def test_edges_express_the_full_machine():
    pairs = {(e["from"], e["to"]): e["kind"] for e in EDGES}
    assert pairs[("harness_setup", "select_kernel")] == "flow"
    assert pairs[("select_kernel", "optimize")] == "flow"
    assert pairs[("optimize", "verify")] == "flow"
    # correctness branch
    assert pairs[("verify", "admit_to_pool")] == "pass"
    assert pairs[("verify", "repair")] == "fail"
    assert pairs[("repair", "verify")] == "retry"
    # outer round loop + terminal
    assert pairs[("admit_to_pool", "select_kernel")] == "loop"
    assert pairs[("discarded", "select_kernel")] == "loop"
    assert pairs[("select_kernel", "finished")] == "stop"


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
    all_non_terminal = ALL_KEYS - {"finished"}
    assert set(CHEAP_PHASES) | set(STRONG_PHASES) == all_non_terminal


def test_graph_payload_node_set_and_states():
    payload = graph_payload("verify")
    assert payload["current"] == "verify"
    ids = [n["id"] for n in payload["nodes"]]
    assert ids == PHASE_ORDER
    by_id = {n["id"]: n for n in payload["nodes"]}
    assert by_id["harness_setup"]["state"] == "done"    # setup ran once
    assert by_id["verify"]["state"] == "current"
    assert by_id["finished"]["state"] == "pending"


def test_graph_payload_edge_set():
    payload = graph_payload("optimize")
    edge_pairs = {(e["from"], e["to"]) for e in payload["edges"]}
    assert ("harness_setup", "select_kernel") in edge_pairs
    assert ("verify", "repair") in edge_pairs
    assert ("repair", "verify") in edge_pairs
    assert ("admit_to_pool", "select_kernel") in edge_pairs
    assert ("discarded", "select_kernel") in edge_pairs
    assert ("select_kernel", "finished") in edge_pairs


def test_graph_payload_active_edge():
    g = graph_payload("optimize")
    assert g["active_edge"] == {"from": "select_kernel", "to": "optimize",
                                "kind": "flow"}
    g_verify = graph_payload("repair")
    assert g_verify["active_edge"]["kind"] in ("fail", "retry", "loop")


def test_graph_payload_terminal():
    g = graph_payload("finished")
    assert g["current"] == "finished"
    assert g["terminal_nodes"][0]["id"] == "finished"
    # outcome legend documents the branch meanings for the UI
    keys = {o["key"] for o in g["outcome_legend"]}
    assert keys == {"admit_to_pool", "discarded", "failed"}
