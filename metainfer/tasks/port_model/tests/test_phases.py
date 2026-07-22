"""Tests for the port-model phase state machine (6 agents)."""

from __future__ import annotations

from metainfer.tasks.port_model.orchestrator.phases import (
    ALL_OUTCOMES,
    BOUNCE_BACK,
    INFRA_FAIL,
    NEEDS_REPAIR,
    OK,
    PHASES,
    Phase,
    TEST_FAIL,
    TRANSITIONS,
    graph_payload,
    is_terminal,
    next_transition,
    phase_label,
)


def test_phase_count():
    phase_ids = {m.id for m in PHASES}
    expected = {
        "P1_weight_analysis",
        "P2_framework_analysis",
        "P3_architect_review",
        "P4_minimal_framework",
        "P5_verify_minimal",
        "P6_port_engine",
        "finished",
    }
    assert expected <= phase_ids


def test_linear_forward_path():
    path: list[Phase] = [
        "P1_weight_analysis",
        "P2_framework_analysis",
        "P3_architect_review",
        "P4_minimal_framework",
        "P5_verify_minimal",
        "P6_port_engine",
    ]
    for i in range(len(path) - 1):
        t = next_transition(path[i], OK)
        assert t is not None, f"no transition from {path[i]} ok"
        assert t.to_phase == path[i + 1], (
            f"{path[i]} ok → {t.to_phase}, expected {path[i + 1]}"
        )


def test_p6_ok_goes_to_finished():
    t = next_transition("P6_port_engine", OK)
    assert t is not None
    assert t.to_phase == "finished"


def test_p3_bounce_back_goes_to_p1():
    t = next_transition("P3_architect_review", BOUNCE_BACK)
    assert t is not None
    assert t.to_phase == "P1_weight_analysis"


def test_p5_fail_goes_back_to_p4():
    t = next_transition("P5_verify_minimal", TEST_FAIL)
    assert t is not None
    assert t.to_phase == "P4_minimal_framework"


def test_p6_needs_repair_self_loops():
    t = next_transition("P6_port_engine", NEEDS_REPAIR)
    assert t is not None
    assert t.to_phase == "P6_port_engine"


def test_logic_fail_stops():
    for phase in (
        "P1_weight_analysis", "P2_framework_analysis",
        "P3_architect_review", "P4_minimal_framework",
        "P5_verify_minimal", "P6_port_engine",
    ):
        t = next_transition(phase, "logic_fail")
        assert t is not None, f"no transition from {phase} logic_fail"
        assert t.to_phase == "finished", f"{phase} logic_fail → {t.to_phase}"


def test_infra_fail_retries_analysis():
    for phase in (
        "P1_weight_analysis", "P2_framework_analysis",
        "P3_architect_review", "P4_minimal_framework",
    ):
        t = next_transition(phase, INFRA_FAIL)
        assert t is not None, f"no transition from {phase} infra_fail"
        assert t.to_phase == phase  # self-loop


def test_terminal_only_finished():
    assert is_terminal("finished") is True
    assert is_terminal("P1_weight_analysis") is False
    assert is_terminal("P6_port_engine") is False


def test_graph_payload_has_all_nodes():
    payload = graph_payload("P1_weight_analysis", OK, "ok")
    # 6 phases (P1-P6).
    assert len(payload["nodes"]) == 6
    assert payload["current"] == "P1_weight_analysis"
    assert isinstance(payload["edges"], list)
    assert len(payload["edges"]) > 0
    assert len(payload["outcome_legend"]) == len(ALL_OUTCOMES)


def test_phase_label_known():
    assert phase_label("P1_weight_analysis")
    assert phase_label("finished") == "finished"
