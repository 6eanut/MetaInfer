"""Phase / Outcome / Transition definitions for fusedmoe-evolve.

4-phase loop:
  A_prepare -> B_evolve -> C_validate -> D_review -> A_prepare (new iter)

Flow:
  - A_prepare: agent writes initial_program.py, evaluator.py, config.yaml
  - B_evolve: oracle runs openevolve subprocess; N internal generations
  - C_validate: oracle runs correctness + perf tests on best kernel
  - D_review: agent reviews trajectory, writes review.md + perf_plan.md
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


# Type aliases
Phase = Literal[
    "idle",
    "A_prepare",
    "B_evolve",
    "C_validate",
    "D_review",
    "finished",
]

Outcome = Literal[
    "ok",
    "logic_fail",
    "infra_fail",
    "perf_regression",
    "aborted",
]


# Runtime constants
OK: Outcome = "ok"
LOGIC_FAIL: Outcome = "logic_fail"
INFRA_FAIL: Outcome = "infra_fail"
PERF_REGRESSION: Outcome = "perf_regression"
ABORTED: Outcome = "aborted"

ALL_OUTCOMES: List[Outcome] = [OK, LOGIC_FAIL, INFRA_FAIL, PERF_REGRESSION, ABORTED]


@dataclass(frozen=True)
class PhaseMeta:
    """Display metadata for one phase."""

    id: Phase
    label: str
    description: str = ""
    is_terminal: bool = False


@dataclass(frozen=True)
class Transition:
    """One edge of the state machine.

    Attributes
    ----------
    carry_failure
        If True, propagate ``ctx.failure`` to the next phase.
    carry_perf
        If True, update ``ctx.last_perf`` from this step's measured perf.
    consume_iteration
        If True, close the current iteration folder and open a fresh one for
        the next phase.
    """

    from_phase: Phase
    on: Outcome
    to_phase: Phase
    label: str = ""
    carry_failure: bool = True
    carry_perf: bool = False
    consume_iteration: bool = True


# --------------------------------------------------------------------------- #
# Phases
# --------------------------------------------------------------------------- #

PHASES: List[PhaseMeta] = [
    PhaseMeta("idle",        "idle",       "not started"),
    PhaseMeta("A_prepare",   "A: Prepare",
              "agent writes initial_program.py, evaluator.py, config.yaml"),
    PhaseMeta("B_evolve",    "B: Evolve",
              "oracle runs openevolve; N internal generations"),
    PhaseMeta("C_validate",  "C: Validate",
              "oracle runs correctness + perf tests on best kernel"),
    PhaseMeta("D_review",    "D: Review",
              "agent reviews trajectory, writes review.md + perf_plan.md"),
    PhaseMeta("finished",    "finished",
              "run ended", is_terminal=True),
]

PHASE_ORDER: List[Phase] = [
    "A_prepare", "B_evolve", "C_validate", "D_review",
]


# --------------------------------------------------------------------------- #
# Transition table
#
# Flow: A → B → C → D → A (new iter)
# B_evolve fail → consume iteration, route to D for review, then new iter
# --------------------------------------------------------------------------- #

TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # ---- intra-iteration forward (same folder) ----
    ("A_prepare",   OK): Transition("A_prepare",   OK, "B_evolve",
                                    label="ok",   carry_failure=False, consume_iteration=False),
    ("B_evolve",    OK): Transition("B_evolve",    OK, "C_validate",
                                    label="ok",   carry_failure=False, consume_iteration=False),
    ("C_validate",  OK): Transition("C_validate",  OK, "D_review",
                                    label="pass", carry_failure=False, carry_perf=True,
                                    consume_iteration=False),
    ("C_validate",  LOGIC_FAIL): Transition("C_validate",  LOGIC_FAIL, "D_review",
                                            label="fail", carry_failure=False,
                                            consume_iteration=False),
    ("D_review",    OK): Transition("D_review",    OK, "A_prepare",
                                    label="new iter", carry_failure=False,
                                    consume_iteration=True),

    # ---- infra failures: retry in place ----
    ("A_prepare",   INFRA_FAIL): Transition("A_prepare",   INFRA_FAIL, "A_prepare",
                                            label="retry", carry_failure=False,
                                            consume_iteration=False),
    ("B_evolve",    INFRA_FAIL): Transition("B_evolve",    INFRA_FAIL, "B_evolve",
                                            label="retry", carry_failure=False,
                                            consume_iteration=False),
    ("C_validate",  INFRA_FAIL): Transition("C_validate",  INFRA_FAIL, "C_validate",
                                            label="retry", carry_failure=False,
                                            consume_iteration=False),
    ("D_review",    INFRA_FAIL): Transition("D_review",    INFRA_FAIL, "D_review",
                                            label="retry", carry_failure=False,
                                            consume_iteration=False),

    # ---- logic failures (agent produced bad output) ----
    ("A_prepare",   LOGIC_FAIL): Transition("A_prepare",   LOGIC_FAIL, "A_prepare",
                                            label="replan", carry_failure=False,
                                            consume_iteration=False),

    # ---- B logic_fail: openevolve produced no valid program ----
    # Route to D so the reviewer can document what happened,
    # then start a fresh iteration
    ("B_evolve",    LOGIC_FAIL): Transition("B_evolve",    LOGIC_FAIL, "D_review",
                                            label="OE fail → review",
                                            carry_failure=True, consume_iteration=True),

    # ---- C perf_regression: passed correctness but slower than best ----
    ("C_validate",  PERF_REGRESSION): Transition("C_validate",  PERF_REGRESSION, "D_review",
                                                  label="regress", carry_failure=False,
                                                  consume_iteration=False),
}


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #


def next_transition(from_phase: Phase, outcome: Outcome) -> Optional[Transition]:
    """Return the transition for ``(from_phase, outcome)`` or ``None``."""
    return TRANSITIONS.get((from_phase, outcome))


def phase_label(p: Phase) -> str:
    for m in PHASES:
        if m.id == p:
            return m.label
    return str(p)


def phase_meta(p: Phase) -> Optional[PhaseMeta]:
    for m in PHASES:
        if m.id == p:
            return m
    return None


def is_terminal(p: Phase) -> bool:
    m = phase_meta(p)
    return bool(m and m.is_terminal)


# --------------------------------------------------------------------------- #
# Graph export — consumed by the WebUI
# --------------------------------------------------------------------------- #


def nodes_for_graph() -> List[Dict[str, str]]:
    """Return node metadata for every phase in :data:`PHASE_ORDER`."""
    return [
        {"id": m.id, "label": m.label, "description": m.description}
        for m in PHASES
        if m.id in PHASE_ORDER
    ]


def edges_for_graph() -> List[Dict[str, str]]:
    """Return deduped ``{from, to, label}`` edges."""
    merged: Dict[Tuple[Phase, Phase], List[str]] = {}
    for (frm, _outc), t in TRANSITIONS.items():
        merged.setdefault((frm, t.to_phase), []).append(t.label or _outc)
    out: List[Dict[str, str]] = []
    for (frm, to), labels in merged.items():
        out.append({
            "from": frm,
            "to": to,
            "label": " / ".join(sorted(set(labels))),
        })
    return out


def outcome_label(o: Outcome) -> str:
    return {
        OK: "ok",
        LOGIC_FAIL: "logic fail",
        INFRA_FAIL: "infra fail",
        PERF_REGRESSION: "perf regression",
        ABORTED: "aborted",
    }.get(o, str(o))
