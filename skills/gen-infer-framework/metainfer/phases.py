"""Phase / Outcome / Transition definitions — the single source of truth for
the orchestrator's state machine.

This module is the **only** place where phase names, their display labels, and
the legal transitions live. The pipeline consults :data:`TRANSITIONS` to decide
"what runs next"; the WebUI's ``/api/state-graph`` reads :data:`PHASES` and
:data:`PHASE_ORDER` plus :func:`edges_for_graph` to render the flow diagram.

**Adding or changing behavior is a one-file edit here:**

1. add the phase to :data:`PHASES` (and :data:`PHASE_ORDER` if it should appear
   in the graph),
2. add the relevant :data:`TRANSITIONS` entries,
3. register a ``_do_<phase>`` handler in :mod:`metainfer.pipeline`.

The WebUI picks up the new nodes/edges automatically — no frontend edit
required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple


# Type aliases (kept as Literals so IDEs / mypy catch typos).
Phase = Literal[
    "idle",
    "A_plan",
    "B_implement",
    "C_test",
    "D_review",
    "E_perf_test",
    "F_perf_plan",
    "finished",
    "failed",
]

Outcome = Literal[
    "ok",
    "logic_fail",
    "infra_fail",
    "perf_regression",
    "aborted",
]


# Runtime constants — useful where you need a value rather than a type.
OK              : Outcome = "ok"
LOGIC_FAIL      : Outcome = "logic_fail"
INFRA_FAIL      : Outcome = "infra_fail"
PERF_REGRESSION : Outcome = "perf_regression"
ABORTED         : Outcome = "aborted"

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
        If True, propagate ``ctx.failure`` to the next phase. If False and the
        outcome was OK, the failure is cleared.
    carry_perf
        If True, update ``ctx.last_perf`` from this step's measured perf
        (typically only on C-pass).
    consume_iteration
        If True, close the current iteration folder and open a fresh one for
        the next phase. If False, the next phase runs in the same folder
        (used for in-place infra retries and intra-iteration forward steps
        like A→B→C→D→E→F).
    """

    from_phase: Phase
    on: Outcome
    to_phase: Phase
    label: str = ""
    carry_failure: bool = True
    carry_perf: bool = False
    consume_iteration: bool = True


# --------------------------------------------------------------------------- #
# Phases — canonical list + display order
# --------------------------------------------------------------------------- #
#
# Flow:
#   A_plan → B_implement → C_test → D_review ──┬─ C ok  → E_perf_test → F_perf_plan → A_plan (new iter)
#                                                └─ C fail → B_implement (new iter)
#
# D_review ALWAYS runs after C (regardless of C outcome). Its egress routing
# (→ E vs → B) is encoded in D's outcome, which the orchestrator derives from
# C's outcome (see _do_review in pipeline.py). E and F only run on the happy
# path; if C failed, D routes the iteration back to B for a redo.
#
PHASES: List[PhaseMeta] = [
    PhaseMeta("idle",        "idle",     "not started"),
    PhaseMeta("A_plan",      "A: Plan",  "planner writes plan.md + test_spec.md"),
    PhaseMeta("B_implement", "B: Implement",
              "implementer writes code + smoke-tests serve.sh"),
    PhaseMeta("C_test",      "C: Correctness Test",
              "run immutable oracle (or test.sh) for correctness only"),
    PhaseMeta("D_review",    "D: Review + Retro",
              "post-test reviewer writes review.md; advisory, does NOT gate. "
              "Routes to E on C-pass, back to B on C-fail"),
    PhaseMeta("E_perf_test", "E: Perf Test",
              "agent writes + runs perf.sh (heavier load) → perf_report.json"),
    PhaseMeta("F_perf_plan", "F: Perf Plan",
              "agent reads perf_report.json + review.md, writes perf_plan.md; "
              "no code changes; next iteration's A executes the plan"),
    PhaseMeta("finished",    "finished", "run ended cleanly", is_terminal=True),
    PhaseMeta("failed",      "failed",   "run ended in failure", is_terminal=True),
]

# Left-to-right display order for the graph. Phases not in this list
# (idle / finished / failed) are rendered as status badges, not graph nodes.
PHASE_ORDER: List[Phase] = [
    "A_plan", "B_implement", "C_test", "D_review", "E_perf_test", "F_perf_plan",
]


# --------------------------------------------------------------------------- #
# Transition table
#
# Keys are (from_phase, outcome). A missing key for a runtime (phase, outcome)
# pair is treated as "abort the run".
#
# D_review's outcome is set by the orchestrator based on what C's outcome was
# (NOT on whether the reviewer agent itself succeeded — D is advisory). So:
#   (D_review, OK)         → E_perf_test   [meaning: C had passed]
#   (D_review, LOGIC_FAIL) → B_implement   [meaning: C had failed]
# --------------------------------------------------------------------------- #

TRANSITIONS: Dict[Tuple[Phase, Outcome], Transition] = {
    # ---- intra-iteration forward (do NOT consume the iteration) ----------- #
    ("A_plan",       OK): Transition("A_plan",       OK, "B_implement",
                                    label="ok",   carry_failure=False, consume_iteration=False),
    ("B_implement",  OK): Transition("B_implement",  OK, "C_test",
                                    label="ok",   carry_failure=False, consume_iteration=False),
    ("C_test",       OK):              Transition("C_test",  OK, "D_review",
                                                  label="pass",   carry_failure=False, consume_iteration=False),
    ("C_test",       LOGIC_FAIL):      Transition("C_test",  LOGIC_FAIL, "D_review",
                                                  label="fail",   carry_failure=False, consume_iteration=False),
    ("C_test",       INFRA_FAIL):      Transition("C_test",  INFRA_FAIL, "D_review",
                                                  label="infra",  carry_failure=False, consume_iteration=False),
    ("C_test",       PERF_REGRESSION): Transition("C_test",  PERF_REGRESSION, "D_review",
                                                  label="regress", carry_failure=False, consume_iteration=False),
    ("D_review",     OK):              Transition("D_review", OK, "E_perf_test",
                                                  label="C ok → perf",
                                                  carry_failure=False, consume_iteration=False),
    ("D_review",     LOGIC_FAIL):      Transition("D_review", LOGIC_FAIL, "B_implement",
                                                  label="C fail → redo",
                                                  consume_iteration=True),
    ("E_perf_test",  OK):              Transition("E_perf_test", OK, "F_perf_plan",
                                                  label="ok", carry_failure=False,
                                                  carry_perf=True, consume_iteration=False),
    ("F_perf_plan",  OK):              Transition("F_perf_plan", OK, "A_plan",
                                                  label="new iter",
                                                  carry_failure=False, consume_iteration=True),

    # ---- infra failures: retry in place, same folder ---------------------- #
    ("A_plan",       INFRA_FAIL): Transition("A_plan",       INFRA_FAIL, "A_plan",
                                             label="retry", carry_failure=False, consume_iteration=False),
    ("B_implement",  INFRA_FAIL): Transition("B_implement",  INFRA_FAIL, "B_implement",
                                             label="retry", carry_failure=False, consume_iteration=False),
    ("E_perf_test",  INFRA_FAIL): Transition("E_perf_test",  INFRA_FAIL, "E_perf_test",
                                             label="retry", carry_failure=False, consume_iteration=False),
    ("F_perf_plan",  INFRA_FAIL): Transition("F_perf_plan",  INFRA_FAIL, "F_perf_plan",
                                             label="retry", carry_failure=False, consume_iteration=False),

    # ---- logic failures at A/B/E/F: redo in place, same folder ------------ #
    # (SubAgentManager already retried 3× internally; one more redo here with
    #  a fresh prompt before burning a new iteration folder.)
    ("A_plan",       LOGIC_FAIL): Transition("A_plan",       LOGIC_FAIL, "A_plan",
                                             label="replan", carry_failure=False, consume_iteration=False),
    ("B_implement",  LOGIC_FAIL): Transition("B_implement",  LOGIC_FAIL, "B_implement",
                                             label="redo",   carry_failure=False, consume_iteration=False),
    ("E_perf_test",  LOGIC_FAIL): Transition("E_perf_test",  LOGIC_FAIL, "E_perf_test",
                                             label="redo",   carry_failure=False, consume_iteration=False),
    ("F_perf_plan",  LOGIC_FAIL): Transition("F_perf_plan",  LOGIC_FAIL, "F_perf_plan",
                                             label="redo",   carry_failure=False, consume_iteration=False),

    # ---- abort: any phase can transition to failed on ABORTED ------------- #
    ("A_plan",       ABORTED): Transition("A_plan",       ABORTED, "failed", label="abort", carry_failure=False),
    ("B_implement",  ABORTED): Transition("B_implement",  ABORTED, "failed", label="abort", carry_failure=False),
    ("C_test",       ABORTED): Transition("C_test",       ABORTED, "failed", label="abort", carry_failure=False),
    ("D_review",     ABORTED): Transition("D_review",     ABORTED, "failed", label="abort", carry_failure=False),
    ("E_perf_test",  ABORTED): Transition("E_perf_test",  ABORTED, "failed", label="abort", carry_failure=False),
    ("F_perf_plan",  ABORTED): Transition("F_perf_plan",  ABORTED, "failed", label="abort", carry_failure=False),
}


# --------------------------------------------------------------------------- #
# Lookup helpers
# --------------------------------------------------------------------------- #


def next_transition(from_phase: Phase, outcome: Outcome) -> Optional[Transition]:
    """Return the transition for ``(from_phase, outcome)`` or ``None`` if no
    edge is defined (the orchestrator treats this as an unrecoverable abort)."""
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
    """Return deduped ``{from, to, label}`` edges.

    Multiple outcomes on the same ``(from, to)`` pair get their labels merged
    (sorted, de-duplicated, ``" / "``-joined) so the graph stays readable.
    """
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
