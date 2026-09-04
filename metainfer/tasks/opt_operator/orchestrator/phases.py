"""Phase state machine + WebUI graph payload for the opt_operator pool evolution.

The loop is a *pool-evolution* engine (OPT_KERNEL_SPEC FR-4/5/6/7): each round
picks a kernel from the admitted pool by quality-weighted sampling, optimizes it
into a candidate, verifies it with the twin harnesses (correctness + benchmark),
repairs a wrong candidate a bounded number of times, then either admits it to the
pool or records a discarded conclusion::

    harness_setup
        -> select_kernel -> optimize -> verify -> [admit_to_pool | discarded]
            -> select_kernel -> ... -> finished

- ``harness_setup``  one-time: certify the genesis kernel into the pool, run the
                    adversarial review of the correctness/benchmark harnesses.
- ``select_kernel``  quality-weighted probability sample from the pool (seeded).
- ``optimize``       land a candidate kernel that improves the selected kernel.
- ``verify``         twin harnesses: correctness gate vs oracle + benchmark.
- ``repair``         bounded (<= max_repairs) retry of a failing candidate.
- ``admit_to_pool``  candidate correct + above the admission score gate.
- ``discarded``      candidate failed correctness after repair, or below the gate.
- ``finished``       terminal.

The ``discarded`` / ``admit_to_pool`` states close one round and loop back to
``select_kernel``; ``finished`` is reachable from ``select_kernel`` when the
iteration budget (or a stop decision) is reached. Requirement confirmation is a
**creation-time** concern (before any run), so there is no in-run await/confirm
phase. ``graph_payload`` renders the machine for the WebUI overview.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PhaseSpec:
    key: str
    label: str
    description: str
    model_tier: str = "strong"   # which LLM tier drives this phase


_PHASE_ORDER = [
    "harness_setup", "select_kernel", "optimize", "verify",
    "repair", "admit_to_pool", "discarded", "finished",
]

# Linear display spine of a single round for the WebUI stepper. Branch states
# (repair / discarded) and the terminal (finished) are laid out around it.
_SPINE = ["harness_setup", "select_kernel", "optimize", "verify",
          "admit_to_pool", "finished"]

# Directed machine edges: (from, to, kind).
# kind: flow=forward, pass/fail=branch from verify, retry=repair re-entry,
#       loop=outer round loop-back, stop=terminal exit.
_EDGES: List[Dict[str, str]] = [
    {"from": "harness_setup", "to": "select_kernel", "kind": "flow"},
    {"from": "select_kernel", "to": "optimize", "kind": "flow"},
    {"from": "optimize", "to": "verify", "kind": "flow"},
    {"from": "verify", "to": "admit_to_pool", "kind": "pass"},
    {"from": "verify", "to": "repair", "kind": "fail"},
    {"from": "repair", "to": "verify", "kind": "retry"},
    {"from": "admit_to_pool", "to": "select_kernel", "kind": "loop"},
    {"from": "discarded", "to": "select_kernel", "kind": "loop"},
    {"from": "select_kernel", "to": "finished", "kind": "stop"},
]

_PHASES: Dict[str, PhaseSpec] = {
    "harness_setup": PhaseSpec(
        "harness_setup", "Setup & certify harness",
        "Certify the genesis kernel and adversarially self-review the "
        "correctness + benchmark harnesses.",
        model_tier="strong"),
    "select_kernel": PhaseSpec(
        "select_kernel", "Select kernel",
        "Sample a kernel from the pool by quality-weighted probability.",
        model_tier="cheap"),
    "optimize": PhaseSpec(
        "optimize", "Optimize",
        "Land a candidate kernel that improves on the selected kernel.",
        model_tier="cheap"),
    "verify": PhaseSpec(
        "verify", "Verify",
        "Run the candidate through the correctness + benchmark harnesses.",
        model_tier="cheap"),
    "repair": PhaseSpec(
        "repair", "Repair",
        "Fix a candidate the correctness gate rejected (bounded retries).",
        model_tier="cheap"),
    "admit_to_pool": PhaseSpec(
        "admit_to_pool", "Admit",
        "Candidate is correct and above the admission gate; enter the pool.",
        model_tier="cheap"),
    "discarded": PhaseSpec(
        "discarded", "Discard",
        "Candidate failed correctness or missed the gate; recorded conclusion.",
        model_tier="cheap"),
    "finished": PhaseSpec(
        "finished", "Finished", "Optimization complete.", model_tier="cheap"),
}

PHASE_ORDER = list(_PHASE_ORDER)
PHASES = dict(_PHASES)
SPINE = list(_SPINE)
EDGES = [dict(e) for e in _EDGES]

# Cheap-tier execution phases — mechanical build/verify/repair/admit work driven
# by the seeded selection; only harness_setup (certify + adversarial review) is a
# strong-model phase.
CHEAP_PHASES = ("select_kernel", "optimize", "verify", "repair",
                "admit_to_pool", "discarded")
STRONG_PHASES = ("harness_setup",)


def phase_spec(key: str) -> PhaseSpec:
    if key not in _PHASES:
        raise ValueError(f"unknown phase {key!r}")
    return _PHASES[key]


def next_phase(key: str) -> Optional[str]:
    """The primary flow edge out of ``key``, or None if unknown/terminal.

    Returns None for terminal ``finished`` and for branch/loop states whose next
    step is decided at runtime (verify -> admit|repair, admit/discard -> select).
    """
    if key == "finished" or key not in _PHASES:
        return None
    for e in _EDGES:
        if e["from"] == key and e["kind"] == "flow":
            return e["to"]
    return None


def is_terminal(key: str) -> bool:
    return key == "finished"


def _index_in_spine(key: str) -> Optional[int]:
    return SPINE.index(key) if key in SPINE else None


def graph_payload(current: str, last_outcome: Optional[str] = None,
                  last_label: Optional[str] = None) -> Dict[str, object]:
    """Render the pool-evolution machine for the WebUI overview.

    ``current`` is the run's active phase; ``last_outcome`` / ``last_label``
    describe the transition that just fired. The machine is a directed graph with
    a branch (verify -> admit|repair), a bounded retry (repair -> verify), an
    outer round loop (admit/discard -> select_kernel) and a terminal stop
    (select_kernel -> finished when the budget is exhausted). Node ``state`` is
    coarse: ``current`` for the active phase, ``done`` for the one-time setup
    after it has run, otherwise ``pending`` (a WebUI stepper resolves loop
    iteration visually from ``iteration`` in run.json).
    """
    current_idx = _index_in_spine(current)
    nodes = []
    for spec in _PHASES.values():
        if spec.key == current:
            state = "current"
        elif spec.key == "harness_setup" and current not in ("", "harness_setup"):
            state = "done"      # setup is one-time; done once the loop starts
        elif (spec.key != "finished" and current_idx is not None
              and _index_in_spine(spec.key) is not None
              and _index_in_spine(spec.key) < current_idx):
            state = "done"
        else:
            state = "pending"
        nodes.append({
            "id": spec.key,
            "label": spec.label,
            "tier": spec.model_tier,
            "state": state,
            "description": spec.description,
        })

    edges = [dict(e) for e in EDGES]

    active_edge = None
    if current != "finished":
        # Highlight the primary flow edge that just produced ``current``.
        for e in EDGES:
            if e["to"] == current and e["kind"] in ("flow", "pass", "fail",
                                                    "retry", "loop"):
                active_edge = {"from": e["from"], "to": e["to"],
                               "kind": e["kind"]}
                break
        if active_edge is None:
            active_edge = {"from": None, "to": current, "kind": "flow"}
    else:
        active_edge = {"from": None, "to": "finished", "kind": "stop"}

    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": [{"id": "finished", "label": "Finished",
                            "description": "Optimization complete."}],
        "outcome_legend": [
            {"key": "admit_to_pool", "label": "Admitted",
             "description": "Correct and above the admission gate -> pool."},
            {"key": "discarded", "label": "Discarded",
             "description": "Failed correctness after repair, or below the gate."},
            {"key": "failed", "label": "Failed",
             "description": "No candidate could be built/launched; round skipped."},
        ],
    }


def tier_for_phase(key: str) -> str:
    return _PHASES[key].model_tier


__all__ = ["PhaseSpec", "PHASE_ORDER", "PHASES", "SPINE", "EDGES",
           "CHEAP_PHASES", "STRONG_PHASES", "phase_spec", "next_phase",
           "is_terminal", "graph_payload", "tier_for_phase"]
