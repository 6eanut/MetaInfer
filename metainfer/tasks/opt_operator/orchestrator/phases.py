"""Phase state machine + WebUI graph payload for the opt_operator loop.

Optimization loop::

    S_baseline -> A_plan -> B_implement -> C_conformance -> D_review
        -> E_perf_test -> F_perf_plan -> (loop to A_plan) ... -> finished

- ``S_baseline``  self-certify the initial champion (mode-A source or mode-B
                    reference-library baseline).
- ``A_plan``       strong model reasons about how to improve (architecture-level).
- ``B_implement``  cheap model lands the plan into a candidate kernel.
- ``C_conformance``build/load + numerics gate against the frozen oracle.
- ``D_review``     strong model reviews the conformance/perf evidence (guidance,
                    not a gate).
- ``E_perf_test``  profile every shape; promote if it beats the incumbent.
- ``F_perf_plan``  strong model analyses the profile and plans the next iteration.

``finished`` is terminal. The phase machine is a tiny DAG with no branching today,
kept simple and deterministic; ``graph_payload`` renders it for the WebUI overview.
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
    "S_baseline", "A_plan", "B_implement", "C_conformance",
    "D_review", "E_perf_test", "F_perf_plan", "finished",
]

_PHASES: Dict[str, PhaseSpec] = {
    "S_baseline": PhaseSpec("S_baseline", "Self-certify baseline",
                            "Certify the initial champion against the frozen oracle.",
                            model_tier="strong"),
    "A_plan": PhaseSpec("A_plan", "Plan optimization",
                        "Strong model devises the next optimization strategy.",
                        model_tier="strong"),
    "B_implement": PhaseSpec("B_implement", "Implement candidate",
                             "Cheap model lands the plan into a candidate kernel.",
                             model_tier="cheap"),
    "C_conformance": PhaseSpec("C_conformance", "Check correctness",
                               "Build/load the candidate and gate it on the oracle.",
                               model_tier="cheap"),
    "D_review": PhaseSpec("D_review", "Review evidence",
                          "Strong model reviews conformance/perf evidence (guidance).",
                          model_tier="strong"),
    "E_perf_test": PhaseSpec("E_perf_test", "Profile & promote",
                             "Profile all shapes; promote if it beats the incumbent.",
                             model_tier="cheap"),
    "F_perf_plan": PhaseSpec("F_perf_plan", "Analyze profile",
                             "Strong model analyzes the profile and plans next.",
                             model_tier="strong"),
    "finished": PhaseSpec("finished", "Finished", "Optimization complete.",
                          model_tier="cheap"),
}

PHASE_ORDER = list(_PHASE_ORDER)
PHASES = dict(_PHASES)

# Cheap-tier execution phases (B_implement / C repair) — these follow a strong
# model's plan, so they can use a cheaper model.
CHEAP_PHASES = ("B_implement", "C_conformance", "E_perf_test")
STRONG_PHASES = ("S_baseline", "A_plan", "D_review", "F_perf_plan")


def phase_spec(key: str) -> PhaseSpec:
    if key not in _PHASES:
        raise ValueError(f"unknown phase {key!r}")
    return _PHASES[key]


def next_phase(key: str) -> Optional[str]:
    """The phase that follows ``key``, or None if unknown/terminal."""
    if key not in _PHASE_ORDER:
        return None
    i = _PHASE_ORDER.index(key)
    return _PHASE_ORDER[i + 1] if i + 1 < len(_PHASE_ORDER) else None


def is_terminal(key: str) -> bool:
    return key == "finished"


def graph_payload(current: str, last_outcome: Optional[str] = None,
                  last_label: Optional[str] = None) -> Dict[str, object]:
    """Render the phase DAG for the WebUI overview (nodes + edges + current).

    ``current`` is the run's active phase; ``last_outcome`` / ``last_label`` are
    forwarded from ``run.json`` for the WebUI to highlight the transition that
    just fired. The phase machine is a linear DAG today, so ``terminal_nodes`` is
    just ``finished`` and there is no branching outcome legend.
    """
    nodes = [
        {
            "id": spec.key,
            "label": spec.label,
            "tier": spec.model_tier,
            "state": ("current" if spec.key == current else
                      "done" if current in _PHASE_ORDER and _PHASE_ORDER.index(spec.key) < _PHASE_ORDER.index(current)
                      else "pending"),
        }
        for spec in _PHASES.values()
    ]
    edges = [
        {"from": _PHASE_ORDER[i], "to": _PHASE_ORDER[i + 1]}
        for i in range(len(_PHASE_ORDER) - 1)
    ]
    active_edge = None
    if current in _PHASE_ORDER:
        i = _PHASE_ORDER.index(current)
        if i > 0:
            active_edge = {"from": _PHASE_ORDER[i - 1], "to": current}
    else:
        active_edge = {"from": None, "to": current}
    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": [{"id": "finished", "label": "Finished",
                            "description": "Optimization complete."}],
        "outcome_legend": [],
    }


def tier_for_phase(key: str) -> str:
    return _PHASES[key].model_tier


__all__ = ["PhaseSpec", "PHASE_ORDER", "PHASES", "CHEAP_PHASES", "STRONG_PHASES",
           "phase_spec", "next_phase", "is_terminal", "graph_payload", "tier_for_phase"]
