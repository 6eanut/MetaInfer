"""State machine definition.

Exports ``graph_payload(current, last_outcome, last_label) -> dict`` so the
WebUI can render the per-task-type phase diagram.

The return dict must have::

    {"current": str, "nodes": [str, ...], "edges": [...],
     "active_edge": str | None, "last_outcome": str | None,
     "terminal_nodes": [str, ...], "outcome_legend": {...}}

Each node is a phase ID. Each edge is::

    {"from": str, "to": str, "label": str, "outcome": str}

``outcome_legend`` maps outcome strings to human-readable descriptions (shown
as tooltips in the UI).

``graph_payload`` is called by the task's own ``_state_readers.py`` when the
WebUI requests ``GET /state-graph``. The shell's ``state_reader.py`` never
imports this module — only the task's web plugin does.
"""

from __future__ import annotations

from typing import Any

# Phase constants — opaque strings the shell prints as-is.  Task CSS
# (e.g. ``X.css``) styles phase pills by matching these values.
PHASE_STEP1 = "step1"
PHASE_STEP2 = "step2"
PHASE_DONE = "done"

NODES = [PHASE_STEP1, PHASE_STEP2, PHASE_DONE]

EDGES = [
    {"from": PHASE_STEP1, "to": PHASE_STEP2, "label": "continue",
     "outcome": "ok"},
    {"from": PHASE_STEP2, "to": PHASE_DONE,  "label": "finish",
     "outcome": "ok"},
    {"from": PHASE_STEP2, "to": PHASE_STEP1, "label": "retry",
     "outcome": "fail"},
]

OUTCOME_LEGEND = {
    "ok":   "step completed normally",
    "fail": "step failed; retrying",
}


def graph_payload(
    current: str,
    last_outcome: str | None = None,
    last_label: str | None = None,
) -> dict[str, Any]:
    """Return a dict ready for the WebUI state-graph panel."""
    active_edge = None
    if last_outcome and last_label:
        for e in EDGES:
            if e["outcome"] == last_outcome and e["label"] == last_label:
                active_edge = e["label"]
                break
    return {
        "current": current,
        "nodes": NODES,
        "edges": EDGES,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": [PHASE_DONE],
        "outcome_legend": OUTCOME_LEGEND,
    }
