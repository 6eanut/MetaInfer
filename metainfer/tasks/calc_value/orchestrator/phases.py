"""Phase definitions for the calc-theoretical-value pipeline.

Linear state machine::

    idle → S0_rough → S1_analyze → S2_graph → S3_calculate → S4_visualize → finished

Each step retries forever on failure (see ``pipeline._run_step_with_retry``);
there is NO ``failed`` phase. The only terminal phase is ``finished``,
which covers success, stopped (e.g. invalid inputs), and externally
interrupted runs.

These phase strings are stored in ``run.json.current_phase`` and rendered
by the WebUI. The frontend's task-detail view checks ``task_type`` to
switch to the calc-value-specific visualization when needed.
"""

from __future__ import annotations

from typing import Dict, List

# Phase strings — written to run.json. Stored as plain strings (not a
# Literal) so that the rest of the orchestrator package, which shares
# state.py with the ABCDEF pipeline, doesn't need its Phase Literal
# extended. state.py treats current_phase as opaque.
S0_ROUGH = "S0_rough"
S1_ANALYZE = "S1_analyze"
S2_GRAPH = "S2_graph"
S3_CALCULATE = "S3_calculate"
S4_VISUALIZE = "S4_visualize"
FINISHED = "finished"
IDLE = "idle"

# Legacy alias — kept so old state files / consumers that read "failed"
# still import cleanly. The orchestrator NEVER writes this value today;
# runs only terminate via ``FINISHED``. New code MUST use ``FINISHED``.
FAILED = "finished"

# Linear ordering (idle / finished are bookkeeping, not steps).
STEP_ORDER: List[str] = [S0_ROUGH, S1_ANALYZE, S2_GRAPH, S3_CALCULATE, S4_VISUALIZE]

HUMAN_LABEL = {
    S0_ROUGH: "S0: Rough single-pass estimate",
    S1_ANALYZE: "S1: Analyze code (2 agents)",
    S2_GRAPH: "S2: Build & validate execution graph",
    S3_CALCULATE: "S3: Calculate FLOPs / mem-traffic (2 angles × 5-way parallel)",
    S4_VISUALIZE: "S4: Generate HTML visualization",
    FINISHED: "finished",
    IDLE: "idle",
}


# --------------------------------------------------------------------------- #
# graph_payload — the ONLY function the WebUI's state-graph endpoint calls.
#
# Same protocol shape as gen_infer_framework.orchestrator.phases.graph_payload
# (see there for the full contract). calc_value's state machine is linear,
# so we synthesize a linear node/edge list from STEP_ORDER.
# --------------------------------------------------------------------------- #


def graph_payload(current, last_outcome, last_label) -> Dict[str, object]:
    """Build the state-graph render payload for the WebUI.

    Linear pipeline: one chain of nodes ``S0 → S1 → ... → S4 → finished``
    with a single ``step`` label on every edge.
    """
    nodes = [{"id": p, "label": HUMAN_LABEL.get(p, p)} for p in STEP_ORDER]
    nodes.append({"id": FINISHED, "label": HUMAN_LABEL.get(FINISHED, "finished"),
                  "description": "pipeline complete"})
    edges = []
    chain = STEP_ORDER + [FINISHED]
    for a, b in zip(chain, chain[1:]):
        edges.append({"from": a, "to": b, "label": "step"})
    active_edge = None
    for e in edges:
        if e["to"] == current:
            active_edge = e
            break
    terminal_nodes = [
        {"id": FINISHED, "label": HUMAN_LABEL.get(FINISHED, "finished"),
         "description": "pipeline complete"},
    ]
    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": terminal_nodes,
        "outcome_legend": [],
    }

