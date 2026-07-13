"""Phase definitions for the calc-theoretical-value pipeline.

Linear state machine::

    idle → S1_analyze → S2_graph → S3_calculate → S4_visualize → finished

Each step retries forever on failure (see ``pipeline._run_step_with_retry``);
there is NO ``failed`` phase. The only terminal phase is ``finished``,
which covers success, stopped (e.g. invalid inputs), and externally
interrupted runs.

These phase strings are stored in ``run.json.current_phase`` and rendered
by the WebUI. The frontend's task-detail view checks ``task_type`` to
switch to the calc-value-specific visualization when needed.
"""

from __future__ import annotations

from typing import List

# Phase strings — written to run.json. Stored as plain strings (not a
# Literal) so that the rest of the orchestrator package, which shares
# state.py with the ABCDEF pipeline, doesn't need its Phase Literal
# extended. state.py treats current_phase as opaque.
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
STEP_ORDER: List[str] = [S1_ANALYZE, S2_GRAPH, S3_CALCULATE, S4_VISUALIZE]

HUMAN_LABEL = {
    S1_ANALYZE: "S1: Analyze code (3 agents)",
    S2_GRAPH: "S2: Build & validate execution graph",
    S3_CALCULATE: "S3: Calculate FLOPs / mem-traffic (3 agents × 42 combos)",
    S4_VISUALIZE: "S4: Generate HTML visualization",
    FINISHED: "finished",
    IDLE: "idle",
}

