"""Task-specific on-disk state readers.

Every task type that creates iterations / charts / state-graph data on disk
provides reader functions here.  The task's ``routes.py`` calls these to feed
HTTP responses.  The shell's ``metainfer/server/state_reader.py`` is
deliberately kept task-agnostic — it only reads the common envelope files
(run.json, timeline.jsonl, agents.json).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def read_iterations(state_dir: Path) -> list[dict[str, Any]]:
    """Read all iteration records from ``<state_dir>/iterations/*.json``.

    Returns a list of plain dicts (the task's private iteration schema).
    The frontend detail view interprets these fields however it wants.
    """
    iters_dir = state_dir / "iterations"
    if not iters_dir.exists():
        return []
    import json
    records = []
    for p in sorted(iters_dir.glob("*.json")):
        try:
            records.append(json.loads(p.read_text(encoding="utf-8")))
        except (ValueError, OSError):
            continue
    return records


def read_state_graph(state_dir: Path) -> dict[str, Any]:
    """Return the current state-machine snapshot for this task.

    Reads ``run.json`` (via :mod:`metainfer.server.state_reader`) and calls
    the task's own ``graph_payload`` to build the UI-able dict.
    """
    from metainfer.server.state_reader import read_run
    run = read_run(state_dir)
    current = run.get("current_phase", "idle")
    last_outcome = run.get("last_outcome")
    last_label = run.get("last_transition_label")

    # Import from the orchestrator side — task-private phase definition.
    from metainfer.tasks.X.orchestrator.phases import graph_payload
    return graph_payload(current, last_outcome, last_label)
