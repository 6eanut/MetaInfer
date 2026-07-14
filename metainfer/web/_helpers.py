"""Shared HTTP helpers for the WebUI.

These used to live inline in :mod:`metainfer.web.app`; they're extracted
here so per-task-type plugin routes can use them without circular
imports against ``app.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from . import tasks as _tasks


def task_or_404(task_id: str):
    """Resolve a task_id to its :class:`TaskEntry`, 404 on missing."""
    entry = _tasks.get_task(task_id)
    if entry is None:
        raise HTTPException(404, f"no such task: {task_id}")
    return entry


def state_dir_for(entry) -> Path:
    """Canonical state dir path for a task entry."""
    return Path(entry.state_dir)


def find_events_file(log_dir: Path) -> Optional[Path]:
    """Locate the events.jsonl produced by SubAgentManager for an agent
    whose log_dir is ``log_dir``. Returns the highest-attempt file if
    several attempts exist (the last attempt is the one that produced
    the final result). Returns None if the dir is missing or empty.
    """
    if not log_dir.is_dir():
        return None
    candidates = sorted(log_dir.glob("*.events.jsonl"))
    if not candidates:
        return None
    # Prefer the highest attempt number; fall back to the last lexically.
    def _attempt(p: Path) -> int:
        # filenames look like "<name>.attempt<N>.events.jsonl"
        for part in p.stem.split("."):
            if part.startswith("attempt"):
                try:
                    return int(part[len("attempt"):])
                except ValueError:
                    pass
        return -1
    return max(candidates, key=_attempt)


def require_task_type(entry, type_name: str) -> None:
    """Raise HTTPException(409) if ``entry.type`` doesn't match.

    Used by routes that were migrated from the old monolithic app — we
    keep the guard at the route layer for safety, since the route path
    prefix alone is suggestive but not authoritative.
    """
    if entry.type != type_name:
        raise HTTPException(
            409, f"task is not a {type_name} task (got {entry.type!r})",
        )
