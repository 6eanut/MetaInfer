"""Task registry: durable list of all tasks the WebUI knows about.

Stored at ``<node_dir>/.metainfer/registry.json`` (see :mod:`metainfer.web.paths`
for the node-rooted layout). Each entry pins a task to its ``state_dir``
(metadata + logs), its ``workspace_dir`` (generated artifacts), the task
type, the launcher used to spawn it, and the last-known PID + status. The
registry is the source of truth for the task list view; everything else
(current phase, iterations, agents, perf, etc.) is read on demand from
the task's ``state_dir``.

Atomic updates via :func:`fcntl.flock` on a sibling lock file so multiple
WebUI processes (or the orchestrator subprocess writing its PID) can
safely read-modify-write.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import paths as _paths


@dataclass
class TaskEntry:
    """One row in the registry."""
    id: str                                 # user-visible task id, unique
    type: str                               # task type id (e.g. "gen-infer-framework")
    label: str                              # short display name
    state_dir: str                          # absolute path to task metadata dir
    created_at: float
    # Absolute path to task's generated-artifacts dir (parallel to
    # state_dir but under <node>/workspaces/). Empty for legacy entries
    # created before the split — callers should fall back to deriving
    # from id via paths.workspace_dir(id) if they need a Path.
    workspace_dir: str = ""
    # Launcher that owns this task: "local" or "remote:<node_id>" (future).
    # "remote:<node_id>" semantics will eventually resolve to a path under
    # <root>/nodes/<node_id>/ on the shared filesystem.
    launcher: str = "local"
    # Last-known orchestrator PID + lifecycle markers. The orchestrator
    # subprocess writes its own PID file under state_dir; the registry
    # caches a copy here for the task list view to render quickly without
    # touching every task dir.
    pid: Optional[int] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


def _read_registry_locked() -> Dict[str, Any]:
    """Read the raw registry dict. Call this while holding the flock."""
    p = _paths.registry_path()
    if not p.exists():
        return {"tasks": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"tasks": []}
    if "tasks" not in data or not isinstance(data["tasks"], list):
        data["tasks"] = []
    return data


def _write_registry_locked(data: Dict[str, Any]) -> None:
    p = _paths.registry_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


def _lock():
    """Context manager that acquires an exclusive flock on the registry
    lock file. Safe across processes. Usage:

        with _lock():
            ...
    """
    from contextlib import contextmanager

    @contextmanager
    def _ctx():
        lock_path = _paths.registry_lock_path()
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        f = open(lock_path, "a+", encoding="utf-8")
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        finally:
            f.close()

    return _ctx()


def list_tasks() -> List[TaskEntry]:
    """Return all tasks in creation order."""
    with _lock():
        data = _read_registry_locked()
    return [TaskEntry(**t) for t in data["tasks"]]


def get_task(task_id: str) -> Optional[TaskEntry]:
    """Return one task by id, or None."""
    with _lock():
        data = _read_registry_locked()
    for t in data["tasks"]:
        if t.get("id") == task_id:
            return TaskEntry(**t)
    return None


def add_task(entry: TaskEntry) -> None:
    """Insert a new task. Raises ValueError if id collides."""
    with _lock():
        data = _read_registry_locked()
        for t in data["tasks"]:
            if t.get("id") == entry.id:
                raise ValueError(f"task id {entry.id!r} already exists")
        data["tasks"].append(asdict(entry))
        _write_registry_locked(data)


def update_task(task_id: str, **patch: Any) -> Optional[TaskEntry]:
    """Patch fields on an existing task (e.g. refresh pid / started_at /
    finished_at). Returns the updated entry, or None if no such task.

    None values in ``patch`` are skipped (treated as 'no change'). Pass
    an explicit sentinel if you need to clear a field."""
    with _lock():
        data = _read_registry_locked()
        for i, t in enumerate(data["tasks"]):
            if t.get("id") == task_id:
                for k, v in patch.items():
                    if v is None:
                        continue
                    t[k] = v
                data["tasks"][i] = t
                _write_registry_locked(data)
                return TaskEntry(**t)
    return None


def remove_task(task_id: str) -> bool:
    """Remove a task from the registry. Returns True if it was present.
    Does NOT delete the state_dir on disk — that's a separate operation
    so it can be confirmed by the user."""
    with _lock():
        data = _read_registry_locked()
        before = len(data["tasks"])
        data["tasks"] = [t for t in data["tasks"] if t.get("id") != task_id]
        if len(data["tasks"]) == before:
            return False
        _write_registry_locked(data)
    return True


def gen_task_id(task_type: str, label: Optional[str] = None) -> str:
    """Generate a unique, readable task id of the form
    ``<slug>-<short-uuid>``. ``task_type`` is the prefix; ``label`` (if
    given) is slugified and prepended for readability."""
    import re
    import uuid
    short = uuid.uuid4().hex[:8]
    if label:
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:32]
        if slug:
            return f"{slug}-{short}"
    # Fall back to type-based prefix
    type_slug = re.sub(r"[^a-z0-9]+", "-", task_type.lower()).strip("-")[:24]
    return f"{type_slug}-{short}"
