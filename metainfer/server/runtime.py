"""Live WebUI runtime state, stored at ``<cwd>/.metainfer/runtime.json``
(or ``$METAINFER_HOME/runtime.json``).

This is the WebUI's own scratchpad of "what I think is running right now".
It is *not* the source of truth — that lives on disk in each task's
``orchestrator.pid`` file, and in the OS process table. ``runtime.json``
exists so that when the WebUI crashes and restarts, it can reconcile
its in-memory picture against the actual process table and:

  - **adopt** orchestrators that are still running but that this fresh
    WebUI instance didn't spawn (orphans from the previous session),
  - **mark dead** registry entries whose orchestrator process has
    disappeared, so the UI doesn't keep hoping,
  - **detect** PID reuse (a stale runtime entry whose recorded PID now
    belongs to an unrelated process).

Schema::

    {
      "webui": {
        "pid": 12345,
        "boot_id": "abcd1234",       # random per WebUI session
        "started_at": 1720000000,
        "hostname": "k100-01"
      },
      "tasks": {
        "<task_id>": {
          "pid": 12346,
          "started_at": 1720000010,  # orchestrator start time, for PID reuse check
          "boot_id": "abcd1234",     # which WebUI session spawned it
          "state_dir": "/.../..."
        }
      }
    }

All updates go through an flock on ``runtime.lock`` and use atomic
``tmp + replace`` writes. Multiple WebUI instances can therefore share
the same ``METAINFER_HOME`` safely.
"""

from __future__ import annotations

import fcntl
import json
import os
import socket
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from . import paths as _paths


# --------------------------------------------------------------------------- #
# Locking + raw IO
# --------------------------------------------------------------------------- #

@contextmanager
def _locked() -> Iterator[None]:
    lock_path = _paths.runtime_lock_path()
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


def _read_raw() -> Dict[str, Any]:
    p = _paths.runtime_path()
    if not p.exists():
        return {"webui": None, "tasks": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"webui": None, "tasks": {}}
    if not isinstance(data, dict):
        return {"webui": None, "tasks": {}}
    data.setdefault("webui", None)
    data.setdefault("tasks", {})
    if not isinstance(data["tasks"], dict):
        data["tasks"] = {}
    return data


def _write_raw(data: Dict[str, Any]) -> None:
    p = _paths.runtime_path()
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(p)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def read_state() -> Dict[str, Any]:
    """Return a copy of the current runtime state. Safe to call from
    anywhere; takes the flock briefly."""
    with _locked():
        return _read_raw()


def get_task_entry(task_id: str) -> Optional[Dict[str, Any]]:
    """Return the runtime entry for one task, or None."""
    with _locked():
        return _read_raw()["tasks"].get(task_id)


def record_webui_start() -> str:
    """Stamp this WebUI session into runtime.json. Returns the new
    ``boot_id`` (a random uuid4 hex prefix). Called once at app startup."""
    boot_id = uuid.uuid4().hex[:12]
    payload = {
        "pid": os.getpid(),
        "boot_id": boot_id,
        "started_at": time.time(),
        "hostname": socket.gethostname(),
    }
    with _locked():
        data = _read_raw()
        data["webui"] = payload
        _write_raw(data)
    return boot_id


def record_webui_exit() -> None:
    """Clear the webui entry on clean shutdown. Best-effort; if the
    WebUI crashes, the stale entry will be replaced on next start
    anyway (see :func:`record_webui_start`)."""
    with _locked():
        data = _read_raw()
        data["webui"] = None
        _write_raw(data)


def record_task_spawn(
    task_id: str, pid: int, state_dir: Path, boot_id: str,
    started_at: Optional[float] = None,
) -> None:
    """Record that this WebUI session (``boot_id``) spawned an
    orchestrator with the given PID for ``task_id``. ``started_at``
    defaults to now; it should match the orchestrator's actual process
    start time (read from /proc/<pid>/stat) so PID reuse can be
    detected later."""
    with _locked():
        data = _read_raw()
        data["tasks"][task_id] = {
            "pid": pid,
            "started_at": started_at or time.time(),
            "boot_id": boot_id,
            "state_dir": str(state_dir),
        }
        _write_raw(data)


def update_task(task_id: str, **patch: Any) -> None:
    """Patch fields on a runtime task entry (e.g. update started_at once
    the orchestrator has written its pid file). No-op if the task isn't
    in runtime. None-valued patches are skipped."""
    with _locked():
        data = _read_raw()
        entry = data["tasks"].get(task_id)
        if not entry:
            return
        for k, v in patch.items():
            if v is None:
                continue
            entry[k] = v
        data["tasks"][task_id] = entry
        _write_raw(data)


def clear_task(task_id: str) -> None:
    """Remove a task from runtime state (after orchestrator exit)."""
    with _locked():
        data = _read_raw()
        if task_id in data["tasks"]:
            del data["tasks"][task_id]
            _write_raw(data)


def replace_tasks(new_tasks: Dict[str, Any]) -> None:
    """Atomically replace the entire ``tasks`` dict. Used by
    reconciliation after startup scan."""
    with _locked():
        data = _read_raw()
        data["tasks"] = dict(new_tasks)
        _write_raw(data)
