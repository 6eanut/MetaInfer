"""Persistent state store for one MetaInfer task.

All state lives under ``<cwd>/.metainfer/state/<task_id>/``:

* ``requirements.json``    — frozen requirements captured by the entry skill
* ``run.json``             — mutable run status (current phase, iteration, ...)
* ``iterations/<n>.json``  — one record per iteration
* ``timeline.jsonl``       — append-only event log (phase transitions, agent events)

The store is intentionally file-based so the WebUI can run in a separate
process and observe state without IPC.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Phase + Outcome are sourced from the state-machine module so there is a
# single source of truth (see :mod:`metainfer.phases`).
from .phases import Phase, Outcome


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


@dataclass
class IterationRecord:
    iteration: int
    goal: str = ""                 # what this iteration is trying to achieve
    start_phase: Phase = "A_plan"  # A or D, depending on outcome of prev iter
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    status: Literal["running", "success", "failed"] = "running"
    failure_reason: Optional[str] = None
    # outcome of the iteration's terminating C step (None until C has run)
    outcome: Optional[Outcome] = None
    phases: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # perf metrics from C step (tokens/s, ms/req, memory MB, etc.)
    perf: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    # True iff the orchestrator process died mid-flight and this record was
    # finalized retroactively on the next resume. Distinguishes a "real"
    # failed C step from an externally-interrupted attempt. The WebUI can
    # use this to render differently (e.g., "interrupted" vs "failed").
    interrupted: bool = False
    # Absolute path to this iteration's retrospective.md (written at the
    # end of E_perf_test). The WebUI reads this file on demand when the
    # user clicks the iteration row. None if E never ran or the retro
    # agent failed to produce the file. Stored as a path (not the full
    # text) so the JSON state file stays small.
    retrospective_path: Optional[str] = None


@dataclass
class RunStatus:
    task_id: str
    task_type: str
    created_at: float
    current_iteration: int = 0
    current_phase: Phase = "idle"
    last_update: float = 0.0
    finished: bool = False
    final_status: Optional[str] = None  # success / failed / aborted
    # last transition the orchestrator took — used by the WebUI to highlight
    # the active edge in the state graph.
    last_outcome: Optional[Outcome] = None
    last_transition_label: Optional[str] = None
    notes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------------- #


class StateStore:
    """File-based state for a single task. Thread-safe via a single RLock."""

    def __init__(self, task_dir: Path) -> None:
        self.task_dir = task_dir
        self.task_dir.mkdir(parents=True, exist_ok=True)
        (self.task_dir / "iterations").mkdir(exist_ok=True)
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #

    @property
    def requirements_path(self) -> Path:
        return self.task_dir / "requirements.json"

    @property
    def run_path(self) -> Path:
        return self.task_dir / "run.json"

    @property
    def timeline_path(self) -> Path:
        return self.task_dir / "timeline.jsonl"

    def iter_path(self, n: int) -> Path:
        return self.task_dir / "iterations" / f"{n:03d}.json"

    def interrupted_iter_path(self, n: int) -> Path:
        """Where to archive an iteration's record when the orchestrator was
        interrupted mid-flight. Sibling to :meth:`iter_path`; same glob
        pattern picks it up for display."""
        return self.task_dir / "iterations" / f"{n:03d}.interrupted.json"

    # ------------------------------------------------------------------ #
    # Requirements
    # ------------------------------------------------------------------ #

    def load_requirements(self) -> Dict[str, Any]:
        return json.loads(self.requirements_path.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ #
    # Run status
    # ------------------------------------------------------------------ #

    def init_run(self, task_id: str, task_type: str) -> RunStatus:
        rs = RunStatus(
            task_id=task_id,
            task_type=task_type,
            created_at=time.time(),
            last_update=time.time(),
        )
        self._write_run(rs)
        return rs

    def init_or_resume(self, task_id: str, task_type: str) -> tuple[RunStatus, bool]:
        """Either initialize a fresh run.json or load the existing one.

        Returns ``(run_status, is_resume)``. ``is_resume`` is True iff a
        ``run.json`` already existed on disk (i.e. the orchestrator has
        run before for this task).
        """
        with self._lock:
            if self.run_path.exists():
                return self.load_run(), True
            return self.init_run(task_id, task_type), False

    def load_run(self) -> RunStatus:
        if not self.run_path.exists():
            raise FileNotFoundError(f"no run.json at {self.run_path}")
        data = json.loads(self.run_path.read_text(encoding="utf-8"))
        return RunStatus(**data)

    def update_run(self, **kwargs: Any) -> RunStatus:
        with self._lock:
            rs = self.load_run()
            for k, v in kwargs.items():
                if hasattr(rs, k):
                    setattr(rs, k, v)
            rs.last_update = time.time()
            self._write_run(rs)
            return rs

    def _write_run(self, rs: RunStatus) -> None:
        tmp = self.run_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(rs), indent=2), encoding="utf-8")
        os.replace(tmp, self.run_path)

    # ------------------------------------------------------------------ #
    # Iteration records
    # ------------------------------------------------------------------ #

    def write_iteration(self, rec: IterationRecord) -> None:
        with self._lock:
            path = self.iter_path(rec.iteration)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(rec), indent=2), encoding="utf-8")
            os.replace(tmp, path)

    def load_iteration(self, n: int) -> Optional[IterationRecord]:
        path = self.iter_path(n)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return IterationRecord(**data)

    def load_all_iterations(self) -> List[IterationRecord]:
        recs = []
        for p in sorted((self.task_dir / "iterations").glob("*.json")):
            data = json.loads(p.read_text(encoding="utf-8"))
            recs.append(IterationRecord(**data))
        return recs

    def delete_iteration(self, n: int) -> bool:
        """Remove the iteration record JSON for ``n``. Used during resume
        to discard an iteration whose folder was incomplete. Returns True
        if a file was removed."""
        with self._lock:
            p = self.iter_path(n)
            if p.exists():
                p.unlink()
                return True
            return False

    def archive_interrupted_iteration(
        self,
        n: int,
        reason: str = "interrupted: orchestrator process exited unexpectedly",
    ) -> bool:
        """Finalize iteration ``n``'s record as failed/interrupted and move
        it aside so the retry can reuse the slot.

        Loads the existing record (if any), stamps ``status="failed"`` with
        ``failure_reason=reason`` and ``ended_at=now``, writes it to
        :meth:`interrupted_iter_path`, then deletes the live record. The
        archived file is still discovered by :meth:`load_all_iterations`
        (same ``*.json`` glob), so the WebUI shows the interrupted attempt
        in the history with its fail reason — instead of silently
        disappearing or, worse, showing as "running".

        Returns True if a record existed and was archived.
        """
        with self._lock:
            src = self.iter_path(n)
            if not src.exists():
                return False
            data = json.loads(src.read_text(encoding="utf-8"))
            now = time.time()
            data.setdefault("status", "failed")
            data["status"] = "failed"
            data["failure_reason"] = reason
            data["ended_at"] = data.get("ended_at") or now
            data["duration_s"] = max(0.0, data["ended_at"] - data.get("started_at", now))
            # marker so the UI / downstream readers can tell interrupted
            # apart from a "real" failed C step. Optional; not enforced.
            data["interrupted"] = True
            dst = self.interrupted_iter_path(n)
            tmp = dst.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(dst)
            src.unlink()
            return True

    # ------------------------------------------------------------------ #
    # Timeline (append-only)
    # ------------------------------------------------------------------ #

    def append_timeline(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        with self._lock:
            entry = {
                "ts": time.time(),
                "type": event_type,
                "payload": payload or {},
            }
            with open(self.timeline_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def load_timeline(self, since: float = 0.0) -> List[Dict[str, Any]]:
        if not self.timeline_path.exists():
            return []
        out = []
        for ln in self.timeline_path.read_text(encoding="utf-8").splitlines():
            try:
                ev = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if ev.get("ts", 0) >= since:
                out.append(ev)
        return out
