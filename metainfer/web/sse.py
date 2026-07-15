"""Server-Sent Events stream for live task updates.

The WebUI server tracks per-task file mtimes on a polling timer (stdlib
only — no watchdog dependency). When any tracked file in a task's
``state_dir`` changes, an event is pushed to every connected SSE client
so the frontend can refetch just the affected panels.

Files watched per task (always):
  - run.json         (phase / iteration transitions)
  - timeline.jsonl   (any append)
  - agents.json      (subagent lifecycle)
  - iterations/*.json (iteration record updates)

Plus per-plugin extra paths declared via
``WebPlugin.extra_watch_paths(entry)`` (e.g. a task package that streams
incremental artifacts into its ``workspace_dir``). This module is
task-type-agnostic — adding a new incremental-progress file for your
task is a plugin-only change, not a public-file edit.

We don't watch the orchestrator.log (too chatty) or code/ (irrelevant to
the dashboard).
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Optional, Set

from . import paths as _paths


# Files (relative to each task's state_dir) whose mtime change should
# trigger a refresh broadcast.
WATCHED_RELPATHS = (
    "run.json",
    "timeline.jsonl",
    "agents.json",
)


def _task_state_dir(task_id: str) -> Path:
    return _paths.task_dir(task_id)


def _scan_task(task_id: str) -> Dict[str, float]:
    """Return a {relative_path: mtime} snapshot of every watched file
    currently present under the task's state_dir. Files that don't exist
    are simply omitted (so missing files don't pollute the diff).

    Two sources:
      1. The always-watched canonical files (``WATCHED_RELPATHS`` +
         ``iterations/*.json``).
      2. Per-plugin extras via ``WebPlugin.extra_watch_paths(entry)`` —
         plugin-supplied paths (under or outside state_dir), keyed by
         their stringified form so the diff is stable across polls.
    """
    from . import tasks as _tasks
    from .registry import get as _get_plugin

    state_dir = _task_state_dir(task_id)
    out: Dict[str, float] = {}
    if not state_dir.exists():
        # Still fall through to the plugin extras — a plugin might
        # watch workspace files even before state_dir is materialized.
        pass
    else:
        for rel in WATCHED_RELPATHS:
            p = state_dir / rel
            if p.exists():
                try:
                    out[rel] = p.stat().st_mtime
                except OSError:
                    pass
        iters_dir = state_dir / "iterations"
        if iters_dir.exists():
            for p in iters_dir.glob("*.json"):
                try:
                    out[f"iterations/{p.name}"] = p.stat().st_mtime
                except OSError:
                    pass

    entry = _tasks.get_task(task_id)
    if entry is None:
        return out
    plugin = _get_plugin(entry.type)
    if plugin is None or plugin.extra_watch_paths is None:
        return out
    try:
        extras = plugin.extra_watch_paths(entry) or []
    except Exception:  # noqa: BLE001 — never let a buggy plugin kill SSE
        return out
    for p in extras:
        if p is None:
            continue
        try:
            p = Path(p)
        except TypeError:
            continue
        if not p.exists():
            continue
        try:
            # Key by absolute path so two extras in different dirs don't collide.
            out[str(p)] = p.stat().st_mtime
        except OSError:
            pass
    return out


class FileWatcher:
    """Polling-based file watcher. Single background asyncio task scans
    every known task every ``interval`` seconds; on diff, pushes events
    to a broadcast queue.

    Designed to survive WebUI restarts: state is rebuilt from disk on
    first scan, so missed events during downtime just look like one big
    initial diff.
    """

    def __init__(self, interval: float = 1.5) -> None:
        self.interval = interval
        self._last: Dict[str, Dict[str, float]] = {}
        self._subs: Set["asyncio.Queue"] = set()
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def subscribe(self) -> "asyncio.Queue":
        q: asyncio.Queue = asyncio.Queue(maxsize=256)
        async with self._lock:
            self._subs.add(q)
        return q

    async def unsubscribe(self, q: "asyncio.Queue") -> None:
        async with self._lock:
            self._subs.discard(q)

    async def _broadcast(self, event: Dict[str, Any]) -> None:
        """Fan-out to every subscriber. Drop on full queue rather than
        block — the SSE client will catch up on the next event."""
        async with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop oldest to make room, then retry.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:  # noqa: BLE001
                    pass

    async def _run(self) -> None:
        while True:
            try:
                await self._scan_once()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 — never let the loop die
                pass
            await asyncio.sleep(self.interval)

    async def _scan_once(self) -> None:
        # Enumerate task ids from the registry (cheap; one file read).
        from . import tasks as _tasks
        all_tasks: Dict[str, float] = {}
        try:
            entries = _tasks.list_tasks()
        except Exception:  # noqa: BLE001
            entries = []
        for entry in entries:
            tid = entry.id
            now = _scan_task(tid)
            prev = self._last.get(tid, {})
            if prev != now:
                # Compute a hint about WHAT changed so the frontend can
                # refresh just the affected panel.
                changed: list = []
                for k, t in now.items():
                    if prev.get(k) != t:
                        changed.append(k)
                for k in prev:
                    if k not in now:
                        changed.append(k)
                self._last[tid] = now
                await self._broadcast({
                    "type": "task_changed",
                    "task_id": tid,
                    "changed": changed,
                    "ts": time.time(),
                })


# Module-level singleton; the FastAPI app uses this.
watcher = FileWatcher()
