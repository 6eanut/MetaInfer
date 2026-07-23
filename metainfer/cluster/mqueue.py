"""File-system message queue: submit → consume → result.

Authority sources & ownership:

- ``cluster/inbox/<worker_id>/<job_id>/`` — **owned by submitter (orchestrator)**,
  consumed by exactly one worker. Contains ``job.json``, ``script.sh`` or
  ``prompt.txt``, ``stdout.log`` / ``stderr.log`` (appended by worker),
  ``status.json``, and optional ``cancel.marker``.

- ``cluster/replies/<orchestrator_id>/<job_id>.result.json`` — **owned by worker**
  once it finishes the job. Orchestrator polls this file.

Robustness guarantees (see plan decision 5, 6):

- Submit uses tmp dir + rename → consumer never sees a partial job directory.
- Consume uses ``link_claim`` on ``<job_dir>/claimed`` → exactly one worker wins.
- Result is tmp + ``os.replace`` → orchestrator sees either old or new, never partial.
- Orphan recovery (orchestrator-side): if a job's ``timeout_s + grace`` elapses
  with no result file, the orchestrator reaps the worker's claims via scoreboard
  and writes a synthetic ``status=worker_dead`` or ``status=timeout`` result.
  **No auto-requeue** (decision: surface as failure).
"""

from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import fs_primitives
from . import paths
from .queue_schema import (
    JobHandle,
    JobResult,
    JobSpec,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_PENDING,
    STATUS_TIMEOUT,
    STATUS_WORKER_DEAD,
    new_job_id,
    now_ts,
)


GRACE_S = 30.0  # extra seconds past timeout_s before orchestrator declares worker_dead


# --------------------------------------------------------------------------- #
# Submit
# --------------------------------------------------------------------------- #
def submit_job(spec: JobSpec) -> str:
    """Atomically enqueue ``spec`` to its target worker's inbox.

    Fills in ``job_id`` (if missing), ``submitted_at``, and ``reply_to`` (the
    absolute path to the future result.json under the submitter's replies dir).
    """
    if not spec.job_id:
        spec.job_id = new_job_id()
    if not spec.submitter:
        raise ValueError("JobSpec.submitter must be set (orchestrator node_id)")
    if not spec.worker_node_id:
        raise ValueError("JobSpec.worker_node_id must be set")
    spec.validate()
    spec.submitted_at = now_ts()
    spec.reply_to = str(paths.result_path(spec.submitter, spec.job_id))

    jdir = paths.job_dir(spec.worker_node_id, spec.job_id)
    jdir.parent.mkdir(parents=True, exist_ok=True)

    # Build in tmp dir, then rename atomically so consumer never sees a half-built job.
    tmp_dir = jdir.with_name(f".{jdir.name}.{uuid.uuid4().hex}.tmp")
    tmp_dir.mkdir(parents=True)
    try:
        # job.json
        (tmp_dir / "job.json").write_text(__import__("json").dumps(spec.to_dict(), indent=2))
        # script or prompt file
        if spec.type == "script":
            (tmp_dir / "script.sh").write_text(spec.script_body)
        else:
            (tmp_dir / "prompt.txt").write_text(spec.prompt_body)
        # Initial status.json
        fs_primitives.atomic_write_json(tmp_dir / "status.json",
                                        {"status": STATUS_PENDING, "ts": now_ts()})
        os.replace(tmp_dir, jdir)
    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return spec.job_id


def read_job(worker_node_id: str, job_id: str) -> Optional[JobSpec]:
    """Read a job's spec from disk (read-only)."""
    jdir = paths.job_dir(worker_node_id, job_id)
    job_json = jdir / "job.json"
    if not job_json.exists():
        return None
    try:
        import json
        return JobSpec.from_dict(json.loads(job_json.read_text()))
    except (OSError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Consume (worker side)
# --------------------------------------------------------------------------- #
def consume_next_job(worker_node_id: str, worker_pid: int) -> Optional[JobHandle]:
    """Poll ``inbox/<worker_node_id>/`` and claim the next available job.

    Returns a :class:`JobHandle` on success, or None if no pending jobs.

    Idempotent: re-running on the same job after crash is safe — the
    ``claimed`` marker persists; subsequent calls skip already-claimed jobs.
    """
    idir = paths.inbox_dir(worker_node_id)
    for entry in sorted(idir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        claimed_marker = paths.job_claimed_marker(entry)
        # Skip jobs that are already claimed.
        if claimed_marker.exists():
            continue
        # Skip cancelled jobs (force-release wrote cancel.marker before we got here).
        if paths.job_cancel_marker(entry).exists():
            continue
        payload = {"worker_pid": worker_pid, "claimed_at": now_ts()}
        if fs_primitives.link_claim(claimed_marker, payload):
            spec = read_job(worker_node_id, entry.name)
            if spec is None:
                # Corrupt or vanished job.json — release claim, move on.
                try:
                    claimed_marker.unlink()
                except FileNotFoundError:
                    pass
                continue
            fs_primitives.atomic_write_json(
                entry / "status.json",
                {"status": "inflight", "ts": now_ts(), "worker_pid": worker_pid},
            )
            return JobHandle(spec=spec, job_dir=str(entry),
                             claimed_at=payload["claimed_at"], worker_pid=worker_pid)
    return None


def is_cancelled(job_dir: str | Path) -> bool:
    return paths.job_cancel_marker(Path(job_dir)).exists()


# --------------------------------------------------------------------------- #
# Result (worker writes, orchestrator reads)
# --------------------------------------------------------------------------- #
def write_result(handle: JobHandle, result: JobResult) -> None:
    """Atomically write the result to the reply path given in the spec.

    Caller is responsible for filling ``result.job_id``. ``status.json`` in the
    job dir is also updated so WebUI/admin can see the final state alongside logs.
    """
    result.job_id = handle.spec.job_id
    reply_to = Path(handle.spec.reply_to)
    fs_primitives.atomic_write_json(reply_to, result.to_dict())
    # Also update job-dir status.json for admin visibility.
    fs_primitives.atomic_write_json(
        Path(handle.job_dir) / "status.json",
        {"status": result.status, "ts": now_ts(),
         "exit_code": result.exit_code, "signal": result.signal},
    )


def read_result(job_id: str, orchestrator_node_id: str,
                timeout_s: float = 0.0, poll_interval_s: float = 0.5) -> Optional[JobResult]:
    """Poll for a result. If ``timeout_s`` <= 0, do a single non-blocking read.

    Returns None on timeout.
    """
    rpath = paths.result_path(orchestrator_node_id, job_id)
    deadline = time.time() + timeout_s if timeout_s > 0 else 0.0
    while True:
        data = fs_primitives.read_claim(rpath)
        if data is not None:
            try:
                return JobResult.from_dict(data)
            except (ValueError, TypeError):
                return None
        if timeout_s <= 0:
            return None
        if time.time() >= deadline:
            return None
        time.sleep(poll_interval_s)


# --------------------------------------------------------------------------- #
# Reset (admin / debug)
# --------------------------------------------------------------------------- #
def reset_queue(worker_node_id: str) -> None:
    """Delete all jobs in a worker's inbox. Use with caution — affects all
    orchestrators that submitted to this worker."""
    idir = paths.inbox_dir(worker_node_id)
    if not idir.exists():
        return
    # Wipe and recreate.
    shutil.rmtree(idir, ignore_errors=True)
    idir.mkdir(parents=True, exist_ok=True)


def reset_reply_queue(orchestrator_node_id: str) -> None:
    """Delete all results addressed to an orchestrator."""
    rdir = paths.replies_dir(orchestrator_node_id)
    if not rdir.exists():
        return
    shutil.rmtree(rdir, ignore_errors=True)
    rdir.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Orphan recovery (orchestrator side)
# --------------------------------------------------------------------------- #
def reap_orphaned_submissions(
    orchestrator_node_id: str,
    from_scoreboard_releaser=None,
    grace_s: float = GRACE_S,
) -> List[str]:
    """Scan all workers' inboxes for jobs submitted by ``orchestrator_node_id``
    that are past their timeout and still have no result. For each:

      - release the GPU slots via the provided ``from_scoreboard_releaser`` (if
        given) — typically :func:`metainfer.cluster.scoreboard.release_gpus`
        with the orchestrator's stored token. If no token is available, the
        reaper falls back to unconditional force_release.
      - write a synthetic result with ``status=worker_dead`` (if worker
        heartbeat is stale) or ``status=timeout`` (worker alive but unresponsive).

    Returns the list of reaped job_ids.

    This is **not** auto-requeue — jobs surfaced here are gone; the orchestrator
    decides whether to retry at the application layer.
    """
    from . import worker_registry
    from . import scoreboard

    reaped: List[str] = []
    # We don't have the original LeaseToken at this layer — callers (typically the
    # SDK) keep tokens in memory. Orphan reaping is a best-effort safety net that
    # force-releases slots unconditionally when worker heartbeat is stale.
    inbox_root = paths.inbox_root()
    if not inbox_root.exists():
        return reaped

    now = now_ts()
    for worker_dir in inbox_root.iterdir():
        if not worker_dir.is_dir():
            continue
        worker_id = worker_dir.name
        worker_alive = worker_registry.is_worker_alive(worker_id)
        for job_entry in list(worker_dir.iterdir()):
            if not job_entry.is_dir() or job_entry.name.startswith("."):
                continue
            job_json = job_entry / "job.json"
            if not job_json.exists():
                continue
            try:
                import json
                spec = JobSpec.from_dict(json.loads(job_json.read_text()))
            except (OSError, ValueError):
                continue
            if spec.submitter != orchestrator_node_id:
                continue
            # Already has a result?
            if paths.result_path(orchestrator_node_id, spec.job_id).exists():
                continue
            # Past timeout?
            if now < spec.submitted_at + spec.timeout_s + grace_s:
                continue
            # Stale — force-release slots.
            for slot in spec.gpu_slots:
                scoreboard.force_release(tuple(slot), reason="orphan-reaper")
            # Synthetic result.
            status = STATUS_WORKER_DEAD if not worker_alive else STATUS_TIMEOUT
            result = JobResult(
                job_id=spec.job_id, status=status, duration_s=now - spec.submitted_at,
                error=f"orchestrator-side reap: worker_alive={worker_alive}",
            )
            fs_primitives.atomic_write_json(
                paths.result_path(orchestrator_node_id, spec.job_id),
                result.to_dict(),
            )
            reaped.append(spec.job_id)
    return reaped


# --------------------------------------------------------------------------- #
# Job listing (for webui)
# --------------------------------------------------------------------------- #
def list_jobs(worker_node_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Snapshot of jobs across one or all workers' inboxes."""
    out: List[Dict[str, Any]] = []
    if worker_node_id is not None:
        worker_dirs = [paths.inbox_dir(worker_node_id)]
    else:
        iroot = paths.inbox_root()
        worker_dirs = [d for d in iroot.iterdir() if d.is_dir()] if iroot.exists() else []

    for worker_dir in worker_dirs:
        if not worker_dir.exists() or not worker_dir.is_dir():
            continue
        for job_entry in worker_dir.iterdir():
            if not job_entry.is_dir() or job_entry.name.startswith("."):
                continue
            spec = read_job(worker_dir.name, job_entry.name)
            if spec is None:
                continue
            status_data = fs_primitives.read_claim(job_entry / "status.json") or {}
            out.append({
                "job_id": spec.job_id,
                "worker_node_id": worker_dir.name,
                "submitter": spec.submitter,
                "type": spec.type,
                "submitted_at": spec.submitted_at,
                "submitted_ago_s": max(0.0, now_ts() - spec.submitted_at),
                "status": status_data.get("status", STATUS_PENDING),
                "claimed": paths.job_claimed_marker(job_entry).exists(),
                "cancelled": paths.job_cancel_marker(job_entry).exists(),
                "timeout_s": spec.timeout_s,
            })
    return out
