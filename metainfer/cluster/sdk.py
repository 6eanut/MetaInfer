"""High-level Python SDK for orchestrators and Agents.

Wraps the lower-level primitives into one-call patterns:

- :class:`RemoteJob` — context manager that acquires GPU slots, submits a job,
  awaits result, and always releases in ``finally``. The recommended primitive.

- :func:`submit_script` / :func:`submit_agent` — sugar over RemoteJob.

- :func:`submit_pp2_ranks` — convenience for the PP2 (pipeline-parallel 2-rank)
  pattern: submits two jobs to two workers simultaneously with torch.distributed
  rendezvous env injected (RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT).

- :func:`tail_stdout` / :func:`tail_stderr` — live log tailing.

Typical orchestrator usage::

    from metainfer.cluster.sdk import submit_script

    result = submit_script(
        worker_node_id="workerA",
        gpu_slots=[("workerA", 0)],
        script_body="nvidia-smi && exit 0",
        timeout_s=120,
    )
    if result.status == "done" and result.exit_code == 0:
        ...

See ``docs/agent-sdk-guide.md`` for the full cookbook.
"""

from __future__ import annotations

import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import mqueue, paths, scoreboard
from .queue_schema import JobResult, JobSpec
from .scoreboard import LeaseToken, Slot


# Default polling interval for read_result. Lower = snappier but more CPU.
DEFAULT_POLL_INTERVAL_S = 0.5


# --------------------------------------------------------------------------- #
# Orchestrator node identity
# --------------------------------------------------------------------------- #
def _orchestrator_node_id() -> str:
    """Best-effort node_id for the orchestrator process. Used as the holder
    for scoreboard claims and the reply_to address.

    Honors ``METAINFER_NODE_ID`` (consistent with metainfer.server.paths.node_id).
    """
    import os
    return os.environ.get("METAINFER_NODE_ID") or socket.gethostname()


# --------------------------------------------------------------------------- #
# RemoteJob context manager
# --------------------------------------------------------------------------- #
class RemoteJob:
    """Context manager wrapping acquire → submit → await → release.

    Usage::

        with RemoteJob(spec) as awaited:
            result = awaited.result  # blocks until worker writes result

    Or with explicit polling::

        with RemoteJob(spec, block=False) as job:
            while not job.result_ready():
                tail = job.tail_stdout()
                ...
            result = job.collect_result()

    On context exit (or exception), the GPU slots are always released via
    :func:`scoreboard.release_gpus`.
    """

    def __init__(self, spec: JobSpec, holder: Optional[str] = None,
                 lease_s: float = scoreboard.DEFAULT_LEASE_S,
                 acquire_deadline_s: float = scoreboard.DEFAULT_DEADLINE_S,
                 block: bool = True,
                 poll_interval_s: float = DEFAULT_POLL_INTERVAL_S) -> None:
        # Fill in submitter if missing
        if not spec.submitter:
            spec.submitter = _orchestrator_node_id()
        if not spec.worker_node_id:
            raise ValueError("JobSpec.worker_node_id must be set")
        self.spec = spec
        self.holder = holder or spec.worker_node_id  # default: worker holds the slot
        self.lease_s = lease_s
        self.acquire_deadline_s = acquire_deadline_s
        self.block = block
        self.poll_interval_s = poll_interval_s

        self._token: Optional[LeaseToken] = None
        self._job_id: Optional[str] = None
        self._result: Optional[JobResult] = None
        self._slots_acquired: List[Slot] = []

    def __enter__(self) -> "RemoteJob":
        # 1. Acquire GPU slots (all-or-nothing).
        if self.spec.gpu_slots:
            self._token = scoreboard.acquire_gpus(
                self.spec.gpu_slots, holder=self.holder,
                job_id="<pending>", lease_s=self.lease_s,
                deadline_s=self.acquire_deadline_s,
            )
            if self._token is None:
                raise TimeoutError(
                    f"could not acquire GPU slots {self.spec.gpu_slots} "
                    f"within {self.acquire_deadline_s}s"
                )
            self._slots_acquired = list(self._token.slots)
            # Update job_id in claim metadata once we have it
            self.spec.gpu_slots = list(self._token.slots)

        # 2. Submit job.
        self._job_id = mqueue.submit_job(self.spec)
        # Backfill job_id into the claim metadata (best-effort)
        if self._token is not None:
            self._token.job_id = self._job_id

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._release_slots()

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #
    @property
    def job_id(self) -> str:
        """The job_id assigned at submit time. Raises if called pre-__enter__."""
        if self._job_id is None:
            raise RuntimeError("job_id not available until __enter__")
        return self._job_id

    @property
    def token(self) -> Optional[LeaseToken]:
        """The LeaseToken for acquired GPU slots (None if no slots or pre-enter)."""
        return self._token

    def result_ready(self) -> bool:
        """Non-blocking check: has the worker written a result yet?"""
        if self._result is not None:
            return True
        r = mqueue.read_result(self.job_id, self.spec.submitter)
        if r is not None:
            self._result = r
            return True
        return False

    def collect_result(self, timeout_s: float = 0.0) -> Optional[JobResult]:
        """Block up to ``timeout_s`` for the result. Returns None on timeout.

        While waiting, periodically runs the orchestrator-side orphan reaper
        so that a dead worker surfaces as ``status=worker_dead`` rather than
        blocking until the timeout elapses.
        """
        if self._result is not None:
            return self._result
        deadline = time.time() + timeout_s if timeout_s > 0 else 0.0
        last_reap = 0.0
        while True:
            r = mqueue.read_result(self.job_id, self.spec.submitter,
                                   timeout_s=0,
                                   poll_interval_s=self.poll_interval_s)
            if r is not None:
                self._result = r
                return r
            # Run orphan reaper every ~5s to surface dead-worker results.
            now = time.time()
            if now - last_reap >= 5.0:
                try:
                    mqueue.reap_orphaned_submissions(self.spec.submitter, grace_s=0.0)
                except Exception:  # noqa: BLE001 — reaper is best-effort
                    pass
                last_reap = now
            if timeout_s <= 0:
                return None
            if now >= deadline:
                return None
            time.sleep(self.poll_interval_s)

    def tail_stdout(self, offset: int = 0) -> bytes:
        """Read stdout.log from ``offset``. Returns bytes (may be empty)."""
        return tail_stdout(self.job_id, self.spec.worker_node_id, offset)

    def tail_stderr(self, offset: int = 0) -> bytes:
        """Read stderr.log from ``offset``. Returns bytes (may be empty)."""
        return tail_stderr(self.job_id, self.spec.worker_node_id, offset)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _release_slots(self) -> None:
        if self._token is None:
            return
        try:
            scoreboard.release_gpus(self._token)
        finally:
            self._token = None
            self._slots_acquired = []


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def submit_script(
    worker_node_id: str,
    script_body: str,
    gpu_slots: Optional[Iterable[Slot]] = None,
    timeout_s: float = 1800.0,
    env: Optional[Dict[str, str]] = None,
    cwd: str = "",
    holder: Optional[str] = None,
    lease_s: float = scoreboard.DEFAULT_LEASE_S,
    acquire_deadline_s: float = scoreboard.DEFAULT_DEADLINE_S,
    block: bool = True,
) -> Tuple[str, Optional[JobResult]]:
    """Submit a script job and (optionally) block for the result.

    Returns ``(job_id, result)``. ``result`` is None if ``block=False`` or if
    the worker hasn't produced a result yet (shouldn't happen with block=True
    and reasonable timeout).
    """
    spec = JobSpec(
        type="script",
        worker_node_id=worker_node_id,
        script_body=script_body,
        gpu_slots=list(gpu_slots) if gpu_slots else [],
        timeout_s=timeout_s,
        env=env or {},
        cwd=cwd,
    )
    with RemoteJob(spec, holder=holder, lease_s=lease_s,
                   acquire_deadline_s=acquire_deadline_s, block=block) as job:
        if not block:
            return job.job_id, None
        # Block up to timeout + grace for the worker to finish
        result = job.collect_result(timeout_s=timeout_s + 60)
        return job.job_id, result


def submit_agent(
    worker_node_id: str,
    prompt_body: str,
    gpu_slots: Optional[Iterable[Slot]] = None,
    timeout_s: float = 1800.0,
    env: Optional[Dict[str, str]] = None,
    cwd: str = "",
    holder: Optional[str] = None,
    lease_s: float = scoreboard.DEFAULT_LEASE_S,
    acquire_deadline_s: float = scoreboard.DEFAULT_DEADLINE_S,
    block: bool = True,
) -> Tuple[str, Optional[JobResult]]:
    """Submit an agent job (worker runs ccb with the prompt). See :func:`submit_script`."""
    spec = JobSpec(
        type="agent",
        worker_node_id=worker_node_id,
        prompt_body=prompt_body,
        gpu_slots=list(gpu_slots) if gpu_slots else [],
        timeout_s=timeout_s,
        env=env or {},
        cwd=cwd,
    )
    with RemoteJob(spec, holder=holder, lease_s=lease_s,
                   acquire_deadline_s=acquire_deadline_s, block=block) as job:
        if not block:
            return job.job_id, None
        result = job.collect_result(timeout_s=timeout_s + 60)
        return job.job_id, result


# --------------------------------------------------------------------------- #
# PP2 multi-rank convenience
# --------------------------------------------------------------------------- #
@dataclass
class PP2RankSpec:
    """Per-rank spec for ``submit_pp2_ranks``: which worker + GPU runs this rank,
    and the shell command to launch it. The SDK injects ``RANK``/``WORLD_SIZE``/
    ``MASTER_ADDR``/``MASTER_PORT`` env vars automatically.
    """
    worker_node_id: str
    gpu_idx: int
    command: str  # shell command to run as this rank (passed as script body)


def submit_pp2_ranks(
    rank0: PP2RankSpec,
    rank1: PP2RankSpec,
    master_port: int = 29500,
    master_addr: Optional[str] = None,
    timeout_s: float = 1800.0,
    extra_env: Optional[Dict[str, str]] = None,
    holder: Optional[str] = None,
) -> Tuple[str, str, Optional[JobResult], Optional[JobResult]]:
    """Submit two rank jobs simultaneously for PP2 distributed testing.

    Both jobs get torch.distributed rendezvous env injected:
      RANK / NODE_RANK / LOCAL_RANK / WORLD_SIZE / NNODES / NPROC_PER_NODE
      MASTER_ADDR (= rank0's worker hostname, unless overridden)
      MASTER_PORT

    Jobs are submitted non-blocking; then both are awaited. Returns
    ``(job_id_0, job_id_1, result_0, result_1)``.

    Per the user's design choice, the ranks are NOT synchronized by the
    orchestrator — the script on each worker is responsible for retrying
    ``init_process_group`` until both sides connect.
    """
    rank0_worker_hostname = master_addr or _lookup_hostname(rank0.worker_node_id)
    common_env = {
        "WORLD_SIZE": "2",
        "NNODES": "2",
        "NPROC_PER_NODE": "1",
        "MASTER_ADDR": rank0_worker_hostname,
        "MASTER_PORT": str(master_port),
        **(extra_env or {}),
    }

    def _build(rank: PP2RankSpec, rank_idx: int) -> Tuple[JobSpec, Dict[str, str]]:
        env = dict(common_env)
        env["RANK"] = str(rank_idx)
        env["NODE_RANK"] = str(rank_idx)
        env["LOCAL_RANK"] = "0"
        spec = JobSpec(
            type="script",
            worker_node_id=rank.worker_node_id,
            script_body=rank.command,
            gpu_slots=[(rank.worker_node_id, rank.gpu_idx)],
            timeout_s=timeout_s,
            env=env,
        )
        return spec, env

    spec0, _ = _build(rank0, 0)
    spec1, _ = _build(rank1, 1)

    # Submit both as RemoteJobs (acquires slots), then await both.
    job0 = RemoteJob(spec0, holder=holder)
    job1 = RemoteJob(spec1, holder=holder)
    # Enter both (acquires slots + submits)
    job0.__enter__()
    job1.__enter__()
    try:
        r0 = job0.collect_result(timeout_s=timeout_s + 60)
        r1 = job1.collect_result(timeout_s=timeout_s + 60)
        return job0.job_id, job1.job_id, r0, r1
    finally:
        job0.__exit__(None, None, None)
        job1.__exit__(None, None, None)


def _lookup_hostname(worker_node_id: str) -> str:
    """Look up a worker's hostname from the registry."""
    from . import worker_registry
    rec = worker_registry.read_worker(worker_node_id)
    if rec is None:
        return worker_node_id
    return rec.hostname or worker_node_id


# --------------------------------------------------------------------------- #
# Log tailing
# --------------------------------------------------------------------------- #
def tail_stdout(job_id: str, worker_node_id: str, offset: int = 0) -> bytes:
    """Read ``<job_dir>/stdout.log`` from ``offset``. Returns b"" if missing."""
    p = paths.job_dir(worker_node_id, job_id) / "stdout.log"
    return _read_from_offset(p, offset)


def tail_stderr(job_id: str, worker_node_id: str, offset: int = 0) -> bytes:
    """Read ``<job_dir>/stderr.log`` from ``offset``. Returns b"" if missing."""
    p = paths.job_dir(worker_node_id, job_id) / "stderr.log"
    return _read_from_offset(p, offset)


def _read_from_offset(path: Path, offset: int) -> bytes:
    try:
        size = path.stat().st_size
    except OSError:
        return b""
    if size <= offset:
        return b""
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read()
