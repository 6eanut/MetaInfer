"""Test-only FakeWorker — runs jobs in-process without spawning subprocesses.

Mirrors the protocol of ``metainfer.worker.daemon.WorkerDaemon`` but executes
a caller-supplied callable instead of spawning ``bash`` / ``ccb``. This lets
unit and e2e tests run jobs deterministically without GPU or network.

Usage::

    fake = FakeWorker(node_id="w0", metainfer_root=tmp_path)
    fake.register()  # write worker_registry record
    fake.start_background()  # begin poll loop in a thread
    # ... orchestrator submits jobs ...
    fake.wait_for_jobs(timeout_s=5)  # block until all results written
    fake.stop()

Or drive manually::

    fake.register()
    handle = fake.consume_one()  # one-shot consume + execute
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from metainfer.cluster import mqueue, paths, topology, worker_registry
from metainfer.cluster.queue_schema import JobHandle, JobResult, JobSpec
from metainfer.worker import jobs


# Default job handler: simulates immediate success.
DefaultHandler = Callable[[JobHandle, str], JobResult]


def _default_handler(handle: JobHandle, own_node_id: str) -> JobResult:
    """Default: pretend the script ran successfully (exit 0)."""
    # Optional: append a fake stdout line so log-tail tests can verify
    try:
        stdout_path = Path(handle.job_dir) / "stdout.log"
        with open(stdout_path, "ab") as f:
            f.write(b"fake-worker: simulated success\n")
    except OSError:
        pass
    return JobResult(job_id=handle.spec.job_id, status="done", exit_code=0,
                     duration_s=0.01)


@dataclass
class FakeWorker:
    node_id: str
    metainfer_root: Optional[str] = None
    ip: str = "10.0.0.1"
    hostname: str = "fake"
    mac: str = "aa:bb:cc:dd:ee:ff"
    gpu_topology: Dict[int, Dict[str, object]] = field(default_factory=dict)
    handler: DefaultHandler = _default_handler
    # If True, do NOT touch heartbeat after register() — used to simulate dead workers.
    simulate_dead_after_register: bool = False

    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None
    _heartbeat_thread: Optional[threading.Thread] = None
    _jobs_done: List[str] = field(default_factory=list)
    _jobs_lock: threading.Lock = field(default_factory=threading.Lock)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def register(self) -> None:
        if self.metainfer_root:
            os.environ["METAINFER_ROOT"] = self.metainfer_root
        worker_registry.register_worker(
            node_id=self.node_id,
            ip=self.ip,
            hostname=self.hostname,
            mac=self.mac,
            gpu_topology=self.gpu_topology,
        )

    def start_background(self, poll_interval_s: float = 0.05) -> None:
        """Begin polling inbox + heartbeating in background threads."""
        self._stop.clear()
        if not self.simulate_dead_after_register:
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, args=(poll_interval_s,), daemon=True
            )
            self._heartbeat_thread.start()
        self._thread = threading.Thread(target=self._poll_loop,
                                        args=(poll_interval_s,), daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=2.0)

    # ------------------------------------------------------------------ #
    # Manual drive (for tests that want synchronous control)
    # ------------------------------------------------------------------ #
    def consume_one(self) -> Optional[JobHandle]:
        """Claim and execute one job synchronously."""
        handle = mqueue.consume_next_job(self.node_id, os.getpid())
        if handle is None:
            return None
        result = self.handler(handle, self.node_id)
        mqueue.write_result(handle, result)
        with self._jobs_lock:
            self._jobs_done.append(handle.spec.job_id)
        return handle

    def wait_for_jobs(self, expected: int, timeout_s: float = 5.0) -> bool:
        """Block until ``expected`` jobs have completed."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._jobs_lock:
                if len(self._jobs_done) >= expected:
                    return True
            time.sleep(0.05)
        return False

    def completed_job_ids(self) -> List[str]:
        with self._jobs_lock:
            return list(self._jobs_done)

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _heartbeat_loop(self, interval_s: float) -> None:
        while not self._stop.is_set():
            worker_registry.touch_heartbeat(self.node_id)
            time.sleep(interval_s * 10)  # heartbeat every ~0.5s in tests

    def _poll_loop(self, interval_s: float) -> None:
        while not self._stop.is_set():
            self.consume_one()
            time.sleep(interval_s)
