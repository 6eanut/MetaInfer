"""Worker daemon main loop.

Single-process, multi-threaded. Main thread:
  (a) touches heartbeat every ``HEARTBEAT_INTERVAL_S``
  (b) polls inbox for new jobs each iteration
  (c) on each new job: spawns a supervisor thread that runs ``jobs.run_job``
      and writes the result back.

Per-job supervisor threads are bounded by ``max_concurrent_jobs`` (default 1 —
GPU jobs rarely benefit from in-worker parallelism and the scoreboard has
already allocated exclusive slots). Excess jobs queue in the inbox until a
slot frees.

Startup:
  - Detects local IP / MAC / hostname / GPU topology
  - Calls ``worker_registry.register_worker`` (overwrites prior record — cold-start safe)
  - Begins main loop

Shutdown:
  - SIGTERM/SIGINT → stop event → wait for in-flight jobs to finish (bounded
    by their remaining timeout) → exit. Does NOT abandon jobs mid-flight.
"""

from __future__ import annotations

import os
import signal
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from metainfer.cluster import mqueue, paths, topology, worker_registry
from metainfer.cluster.queue_schema import JobHandle
from metainfer.worker import jobs


HEARTBEAT_INTERVAL_S = worker_registry.HEARTBEAT_INTERVAL_S  # 15s
POLL_INTERVAL_S = 1.0


@dataclass
class WorkerConfig:
    node_id: str
    metainfer_root: Optional[str] = None
    ip: Optional[str] = None
    hostname: Optional[str] = None
    mac: Optional[str] = None
    max_concurrent_jobs: int = 1


class WorkerDaemon:
    """Long-running worker process. Call :meth:`run_forever` to start."""

    def __init__(self, cfg: WorkerConfig) -> None:
        self.cfg = cfg
        self._stop = threading.Event()
        self._inflight = threading.Semaphore(cfg.max_concurrent_jobs)
        self._inflight_count = 0
        self._inflight_lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Register worker and begin main loop. Returns when stop() is called."""
        self._register()
        # Install signal handlers (no-op when called from non-main thread).
        try:
            signal.signal(signal.SIGTERM, self._on_signal)
            signal.signal(signal.SIGINT, self._on_signal)
        except (ValueError, OSError):
            pass

        last_hb = 0.0
        while not self._stop.is_set():
            # Touch heartbeat at the configured cadence
            now = time.time()
            if now - last_hb >= HEARTBEAT_INTERVAL_S:
                worker_registry.touch_heartbeat(self.cfg.node_id)
                last_hb = now

            # Try to claim a job if we have an inflight slot
            if self._inflight.acquire(blocking=False):
                try:
                    handle = mqueue.consume_next_job(self.cfg.node_id, os.getpid())
                    if handle is None:
                        # No job available — release the slot and sleep.
                        self._inflight.release()
                        time.sleep(POLL_INTERVAL_S)
                        continue
                    # Launch supervisor thread
                    with self._inflight_lock:
                        self._inflight_count += 1
                    t = threading.Thread(target=self._supervise, args=(handle,),
                                         daemon=True,
                                         name=f"job-{handle.spec.job_id[:8]}")
                    self._threads.append(t)
                    t.start()
                except Exception:  # noqa: BLE001
                    self._inflight.release()
                    raise
            else:
                time.sleep(POLL_INTERVAL_S)

        # Drain in-flight jobs
        for t in self._threads:
            if t.is_alive():
                t.join(timeout=60.0)

    def run_forever(self) -> None:
        """Alias for :meth:`start` — the conventional entry point name."""
        self.start()

    def stop(self) -> None:
        self._stop.set()

    def _on_signal(self, signum, frame) -> None:  # noqa: ARG002
        self._stop.set()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _register(self) -> None:
        if self.cfg.metainfer_root:
            os.environ["METAINFER_ROOT"] = self.cfg.metainfer_root
        ip = self.cfg.ip or worker_registry.detect_local_ip()
        hostname = self.cfg.hostname or socket.gethostname()
        mac = self.cfg.mac or worker_registry.detect_local_mac()
        topo = topology.detect_gpu_topology()
        worker_registry.register_worker(
            node_id=self.cfg.node_id,
            ip=ip,
            hostname=hostname,
            mac=mac,
            gpu_topology=topo,
        )

    def _supervise(self, handle: JobHandle) -> None:
        """Run one job in its own thread and write result back."""
        try:
            result = jobs.run_job(handle, self.cfg.node_id)
            mqueue.write_result(handle, result)
        except Exception as e:  # noqa: BLE001
            # Last-resort error path: write a failed result so the orchestrator
            # is not left polling forever.
            try:
                from metainfer.cluster.queue_schema import JobResult
                mqueue.write_result(handle, JobResult(
                    job_id=handle.spec.job_id, status="failed",
                    error=f"worker supervisor crash: {e!s}",
                ))
            except Exception:  # noqa: BLE001
                pass
        finally:
            with self._inflight_lock:
                self._inflight_count -= 1
            self._inflight.release()
