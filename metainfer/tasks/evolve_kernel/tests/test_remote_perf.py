"""H.5 integration test: evolve_kernel perf phase routes through cluster SDK
when ``worker_nodes`` is configured.

Uses FakeWorker to simulate a remote worker that writes a perf-harness-style
JSON blob to stdout.log. Verifies:
  1. run_perf_test returns success and parses JSON from remote stdout.
  2. worker_status='done' is set on the result dict.
  3. Worker failure (status=worker_dead) is surfaced as worker_failure and
     causes run_perf_test to return passed=False with worker_status set.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from metainfer.cluster import paths as cluster_paths, worker_registry
from metainfer.cluster.queue_schema import JobHandle, JobResult
from metainfer.tasks.evolve_kernel.orchestrator.harness import run_perf_test
from metainfer.testing.fake_worker import FakeWorker


def _setup_worker(tmp_path: Path) -> FakeWorker:
    os.environ["METAINFER_ROOT"] = str(tmp_path)
    worker_registry.register_worker(
        node_id="w0", ip="10.0.0.1", hostname="fake",
        mac="aa:bb:cc:dd:ee:ff",
        gpu_topology={0: {"name": "fakeGPU", "total_memory_mib": 1024}},
    )
    # Need a heartbeat file so the worker isn't immediately marked dead.
    from metainfer.cluster import worker_registry as wr
    wr.touch_heartbeat("w0")
    return FakeWorker(
        node_id="w0", metainfer_root=str(tmp_path),
        gpu_topology={0: {"name": "fakeGPU", "total_memory_mib": 1024}},
    )


def test_run_perf_test_remote_success(tmp_path: Path, monkeypatch):
    """Remote perf path: worker writes valid JSON → parsed and passed=True."""
    fake = _setup_worker(tmp_path)
    harness_path = tmp_path / "perf_harness.py"
    harness_path.write_text("# fake harness\n")
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text("# fake kernel\n")

    def handler(handle: JobHandle, own_node_id: str) -> JobResult:
        out = {
            "passed": True,
            "ref_median_ms": 10.0,
            "evo_median_ms": 5.0,
            "overall_speedup": 2.0,
        }
        stdout_log = Path(handle.job_dir) / "stdout.log"
        stdout_log.write_text(json.dumps(out))
        return JobResult(job_id=handle.spec.job_id, status="done", exit_code=0,
                         duration_s=0.01)

    fake.handler = handler
    fake.start_background()
    try:
        ok, result = run_perf_test(
            harness_path, kernel_path, timeout_s=30,
            worker_nodes=["w0"],
        )
    finally:
        fake.stop()

    assert ok is True
    assert result["passed"] is True
    assert result["worker_status"] == "done"
    assert result["worker_node"] == "w0"
    assert result["overall_speedup"] == 2.0


def test_run_perf_test_remote_worker_dead(tmp_path: Path):
    """Worker dies before producing result → surfaced as worker_dead."""
    os.environ["METAINFER_ROOT"] = str(tmp_path)
    # Register then immediately sabotage heartbeat to look stale.
    worker_registry.register_worker(
        node_id="w0", ip="10.0.0.1", hostname="fake",
        mac="aa:bb:cc:dd:ee:ff",
        gpu_topology={0: {"name": "fakeGPU", "total_memory_mib": 1024}},
    )
    # Backdate heartbeat by touching with old mtime.
    hb = cluster_paths.worker_heartbeat("w0")
    import time as _t
    old = _t.time() - 600
    os.utime(hb, (old, old))

    harness_path = tmp_path / "perf_harness.py"
    harness_path.write_text("# fake\n")
    kernel_path = tmp_path / "kernel.py"
    kernel_path.write_text("# fake\n")

    # No FakeWorker running — the submit will get reaped as worker_dead.
    ok, result = run_perf_test(
        harness_path, kernel_path, timeout_s=10,
        worker_nodes=["w0"],
    )
    assert ok is False
    assert result["worker_status"] in ("worker_dead", "unknown", "timeout")
