"""Tests for metainfer.worker.daemon + metainfer.worker.jobs.

These tests exercise real subprocess execution (bash scripts only) against the
mqueue/fs_primitives layers. ccb-dependent paths are covered by FakeWorker
end-to-end tests in metainfer/tests/e2e/.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from metainfer.cluster import mqueue, paths, scoreboard, worker_registry
from metainfer.cluster.queue_schema import JobSpec
from metainfer.testing.fake_worker import FakeWorker
from metainfer.worker import jobs
from metainfer.worker.jobs import run_job
from metainfer.cluster.queue_schema import JobHandle


@pytest.fixture(autouse=True)
def _scratch_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------- #
# jobs.run_job — script path
# --------------------------------------------------------------------------- #
def _claim(spec: JobSpec) -> JobHandle:
    """Manually mark a job as claimed and return a handle (bypasses the daemon)."""
    jdir = paths.job_dir(spec.worker_node_id, spec.job_id)
    from metainfer.cluster import fs_primitives
    fs_primitives.link_claim(paths.job_claimed_marker(jdir),
                              {"worker_pid": 1, "claimed_at": time.time()})
    return JobHandle(spec=spec, job_dir=str(jdir),
                     claimed_at=time.time(), worker_pid=1)


def test_run_job_script_success() -> None:
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="echo 'hi from worker'\nexit 0\n", timeout_s=5.0)
    mqueue.submit_job(spec)
    handle = _claim(spec)

    result = run_job(handle, own_node_id="w0")
    assert result.status == "done"
    assert result.exit_code == 0
    # stdout was streamed to file
    out = (Path(handle.job_dir) / "stdout.log").read_text()
    assert "hi from worker" in out


def test_run_job_script_nonzero_exit() -> None:
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="echo to-stderr 1>&2\nexit 7\n", timeout_s=5.0)
    mqueue.submit_job(spec)
    handle = _claim(spec)

    result = run_job(handle, own_node_id="w0")
    assert result.status == "done"
    assert result.exit_code == 7
    err = (Path(handle.job_dir) / "stderr.log").read_text()
    assert "to-stderr" in err


def test_run_job_script_with_env() -> None:
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body='echo "MY_VAR=$MY_VAR"\n',
                   timeout_s=5.0,
                   env={"MY_VAR": "hello-from-orch"})
    mqueue.submit_job(spec)
    handle = _claim(spec)
    result = run_job(handle, own_node_id="w0")
    assert result.exit_code == 0
    out = (Path(handle.job_dir) / "stdout.log").read_text()
    assert "MY_VAR=hello-from-orch" in out


def test_run_job_cuda_visible_devices_set() -> None:
    """Slots on this worker should map to CUDA_VISIBLE_DEVICES."""
    worker_registry.register_worker("w0", "ip", "h", "m",
                                     {0: {"uuid": "a"}, 1: {"uuid": "b"}})
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body='echo "CVD=$CUDA_VISIBLE_DEVICES"\n',
                   timeout_s=5.0,
                   gpu_slots=[("w0", 1), ("w1", 0), ("w0", 0)])
    mqueue.submit_job(spec)
    handle = _claim(spec)
    run_job(handle, own_node_id="w0")
    out = (Path(handle.job_dir) / "stdout.log").read_text()
    # Sorted indices from w0: 0,1
    assert "CVD=0,1" in out


def test_run_job_no_cuda_when_no_slots_on_this_node() -> None:
    """Slots exist but none on this worker — CUDA_VISIBLE_DEVICES unset (or empty)."""
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body='echo "CVD=${CUDA_VISIBLE_DEVICES:-unset}"\n',
                   timeout_s=5.0,
                   gpu_slots=[("w1", 0)])
    mqueue.submit_job(spec)
    handle = _claim(spec)
    run_job(handle, own_node_id="w0")
    out = (Path(handle.job_dir) / "stdout.log").read_text()
    assert "CVD=unset" in out


# --------------------------------------------------------------------------- #
# Timeout escalation
# --------------------------------------------------------------------------- #
def test_run_job_timeout_returns_timeout_status() -> None:
    """Long-running script killed by deadline → status=timeout."""
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="sleep 30\n", timeout_s=1.0)
    mqueue.submit_job(spec)
    handle = _claim(spec)

    t0 = time.time()
    result = run_job(handle, own_node_id="w0")
    elapsed = time.time() - t0

    assert result.status == "timeout"
    # Elapsed should be ~1s timeout + up to 5s kill grace — well under 30s
    assert elapsed < 10.0


# --------------------------------------------------------------------------- #
# Cancel via marker
# --------------------------------------------------------------------------- #
def test_run_job_cancel_marker_triggers_sigterm() -> None:
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="sleep 30\n", timeout_s=30.0)
    mqueue.submit_job(spec)
    handle = _claim(spec)

    # Write cancel marker just before running
    paths.job_cancel_marker(Path(handle.job_dir)).write_text('{"reason":"test"}')

    result = run_job(handle, own_node_id="w0")
    assert result.status == "cancelled"


def test_run_job_cancel_marker_arrives_mid_run() -> None:
    """cancel.marker appears while the child is running."""
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="sleep 10\n", timeout_s=30.0)
    mqueue.submit_job(spec)
    handle = _claim(spec)

    # Spawn a thread that writes the marker after 0.3s
    import threading
    def canceller() -> None:
        time.sleep(0.3)
        paths.job_cancel_marker(Path(handle.job_dir)).write_text('{"reason":"late"}')
    threading.Thread(target=canceller, daemon=True).start()

    t0 = time.time()
    result = run_job(handle, own_node_id="w0")
    elapsed = time.time() - t0

    assert result.status == "cancelled"
    # Should die within ~1s of the marker (0.5s poll + kill grace)
    assert elapsed < 5.0


# --------------------------------------------------------------------------- #
# FakeWorker end-to-end
# --------------------------------------------------------------------------- #
def test_fake_worker_completes_submitted_job() -> None:
    fake = FakeWorker(node_id="w0")
    fake.register()
    fake.start_background()

    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="echo hi\n", timeout_s=5.0)
    job_id = mqueue.submit_job(spec)

    assert fake.wait_for_jobs(expected=1, timeout_s=2.0)
    assert job_id in fake.completed_job_ids()

    result = mqueue.read_result(job_id, "orch")
    assert result is not None
    assert result.status == "done"

    fake.stop()


def test_fake_worker_custom_handler() -> None:
    """Tests can inject their own job handler."""
    seen: list[str] = []

    def custom(handle: JobHandle, own: str) -> object:
        seen.append(handle.spec.job_id)
        from metainfer.cluster.queue_schema import JobResult
        return JobResult(job_id=handle.spec.job_id, status="done", exit_code=42)

    fake = FakeWorker(node_id="w0", handler=custom)
    fake.register()
    fake.start_background()

    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="echo x\n", timeout_s=5.0)
    job_id = mqueue.submit_job(spec)

    assert fake.wait_for_jobs(expected=1, timeout_s=2.0)
    result = mqueue.read_result(job_id, "orch")
    assert result is not None
    assert result.exit_code == 42
    fake.stop()
