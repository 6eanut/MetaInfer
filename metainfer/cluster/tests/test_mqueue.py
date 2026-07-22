"""Tests for metainfer.cluster.mqueue."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import pytest

from metainfer.cluster import mqueue, paths, scoreboard, worker_registry
from metainfer.cluster.queue_schema import JobResult, JobSpec


@pytest.fixture(autouse=True)
def _scratch_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    return tmp_path


def _make_spec(worker: str = "w0", submitter: str = "orch",
               script_body: str = "echo hello\n",
               timeout_s: float = 60.0) -> JobSpec:
    return JobSpec(
        type="script",
        worker_node_id=worker,
        submitter=submitter,
        script_body=script_body,
        timeout_s=timeout_s,
    )


# --------------------------------------------------------------------------- #
# Submit / consume / result round-trip
# --------------------------------------------------------------------------- #
def test_submit_creates_job_dir_atomically() -> None:
    spec = _make_spec()
    job_id = mqueue.submit_job(spec)
    jdir = paths.job_dir("w0", job_id)
    assert (jdir / "job.json").exists()
    assert (jdir / "script.sh").read_text() == "echo hello\n"
    assert (jdir / "status.json").exists()
    # reply_to is populated
    loaded = mqueue.read_job("w0", job_id)
    assert loaded is not None
    assert loaded.reply_to.endswith(f"replies/orch/{job_id}.result.json")


def test_submit_agent_writes_prompt_file() -> None:
    spec = JobSpec(type="agent", worker_node_id="w0", submitter="orch",
                   prompt_body="do the thing")
    job_id = mqueue.submit_job(spec)
    jdir = paths.job_dir("w0", job_id)
    assert (jdir / "prompt.txt").read_text() == "do the thing"
    assert not (jdir / "script.sh").exists()


def test_consume_next_job_claims_and_returns_handle() -> None:
    spec = _make_spec()
    job_id = mqueue.submit_job(spec)
    handle = mqueue.consume_next_job("w0", worker_pid=123)
    assert handle is not None
    assert handle.spec.job_id == job_id
    assert handle.worker_pid == 123
    # claimed marker now exists
    assert paths.job_claimed_marker(paths.job_dir("w0", job_id)).exists()
    # status flipped to inflight
    status = json.loads((paths.job_dir("w0", job_id) / "status.json").read_text())
    assert status["status"] == "inflight"


def test_consume_next_job_skips_already_claimed() -> None:
    job_id = mqueue.submit_job(_make_spec())
    h1 = mqueue.consume_next_job("w0", worker_pid=1)
    assert h1 is not None
    h2 = mqueue.consume_next_job("w0", worker_pid=2)
    assert h2 is None


def test_consume_returns_none_when_empty() -> None:
    assert mqueue.consume_next_job("w0", worker_pid=1) is None


def test_write_then_read_result() -> None:
    mqueue.submit_job(_make_spec())
    handle = mqueue.consume_next_job("w0", worker_pid=1)
    assert handle is not None

    result = JobResult(job_id=handle.spec.job_id, status="done",
                       exit_code=0, duration_s=1.5)
    mqueue.write_result(handle, result)

    out = mqueue.read_result(handle.spec.job_id, "orch")
    assert out is not None
    assert out.status == "done"
    assert out.exit_code == 0
    assert out.duration_s == 1.5


def test_read_result_blocks_then_returns(tmp_path: Path) -> None:
    """read_result with timeout polls until result appears."""
    mqueue.submit_job(_make_spec())
    handle = mqueue.consume_next_job("w0", worker_pid=1)
    assert handle is not None

    result_written: dict = {"done": False}

    def writer() -> None:
        time.sleep(0.2)
        mqueue.write_result(handle, JobResult(job_id=handle.spec.job_id,
                                              status="done", exit_code=0))
        result_written["done"] = True

    threading.Thread(target=writer, daemon=True).start()
    out = mqueue.read_result(handle.spec.job_id, "orch", timeout_s=2.0)
    assert result_written["done"] is True
    assert out is not None
    assert out.status == "done"


# --------------------------------------------------------------------------- #
# Consume race
# --------------------------------------------------------------------------- #
def test_consume_race_unique_winner() -> None:
    """Multiple worker threads try to consume the same single job. Exactly one wins."""
    job_id = mqueue.submit_job(_make_spec())
    winners: list[int] = []
    lock = threading.Lock()

    def consumer(pid: int) -> None:
        h = mqueue.consume_next_job("w0", worker_pid=pid)
        if h is not None:
            with lock:
                winners.append(pid)

    with ThreadPoolExecutor(max_workers=10) as ex:
        list(ex.map(consumer, range(10)))

    assert len(winners) == 1


# --------------------------------------------------------------------------- #
# Producer crash tolerance
# --------------------------------------------------------------------------- #
def test_consume_skips_partial_tmp_dir() -> None:
    """A leftover .tmp dir from a crashed submit must not break consumption."""
    idir = paths.inbox_dir("w0")
    leftover_tmp = idir / ".somejob.abc.tmp"
    leftover_tmp.mkdir(parents=True)

    # Submit a real job in parallel
    real_job = mqueue.submit_job(_make_spec())
    handle = mqueue.consume_next_job("w0", worker_pid=1)
    assert handle is not None
    assert handle.spec.job_id == real_job


def test_consume_skips_cancelled_job() -> None:
    """A cancel.marker present before consumption makes the job unclaimable."""
    mqueue.submit_job(_make_spec())
    jdir = paths.job_dir("w0", _last_job_id("w0"))
    paths.job_cancel_marker(jdir).write_text('{"reason":"test"}')
    handle = mqueue.consume_next_job("w0", worker_pid=1)
    assert handle is None


# --------------------------------------------------------------------------- #
# Orphan reaper
# --------------------------------------------------------------------------- #
def test_reap_orphaned_submissions_writes_worker_dead_result() -> None:
    worker_registry.register_worker("w0", "ip", "h", "m", {})
    # Stale worker
    hb = paths.worker_heartbeat("w0")
    old = time.time() - 600
    import os as _os
    _os.utime(hb, (old, old))

    spec = _make_spec(timeout_s=0.1)
    job_id = mqueue.submit_job(spec)
    # Wait past timeout + grace. Force grace_s small.
    time.sleep(0.2)

    reaped = mqueue.reap_orphaned_submissions("orch", grace_s=0.0)
    assert job_id in reaped

    out = mqueue.read_result(job_id, "orch")
    assert out is not None
    assert out.status == "worker_dead"


def test_reap_skips_jobs_with_existing_result() -> None:
    worker_registry.register_worker("w0", "ip", "h", "m", {})
    spec = _make_spec(timeout_s=0.1)
    job_id = mqueue.submit_job(spec)
    handle = mqueue.consume_next_job("w0", worker_pid=1)
    assert handle is not None
    mqueue.write_result(handle, JobResult(job_id=job_id, status="done", exit_code=0))
    time.sleep(0.2)

    reaped = mqueue.reap_orphaned_submissions("orch", grace_s=0.0)
    assert job_id not in reaped


def test_reap_force_releases_scoreboard_slots() -> None:
    """Orphan reaper must release GPU slots so they don't leak."""
    worker_registry.register_worker("w0", "ip", "h", "m", {0: {"uuid": "x"}})
    # Pre-acquire a slot in the name of this job (simulating orchestrator side)
    tok = scoreboard.acquire_gpus([("w0", 0)], holder="orch", job_id="j", deadline_s=1.0)
    assert tok is not None

    spec = _make_spec(timeout_s=0.1)
    spec.gpu_slots = [("w0", 0)]
    job_id = mqueue.submit_job(spec)
    # Backdate submitted_at by editing job.json directly.
    jdir = paths.job_dir("w0", job_id)
    data = json.loads((jdir / "job.json").read_text())
    data["submitted_at"] = time.time() - 1000
    (jdir / "job.json").write_text(json.dumps(data))

    reaped = mqueue.reap_orphaned_submissions("orch", grace_s=0.0)
    assert job_id in reaped
    # Slot now free
    claims = scoreboard.list_claims()
    assert all(not (c["node_id"] == "w0" and c["gpu_idx"] == 0) for c in claims)


# --------------------------------------------------------------------------- #
# Reset
# --------------------------------------------------------------------------- #
def test_reset_queue_wipes_inbox() -> None:
    mqueue.submit_job(_make_spec())
    mqueue.reset_queue("w0")
    assert not any(paths.inbox_dir("w0").iterdir())


def test_reset_reply_queue_wipes_results() -> None:
    spec = _make_spec()
    job_id = mqueue.submit_job(spec)
    handle = mqueue.consume_next_job("w0", worker_pid=1)
    assert handle is not None
    mqueue.write_result(handle, JobResult(job_id=job_id, status="done", exit_code=0))
    assert paths.result_path("orch", job_id).exists()

    mqueue.reset_reply_queue("orch")
    assert not paths.result_path("orch", job_id).exists()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _last_job_id(worker: str) -> str:
    idir = paths.inbox_dir(worker)
    for entry in sorted(idir.iterdir(), reverse=True):
        if entry.is_dir() and not entry.name.startswith("."):
            return entry.name
    raise AssertionError("no jobs in inbox")
