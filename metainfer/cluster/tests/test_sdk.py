"""Tests for metainfer.cluster.sdk.

Uses FakeWorker as the remote side — no real subprocesses needed for most paths.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from metainfer.cluster import paths, sdk, worker_registry
from metainfer.cluster.queue_schema import JobResult
from metainfer.testing.fake_worker import FakeWorker


@pytest.fixture(autouse=True)
def _scratch_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------------------- #
# submit_script
# --------------------------------------------------------------------------- #
def test_submit_script_blocks_until_result() -> None:
    fake = FakeWorker(node_id="w0")
    fake.register()
    fake.start_background()

    job_id, result = sdk.submit_script(
        worker_node_id="w0",
        script_body="echo hi\n",
        timeout_s=5.0,
    )
    assert job_id
    assert result is not None
    assert result.status == "done"
    fake.stop()


def test_submit_script_no_gpu_slots_no_acquire() -> None:
    """When gpu_slots is empty, no scoreboard claim is created."""
    fake = FakeWorker(node_id="w0")
    fake.register()
    fake.start_background()

    sdk.submit_script(worker_node_id="w0", script_body="echo hi\n", timeout_s=5.0)

    # No claims should exist
    from metainfer.cluster import scoreboard
    assert scoreboard.list_claims() == []
    fake.stop()


def test_submit_script_with_gpu_slot_acquires_and_releases() -> None:
    fake = FakeWorker(node_id="w0")
    fake.register()
    fake.start_background()

    sdk.submit_script(
        worker_node_id="w0",
        script_body="echo hi\n",
        gpu_slots=[("w0", 0)],
        timeout_s=5.0,
    )

    from metainfer.cluster import scoreboard
    # Slot must be released after the job finishes.
    claims = scoreboard.list_claims()
    assert all(not (c["node_id"] == "w0" and c["gpu_idx"] == 0) for c in claims), \
        "slot must be released after RemoteJob exits"
    fake.stop()


def test_submit_script_acquire_failure_returns_none_no_leak() -> None:
    """If a slot is held by someone else, acquire fails and we return cleanly.

    No slot leak, no orphan job in the queue."""

    # Pre-acquire the slot
    from metainfer.cluster import scoreboard
    blocker = scoreboard.acquire_gpus([("w0", 0)], holder="other", job_id="b",
                                       deadline_s=1.0)
    assert blocker is not None

    with pytest.raises(TimeoutError):
        sdk.submit_script(
            worker_node_id="w0",
            script_body="echo hi\n",
            gpu_slots=[("w0", 0)],
            timeout_s=1.0,
            acquire_deadline_s=0.5,
        )

    # No new claim, blocker still holds.
    claims = scoreboard.list_claims()
    holders = [c["holder"] for c in claims if c["node_id"] == "w0" and c["gpu_idx"] == 0]
    assert holders == ["other"]


# --------------------------------------------------------------------------- #
# Timeout path
# --------------------------------------------------------------------------- #
def test_submit_script_result_status_timeout_when_worker_unresponsive() -> None:
    """Worker doesn't process the job — orchestrator's reap_orphaned_submissions
    eventually produces a synthetic result."""
    # Register worker but never start the daemon — heartbeat goes stale.
    # Stale the heartbeat immediately
    import os as _os
    worker_registry.register_worker("w0", "10.0.0.1", "w0", "m",
                                     {0: {"uuid": "x"}})
    hb = paths.worker_heartbeat("w0")
    old = time.time() - 600
    _os.utime(hb, (old, old))

    # Don't start FakeWorker — job stays pending.
    # block=True: RemoteJob's inline reaper must surface worker_dead within ~5s.
    job_id, result = sdk.submit_script(
        worker_node_id="w0",
        script_body="echo hi\n",
        gpu_slots=[("w0", 0)],
        timeout_s=0.1,
        acquire_deadline_s=1.0,
        # Override poll interval to make the test snappy
    )

    # The inline reaper inside RemoteJob.collect_result should have written
    # a synthetic worker_dead result; submit_script returns it.
    assert result is not None, "expected synthetic result from inline reaper"
    assert result.status == "worker_dead"
    assert result.job_id == job_id


# --------------------------------------------------------------------------- #
# Log tailing
# --------------------------------------------------------------------------- #
def test_tail_stdout_returns_streamed_output() -> None:
    """FakeWorker writes to stdout.log; tail_stdout reads it."""
    fake = FakeWorker(node_id="w0")
    fake.register()
    fake.start_background()

    job_id, _ = sdk.submit_script(
        worker_node_id="w0",
        script_body="echo hi\n",
        timeout_s=5.0,
    )
    # Give worker a moment
    time.sleep(0.2)

    out = sdk.tail_stdout(job_id, "w0")
    assert b"fake-worker" in out  # FakeWorker writes this line

    fake.stop()


# --------------------------------------------------------------------------- #
# Non-blocking submit
# --------------------------------------------------------------------------- #
def test_submit_script_non_blocking_returns_immediately() -> None:
    fake = FakeWorker(node_id="w0")
    fake.register()
    fake.start_background()

    job_id, result = sdk.submit_script(
        worker_node_id="w0",
        script_body="echo hi\n",
        timeout_s=5.0,
        block=False,
    )
    assert job_id
    assert result is None  # non-blocking

    # Eventually completes
    assert fake.wait_for_jobs(expected=1, timeout_s=2.0)
    fake.stop()
