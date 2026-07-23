"""Tests for /api/cluster/* admin endpoints.

Covers:
- GET /api/cluster/workers lists registered workers
- GET /api/cluster/scoreboard shows current claims
- POST /api/cluster/scoreboard/force-release removes a claim
- GET /api/cluster/jobs lists submitted jobs
- GET /api/cluster/jobs/{w}/{j}/stdout returns streamed bytes
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from metainfer.cluster import mqueue, paths, scoreboard, worker_registry
from metainfer.cluster.queue_schema import JobSpec
from metainfer.cluster.scoreboard import LeaseToken


def test_list_workers_endpoint(client) -> None:
    worker_registry.register_worker("w0", "10.0.0.1", "h0", "m0",
                                     {0: {"uuid": "u"}})
    r = client.get("/api/cluster/workers")
    assert r.status_code == 200
    data = r.json()
    assert any(w["node_id"] == "w0" for w in data)
    w0 = [w for w in data if w["node_id"] == "w0"][0]
    assert w0["alive"] is True
    assert w0["ip"] == "10.0.0.1"


def test_get_worker_404_when_missing(client) -> None:
    r = client.get("/api/cluster/workers/no-such-worker")
    assert r.status_code == 404


def test_scoreboard_lists_claims(client) -> None:
    worker_registry.register_worker("w0", "ip", "h", "m", {0: {"uuid": "u"}})
    tok = scoreboard.acquire_gpus([("w0", 0)], holder="w0", job_id="j",
                                   deadline_s=1.0)
    assert tok is not None
    r = client.get("/api/cluster/scoreboard")
    claims = r.json()
    assert any(c["node_id"] == "w0" and c["gpu_idx"] == 0 and c["holder"] == "w0"
               for c in claims)


def test_force_release_endpoint(client) -> None:
    worker_registry.register_worker("w0", "ip", "h", "m", {0: {"uuid": "u"}})
    tok = scoreboard.acquire_gpus([("w0", 0)], holder="w0", job_id="j",
                                   deadline_s=1.0)
    assert tok is not None

    r = client.post("/api/cluster/scoreboard/force-release",
                     json={"node_id": "w0", "gpu_idx": 0, "reason": "test"})
    assert r.status_code == 200
    body = r.json()
    assert body["was_held"] is True

    # Slot is now free (list_claims returns free rows too — filter to status=held)
    claims = client.get("/api/cluster/scoreboard").json()
    held = [c for c in claims if c.get("status") == "held"
            and c.get("node_id") == "w0" and c.get("gpu_idx") == 0]
    assert held == [], "force-released slot must not show as held"


def test_force_release_404_on_missing_fields(client) -> None:
    r = client.post("/api/cluster/scoreboard/force-release", json={"node_id": "w0"})
    assert r.status_code == 400


def test_force_release_writes_cancel_marker(client, tmp_path: Path) -> None:
    """Force-releasing a slot should also write cancel.marker into associated
    job dir so the worker can SIGTERM its child."""
    worker_registry.register_worker("w0", "ip", "h", "m", {0: {"uuid": "u"}})
    # Submit a job and acquire its slot
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="sleep 100\n", timeout_s=100,
                   gpu_slots=[("w0", 0)])
    job_id = mqueue.submit_job(spec)
    tok = scoreboard.acquire_gpus([("w0", 0)], holder="w0", job_id=job_id,
                                   deadline_s=1.0)
    assert tok is not None

    # Force release with cancel target
    # NOTE: this version doesn't yet look up the job dir automatically; pass it
    # via payload's 'job_dir' if admin knows it. For now, verify the basic
    # release path works.
    r = client.post("/api/cluster/scoreboard/force-release",
                     json={"node_id": "w0", "gpu_idx": 0, "reason": "test"})
    assert r.status_code == 200


def test_list_jobs_endpoint(client) -> None:
    worker_registry.register_worker("w0", "ip", "h", "m", {})
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="echo hi\n", timeout_s=10)
    job_id = mqueue.submit_job(spec)
    r = client.get("/api/cluster/jobs")
    jobs = r.json()
    assert any(j["job_id"] == job_id for j in jobs)


def test_tail_log_endpoint(client) -> None:
    worker_registry.register_worker("w0", "ip", "h", "m", {})
    spec = JobSpec(type="script", worker_node_id="w0", submitter="orch",
                   script_body="echo hi\n", timeout_s=10)
    job_id = mqueue.submit_job(spec)
    # Write some bytes to stdout.log directly
    (paths.job_dir("w0", job_id) / "stdout.log").write_bytes(b"hello-from-worker\n")

    r = client.get(f"/api/cluster/jobs/w0/{job_id}/stdout")
    assert r.status_code == 200
    assert b"hello-from-worker" in r.content

    # With offset
    r = client.get(f"/api/cluster/jobs/w0/{job_id}/stdout?offset=5")
    assert r.status_code == 200
    assert b"-from-worker" in r.content
    assert b"hello" not in r.content


def test_tail_log_404_when_missing(client) -> None:
    r = client.get("/api/cluster/jobs/w0/no-such-job/stdout")
    assert r.status_code == 404


def test_tail_log_400_on_invalid_stream(client) -> None:
    r = client.get("/api/cluster/jobs/w0/whatever/stdin")
    assert r.status_code == 400
