"""I.5 E2E: WebUI cold restart.

Per CLAUDE.md invariant: restarting the WebUI process must not lose cluster
state. Workers, scoreboard claims, and jobs are all SSOT-backed by files —
the new process must read the same state.

Writes some state via the SDK, then constructs a fresh TestClient (simulating
a process restart) and verifies the new server sees the same workers/claims/jobs.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from metainfer.cluster import paths, scoreboard, worker_registry
from metainfer.cluster.queue_schema import JobSpec
from metainfer.cluster import mqueue
from metainfer.server.app import create_app


@pytest.fixture(autouse=True)
def _scratch_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    return tmp_path


def test_cluster_state_survives_webui_restart(tmp_path: Path) -> None:
    # Seed state.
    worker_registry.register_worker(
        "w0", "10.0.0.1", "fake", "aa:bb",
        gpu_topology={0: {"name": "fakeGPU", "total_memory_mib": 1024}},
    )
    worker_registry.register_worker(
        "w1", "10.0.0.2", "fake2", "cc:dd",
        gpu_topology={0: {"name": "fakeGPU", "total_memory_mib": 1024}},
    )
    # Touch heartbeats so they look alive.
    worker_registry.touch_heartbeat("w0")
    worker_registry.touch_heartbeat("w1")
    # Acquire a slot.
    token = scoreboard.acquire_gpus(
        [("w0", 0)], holder="orch", job_id="j-restart-1", deadline_s=5.0,
    )
    assert token is not None
    # Submit a job.
    spec = JobSpec(
        job_id="j-restart-1", type="script", worker_node_id="w0",
        gpu_slots=[("w0", 0)], timeout_s=60.0, submitter="orch",
        submitted_at=time.time(), script_body="echo hi\n",
    )
    mqueue.submit_job(spec)

    # First server instance.
    client1 = TestClient(create_app())
    workers_1 = client1.get("/api/cluster/workers").json()
    score_1 = client1.get("/api/cluster/scoreboard").json()
    jobs_1 = client1.get("/api/cluster/jobs").json()
    assert len(workers_1) == 2
    assert any(c["node_id"] == "w0" and c["gpu_idx"] == 0 for c in score_1)
    assert any(j["job_id"] == "j-restart-1" for j in jobs_1)

    # Simulate restart: drop client + app, build fresh ones.
    client2 = TestClient(create_app())
    workers_2 = client2.get("/api/cluster/workers").json()
    score_2 = client2.get("/api/cluster/scoreboard").json()
    jobs_2 = client2.get("/api/cluster/jobs").json()

    # State must be identical.
    assert {w["node_id"] for w in workers_2} == {"w0", "w1"}
    assert any(c["node_id"] == "w0" and c["gpu_idx"] == 0 for c in score_2), \
        "scoreboard claim must survive restart"
    assert any(j["job_id"] == "j-restart-1" for j in jobs_2), \
        "inbox jobs must survive restart"

    # Cleanup.
    scoreboard.release_gpus(token)
