"""HTTP routes for cluster-wide coordination state.

Mounted at ``/api/cluster/*`` by :func:`metainfer.server.app.create_app`.

Stateless: every read goes to disk via :mod:`metainfer.cluster.worker_registry`
and :mod:`metainfer.cluster.scoreboard`. WebUI restart-safe.

Endpoints (workers):
- ``GET /api/cluster/workers`` — list all registered workers with liveness
- ``GET /api/cluster/workers/{node_id}`` — single worker detail

Endpoints (scoreboard): added by Module C.5.

Endpoints (job logs): added by Module G.4.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from metainfer.cluster import worker_registry
from metainfer.cluster import scoreboard
from metainfer.cluster.fs_primitives import is_stale_heartbeat
from metainfer.cluster.paths import worker_heartbeat


def build_router() -> APIRouter:
    router = APIRouter(tags=["cluster"])

    # ------------------------------------------------------------------ #
    # Workers
    # ------------------------------------------------------------------ #
    @router.get("/workers")
    def list_workers() -> List[Dict[str, Any]]:
        out = []
        for rec in worker_registry.list_workers():
            stale = is_stale_heartbeat(worker_heartbeat(rec.node_id),
                                       stale_after_s=worker_registry.STALE_AFTER_S)
            out.append({
                **rec.to_dict(),
                "alive": not stale,
                "last_heartbeat_ago_s": _heartbeat_age(rec.node_id),
            })
        return out

    @router.get("/workers/{node_id}")
    def get_worker(node_id: str) -> Dict[str, Any]:
        rec = worker_registry.read_worker(node_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"worker {node_id!r} not registered")
        stale = is_stale_heartbeat(worker_heartbeat(node_id),
                                   stale_after_s=worker_registry.STALE_AFTER_S)
        return {
            **rec.to_dict(),
            "alive": not stale,
            "last_heartbeat_ago_s": _heartbeat_age(node_id),
        }

    # ------------------------------------------------------------------ #
    # Scoreboard
    # ------------------------------------------------------------------ #
    @router.get("/scoreboard")
    def get_scoreboard() -> List[Dict[str, Any]]:
        return scoreboard.list_claims(node_id=None)

    @router.post("/scoreboard/force-release")
    def force_release(payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload.get("node_id")
        gpu_idx = payload.get("gpu_idx")
        if not isinstance(node_id, str) or not isinstance(gpu_idx, int):
            raise HTTPException(status_code=400,
                                detail="node_id (str) and gpu_idx (int) required")
        slot = (node_id, gpu_idx)
        # Read the claim before breaking it so we can write cancel.marker
        # into the worker's job_dir (signals the worker subprocess to SIGTERM).
        from metainfer.cluster.paths import gpu_claim_path, job_dir
        from metainfer.cluster.fs_primitives import read_claim
        claim = read_claim(gpu_claim_path(node_id, gpu_idx))
        cancel_jd = None
        if claim is not None:
            jid = claim.get("job_id")
            if jid:
                # The slot's node_id is the worker that hosts the job's inbox.
                jd = job_dir(node_id, jid)
                if jd.exists():
                    cancel_jd = jd
        existed = scoreboard.force_release(
            slot,
            reason=str(payload.get("reason", "admin-kill")),
            cancel_job_dir=cancel_jd,
        )
        return {"node_id": node_id, "gpu_idx": gpu_idx, "was_held": existed}

    # ------------------------------------------------------------------ #
    # Jobs (inbox + replies listing, log tailing)
    # ------------------------------------------------------------------ #
    @router.get("/jobs")
    def list_jobs(worker_node_id: Optional[str] = None) -> List[Dict[str, Any]]:
        from metainfer.cluster import mqueue
        return mqueue.list_jobs(worker_node_id=worker_node_id)

    @router.get("/jobs/{worker_node_id}/{job_id}/{stream}")
    def tail_job_log(worker_node_id: str, job_id: str, stream: str,
                     offset: int = 0) -> bytes:
        """Tail stdout.log or stderr.log for one job. Returns raw bytes."""
        if stream not in ("stdout", "stderr"):
            raise HTTPException(status_code=400,
                                detail="stream must be 'stdout' or 'stderr'")
        from metainfer.cluster import sdk
        from metainfer.cluster.paths import job_dir
        # Verify job dir exists (404 otherwise)
        jdir = job_dir(worker_node_id, job_id)
        if not jdir.exists():
            raise HTTPException(status_code=404, detail="job dir not found")
        if stream == "stdout":
            return sdk.tail_stdout(job_id, worker_node_id, offset=offset)
        return sdk.tail_stderr(job_id, worker_node_id, offset=offset)

    return router


def _heartbeat_age(node_id: str) -> float | None:
    """Seconds since the worker's last heartbeat, or None if missing."""
    try:
        from pathlib import Path
        p = worker_heartbeat(node_id)
        return max(0.0, time.time() - Path(p).stat().st_mtime)
    except OSError:
        return None
