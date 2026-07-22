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
        existed = scoreboard.force_release(
            (node_id, gpu_idx),
            reason=str(payload.get("reason", "admin-kill")),
        )
        return {"node_id": node_id, "gpu_idx": gpu_idx, "was_held": existed}

    return router


def _heartbeat_age(node_id: str) -> float | None:
    """Seconds since the worker's last heartbeat, or None if missing."""
    try:
        from pathlib import Path
        p = worker_heartbeat(node_id)
        return max(0.0, time.time() - Path(p).stat().st_mtime)
    except OSError:
        return None
