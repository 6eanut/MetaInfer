"""FastAPI router for opt_operator.

Endpoints (mounted by the shell at ``/api/opt-operator/{task_id}``):
  - ``/state-graph``  phase DAG for the macro view
  - ``/overview``     macro: run + state graph + lineage + reference + GPU pool + summary
  - ``/lineage``      champion lineage curve
  - ``/iterations``, ``/iterations/{n}``, ``/iterations/{n}/conformance``  drill-in
  - ``/events``       SSE live updates (run.json / timeline / agents / iterations)
  - ``/qa``           generic offline-QA triplet
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from metainfer.server import sse as _sse
from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
)
from metainfer.server.qa_routes import register_qa_routes

from . import _state_readers

PLUGIN_TYPE = "opt-operator"


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    def _state_dir(task_id: str) -> Path:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return state_dir_for(entry)

    @router.get("/state-graph")
    def state_graph(task_id: str) -> Dict[str, Any]:
        return _state_readers.read_state_graph(_state_dir(task_id))

    @router.get("/overview")
    def overview(task_id: str) -> Dict[str, Any]:
        return _state_readers.read_overview(_state_dir(task_id))

    @router.get("/lineage")
    def lineage(task_id: str) -> List[Dict[str, Any]]:
        return _state_readers.read_lineage(_state_dir(task_id))

    @router.get("/pool")
    def pool(task_id: str) -> Dict[str, Any]:
        return _state_readers.read_pool(_state_dir(task_id))

    @router.get("/harness")
    def harness(task_id: str) -> Dict[str, Any]:
        return _state_readers.read_harness_reviews(_state_dir(task_id))

    @router.get("/iterations")
    def iterations(task_id: str) -> List[Dict[str, Any]]:
        return _state_readers.read_iterations(_state_dir(task_id))

    @router.get("/iterations/{n}")
    def iteration_detail(task_id: str, n: int) -> Dict[str, Any]:
        rec = _state_readers.read_iteration(_state_dir(task_id), n)
        if rec is None:
            raise HTTPException(404, f"no iteration {n} for task {task_id}")
        return rec

    @router.get("/iterations/{n}/conformance")
    def conformance(task_id: str, n: int) -> Dict[str, Any]:
        data = _state_readers.read_conformance(_state_dir(task_id), n)
        if data is None:
            raise HTTPException(404, f"no conformance for iteration {n}")
        return data

    @router.get("/events")
    async def events(request: Request) -> StreamingResponse:
        await _sse.watcher.start()
        q = await _sse.watcher.subscribe()

        async def gen():
            try:
                yield "event: hello\ndata: {}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                await _sse.watcher.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    register_qa_routes(router, plugin, prefix="/qa")

    return router
