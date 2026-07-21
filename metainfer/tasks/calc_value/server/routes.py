"""FastAPI router for the calc-theoretical-value task type.

This module builds a single :class:`fastapi.APIRouter` carrying every
HTTP route calc_value exposes. The shell mounts it under
``/api/{type}/{task_id}`` so every route below lands at:

    /api/{type}/{task_id}/calc/graph
    /api/{type}/{task_id}/calc/compute
    /api/{type}/{task_id}/calc/viz
    /api/{type}/{task_id}/calc/summary
    /api/{type}/{task_id}/calc/iterations
    /api/{type}/{task_id}/calc/rough
    /api/{type}/{task_id}/calc/cells
    /api/{type}/{task_id}/calc/cell/{compound}/{angle}/{round_idx}
    /api/{type}/{task_id}/calc/qa[/start|/<sid>]
    /api/{type}/{task_id}/iterations[/{n}[/retrospective]]
    /api/{type}/{task_id}/charts
    /api/{type}/{task_id}/state-graph
    /api/{type}/{task_id}/control

The first block (``/calc/...``) reads calc-specific artifacts from the
task's ``workspace_dir`` via :mod:`._readers`. The second block
(``/iterations``, ``/charts``, ``/state-graph``, ``/retrospective``)
reads the orchestrator's iteration records / phases from ``state_dir``
— these used to live in the shell but are task-shaped, so they now
live with the task package.

Type guard is enforced at the route layer via
:func:`metainfer.server._helpers.require_task_type` for safety.
"""

from __future__ import annotations

import shutil
import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request
import asyncio
from fastapi.responses import HTMLResponse

from metainfer.server import launcher as _launcher
from metainfer.server import state_reader as _sr
from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
    workspace_dir_for,
)
from metainfer.server.qa_routes import register_qa_routes

from . import _readers, _state_readers

PLUGIN_TYPE = "calc-theoretical-value"

# Step order: when re-running step N, we wipe steps N..last.
# These live under workspace_dir (e.g. workspace_dir/step0/).
_STEP_DIRS = ["step0", "step1", "step2", "step3", "step4"]
_STEP_INDEX = {s: i for i, s in enumerate(_STEP_DIRS)}

# Map WebUI-facing step labels to step dir names.
_STEP_MAP = {
    "S0_rough": "step0",
    "S1_analyze": "step1",
    "S2_graph": "step2",
    "S3_calculate": "step3",
    "S4_visualize": "step4",
}


def build_router(plugin) -> APIRouter:
    """Build the calc_value router. ``plugin`` is the WebPlugin itself,
    passed in by :func:`metainfer.server.app.create_app` so we can hand it
    to the generic QA helper without a circular import."""
    router = APIRouter()

    # ----------------------------------------------------------------- #
    # calc-specific workspace reads
    # ----------------------------------------------------------------- #
    @router.get("/calc/graph")
    def calc_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return _readers.read_graph(wd)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    @router.get("/calc/compute")
    def calc_compute(task_id: str, batch_size: int = 1, seq_len: int = 1) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return _readers.compute(wd, batch_size, seq_len)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/calc/viz")
    def calc_viz(task_id: str) -> HTMLResponse:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return HTMLResponse(_readers.read_viz(wd, task_id))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    @router.get("/calc/summary")
    def calc_summary(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _readers.read_summary(workspace_dir_for(entry))

    @router.get("/calc/iterations")
    def calc_iterations(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _readers.read_iterations(workspace_dir_for(entry))

    @router.get("/calc/rough")
    def calc_rough(task_id: str, request: Request) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        qp = request.query_params
        bs_str = qp.get("batch_size")
        sl_str = qp.get("seq_len")
        bs = None
        sl = None
        if bs_str is not None:
            try:
                bs = int(bs_str)
            except ValueError:
                raise HTTPException(400, "batch_size / seq_len must be integers")
        if sl_str is not None:
            try:
                sl = int(sl_str)
            except ValueError:
                raise HTTPException(400, "batch_size / seq_len must be integers")
        try:
            return _readers.read_rough(wd, bs, sl)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/calc/cells")
    def calc_cells(task_id: str, request: Request) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        qp = request.query_params
        bs_str = qp.get("batch_size")
        sl_str = qp.get("seq_len")
        bs = None
        sl = None
        if bs_str is not None:
            try:
                bs = int(bs_str)
            except ValueError:
                raise HTTPException(400, "batch_size / seq_len must be integers")
        if sl_str is not None:
            try:
                sl = int(sl_str)
            except ValueError:
                raise HTTPException(400, "batch_size / seq_len must be integers")
        try:
            return _readers.read_cells(wd, bs, sl)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    @router.get("/calc/cell/{compound}/{angle}/{round_idx}")
    def calc_cell_detail(
        task_id: str, compound: str, angle: str, round_idx: int,
    ) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return _readers.read_cell_detail(wd, compound, angle, round_idx)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))

    # ----------------------------------------------------------------- #
    # Orchestrator iteration records / charts / state-graph / retro
    # (used to be shell endpoints; now task-owned because the record
    # schema is task-specific)
    # ----------------------------------------------------------------- #
    @router.get("/iterations")
    def calc_orch_iterations(task_id: str) -> list:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_iterations(state_dir_for(entry))

    @router.get("/iterations/{n}")
    def calc_orch_iteration_detail(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        rec = _state_readers.read_iteration(state_dir_for(entry), n)
        if rec is None:
            raise HTTPException(404, f"no iteration {n} for task {task_id}")
        return rec

    @router.get("/iterations/{n}/retrospective")
    def calc_orch_retrospective(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_retrospective(state_dir_for(entry), n)

    @router.get("/charts")
    def calc_orch_charts(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_charts(state_dir_for(entry))

    @router.get("/state-graph")
    def calc_orch_state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _state_readers.read_state_graph(state_dir_for(entry))

    # ----------------------------------------------------------------- #
    # Offline QA over agent conversation history
    # ----------------------------------------------------------------- #
    # ----------------------------------------------------------------- #
    # Control: kill / restart / rerun_step
    # ----------------------------------------------------------------- #
    @router.post("/control")
    async def calc_control(task_id: str, request: Request) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        body = await request.json()
        action = body.get("action")
        launcher = _launcher.get_default_launcher()

        if action == "rerun_step":
            step_label = body.get("step")
            step_dir = _STEP_MAP.get(step_label or "")
            if not step_dir:
                valid = ", ".join(_STEP_MAP.keys())
                raise HTTPException(
                    400,
                    f"unknown step {step_label!r}; valid: {valid}",
                )

            sd = state_dir_for(entry)
            wd = workspace_dir_for(entry)

            # Kill the orchestrator if it's still running.
            status = launcher.status(task_id)
            if status.running:
                launcher.kill(task_id, force=True)
                await asyncio.sleep(0.5)

            # Wipe the requested step and all subsequent steps.
            start_idx = _STEP_INDEX[step_dir]
            removed: list[str] = []
            for d in _STEP_DIRS[start_idx:]:
                dp = wd / d
                if dp.exists():
                    try:
                        shutil.rmtree(dp, ignore_errors=True)
                    except OSError:
                        pass
                    removed.append(d)

            # Stamp a timeline event so the audit trail records the wipe.
            _sr.append_timeline_event(sd, "rerun_step", {
                "task_id": task_id,
                "step": step_label,
                "removed": removed,
                "rerun_at": time.time(),
            })

            # Restart — the orchestrator picks up from the first missing step.
            req_data = _sr.read_requirements(sd)
            if req_data is None:
                raise HTTPException(400, "no requirements.json to restart from")
            pid = launcher.start(task_id, req_data, sd, wd)
            return {
                "ok": True,
                "action": "rerun_step",
                "step": step_label,
                "removed": removed,
                "pid": pid,
            }

        if action == "kill":
            force = bool(body.get("force", False))
            ok = launcher.kill(task_id, force=force)
            return {"ok": ok, "action": "kill", "force": force}

        if action == "restart":
            sd = state_dir_for(entry)
            req_data = _sr.read_requirements(sd)
            if req_data is None:
                raise HTTPException(400, "no requirements.json to restart from")
            status = launcher.status(task_id)
            if status.running:
                launcher.kill(task_id, force=True)
                await asyncio.sleep(0.5)
            wd = workspace_dir_for(entry)
            pid = launcher.start(task_id, req_data, sd, wd)
            return {"ok": True, "action": "restart", "pid": pid}

        raise HTTPException(400, f"unknown action: {action!r}")

    # ----------------------------------------------------------------- #
    # Offline QA over agent conversation history
    # ----------------------------------------------------------------- #
    register_qa_routes(router, plugin, prefix="/calc/qa")

    return router

