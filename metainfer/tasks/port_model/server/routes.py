"""API routes for port-model, mounted at /api/port-model/{task_id}.

Endpoints:
    GET  /iterations                — all iteration records
    GET  /iterations/{n}            — one iteration detail
    GET  /state-graph               — phase graph payload
    GET  /memory/{name}             — canonical memory/<name>.md
    GET  /phase-summary/{phase}     — phase workdir's summary.md
    GET  /dumps                     — P5 hidden_state dump index
    GET  /p6-iterations             — P6 attempt dirs + verdicts
    POST /control                   — {action: rerun_step|kill|restart}
    QA  /qa/...                     — offline QA over agent transcripts
"""

from __future__ import annotations

import asyncio
import shutil
import time
from typing import Any, Dict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse

from metainfer.server import launcher as _launcher
from metainfer.server import state_reader as _sr
from metainfer.server._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
    workspace_dir_for,
)
from metainfer.server.qa_routes import register_qa_routes

from ._state_readers import (
    read_iteration,
    read_iterations,
    read_memory_markdown,
    read_p5_dumps_index,
    read_p6_iterations,
    read_phase_summary,
    read_state_graph,
)


PLUGIN_TYPE = "port-model"

# Step ordering — when re-running step N, we wipe steps N..last.
# These correspond to the 6 phases + their workdir subdirs.
_STEP_DIRS = ["p1", "p2", "p3", "p4", "p5", "p6"]
_STEP_INDEX = {s: i for i, s in enumerate(_STEP_DIRS)}

_STEP_MAP = {
    "P1_weight_analysis": "p1",
    "P2_framework_analysis": "p2",
    "P3_architect_review": "p3",
    "P4_minimal_framework": "p4",
    "P5_verify_minimal": "p5",
    "P6_port_engine": "p6",
}


def build_router(plugin) -> APIRouter:
    router = APIRouter()

    # ----------------------------------------------------------------- #
    # Orchestrator iteration records / state-graph
    # ----------------------------------------------------------------- #
    @router.get("/iterations")
    def port_iterations(task_id: str) -> list:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return read_iterations(state_dir_for(entry))

    @router.get("/iterations/{n}")
    def port_iteration_detail(task_id: str, n: int) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        rec = read_iteration(state_dir_for(entry), n)
        if rec is None:
            raise HTTPException(404, f"no iteration {n} for task {task_id}")
        return rec

    @router.get("/state-graph")
    def port_state_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return read_state_graph(state_dir_for(entry))

    # ----------------------------------------------------------------- #
    # Per-phase artifacts
    # ----------------------------------------------------------------- #
    @router.get("/memory/{name}")
    def port_memory(task_id: str, name: str) -> PlainTextResponse:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        content = read_memory_markdown(workspace_dir_for(entry), name)
        if content is None:
            raise HTTPException(404, f"memory/{name}.md not found")
        return PlainTextResponse(content, media_type="text/markdown")

    @router.get("/phase-summary/{phase}")
    def port_phase_summary(task_id: str, phase: str) -> PlainTextResponse:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        content = read_phase_summary(workspace_dir_for(entry), phase)
        if content is None:
            raise HTTPException(404, f"no summary.md for phase {phase}")
        return PlainTextResponse(content, media_type="text/markdown")

    @router.get("/dumps")
    def port_dumps(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return read_p5_dumps_index(workspace_dir_for(entry))

    @router.get("/p6-iterations")
    def port_p6_iters(task_id: str) -> list:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return read_p6_iterations(workspace_dir_for(entry))

    # ----------------------------------------------------------------- #
    # Control: rerun_step / kill / restart
    # ----------------------------------------------------------------- #
    @router.post("/control")
    async def port_control(task_id: str, request: Request) -> Dict[str, Any]:
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
                    400, f"unknown step {step_label!r}; valid: {valid}"
                )

            sd = state_dir_for(entry)
            wd = workspace_dir_for(entry)

            # Kill orchestrator if still running.
            status = launcher.status(task_id)
            if status.running:
                launcher.kill(task_id, force=True)
                await asyncio.sleep(0.5)

            # Wipe the requested step's workdir AND every subsequent
            # step's workdir, since downstream artifacts depend on
            # upstream outputs. Also clear the canonical memory copy
            # for steps that produce one (P1, P3).
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
                dp.mkdir(parents=True, exist_ok=True)
            # Drop canonical memory artifacts so resume logic re-detects —
            # but ONLY for steps at or below the one being re-run. Re-running
            # P6 (start_idx=5) must NOT wipe P1/P3 canonical copies, since
            # P6 reads them and they are not regenerated.
            _memory_artifact_step = {
                "p1_weight_analysis": _STEP_INDEX["p1"],
                "p3_consolidated_spec": _STEP_INDEX["p3"],
            }
            for art_name, art_step_idx in _memory_artifact_step.items():
                if start_idx > art_step_idx:
                    continue
                mp = wd / "memory" / f"{art_name}.md"
                if mp.is_file():
                    try:
                        mp.unlink()
                    except OSError:
                        pass
            # If we wiped P5 or later, also clear dumps/.
            if start_idx <= _STEP_INDEX["p5"]:
                dp = wd / "dumps"
                if dp.exists():
                    shutil.rmtree(dp, ignore_errors=True)
                dp.mkdir(parents=True, exist_ok=True)

            _sr.append_timeline_event(sd, "rerun_step", {
                "task_id": task_id, "step": step_label, "removed": removed,
                "rerun_at": time.time(),
            })

            req_data = _sr.read_requirements(sd)
            if req_data is None:
                raise HTTPException(400, "no requirements.json to restart from")
            pid = launcher.start(task_id, req_data, sd, wd)
            return {
                "ok": True, "action": "rerun_step",
                "step": step_label, "removed": removed, "pid": pid,
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
    register_qa_routes(router, plugin, prefix="/qa")
    return router
