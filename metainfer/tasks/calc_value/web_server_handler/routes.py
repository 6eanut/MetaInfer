"""FastAPI route handlers for the calc-theoretical-value task type.

All 11 ``/api/tasks/{task_id}/calc/...`` routes live here. They delegate
to :mod:`._readers` for disk reads (against the task's ``workspace_dir``
— where step0..step4 outputs physically live) and to
:mod:`metainfer.web.qa` for the offline-analyst feature (sessions stored
under the task's ``state_dir``). Type guard is enforced at the route
layer via :func:`metainfer.web._helpers.require_task_type` for safety —
the ``/calc/`` prefix is suggestive but not authoritative.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from metainfer.web._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
    workspace_dir_for,
)
from . import _readers

PLUGIN_TYPE = "calc-theoretical-value"


def register_routes(app: FastAPI, deps) -> None:
    """Mount the 11 calc routes onto ``app``.

    ``deps`` is a :class:`metainfer.web.registry.WebDeps`; we don't
    currently use it (calc routes are fully self-contained against the
    state_dir + workspace_dir), but it's accepted for symmetry with
    other plugins.
    """

    @app.get("/api/tasks/{task_id}/calc/graph")
    def calc_graph(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return _readers.read_graph(wd)
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/tasks/{task_id}/calc/compute")
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

    @app.get("/api/tasks/{task_id}/calc/viz")
    def calc_viz(task_id: str) -> HTMLResponse:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        wd = workspace_dir_for(entry)
        try:
            return HTMLResponse(_readers.read_viz(wd))
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc))

    @app.get("/api/tasks/{task_id}/calc/summary")
    def calc_summary(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _readers.read_summary(workspace_dir_for(entry))

    @app.get("/api/tasks/{task_id}/calc/iterations")
    def calc_iterations(task_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        return _readers.read_iterations(workspace_dir_for(entry))

    @app.get("/api/tasks/{task_id}/calc/rough")
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

    @app.get("/api/tasks/{task_id}/calc/cells")
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

    @app.get("/api/tasks/{task_id}/calc/cell/{compound}/{angle}/{round_idx}")
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
    # Offline QA over agent conversation history
    # ----------------------------------------------------------------- #
    # Lets the user click an agent in the iterations panel and ask a
    # follow-up question. A fresh ccb subprocess (the "analyst") is
    # spawned with read access to the agent's events.jsonl transcript;
    # the analyst answers based on what the original agent actually
    # did. QA sessions are stored under state_dir (metadata), so we
    # use state_dir_for here rather than workspace_dir_for. See
    # metainfer/web/qa.py for lifecycle / storage.

    @app.post("/api/tasks/{task_id}/calc/qa/start")
    def calc_qa_start(task_id: str, body: dict) -> Dict[str, Any]:
        from ... import qa as _qa
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        sd = state_dir_for(entry)
        try:
            sid = _qa.start_qa_session(sd, body or {})
        except _qa.EventsFileNotFound as exc:
            raise HTTPException(404, str(exc))
        except _qa.BudgetExhausted as exc:
            # 429 + Retry-After-ish body so the client can surface a
            # meaningful "task over budget" message instead of a generic
            # 500.
            raise HTTPException(429, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"session_id": sid, "task_id": task_id}

    @app.get("/api/tasks/{task_id}/calc/qa/{session_id}")
    def calc_qa_get(task_id: str, session_id: str) -> Dict[str, Any]:
        from ... import qa as _qa
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        sd = state_dir_for(entry)
        sess = _qa.get_qa_session(sd, session_id)
        if sess is None:
            raise HTTPException(404, f"no such qa session: {session_id}")
        return sess

    @app.get("/api/tasks/{task_id}/calc/qa")
    def calc_qa_list(
        task_id: str,
        step: Optional[str] = None,
        round: Optional[str] = None,
        agent: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, PLUGIN_TYPE)
        sd = state_dir_for(entry)
        from ... import qa as _qa
        sessions = _qa.list_qa_sessions(
            sd, step=step, round_=round, agent=agent,
        )
        return {"sessions": sessions}
