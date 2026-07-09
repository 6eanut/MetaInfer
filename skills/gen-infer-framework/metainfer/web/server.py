"""FastAPI backend for the MetaInfer dashboard.

The server reads the file-based state store + iteration records + the live
SubAgentManager snapshot and exposes JSON endpoints for the frontend.

Run inline (in a daemon thread) alongside the orchestrator via
:func:`start_server_in_thread`, or standalone via :func:`run_server`
(``metainfer web <state_dir>``).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from .. import phases as P
from ..state import StateStore
from ..subagent_manager import SubAgentManager


STATIC_DIR = Path(__file__).resolve().parent / "static"


def _build_app(
    store: StateStore,
    manager: Optional[SubAgentManager],
    iterations_root: Path,
) -> FastAPI:
    app = FastAPI(title="MetaInfer Dashboard")

    # --- Requirements -------------------------------------------------- #
    @app.get("/api/requirements")
    def requirements() -> Dict[str, Any]:
        try:
            return store.load_requirements()
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))

    # --- Run status ---------------------------------------------------- #
    @app.get("/api/run")
    def run_status() -> Dict[str, Any]:
        try:
            rs = store.load_run()
            return json.loads(json.dumps(rs.__dict__, default=str))
        except FileNotFoundError:
            return {"finished": False, "current_phase": "idle", "current_iteration": 0}

    # --- Iterations ---------------------------------------------------- #
    @app.get("/api/iterations")
    def iterations() -> List[Dict[str, Any]]:
        return [r.__dict__ for r in store.load_all_iterations()]

    @app.get("/api/iterations/{n}")
    def iteration_detail(n: int) -> Dict[str, Any]:
        rec = store.load_iteration(n)
        if rec is None:
            raise HTTPException(404, f"iteration {n} not found")
        return rec.__dict__

    # --- Agents -------------------------------------------------------- #
    @app.get("/api/agents")
    def agents() -> Dict[str, Any]:
        if manager is None:
            return {"agents": []}
        return {"agents": manager.snapshot()}

    # --- Timeline ------------------------------------------------------ #
    @app.get("/api/timeline")
    def timeline(since: float = 0.0) -> Dict[str, Any]:
        return {"events": store.load_timeline(since=since)}

    # --- Charts data --------------------------------------------------- #
    @app.get("/api/charts")
    def charts() -> Dict[str, Any]:
        recs = store.load_all_iterations()
        durations = [{"x": r.iteration, "y": round(r.duration_s, 1)} for r in recs if r.duration_s]
        # perf: collect all metric keys across iterations
        perf_keys: List[str] = []
        for r in recs:
            for k in r.perf:
                if k not in perf_keys:
                    perf_keys.append(k)
        perf_series = []
        for k in perf_keys:
            series = [{"x": r.iteration, "y": r.perf.get(k)}
                      for r in recs if k in r.perf]
            perf_series.append({"metric": k, "points": series})
        return {
            "durations": durations,
            "perf_series": perf_series,
            "iteration_status": [
                {"iteration": r.iteration, "status": r.status,
                 "goal": r.goal or ""} for r in recs
            ],
        }

    # --- Phase graph (current state) ---------------------------------- #
    @app.get("/api/state-graph")
    def state_graph() -> Dict[str, Any]:
        """Render the flow diagram directly from :mod:`metainfer.phases`.

        The frontend never hardcodes nodes/edges — it consumes whatever this
        endpoint returns. Editing the transition table in ``phases.py`` is
        enough to update the diagram.
        """
        try:
            rs = store.load_run()
            current: str = rs.current_phase
            last_outcome = rs.last_outcome
            last_label = rs.last_transition_label
        except FileNotFoundError:
            current, last_outcome, last_label = "idle", None, None

        nodes = P.nodes_for_graph()
        edges = P.edges_for_graph()

        # Compute the "active edge" — the transition the orchestrator just
        # took — so the frontend can highlight it. We match on the
        # destination phase + one of the labels merged into the edge.
        active_edge: Optional[Dict[str, str]] = None
        if last_label:
            for e in edges:
                if e["to"] == current and last_label in e["label"].split(" / "):
                    active_edge = {"from": e["from"], "to": e["to"], "label": last_label}
                    break

        return {
            "current": current,
            "nodes": nodes,
            "edges": edges,
            "active_edge": active_edge,
            "last_outcome": last_outcome,
            "outcome_legend": [
                {"id": o, "label": P.outcome_label(o)} for o in P.ALL_OUTCOMES
            ],
        }

    # --- Static frontend ---------------------------------------------- #
    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def run_server(
    store: StateStore,
    manager: Optional[SubAgentManager],
    iterations_root: Path,
    port: int = 8765,
    host: str = "127.0.0.1",
) -> None:
    """Block forever running uvicorn. Use from the CLI's `web` subcommand."""
    import uvicorn
    app = _build_app(store, manager, iterations_root)
    uvicorn.run(app, host=host, port=port, log_level="info")


def start_server_in_thread(
    store: StateStore,
    manager: Optional[SubAgentManager],
    iterations_root: Path,
    port: int = 8765,
    host: str = "127.0.0.1",
) -> threading.Thread:
    """Start the dashboard in a daemon thread. Returns immediately."""
    import uvicorn
    app = _build_app(store, manager, iterations_root)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, name="metainfer-web", daemon=True)
    t.start()
    return t
