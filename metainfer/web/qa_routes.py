"""Generic per-task-type QA routes.

This helper mounts the ``POST /qa/start`` + ``GET /qa/<sid>`` +
``GET /qa`` triplet for a task type, backed by the task-type-agnostic
:mod:`metainfer.web.qa` engine. Each task plugin that wants offline-QA
support calls :func:`register_qa_routes` from its ``register_routes``
hook — no boilerplate copy-pasted per package.

Two modes are supported by the same routes:

* **Frontend-driven** (default): the request body contains
  ``events_file`` (and optionally ``target_workdir``, ``target_label``).
  The route forwards them straight to the QA engine.
* **Server-side resolution**: the body contains a tuple like
  ``{step, round, agent}`` or ``{iteration, agent}``. The route calls
  ``plugin.qa_config.resolve_target(state_dir, body)`` to map the tuple
  to an ``events_file`` path before invoking the engine.

The plugin's :class:`~metainfer.web.registry.WebPlugin` already holds a
``qa_config``; this helper just consumes it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException

from . import qa as _qa
from ._helpers import (
    require_task_type,
    state_dir_for,
    task_or_404,
)
from .registry import WebPlugin


def register_qa_routes(
    app: FastAPI,
    plugin: WebPlugin,
    *,
    prefix: str = "/qa",
) -> None:
    """Mount the three QA routes onto ``app`` for ``plugin.type``.

    ``prefix`` is the URL path suffix appended after
    ``/api/tasks/{task_id}``; defaults to ``"/qa"``. Plugins with an
    existing stable URL (e.g. a task that historically mounted QA at
    ``/<plugin>/qa``) can override it.

    Routes mounted:
      - ``POST /api/tasks/{task_id}{prefix}/start``
      - ``GET  /api/tasks/{task_id}{prefix}/{session_id}``
      - ``GET  /api/tasks/{task_id}{prefix}``
    """
    plugin_type = plugin.type
    qa_config = plugin.qa_config

    @app.post(f"/api/tasks/{{task_id}}{prefix}/start")
    def _qa_start(task_id: str, body: dict) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, plugin_type)
        sd = state_dir_for(entry)
        payload = body or {}
        # If the caller didn't pass an explicit events_file, try
        # resolving via the plugin's pathsolver (server-side resolution).
        if not (payload.get("events_file") or "").strip() and qa_config is not None:
            try:
                resolved = qa_config.resolve_target(sd, payload)
            except FileNotFoundError as exc:
                raise HTTPException(404, str(exc))
            except ValueError as exc:
                raise HTTPException(400, str(exc))
            except NotImplementedError as exc:
                raise HTTPException(501, str(exc))
            # pathsolvers return Path objects for events_file / target_workdir;
            # the QA engine expects strings.
            if resolved.get("events_file") is not None:
                resolved["events_file"] = str(resolved["events_file"])
            if resolved.get("target_workdir") is not None:
                resolved["target_workdir"] = str(resolved["target_workdir"])
            # Merge resolved fields back; caller-provided label wins.
            payload = {
                **resolved,
                **{k: v for k, v in payload.items()
                   if k not in ("events_file", "target_workdir", "target_label")},
            }
        try:
            sid = _qa.start_qa_session(sd, payload)
        except _qa.EventsFileNotFound as exc:
            raise HTTPException(404, str(exc))
        except _qa.BudgetExhausted as exc:
            raise HTTPException(429, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"session_id": sid, "task_id": task_id}

    @app.get(f"/api/tasks/{{task_id}}{prefix}/{{session_id}}")
    def _qa_get(task_id: str, session_id: str) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, plugin_type)
        sd = state_dir_for(entry)
        sess = _qa.get_qa_session(sd, session_id)
        if sess is None:
            raise HTTPException(404, f"no such qa session: {session_id}")
        return sess

    @app.get(f"/api/tasks/{{task_id}}{prefix}")
    def _qa_list(
        task_id: str,
        step: Optional[str] = None,
        round: Optional[str] = None,
        agent: Optional[str] = None,
        iteration: Optional[str] = None,
    ) -> Dict[str, Any]:
        entry = task_or_404(task_id)
        require_task_type(entry, plugin_type)
        sd = state_dir_for(entry)
        sessions = _qa.list_qa_sessions(
            sd, step=step, round_=round, agent=agent,
        )
        if iteration is not None:
            sessions = [s for s in sessions if str(s.get("iteration")) == str(iteration)]
        return {"sessions": sessions}
