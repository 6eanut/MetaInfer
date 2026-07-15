"""FastAPI WebUI server. The long-lived main process.

Responsibilities:
  - Serve the static SPA frontend (``metainfer/static/``)
  - Expose JSON API for task list / detail / control
  - Spawn / kill per-task orchestrator subprocesses via the launcher
  - Push live updates via SSE

No in-memory task state — every read endpoint derives from files on
disk (state_reader), every write goes through the launcher / task
registry.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from . import forms as _forms
from . import launcher as _launcher
from . import paths as _paths
from . import reconcile as _reconcile
from . import sse as _sse
from . import state_reader as _sr
from . import tasks as _tasks
from ._helpers import (
    state_dir_for as _state_dir_for,
    task_or_404 as _task_or_404,
    workspace_dir_for as _workspace_dir_for,
)
from .registry import WebDeps as _WebDeps
from .registry import WebPlugin as _WebPlugin
from .registry import all_plugins as _all_web_plugins
from .registry import get as _get_web_plugin
from .. import tasks as _task_packages  # noqa: F401 — side-effect: register all task plugins (orchestrator + web)
from ..orchestrator.paths import repo_root as _repo_root


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


# Backward-compat aliases — keep the historical names that the rest of
# app.py uses inline. Both point at the same callable as the extracted
# helpers, so behavior is unchanged.


def _plugin_view_hint(task_type: str) -> Dict[str, Any]:
    """Return the frontend detail-view hint fields for a task type.

    Plugins set ``detail_view_module`` (an importmap key) on their
    WebPlugin; the frontend uses it to dynamically dispatch the task
    detail view. Returns an empty dict when no plugin is registered for
    this task type (the frontend then renders its default view).
    """
    plugin = _get_web_plugin(task_type)
    if plugin is None or not plugin.detail_view_module:
        return {}
    return {
        "detail_view_module": plugin.detail_view_module,
        "detail_view_export": plugin.detail_view_export,
    }


def create_app() -> FastAPI:
    app = FastAPI(title="MetaInfer", docs_url=None, redoc_url=None)

    # Reconcile runtime state with the actual process table. This makes
    # the WebUI crash-safe: on restart, any orchestrator subprocesses
    # that survived from the previous session are picked back up, and
    # stale entries for orchestrators that have died are cleaned.
    try:
        _reconcile.reconcile()
    except Exception as e:  # noqa: BLE001 — startup must not crash
        import sys
        print(f"[metainfer-web] reconciliation failed: {e!r}", file=sys.stderr)

    # ------------------------------------------------------------------ #
    # Task types / forms
    # ------------------------------------------------------------------ #
    @app.get("/api/task-types")
    def task_types() -> List[Dict[str, str]]:
        return _forms.list_task_types()

    @app.get("/api/task-types/{task_type}/schema")
    def task_type_schema(task_type: str) -> Dict[str, Any]:
        schema = _forms.load_form_schema(task_type)
        if schema is None:
            raise HTTPException(404, f"unknown task type: {task_type}")
        return schema

    # ------------------------------------------------------------------ #
    # Task CRUD
    # ------------------------------------------------------------------ #
    @app.get("/api/tasks")
    def list_tasks() -> Dict[str, Any]:
        launcher = _launcher.get_default_launcher()
        out = []
        for e in _tasks.list_tasks():
            status = launcher.status(e.id).to_dict()
            out.append({
                "id": e.id, "type": e.type, "label": e.label,
                "state_dir": e.state_dir, "created_at": e.created_at,
                "launcher": e.launcher,
                "status": status,
                **_plugin_view_hint(e.type),
            })
        return {"tasks": out}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        sd = _state_dir_for(entry)
        launcher = _launcher.get_default_launcher()
        return {
            "id": entry.id, "type": entry.type, "label": entry.label,
            "state_dir": entry.state_dir, "created_at": entry.created_at,
            "launcher": entry.launcher,
            "status": launcher.status(task_id).to_dict(),
            "requirements": _sr.read_requirements(sd),
            "run": _sr.read_run(sd),
            **_plugin_view_hint(entry.type),
        }

    @app.get("/api/tasks/{task_id}/run")
    def task_run(task_id: str) -> Dict[str, Any]:
        """RunStatus only — convenience for clients that don't need the
        full task envelope. Same file as ``get_task()['run']``."""
        entry = _task_or_404(task_id)
        return _sr.read_run(_state_dir_for(entry))

    @app.post("/api/tasks")
    async def create_task(request: Request) -> Dict[str, Any]:
        try:
            body = await request.json()
        except Exception:  # noqa: BLE001 — malformed JSON
            raise HTTPException(400, "request body must be valid JSON")
        if not isinstance(body, dict):
            raise HTTPException(400, "request body must be a JSON object")
        task_type = body.get("type")
        label = body.get("label") or ""
        answers = body.get("answers") or {}
        extra_args = body.get("extra_args") or []
        if not task_type:
            raise HTTPException(400, "missing 'type'")
        # Validate against schema before doing anything destructive.
        v = _forms.validate_submission(task_type, answers)
        if not v["ok"]:
            raise HTTPException(400, detail={"errors": v["errors"]})
        # Generate a unique task id.
        task_id = _tasks.gen_task_id(task_type, label)
        sd = _paths.task_dir(task_id)
        sd.mkdir(parents=True, exist_ok=True)
        # Workspace dir holds task-package-generated artifacts (each task
        # type owns its own workspace subtree); state_dir holds metadata + logs.
        wd = _paths.workspace_dir(task_id)
        wd.mkdir(parents=True, exist_ok=True)
        # Build the requirements.json the orchestrator will read.
        _plugin = _get_web_plugin(task_type)
        _default_label = (_plugin.label if _plugin else "") or task_type
        requirements = {
            "task_id": task_id,
            "task_type": task_type,
            "raw_request": body.get("raw_request") or "",
            "label": label or _default_label,
            **answers,
        }
        # Register first (so list view picks it up even if spawn fails).
        entry = _tasks.TaskEntry(
            id=task_id, type=task_type, label=label or _default_label,
            state_dir=str(sd), workspace_dir=str(wd),
            created_at=time.time(), launcher="local",
        )
        _tasks.add_task(entry)
        # Spawn the orchestrator.
        launcher = _launcher.get_default_launcher()
        try:
            pid = launcher.start(task_id, requirements, sd, wd, extra_args=extra_args)
        except Exception as e:  # noqa: BLE001
            # Spawn failed — keep the registration so the user can see
            # the error in the UI, but mark as not running.
            _tasks.update_task(task_id, pid=None, finished_at=time.time())
            raise HTTPException(500, detail={"error": f"spawn failed: {e!r}"})
        return {"task_id": task_id, "pid": pid,
                "state_dir": str(sd), "workspace_dir": str(wd)}

    @app.delete("/api/tasks/{task_id}")
    def delete_task(task_id: str, purge: bool = False) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        # Kill the orchestrator if it's still running.
        launcher = _launcher.get_default_launcher()
        if launcher.status(task_id).running:
            launcher.kill(task_id, force=True)
        # Optionally remove state_dir + workspace_dir from disk.
        removed_files = False
        if purge:
            import shutil
            sd = Path(entry.state_dir)
            if sd.exists():
                shutil.rmtree(sd, ignore_errors=True)
                removed_files = True
            wd = _workspace_dir_for(entry)
            if wd.exists():
                shutil.rmtree(wd, ignore_errors=True)
                removed_files = True
        _tasks.remove_task(task_id)
        return {"removed_from_registry": True, "purged_files": removed_files}

    @app.post("/api/tasks/{task_id}/control")
    async def control_task(task_id: str, request: Request) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        body = await request.json()
        action = body.get("action")
        launcher = _launcher.get_default_launcher()
        if action == "kill":
            force = bool(body.get("force", False))
            ok = launcher.kill(task_id, force=force)
            return {"ok": ok, "action": "kill", "force": force}
        if action == "restart":
            sd = _state_dir_for(entry)
            req = _sr.read_requirements(sd)
            if req is None:
                raise HTTPException(400, "no requirements.json to restart from")
            # Capture state BEFORE killing so the audit trail records
            # what we're resuming from. The orchestrator's own
            # `orchestrator_resume` event will follow once it starts;
            # our `restart_initiated` here explains WHY a new process
            # appeared.
            prior_status = launcher.status(task_id).to_dict()
            prior_run = _sr.read_run(sd) or {}
            _sr.append_timeline_event(sd, "restart_initiated", {
                "task_id": task_id,
                "prior_pid": prior_status.get("pid"),
                "prior_running": prior_status.get("running"),
                "prior_phase": prior_run.get("current"),
                "prior_iteration": prior_run.get("iteration_count")
                                or prior_run.get("iteration"),
                "prior_outcome": prior_run.get("outcome"),
                # The new orchestrator will call init_or_resume() which
                # detects run.json + iterations/, picks the right phase
                # via _prepare_resume(), and continues forward. NO
                # state is cleared on restart — iterations, code, logs,
                # timeline all persist.
                "resume_mode": "preserve_state",
            })
            if prior_status.get("running"):
                launcher.kill(task_id, force=True)
                # Give it a beat to die. The orchestrator's SIGTERM
                # handler calls manager.shutdown() before os._exit(),
                # which can take a moment if sub-agents are mid-call.
                await asyncio.sleep(0.5)
            wd = _workspace_dir_for(entry)
            pid = launcher.start(task_id, req, sd, wd)
            return {
                "ok": True, "action": "restart", "pid": pid,
                "prior_status": prior_status,
            }
        if action == "reset":
            # Destructive: wipe everything except requirements.json so the
            # task is back to its just-created state. Only allowed when
            # the orchestrator is stopped — caller must kill first.
            if launcher.status(task_id).running:
                raise HTTPException(
                    409, "task is still running; kill it before resetting",
                )
            sd = _state_dir_for(entry)
            prior_run = _sr.read_run(sd) or {}
            tid = prior_run.get("task_id") or task_id
            ttype = prior_run.get("task_type") or entry.type
            wd = _workspace_dir_for(entry)
            summary = _sr.reset_state_dir(sd, wd, tid, ttype)
            # Note: we deliberately don't touch the registry here — the
            # stale pid/finished_at values are inert because launcher.status
            # reads the (now-wiped) pid file and reports "not running".
            # The next `launcher.start` will refresh the registry with the
            # new pid.
            return {"ok": True, "action": "reset", **summary}
        raise HTTPException(400, f"unknown action: {action}")

    # ------------------------------------------------------------------ #
    # Task detail data (all file-derived)
    # ------------------------------------------------------------------ #
    @app.get("/api/tasks/{task_id}/iterations")
    def task_iterations(task_id: str) -> List[Dict[str, Any]]:
        entry = _task_or_404(task_id)
        return _sr.read_iterations(_state_dir_for(entry))

    @app.get("/api/tasks/{task_id}/iterations/{n}")
    def task_iteration_detail(task_id: str, n: int) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        rec = _sr.read_iteration(_state_dir_for(entry), n)
        if rec is None:
            raise HTTPException(404, f"no iteration {n} for task {task_id}")
        return rec

    @app.get("/api/tasks/{task_id}/iterations/{n}/retrospective")
    def task_retrospective(task_id: str, n: int) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        return _sr.read_retrospective(_state_dir_for(entry), n)

    @app.get("/api/tasks/{task_id}/timeline")
    def task_timeline(task_id: str, since: float = 0.0) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        return {"events": _sr.read_timeline(_state_dir_for(entry), since=since)}

    @app.get("/api/tasks/{task_id}/charts")
    def task_charts(task_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        return _sr.read_charts(_state_dir_for(entry))

    @app.get("/api/tasks/{task_id}/state-graph")
    def task_state_graph(task_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        return _sr.read_state_graph(_state_dir_for(entry))

    @app.get("/api/tasks/{task_id}/agents")
    def task_agents(task_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        return _sr.read_agents(_state_dir_for(entry))

    @app.get("/api/tasks/{task_id}/token-budget")
    def task_token_budget(task_id: str) -> Dict[str, Any]:
        """Return the task's token-cost budget snapshot.

        Reads ``<state_dir>/token_budget.json`` directly (no in-process
        TokenBudget construction needed on the read side) and returns a
        flat dict the WebUI can render as a progress bar. Returns 200
        with ``{"configured": false}`` when no budget file exists yet
        — the caller renders nothing in that case.
        """
        entry = _task_or_404(task_id)
        budget_path = _state_dir_for(entry) / "token_budget.json"
        if not budget_path.exists():
            return {"configured": False}
        try:
            data = json.loads(budget_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return {"configured": False, "error": "budget file unreadable"}
        cfg = data.get("config") or {}
        totals = data.get("totals") or {}
        used = float(totals.get("total_cost_usd", 0.0))
        limit = cfg.get("max_cost_usd")
        hard = cfg.get("max_cost_usd_hard")
        exhausted = bool(totals.get("exhausted"))
        hard_exhausted = bool(totals.get("hard_exhausted"))
        pct = None
        if isinstance(limit, (int, float)) and limit > 0:
            pct = round(min(100.0, (used / float(limit)) * 100.0), 2)
        return {
            "configured": True,
            "used_cost_usd": used,
            "limit_cost_usd": limit,
            "hard_limit_cost_usd": hard,
            "exhausted": exhausted,
            "hard_exhausted": hard_exhausted,
            "used_pct": pct,
            "agent_count": int(totals.get("agent_count", 0)),
            "total_input_tokens": int(totals.get("total_input_tokens", 0)),
            "total_output_tokens": int(totals.get("total_output_tokens", 0)),
            "total_cache_read_input_tokens": int(
                totals.get("total_cache_read_input_tokens", 0)),
            "per_source": data.get("per_source") or {},
            "per_phase": data.get("per_phase") or {},
        }

    @app.post("/api/tasks/{task_id}/token-budget")
    def task_token_budget_update(task_id: str, body: dict) -> Dict[str, Any]:
        """Adjust the task's cost limit at runtime.

        Body: ``{"max_cost_usd": <float|null>, "max_cost_usd_hard": <float|null>}``.
        Keys are optional — only the ones you pass get updated; pass
        ``null`` to clear that limit.

        Effects:
          - Atomically rewrites ``token_budget.json::config`` with the
            new limits. If the file doesn't yet exist (legacy task
            created before this feature, or a freshly-created task
            whose orchestrator hasn't started), it gets CREATED here
            with empty totals — so you can retroactively cap a task
            that's already running without budget enforcement.
          - If the orchestrator is still alive, its in-memory
            :class:`TokenBudget` notices the file mtime change on its
            next snapshot/poll and picks up the new limit. No IPC
            needed.
          - If the run has already aborted (``final_status=aborted``),
            the user should click Restart after raising the budget;
            the new orchestrator process loads the updated limit from
            disk automatically.

        Returns the post-update snapshot (same shape as the GET endpoint).
        """
        entry = _task_or_404(task_id)
        sd = _state_dir_for(entry)
        budget_path = sd / "token_budget.json"
        # Read existing file, mutate config block only, write back.
        # If no file exists yet, start from an empty skeleton.
        if budget_path.exists():
            try:
                data = json.loads(budget_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise HTTPException(500, f"cannot read budget file: {exc}")
        else:
            data = {
                "schema_version": 1,
                "config": {},
                "totals": {
                    "total_cost_usd": 0.0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_cache_read_input_tokens": 0,
                    "agent_count": 0,
                    "exhausted": False,
                    "hard_exhausted": False,
                },
                "per_source": {},
                "per_phase": {},
                "records": [],
            }
        if not isinstance(data, dict):
            data = {}
        cfg = data.setdefault("config", {})
        if not isinstance(cfg, dict):
            cfg = {}
            data["config"] = cfg
        body = body or {}
        if "max_cost_usd" in body:
            v = body["max_cost_usd"]
            cfg["max_cost_usd"] = (float(v) if v is not None else None)
        if "max_cost_usd_hard" in body:
            v = body["max_cost_usd_hard"]
            cfg["max_cost_usd_hard"] = (float(v) if v is not None else None)
        # Recompute totals.exhausted so the GET response is immediately
        # consistent (the orchestrator's hot-reload will fix it on its
        # own next read, but we want the response to be truthful).
        totals = data.setdefault("totals", {})
        if not isinstance(totals, dict):
            totals = {}
            data["totals"] = totals
        # Don't lose existing totals when creating fresh on legacy task
        totals.setdefault("total_cost_usd", 0.0)
        used = float(totals.get("total_cost_usd", 0.0))
        soft = cfg.get("max_cost_usd")
        hard = cfg.get("max_cost_usd_hard")
        totals["exhausted"] = bool(
            isinstance(soft, (int, float)) and used >= soft)
        totals["hard_exhausted"] = bool(
            isinstance(hard, (int, float)) and used >= hard)
        # Atomic write. Bump mtime explicitly via os.utime so the
        # orchestrator's hot-reload (which keys off mtime) always
        # notices — even when the write happens within the same
        # second as the previous one.
        tmp = budget_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(budget_path)
        os.utime(budget_path)  # bump mtime to "now"
        # Echo back the updated snapshot by re-reading via GET handler.
        return task_token_budget(task_id)

    @app.get("/api/tasks/{task_id}/log")
    def task_log(task_id: str, tail_bytes: int = 65536) -> Dict[str, Any]:
        """Tail of the orchestrator's stdout+stderr log."""
        entry = _task_or_404(task_id)
        p = _state_dir_for(entry) / "orchestrator.log"
        if not p.exists():
            return {"content": "", "truncated": False}
        try:
            data = p.read_bytes()
        except OSError:
            return {"content": "", "truncated": False}
        truncated = len(data) > tail_bytes
        if truncated:
            data = data[-tail_bytes:]
        return {
            "content": data.decode("utf-8", errors="replace"),
            "truncated": truncated,
        }

    # ------------------------------------------------------------------ #
    # Plugin routes (per task type)
    # ------------------------------------------------------------------ #
    # Each WebPlugin registers its task-type-specific routes here. Adding
    # a new task type: drop a self-contained package under
    # ``metainfer/tasks/<name>/`` with a ``web_server_handler/plugin.py``
    # that calls ``register(WebPlugin(...))``. Auto-discovery via
    # ``metainfer/tasks/__init__.py`` picks it up; no edits to app.py.
    _web_deps = _WebDeps(
        repo_root=_repo_root(),
        get_launcher=_launcher.get_default_launcher,
    )
    for _plugin in _all_web_plugins():
        if _plugin.register_routes is not None:
            _plugin.register_routes(app, _web_deps, _plugin)

    # ------------------------------------------------------------------ #
    # SSE stream
    # ------------------------------------------------------------------ #
    @app.get("/api/events")
    async def events(request: Request) -> StreamingResponse:
        await _sse.watcher.start()
        q = await _sse.watcher.subscribe()

        async def gen():
            try:
                # Initial hello so the client knows the stream is alive.
                yield "event: hello\ndata: {}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"event: {ev['type']}\ndata: {json.dumps(ev)}\n\n"
                    except asyncio.TimeoutError:
                        # Comment-line keepalive — keeps proxies from
                        # closing the connection during quiet periods.
                        yield ": keepalive\n\n"
            finally:
                await _sse.watcher.unsubscribe(q)

        return StreamingResponse(
            gen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering
            },
        )

    # ------------------------------------------------------------------ #
    # Static frontend
    # ------------------------------------------------------------------ #
    def _cache_bust_token() -> str:
        """Compute a version token from the latest mtime across all
        static files (web shell + every plugin's frontend dir).
        Embedded into index.html as ?v=<token> on every JS/CSS URL so
        the browser fetches fresh modules after any code change.
        """
        mtimes: List[float] = []
        roots = [STATIC_DIR] + [
            p.frontend_dir for p in _all_web_plugins()
            if p.frontend_dir and p.frontend_dir.exists()
        ]
        for root in roots:
            try:
                mtimes.extend(pp.stat().st_mtime for pp in root.rglob("*") if pp.is_file())
            except OSError:
                pass
        return str(int(max(mtimes))) if mtimes else "0"

    # ------------------------------------------------------------------ #
    # Importmap + stylesheet merge — server-side single source of truth
    # ------------------------------------------------------------------ #
    #
    # The shell provides default entries for shared widgets (charts /
    # state-graph / iterations-table / etc.) directly inside
    # ``index.html``'s ``<script type="importmap">`` block. Plugins that
    # want to ship a divergent version of one of these widgets simply
    # include the same key in their ``importmap_entries`` — the merge
    # below lets plugin entries OVERRIDE shell entries with the same key.
    #
    # In addition, every ``*.js`` directly under a plugin's
    # ``frontend_dir`` is auto-registered as ``app/<stem>`` →
    # ``/static/plugins/<type>/<file>?v=<token>``. Plugin authors only
    # need to populate ``importmap_entries`` for keys that DON'T follow
    # the ``app/<filename-stem>`` convention, or to override shell
    # entries (e.g. ship a divergent ``app/state-graph``).
    #
    # Plugin stylesheets (``extra_stylesheets``) are injected as
    # ``<link>`` tags right after the shell stylesheet, so a plugin can
    # ship task-type-specific CSS without editing the shell's
    # ``styles.css``.
    #
    # We merge on the SERVER (rather than emitting duplicate keys and
    # relying on browser last-wins semantics) because duplicate importmap
    # keys are spec-undefined and Chrome rejects the entire importmap on
    # duplicates in some versions — server merge is the safe path.

    _SHELL_IMPORTMAP_CACHE: Dict[str, Any] = {}

    def _shell_importmap() -> Dict[str, str]:
        """Parse the shell importmap from index.html once, cache it."""
        if _SHELL_IMPORTMAP_CACHE:
            return _SHELL_IMPORTMAP_CACHE["entries"]
        import re
        html_text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        m = re.search(
            r'<script[^>]*type="importmap"[^>]*>(.*?)</script>',
            html_text, re.DOTALL,
        )
        if not m:
            _SHELL_IMPORTMAP_CACHE["entries"] = {}
            return {}
        try:
            block = json.loads(m.group(1))
            entries = dict(block.get("imports", {}))
        except (ValueError, TypeError):
            entries = {}
        _SHELL_IMPORTMAP_CACHE["entries"] = entries
        return entries

    def _plugin_auto_importmap(plugin: "_WebPlugin") -> Dict[str, str]:
        """Auto-discover importmap entries by globbing the plugin's
        ``frontend_dir`` for ``*.js``. Each file becomes
        ``app/<stem>`` → ``/static/plugins/<type>/<file>`` (CACHE_BUST
        substituted later). Explicit ``importmap_entries`` on the plugin
        override these."""
        out: Dict[str, str] = {}
        if not plugin.frontend_dir or not plugin.frontend_dir.is_dir():
            return out
        prefix = f"/static/plugins/{plugin.type}"
        try:
            files = sorted(plugin.frontend_dir.glob("*.js"))
        except OSError:
            return out
        for p in files:
            key = f"app/{p.stem}"
            out[key] = f"{prefix}/{p.name}?v=CACHE_BUST"
        return out

    def _merged_importmap(token: str) -> Dict[str, str]:
        """Merge shell + plugin importmap entries with a deliberate
        precedence (lowest → highest):

          1. Per-plugin auto-discovered entries (``app/<stem>`` for every
             ``*.js`` under ``frontend_dir``).
          2. Shell entries (the ``<script type="importmap">`` block in
             ``index.html``).
          3. Per-plugin explicit ``importmap_entries``.

        This means a plugin that happens to ship a file named
        ``utils.js`` does NOT silently hijack the shell's ``app/utils``.
        To override a shell entry, a plugin must opt in by listing it
        explicitly in ``importmap_entries`` — accidental filename
        collisions stay safe. Within (3), later-registered plugins win
        over earlier ones, mirroring the historical behavior.
        """
        merged: Dict[str, str] = {}
        for plugin in _all_web_plugins():
            for k, v in _plugin_auto_importmap(plugin).items():
                merged[k] = v.replace("CACHE_BUST", token)
        for k, v in _shell_importmap().items():
            merged[k] = v.replace("CACHE_BUST", token)
        for plugin in _all_web_plugins():
            for k, v in plugin.importmap_entries.items():
                merged[k] = v.replace("CACHE_BUST", token)
        return merged

    def _plugin_stylesheet_links(token: str) -> str:
        """Render ``<link>`` tags for every plugin stylesheet, in
        registration order. Returns the concatenated HTML (may be empty).

        Filename validation: each entry must be a bare filename (no path
        separators, no ``..``) ending in ``.css``. This is a defense-
        in-depth on top of the later ``relative_to`` check — it rejects
        backslash forms and odd inputs early, so a buggy or malicious
        plugin can't get any variety of escape into the rendered HTML.
        """
        import re as _re
        # One filename, ``.css`` suffix, no separators of any kind, no
        # leading dot. Allows ``foo.css`` / ``foo-bar.css`` / ``foo_1.css``.
        _VALID = _re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*\.css$")
        parts: List[str] = []
        for plugin in _all_web_plugins():
            if not plugin.extra_stylesheets or not plugin.frontend_dir:
                continue
            for fname in plugin.extra_stylesheets:
                if not isinstance(fname, str) or not _VALID.match(fname):
                    continue  # reject malformed entry; never inject raw
                target = plugin.frontend_dir / fname
                if not target.is_file():
                    # Form is valid but the file doesn't exist (typo in
                    # plugin config, or a stale entry). Skip rather than
                    # emit a link that 404s.
                    continue
                try:
                    rel = target.resolve().relative_to(
                        plugin.frontend_dir.resolve()
                    )
                except (ValueError, OSError):
                    continue  # symlink/resolution escape; skip
                url = (
                    f"/static/plugins/{plugin.type}/{rel.as_posix()}?v={token}"
                )
                parts.append(
                    f'  <link rel="stylesheet" href="{url}" />'
                )
        return "\n".join(parts)

    @app.get("/")
    def index() -> HTMLResponse:
        html_text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        token = _cache_bust_token()
        merged = _merged_importmap(token)
        # Re-serialize the importmap block with the merged entries. This
        # avoids the duplicate-key problem (browsers' importmap spec
        # rejects duplicate top-level keys) and gives plugins a clean
        # override path.
        import re
        new_block = json.dumps({"imports": merged}, indent=2)
        # Preserve the 6-space indent used inside <script> in index.html
        # so the diff stays small. The regex matches the whole script
        # block regardless of original indent.
        indented = "\n".join("      " + line for line in new_block.splitlines())
        html_text = re.sub(
            r'(<script[^>]*type="importmap"[^>]*>)(.*?)(</script>)',
            lambda m: m.group(1) + "\n" + indented + "\n  " + m.group(3),
            html_text, count=1, flags=re.DOTALL,
        )
        # Inject plugin stylesheets right after the shell stylesheet.
        # The shell index.html has exactly one <link rel="stylesheet" href="/static/styles.css?v=CACHE_BUST">.
        plugin_links = _plugin_stylesheet_links(token)
        if plugin_links:
            html_text = html_text.replace(
                '<link rel="stylesheet" href="/static/styles.css?v=CACHE_BUST" />',
                '<link rel="stylesheet" href="/static/styles.css?v=CACHE_BUST" />\n'
                + plugin_links,
                1,
            )
        # Substitute CACHE_BUST anywhere else in the doc (e.g. <link>
        # stylesheet, the bootstrap main.js script tag).
        html_text = html_text.replace("CACHE_BUST", token)
        return HTMLResponse(
            content=html_text,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    # Each plugin with bundled frontend assets gets its own mount point
    # at /static/plugins/<type>/. The plugin's importmap_entries point
    # at URLs under this mount. Mount these BEFORE the generic /static
    # catch-all so the more-specific path wins.
    for _plugin in _all_web_plugins():
        if _plugin.frontend_dir and _plugin.frontend_dir.exists():
            app.mount(
                f"/static/plugins/{_plugin.type}",
                StaticFiles(directory=str(_plugin.frontend_dir)),
                name=f"static_plugin_{_plugin.type}",
            )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Force no-cache on JS / CSS module responses so the browser always
    # revalidates. ES modules are aggressively cached by the browser; an
    # old state-graph.js with the VNode-concat bug will keep loading
    # from cache even after a Ctrl-Shift-R unless we explicitly send
    # no-cache. Revalidation is cheap (304s when nothing changed).
    @app.middleware("http")
    async def _no_cache_modules(request: Request, call_next):
        resp = await call_next(request)
        path = request.url.path
        if path.startswith("/static/") and (path.endswith(".js") or path.endswith(".css")):
            resp.headers["Cache-Control"] = "no-cache, must-revalidate"
        # All /api/ endpoints read live state from disk (run.json,
        # agents.json, iterations/, ...). Without no-store the browser
        # applies heuristic caching and the Live sub-agents / Last
        # output columns go stale even though the orchestrator is
        # actively rewriting agents.json. The cost of no-store is one
        # small file read per poll — negligible.
        elif path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # On clean shutdown, clear the WebUI entry in runtime.json so the
    # next boot doesn't think the previous session is still running.
    # (If the WebUI crashes, reconcile() overwrites the stale entry on
    # next start anyway — this is just hygiene.)
    @app.on_event("shutdown")
    async def _on_shutdown():
        try:
            from . import runtime as _runtime
            _runtime.record_webui_exit()
        except Exception:  # noqa: BLE001 — never fail shutdown
            pass

    return app


# Module-level app instance for uvicorn / `metainfer-web` entry point.
app = create_app()


def main() -> int:
    """Entry point for the `metainfer-web` console script."""
    import uvicorn
    import os
    host = os.environ.get("METAINFER_HOST", "127.0.0.1")
    port = int(os.environ.get("METAINFER_PORT", "8765"))
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
