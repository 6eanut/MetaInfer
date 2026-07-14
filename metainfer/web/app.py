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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.responses import StreamingResponse

from . import forms as _forms
from . import launcher as _launcher
from . import paths as _paths
from . import reconcile as _reconcile
from . import sse as _sse
from . import state_reader as _sr
from . import tasks as _tasks


STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def _task_or_404(task_id: str):
    entry = _tasks.get_task(task_id)
    if entry is None:
        raise HTTPException(404, f"no such task: {task_id}")
    return entry


def _state_dir_for(entry) -> Path:
    return Path(entry.state_dir)


def _find_events_file(log_dir: Path) -> Optional[Path]:
    """Locate the events.jsonl produced by SubAgentManager for an agent
    whose log_dir is ``log_dir``. Returns the highest-attempt file if
    several attempts exist (the last attempt is the one that produced
    the final result). Returns None if the dir is missing or empty.
    """
    if not log_dir.is_dir():
        return None
    candidates = sorted(log_dir.glob("*.events.jsonl"))
    if not candidates:
        return None
    # Prefer the highest attempt number; fall back to the last lexically.
    def _attempt(p: Path) -> int:
        # filenames look like "<name>.attempt<N>.events.jsonl"
        for part in p.stem.split("."):
            if part.startswith("attempt"):
                try:
                    return int(part[len("attempt"):])
                except ValueError:
                    pass
        return -1
    return max(candidates, key=_attempt)


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
        # Apply YAML defaults for fields whose submitted value is empty or
        # a placeholder (the frontend sends "默认 " for unfilled text fields).
        schema = _forms.load_form_schema(task_type)
        if schema:
            for field in schema["fields"]:
                key = field["key"]
                submitted = (answers.get(key) or "").strip()
                if not submitted or submitted == "默认":
                    default = field.get("default")
                    if default is not None and str(default).strip():
                        answers[key] = default
        # Generate a unique task id.
        task_id = _tasks.gen_task_id(task_type, label)
        sd = _paths.task_dir(task_id)
        sd.mkdir(parents=True, exist_ok=True)
        # Build the requirements.json the orchestrator will read.
        meta = _forms.TASK_TYPE_META.get(task_type, {})
        requirements = {
            "task_id": task_id,
            "task_type": task_type,
            "raw_request": body.get("raw_request") or "",
            "label": label or meta.get("label", task_type),
            **answers,
        }
        # Register first (so list view picks it up even if spawn fails).
        entry = _tasks.TaskEntry(
            id=task_id, type=task_type, label=label or meta.get("label", task_id),
            state_dir=str(sd), created_at=time.time(), launcher="local",
        )
        _tasks.add_task(entry)
        # Spawn the orchestrator.
        launcher = _launcher.get_default_launcher()
        try:
            pid = launcher.start(task_id, requirements, sd, extra_args=extra_args)
        except Exception as e:  # noqa: BLE001
            # Spawn failed — keep the registration so the user can see
            # the error in the UI, but mark as not running.
            _tasks.update_task(task_id, pid=None, finished_at=time.time())
            raise HTTPException(500, detail={"error": f"spawn failed: {e!r}"})
        return {"task_id": task_id, "pid": pid, "state_dir": str(sd)}

    @app.delete("/api/tasks/{task_id}")
    def delete_task(task_id: str, purge: bool = False) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        # Kill the orchestrator if it's still running.
        launcher = _launcher.get_default_launcher()
        if launcher.status(task_id).running:
            launcher.kill(task_id, force=True)
        # Optionally remove state_dir from disk.
        removed_files = False
        if purge:
            import shutil
            sd = Path(entry.state_dir)
            if sd.exists():
                shutil.rmtree(sd, ignore_errors=True)
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
            pid = launcher.start(task_id, req, sd)
            return {
                "ok": True, "action": "restart", "pid": pid,
                "prior_status": prior_status,
            }
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
    # Calc-theoretical-value endpoints
    # ------------------------------------------------------------------ #
    # These three endpoints only apply to tasks of type
    # calc-theoretical-value. They read the artifacts produced by the
    # calc_value orchestrator's 4-step pipeline:
    #   * step2/graph.json  (graph)
    #   * step3/final/*.py  (per-node calc scripts)
    #   * step4/viz.html    (HTML visualization)
    # The /compute endpoint imports each per-node calc.py and runs it
    # deterministically — never trusts an LLM for numeric computation.

    @app.get("/api/tasks/{task_id}/calc/graph")
    def calc_graph(task_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        if entry.type != "calc-theoretical-value":
            raise HTTPException(409, "task is not a calc-theoretical-value task")
        sd = _state_dir_for(entry)
        graph_path = sd / "step2" / "graph.json"
        if not graph_path.exists():
            raise HTTPException(404, "graph.json not built yet")
        return json.loads(graph_path.read_text(encoding="utf-8"))

    @app.get("/api/tasks/{task_id}/calc/compute")
    def calc_compute(task_id: str, batch_size: int = 1, seq_len: int = 1) -> Dict[str, Any]:
        """Run every per-compound calc.py at the given shape and return
        per-instance ``{tflops, access_gb}`` keyed by compound_id, plus
        totals aggregated as ``Σ per_compound * section.repeat_count``.

        Deterministic — no LLM in the loop. ``compound_id`` is the
        section-prefixed filename stem (``<section_id>__<node_id>``),
        matching what step3 writes; ``per_compound`` values are
        per-instance (one layer), so the frontend multiplies by
        ``section.repeat_count`` for display and we do the same here for
        the totals.
        """
        entry = _task_or_404(task_id)
        if entry.type != "calc-theoretical-value":
            raise HTTPException(409, "task is not a calc-theoretical-value task")
        if batch_size <= 0 or seq_len <= 0:
            raise HTTPException(400, "batch_size and seq_len must be positive")
        sd = _state_dir_for(entry)
        final_dir = sd / "step3" / "final"
        if not final_dir.exists():
            raise HTTPException(404, "calc scripts not built yet")
        # Import the calc_value deterministic helpers lazily so the
        # WebUI doesn't pay the import cost for non-calc tasks.
        from ..orchestrator.tasks.calc_value import deterministic as _det
        per_compound: Dict[str, Dict[str, float]] = {}
        per_compound_meta: Dict[str, Dict[str, Any]] = {}
        total_tflops = 0.0
        total_gb = 0.0
        errors: Dict[str, str] = {}
        # Walk *.py scripts; resolve each script's repeat_count from its
        # sibling .meta.json (defaults to 1 if meta missing or unset).
        for script in sorted(final_dir.glob("*.py")):
            compound_id = script.stem
            meta_path = final_dir / f"{compound_id}.meta.json"
            repeat = 1
            section_id: Optional[str] = None
            section_kind: Optional[str] = None
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    repeat = int(meta.get("section_repeat_count") or 1)
                    section_id = meta.get("section_id")
                    section_kind = meta.get("section_kind")
                    per_compound_meta[compound_id] = {
                        "section_id": section_id,
                        "section_kind": section_kind,
                        "repeat_count": repeat,
                        "node_id": meta.get("node_id"),
                    }
                except (ValueError, OSError):
                    pass
            try:
                mod = _det.load_calc_module(
                    script, module_name=f"_calc_web_{compound_id}",
                )
                tflops, gb = _det.call_calc(mod, batch_size, seq_len)
                per_compound[compound_id] = {"tflops": tflops, "access_gb": gb}
                total_tflops += tflops * repeat
                total_gb += gb * repeat
            except Exception as exc:  # noqa: BLE001
                errors[compound_id] = f"{type(exc).__name__}: {exc}"
                per_compound[compound_id] = {"tflops": 0.0, "access_gb": 0.0}
        # approximate_compounds: read flags from each meta file.
        approximate_compounds: List[str] = []
        for compound_id, meta in per_compound_meta.items():
            # Re-read approximate flag (we only kept a subset above).
            mp = final_dir / f"{compound_id}.meta.json"
            try:
                full = json.loads(mp.read_text(encoding="utf-8"))
                if full.get("approximate"):
                    approximate_compounds.append(compound_id)
            except (ValueError, OSError):
                pass
        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "per_compound": per_compound,
            # Legacy alias for older frontends; same data as per_compound.
            "per_node": per_compound,
            "compound_meta": per_compound_meta,
            "totals": {
                "tflops": total_tflops,
                "access_gb": total_gb,
                "arithmetic_intensity": (
                    total_tflops / total_gb if total_gb > 0 else 0.0
                ),
            },
            "approximate_compounds": approximate_compounds,
            "approximate_nodes": approximate_compounds,
            "errors": errors,
        }

    @app.get("/api/tasks/{task_id}/calc/viz")
    def calc_viz(task_id: str) -> HTMLResponse:
        """Serve the generated HTML visualization. Inline (no iframe
        sandbox needed) since the WebUI already trusts its own output
        and the HTML is generated locally by an LLM agent."""
        entry = _task_or_404(task_id)
        if entry.type != "calc-theoretical-value":
            raise HTTPException(409, "task is not a calc-theoretical-value task")
        sd = _state_dir_for(entry)
        viz_path = sd / "step4" / "viz.html"
        if not viz_path.exists():
            raise HTTPException(404, "viz.html not built yet")
        return HTMLResponse(viz_path.read_text(encoding="utf-8"))

    @app.get("/api/tasks/{task_id}/calc/summary")
    def calc_summary(task_id: str) -> Dict[str, Any]:
        """Step-by-step pipeline progress for the calc-value task."""
        entry = _task_or_404(task_id)
        if entry.type != "calc-theoretical-value":
            raise HTTPException(409, "task is not a calc-theoretical-value task")
        sd = _state_dir_for(entry)
        out: Dict[str, Any] = {"steps": {}}
        # Step 1
        s1 = sd / "step1" / "memory.json"
        out["steps"]["s1_analyze"] = {
            "done": s1.exists(),
            "memory_path": str(s1) if s1.exists() else None,
        }
        # Step 2
        s2 = sd / "step2" / "graph.json"
        graph_node_count = 0          # template nodes (per-section)
        aggregated_node_count = 0     # × repeat_count
        section_count = 0
        sections_summary: List[Dict[str, Any]] = []
        if s2.exists():
            try:
                from ..orchestrator.tasks.calc_value import deterministic as _det
                g = _det.normalize_graph(json.loads(s2.read_text(encoding="utf-8")))
                graph_node_count = _det.section_node_count(g)
                aggregated_node_count = _det.aggregated_node_count(g)
                section_count = len(g.get("sections") or [])
                for sec in g.get("sections") or []:
                    if not isinstance(sec, dict):
                        continue
                    rc = (sec.get("repeat_count") if sec.get("kind") == "layer_template"
                          else 1)
                    sections_summary.append({
                        "id": sec.get("id"),
                        "kind": sec.get("kind"),
                        "repeat_count": rc,
                        "node_count": len(((sec.get("graph") or {}).get("nodes")) or []),
                        "edge_count": len(((sec.get("graph") or {}).get("edges")) or []),
                    })
            except ValueError:
                pass
        out["steps"]["s2_graph"] = {
            "done": s2.exists(),
            "graph_path": str(s2) if s2.exists() else None,
            "node_count": graph_node_count,
            "aggregated_node_count": aggregated_node_count,
            "section_count": section_count,
            "sections": sections_summary,
        }
        # Step 3
        final_dir = sd / "step3" / "final"
        calc_scripts = list(final_dir.glob("*.py")) if final_dir.exists() else []
        out["steps"]["s3_calculate"] = {
            "done": len(calc_scripts) > 0,
            "final_dir": str(final_dir),
            "node_count": len(calc_scripts),
        }
        # Step 4
        s4 = sd / "step4" / "viz.html"
        out["steps"]["s4_visualize"] = {
            "done": s4.exists(),
            "viz_path": str(s4) if s4.exists() else None,
        }
        return out

    @app.get("/api/tasks/{task_id}/calc/iterations")
    def calc_iterations(task_id: str) -> Dict[str, Any]:
        """Per-round, per-agent analysis results for every step.

        Surfaces each agent's individual output (including disagreements)
        so the user can audit the convergence process, not just the
        final consensus. Reads artifacts already on disk — no extra
        writes from the orchestrator side.
        """
        entry = _task_or_404(task_id)
        if entry.type != "calc-theoretical-value":
            raise HTTPException(409, "task is not a calc-theoretical-value task")
        sd = _state_dir_for(entry)
        out: Dict[str, Any] = {"s1_analyze": [], "s2_graph": [], "s3_calculate": []}

        # ---- Step 1: round_NN/agent_X/memory.json ----
        s1 = sd / "step1"
        if s1.exists():
            rounds = sorted(d for d in s1.iterdir() if d.is_dir()
                            and d.name.startswith("round_"))
            for r in rounds:
                round_idx = int(r.name.split("_")[1])
                agents = []
                for a in sorted(x for x in r.iterdir() if x.is_dir()
                                and x.name.startswith("agent_")):
                    mem_p = a / "memory.json"
                    memory = None
                    if mem_p.exists():
                        try:
                            memory = json.loads(mem_p.read_text(encoding="utf-8"))
                        except ValueError:
                            memory = None
                    resp_p = a / "response.txt"
                    response_excerpt = None
                    if resp_p.exists():
                        try:
                            response_excerpt = resp_p.read_text(encoding="utf-8")[:8000]
                        except OSError:
                            response_excerpt = None
                    err_p = a / "parse_error.txt"
                    parse_error = err_p.read_text(encoding="utf-8")[:2000] \
                        if err_p.exists() else None
                    # Locate events.jsonl + workdir so QA can target this
                    # agent. The events file lives under round_dir/logs/
                    # <name>/<name>.attempt0.events.jsonl (see
                    # subagent_manager.AgentSpec.events_file).
                    events_file = _find_events_file(r / "logs" / a.name)
                    agents.append({
                        "name": a.name,
                        "has_memory": memory is not None,
                        "memory": memory,
                        "response_excerpt": response_excerpt,
                        "parse_error": parse_error,
                        "events_file": str(events_file) if events_file else None,
                        "target_workdir": str(a),
                    })
                # Disputes for this round (re-derived from the per-agent
                # memories by running the deterministic merge again).
                # Decoupled from the merged-file existence check so the
                # panel still surfaces disagreements even mid-round.
                disputes = []
                agent_mems = [a.get("memory") for a in agents
                              if a.get("memory")]
                if len(agent_mems) >= 2:
                    try:
                        from ..orchestrator.calc_value import deterministic as _det
                        _, disp = _det.merge_memories(agent_mems)
                        disputes = disp
                    except Exception:  # noqa: BLE001
                        pass
                merged_p = s1 / f"memory.round_{round_idx:02d}.json"
                out["s1_analyze"].append({
                    "round": round_idx,
                    "agents": agents,
                    "disputes": disputes,
                    "converged": len(disputes) == 0,
                })

        # ---- Step 2: rounds/<NN>_(validate|fix|build)/ ----
        s2 = sd / "step2"
        if s2.exists():
            rroot = s2 / "rounds"
            if rroot.exists():
                # Track the graph.json snapshot per round (if present)
                # and the verdicts.json for validate rounds.
                rounds = sorted(rroot.iterdir(), key=lambda p: p.name)
                for r in rounds:
                    if not r.is_dir():
                        continue
                    label = r.name  # e.g. "00_build", "01_validate", "02_fix"
                    kind = "build" if "_build" in label else (
                        "validate" if "_validate" in label else (
                            "fix" if "_fix" in label else "other"))
                    entry_rec: Dict[str, Any] = {
                        "dir": label, "kind": kind,
                    }
                    gj = r / "graph.json"
                    if gj.exists():
                        try:
                            from ..orchestrator.tasks.calc_value import deterministic as _det
                            g = _det.normalize_graph(
                                json.loads(gj.read_text(encoding="utf-8"))
                            )
                            entry_rec["node_count"] = _det.section_node_count(g)
                            entry_rec["edge_count"] = _det.section_edge_count(g)
                            entry_rec["aggregated_node_count"] = (
                                _det.aggregated_node_count(g)
                            )
                            entry_rec["section_count"] = (
                                len(g.get("sections") or [])
                            )
                            entry_rec["sections"] = [
                                {
                                    "id": sec.get("id"),
                                    "kind": sec.get("kind"),
                                    "repeat_count": (
                                        sec.get("repeat_count")
                                        if sec.get("kind") == "layer_template"
                                        else 1
                                    ),
                                    "node_count": len(
                                        ((sec.get("graph") or {}).get("nodes")) or []
                                    ),
                                    "edge_count": len(
                                        ((sec.get("graph") or {}).get("edges")) or []
                                    ),
                                }
                                for sec in (g.get("sections") or [])
                                if isinstance(sec, dict)
                            ]
                        except ValueError:
                            entry_rec["node_count"] = None
                    vj = r / "verdicts.json"
                    if vj.exists():
                        try:
                            verdicts = json.loads(vj.read_text(encoding="utf-8"))
                            entry_rec["verdicts"] = verdicts
                            entry_rec["pass"] = sum(
                                1 for v in verdicts
                                if isinstance(v, dict) and v.get("verdict") == "pass")
                            entry_rec["reject"] = sum(
                                1 for v in verdicts
                                if isinstance(v, dict) and v.get("verdict") == "reject")
                        except ValueError:
                            entry_rec["verdicts"] = []
                    # Per-validator raw responses (one per node).
                    validators = []
                    for vdir in sorted(r.iterdir()):
                        if not vdir.is_dir() or not vdir.name.startswith("validator_"):
                            continue
                        resp_p = vdir / "response.txt"
                        v_ef = _find_events_file(r / "logs" / vdir.name)
                        validators.append({
                            "name": vdir.name,
                            "response_excerpt": (
                                resp_p.read_text(encoding="utf-8")[:1500]
                                if resp_p.exists() else None),
                            "events_file": str(v_ef) if v_ef else None,
                            "target_workdir": str(vdir),
                        })
                    if validators:
                        entry_rec["validators"] = validators
                    out["s2_graph"].append(entry_rec)

        # ---- Step 3: rounds/<node>/round_NN/writer_X/ ----
        s3 = sd / "step3" / "rounds"
        if s3.exists():
            nodes = []
            for ndir in sorted(s3.iterdir()):
                if not ndir.is_dir():
                    continue
                node_rec: Dict[str, Any] = {
                    "node_id": ndir.name, "rounds": [],
                    "compound_id": ndir.name,
                }
                # Compound id is ``<section_id>__<node_id>`` (sanitized).
                # Resolve the bare node_id + section context from the
                # sibling final/<compound>.meta.json if present.
                meta_p = sd / "step3" / "final" / f"{ndir.name}.meta.json"
                if meta_p.exists():
                    try:
                        m = json.loads(meta_p.read_text(encoding="utf-8"))
                        node_rec["node_id"] = m.get("node_id") or ndir.name
                        node_rec["section_id"] = m.get("section_id")
                        node_rec["section_kind"] = m.get("section_kind")
                        node_rec["section_repeat_count"] = (
                            m.get("section_repeat_count")
                        )
                    except (ValueError, OSError):
                        pass
                for rdir in sorted(ndir.iterdir()):
                    if not rdir.is_dir() or not rdir.name.startswith("round_"):
                        continue
                    round_idx = int(rdir.name.split("_")[1])
                    writers = []
                    for wdir in sorted(rdir.iterdir()):
                        if not wdir.is_dir() or not wdir.name.startswith("writer_"):
                            continue
                        calc_p = wdir / "calc.py"
                        resp_p = wdir / "response.txt"
                        err_p = wdir / "error.txt"
                        w_ef = _find_events_file(rdir / "logs" / wdir.name)
                        writers.append({
                            "name": wdir.name,
                            "has_script": calc_p.exists(),
                            "script_excerpt": (
                                calc_p.read_text(encoding="utf-8")[:2500]
                                if calc_p.exists() else None),
                            "response_excerpt": (
                                resp_p.read_text(encoding="utf-8")[:1500]
                                if resp_p.exists() else None),
                            "events_file": str(w_ef) if w_ef else None,
                            "target_workdir": str(wdir),
                            "error": (
                                err_p.read_text(encoding="utf-8")[:500]
                                if err_p.exists() else None),
                        })
                    rec: Dict[str, Any] = {"round": round_idx, "writers": writers}
                    comp_p = rdir / "comparison.json"
                    if comp_p.exists():
                        try:
                            comp = json.loads(comp_p.read_text(encoding="utf-8"))
                            rec["ok"] = bool(comp.get("ok"))
                            rec["mismatch_count"] = len(comp.get("mismatches") or [])
                            rec["mismatches_excerpt"] = (
                                comp.get("mismatches") or [])[:5]
                        except ValueError:
                            pass
                    med_p = rdir / "median_fallback.json"
                    if med_p.exists():
                        rec["median_fallback"] = True
                    node_rec["rounds"].append(rec)
                nodes.append(node_rec)
            out["s3_calculate"] = nodes

        return out

    # ------------------------------------------------------------------ #
    # Offline QA over agent conversation history
    # ------------------------------------------------------------------ #
    # Lets the user click an agent in the iterations panel and ask a
    # follow-up question. A fresh ccb subprocess (the "analyst") is
    # spawned with read access to the agent's events.jsonl transcript;
    # the analyst answers based on what the original agent actually
    # did. See metainfer/web/qa.py for lifecycle / storage.

    @app.post("/api/tasks/{task_id}/calc/qa/start")
    def calc_qa_start(task_id: str, body: dict) -> Dict[str, Any]:
        """Body: {events_file, target_workdir?, target_label?, question,
                  step?, round?, round_label?, agent?}.

        Returns {session_id} immediately; the analyst runs in a daemon
        thread. Poll GET /qa/<session_id> for the answer.
        """
        from . import qa as _qa
        entry = _task_or_404(task_id)
        if entry.type != "calc-theoretical-value":
            raise HTTPException(409, "task is not a calc-theoretical-value task")
        sd = _state_dir_for(entry)
        try:
            sid = _qa.start_qa_session(sd, body or {})
        except _qa.EventsFileNotFound as exc:
            raise HTTPException(404, str(exc))
        except ValueError as exc:
            raise HTTPException(400, str(exc))
        return {"session_id": sid, "task_id": task_id}

    @app.get("/api/tasks/{task_id}/calc/qa/{session_id}")
    def calc_qa_get(task_id: str, session_id: str) -> Dict[str, Any]:
        entry = _task_or_404(task_id)
        if entry.type != "calc-theoretical-value":
            raise HTTPException(409, "task is not a calc-theoretical-value task")
        sd = _state_dir_for(entry)
        from . import qa as _qa
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
        """List QA sessions, optionally filtered by target agent identity.
        Query params: ?step=&round=&agent="""
        entry = _task_or_404(task_id)
        if entry.type != "calc-theoretical-value":
            raise HTTPException(409, "task is not a calc-theoretical-value task")
        sd = _state_dir_for(entry)
        from . import qa as _qa
        sessions = _qa.list_qa_sessions(
            sd, step=step, round_=round, agent=agent,
        )
        return {"sessions": sessions}

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
        static files. Embedded into index.html as ?v=<token> on every
        JS/CSS URL so the browser fetches fresh modules after any code
        change. Cheap to compute (~1ms).
        """
        try:
            mtimes = [p.stat().st_mtime for p in STATIC_DIR.rglob("*") if p.is_file()]
            return str(int(max(mtimes))) if mtimes else "0"
        except OSError:
            return "0"

    @app.get("/")
    def index() -> HTMLResponse:
        # Replace the CACHE_BUST placeholder in index.html with the
        # current static-dir mtime token. This pins every module URL
        # with a version query string so the browser never serves a
        # stale cached JS after a code change.
        html_text = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        token = _cache_bust_token()
        html_text = html_text.replace("CACHE_BUST", token)
        return HTMLResponse(
            content=html_text,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
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
