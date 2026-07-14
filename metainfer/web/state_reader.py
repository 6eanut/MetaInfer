"""Read-only access to a task's on-disk observable state.

The WebUI never imports the orchestrator package (no in-memory coupling).
Instead it reads the files the orchestrator writes under each task's
``state_dir``:

    <state_dir>/
    ├── requirements.json     # frozen inputs
    ├── run.json              # RunStatus
    ├── timeline.jsonl        # append-only events
    ├── iterations/<n>.json   # per-iteration records
    └── agents.json           # SubAgentManager snapshot

All reads are defensive: missing files return None / empty defaults
rather than raising. This is what lets the WebUI render a half-spawned
task whose orchestrator hasn't written anything yet.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def read_requirements(state_dir: Path) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "requirements.json", None)


def read_run(state_dir: Path) -> Dict[str, Any]:
    """Return RunStatus dict, or a default 'idle' sentinel if missing."""
    default = {
        "task_id": None, "task_type": None, "created_at": 0,
        "current_iteration": 0, "current_phase": "idle",
        "last_update": 0, "finished": False, "final_status": None,
        "last_outcome": None, "last_transition_label": None, "notes": [],
    }
    data = _load_json(state_dir / "run.json", None)
    if data is None:
        return default
    # Merge with defaults so missing keys don't crash the frontend.
    return {**default, **data}


def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    """All iteration records, sorted by iteration number."""
    iters_dir = state_dir / "iterations"
    if not iters_dir.exists():
        return []
    out = []
    for p in sorted(iters_dir.glob("*.json")):
        data = _load_json(p, None)
        if data is not None:
            out.append(data)
    return out


def read_iteration(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "iterations" / f"{n:03d}.json", None)


def read_timeline(state_dir: Path, since: float = 0.0) -> List[Dict[str, Any]]:
    path = state_dir / "timeline.jsonl"
    if not path.exists():
        return []
    out = []
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("ts", 0) >= since:
            out.append(ev)
    return out


def read_agents(state_dir: Path) -> Dict[str, Any]:
    """SubAgentManager snapshot. Returns ``{ts: 0, agents: []}`` when
    no orchestrator has written anything yet (e.g. orchestrator hasn't
    started, or hasn't spawned any sub-agents yet)."""
    default = {"ts": 0, "agents": []}
    return _load_json(state_dir / "agents.json", default)


def read_charts(state_dir: Path) -> Dict[str, Any]:
    """Aggregate perf-per-iteration + durations for the charts panel.
    Reads iteration records and reshapes them — same logic the old
    in-process endpoint did, but purely file-based now."""
    recs = read_iterations(state_dir)
    durations = [
        {"x": r.get("iteration", 0), "y": round(r.get("duration_s", 0) or 0, 1)}
        for r in recs if r.get("duration_s")
    ]
    perf_keys: List[str] = []
    for r in recs:
        for k in (r.get("perf") or {}):
            if k not in perf_keys:
                perf_keys.append(k)
    perf_series = []
    for k in perf_keys:
        series = [
            {"x": r.get("iteration", 0), "y": (r.get("perf") or {}).get(k)}
            for r in recs if r.get("perf") and k in r["perf"]
        ]
        perf_series.append({"metric": k, "points": series})
    return {
        "durations": durations,
        "perf_series": perf_series,
        "iteration_status": [
            {
                "iteration": r.get("iteration", 0),
                "status": r.get("status", "running"),
                "goal": r.get("goal") or "",
            }
            for r in recs
        ],
    }


def read_retrospective(state_dir: Path, n: int) -> Dict[str, Any]:
    """Return the retrospective payload for iteration ``n``. Mirrors the
    shape the old in-process endpoint returned, so the frontend modal
    logic doesn't change."""
    rec = read_iteration(state_dir, n)
    if rec is None:
        return {"has_retrospective": False, "markdown": "no such iteration",
                "path": None, "this_perf": {}, "prev_perf": {}, "iteration": n}
    prev = read_iteration(state_dir, n - 1) if n > 1 else None
    prev_perf = dict(prev.get("perf") or {}) if prev else {}
    this_perf = dict(rec.get("perf") or {})
    path_str = rec.get("retrospective_path")
    markdown = ""
    has = False
    if path_str:
        p = Path(path_str)
        if p.is_file():
            try:
                markdown = p.read_text(encoding="utf-8", errors="replace")
                has = True
            except OSError:
                markdown = ""
    if not has:
        # Same status-aware placeholder logic as the old endpoint.
        status = rec.get("status")
        if status == "running":
            reason = ("This iteration is still running — no retrospective "
                      "has been produced yet.")
        elif status == "failed":
            reason = (
                "This iteration failed and the postmortem agent didn't "
                f"produce a retrospective file. Failure reason: "
                f"`{rec.get('failure_reason') or 'unknown'}`."
            )
        elif rec.get("phases") and "E_perf_test" not in rec["phases"]:
            reason = ("This iteration hasn't reached the perf-test (E) "
                      "phase yet, so no retrospective was written.")
        else:
            reason = ("The retrospective agent didn't produce a file. "
                      "Check the iteration's logs directory.")
        markdown = (
            f"# Iteration {n} — no retrospective available\n\n"
            f"{reason}\n\n"
            f"## Raw perf data\n\n"
            f"- this iteration: `{this_perf or 'no data'}`\n"
            f"- previous iteration: `{prev_perf or 'no data'}`\n"
        )
    return {
        "has_retrospective": has,
        "path": path_str,
        "markdown": markdown,
        "this_perf": this_perf,
        "prev_perf": prev_perf,
        "iteration": n,
    }


def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    """Return nodes / edges / active_edge / current for the state-graph
    panel. This is the ONE endpoint that has to import the orchestrator
    package — the transition table is static metadata, not task state.
    We import lazily so a broken/missing orchestrator package doesn't
    take down the whole WebUI.

    The phase vocabulary is task-type-specific. We resolve the right
    ``phases_module`` from the task plugin registry (each plugin
    declares its own). Unknown / unregistered task types fall back to
    gen-infer-framework's phases (the only plugin with a real multi-edge
    state graph today)."""
    run = read_run(state_dir)
    current = run.get("current_phase", "idle")
    task_type = (run.get("task_type") or "").strip()
    try:
        from ..orchestrator.tasks import get_task
        try:
            plugin = get_task(task_type)
            phases_module = plugin.phases_module
        except KeyError:
            # Unknown task_type — best-effort fallback so the UI still
            # renders *something* rather than 500'ing.
            phases_module = (
                "metainfer.tasks.gen_infer_framework.orchestrator.phases"
            )
        import importlib
        P = importlib.import_module(phases_module)
    except Exception as e:  # noqa: BLE001
        return {"error": f"orchestrator phases module unavailable: {e!r}"}
    last_outcome = run.get("last_outcome")
    last_label = run.get("last_transition_label")

    if hasattr(P, "nodes_for_graph") and hasattr(P, "edges_for_graph"):
        # Multi-edge state-graph plugin (gen-infer-framework style).
        nodes = P.nodes_for_graph()
        edges = P.edges_for_graph()
        active_edge = None
        if last_label:
            for e in edges:
                if e["to"] == current and last_label in e["label"].split(" / "):
                    active_edge = {
                        "from": e["from"], "to": e["to"], "label": last_label,
                    }
                    break
        terminal_nodes = [
            {"id": m.id, "label": m.label, "description": m.description}
            for m in P.PHASES if m.is_terminal
        ]
        outcome_legend = [
            {"id": o, "label": P.outcome_label(o)} for o in P.ALL_OUTCOMES
        ]
    else:
        # Linear pipeline plugin (calc-value style). phases.py exposes
        # STEP_ORDER + HUMAN_LABEL only — synthesize a linear graph.
        order = list(getattr(P, "STEP_ORDER", []))
        labels = getattr(P, "HUMAN_LABEL", {})
        nodes = [{"id": p, "label": labels.get(p, p)} for p in order]
        if getattr(P, "FINISHED", None):
            nodes.append({"id": P.FINISHED, "label": labels.get(P.FINISHED, "finished")})
        edges = []
        for a, b in zip(order, order[1:] + ([P.FINISHED] if hasattr(P, "FINISHED") else [])):
            edges.append({"from": a, "to": b, "label": "step"})
        active_edge = None
        for e in edges:
            if e["to"] == current:
                active_edge = e
                break
        terminal_nodes = (
            [{"id": P.FINISHED, "label": labels.get(P.FINISHED, "finished"),
              "description": "pipeline complete"}]
            if hasattr(P, "FINISHED") else []
        )
        outcome_legend = []

    return {
        "current": current,
        "nodes": nodes,
        "edges": edges,
        "active_edge": active_edge,
        "last_outcome": last_outcome,
        "terminal_nodes": terminal_nodes,
        "outcome_legend": outcome_legend,
    }


# --------------------------------------------------------------------------- #
# Write helpers (very limited)
# --------------------------------------------------------------------------- #
# The WebUI is read-only by design — but the restart flow needs to stamp
# explicit audit events into timeline.jsonl so it's visible WHY each
# orchestrator was killed + respawned. We don't touch run.json, agents.json,
# iterations/, code/, logs/ — those belong to the orchestrator. timeline.jsonl
# is append-only JSONL, safe for the WebUI to append to without coordination.

def append_timeline_event(
    state_dir: Path, event_type: str, payload: Optional[Dict[str, Any]] = None,
) -> None:
    """Append one event to ``<state_dir>/timeline.jsonl``.

    Format matches what the orchestrator's StateStore writes (see
    ``state.py:append_timeline``): ``{"ts": float, "type": str, "payload": dict}``.
    Used by the WebUI to record lifecycle events it initiated (e.g.
    ``restart_initiated``), so the timeline gives a full audit trail
    spanning both orchestrator and WebUI actions.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "type": event_type,
        "payload": payload or {},
    }
    with open(state_dir / "timeline.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def reset_state_dir(
    state_dir: Path, task_id: str, task_type: str,
) -> Dict[str, Any]:
    """Wipe everything in ``state_dir`` except ``requirements.json``.

    Removes run.json, timeline.jsonl, orchestrator.log, orchestrator.pid,
    agents.json, and all subdirectories (iterations/, code/, logs/,
    step0..4/, etc.). Then writes a fresh ``run.json`` matching the
    RunStatus defaults so the WebUI shows a clean idle state immediately,
    and stamps a single ``task_reset`` timeline event so the reset itself
    is auditable.

    Caller MUST ensure the orchestrator is not running — this function
    does not check.
    """
    import shutil
    state_dir = Path(state_dir)
    keep = {"requirements.json"}
    removed: List[str] = []
    if state_dir.exists():
        for p in state_dir.iterdir():
            if p.name in keep:
                continue
            is_dir = p.is_dir() and not p.is_symlink()
            try:
                if is_dir:
                    shutil.rmtree(p, ignore_errors=True)
                else:
                    p.unlink()
                removed.append(p.name + ("/" if is_dir else ""))
            except OSError:
                pass
    now = time.time()
    fresh_run = {
        "task_id": task_id,
        "task_type": task_type,
        "created_at": now,
        "current_iteration": 0,
        "current_phase": "idle",
        "last_update": now,
        "finished": False,
        "final_status": None,
        "last_outcome": None,
        "last_transition_label": None,
        "notes": [],
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "run.json").write_text(
        json.dumps(fresh_run, indent=2), encoding="utf-8",
    )
    append_timeline_event(state_dir, "task_reset", {
        "task_id": task_id, "reset_at": now, "removed_count": len(removed),
    })
    return {"removed": removed, "run": fresh_run}
