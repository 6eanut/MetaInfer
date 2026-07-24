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
    """Return RunStatus dict, or a default 'idle' sentinel if missing.

    ``task_type`` is intentionally absent — its authoritative source is
    ``requirements.json::task_type`` (read via :func:`read_requirements`),
    not run.json. The frontend gets type from the registry entry.
    """
    default = {
        "task_id": None,
        "current_iteration": 0, "current_phase": "idle",
        "last_update": 0, "finished": False, "final_status": None,
        "last_outcome": None, "last_transition_label": None, "notes": [],
    }
    data = _load_json(state_dir / "run.json", None)
    if data is None:
        return default
    # Drop legacy fields if an old run.json still has them — single
    # source of truth is requirements.json (task_type) and registry.json
    # (created_at). Merge with defaults so missing keys don't crash the
    # frontend.
    for _legacy in ("task_type", "created_at"):
        data.pop(_legacy, None)
    return {**default, **data}


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


# --------------------------------------------------------------------------- #
# Agent output tail
# --------------------------------------------------------------------------- #
def read_agent_tail(
    state_dir: Path, agent_name: str, max_events: int = 50,
) -> Dict[str, Any]:
    """Tail of one agent's recent activity, parsed from its stream-json log.

    The WebUI exposes this so the operator can see what an agent is currently
    doing — not just "is it alive" (which ``agents.json`` already answers) but
    "is it heading in the right direction". Without this, a stuck or
    mis-directed agent looks identical to a productive one.

    Source: ``logs/<phase>/iter_<NN>/<agent>.attempt<N>.events.jsonl`` written
    by ``metainfer.orchestrator.subagent_manager`` (stream-json from ccb).
    Each line is one stream event with ``type`` in {system, user, assistant}.

    We extract the *meaningful* events — assistant text blocks and tool uses —
    from the tail, skipping system / meta / result-only lines. This is enough
    signal for the operator to spot a misdirected agent without flooding the
    browser with raw stream-json noise.

    Returns ``{agent_name, found, log_file?, attempt?, events: [...]}``.
    ``found=False`` if the agent isn't in ``agents.json`` (caller 404s).
    """
    snap = _load_json(state_dir / "agents.json", {"ts": 0, "agents": []})
    agents = snap.get("agents", []) if isinstance(snap, dict) else []
    match = next((a for a in agents if a.get("name") == agent_name), None)
    if match is None:
        return {"agent_name": agent_name, "found": False, "events": []}

    log_file = match.get("log_file") or ""
    if not log_file:
        return {
            "agent_name": agent_name, "found": True, "log_file": "",
            "attempt": match.get("attempt"), "events": [],
        }

    # Prefer the .events.jsonl sibling (structured) over the .log (raw).
    # The orchestrator writes both with the same prefix; .events.jsonl is
    # line-delimited stream-json, .log is the human-readable rendering.
    events_path = Path(str(log_file).replace(".log", ".events.jsonl"))
    if not events_path.exists():
        # Fall back to raw log tail — last N lines as text blobs.
        try:
            raw = Path(log_file).read_text(encoding="utf-8", errors="replace")
        except OSError:
            raw = ""
        lines = [ln for ln in raw.splitlines() if ln.strip()][-max_events:]
        return {
            "agent_name": agent_name, "found": True,
            "log_file": log_file, "attempt": match.get("attempt"),
            "events": [{"type": "raw", "text": ln} for ln in lines],
        }

    # Parse the JSONL, keep assistant + tool_use events from the tail.
    # Stream-json schema (Anthropic):
    #   {"type": "assistant", "message": {"content": [{type:"text",text:"..."},
    #                                                 {type:"tool_use",name:"...",input:{}}]}}
    #   {"type": "user", "message": {"content": [{type:"tool_result",content:"..."}]},
    #    "tool_use_result": {...}}
    parsed: List[Dict[str, Any]] = []
    try:
        with open(events_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                evt = _stream_event_to_summary(d)
                if evt is not None:
                    parsed.append(evt)
    except OSError:
        pass

    # Keep the last max_events meaningful events.
    parsed = parsed[-max_events:]
    return {
        "agent_name": agent_name, "found": True,
        "log_file": log_file, "attempt": match.get("attempt"),
        "events": parsed,
    }


def _stream_event_to_summary(d: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Reduce one stream-json line to a display-friendly summary, or None
    if the line carries no operator-relevant signal.

    Returned shapes:
      ``{"type": "text", "text": "..."}`` — assistant free-form text
      ``{"type": "tool_use", "name": "Bash", "input_brief": "..."}`` — tool call
      ``{"type": "tool_result", "name": "Bash", "brief": "..."}`` — tool output
    """
    etype = d.get("type")
    if etype == "assistant":
        msg = d.get("message") or {}
        content = msg.get("content") or []
        if isinstance(content, list) and content:
            last = content[-1]
            if isinstance(last, dict):
                if last.get("type") == "text":
                    return {"type": "text", "text": str(last.get("text", ""))[:500]}
                if last.get("type") == "tool_use":
                    name = str(last.get("name", "?"))
                    inp = last.get("input") or {}
                    # Brief: for Bash, the command; for Edit/Write, the path;
                    # for Read, the path. Other tools: json first 200 chars.
                    brief = _tool_input_brief(name, inp)
                    return {"type": "tool_use", "name": name, "input_brief": brief}
        return None
    if etype == "user":
        msg = d.get("message") or {}
        content = msg.get("content") or []
        if isinstance(content, list) and content:
            last = content[-1]
            if isinstance(last, dict) and last.get("type") == "tool_result":
                # tool_result content can be string or list of blocks.
                rc = last.get("content")
                if isinstance(rc, list) and rc:
                    txt = ""
                    for blk in rc:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            txt = str(blk.get("text", ""))
                            break
                else:
                    txt = str(rc or "")
                return {"type": "tool_result", "brief": txt[:300]}
        return None
    return None


def _tool_input_brief(name: str, inp: Any) -> str:
    """Pull the most identifying bit out of a tool_use input for one-glance display."""
    if not isinstance(inp, dict):
        return ""
    if name in ("Bash",):
        return str(inp.get("command", ""))[:200]
    if name in ("Read", "Write", "Edit"):
        return str(inp.get("file_path", ""))
    if name in ("Glob",):
        return str(inp.get("pattern", ""))
    if name in ("Grep",):
        return str(inp.get("pattern", ""))
    try:
        return json.dumps(inp)[:200]
    except (TypeError, ValueError):
        return ""


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
    from metainfer.server.filelock import lock_file
    timeline_path = state_dir / "timeline.jsonl"
    with lock_file(timeline_path):
        with open(timeline_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")


def reset_state_dir(
    state_dir: Path, workspace_dir: Path, task_id: str,
) -> Dict[str, Any]:
    """Wipe everything in ``state_dir`` except ``requirements.json``,
    and wipe the entire ``workspace_dir``.

    For state_dir: removes run.json, timeline.jsonl, orchestrator.log,
    orchestrator.pid, agents.json, and all subdirectories (iterations/,
    logs/, etc.). For workspace_dir: removes the whole tree (iteration
    code, step outputs) and recreates an empty dir. Then writes a fresh
    ``run.json`` matching the RunStatus defaults so the WebUI shows a
    clean idle state immediately, and stamps a single ``task_reset``
    timeline event so the reset itself is auditable.

    ``task_type`` is intentionally NOT a parameter — it lives in
    requirements.json (which is preserved across reset) and the registry
    entry (which the caller already has).

    Caller MUST ensure the orchestrator is not running — this function
    does not check.
    """
    import shutil
    state_dir = Path(state_dir)
    workspace_dir = Path(workspace_dir)
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
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir, ignore_errors=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    fresh_run = {
        "task_id": task_id,
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
    run_path = state_dir / "run.json"
    tmp = run_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(fresh_run, indent=2), encoding="utf-8")
    tmp.replace(run_path)
    append_timeline_event(state_dir, "task_reset", {
        "task_id": task_id, "reset_at": now, "removed_count": len(removed),
        "workspace_reset": True,
    })
    return {"removed": removed, "workspace_reset": True, "run": fresh_run}
