"""Bootstrap + runner for the port-model orchestrator.

Mirrors :mod:`metainfer.tasks.calc_value.orchestrator.orchestrator`:
uses the shared :mod:`metainfer.orchestrator._bootstrap` helpers (PID
file stamping, signal handlers, SubAgentManager factory), wires the
per-task :class:`TokenBudget`, and dispatches into the
:class:`Pipeline` state machine.

State-directory layout::

      <state_dir>/
      ├── requirements.json
      ├── orchestrator.pid
      ├── orchestrator.log
      ├── run.json
      ├── timeline.jsonl
      ├── agents.json
      ├── token_budget.json
      ├── iterations/<NNN>.json
      └── logs/<phase>/<...>/

Workspace layout::

      <workspace_dir>/
      ├── p1/, p2/, p3/, p4/, p5/, p6/   ← per-phase agent workdirs
      ├── memory/                         ← canonical consolidated artifacts
      ├── dumps/                          ← P5 hidden_state dumps (golden)
      └── .git/                           ← per-phase audit commits land here
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.orchestrator import paths as _orch_paths
from metainfer.orchestrator._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from metainfer.orchestrator.requirements import req_field
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.token_budget import (
    TokenBudget,
    resolve_budget_limits,
)

from . import phases as _phases
from .pipeline import Pipeline, PipelineConfig


# --------------------------------------------------------------------------- #
# State directory layout
# --------------------------------------------------------------------------- #

def _task_subdirs(state_dir: Path, workspace_dir: Path) -> Dict[str, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {
        "state_dir": state_dir,
        "workspace_dir": workspace_dir,
        "requirements": state_dir / "requirements.json",
        "pid_file": state_dir / "orchestrator.pid",
        "log_file": state_dir / "orchestrator.log",
        "run_file": state_dir / "run.json",
        "timeline_file": state_dir / "timeline.jsonl",
        "agents_file": state_dir / "agents.json",
        # Workspace subdirs:
        "memory_dir": workspace_dir / "memory",
        "dumps_dir": workspace_dir / "dumps",
        "p1_dir": workspace_dir / "p1",
        "p2_dir": workspace_dir / "p2",
        "p3_dir": workspace_dir / "p3",
        "p4_dir": workspace_dir / "p4",
        "p5_dir": workspace_dir / "p5",
        "p6_dir": workspace_dir / "p6",
    }
    for k in ("memory_dir", "dumps_dir", "p1_dir", "p2_dir", "p3_dir",
              "p4_dir", "p5_dir", "p6_dir"):
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def _validate_inputs(req: Dict[str, Any]) -> Optional[str]:
    """Sanity-check user inputs BEFORE booting any agents."""
    model_path = req_field(req, "model_params_path")
    target_fw = req_field(req, "target_framework_dir")
    if not model_path:
        return "model_params_path is required"
    if not target_fw:
        return "target_framework_dir is required"
    mp = Path(model_path)
    fp = Path(target_fw)
    if not mp.exists():
        return f"model_params_path does not exist: {mp}"
    if not fp.exists():
        return f"target_framework_dir does not exist: {fp}"
    # config.json is required if model_path is a dir; OK if it's a single file.
    if mp.is_dir() and not (mp / "config.json").exists():
        return f"model_params_path directory must contain config.json: {mp}"

    # Validate reference_sources is a list of dicts with `path` keys that exist.
    refs = req_field(req, "reference_sources") or []
    if not isinstance(refs, list):
        return "reference_sources must be a list of {path, notes} objects"
    for i, r in enumerate(refs):
        if not isinstance(r, dict):
            return f"reference_sources[{i}] must be an object with a 'path' key"
        p = r.get("path")
        if not p:
            return f"reference_sources[{i}].path is required"
        if not Path(p).exists():
            return f"reference_sources[{i}].path does not exist: {p}"
    return None


def _normalize_reference_sources(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Always return a list, even if the user passed JSON-in-string."""
    refs = req_field(req, "reference_sources")
    if refs is None:
        return []
    if isinstance(refs, list):
        return refs
    if isinstance(refs, str):
        s = refs.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return parsed
        except ValueError:
            pass
        # Fall back to line-per-path with optional `# notes`.
        out = []
        for line in s.splitlines():
            line = line.strip()
            if not line:
                continue
            if "#" in line:
                path, _, notes = line.partition("#")
                out.append({"path": path.strip(), "notes": notes.strip()})
            else:
                out.append({"path": line, "notes": ""})
        return out
    return []


def _build_budget(state_dir: Path, req: Dict[str, Any]) -> Optional[TokenBudget]:
    soft, hard = resolve_budget_limits(state_dir, req)
    if soft is None and hard is None:
        return None
    return TokenBudget(
        state_dir,
        max_cost_usd=soft,
        max_cost_usd_hard=hard,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    extra_claude_args: Optional[list] = None,
    effort: str = "max",
) -> int:
    """port-model orchestrator entry point."""
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(requirements_path.read_text(encoding="utf-8"))
    task_id = req.get("task_id", "task")

    set_process_name("metainfer-pm-orch")

    if state_dir is None or workspace_dir is None:
        from metainfer.server import paths as _web_paths
        if state_dir is None:
            state_dir = _web_paths.task_dir(task_id)
        if workspace_dir is None:
            workspace_dir = _web_paths.workspace_dir(task_id)
    paths = _task_subdirs(state_dir, workspace_dir)

    # Copy requirements.json into state_dir if invoked from elsewhere.
    target_req = paths["requirements"]
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    write_pid_file(paths["pid_file"], task_id)

    # Hard input validation.
    err = _validate_inputs(req)
    if err:
        print(f"[port-model] FATAL: {err}", flush=True)
        store = StateStore(state_dir)
        store.init_or_resume(task_id)
        store.update_run(
            current_phase=_phases.FINISHED if hasattr(_phases, "FINISHED") else "finished",
            finished=True, final_status="stopped",
            last_transition_label=f"input validation: {err}",
        )
        store.append_timeline("port_model.start.stopped", {"error": err})
        clear_pid_file(paths["pid_file"])
        return 2

    # Normalize reference_sources to a real list.
    refs = _normalize_reference_sources(req)
    req["reference_sources"] = refs

    repo_root = _orch_paths.repo_root()
    model_path = Path(req_field(req, "model_params_path") or "").resolve()
    target_fw = Path(req_field(req, "target_framework_dir") or "").resolve()
    extra_add_dirs: List[Path] = [repo_root, workspace_dir, model_path, target_fw]
    for r in refs:
        rp = Path(r.get("path") or "")
        if rp:
            extra_add_dirs.append(rp.resolve())

    store = StateStore(state_dir)
    _, is_resume = store.init_or_resume(task_id)
    if not is_resume:
        store.update_run(current_phase="idle")
    store.append_timeline(
        "port_model.start",
        {
            "task_id": task_id,
            "model_params_path": str(model_path),
            "target_framework_dir": str(target_fw),
            "reference_count": len(refs),
            "resume": is_resume,
        },
    )

    budget = _build_budget(state_dir, req)
    if budget is not None:
        store_for_cb = store
        budget._on_recorded = lambda rec, snap: store_for_cb.append_timeline(
            "token_usage",
            {
                "agent": rec.agent,
                "source": rec.source,
                "phase": rec.phase,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "cache_read_input_tokens": rec.cache_read_input_tokens,
                "cost_usd": rec.total_cost_usd,
                "running_total_cost_usd": snap.total_cost_usd,
                "running_total_input_tokens": snap.total_input_tokens,
                "running_total_output_tokens": snap.total_output_tokens,
                "agent_count": snap.agent_count,
                "exhausted": snap.exhausted,
            },
        )

    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=extra_add_dirs,
        snapshot_file=paths["agents_file"],
        max_concurrent=4,
        budget=budget,
    )
    if budget is not None and budget.max_cost_usd_hard is not None:
        budget._on_hard = lambda: manager.shutdown()

    cfg = PipelineConfig(
        state_dir=state_dir,
        workspace_dir=workspace_dir,
        p1_dir=paths["p1_dir"],
        p2_dir=paths["p2_dir"],
        p3_dir=paths["p3_dir"],
        p4_dir=paths["p4_dir"],
        p5_dir=paths["p5_dir"],
        p6_dir=paths["p6_dir"],
        memory_dir=paths["memory_dir"],
        dumps_dir=paths["dumps_dir"],
        target_fw_dir=target_fw,
        model_params_path=model_path,
        reference_sources=refs,
        user_notes=req_field(req, "user_notes") or "",
    )

    print(f"[port-model] task_id         = {task_id}")
    print(f"[port-model] state dir       = {state_dir}")
    print(f"[port-model] workspace dir   = {workspace_dir}")
    print(f"[port-model] model path      = {model_path}")
    print(f"[port-model] target fw dir   = {target_fw}")
    print(f"[port-model] references      = {len(refs)}")
    print(f"[port-model] resume          = {is_resume}")
    print(f"[port-model] budget          = "
          f"${budget.snapshot().total_cost_usd:.2f}/${budget.snapshot().limit_cost_usd:.2f}"
          if budget else "[port-model] budget          = (none)")

    restore_signals = install_subagent_shutdown_handlers(
        manager, pid_file=paths["pid_file"]
    )

    exit_code = 0
    try:
        pipeline = Pipeline(
            req=req, store=store, cfg=cfg, manager=manager, budget=budget,
            extra_claude_args=extra_claude_args,
        )
        exit_code = pipeline.run()
    except Exception as exc:  # noqa: BLE001 — top-level guard
        import traceback
        print(f"[port-model] pipeline crashed: {exc}\n{traceback.format_exc()}",
              flush=True)
        store.update_run(
            current_phase="finished", finished=True, final_status="stopped",
            last_transition_label=f"crash: {type(exc).__name__}: {exc}",
        )
        store.append_timeline("port_model.crash", {"error": str(exc)})
        exit_code = 1
    finally:
        restore_signals()
        try:
            manager.shutdown()
        except Exception:  # noqa: BLE001
            pass
        clear_pid_file(paths["pid_file"])
    return exit_code
