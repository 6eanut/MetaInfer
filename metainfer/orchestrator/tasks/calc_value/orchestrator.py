"""Bootstrap + linear runner for the calc-value orchestrator.

Mirrors the structure of :mod:`metainfer.orchestrator.gen_infer_framework.orchestrator`
(PID file stamping, signal handlers, SubAgentManager init — all shared via
:mod:`metainfer.orchestrator._bootstrap`) but with a linear 4-step pipeline
instead of the ABCDEF iteration loop.

Differences from the gen-infer-framework orchestrator:

* No iterations/, code/, or logs/<n>/ subdirs — we have step1/.../step4/.
* SubAgentManager's ``extra_add_dirs`` includes the user's ``model_dir``
  and ``framework_source_dir`` so every agent has read-only access to
  the model + framework source.
* State directory layout::

      <state_dir>/
      ├── requirements.json
      ├── orchestrator.pid
      ├── orchestrator.log
      ├── run.json
      ├── timeline.jsonl
      ├── agents.json
      ├── step0/{agent_rough/, per_node/, rough_graph.json, rough_results.json}
      ├── step1/{agent_a,agent_b,agent_c}/, memory.json
      ├── step2/{graph.json, rounds/<n>/node_<id>/}
      ├── step3/{rounds/<n>/, cells/, final/}
      └── step4/viz.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from ... import paths as _orch_paths
from ..._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from ...state import StateStore
from . import phases as _phases
from .pipeline import run_pipeline


# --------------------------------------------------------------------------- #
# State directory layout
# --------------------------------------------------------------------------- #

def _task_subdirs(state_dir: Path) -> Dict[str, Path]:
    """Create and return the canonical paths under ``state_dir``."""
    state_dir.mkdir(parents=True, exist_ok=True)
    step0 = state_dir / "step0"
    step1 = state_dir / "step1"
    step2 = state_dir / "step2"
    step3 = state_dir / "step3"
    step4 = state_dir / "step4"
    for p in (step0, step1, step2, step3, step4):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "state_dir": state_dir,
        "requirements": state_dir / "requirements.json",
        "pid_file": state_dir / "orchestrator.pid",
        "log_file": state_dir / "orchestrator.log",
        "run_file": state_dir / "run.json",
        "timeline_file": state_dir / "timeline.jsonl",
        "agents_file": state_dir / "agents.json",
        "step0_dir": step0,
        "step1_dir": step1,
        "step2_dir": step2,
        "step3_dir": step3,
        "step4_dir": step4,
    }


def _write_pid_file(pid_file: Path, task_id: str) -> None:
    # Thin local wrapper kept for backward-compat with call sites in this
    # file; delegates to the shared bootstrap helper.
    write_pid_file(pid_file, task_id)


def _clear_pid_file(pid_file: Path) -> None:
    clear_pid_file(pid_file)


def _validate_inputs(req: Dict[str, Any]) -> Optional[str]:
    """Sanity-check the user's inputs BEFORE booting any agents.

    Returns an error message string if inputs are unusable, else None.
    Critically, NEVER writes to or modifies the user's paths — just
    stats them.
    """
    model_dir = req.get("model_dir")
    fw_dir = req.get("framework_source_dir")
    if not model_dir:
        return "model_dir is required"
    if not fw_dir:
        return "framework_source_dir is required"
    mp = Path(model_dir)
    fp = Path(fw_dir)
    if not mp.exists():
        return f"model_dir does not exist: {mp}"
    if not fp.exists():
        return f"framework_source_dir does not exist: {fp}"
    if not (mp / "config.json").exists():
        return f"model_dir must contain config.json: {mp}"
    return None


def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    extra_claude_args: Optional[list] = None,
    effort: str = "max",
) -> int:
    """Per-task calc-value orchestrator entry point."""
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(requirements_path.read_text(encoding="utf-8"))
    task_id = req.get("task_id", "task")

    # Friendly process name (kernel comm, 15-char limit).
    set_process_name("metainfer-cv-orch")

    if state_dir is None:
        state_dir = Path.cwd() / ".metainfer" / "tasks" / task_id
    paths = _task_subdirs(state_dir)

    # Copy requirements into state_dir if invoked from elsewhere.
    target_req = paths["requirements"]
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    _write_pid_file(paths["pid_file"], task_id)

    # Hard input validation — refuse to start if inputs are unusable
    # (missing paths, missing config.json, etc). This is NOT a "Fail"
    # state: the system can't even begin exploring because the user
    # gave it nothing to work with. We mark as ``stopped`` so the WebUI
    # shows a distinct non-Fail state.
    err = _validate_inputs(req)
    if err:
        print(f"[calc-value] FATAL: {err}", flush=True)
        store = StateStore(state_dir)
        rs, _ = store.init_or_resume(task_id, "calc-theoretical-value")
        store.update_run(
            current_phase=_phases.FINISHED,
            finished=True,
            final_status="stopped",
            last_transition_label=f"input validation: {err}",
        )
        store.append_timeline("calc_value.start.stopped", {"error": err})
        _clear_pid_file(paths["pid_file"])
        return 2

    # Extra add-dirs: every agent can read the model files, the framework
    # source, the notebooks (knowledge base), and the repo root (so agents
    # can read the calc_value package itself if needed for context).
    repo_root = _orch_paths.repo_root()
    notebooks_dir = _orch_paths.notebooks_dir()
    model_dir = Path(req["model_dir"]).resolve()
    framework_dir = Path(req["framework_source_dir"]).resolve()
    extra_add_dirs = [notebooks_dir, repo_root, model_dir, framework_dir]

    store = StateStore(state_dir)
    rs, is_resume = store.init_or_resume(task_id, "calc-theoretical-value")
    if not is_resume:
        store.update_run(current_phase=_phases.IDLE)
    store.append_timeline(
        "calc_value.start",
        {"task_id": task_id, "model_dir": str(model_dir),
         "framework_source_dir": str(framework_dir),
         "resume": is_resume},
    )

    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=extra_add_dirs,
        snapshot_file=paths["agents_file"],
        max_concurrent=5,
    )

    print(f"[calc-value] task_id        = {task_id}")
    print(f"[calc-value] state dir      = {state_dir}")
    print(f"[calc-value] model dir      = {model_dir}")
    print(f"[calc-value] framework dir  = {framework_dir}")
    print(f"[calc-value] resume         = {is_resume}")
    print(f"[calc-value] starting 5-step pipeline (S0 rough → S1 → S2 → S3 → S4).")

    restore_signals = install_subagent_shutdown_handlers(
        manager, pid_file=paths["pid_file"]
    )

    exit_code = 0
    try:
        exit_code = run_pipeline(
            req=req,
            store=store,
            manager=manager,
            paths=paths,
        )
    except Exception as exc:  # noqa: BLE001 — top-level guard
        import traceback
        print(f"[calc-value] pipeline crashed: {exc}\n{traceback.format_exc()}",
              flush=True)
        # The pipeline's step-level retry loop absorbs virtually all
        # errors. If something escapes here it's a true infrastructure
        # crash (e.g. KeyboardInterrupt, OSError out of disk). Mark as
        # ``stopped`` so the run is visible as halted but NOT as Fail —
        # the user can fix the underlying issue and resume.
        store.update_run(
            current_phase=_phases.FINISHED,
            finished=True,
            final_status="stopped",
            last_transition_label=f"crash: {type(exc).__name__}: {exc}",
        )
        store.append_timeline("calc_value.crash", {"error": str(exc)})
        exit_code = 1
    finally:
        restore_signals()
        try:
            manager.shutdown()
        except Exception:  # noqa: BLE001
            pass
        clear_pid_file(paths["pid_file"])
    return exit_code
