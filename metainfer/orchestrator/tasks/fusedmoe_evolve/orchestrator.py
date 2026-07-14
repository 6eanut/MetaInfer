"""Bootstrap + entry point for the fusedmoe-evolve orchestrator.

This is the per-task subprocess the WebUI spawns when
``requirements.json.task_type == "fusedmoe-evolve"``. It reads the
requirements, sets up the state directory, boots a SubAgentManager, and
hands control to the 4-phase A-B-C-D iteration loop in :mod:`.pipeline`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from ..._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from ...paths import notebooks_dir as _notebooks_dir
from ...paths import repo_root as _repo_root
from ...state import StateStore
from .pipeline import Orchestrator, OrchestratorConfig


def _task_subdirs(state_dir: Path) -> Dict[str, Path]:
    """Return the canonical sub-paths under ``state_dir``."""
    state_dir.mkdir(parents=True, exist_ok=True)
    code = state_dir / "code"
    logs = state_dir / "logs"
    iterations_state = state_dir / "iterations"
    for p in (code, logs, iterations_state):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "state_dir": state_dir,
        "code_root": code,
        "logs_root": logs,
        "iterations_state": iterations_state,
        "requirements": state_dir / "requirements.json",
        "pid_file": state_dir / "orchestrator.pid",
        "log_file": state_dir / "orchestrator.log",
        "run_file": state_dir / "run.json",
        "timeline_file": state_dir / "timeline.jsonl",
        "agents_file": state_dir / "agents.json",
    }


def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    max_iterations: Optional[int] = None,
    extra_claude_args: Optional[list] = None,
    effort: str = "max",
    openevolve_path: Optional[Path] = None,
    openevolve_iterations: Optional[int] = None,
) -> int:
    """Per-task orchestrator entry point.

    Reads ``requirements.json``, runs the A-B-C-D loop to completion, and exits.
    """
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(
        requirements_path.read_text(encoding="utf-8")
    )
    task_id = req.get("task_id", "task")

    set_process_name("metainfer-orch")

    if state_dir is None:
        state_dir = Path.cwd() / ".metainfer" / "tasks" / task_id
    paths = _task_subdirs(state_dir)

    # Copy requirements into state_dir
    target_req = paths["requirements"]
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(
            requirements_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    write_pid_file(paths["pid_file"], task_id)

    repo_root = _repo_root()
    notebooks_dir = _notebooks_dir()
    logs_root = paths["logs_root"]
    iterations_root = paths["code_root"]

    store = StateStore(state_dir)
    cfg = OrchestratorConfig(
        workdir=state_dir,
        repo_root=repo_root,
        notebooks_dir=notebooks_dir,
        iterations_root=iterations_root,
        logs_root=logs_root,
        state_dir=state_dir,
        max_iterations=max_iterations or _extract_max_iter(req, default=10),
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        extra_claude_args=list(extra_claude_args or []),
        openevolve_path=openevolve_path or _resolve_openevolve_path(),
        openevolve_iterations=(
            openevolve_iterations
            or _extract_openevolve_iterations(req, default=50)
        ),
    )

    manager = make_subagent_manager(
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=[notebooks_dir, repo_root, logs_root],
        snapshot_file=paths["agents_file"],
    )
    orch = Orchestrator(req=req, store=store, cfg=cfg, manager=manager)

    print(f"[metainfer] task_id        = {task_id}")
    print(f"[metainfer] state dir      = {state_dir}")
    print(f"[metainfer] code dir       = {iterations_root}")
    print(f"[metainfer] logs dir       = {logs_root}")
    print(f"[metainfer] notebooks      = {notebooks_dir}")
    print(f"[metainfer] openevolve     = {cfg.openevolve_path}")
    print(f"[metainfer] OE iterations  = {cfg.openevolve_iterations}")
    print(f"[metainfer] orchestrator starting; WebUI is in a separate process.")

    restore_signals = install_subagent_shutdown_handlers(
        manager, pid_file=paths["pid_file"]
    )

    try:
        orch.run()
    finally:
        restore_signals()
        clear_pid_file(paths["pid_file"])
    return 0


def _extract_max_iter(req: Dict[str, Any], default: int = 10) -> int:
    """Read max_iterations from requirements."""
    v = req.get("max_iterations")
    if v is None:
        v = req.get("answers", {}).get("max_iterations")
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _extract_openevolve_iterations(req: Dict[str, Any], default: int = 50) -> int:
    """Read openevolve_iterations from requirements."""
    v = req.get("openevolve_iterations")
    if v is None:
        v = req.get("answers", {}).get("openevolve_iterations")
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _resolve_openevolve_path() -> Path:
    """Resolve openevolve_path: CLI arg > env var > default."""
    env = os.environ.get("METAINFER_OPENEVOLVE_PATH")
    if env:
        return Path(env)
    return Path("/home/jiakai/0716-fusedmoe-sglang/openevolve")
