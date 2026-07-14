"""CLI for the fusedmoe-evolve orchestrator subprocess.

The orchestrator runs as a child of the WebUI server. One orchestrator per
task, spawned when the user submits a new task via the web form.

Direct CLI usage (for debugging without the WebUI):

    python -m metainfer.orchestrator.tasks.fusedmoe_evolve.cli run requirements.json
    python -m metainfer.orchestrator.tasks.fusedmoe_evolve.cli run requirements.json --state-dir /path/to/task
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_CLAUDE_BIN = "ccb"
DEFAULT_PERMISSION_MODE = "bypassPermissions"
_VALID_PERMISSION_MODES = ("default", "acceptEdits", "plan", "bypassPermissions", "auto")
DEFAULT_EFFORT = "max"
_VALID_EFFORTS = ("low", "medium", "high", "max")


def _resolve_claude_bin(cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return os.environ.get("METAINFER_CLAUDE_BIN", DEFAULT_CLAUDE_BIN)


def _resolve_permission_mode(cli_value: str | None) -> str:
    if cli_value:
        v = cli_value
    else:
        v = os.environ.get("METAINFER_PERMISSION_MODE", DEFAULT_PERMISSION_MODE)
    if v not in _VALID_PERMISSION_MODES:
        raise SystemExit(
            f"invalid permission mode {v!r}; expected one of "
            f"{', '.join(_VALID_PERMISSION_MODES)}"
        )
    return v


def _resolve_effort(cli_value: str | None) -> str:
    if cli_value:
        v = cli_value
    else:
        v = os.environ.get("METAINFER_EFFORT", DEFAULT_EFFORT)
    if v not in _VALID_EFFORTS:
        raise SystemExit(
            f"invalid effort {v!r}; expected one of {', '.join(_VALID_EFFORTS)}"
        )
    return v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="metainfer-fusedmoe-evolve",
        description="MetaInfer FusedMoE Evolve orchestrator — 4-phase openevolve-driven kernel optimization.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the orchestrator on a requirements.json")
    run_p.add_argument("requirements", type=Path, help="Path to requirements.json")
    run_p.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=(
            "Where to put all task artifacts (state, code, logs). Default: "
            "<cwd>/.metainfer/tasks/<task_id>/."
        ),
    )
    run_p.add_argument(
        "--claude-bin",
        default=None,
        help=(
            "Claude Code binary to shell out to for sub-agents "
            f"(default: env METAINFER_CLAUDE_BIN or {DEFAULT_CLAUDE_BIN!r})"
        ),
    )
    run_p.add_argument(
        "--permission-mode",
        default=None,
        choices=_VALID_PERMISSION_MODES,
        help=(
            "Claude Code permission mode for sub-agents (default: env "
            f"METAINFER_PERMISSION_MODE or {DEFAULT_PERMISSION_MODE!r})."
        ),
    )
    run_p.add_argument("--model", default=None, help="Override model for sub-agents")
    run_p.add_argument(
        "--effort",
        default=None,
        choices=_VALID_EFFORTS,
        help=(
            "Claude Code effort level, controls extended-thinking budget per "
            f"turn (default: env METAINFER_EFFORT or {DEFAULT_EFFORT!r})."
        ),
    )
    run_p.add_argument("--max-iterations", type=int, default=None,
                       help="Override max iterations")
    run_p.add_argument("--extra-claude-arg", action="append", default=[],
                       help="Extra arg(s) forwarded to claude -p")
    run_p.add_argument("--openevolve-path", type=Path, default=None,
                       help="Path to openevolve installation directory")
    run_p.add_argument("--openevolve-iterations", type=int, default=None,
                       help="OpenEvolve internal iterations per MetaInfer iteration")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from .orchestrator import run_with_requirements
        return run_with_requirements(
            requirements_path=args.requirements,
            state_dir=args.state_dir,
            claude_bin=_resolve_claude_bin(args.claude_bin),
            permission_mode=_resolve_permission_mode(args.permission_mode),
            model=args.model,
            max_iterations=args.max_iterations,
            extra_claude_args=args.extra_claude_arg,
            effort=_resolve_effort(args.effort),
            openevolve_path=args.openevolve_path,
            openevolve_iterations=args.openevolve_iterations,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
