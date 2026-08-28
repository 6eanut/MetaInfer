"""CLI for the opt_operator orchestrator subprocess.

Framework contract (required)::

    python -m metainfer.tasks.opt_operator.orchestrator.cli run \\
        <requirements.json> --state-dir … --workspace-dir …

Task-specific flags ``--model-strong`` / ``--model-cheap`` select the LLM tier for
reasoning (plan/review/analyze) vs execution (implement/repair/write). They fall
back to env ``METAINFER_MODEL_STRONG`` / ``METAINFER_MODEL_CHEAP``, then to
``--model``.
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


def _resolve_claude_bin(cli_value):
    return cli_value or os.environ.get("METAINFER_CLAUDE_BIN", DEFAULT_CLAUDE_BIN)


def _resolve_permission_mode(cli_value):
    v = cli_value or os.environ.get("METAINFER_PERMISSION_MODE", DEFAULT_PERMISSION_MODE)
    if v not in _VALID_PERMISSION_MODES:
        raise SystemExit(f"invalid permission mode {v!r}; expected one of {', '.join(_VALID_PERMISSION_MODES)}")
    return v


def _resolve_effort(cli_value):
    v = cli_value or os.environ.get("METAINFER_EFFORT", DEFAULT_EFFORT)
    if v not in _VALID_EFFORTS:
        raise SystemExit(f"invalid effort {v!r}; expected one of {', '.join(_VALID_EFFORTS)}")
    return v


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="metainfer-orchestrator",
        description="MetaInfer opt_operator orchestrator (spawned by WebUI).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run orchestrator on requirements.json")
    run_p.add_argument("requirements", type=Path)
    run_p.add_argument("--state-dir", type=Path, default=None)
    run_p.add_argument("--workspace-dir", type=Path, default=None)
    run_p.add_argument("--claude-bin", default=None)
    run_p.add_argument("--permission-mode", default=None, choices=_VALID_PERMISSION_MODES)
    run_p.add_argument("--model", default=None, help="Default model for sub-agents")
    run_p.add_argument("--model-strong", default=None, help="Strong tier (plan/review/analyze)")
    run_p.add_argument("--model-cheap", default=None, help="Cheap tier (implement/repair/write)")
    run_p.add_argument("--effort", default=None, choices=_VALID_EFFORTS)
    run_p.add_argument("--max-iterations", type=int, default=None)
    run_p.add_argument("--extra-claude-arg", action="append", default=[])

    args = parser.parse_args(argv)
    if args.cmd == "run":
        from .orchestrator import run_with_requirements
        return run_with_requirements(
            requirements_path=args.requirements,
            state_dir=args.state_dir,
            workspace_dir=args.workspace_dir,
            claude_bin=_resolve_claude_bin(args.claude_bin),
            permission_mode=_resolve_permission_mode(args.permission_mode),
            model=args.model,
            max_iterations=args.max_iterations,
            extra_claude_args=args.extra_claude_arg,
            effort=_resolve_effort(args.effort),
            model_strong=args.model_strong,
            model_cheap=args.model_cheap,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
