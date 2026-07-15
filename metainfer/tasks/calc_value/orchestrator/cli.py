"""CLI for the calc-value orchestrator subprocess.

Mirrors :mod:`metainfer.orchestrator.gen_infer_framework.cli` (shared
CLI shape) but dispatches into
:mod:`metainfer.orchestrator.calc_value.orchestrator`.

Direct CLI usage::

    python -m metainfer.orchestrator.calc_value.cli run requirements.json \\
        --state-dir /path/to/task
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


DEFAULT_CLAUDE_BIN = "ccb"
DEFAULT_PERMISSION_MODE = "bypassPermissions"
_VALID_PERMISSION_MODES = ("default", "acceptEdits", "plan", "bypassPermissions", "auto")
DEFAULT_EFFORT = "max"
_VALID_EFFORTS = ("low", "medium", "high", "max")


def _resolve(arg_val, env_var, default, valid=None):
    v = arg_val or __import__("os").environ.get(env_var, default)
    if valid and v not in valid:
        raise SystemExit(f"invalid value {v!r}; expected one of {valid}")
    return v


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="metainfer-calc-value",
        description="MetaInfer calc-theoretical-value orchestrator.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the calc-value orchestrator")
    run_p.add_argument("requirements", type=Path, help="Path to requirements.json")
    run_p.add_argument("--state-dir", type=Path, default=None,
                       help="Metadata dir (run.json, timeline.jsonl, agents.json, etc.).")
    run_p.add_argument("--workspace-dir", type=Path, default=None,
                       help="Generated-artifacts dir (step0..step4 outputs).")
    run_p.add_argument("--claude-bin", default=None)
    run_p.add_argument("--permission-mode", default=None,
                       choices=_VALID_PERMISSION_MODES)
    run_p.add_argument("--model", default=None)
    run_p.add_argument("--effort", default=None, choices=_VALID_EFFORTS)
    run_p.add_argument("--extra-claude-arg", action="append", default=[])

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from .orchestrator import run_with_requirements
        return run_with_requirements(
            requirements_path=args.requirements,
            state_dir=args.state_dir,
            workspace_dir=args.workspace_dir,
            claude_bin=_resolve(args.claude_bin, "METAINFER_CLAUDE_BIN", DEFAULT_CLAUDE_BIN),
            permission_mode=_resolve(args.permission_mode, "METAINFER_PERMISSION_MODE",
                                     DEFAULT_PERMISSION_MODE, _VALID_PERMISSION_MODES),
            model=args.model,
            extra_claude_args=args.extra_claude_arg,
            effort=_resolve(args.effort, "METAINFER_EFFORT", DEFAULT_EFFORT, _VALID_EFFORTS),
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
