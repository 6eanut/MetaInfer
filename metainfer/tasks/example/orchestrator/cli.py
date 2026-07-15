"""CLI entry point for the orchestrator subprocess.

The launcher spawns::

    python -m <cli_module> run <requirements.json> --state-dir … --workspace-dir …

This module parses those arguments and delegates to the orchestrator's
``run_with_requirements``.

The ``run`` subcommand + ``--state-dir`` / ``--workspace-dir`` flags are
REQUIRED by the framework contract (§6d). Additional flags (e.g.
``--iter-limit``, ``--dry-run``) are task-defined and ignored by the shell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metainfer-orchestrator")
    sub = parser.add_subparsers(dest="command")

    run_p = sub.add_parser("run")
    run_p.add_argument("requirements", type=Path,
                       help="Path to requirements.json")
    run_p.add_argument("--state-dir", type=Path, required=True)
    run_p.add_argument("--workspace-dir", type=Path, required=True)
    # Task-specific flags — shell does NOT pass these; add whatever you need.
    run_p.add_argument("--iter-limit", type=int, default=10)
    run_p.add_argument("--dry-run", action="store_true")

    args = parser.parse_args(argv)
    if args.command != "run":
        parser.print_help()
        return 1

    # In a real task, delegate to the orchestrator here:
    # from .orchestrator import run_with_requirements
    # import json
    # req = json.loads(args.requirements.read_text())
    # return run_with_requirements(
    #     req,
    #     state_dir=args.state_dir,
    #     workspace_dir=args.workspace_dir,
    #     iter_limit=args.iter_limit,
    #     dry_run=args.dry_run,
    # )

    print(f"[example] would run with req={args.requirements} "
          f"state_dir={args.state_dir} workspace_dir={args.workspace_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
