"""CLI entry points for MetaInfer."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


# The Claude Code binary the sub-agent manager shells out to. Resolution:
#   1. --claude-bin CLI flag (highest priority)
#   2. METAINFER_CLAUDE_BIN env var
#   3. "ccb" (sensible default; override per environment if needed)
DEFAULT_CLAUDE_BIN = "ccb"

# Claude Code permission mode for sub-agents. Sub-agents are non-interactive
# (`-p` with stdin), so `default` mode hangs on every Edit/Write prompt.
#   - `auto`              = accept ALL tool uses (Edit/Write/Bash/...) without
#                           prompting. Works under root. DEFAULT.
#   - `acceptEdits`       = auto-accept Edit/Write only; Bash still prompts.
#                           Hangs on any sub-agent that needs shell.
#   - `bypassPermissions` = same idea as `auto` but maps to
#                           `--dangerously-skip-permissions`, which Claude
#                           Code refuses to run when EUID=0. Don't use as root.
#   - `plan`              = planning mode (read-only). Not for execution agents.
#   - `default`           = prompt for everything. Hangs sub-agents.
# Resolution:
#   1. --permission-mode CLI flag
#   2. METAINFER_PERMISSION_MODE env var
#   3. "auto"
DEFAULT_PERMISSION_MODE = "auto"
_VALID_PERMISSION_MODES = ("default", "acceptEdits", "plan", "bypassPermissions", "auto")

# Claude Code effort level controls extended-thinking budget per turn.
# Choices: low / medium / high / max. Default "max" — iteration logs showed
# reviewers hitting the default ceiling mid-analysis (13k thinking tokens
# produced but truncated), so we lift the cap. Override per-invocation via
# METAINFER_EFFORT or --effort.
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
    # Fail fast: bypassPermissions as root dies immediately inside every
    # sub-agent invocation ("--dangerously-skip-permissions cannot be used
    # with root/sudo privileges"). Better to surface this at orchestrator
    # startup with an actionable hint than waste iteration 1's three
    # retries discovering it.
    if v == "bypassPermissions" and os.geteuid() == 0:
        raise SystemExit(
            "permission-mode 'bypassPermissions' is incompatible with root: "
            "Claude Code refuses --dangerously-skip-permissions when EUID=0. "
            "Re-run without --permission-mode (defaults to 'auto', which "
            "accepts all tool uses including Bash, and works as root), or "
            "explicitly pass --permission-mode auto."
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
    parser = argparse.ArgumentParser(prog="metainfer", description="MetaInfer orchestrator CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run the orchestrator on a requirements.json")
    run_p.add_argument("requirements", type=Path, help="Path to requirements.json")
    run_p.add_argument("--port", type=int, default=8765, help="WebUI port (default 8765)")
    run_p.add_argument("--no-web", action="store_true", help="Disable the WebUI server")
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
            f"METAINFER_PERMISSION_MODE or {DEFAULT_PERMISSION_MODE!r}). "
            "'auto' = accept all tool uses, works under root (RECOMMENDED); "
            "'acceptEdits' = auto-accept Edit/Write only (Bash still hangs "
            "sub-agents); 'bypassPermissions' = same as auto but refused "
            "by Claude Code when EUID=0."
        ),
    )
    run_p.add_argument("--model", default=None, help="Override model for sub-agents")
    run_p.add_argument(
        "--effort",
        default=None,
        choices=_VALID_EFFORTS,
        help=(
            "Claude Code effort level, controls extended-thinking budget per "
            f"turn (default: env METAINFER_EFFORT or {DEFAULT_EFFORT!r}). "
            "'max' = no cap on thinking; 'low' = minimal thinking, fastest. "
            "Lift the cap when sub-agents get cut off mid-analysis; lower "
            "it when they waste wall-clock on unnecessary reasoning."
        ),
    )
    run_p.add_argument("--max-iterations", type=int, default=None,
                       help="Override max iterations (defaults to requirements.max_iterations or 20)")
    run_p.add_argument("--extra-claude-arg", action="append", default=[],
                       help="Extra arg(s) forwarded to claude -p")
    run_p.add_argument(
        "--no-keepalive",
        action="store_true",
        help=(
            "Exit the orchestrator process as soon as the run finishes "
            "(default: keep the WebUI alive after completion so results "
            "can be browsed; the next `metainfer run` in the same CWD "
            "auto-takes-over and kills this one)."
        ),
    )

    web_p = sub.add_parser("web", help="Start only the WebUI (read-only) for an existing task")
    web_p.add_argument("state_dir", type=Path, help="Path to .metainfer/state/<task_id>/")
    web_p.add_argument("--iterations-root", type=Path, default=None,
                       help="Path to the iteration CODE root (default: <cwd>/<task_id>/)")
    web_p.add_argument("--port", type=int, default=8765)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        from .orchestrator import run_with_requirements
        return run_with_requirements(
            requirements_path=args.requirements,
            web_port=args.port,
            no_web=args.no_web,
            claude_bin=_resolve_claude_bin(args.claude_bin),
            permission_mode=_resolve_permission_mode(args.permission_mode),
            model=args.model,
            max_iterations=args.max_iterations,
            extra_claude_args=args.extra_claude_arg,
            keepalive=not args.no_keepalive,
            effort=_resolve_effort(args.effort),
        )
    if args.cmd == "web":
        from .state import StateStore
        from .web.server import run_server
        store = StateStore(args.state_dir)
        # state_dir = <cwd>/.metainfer/state/<task_id>
        # In the new layout, iteration CODE lives directly under <cwd>/<task_id>/.
        # We try to derive that from the state_dir's siblings; if the user
        # has a non-standard layout, --iterations-root overrides.
        iterations_root = args.iterations_root
        if iterations_root is None:
            task_id = args.state_dir.name
            metainfer_root = args.state_dir.parents[1]   # .../.metainfer
            cwd = metainfer_root.parent                  # <cwd>
            new_layout = cwd / task_id
            old_layout = metainfer_root / "iterations" / task_id
            if new_layout.is_dir():
                iterations_root = new_layout
            elif old_layout.is_dir():
                # Legacy layout (pre-refactor) — keep working for old tasks.
                iterations_root = old_layout
            else:
                # Neither exists yet; assume new layout (will be created on
                # the next orchestrator run).
                iterations_root = new_layout
        run_server(store=store, manager=None, iterations_root=iterations_root, port=args.port)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
