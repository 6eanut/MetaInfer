#!/usr/bin/env python3
"""Save / load / validate the Phase 0 interactive inputs for
metainfer-find-low-hanging-kernel. The orchestrator uses this to persist
user answers across runs and to offer a "use saved config" fast-path on launch.

The saved file is `<cwd>/.metainfer-inputs.json`. It is a single file per
working directory (NOT per run) — that's the point: subsequent launches in the
same directory get the same inputs back.

Subcommands:

  save <file> --tracing-file P --model-dir D --framework-src S
       [--cli-args STR] [--env-vars STR] [--env-file PATH] [--log-file PATH]
       [--overwrite]
       Writes the inputs file. Refuses to overwrite an existing one unless
       --overwrite is given.

  show [<file>]
       Pretty-print the saved inputs.

  validate [<file>]
       Check that the required paths still exist. Returns exit 0 if all OK,
       non-zero with a per-field diagnostic otherwise.

  exists [<file>]
       Exit 0 if the file exists, exit 1 otherwise. Used by the orchestrator
       to decide whether to offer the fast-path.

The default file path is `.metainfer-inputs.json` in the current working
directory, but a different path may be passed as the first positional arg.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any


DEFAULT_NAME = ".metainfer-inputs.json"
REQUIRED = ["tracing_file", "model_dir", "framework_src"]
OPTIONAL = ["cli_args", "env_vars", "env_file", "log_file"]


def _resolve(path: str | None) -> str:
    return os.path.abspath(path or os.path.join(os.getcwd(), DEFAULT_NAME))


def _exists(path: str) -> bool:
    return os.path.isfile(path)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def cmd_save(args) -> int:
    path = _resolve(args.file)
    if _exists(path) and not args.overwrite:
        print(f"refusing to overwrite existing {path} (pass --overwrite)", file=sys.stderr)
        return 2
    inputs: dict[str, Any] = {}
    for k in REQUIRED:
        v = getattr(args, k)
        if not v:
            print(f"missing required --{k.replace('_','-')}", file=sys.stderr)
            return 2
        inputs[k] = os.path.abspath(v)
    for k in OPTIONAL:
        v = getattr(args, k)
        if v:
            inputs[k] = v if k in ("cli_args", "env_vars") else os.path.abspath(v)
    record = {
        "schema_version": 1,
        "saved_at": _now(),
        "cwd_at_save": os.getcwd(),
        "inputs": inputs,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"wrote {path}", file=sys.stderr)
    return 0


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def cmd_show(args) -> int:
    path = _resolve(args.file)
    if not _exists(path):
        print(f"no inputs file at {path}", file=sys.stderr)
        return 1
    print(json.dumps(_load(path), indent=2))
    return 0


def cmd_validate(args) -> int:
    path = _resolve(args.file)
    if not _exists(path):
        print(f"no inputs file at {path}", file=sys.stderr)
        return 1
    rec = _load(path)
    inputs = rec.get("inputs", {})
    problems = []
    for k in REQUIRED:
        v = inputs.get(k)
        if not v:
            problems.append(f"{k}: missing")
            continue
        if not os.path.exists(v):
            problems.append(f"{k}: path does not exist: {v}")
    for k in ("env_file", "log_file"):
        v = inputs.get(k)
        if v and not os.path.exists(v):
            problems.append(f"{k}: path does not exist: {v}")
    if problems:
        for p in problems:
            print(f"INVALID: {p}", file=sys.stderr)
        return 1
    print(json.dumps({"ok": True, "saved_at": rec.get("saved_at"),
                       "cwd_at_save": rec.get("cwd_at_save")}, indent=2))
    return 0


def cmd_exists(args) -> int:
    path = _resolve(args.file)
    return 0 if _exists(path) else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("save")
    s.add_argument("file", nargs="?")
    s.add_argument("--tracing-file", required=True)
    s.add_argument("--model-dir", required=True)
    s.add_argument("--framework-src", required=True)
    s.add_argument("--cli-args", default=None)
    s.add_argument("--env-vars", default=None)
    s.add_argument("--env-file", default=None)
    s.add_argument("--log-file", default=None)
    s.add_argument("--overwrite", action="store_true")
    s.set_defaults(func=cmd_save)

    sh = sub.add_parser("show")
    sh.add_argument("file", nargs="?")
    sh.set_defaults(func=cmd_show)

    v = sub.add_parser("validate")
    v.add_argument("file", nargs="?")
    v.set_defaults(func=cmd_validate)

    e = sub.add_parser("exists")
    e.add_argument("file", nargs="?")
    e.set_defaults(func=cmd_exists)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
