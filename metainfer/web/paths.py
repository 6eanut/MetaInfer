"""Filesystem paths for the WebUI server.

Two-level directory split, with a per-node layer at the top so multiple
machines can share a filesystem without write conflicts::

    $METAINFER_ROOT/                          (defaults to <cwd>)
    └── nodes/
        └── <node_id>/                        ($METAINFER_NODE_ID or hostname)
            ├── workspaces/                   iteration-generated artifacts
            │   └── <task_id>/                one dir per task (gf: 001/, 002/...;
            │                                  calc_value: step0..4/)
            └── .metainfer/                   metadata + logs + prompts
                ├── registry.json             global task list
                ├── registry.lock             flock for atomic registry updates
                ├── runtime.json              live WebUI/orchestrator PID state
                ├── runtime.lock              flock for runtime.json
                └── tasks/
                    └── <task_id>/            per-task bookkeeping:
                        ├── requirements.json
                        ├── run.json
                        ├── timeline.jsonl
                        ├── orchestrator.{pid,log}
                        ├── agents.json
                        ├── token_budget.json
                        ├── iterations/*.json  (gf only — per-iter records)
                        └── logs/<NNN>/        (gf only — prompts/oracle/server)

Why the split: artifacts the user cares about (the generated framework code)
live in a visible, navigable ``workspaces/`` tree; orchestrator bookkeeping
and transient logs stay hidden in ``.metainfer/``. The outermost ``nodes/``
layer lets a future central controller scan ``nodes/*/`` to see global state
across a shared filesystem.

Overrides via env vars:
  - ``METAINFER_ROOT`` — top-level root (use a shared mount point in multi-node)
  - ``METAINFER_NODE_ID`` — current machine's id (defaults to hostname)
"""

from __future__ import annotations

import os
import socket
from pathlib import Path


# Capture the cwd once at module import. The WebUI process doesn't
# normally chdir during its lifetime, but pinning the value means any
# incidental os.chdir (e.g. in a third-party plugin) can't silently
# relocate the root out from under us.
_CWD_AT_START = Path.cwd().resolve()


def root_dir() -> Path:
    """Top-level root directory. Honors ``METAINFER_ROOT`` env var;
    defaults to ``<cwd>`` (captured at import time). Multi-node setups
    point this at a shared filesystem mount so every node writes under
    the same tree (each under its own ``nodes/<id>/`` subdirectory)."""
    override = os.environ.get("METAINFER_ROOT")
    p = Path(override).expanduser().resolve() if override else _CWD_AT_START
    return p


def node_id() -> str:
    """Identifier for the current machine. Honors ``METAINFER_NODE_ID``
    env var; defaults to ``socket.gethostname()``. Used as the per-node
    subdirectory name under ``<root>/nodes/`` so two machines sharing a
    filesystem never collide."""
    return os.environ.get("METAINFER_NODE_ID") or socket.gethostname()


def node_dir() -> Path:
    """This node's directory: ``<root>/nodes/<node_id>/``. Created on
    first access. Every path the WebUI and orchestrator writes on this
    machine lives somewhere under here."""
    p = root_dir() / "nodes" / node_id()
    p.mkdir(parents=True, exist_ok=True)
    return p


def home_dir() -> Path:
    """Metadata root for this node: ``<node_dir>/.metainfer/``. Holds
    registry.json + runtime.json + tasks/<id>/ (per-task bookkeeping).
    Created on first access.

    Note: ``METAINFER_HOME`` env var is no longer honored — the layout
    now derives from ``METAINFER_ROOT`` + ``METAINFER_NODE_ID``. Tests
    that used to set ``METAINFER_HOME`` should set ``METAINFER_ROOT``
    instead.
    """
    p = node_dir() / ".metainfer"
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspaces_root() -> Path:
    """Artifacts root for this node: ``<node_dir>/workspaces/``. Each
    task's generated artifacts (iteration code, step outputs) live in
    a per-task subdirectory here."""
    p = node_dir() / "workspaces"
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspace_dir(task_id: str) -> Path:
    """Path to one task's workspace directory (generated artifacts).
    Does NOT auto-create — the launcher creates it at spawn time so a
    stray read here doesn't litter the filesystem."""
    return workspaces_root() / task_id


def tasks_root() -> Path:
    """Where each task's metadata directory lives (one subdir per task)."""
    p = home_dir() / "tasks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def task_dir(task_id: str) -> Path:
    """Path to one task's metadata directory."""
    return tasks_root() / task_id


def registry_path() -> Path:
    """Global registry file (JSON list of all known tasks)."""
    return home_dir() / "registry.json"


def registry_lock_path() -> Path:
    """flock target for atomic registry read-modify-write."""
    return home_dir() / "registry.lock"


def runtime_path() -> Path:
    """Live runtime state: which WebUI instance is running + which
    orchestrator PIDs it spawned. Used for crash-recovery reconciliation
    on WebUI restart — see :mod:`metainfer.web.runtime`."""
    return home_dir() / "runtime.json"


def runtime_lock_path() -> Path:
    """flock target for atomic runtime.json updates."""
    return home_dir() / "runtime.lock"
