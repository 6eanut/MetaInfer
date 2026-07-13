"""Filesystem paths for the WebUI server.

All persistent state lives under one root. Default location is
``<cwd>/.metainfer/`` — i.e. a ``.metainfer`` subdirectory of whatever
directory the WebUI was launched from. This keeps each MetaInfer
deployment self-contained next to its notebooks/ + metainfer/ source
tree instead of polluting ``~``.

Layout::

    <cwd>/.metainfer/
    ├── registry.json       # global task list (one entry per task)
    ├── registry.lock       # flock for atomic registry updates
    ├── runtime.json        # live WebUI/orchestrator PID state
    ├── runtime.lock        # flock for runtime.json
    └── tasks/
        └── <task_id>/      # one dir per task (see orchestrator.py for layout)

Override with the ``METAINFER_HOME`` env var for tests / custom installs
(e.g. pointing multiple WebUI instances at separate roots).
"""

from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    """Root directory for all MetaInfer persistent state. Honors
    ``METAINFER_HOME`` env var; defaults to ``<cwd>/.metainfer`` (a
    ``.metainfer`` subdirectory of the current working directory at
    process start). The cwd is captured at import time so later
    ``os.chdir`` calls don't shift the root out from under us.
    """
    override = os.environ.get("METAINFER_HOME")
    if override:
        p = Path(override).expanduser()
    else:
        p = _CWD_AT_START / ".metainfer"
    p.mkdir(parents=True, exist_ok=True)
    return p


# Capture the cwd once at module import. The WebUI process doesn't
# normally chdir during its lifetime, but pinning the value means any
# incidental os.chdir (e.g. in a third-party plugin) can't silently
# relocate the state root to a different directory.
_CWD_AT_START = Path.cwd().resolve()


def tasks_root() -> Path:
    """Where each task's directory lives (one subdir per task)."""
    p = home_dir() / "tasks"
    p.mkdir(parents=True, exist_ok=True)
    return p


def task_dir(task_id: str) -> Path:
    """Path to one task's state directory."""
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
