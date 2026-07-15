"""Task plugin registry.

Single source of truth for "which task types does this MetaInfer install
know about?" Every consumer (launcher, WebUI, process detector) reads
this registry instead of maintaining its own if/else branch over
``task_type``.

A task plugin is registered by importing its subpackage — the
subpackage's ``__init__.py`` calls :func:`register` with its
:class:`TaskPlugin` descriptor. The bottom of this file imports every
shipped plugin so registration happens at framework import time.

Third-party plugins would call :func:`register` from their own package's
init (after importing :mod:`metainfer.orchestrator.tasks`). We don't
support entry-point discovery today; plugins must be importable from
inside the metainfer package.
"""

from __future__ import annotations

from typing import Dict, List

from .base import TaskPlugin


_REGISTRY: Dict[str, TaskPlugin] = {}


def register(plugin: TaskPlugin) -> None:
    """Register a task plugin. Called from each plugin's __init__.

    Raises ``ValueError`` on duplicate task_type — a typo that aliases
    two plugins to the same key would otherwise silently shadow one.
    """
    if plugin.task_type in _REGISTRY:
        raise ValueError(
            f"duplicate task_type {plugin.task_type!r}; already registered "
            f"by {_REGISTRY[plugin.task_type].cli_module!r}"
        )
    _REGISTRY[plugin.task_type] = plugin


def get_task(task_type: str) -> TaskPlugin:
    """Look up the plugin for ``task_type``.

    Raises ``KeyError`` for unknown task types — we deliberately do NOT
    return None or fall back to a default. An unknown task_type is a bug
    (typo in requirements.json, missing plugin import, unregistered new
    task type) and should fail fast at dispatch rather than silently
    running the wrong pipeline.
    """
    if task_type not in _REGISTRY:
        raise KeyError(
            f"unknown task_type {task_type!r}; registered types: "
            f"{', '.join(sorted(_REGISTRY))}"
        )
    return _REGISTRY[task_type]


def all_tasks() -> List[TaskPlugin]:
    """Return all registered plugins (iteration order is registration
    order, not sorted — callers that need deterministic order should
    sort by ``task_type``)."""
    return list(_REGISTRY.values())


def all_cli_modules() -> List[str]:
    """Convenience for process detectors: every registered CLI module
    path. Used by :func:`metainfer.server.proc.is_orchestrator_process` to
    recognize orchestrator subprocesses regardless of task type."""
    return [p.cli_module for p in _REGISTRY.values()]


# --------------------------------------------------------------------------- #
# Task package discovery
# --------------------------------------------------------------------------- #
#
# Auto-discovery now lives in :mod:`metainfer.tasks` (which imports each
# self-contained task package under ``metainfer/tasks/<name>/``, and
# each task package's __init__. calls ``register(...)`` here). Importing
# ``metainfer.tasks`` as a side effect of web app startup / launcher
# use populates this registry.
#
# We deliberately do NOT import ``metainfer.tasks`` from here — that
# would create a circular import (task packages import this module to
# call ``register``).


__all__ = [
    "TaskPlugin",
    "register",
    "get_task",
    "all_tasks",
    "all_cli_modules",
]
