"""Backward-compat re-export.

The real registry now lives at :mod:`metainfer.orchestrator.tasks`.
This module re-exports the legacy names so any external code that
imports ``metainfer.orchestrator.registry`` keeps working.

New code should import from :mod:`metainfer.orchestrator.tasks` directly.
"""

from .tasks import (
    TaskPlugin as OrchestratorEntry,  # legacy alias
    all_cli_modules,
    all_tasks,
    get_task as get_orchestrator,
    register,
)

# Legacy name: ORCHESTRATORS (dict keyed by task_type).
ORCHESTRATORS = {p.task_type: p for p in all_tasks()}


__all__ = [
    "OrchestratorEntry",
    "ORCHESTRATORS",
    "all_cli_modules",
    "all_tasks",
    "get_orchestrator",
    "register",
]
