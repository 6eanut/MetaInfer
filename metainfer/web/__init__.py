"""WebUI backend package.

This package is the long-lived main process. It serves:
  - the static SPA frontend
  - JSON API endpoints that read observable state from each task's
    ``<state_dir>/`` (no in-memory coupling to any orchestrator)
  - an SSE stream for live updates
  - per-task orchestrator subprocess management via :mod:`metainfer.web.launcher`

Orchestrators live in :mod:`metainfer.orchestrator` (shared infrastructure)
and the self-contained task packages under :mod:`metainfer.tasks`. One
orchestrator subprocess is spawned per task, dispatched by ``task_type``
via the plugin registries (``metainfer.orchestrator.tasks`` and
``metainfer.web.registry``).
"""
