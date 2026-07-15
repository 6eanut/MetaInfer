"""The orchestrator package: shared infrastructure for task pipelines.

Top-level modules here are **shared** by every orchestrator:

* :mod:`metainfer.orchestrator.state`          — RunStatus / StateStore (file-based state)
* :mod:`metainfer.orchestrator.iteration`      — iteration workspace helpers
* :mod:`metainfer.orchestrator.subagent_manager` — ccb subprocess lifecycle
* :mod:`metainfer.orchestrator.paths`          — repo path helpers
* :mod:`metainfer.orchestrator.gpu_preflight`  — GPU availability check
* :mod:`metainfer.orchestrator.oracles`        — per-task-type oracle registry
* :mod:`metainfer.orchestrator._bootstrap`     — shared PID / signal / manager setup
* :mod:`metainfer.orchestrator.tasks`          — TaskPlugin registry (task_type → cli/phases modules)

Each task type lives in its own self-contained subpackage under
``metainfer/tasks/<task_pkg>/`` — pipeline, phases, prompts, bootstrap
entry point, web plugin, frontend assets, tests. Adding a task type does
NOT require editing any file in this package; the registry is populated
by side-effect of importing the task package.

The WebUI dispatches to orchestrators via the registry; see
``metainfer.server.launcher``.
"""

__version__ = "0.3.0"
