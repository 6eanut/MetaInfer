"""The orchestrator package: per-task-type pipelines + shared infrastructure.

Top-level modules here are **shared** by every orchestrator:

* :mod:`metainfer.orchestrator.state`          — RunStatus / StateStore (file-based state)
* :mod:`metainfer.orchestrator.iteration`      — iteration workspace helpers
* :mod:`metainfer.orchestrator.subagent_manager` — ccb subprocess lifecycle
* :mod:`metainfer.orchestrator.paths`          — repo path helpers
* :mod:`metainfer.orchestrator.gpu_preflight`  — GPU availability check
* :mod:`metainfer.orchestrator.oracles`        — per-task-type oracle registry
* :mod:`metainfer.orchestrator._bootstrap`     — shared PID / signal / manager setup
* :mod:`metainfer.orchestrator.registry`       — task_type → orchestrator dispatch table

Each task type has its own subpackage with its pipeline, phases, prompts,
and bootstrap entry point:

* :mod:`metainfer.orchestrator.gen_infer_framework`
  — the 6-phase ABCDEF iteration loop (plan→implement→test→review→perf→perf_plan)

* :mod:`metainfer.orchestrator.calc_value`
  — the linear 4-step calc-theoretical-value pipeline (analyze→graph→calculate→visualize)

The WebUI dispatches to these via the registry; see
``metainfer.web.launcher`` and :func:`metainfer.orchestrator.registry.get_orchestrator`.
"""

__version__ = "0.3.0"
