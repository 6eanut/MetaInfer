"""TaskPlugin contract.

A task plugin bundles everything the framework needs to know about a task
type to dispatch + render it, WITHOUT bundling the task's pipeline logic
itself. The pipeline lives in the same subpackage and is invoked through
``cli_module``; the framework never imports it directly.

Design intent
-------------
* **High cohesion**: every file specific to one task type lives in that
  task's subpackage (``tasks/<name>/``). Pipeline, prompts, phases,
  oracles, and data files are colocated.
* **Low coupling**: a task plugin depends only on the framework's shared
  interfaces (``state.StateStore``, ``subagent_manager.SubAgentManager``,
  ``oracles.base.Oracle`` ABCs). It never imports another task plugin.
  The framework never imports a task plugin's pipeline — it only reads
  metadata from the plugin descriptor.

Adding a new task type:

1. Create ``metainfer/orchestrator/tasks/<name>/`` with the standard
   layout (``plugin.py``, ``cli.py``, ``orchestrator.py``, ``pipeline.py``,
   ``phases.py``, optional ``oracles/``, optional ``prompts.py``).
2. Define a :class:`TaskPlugin` instance in ``plugin.py`` and call
   :func:`register` from ``__init__.py``.
3. Add an import line to :mod:`metainfer.orchestrator.tasks.__init__`
   so the plugin module is loaded at package import time.

That's it — launcher, process detector, WebUI state graph, and forms
all pick up the new task type via the registry.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskPlugin:
    """Static metadata describing one task type.

    Attributes
    ----------
    task_type:
        The ``task_type`` string written to ``requirements.json``. This
        is the dispatch key the launcher and WebUI use. Must be unique
        across all registered plugins.
    name:
        Human-readable display name (e.g. "Inference Framework Optimizer").
        Used in docs and possibly UI headers.
    description:
        One-line description of what this task does.
    cli_module:
        Dotted module path of the orchestrator CLI entry point. The
        launcher invokes ``python -m <cli_module> run requirements.json
        --state-dir ...``. The CLI module is responsible for parsing
        args and dispatching to its orchestrator's ``run_with_requirements``.
    phases_module:
        Dotted module path of the module that defines :data:`PHASES`,
        :data:`TRANSITIONS`, :func:`nodes_for_graph`,
        :func:`edges_for_graph`, etc. The WebUI's state-graph endpoint
        imports this lazily to render the per-task-type phase diagram.
        Set to empty string if the task has no multi-node state graph
        (e.g. a linear pipeline that doesn't need a graph view).

    Notes
    -----
    Oracles are deliberately NOT in this descriptor. Each pipeline
    imports its own oracles directly (``from .oracles.correctness import
    CorrectnessOracle``). The framework doesn't need to know which
    oracles a task has — that's an internal implementation detail of
    the task's pipeline. This keeps the descriptor minimal and avoids
    the prior asymmetry where one oracle type was registered and another
    was hardcoded.
    """

    task_type: str
    name: str
    description: str
    cli_module: str
    phases_module: str = ""


__all__ = ["TaskPlugin"]
