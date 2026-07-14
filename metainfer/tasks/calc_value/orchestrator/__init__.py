"""calc-theoretical-value task plugin: linear 4-step pipeline.

This subpackage is **self-contained**: pipeline, prompts, phases,
deterministic helpers, and the four step implementations all live here.
calc-value has no oracles — verification is purely deterministic (see
:mod:`.deterministic`), which is why the plugin doesn't have an
``oracles/`` subpackage.

The framework (:mod:`metainfer.orchestrator`) provides shared
infrastructure but does not import this package's pipeline directly —
the launcher invokes the CLI module declared in :data:`plugin.PLUGIN`
via the registry.

Pipeline shape::

    S1_analyze ──▶ S2_graph ──▶ S3_calculate ──▶ S4_visualize ──▶ done

Each step is internally self-converging:

* S1: 2 independent agents analyze code from 2 angles → memory.json
* S2: build graph.json, then per-node LLM validation, iterate to convergence
* S3: 2 independent agents write Python calc functions per node; deterministic
  verification at the canonical shape (B=1, S=512); median-fallback after
  3 rounds. The WebUI re-runs the final calc.py at arbitrary shapes on demand.
* S4: one agent generates an HTML visualization that calls back into the
  WebUI's /compute endpoint at runtime
"""

from metainfer.orchestrator.tasks import register
from .plugin import PLUGIN

register(PLUGIN)
