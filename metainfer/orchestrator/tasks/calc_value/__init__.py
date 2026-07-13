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

* S1: 3 independent agents analyze code from 3 angles → memory.json
* S2: build graph.json, then per-node LLM validation, iterate to convergence
* S3: 3 independent agents write Python calc functions per node; deterministic
  verification on a 7×6 cartesian product (seq_len × batch_size = 42 combos);
  median-fallback after 15 rounds
* S4: one agent generates an HTML visualization that calls back into the
  WebUI's /compute endpoint at runtime
"""

from .. import register
from .plugin import PLUGIN

register(PLUGIN)
