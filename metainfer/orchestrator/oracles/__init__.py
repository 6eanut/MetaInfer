"""Oracle framework: shared base classes + judge utility.

This package is the SHARED layer. It contains only infrastructure that
every task plugin's oracle can build on:

* :class:`Oracle` / :class:`OracleResult` / :class:`OracleCaseResult` —
  the abstract result contract that pipelines expect from any oracle.
* :mod:`.judge` — LLM-judge helpers (any correctness oracle that needs
  an LLM verdict on responses reuses these).

Concrete oracle implementations (correctness, perf, etc.) live WITH
their task plugins, not here. Each task plugin's ``oracles/``
subpackage imports from this layer:

    from ....oracles.base import Oracle, OracleResult
    from ....oracles.judge import run_judge_batch

There is intentionally NO task_type → oracle registry here. The old
``_REGISTRY`` dispatch was removed because:

1. It only ever covered the correctness dimension (perf was hardcoded
   in the pipeline), creating a misleading asymmetry.
2. The pipeline that consumes an oracle lives in the same package as
   the oracle, so direct imports are clearer than registry indirection.
3. A new task type adding an oracle shouldn't have to edit framework
   state — it just drops a module into its own ``oracles/`` subpackage.
"""

from .base import Oracle, OracleCaseResult, OracleResult
from .judge import (
    JudgeInput,
    heuristic_verdict,
    run_judge_batch,
)


__all__ = [
    "Oracle",
    "OracleResult",
    "OracleCaseResult",
    "JudgeInput",
    "run_judge_batch",
    "heuristic_verdict",
]
