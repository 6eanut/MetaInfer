"""opt_operator orchestrator package: contract-first, self-certifying operator
optimization (HIP / Triton → K100) with an append-only champion lineage ledger.

Registers the :class:`TaskPlugin` for the ``opt-operator`` task type.
"""

from metainfer.orchestrator.tasks import register

from .plugin import PLUGIN

register(PLUGIN)

__all__ = ["PLUGIN"]
