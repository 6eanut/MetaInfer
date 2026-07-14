"""Oracles for the fusedmoe-evolve task.

- :class:`OpenEvolveOracle`: shells out to openevolve-run.py for B_evolve.
- :class:`ValidateOracle`: runs correctness + perf tests for C_validate.
"""

from .openevolve_oracle import OpenEvolveOracle
from .validate_oracle import ValidateOracle

__all__ = ["OpenEvolveOracle", "ValidateOracle"]
