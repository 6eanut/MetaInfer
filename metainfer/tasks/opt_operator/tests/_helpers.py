"""Shared fixtures for opt_operator orchestrator tests (no numpy/torch)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List


RMSNORM_CONTRACT: Dict[str, Any] = {
    "name": "RMSNorm",
    "entrypoint": "rmsnorm_kernel",
    "language": "triton",
    "shape_mode": "general",
    "inputs": [
        {"name": "X", "dtype": "fp16", "shape": ["B", "S", "H"]},
        {"name": "W", "dtype": "fp16", "shape": ["H"]},
    ],
    "outputs": [
        {"name": "Y", "dtype": "fp16", "shape": ["B", "S", "H"]},
    ],
    "shapes": {"B": 1, "S": [2048, 8192], "H": [128, 512]},
    "numerics": {"abs_tol": 1e-3, "rel_tol": 1e-2},
    "constraints": "deterministic",
}


def contract_dict(**mutations: Any) -> Dict[str, Any]:
    """A copy of the RMSNorm contract with optional mutations applied."""
    data = copy.deepcopy(RMSNORM_CONTRACT)
    data.update(mutations)
    return data


def _fill(shape: List[int], *, base: float = 1.0, nan: bool = False):
    """Deterministic nested-list fill of ``shape`` (NaN leaves when ``nan``)."""
    if not shape:
        return float("nan") if nan else base
    return [_fill(shape[1:], base=base, nan=nan) for _ in range(shape[0])]


class FakeExecutor:
    """Pure-python ReferenceExecutor for tests.

    ``run`` computes outputs deterministically from the dims alone (so two calls
    agree unless ``nondeterministic``). Config flags force each review-gate check
    to fail:
      - ``nondeterministic``: outputs differ between two runs.
      - ``wrong_shape``:     output shape does not match the contract.
      - ``nonfinite``:       an output contains NaN.
      - ``crash``:           ``run`` raises on the first case.
    """

    def __init__(
        self,
        *,
        nondeterministic: bool = False,
        wrong_shape: bool = False,
        nonfinite: bool = False,
        crash: bool = False,
    ) -> None:
        self.nondeterministic = nondeterministic
        self.wrong_shape = wrong_shape
        self.nonfinite = nonfinite
        self.crash = crash
        self.call_count = 0

    def generate_inputs(self, contract, dims) -> Dict[str, Any]:
        return {t.name: _fill(t.resolved_shape(dims)) for t in contract.inputs}

    def run(self, reference_source: str, contract, dims) -> Dict[str, Any]:
        self.call_count += 1
        if self.crash and self.call_count == 1:
            raise RuntimeError("fake reference crashed")
        # In nondeterministic mode the value drifts by 1.0 per call, so two runs
        # differ far beyond the review tolerances.
        base = float(self.call_count) if self.nondeterministic else 2.0
        out: Dict[str, Any] = {}
        for t in contract.outputs:
            shape = t.resolved_shape(dims)
            if self.wrong_shape:
                shape = [max(1, shape[0] + 1)]
            out[t.name] = _fill(shape, base=base, nan=self.nonfinite)
        return out
