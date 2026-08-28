"""Numerics-conformance gate — candidate outputs vs the frozen oracle.

Language-agnostic: the kernel adapter produces the *same* dict of output tensors
(keyed by contract output name) whether the kernel is HIP or Triton, so this gate
compares them against the oracle with a single piece of logic.

A candidate passes a case only if **every** output tensor is within the
contract's ``abs_tol`` / ``rel_tol``. The gate returns a structured
:class:`ConformanceReport` with per-case max abs/rel error, which drives both the
C_conformance phase verdict and the WebUI drill-in table.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from ._compare import max_abs_rel_error, shape_of
from .contract import CaseSpec, OperatorContract


class ConformanceError(ValueError):
    """Candidate outputs are structurally incompatible (bad shape / missing key)."""


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    max_abs_err: float
    max_rel_err: float
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ConformanceReport:
    passed: bool
    results: List[CaseResult]
    contract_name: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "contract_name": self.contract_name,
            "results": [r.as_dict() for r in self.results],
        }

    def result(self, case_id: str) -> Optional[CaseResult]:
        for r in self.results:
            if r.case_id == case_id:
                return r
        return None


def _check_case(
    contract: OperatorContract,
    case: CaseSpec,
    oracle_out: Dict[str, Any],
    candidate_out: Dict[str, Any],
) -> CaseResult:
    """Compare one case's oracle vs candidate outputs within contract tolerances."""
    abs_tol = float(contract.numerics.get("abs_tol", 1e-3))
    rel_tol = float(contract.numerics.get("rel_tol", 1e-2))
    max_abs, max_rel, detail = 0.0, 0.0, ""
    ok = True

    for t in contract.outputs:
        name = t.name
        if name not in candidate_out:
            return CaseResult(case.id, False, math.inf, math.inf,
                              f"missing output {name!r}")
        if name not in oracle_out:
            return CaseResult(case.id, False, math.inf, math.inf,
                              f"oracle missing {name!r}")
        got_shape = shape_of(candidate_out[name])
        want_shape = t.resolved_shape(case.dims)
        if got_shape != want_shape:
            return CaseResult(case.id, False, math.inf, math.inf,
                              f"{name}: shape {got_shape} != {want_shape}")
        a, r = max_abs_rel_error(candidate_out[name], oracle_out[name])
        max_abs = max(max_abs, a)
        max_rel = max(max_rel, r)
        if a > abs_tol and r > rel_tol:
            ok = False
            detail = f"{name}: abs={a:.3g} rel={r:.3g}"

    return CaseResult(case.id, ok, max_abs, max_rel, detail)


def conformance_report(
    contract: OperatorContract,
    oracle_outputs: Dict[str, Dict[str, Any]],
    candidate_outputs: Dict[str, Dict[str, Any]],
) -> ConformanceReport:
    """Compare candidate vs oracle outputs across the full case matrix.

    ``oracle_outputs`` / ``candidate_outputs`` are keyed by ``case_id``, each
    mapping output-name -> tensor. A missing case in either is a failure.
    """
    cases = contract.generate_cases()
    results: List[CaseResult] = []
    for case in cases:
        if case.id not in oracle_outputs:
            results.append(CaseResult(case.id, False, math.inf, math.inf,
                                      "oracle missing this case"))
            continue
        if case.id not in candidate_outputs:
            results.append(CaseResult(case.id, False, math.inf, math.inf,
                                      "candidate missing this case"))
            continue
        results.append(_check_case(
            contract, case,
            oracle_outputs[case.id], candidate_outputs[case.id],
        ))
    passed = all(r.passed for r in results)
    return ConformanceReport(passed, results, contract.name)


__all__ = ["ConformanceError", "CaseResult", "ConformanceReport", "conformance_report"]
