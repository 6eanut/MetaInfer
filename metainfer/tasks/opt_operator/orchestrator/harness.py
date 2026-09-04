"""Twin validation harnesses — correctness + benchmark (OPT_KERNEL_SPEC FR-2/3).

The pipeline validates every candidate through two *harnesses*, each of which
bundles the reproducible metadata an auditor needs to trust its verdict:

- :class:`CorrectnessHarness` — decides whether a candidate's outputs match the
  frozen oracle within the contract tolerances over the shape-sweep. Metadata:
  the shape set, the numerics tolerances, and the oracle's origin + digest.
- :class:`BenchmarkHarness` — decides how much faster (or slower) a candidate
  is than baseline on the *same* shape set, measured with warmup + multiple
  reps + a stable statistic. Metadata: warmup, reps, statistic, shape set, so
  the WebUI can annotate every speedup with exactly how it was measured.

Both harnesses are thin, injectable bindings: they carry metadata and hand the
actual GPU/toolchain work to a provided :class:`backend.Backend`, so they are
pure-Python and unit-testable with a fake backend. Production wiring lives on
:class:`RealBackend` (``make_correctness_harness`` / ``make_benchmark_harness``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .contract import OperatorContract
from .oracle import FrozenOracle


class HarnessError(ValueError):
    pass


def _shape_ids(contract: OperatorContract) -> List[str]:
    return [c.id for c in contract.generate_cases()]


def _numerics(contract: OperatorContract) -> Dict[str, Any]:
    return dict(contract.numerics)


# --------------------------------------------------------------------------- #
# Correctness harness
# --------------------------------------------------------------------------- #

@dataclass
class CorrectnessHarness:
    contract: OperatorContract
    oracle: FrozenOracle
    shape_ids: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.shape_ids = self.shape_ids or _shape_ids(self.contract)

    @property
    def meta(self) -> Dict[str, Any]:
        return {
            "kind": "correctness",
            "shape_ids": self.shape_ids,
            "shape_count": len(self.shape_ids),
            "numerics": _numerics(self.contract),
            "oracle_origin": self.oracle.origin,
            "oracle_digest": self.oracle.digest,
        }

    def run(self, backend, build, job_id: str):
        """Run the gate over the full case matrix and return a ConformanceReport."""
        return backend.conformance(self.contract, self.oracle, build, job_id)


# --------------------------------------------------------------------------- #
# Benchmark harness
# --------------------------------------------------------------------------- #

@dataclass
class BenchmarkHarness:
    contract: OperatorContract
    baseline_digest: Optional[str] = None     # the genesis kernel speedups compare to
    warmup: int = 2
    reps: int = 10
    statistic: str = "median"                 # "median" | "mean"
    shape_ids: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self.shape_ids = self.shape_ids or _shape_ids(self.contract)
        if self.statistic not in ("median", "mean"):
            raise HarnessError(
                f"benchmark statistic must be median|mean, got {self.statistic!r}")
        if int(self.reps) < 1:
            raise HarnessError("benchmark needs at least 1 rep")

    @property
    def meta(self) -> Dict[str, Any]:
        return {
            "kind": "benchmark",
            "shape_ids": self.shape_ids,
            "shape_count": len(self.shape_ids),
            "warmup": int(self.warmup),
            "reps": int(self.reps),
            "statistic": self.statistic,
            "baseline_digest": self.baseline_digest,
        }

    def run(self, backend, build, job_id: str) -> Dict[str, Any]:
        """Profile the candidate on the shape set and return per-shape latency."""
        return backend.profile(self.contract, build, job_id, reps=self.reps)


__all__ = ["HarnessError", "CorrectnessHarness", "BenchmarkHarness",
           "_shape_ids", "_numerics"]
