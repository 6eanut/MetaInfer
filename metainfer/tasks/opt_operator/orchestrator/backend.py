"""Production backend for the pipeline — wires build/adapter/conformance/profiler.

Implements the :class:`pipeline.Backend` seam with the real modules. Every method
is GPU/toolchain-bound (HIP compile, Triton JIT, hipprof timing) so it raises a
clear error when the environment has no K100 / ROCm; the pipeline loop itself is
exercised by tests via a fake backend instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from .build import BuildProfile, build_kernel
from .conformance import ConformanceReport, conformance_report
from .contract import OperatorContract
from .kernel_adapter import make_adapter
from .oracle import FrozenOracle
from .profiler import PerfResult, profile_cases


class BackendError(RuntimeError):
    pass


class RealBackend:
    def __init__(self, pool, executor, default_language: str = "triton") -> None:
        self.pool = pool
        self.executor = executor
        self.default_language = default_language

    def build(self, source: str, language: str, contract: OperatorContract,
              kernel_dir: Path):
        if language not in ("hip", "triton"):
            language = self.default_language
        return build_kernel(source, contract, kernel_dir,
                            BuildProfile.default_for(language))

    def conformance(self, contract: OperatorContract, oracle: FrozenOracle,
                    build, job_id: str) -> ConformanceReport:
        adapter = make_adapter(build.language, None)
        oracle_outputs: Dict[str, Dict] = {}
        candidate_outputs: Dict[str, Dict] = {}
        for case in contract.generate_cases():
            with self.pool.acquire_one(job_id) as lease:  # noqa: F841 (lease pins the GPU)
                inputs = self.executor.generate_inputs(contract, case.dims)
                oracle_outputs[case.id] = self.executor.run(
                    oracle.reference_source, contract, case.dims)
                candidate_outputs[case.id] = adapter.run_case(
                    build, contract, case, inputs)
        return conformance_report(contract, oracle_outputs, candidate_outputs)

    def profile(self, contract: OperatorContract, build, job_id: str,
                reps: int) -> Dict[str, PerfResult]:
        return profile_cases(self.pool, contract, build, job_id=job_id, reps=reps)


__all__ = ["BackendError", "RealBackend"]
