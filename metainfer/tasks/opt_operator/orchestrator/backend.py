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
from .harness import BenchmarkHarness, CorrectnessHarness
from .kernel_adapter import make_adapter, triton_runner
from .oracle import FrozenOracle
from .profiler import PerfResult, profile_cases


class BackendError(RuntimeError):
    pass


class RealBackend:
    def __init__(self, pool, executor, default_language: str = "triton",
                 lease_timeout_s: float = 300.0, *,
                 warmup: int = 2, statistic: str = "median",
                 reps: int = 10) -> None:
        self.pool = pool
        self.executor = executor
        self.default_language = default_language
        self.lease_timeout_s = lease_timeout_s
        self.warmup = warmup
        self.statistic = statistic
        self.reps = reps

    # -- twin-harness construction (口径 annotation / adversarial review) -- #

    def make_correctness_harness(self, contract: OperatorContract,
                                 oracle: FrozenOracle) -> CorrectnessHarness:
        return CorrectnessHarness(contract, oracle)

    def make_benchmark_harness(self, contract: OperatorContract,
                               baseline_digest: Optional[str] = None
                               ) -> BenchmarkHarness:
        return BenchmarkHarness(contract, baseline_digest=baseline_digest,
                                warmup=self.warmup, reps=self.reps,
                                statistic=self.statistic)

    def build(self, source: str, language: str, contract: OperatorContract,
              kernel_dir: Path):
        if language not in ("hip", "triton"):
            language = self.default_language
        return build_kernel(source, contract, kernel_dir,
                            BuildProfile.default_for(language))

    def conformance(self, contract: OperatorContract, oracle: FrozenOracle,
                    build, job_id: str) -> ConformanceReport:
        if build.language == "triton":
            adapter = make_adapter(build.language, triton_runner)
        else:
            raise BackendError(
                f"real backend not wired for language {build.language!r}; use triton")
        oracle_outputs: Dict[str, Dict] = {}
        candidate_outputs: Dict[str, Dict] = {}
        for case in contract.generate_cases():
            with self.pool.acquire_one(job_id, self.lease_timeout_s) as lease:  # noqa: F841 (lease pins the GPU)
                inputs = self.executor.generate_inputs(contract, case.dims)
                oracle_outputs[case.id] = self.executor.run(
                    oracle.reference_source, contract, case.dims)
                candidate_outputs[case.id] = adapter.run_case(
                    build, contract, case, inputs)
        return conformance_report(contract, oracle_outputs, candidate_outputs)

    def profile(self, contract: OperatorContract, build, job_id: str,
                reps: int) -> Dict[str, PerfResult]:
        return profile_cases(self.pool, contract, build, job_id=job_id, reps=reps)

    def oracle_outputs(self, contract: OperatorContract, oracle: FrozenOracle,
                       job_id: str) -> Dict[str, Dict]:
        """Per-case oracle outputs for the adversarial correctness self-review.

        The frozen reference runs on CPU (numpy executor) so this needs no GPU
        lease; it only yields the tensors the adversarial review then perturbs to
        prove the conformance gate is not a rubber stamp."""
        out: Dict[str, Dict] = {}
        for case in contract.generate_cases():
            out[case.id] = self.executor.run(
                oracle.reference_source, contract, case.dims)
        return out


__all__ = ["BackendError", "RealBackend"]
