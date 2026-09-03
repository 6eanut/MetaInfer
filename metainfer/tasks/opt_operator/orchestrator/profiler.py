"""Perf profiler — per-shape latency via a hipprof-style runner, GPU-pool dispatched.

A :class:`PerfResult` records one case's measured latency. Profiling is a
validation task that goes through the idle-dispatch GPU pool: acquire a slot,
profile on it, release. The actual hipprof invocation is injectable so tests mock
the timing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Optional

from .build import BuildResult
from .contract import CaseSpec, OperatorContract
from .gpu_pool import GpuLease


class ProfilerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PerfResult:
    case_id: str
    latency_ns: float
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# timer(build, contract, case, lease, reps) -> latency_ns
Timer = Callable[[BuildResult, OperatorContract, CaseSpec, GpuLease, int], float]


def default_timer(build: BuildResult, contract: OperatorContract, case: CaseSpec,
                  lease: GpuLease, reps: int) -> float:
    """Time a Triton kernel over one case via CUDA events (median of ``reps``).

    Inputs are regenerated deterministically per case, the kernel is warmed up to
    force JIT, then launch latency is recorded per rep with CUDA events. Returns
    median latency in nanoseconds. Requires torch/triton on a K100/ROCm runtime.
    """
    import statistics  # noqa: PLC0415
    try:
        import torch  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover — env-specific
        raise ProfilerError("torch/ROCm required to time a kernel") from exc

    from .kernel_adapter import triton_runner
    from .oracle import NumpyReferenceExecutor

    inputs = NumpyReferenceExecutor().generate_inputs(contract, case.dims)
    try:
        for _ in range(2):
            triton_runner(build, contract, case, inputs)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001 — surface as a ProfilerError
        raise ProfilerError(f"kernel failed to run while timing: {exc}") from exc

    latencies_ns: list = []
    for _ in range(max(1, reps)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        triton_runner(build, contract, case, inputs)
        end.record()
        torch.cuda.synchronize()
        latencies_ns.append(start.elapsed_time(end) * 1e6)  # ms -> ns
    return float(statistics.median(latencies_ns))


def profile_case(
    pool,
    contract: OperatorContract,
    case: CaseSpec,
    build: BuildResult,
    *,
    job_id: str,
    lease_timeout_s: float = 300.0,
    reps: int = 10,
    timer: Optional[Timer] = None,
) -> PerfResult:
    """Acquire an idle GPU, profile one case on it, and release (orchestrator-owned)."""
    timer_fn = timer or default_timer
    with pool.acquire_one(job_id, lease_timeout_s) as lease:
        latency = timer_fn(build, contract, case, lease, reps)
    return PerfResult(case_id=case.id, latency_ns=float(latency))


def profile_cases(
    pool,
    contract: OperatorContract,
    build: BuildResult,
    *,
    job_id: str,
    cases=None,
    lease_timeout_s: float = 300.0,
    reps: int = 10,
    timer: Optional[Timer] = None,
) -> Dict[str, PerfResult]:
    """Profile a set of cases (default: the contract's full case matrix)."""
    case_list = cases if cases is not None else contract.generate_cases()
    return {
        case.id: profile_case(pool, contract, case, build, job_id=job_id,
                              lease_timeout_s=lease_timeout_s, reps=reps, timer=timer)
        for case in case_list
    }


__all__ = ["ProfilerError", "PerfResult", "profile_case", "profile_cases",
           "default_timer", "Timer"]
