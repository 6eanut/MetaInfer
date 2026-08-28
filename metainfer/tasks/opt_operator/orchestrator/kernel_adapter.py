"""Unified kernel adapter — HIP / Triton -> identical output tensors.

The plan's "same comparison logic for both languages" hinges on this adapter: a
built kernel (HIP or Triton) is launched over a concrete case + inputs, and the
adapter returns a **dict of output tensors keyed by contract output name** — the
exact same shape produced regardless of language. From there the conformance gate
and oracle compare candidates with one shared implementation.

Real launching (running a compiled HIP driver / a Triton ``@jit`` module) is
environment-bound and injected as a ``runner`` callable, so the adapter is
unit-testable with a pure-Python fake.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Protocol

from .build import BuildResult, LANGUAGES
from .contract import CaseSpec, OperatorContract


class AdapterError(ValueError):
    pass


# runner(build, contract, case, inputs) -> dict[output_name, tensor]
Runner = Callable[[BuildResult, OperatorContract, CaseSpec, Dict[str, Any]],
                  Dict[str, Any]]


@dataclass(frozen=True)
class KernelAdapter:
    """Bridges one language's launch path into language-agnostic output tensors."""

    language: str
    runner: Runner

    def __post_init__(self) -> None:
        if self.language not in LANGUAGES:
            raise AdapterError(f"language must be one of {LANGUAGES}, got {self.language!r}")

    def run_case(self, build: BuildResult, contract: OperatorContract,
                 case: CaseSpec, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not build.ok:
            raise AdapterError(f"cannot run case {case.id}: build failed: {build.error}")
        outputs = self.runner(build, contract, case, inputs)
        if not isinstance(outputs, dict):
            raise AdapterError(
                f"runner for {self.language!r} must return a dict keyed by output name")
        return outputs


def make_adapter(language: str, runner: Runner) -> KernelAdapter:
    """Build an adapter for ``language`` wrapping ``runner``."""
    return KernelAdapter(language=language, runner=runner)


# --------------------------------------------------------------------------- #
# Default runners (environment-bound; overridden by the orchestrator when a
# toolchain is present). The HIP runner would exec a generated HIP driver that
# calls the entrypoint and dumps outputs; the Triton runner loads a @triton.jit
# module and launches it.
# --------------------------------------------------------------------------- #

def hip_runner(build: BuildResult, contract: OperatorContract, case: CaseSpec,
               inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run a compiled HIP driver. Stub: raises unless a real launcher is wired."""
    raise AdapterError("HIP execution requires a real K100/ROCm launcher")


def triton_runner(build: BuildResult, contract: OperatorContract, case: CaseSpec,
                  inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run a Triton @jit module. Stub: raises unless a real launcher is wired."""
    raise AdapterError("Triton execution requires a real ROCm/torch launcher")


__all__ = [
    "AdapterError", "KernelAdapter", "make_adapter",
    "hip_runner", "triton_runner", "Runner",
]
