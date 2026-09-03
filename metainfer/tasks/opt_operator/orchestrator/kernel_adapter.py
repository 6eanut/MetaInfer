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

import hashlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
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


# contract dtype -> torch dtype name (torch imported lazily at runtime)
_TORCH_DTYPES = {
    "fp16": "float16", "bf16": "bfloat16", "fp32": "float32",
    "fp64": "float64", "int8": "int8", "int4": "int8",
}

# artifact path -> (content sha256, loaded module). Re-load only when the staged
# source file changes (each iteration stages a fresh candidate to the same path).
_module_cache: Dict[str, tuple] = {}


def _load_triton_module(build: BuildResult):
    """Dynamically import the staged Triton source, caching by file digest."""
    artifact = build.artifact
    p = Path(artifact)
    try:
        digest = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""
    except OSError:
        digest = ""
    key = str(artifact)
    hit = _module_cache.get(key)
    if hit is not None and hit[0] == digest:
        return hit[1]
    spec = importlib.util.spec_from_file_location("optop_staged_kernel", artifact)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _module_cache[key] = (digest, module)
    return module


def _materialize_inputs(contract: OperatorContract,
                        inputs: Dict[str, Any]) -> list:
    """Convert numpy inputs -> list of CUDA torch tensors in contract.inputs order."""
    import torch  # noqa: PLC0415  (torch present only on the K100 runtime)
    tensors = []
    for t in contract.inputs:
        dt = _TORCH_DTYPES.get(t.dtype)
        if dt is None:
            raise AdapterError(f"no torch dtype for contract dtype {t.dtype!r}")
        arr = inputs[t.name]
        tns = torch.from_numpy(arr).contiguous().to(getattr(torch, dt))
        tensors.append(tns.to("cuda"))
    return tensors


def triton_runner(build: BuildResult, contract: OperatorContract, case: CaseSpec,
                  inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Run a staged Triton @jit module over one case.

    The staged module must expose a callable ``launch`` (see the ABI block added
    to the source-generating prompts) whose positional args are the CUDA tensors
    in ``contract.inputs`` order and whose return is the single output tensor.
    Returns ``{output_name: tensor}`` so conformance compares language-agnostically.
    """
    if build.language != "triton":
        raise AdapterError("triton_runner requires a triton build")
    module = _load_triton_module(build)
    launch = getattr(module, "launch", None)
    if launch is None:
        launch = getattr(module, contract.entrypoint, None)
    if launch is None:
        raise AdapterError(
            f"triton module {build.artifact!r} has no callable launch/{contract.entrypoint}")
    if len(contract.outputs) != 1:
        raise AdapterError("triton backend supports single-output contracts only")
    tensors = _materialize_inputs(contract, inputs)
    out = launch(*tensors)
    return {contract.outputs[0].name: out}


__all__ = [
    "AdapterError", "KernelAdapter", "make_adapter",
    "hip_runner", "triton_runner", "Runner",
    "_load_triton_module", "_materialize_inputs",
]
