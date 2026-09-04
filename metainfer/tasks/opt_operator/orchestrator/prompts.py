"""Prompt templates for each opt_operator phase.

Each builder returns the prompt text handed to an agent for a pool-evolution
step. The prompts are self-contained: contract summary + oracle identity +
(where relevant) the selected kernel's source and its score. Models are tiered
by role: strong models certify/review (harness_setup), cheap models optimize and
repair.
"""

from __future__ import annotations

from typing import Any, Dict


def _contract_summary(contract) -> str:
    lines = [
        f"operator: {contract.name}",
        f"entrypoint: {contract.entrypoint}",
        f"language: {contract.language}",
        f"shape_mode: {contract.shape_mode}",
        f"inputs: {[(t.name, t.dtype, t.shape) for t in contract.inputs]}",
        f"outputs: {[(t.name, t.dtype, t.shape) for t in contract.outputs]}",
        f"numerics: {contract.numerics}",
    ]
    if contract.constraints:
        lines.append(f"constraints: {contract.constraints}")
    return "\n".join(lines)


# Runner-facing ABI every emitted *Triton* kernel must satisfy. The runner loads
# the staged source as a module and calls ``launch(*inputs_cuda)``; ``launch``
# returns the single output CUDA tensor. Inputs arrive in ``contract.inputs``
# order, already on CUDA in the contract dtypes.
def _triton_source_abi(contract) -> str:
    in_sig = ", ".join(t.name for t in contract.inputs)
    out = contract.outputs[0] if contract.outputs else None
    out_name = out.name if out else "C"
    out_dtype_hint = out.dtype if out else "the contract output dtype"
    return (
        f"SOURCE ABI (you MUST satisfy — the conformance/perf runner imports your "
        f"source as a module and calls `launch`):\n"
        f"- Write ONE complete Python module using torch + triton. No bare C/C++.\n"
        f"- Define your `@triton.jit` kernel(s) inside the module.\n"
        f"- Define a module-level function `launch` whose positional parameters are "
        f"EXACTLY the contract inputs IN ORDER, each already a CUDA torch tensor "
        f"of the matching dtype:\n"
        f"    def launch({in_sig}):  ...\n"
        f"  Derive M/N/K from the input tensor shapes at runtime.\n"
        f"- `launch` must RETURN the single output tensor "
        f"({out_name}, dtype {out_dtype_hint}, CUDA, shape [M,N]) — do NOT store "
        f"to a global/output arg.\n"
        f"- int8 x int8 GEMM: accumulate in int32, then multiply the fp32 scale "
        f"factors (a_scale per row x b_scale per column) in fp32, then cast the "
        f"final value to the output dtype with `.to(...)`. Match the reference's "
        f"numerics within the contract tolerances across the whole shape-sweep.\n"
        f"- Return JSON: {{\"language\": \"triton\", \"source\": \"<full module source>\"}}"
    )


def _hip_note() -> str:
    return ("Only Triton is currently executable on this run. If you would emit HIP, "
            "emit the Triton equivalent instead.\n\n")


def baseline_prompt(contract, oracle, ctx: Dict[str, Any]) -> str:
    """harness_setup (mode B): generate a correct, readable genesis kernel."""
    return (
        f"Generate a correct, readable baseline kernel for the operator below.\n\n"
        f"{_contract_summary(contract)}\n\n"
        f"Frozen oracle origin: {oracle.origin} (digest {oracle.digest[:16]}…)\n\n"
        f"The reference defines `forward(**inputs) -> outputs`; your kernel must "
        f"reproduce that numerics within the contract tolerances across the full "
        f"shape-sweep. Prefer simplicity and correctness over speed for the baseline.\n\n"
        f"{_hip_note()}{_triton_source_abi(contract)}"
    )


def optimize_prompt(contract, selected_source: str, selected_latency,
                    quality, ctx: Dict[str, Any]) -> str:
    """optimize: land a candidate that improves on a *selected* pool kernel."""
    selected = selected_source or "<source unavailable>"
    lat = ("n/a" if selected_latency is None
           else f"{selected_latency:.0f} ns representative latency")
    qual = ("n/a" if quality is None else f"{quality:.3f}x vs baseline")
    return (
        f"Improve the kernel selected for optimization on: {contract.name}.\n\n"
        f"{_contract_summary(contract)}\n\n"
        f"shape_mode is '{contract.shape_mode}'. "
        f"({'TARGET this specific shape/neighborhood for maximal speedup' if contract.shape_mode == 'targeted' else 'Write GENERAL code that is fast across the full shape-sweep range'}).\n\n"
        f"Selected kernel ({lat}, {qual}):\n"
        f"```\n{selected}\n```\n\n"
        f"Produce a complete, compilable kernel source that is faster than the "
        f"selected kernel while preserving exact output semantics. The correctness "
        f"harness will compare against the frozen oracle.\n\n"
        f"{_hip_note()}{_triton_source_abi(contract)}"
    )


def repair_prompt(contract, conformance_failures,
                  ctx: Dict[str, Any]) -> str:
    """repair: fix a candidate the correctness harness rejected (structured fails)."""
    fails = "\n".join(f"- {f}" for f in (conformance_failures or ["unknown"]))
    return (
        f"The candidate failed the correctness harness against the oracle. Fix it.\n\n"
        f"Contract: {contract.name} ({contract.language}).\n"
        f"Failures:\n{fails}\n\n"
        f"{_hip_note()}{_triton_source_abi(contract)}"
    )


__all__ = ["baseline_prompt", "optimize_prompt", "repair_prompt"]
