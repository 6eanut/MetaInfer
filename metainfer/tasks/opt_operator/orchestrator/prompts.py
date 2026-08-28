"""Prompt templates for each opt_operator phase.

Each builder returns the prompt text handed to an agent. The pipeline passes a
structured ``ctx`` dict (contract summary, champion perf, conformance evidence,
prior plan) so the prompt is self-contained. Models are tiered by role: strong
models plan/review/analyze; cheap models implement/repair/write.
"""

from __future__ import annotations

from typing import Any, Dict

from .phases import phase_spec


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


def baseline_prompt(contract, oracle, ctx: Dict[str, Any]) -> str:
    """S_baseline: certify / generate the initial naive champion (mode B)."""
    return (
        f"Generate a correct, readable baseline kernel for the operator below.\n\n"
        f"{_contract_summary(contract)}\n\n"
        f"Frozen oracle origin: {oracle.origin} (digest {oracle.digest[:16]}…)\n\n"
        f"The reference defines `forward(**inputs) -> outputs`; your kernel must "
        f"reproduce that numerics within the contract tolerances across the full "
        f"shape-sweep. Prefer simplicity and correctness over speed for the baseline.\n\n"
        f"Return JSON: {{\"language\": \"hip\"|\"triton\", \"source\": \"<kernel source>\"}}"
    )


def certify_prompt(contract, oracle, champion, ctx: Dict[str, Any]) -> str:
    """S_baseline: certify an already-provided baseline (mode A / library baseline)."""
    return (
        f"Certify the baseline champion for: {contract.name}.\n\n"
        f"{_contract_summary(contract)}\n\n"
        f"Oracle origin {oracle.origin}. The baseline will be conformance-scanned "
        f"against the oracle; review the source for obvious correctness issues.\n\n"
        f"Return JSON: {{\"verdict\": \"ok\"|\"needs_fix\", \"note\": \"...\"}}"
    )


def plan_prompt(contract, oracle, champion, perf, ctx: Dict[str, Any]) -> str:
    """A_plan: strong model devises the next optimization strategy."""
    return (
        f"Devise an optimization strategy for: {contract.name}.\n\n"
        f"{_contract_summary(contract)}\n\n"
        f"shape_mode is '{contract.shape_mode}'. "
        f"({'TARGET this specific shape/neighborhood for maximal speedup' if contract.shape_mode == 'targeted' else 'Write GENERAL code that is fast across the full shape-sweep range'}).\n\n"
        f"Current champion latency per shape: {perf or 'not yet profiled'}\n\n"
        f"Describe the kernel transformation (tiling, vectorization, constexpr "
        f"specialization, fusion, …) as a concrete plan the implementer can land.\n\n"
        f"Return JSON: {{\"approach\": \"<detailed plan>\", \"done\": false, \"detail\": \"<one line>\"}}"
    )


def implement_prompt(contract, plan, ctx: Dict[str, Any]) -> str:
    """B_implement: cheap model lands the strong model's plan into a candidate."""
    return (
        f"Implement the approved optimization plan as a kernel for: {contract.name}.\n\n"
        f"Plan from the architect:\n{plan}\n\n"
        f"Contract:\n{_contract_summary(contract)}\n\n"
        f"Produce complete, compilable kernel source implementing the plan. "
        f"Preserve exact output semantics; the conformance gate will compare "
        f"against the frozen oracle.\n\n"
        f"Return JSON: {{\"language\": \"hip\"|\"triton\", \"source\": \"<kernel source>\"}}"
    )


def repair_prompt(contract, plan, conformance_failures, ctx: Dict[str, Any]) -> str:
    """B_implement repair loop: cheap model fixes conformance failures."""
    fails = "\n".join(f"- {f}" for f in (conformance_failures or ["unknown"]))
    return (
        f"The candidate failed conformance against the oracle. Fix it.\n\n"
        f"Contract: {contract.name} ({contract.language}).\n"
        f"Failures:\n{fails}\n\n"
        f"Original plan:\n{plan}\n\n"
        f"Return JSON: {{\"language\": \"hip\"|\"triton\", \"source\": \"<corrected kernel source>\"}}"
    )


def review_prompt(contract, conformance, perf, ctx: Dict[str, Any]) -> str:
    """D_review: strong model reviews conformance/perf evidence (guidance, not a gate)."""
    return (
        f"Review the evidence for the {contract.name} candidate.\n\n"
        f"Conformance:\n{conformance or 'n/a'}\n"
        f"Perf evidence:\n{perf or 'not yet profiled'}\n\n"
        f"Advise the next step as human-readable guidance (tradeoffs, risks, "
        f"what to try next). This is guidance, not a hard gate.\n\n"
        f"Return JSON: {{\"guidance\": \"<advice>\"}}"
    )


def perf_plan_prompt(contract, perf, incumbent, ctx: Dict[str, Any]) -> str:
    """F_perf_plan: strong model analyzes the profile and plans the next iteration."""
    return (
        f"Analyze the per-shape profile for {contract.name} and plan the next iteration.\n\n"
        f"Profile:\n{perf or 'n/a'}\n\n"
        f"Incumbent champion:\n{incumbent or 'n/a'}\n\n"
        f"Decide: is further optimization promising, or should we stop?\n\n"
        f"Return JSON: {{\"next_plan\": \"<analysis + next steps>\", \"done\": <true|false>}}"
    )


__all__ = [
    "baseline_prompt", "certify_prompt", "plan_prompt", "implement_prompt",
    "repair_prompt", "review_prompt", "perf_plan_prompt",
]
