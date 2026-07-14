"""Prompt templates for fusedmoe-evolve sub-agents.

Each builder returns a fully-rendered prompt string for one phase.

Phases:
  - A_prepare: agent writes initial_program.py, evaluator.py, config.yaml
  - D_review: agent reviews evolution trajectory, writes review.md + perf_plan.md
  - C_repair: debugger fixes evolved_kernel.py after C_validate failure
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


NOTEBOOKS_HINT = """A knowledge base of reference designs, known pitfalls, and worked
examples lives in the `notebooks/` directory.

Read it EFFICIENTLY:
- Use the Read tool directly. Do NOT spawn sub-agents.
- Start with `Glob notebooks/**/*.md` to see the layout, then Read only
  the files whose names match this iteration's task (typically 3-6 files).
- Do not re-read a file you have already read in this session.
"""

PREV_ITER_LOGS_SUBDIR = "prev-iter"


def _render_req(req: Dict[str, Any]) -> str:
    """Render the frozen requirements for prompt injection."""
    lines = [f"- task_type: {req.get('task_type', '?')}",
             f"- task_id: {req.get('task_id', '?')}",
             f"- raw_request: {req.get('raw_request', '')}"]
    _skip = {"task_type", "task_id", "raw_request", "answers"}
    for k, v in req.items():
        if k in _skip:
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        lines.append(f"- {k}: {v}")
    answers = req.get("answers") or {}
    for k, v in answers.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        lines.append(f"- {k}: {v}")
    return "\n".join(lines)


def _prev_logs_section(
    prev_failure: Optional[str],
    prev_logs_dir: Optional[Path] = None,
) -> str:
    """Render the 'previous iteration diagnostics' block."""
    if not prev_failure:
        return ""
    if prev_logs_dir is None:
        snap = f".metainfer-logs/{PREV_ITER_LOGS_SUBDIR}"
        p_oracle = f"{snap}/oracle-report.json"
    else:
        snap = prev_logs_dir
        p_oracle = snap / "oracle-report.json"
    return f"""

# Previous iteration's diagnostic logs (READ BEFORE CODING)
The previous iteration failed. Its diagnostic artifacts are at:
  - {p_oracle} — full validation verdict
  - Other logs under the same directory

The `prev_failure` text above is a summary. BEFORE writing any code, open
these files and identify the concrete root cause.
"""


def _review_feedback_section(review_feedback: Optional[str]) -> str:
    """Render the 'previous iteration's reviewer suggestions' block."""
    if not review_feedback:
        return ""
    return f"""

# Previous iteration's review (ACT ON IT)
The previous D_review wrote concrete improvement suggestions:
{review_feedback}

Address each suggestion explicitly.
"""


def _perf_plan_section(perf_plan: Optional[str]) -> str:
    """Render the 'previous iteration's perf plan' block."""
    if not perf_plan:
        return ""
    return f"""

# Previous iteration's perf plan (THIS IS YOUR BRIEF)
{perf_plan}

Address each planned optimization item explicitly.
"""


# --------------------------------------------------------------------------- #
# A_prepare — Prepare files for OpenEvolve
# --------------------------------------------------------------------------- #


def prepare_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    prev_failures: Optional[str] = None,
    review_feedback: Optional[str] = None,
    perf_plan: Optional[str] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    return f"""You are an expert ML kernel engineer preparing files for OpenEvolve.

YOUR TASK:
1. Read the target kernel source from:
   {req.get('target_kernel_source', '?')}/{req.get('target_kernel_file', '?')}
   Locate the function "{req.get('target_kernel_func', '?')}".

2. Create {iter_dir}/initial_program.py:
   - Copy all imports and helper functions needed by the kernel
   - Wrap the kernel function body in EVOLVE-BLOCK-START / EVOLVE-BLOCK-END
     markers. Everything INSIDE these markers will be evolved by OpenEvolve.
     Everything OUTSIDE stays fixed.
   - Add an invoke wrapper function at the bottom that:
     * Takes standard FusedMoE inputs (A, B, bias, C, topk_weights,
       sorted_token_ids, expert_ids, num_tokens_post_padded, config...)
     * Calls the evolved kernel inside a Triton grid launch
     * Returns the output tensor
   - The program MUST be self-contained and importable (no sglang dependency)

3. Create {iter_dir}/evaluator.py:
   - Must export a function `evaluate(program_text: str) -> dict`
   - The `program_text` parameter will be the raw text of the evolved
     initial_program.py (including any modifications OpenEvolve made)
   - evaluator.py must:
     a. Dynamically import/exec the evolved program text
     b. Extract the kernel function from it
     c. Run correctness tests against a PyTorch reference MoE
        (implemented inside evaluator.py)
     d. Benchmark performance (GPU time via CUDA events or triton.testing)
     e. Return {{"combined_score": float, "metrics": {{...}},
                "artifacts": {{...}}}}
   - combined_score MUST be:
     * -1000 if kernel crashes or produces NaN/Inf
     * 0-50 if correctness tests fail (proportional to % passed)
     * 50 + 50 x speedup_ratio if all correctness passes
   - Use proper error handling: catch all exceptions, never crash

4. Create {iter_dir}/config.yaml for OpenEvolve:
   - LLM config from req: api_base={req.get('llm_api_base', '?')}, model={req.get('llm_model', '?')}
   - database: population_size=25, num_islands=3, feature_dimensions=["score"]
   - evaluator: timeout=300, cascade_evaluation=false
   - prompt.system_message: Expert Triton kernel optimization guidance
   - max_iterations: {req.get('openevolve_iterations', '50')}
   - diff_based_evolution: true

IMPORTANT CONSTRAINTS FOR initial_program.py:
- The EVOLVE-BLOCK must contain ONLY the kernel computation logic
- Do NOT mark the function signature or grid launch as evolvable
- The kernel MUST maintain numerical correctness
- Preserve the existing compute_type/tl.constexpr/tl.dot API

IF THIS IS NOT ITERATION 1:
- Read {iter_dir}/../<prev>/logs/review.md for last round's findings
- Adjust evaluator.py's scoring weights based on review suggestions
- Modify config.yaml's system_message to steer evolution toward
  promising optimization directions

# Task requirements (frozen)
{_render_req(req)}

# Working directory for this iteration
{iter_dir}

# Knowledge base
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

# Previous iteration failures (if any)
{prev_failures or "(none — this is the first iteration)"}
{_prev_logs_section(prev_failures, prev_snap)}
{_perf_plan_section(perf_plan)}
{_review_feedback_section(review_feedback)}

# Deliverables
Write exactly three files inside `{iter_dir}`:
1. `initial_program.py` — self-contained kernel with EVOLVE-BLOCK markers
2. `evaluator.py` — scoring function for OpenEvolve
3. `config.yaml` — OpenEvolve configuration

Do NOT run openevolve. Do NOT modify anything outside {iter_dir}.
"""

# --------------------------------------------------------------------------- #
# D_review — Review evolution trajectory
# --------------------------------------------------------------------------- #


def review_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    outcome: Optional[str] = None,
    failure: Optional[str] = None,
    perf: Optional[Dict[str, float]] = None,
    logs_dir: Optional[Path] = None,
) -> str:
    outcome_line = (
        f"- C outcome: **{outcome}**"
        if outcome is not None
        else "- C outcome: (not provided)"
    )
    failure_block = (
        f"- Failure reason:\n```\n{failure}\n```"
        if failure
        else "- Failure reason: (none — C passed)"
    )
    perf_block = (
        f"- Measured perf: {json.dumps(perf, sort_keys=True)}"
        if perf
        else "- Measured perf: (none)"
    )
    if logs_dir is not None:
        logs_section = f"""# Diagnostic logs
This iteration's logs live under:
  {logs_dir}/
Open the relevant files there:
  - {logs_dir}/oracle-report.json — full per-case verdict
  - {logs_dir}/openevolve-output.log — openevolve stdout/stderr"""
        review_path = logs_dir / "review.md"
    else:
        logs_section = """# Diagnostic logs
`.metainfer-logs/` in this iteration directory contains the logs."""
        review_path = iter_dir / "review.md"
    return f"""You are an expert kernel performance reviewer.

YOUR TASK:
1. Read the openevolve evolution trajectory:
   - Check if {iter_dir}/openevolve_output/ exists
   - Read the checkpoints for evolution progress data
   - Read {iter_dir}/evolved_kernel.py (the best result)

2. Read the validation results:
   - {logs_dir}/oracle-report.json (C_validate output)
   - Per-case correctness pass/fail + perf metrics

3. Analyze:
   - Did evolution produce a measurable speedup?
   - Which optimization strategies were most effective?
   - Did evolution converge prematurely (scores flatlined)?
   - Any shapes/dtypes where correctness or perf regressed?
   - Compare vs openevolve internal best_score progression

4. Write {review_path} with:
   - Summary of this iteration's outcome
   - Best kernel's performance delta vs baseline
   - Analysis of the evolution trajectory
   - Concrete suggestions for the next iteration:
     * What optimization direction to try next
     * What to change in evaluator.py scoring
     * Whether to adjust openevolve config

5. Write {iter_dir}/perf_plan.md with:
   - Specific optimization target for next iteration
   - One-line goal (e.g., "target memory coalescing in the K-loop")

# Task requirements (frozen)
{_render_req(req)}

# This iteration's test outcome
{outcome_line}
{failure_block}
{perf_block}

{logs_section}

# Working directory (code only — logs are NOT here, see above)
{iter_dir}

# Knowledge base
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

# Checks
- Did the evolved kernel maintain correctness?
- If C failed: what is the root cause?
- If C passed: what's the next likely bottleneck?

# Deliverables
Write `{review_path}` and `{iter_dir}/perf_plan.md`.

Do NOT modify the implementation. Review only.
"""


# --------------------------------------------------------------------------- #
# C_validate repair prompts
# --------------------------------------------------------------------------- #


def c_repair_prompt(
    req: Dict[str, Any],
    iter_dir: Path,
    notebooks_dir: Path,
    iteration: int,
    attempt: int,
    max_attempts: int,
    failure: Optional[str],
    logs_dir: Optional[Path] = None,
) -> str:
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    remaining = max(0, max_attempts - attempt)
    return f"""You are the **C-STEP DEBUGGER** for MetaInfer iteration #{iteration},
repair attempt {attempt} of {max_attempts} ({remaining} attempt(s) remaining
after this one before the iteration gives up and routes to D for review).

The correctness validation (C_validate step) just FAILED with this reason:

```
{failure or "(no failure detail provided — read the diagnostic logs below)"}
```

Your job: **identify ONE root cause from the failure reason above, make a
MINIMAL fix to the evolved kernel code, and STOP.** The orchestrator re-runs
validation as soon as you exit.

# Task requirements (frozen)
{_render_req(req)}

# Working directory (pre-populated by orchestrator — do NOT re-copy)
{iter_dir}
The evolved kernel is at `{iter_dir}/evolved_kernel.py`. Edit it in-place.

# Iteration mode: TARGETED REPAIR (do NOT regenerate)
This is NOT a fresh implementation pass. Touch only the kernel code that
the failure reason points at. Regenerating unrelated files re-introduces
bugs that were already fixed.

# Failure-context diagnostic logs
The full oracle-report.json for this iteration's failing C step lives in:
  {logs_dir}/
Open the relevant files there for the full per-case verdict.
{_prev_logs_section(failure, prev_snap)}

# Knowledge base
{NOTEBOOKS_HINT}
Knowledge base path: {notebooks_dir}

# Discipline (MANDATORY)
1. **Diagnose first.** Re-read the failure reason above. Quote the
   specific symptom before editing.
2. **Fix ONE root cause per attempt.** Pick the one the failure reason
   actually points at.
3. **Minimal diff.** Use Edit, not Write. Do not rewrite whole functions
   unless the bug is structural.
4. **Do NOT re-run validation yourself.** That's the orchestrator's job.

# Deliverable
A minimal code change in `{iter_dir}/evolved_kernel.py`.

**MANDATORY — write a structured repair log** to:
  `{logs_dir}/c-repair-attempt{attempt}.md`

```markdown
# C-step repair attempt {attempt}/{max_attempts}

## Error reason (input)
<one paragraph: what the failing C step reported>

## Root cause hypothesis
<one paragraph: what you believe is the underlying bug. Cite file:line.>

## Fix applied
<bullet list of concrete edits>

## Verification
<one paragraph: what local check(s) you ran>

## Expected next-step outcome
<one sentence>
```
"""


def c_repair_followup_prompt(
    iteration: int,
    attempt: int,
    max_attempts: int,
    new_failure: Optional[str],
    logs_dir: Path,
) -> str:
    remaining = max(0, max_attempts - attempt)
    return f"""The C-step re-run after your previous fix still FAILED. This is
repair attempt {attempt} of {max_attempts} ({remaining} remaining after this).

**New failure from the re-run:**

```
{new_failure or "(no failure detail provided — open the diagnostic logs below)"}
```

You're running in a resumed session — everything you diagnosed, read, and
edited last turn is still in your context. Do NOT re-bootstrap.

Same discipline as before:
1. Identify the ONE root cause the new failure points at.
2. Make a minimal Edit to `evolved_kernel.py` (no rewrites of unrelated code).
3. **MANDATORY**: overwrite `{logs_dir}/c-repair-attempt{attempt}.md`
   with the same 5-section structure as before.

Be terse. Stop as soon as the fix is applied and the .md is written.
"""
