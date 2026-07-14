"""Prompt templates for fusedmoe-evolve sub-agents."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

PREV_ITER_LOGS_SUBDIR = "prev-iter"

def _render_req(req: Dict[str, Any]) -> str:
    lines = [
        "- task_type: %s" % req.get('task_type', '?'),
        "- task_id: %s" % req.get('task_id', '?'),
    ]
    skip = {"task_type", "task_id", "raw_request", "answers"}
    for k, v in req.items():
        if k in skip:
            continue
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        lines.append("- %s: %s" % (k, v))
    answers = req.get("answers") or {}
    for k, v in answers.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(str(x) for x in v) if v else "(none)"
        lines.append("- %s: %s" % (k, v))
    return "\n".join(lines)


def _prev_logs_section(prev_failure, prev_logs_dir=None):
    if not prev_failure:
        return ""
    if prev_logs_dir is None:
        p = ".metainfer-logs/%s/oracle-report.json" % PREV_ITER_LOGS_SUBDIR
    else:
        p = str(prev_logs_dir / "oracle-report.json")
    return "\nPrev failure: %s\nDiagnostic logs: %s\n" % (prev_failure[:200], p)


def _review_feedback_section(text):
    if not text:
        return ""
    return "\n# Previous review (ACT ON IT)\n%s\n" % text[:4096]


def _perf_plan_section(text):
    if not text:
        return ""
    return "\n# Previous perf plan (YOUR BRIEF)\n%s\n" % text[:4096]


# --------------------------------------------------------------------------- #
# A_prepare
# --------------------------------------------------------------------------- #

def prepare_prompt(req, iter_dir, notebooks_dir, iteration,
                   prev_failures=None, review_feedback=None, perf_plan=None,
                   logs_dir=None):
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    ks = req.get('target_kernel_source', '')
    kf = req.get('target_kernel_file', '')
    kfunc = req.get('target_kernel_func', 'fused_moe_kernel')
    api_base = req.get('llm_api_base', '')
    model = req.get('llm_model', '')
    oe_iters = req.get('openevolve_iterations', '50')
    lines = []
    lines.append("STOP IMMEDIATELY after creating the 3 files. Do NOT verify, test, or spawn sub-agents. Write files and exit.")
    lines.append("")
    lines.append("Write these 3 files in %s:" % iter_dir)
    lines.append("")
    lines.append("1) initial_program.py")
    lines.append("   Read kernel '%s' from %s/%s" % (kfunc, ks, kf))
    lines.append("   Copy imports + deps needed by the kernel.")
    lines.append("   Wrap kernel COMPUTATION body in EVOLVE-BLOCK-START/END markers.")
    lines.append("   Add invoke() function that takes (A, B, topk_weights, expert_ids, ...)")
    lines.append("   and runs the kernel via triton grid launch. No sglang imports.")
    lines.append("")
    lines.append("2) evaluator.py -- CRITICAL: handle file path OR raw text input")
    lines.append("   def evaluate(program_text: str) -> dict:")
    lines.append("     if os.path.isfile(program_text) and chr(10) not in program_text:")
    lines.append("         with open(program_text) as f: source = f.read()")
    lines.append("     else: source = program_text")
    lines.append("     ns = {}; exec(source, ns)")
    lines.append("     # Get invoke/kernel from ns. Use torch+triton only. No sglang.")
    lines.append("     # Run random inputs vs torch.einsum+topk reference")
    lines.append("     # Return MUST have: {combined_score: float, passed: bool}")
    lines.append("     # combined_score: -1000 if crash/NaN, 0-50 if fail, 50+50*speedup if pass")
    lines.append("     # Catch ALL exceptions, return {combined_score:-1000, passed:False}")
    lines.append("")
    lines.append("3) config.yaml — OpenEvolve YAML with EXACT keys:")
    lines.append("   llm:")
    lines.append("     primary_model: '%s'" % model)
    lines.append("     api_base: '%s'" % api_base)
    lines.append("     api_key: '${OPENAI_API_KEY}'")
    lines.append("   database:")
    lines.append("     population_size: 25")
    lines.append("     num_islands: 3")
    lines.append("   evaluator:")
    lines.append("     timeout: 300")
    lines.append("     cascade_evaluation: false")
    lines.append("   max_iterations: %s" % oe_iters)
    lines.append("   diff_based_evolution: true")
    lines.append("")
    lines.append("Requirements:")
    lines.append(_render_req(req))
    lines.append("")
    lines.append("Working dir: %s" % iter_dir)
    lines.append("Iteration: %s" % iteration)
    lines.append(_perf_plan_section(perf_plan))
    lines.append(_review_feedback_section(review_feedback))
    lines.append(_prev_logs_section(prev_failures, prev_snap))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# D_review
# --------------------------------------------------------------------------- #

def review_prompt(req, iter_dir, notebooks_dir, iteration,
                  outcome=None, failure=None, perf=None, logs_dir=None):
    lines = []
    lines.append("STOP IMMEDIATELY after writing 2 files. Do NOT analyze endlessly. Write and exit.")
    lines.append("")
    lines.append("Previous B_evolve: outcome=%s failure=%s" % (outcome or "?", failure or "none"))
    lines.append("")
    lines.append("Write these 2 files:")
    lines.append("1) %s/review.md — Brief summary:" % (logs_dir or iter_dir))
    lines.append("   - What happened this iteration (1 sentence)")
    lines.append("   - Root cause of the failure (1 sentence)")
    lines.append("   - 1-2 concrete fix suggestions for next iteration")
    lines.append("   Keep under 500 words. Be terse.")
    lines.append("")
    lines.append("2) %s/perf_plan.md — One-line goal:" % iter_dir)
    lines.append("   e.g. \"fix evaluator syntax error\" or \"target memory coalescing\"")
    lines.append("")
    lines.append("Working dir: %s  Iter: %s" % (iter_dir, iteration))
    lines.append("Do NOT run commands. Do NOT read openevolve output. Just write the files.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# C_validate repair
# --------------------------------------------------------------------------- #

def c_repair_prompt(req, iter_dir, notebooks_dir, iteration,
                    attempt, max_attempts, failure, logs_dir=None):
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    remaining = max(0, max_attempts - attempt)
    lines = []
    lines.append("C-STEP DEBUGGER: iteration %s, repair %s/%s (%s remaining)" %
                 (iteration, attempt, max_attempts, remaining))
    lines.append("")
    lines.append("The evolved kernel at %s/evolved_kernel.py FAILED validation:" % iter_dir)
    lines.append("  %s" % (failure or "(no detail)"))
    lines.append("")
    lines.append("Fix ONE root cause. Make minimal Edit to evolved_kernel.py. Do NOT regenerate.")
    lines.append("")
    lines.append("Write repair log to: %s/c-repair-attempt%s.md" %
                 ((logs_dir or iter_dir), attempt))
    lines.append("Requirements:")
    lines.append(_render_req(req))
    lines.append(_prev_logs_section(failure, prev_snap))
    return "\n".join(lines)


def c_repair_followup_prompt(iteration, attempt, max_attempts, new_failure, logs_dir):
    remaining = max(0, max_attempts - attempt)
    lines = []
    lines.append("C-step re-run STILL FAILS: repair %s/%s (%s remaining)" %
                 (iteration, attempt, remaining))
    lines.append("New failure: %s" % (new_failure or "(none)"))
    lines.append("Resumed session. Fix ONE root cause. Minimal Edit.")
    lines.append("Write %s/c-repair-attempt%s.md" % (logs_dir, attempt))
    return "\n".join(lines)
