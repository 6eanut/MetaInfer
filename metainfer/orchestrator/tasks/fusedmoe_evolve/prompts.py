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
    """Agent only called when review feedback exists. Job: tweak config.yaml."""
    prev_snap = logs_dir / PREV_ITER_LOGS_SUBDIR if logs_dir is not None else None
    lines = []
    lines.append("STOP IMMEDIATELY. Your ONLY job: read review feedback and update %s/config.yaml accordingly." % iter_dir)
    lines.append("evaluator.py and initial_program.py are already verified — do NOT touch them.")
    lines.append("")
    if review_feedback:
        lines.append("Review feedback (ACT ON THIS):")
        lines.append(review_feedback[:3000])
    if perf_plan:
        lines.append("Perf plan: %s" % perf_plan[:1000])
    if prev_failures:
        lines.append("Previous failures: %s" % str(prev_failures)[:1000])
    lines.append("")
    lines.append("Adjust ONLY these config.yaml keys based on feedback:")
    lines.append("- max_iterations (if convergence was too slow/fast)")
    lines.append("- temperature (if exploration was too narrow/broad)")
    lines.append("- population_size (if diversity was low)")
    lines.append("- checkpoint_interval")
    lines.append("- system_message (add missing DCU architecture hints)")
    lines.append("")
    lines.append("Iter: %s. Write the updated config.yaml, then exit." % iteration)
    return "\n".join(lines)

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
