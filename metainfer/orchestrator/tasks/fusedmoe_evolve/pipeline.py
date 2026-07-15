"""Deterministic state-machine-driven orchestrator for fusedmoe-evolve.

4-phase loop::

    phase = "A_prepare"
    while not terminal(phase):
        if no open iteration folder: open one
        outcome = run_phase(phase, ...)
        t = TRANSITIONS[(phase, outcome)]
        update ctx (carry_failure / carry_perf)
        record into iteration record + timeline
        if t.consume_iteration: close the folder
        phase = t.to_phase

Flow: A_prepare → B_evolve → C_validate → D_review → A_prepare (new iter)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import phases as P
from ...iteration import IterationWorkspace
from .prompts import (
    c_repair_followup_prompt,
    c_repair_prompt,
    prepare_prompt,
    review_prompt,
)
from ...state import IterationRecord, StateStore
from ...subagent_manager import AgentSpec, SubAgentManager


JSON_LINE_RE = re.compile(r"\{.*\}")

MAX_PHASE_ATTEMPTS = 3
PERF_REGRESSION_THRESHOLD = 0.20


# --------------------------------------------------------------------------- #
# IterationContext
# --------------------------------------------------------------------------- #


@dataclass
class IterationContext:
    """Mutable state threaded through the orchestrator loop."""

    failure: Optional[str] = None
    last_perf: Optional[Dict[str, float]] = None
    best_perf: Optional[Dict[str, float]] = None
    this_iter_perf: Optional[Dict[str, float]] = None
    last_outcome: Optional[P.Outcome] = None
    review_feedback: Optional[str] = None
    perf_plan: Optional[str] = None
    phase_attempts: Dict[str, int] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


@dataclass
class OrchestratorConfig:
    workdir: Path
    repo_root: Path
    notebooks_dir: Path
    iterations_root: Path
    state_dir: Path
    logs_root: Optional[Path] = None
    max_iterations: int = 10

    # Agent tuning
    plan_timeout_s: int = 1800      # A_prepare timeout
    impl_timeout_s: int = 3600      # (kept for compat)
    review_timeout_s: int = 1800    # D_review timeout
    optimize_timeout_s: int = 3600  # (kept for compat)
    stuck_timeout_s: int = 600
    retro_timeout_s: int = 600

    max_c_retries: int = 3

    # OpenEvolve specific
    openevolve_path: Path = Path("/home/jiakai/0716-fusedmoe-sglang/openevolve")
    openevolve_iterations: int = 50
    openevolve_timeout_s: int = 7200   # per B_evolve oracle run
    validate_timeout_s: int = 1200     # per C_validate oracle run

    # Claude Code
    claude_bin: str = "ccb"
    model: Optional[str] = None
    permission_mode: str = "bypassPermissions"
    extra_claude_args: List[str] = field(default_factory=list)
    primary_perf_metric: Optional[str] = "best_score"


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #


class Orchestrator:
    def __init__(
        self,
        req: Dict[str, Any],
        store: StateStore,
        cfg: OrchestratorConfig,
        manager: Optional[SubAgentManager] = None,
    ) -> None:
        self.req = req
        self.store = store
        self.cfg = cfg
        self.manager = manager or SubAgentManager(
            claude_bin=cfg.claude_bin,
            default_model=cfg.model,
            permission_mode=cfg.permission_mode,
            extra_add_dirs=[
                cfg.notebooks_dir,
                *([cfg.logs_root] if cfg.logs_root else []),
            ],
        )
        self.workspace = IterationWorkspace(
            cfg.iterations_root, logs_root=cfg.logs_root,
        )
        self._stop = False
        self.nooped = False

    def _logs_dir_for(self, n: int) -> Path:
        return self.workspace.logs_dir_for(n)

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        task_id = self.req.get("task_id", "task")
        _, is_resume = self.store.init_or_resume(
            task_id=task_id, task_type=self.req.get("task_type", "unknown"),
        )

        resume_from: Optional[Dict[str, Any]] = None
        if is_resume:
            rs = self.store.load_run()
            if rs.finished:
                self.store.append_timeline(
                    "orchestrator_restart",
                    {"prior_final_status": rs.final_status,
                     "prior_phase": rs.current_phase},
                )
                print(f"[metainfer] task {task_id!r} was finished "
                      f"(status={rs.final_status}); restarting on user request")
                self.store.update_run(
                    finished=False,
                    final_status=None,
                    current_phase="idle",
                    last_outcome=None,
                    last_transition_label=None,
                )
            self.store.append_timeline("orchestrator_resume", {"task_id": task_id})
            print(f"[metainfer] resuming task {task_id!r} from existing state")
            resume_from = self._prepare_resume()
        else:
            self.store.append_timeline("orchestrator_start", {"task_id": task_id})

        try:
            self._loop(resume_from=resume_from)
        except KeyboardInterrupt:
            self.store.append_timeline("orchestrator_abort", {"reason": "keyboard-interrupt"})
            self.store.update_run(finished=True, final_status="aborted",
                                  current_phase="finished")
        finally:
            self.manager.shutdown()
            self.store.append_timeline("orchestrator_end", {"task_id": task_id})

    def _prepare_resume(self) -> Dict[str, Any]:
        """Inspect existing iteration state and figure out where to restart."""
        discarded = self.workspace.discard_latest_incomplete()
        if discarded is not None:
            old_rec = self.store.load_iteration(discarded)
            interrupted_in = None
            if old_rec is not None and old_rec.phases:
                interrupted_in = max(
                    old_rec.phases.keys(),
                    key=lambda k: old_rec.phases[k].get("started_at", 0),
                )
            reason = (
                f"user interrupted (orchestrator process exited mid-"
                f"{interrupted_in or 'flight'})"
                if interrupted_in
                else "user interrupted (orchestrator process exited unexpectedly)"
            )
            self.store.archive_interrupted_iteration(discarded, reason=reason)
            self.store.append_timeline(
                "iteration_interrupted",
                {"iteration": discarded, "reason": reason,
                 "restart_from": (old_rec.start_phase if old_rec else "A_prepare")},
            )
            print(f"[metainfer] iteration {discarded:03d} marked interrupted")
            start_phase = (old_rec.start_phase if old_rec else "A_prepare") or "A_prepare"
            iter_num = discarded
            carried_failure = None
            last_outcome: Optional[P.Outcome] = None
            prev_rec = self.store.load_iteration(iter_num - 1) if iter_num > 1 else None
            if prev_rec is not None:
                last_outcome = prev_rec.outcome
        else:
            last_complete = self.workspace.latest_complete_number()
            prev_rec = self.store.load_iteration(last_complete) if last_complete else None
            iter_num = last_complete + 1
            start_phase = "A_prepare"
            carried_failure = None
            last_outcome = prev_rec.outcome if prev_rec is not None else None

        return {
            "iter_num": iter_num,
            "start_phase": start_phase,
            "carried_failure": carried_failure,
            "last_outcome": last_outcome,
        }

    def stop(self) -> None:
        self._stop = True

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #

    def _loop(self, resume_from: Optional[Dict[str, Any]] = None) -> None:
        max_iters = self._resolve_max_iterations()
        ctx = IterationContext()

        if resume_from is not None:
            phase: P.Phase = resume_from["start_phase"]
            iter_num = resume_from["iter_num"] - 1
            ctx.failure = resume_from.get("carried_failure")
            ctx.last_outcome = resume_from.get("last_outcome")
        else:
            phase = "A_prepare"
            iter_num = 0

        iter_dir: Optional[Path] = None
        iter_rec: Optional[IterationRecord] = None
        final_status: Optional[str] = None

        while not self._stop and not P.is_terminal(phase):
            # ---- open a fresh iteration folder if needed ------------------ #
            if iter_dir is None:
                if iter_num >= max_iters:
                    final_status = "success" if ctx.last_outcome == P.OK else "stopped"
                    phase = "finished"
                    break
                iter_num += 1
                iter_dir = self.workspace.open_iteration(iter_num)
                iter_rec = IterationRecord(
                    iteration=iter_num,
                    started_at=time.time(),
                    start_phase=phase,
                )
                self.store.write_iteration(iter_rec)
                ctx.this_iter_perf = None
                self.store.update_run(
                    current_iteration=iter_num,
                    current_phase=phase,
                    last_outcome=ctx.last_outcome,
                )
                self.store.append_timeline(
                    "iteration_start",
                    {"iteration": iter_num, "start_phase": phase,
                     "carried_failure": ctx.failure is not None},
                )
                ctx.phase_attempts.clear()

            # ---- run the phase ------------------------------------------- #
            assert iter_rec is not None
            self._set_phase(iter_num, iter_dir, phase)
            outcome, perf, failure = self._run_phase(phase, iter_num, iter_dir, iter_rec, ctx)

            ctx.last_outcome = outcome
            ctx.phase_attempts[phase] = ctx.phase_attempts.get(phase, 0) + 1

            # Record phase outcome
            phase_rec = iter_rec.phases.setdefault(phase, {})
            phase_rec["outcome"] = outcome
            phase_rec["attempts"] = ctx.phase_attempts[phase]
            phase_rec["ended_at"] = time.time()
            if failure:
                phase_rec["failure"] = failure
            if perf:
                phase_rec["perf"] = perf

            # ---- escalation: too many attempts --------------------------- #
            if outcome != P.OK and ctx.phase_attempts[phase] >= MAX_PHASE_ATTEMPTS:
                self.store.append_timeline(
                    "phase_escalation",
                    {"phase": phase, "attempts": ctx.phase_attempts[phase]},
                )
                forced_failure = (
                    failure
                    or f"{phase} exceeded {MAX_PHASE_ATTEMPTS} in-place attempts "
                       f"(last outcome={outcome})"
                )
                self._close_iteration(
                    iter_rec, status="failed",
                    failure=forced_failure, perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                ctx.failure = forced_failure
                ctx.last_outcome = outcome
                iter_dir = None
                iter_rec = None
                phase = "A_prepare"
                continue

            # ---- consult the transition table ---------------------------- #
            t = P.next_transition(phase, outcome)
            if t is None:
                err = f"no transition defined for ({phase}, {outcome})"
                forced_failure = failure or err
                self._close_iteration(
                    iter_rec, status="failed",
                    failure=err, perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                ctx.failure = forced_failure
                ctx.last_outcome = outcome
                iter_dir = None
                iter_rec = None
                phase = "A_prepare"
                continue

            # ---- update ctx ------------------------------------------------ #
            if t.carry_failure:
                ctx.failure = failure
            elif outcome == P.OK:
                ctx.failure = None

            if perf:
                if t.carry_perf:
                    ctx.last_perf = perf
                ctx.best_perf = _merge_best(ctx.best_perf, perf)
                ctx.this_iter_perf = perf

            # ---- record + advance ----------------------------------------- #
            self.store.append_timeline(
                "transition",
                {"from": phase, "outcome": outcome, "to": t.to_phase,
                 "label": t.label, "iteration": iter_num,
                 "consume_iteration": t.consume_iteration},
            )
            self.store.update_run(
                current_phase=t.to_phase,
                last_outcome=outcome,
                last_transition_label=t.label,
                current_iteration=iter_num,
            )

            if t.consume_iteration:
                iter_status = "success" if outcome == P.OK else "failed"
                self._close_iteration(
                    iter_rec,
                    status=iter_status,
                    failure=(failure if outcome != P.OK else None),
                    perf=ctx.this_iter_perf,
                    outcome=outcome,
                )
                iter_dir = None
                iter_rec = None

            phase = t.to_phase

        # ---- loop done ----------------------------------------------------- #
        if final_status is None:
            final_status = "success" if ctx.last_outcome == P.OK else "stopped"
        self.store.update_run(
            finished=True,
            final_status=final_status,
            current_phase="finished",
            last_outcome=ctx.last_outcome,
        )

    # ------------------------------------------------------------------ #
    # Phase dispatcher
    # ------------------------------------------------------------------ #

    def _run_phase(
        self,
        phase: P.Phase,
        iter_num: int,
        iter_dir: Path,
        rec: IterationRecord,
        ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        if phase == "A_prepare":
            return self._do_prepare(iter_num, iter_dir, ctx)
        if phase == "B_evolve":
            return self._do_evolve(iter_num, iter_dir, ctx)
        if phase == "C_validate":
            return self._do_validate(iter_num, iter_dir, ctx)
        if phase == "D_review":
            return self._do_review(iter_num, iter_dir, ctx)
        raise ValueError(f"no handler for phase {phase!r}")

    # ---- A: prepare — copy templates + agent writes config.yaml ----------- #

    def _do_prepare(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        # Resolve template directory (shipped with the plugin)
        _template_dir = Path(__file__).parent / "oracles" / "templates"

        # Copy verified evaluator.py, initial_program.py, config.yaml,
        # and reference/ from templates. Agent does NOT write these from
        # scratch — the hand-crafted versions in templates/ are battle-tested.
        import shutil
        for _fname in ("evaluator.py", "initial_program.py", "config.yaml"):
            _src = _template_dir / _fname
            _dst = iter_dir / _fname
            if _src.is_file():
                shutil.copy2(_src, _dst)
        _ref_src = _template_dir / "reference"
        _ref_dst = iter_dir / "reference"
        if _ref_src.is_dir() and not _ref_dst.exists():
            shutil.copytree(_ref_src, _ref_dst)

        # If no review feedback from previous D_review, skip the agent
        # entirely — template files are already the proven baseline.
        # Agent only runs to adjust config.yaml when review suggests changes.
        # Only run agent if there is review feedback worth acting on.
        # On the first iteration, template config.yaml is used as-is.
        if ctx.review_feedback or ctx.perf_plan:
            ok, err, mode, _sid = self._run_agent(
                name=f"iter{n}-preparer", role="preparer", iteration=n, iter_dir=iter_dir,
                prompt=prepare_prompt(
                    req=self.req, iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n, prev_failures=ctx.failure,
                    review_feedback=ctx.review_feedback,
                    perf_plan=ctx.perf_plan,
                    logs_dir=self._logs_dir_for(n),
                ),
                timeout=self.cfg.plan_timeout_s,
            )
            if ok:
                return P.OK, None, None
            return _failure_outcome(mode), None, err
        return P.OK, None, None

    # ---- B: evolve (oracle) ---------------------------------------------- #

    def _do_evolve(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        from .oracles.openevolve_oracle import OpenEvolveOracle

        oracle = OpenEvolveOracle()
        report_dir = self._logs_dir_for(n)
        report_dir.mkdir(parents=True, exist_ok=True)

        self.store.append_timeline("oracle_start",
                                   {"iteration": n, "oracle": "openevolve"})

        try:
            result = oracle.run(
                iter_dir=iter_dir,
                req=self.req,
                report_dir=report_dir,
                timeout_s=self.cfg.openevolve_timeout_s,
                openevolve_path=self.cfg.openevolve_path,
                openevolve_iterations=self.cfg.openevolve_iterations,
            )
        except Exception as e:
            self.store.append_timeline("oracle_end",
                                       {"iteration": n, "passed": False,
                                        "error": repr(e)})
            return P.INFRA_FAIL, None, f"B (evolve) oracle crash: {e!r}"

        self.store.append_timeline("oracle_end",
                                   {"iteration": n, "passed": result.passed,
                                    "perf": result.perf})

        if result.passed:
            return P.OK, result.perf or None, None

        return P.LOGIC_FAIL, result.perf, result.failure_reason

    # ---- C: validate (oracle with repair loop) --------------------------- #

    def _do_validate(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        from .oracles.validate_oracle import ValidateOracle

        oracle = ValidateOracle()
        max_attempts = max(1, int(self.cfg.max_c_retries))
        last_outcome: Optional[P.Outcome] = None
        last_perf: Optional[Dict[str, float]] = None
        last_failure: Optional[str] = None
        repair_log_path = self._logs_dir_for(n) / "c-repairs.jsonl"
        try:
            repair_log_path.write_text("", encoding="utf-8")
        except OSError:
            pass

        for attempt in range(1, max_attempts + 1):
            self.store.append_timeline("c_validate_attempt", {
                "iteration": n, "attempt": attempt, "max": max_attempts,
            })

            t_attempt_start = time.time()

            # Run oracle
            report_dir = self._logs_dir_for(n)
            report_dir.mkdir(parents=True, exist_ok=True)
            self.store.append_timeline("oracle_start",
                                       {"iteration": n, "oracle": "validate"})

            try:
                result = oracle.run(
                    iter_dir=iter_dir,
                    req=self.req,
                    report_dir=report_dir,
                    timeout_s=self.cfg.validate_timeout_s,
                )
            except Exception as exc:
                err = f"validate oracle exception: {exc!r}"
                self.store.append_timeline("oracle_end",
                                           {"iteration": n, "passed": False, "error": err})
                return P.INFRA_FAIL, None, err

            self.store.append_timeline("oracle_end", {
                "iteration": n, "passed": result.passed,
                "failure_reason": result.failure_reason,
            })

            if result.passed:
                outcome = P.OK
            else:
                outcome = P.LOGIC_FAIL

            perf = result.perf or None
            failure = result.failure_reason
            last_outcome, last_perf, last_failure = outcome, perf, failure

            # Happy path
            if outcome == P.OK:
                if perf and ctx.best_perf and self.cfg.primary_perf_metric:
                    if _is_regression(ctx.best_perf, perf,
                                      metric=self.cfg.primary_perf_metric,
                                      threshold=PERF_REGRESSION_THRESHOLD):
                        return P.PERF_REGRESSION, perf, (
                            f"perf regression on {self.cfg.primary_perf_metric}: "
                            f"best={ctx.best_perf.get(self.cfg.primary_perf_metric)} "
                            f"new={perf.get(self.cfg.primary_perf_metric)}"
                        )
                return P.OK, perf, None

            # INFRA_FAIL surfaces immediately
            if outcome != P.LOGIC_FAIL:
                return outcome, perf, failure

            # LOGIC_FAIL: dispatch debugger if budget remains
            if attempt >= max_attempts:
                self.store.append_timeline("c_validate_budget_exhausted", {
                    "iteration": n, "attempts": attempt,
                    "final_failure": (failure or "")[:500],
                })
                break

            # Repair: dispatch debugger agent
            self.store.append_timeline("c_validate_repair_start", {
                "iteration": n, "attempt": attempt,
                "reason": (failure or "")[:500],
            })
            dbg_name = f"iter{n}-c-debugger.attempt{attempt}"
            ok, err, mode, _sid = self._run_agent(
                name=dbg_name,
                role="c_debugger",
                iteration=n, iter_dir=iter_dir,
                prompt=c_repair_prompt(
                    req=self.req, iter_dir=iter_dir,
                    notebooks_dir=self.cfg.notebooks_dir,
                    iteration=n, attempt=attempt, max_attempts=max_attempts,
                    failure=failure, logs_dir=self._logs_dir_for(n),
                ),
                timeout=self.cfg.impl_timeout_s,
            )
            if not ok:
                self.store.append_timeline("c_validate_repair_agent_fail", {
                    "iteration": n, "attempt": attempt,
                    "error": err, "mode": mode,
                })

        return last_outcome or P.LOGIC_FAIL, last_perf, last_failure

    # ---- D: review ------------------------------------------------------- #

    def _do_review(
        self, n: int, iter_dir: Path, ctx: IterationContext,
    ) -> Tuple[P.Outcome, Optional[Dict[str, float]], Optional[str]]:
        c_outcome = ctx.last_outcome
        c_failure = ctx.failure
        c_perf = ctx.last_perf

        ok, _err, _mode, _sid = self._run_agent(
            name=f"iter{n}-reviewer", role="reviewer", iteration=n, iter_dir=iter_dir,
            prompt=review_prompt(
                req=self.req, iter_dir=iter_dir,
                notebooks_dir=self.cfg.notebooks_dir, iteration=n,
                outcome=c_outcome, failure=c_failure, perf=c_perf,
                logs_dir=self._logs_dir_for(n),
            ),
            timeout=self.cfg.review_timeout_s,
        )

        # Capture review.md as next-iteration feedback
        review_path = self._logs_dir_for(n) / "review.md"
        if review_path.is_file():
            try:
                text = review_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 8192:
                    text = text[:8160] + "\n...[truncated]..."
                ctx.review_feedback = text
            except OSError:
                pass

        # Capture perf_plan.md for next A_prepare
        plan_path = iter_dir / "perf_plan.md"
        if plan_path.is_file():
            try:
                text = plan_path.read_text(encoding="utf-8", errors="replace")
                if len(text) > 12288:
                    text = text[:12256] + "\n...[truncated]..."
                ctx.perf_plan = text
            except OSError:
                pass

        self.store.append_timeline(
            "review_done",
            {"iteration": n, "c_outcome": c_outcome,
             "review_feedback_captured": ctx.review_feedback is not None,
             "perf_plan_captured": ctx.perf_plan is not None,
             "reviewer_agent_ok": ok},
        )

        # D is ALWAYS advisory — return OK regardless of reviewer agent
        # success; the transition table routes based on this:
        #   (D_review, OK) -> A_prepare (new iter)
        return P.OK, c_perf, None

    # ------------------------------------------------------------------ #
    # Agent runner
    # ------------------------------------------------------------------ #

    def _run_agent(
        self,
        name: str,
        role: str,
        iteration: int,
        iter_dir: Path,
        prompt: str,
        timeout: int,
        *,
        resume_session_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Launch one sub-agent. Returns ``(ok, error, failure_mode, session_id)``."""
        logs_dir = self._logs_dir_for(iteration)
        logs_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = logs_dir / f"{name}.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        spec = AgentSpec(
            name=name,
            role=role,
            prompt_file=prompt_file,
            workdir=iter_dir,
            log_dir=logs_dir,
            timeout_s=timeout,
            stuck_timeout_s=self.cfg.stuck_timeout_s,
            extra_args=list(self.cfg.extra_claude_args),
            session_id=session_id,
            resume_session_id=resume_session_id,
        )
        self.store.append_timeline(
            "agent_launch", {"name": name, "role": role, "iteration": iteration,
                             "resume_from": resume_session_id})
        self.manager.launch(spec)
        result = self.manager.result(name)
        if result is None:
            return False, "no result recorded", "infra", None

        try:
            rec = self.store.load_iteration(iteration)
            if rec is not None:
                rec.phases.setdefault(role, {}).update({
                    "duration_s": result.duration_s,
                    "success": result.success,
                    "error": result.error,
                    "failure_mode": result.failure_mode,
                    "attempts": result.attempts,
                    "final_text_head": result.final_text[:1000],
                    "log_file": str(spec.log_file(result.attempts)),
                    "session_id": result.session_id,
                })
                self.store.write_iteration(rec)
        except Exception:
            pass

        self.store.append_timeline(
            "agent_end",
            {"name": name, "role": role, "iteration": iteration,
             "success": result.success, "error": result.error,
             "failure_mode": result.failure_mode,
             "duration_s": result.duration_s, "attempts": result.attempts,
             "session_id": result.session_id},
        )
        return result.success, result.error, result.failure_mode, result.session_id

    # ------------------------------------------------------------------ #
    # Phase / iteration bookkeeping
    # ------------------------------------------------------------------ #

    def _set_phase(self, n: int, iter_dir: Path, phase: P.Phase) -> None:
        self.store.update_run(current_iteration=n, current_phase=phase)
        self.store.append_timeline("phase_start", {"iteration": n, "phase": phase})
        rec = self.store.load_iteration(n)
        if rec is not None:
            rec.phases.setdefault(phase, {"started_at": time.time()})
            self.store.write_iteration(rec)

    def _close_iteration(
        self,
        rec: IterationRecord,
        *,
        status: str,
        failure: Optional[str],
        perf: Optional[Dict[str, float]],
        outcome: Optional[P.Outcome],
    ) -> None:
        rec.ended_at = time.time()
        rec.duration_s = rec.ended_at - rec.started_at
        rec.status = status
        rec.failure_reason = failure
        rec.outcome = outcome
        if perf:
            rec.perf = perf
        self.store.write_iteration(rec)
        self.workspace.mark_complete(rec.iteration)
        self.store.append_timeline(
            "iteration_end",
            {"iteration": rec.iteration, "status": status, "outcome": outcome,
             "perf": perf, "failure_reason": failure, "duration_s": rec.duration_s},
        )

    def _resolve_max_iterations(self) -> int:
        v = self.req.get("max_iterations")
        if v is None:
            v = self.req.get("answers", {}).get("max_iterations")
        if v is None:
            return self.cfg.max_iterations
        try:
            return int(v)
        except (TypeError, ValueError):
            return self.cfg.max_iterations


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _failure_outcome(mode: Optional[str]) -> P.Outcome:
    if mode == "infra":
        return P.INFRA_FAIL
    return P.LOGIC_FAIL


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _merge_best(
    best: Optional[Dict[str, float]], new: Dict[str, float]
) -> Dict[str, float]:
    """Keep the max of each metric (higher is better)."""
    out = dict(best or {})
    for k, v in new.items():
        if k not in out or v > out[k]:
            out[k] = v
    return out


def _is_regression(
    best: Dict[str, float],
    new: Dict[str, float],
    metric: str,
    threshold: float,
) -> bool:
    if metric not in best or metric not in new:
        return False
    b = best[metric]
    if not b:
        return False
    return (new[metric] / b) < (1.0 - threshold)
