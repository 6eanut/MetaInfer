"""Optimization loop: S_baseline self-certification + A→…→F iterations.

The pipeline wires the contract, frozen oracle, GPU pool, ledger, backend (build /
conformance / profile) and an LLM agent runner into a deterministic phase loop::

    S_baseline → A_plan → B_implement → C_conformance → D_review
        → E_perf_test → F_perf_plan → (loop to A_plan) … → finished

Model tiering follows :data:`phases.STRONG_PHASES` / :data:`phases.CHEAP_PHASES`:
strong models plan/review/analyze; cheap models implement/repair/write. The tier
is passed to the agent runner, which in production maps it onto ``AgentSpec.model``.

Promotion happens only in :meth:`Pipeline._perf_test` — a candidate that is
conformance-clean and beats the incumbent per-shape within a noise margin is
appended to the :class:`ChampionLedger`. Everything else is deterministic and
replayable from ``run.json`` + the ledger.

The ``Backend`` and ``agent_runner`` are injected so the loop is testable with
pure-Python fakes (no GPU, no numpy, no LLM).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from metainfer.orchestrator.iteration import IterationWorkspace
from metainfer.orchestrator.state import StateStore

from . import prompts
from .contract import OperatorContract
from .iteration_record import IterationRecord
from .ledger import CaseMetric, ChampionLedger, LedgerEntry
from .oracle import FrozenOracle
from .phases import (
    CHEAP_PHASES,
    PHASE_ORDER,
    STRONG_PHASES,
    next_phase,
    tier_for_phase,
)
from .conformance import ConformanceReport
from .profiler import PerfResult


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Config + injectable seams
# --------------------------------------------------------------------------- #

@dataclass
class PipelineConfig:
    max_iterations: int = 20
    noise_factor: float = 0.10        # perf promotion threshold (beats by this margin)
    regression_factor: float = 0.10   # regression threshold (worse by this margin = no promote)
    max_repairs: int = 3              # B_implement conformance repair attempts
    lease_timeout_s: float = 300.0
    perf_reps: int = 10
    job_id: str = "opt_operator"


class Backend(Protocol):
    """Build / conformance / profile seam. Production impl wires build.py +
    kernel_adapter.py + conformance.py + profiler.py; tests inject fakes."""

    def build(self, source: str, language: str, contract: OperatorContract,
              kernel_dir: Path):
        ...

    def conformance(self, contract: OperatorContract, oracle: FrozenOracle,
                    build, job_id: str) -> ConformanceReport:
        ...

    def profile(self, contract: OperatorContract, build, job_id: str,
                reps: int) -> Dict[str, PerfResult]:
        ...


# agent runner returns a parsed dict of structured evidence for the phase
AgentRunner = Callable[[str, str, str, Path, int], Dict[str, Any]]


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

class Pipeline:
    def __init__(
        self,
        *,
        store: StateStore,
        workspace: IterationWorkspace,
        backend: Backend,
        agent_runner: AgentRunner,
        ledger: ChampionLedger,
        contract: OperatorContract,
        oracle: FrozenOracle,
        initial_source: Optional[str] = None,
        initial_language: Optional[str] = None,
        cfg: Optional[PipelineConfig] = None,
    ) -> None:
        self.store = store
        self.workspace = workspace
        self.backend = backend
        self.agent_runner = agent_runner
        self.ledger = ledger
        self.contract = contract
        self.oracle = oracle
        self.initial_source = initial_source
        self.initial_language = initial_language
        self.cfg = cfg or PipelineConfig()
        self._champ: Optional[LedgerEntry] = None
        self._champ_perf: Dict[str, float] = {}
        self._plan = ""
        self.done = False

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self, *, is_resume: bool = False) -> None:
        run = self.store.load_run()
        start_cycle = run.current_iteration
        if is_resume and self.ledger.current_champion() is not None:
            self._load_champion()
            start_cycle = run.current_iteration
            self.done = run.finished
        else:
            self._s_baseline()
            start_cycle = 0

        if self.done:
            self._finish()
            return

        n = start_cycle + 1
        while n <= self.cfg.max_iterations:
            self._run_cycle(n)
            if self.done:
                break
            n += 1
        self._finish()

    # ------------------------------------------------------------------ #
    # S_baseline — self-certify the initial champion
    # ------------------------------------------------------------------ #

    def _s_baseline(self) -> None:
        self._set_phase(0, "S_baseline")
        source, language = self.initial_source, self.initial_language
        if not source:
            # Mode B without a library baseline: strong agent generates a naive
            # correct kernel, which we then build + certify.
            resp = self._agent(0, "S_baseline", tier_for_phase("S_baseline"),
                               prompts.baseline_prompt(self.contract, self.oracle, {}))
            source = resp.get("source")
            language = resp.get("language") or self.contract.language
            if not source:
                raise PipelineError("S_baseline agent returned no baseline source")

        build = self._build_and_check(source, language)
        report = self.backend.conformance(self.contract, self.oracle, build,
                                          self.cfg.job_id)
        if not report.passed:
            raise PipelineError(
                f"baseline failed conformance: {_conformance_summary(report)}")

        perf = self.backend.profile(self.contract, build, self.cfg.job_id,
                                    self.cfg.perf_reps)
        metrics = {
            cid: CaseMetric(latency_ns=r.latency_ns, speedup=1.0)
            for cid, r in perf.items()
        }
        self._promote(build, metrics, parent=None)
        self._close_iteration(IterationRecord(
            iteration=0, phase="S_baseline", status="success",
            candidate_source=source, candidate_language=language,
            candidate_digest=build.digest, conformance=report.as_dict(),
            perf=_perf_dict(perf), promoted=True,
        ))

    # ------------------------------------------------------------------ #
    # One optimization cycle: A → B → C → D → E → F
    # ------------------------------------------------------------------ #

    def _run_cycle(self, n: int) -> None:
        rec = IterationRecord(iteration=n, phase="A_plan", started_at=time.time())

        # A_plan — strong model strategy
        self._set_phase(n, "A_plan")
        plan_resp = self._agent(
            n, "A_plan", tier_for_phase("A_plan"),
            prompts.plan_prompt(self.contract, self.oracle, self._champ,
                                self._champ_perf, {}))
        self._plan = plan_resp.get("approach", "")
        rec.plan = {"approach": self._plan,
                    "detail": plan_resp.get("detail", "")}
        if plan_resp.get("done"):
            self.done = True
            self._close_iteration(rec)
            return

        # B_implement + C_conformance — cheap model lands the plan; repair loop
        self._set_phase(n, "B_implement")
        candidate_source, candidate_language, candidate_build = \
            self._implement(n, self._plan, rec)

        self._set_phase(n, "C_conformance")
        report = self.backend.conformance(self.contract, self.oracle,
                                          candidate_build, self.cfg.job_id)
        failures = _conformance_failures(report)
        repairs = 0
        while not report.passed and repairs < self.cfg.max_repairs:
            repairs += 1
            self._set_phase(n, "C_conformance")
            resp = self._agent(
                n, "B_implement", tier_for_phase("B_implement"),
                prompts.repair_prompt(self.contract, self._plan, failures, {}))
            candidate_source = resp.get("source")
            candidate_language = resp.get("language") or candidate_language
            candidate_build = self._build_and_check(candidate_source, candidate_language)
            report = self.backend.conformance(self.contract, self.oracle,
                                              candidate_build, self.cfg.job_id)
            failures = _conformance_failures(report)
        rec.candidate_source = candidate_source
        rec.candidate_language = candidate_language
        rec.candidate_digest = candidate_build.digest
        rec.conformance = report.as_dict()

        # D_review — strong model guidance (not a gate)
        self._set_phase(n, "D_review")
        review = self._agent(
            n, "D_review", tier_for_phase("D_review"),
            prompts.review_prompt(self.contract, report.as_dict(),
                                  self._champ_perf, {}))
        rec.guidance = review.get("guidance")

        # E_perf_test — profile + promote
        self._set_phase(n, "E_perf_test")
        promoted = False
        new_perf: Dict[str, PerfResult] = {}
        if report.passed:
            new_perf = self.backend.profile(self.contract, candidate_build,
                                            self.cfg.job_id, self.cfg.perf_reps)
            metrics, promoted = self._should_promote(new_perf)
            if promoted:
                self._promote(candidate_build, metrics, parent=self._champ)
        rec.perf = _perf_dict(new_perf)
        rec.promoted = promoted

        # F_perf_plan — strong model analysis + stop decision
        self._set_phase(n, "F_perf_plan")
        plan2 = self._agent(
            n, "F_perf_plan", tier_for_phase("F_perf_plan"),
            prompts.perf_plan_prompt(self.contract, _perf_dict(new_perf),
                                     self._champ_perf, {}))
        rec.notes.append(plan2.get("next_plan", ""))
        if plan2.get("done"):
            self.done = True

        rec.status = "success"
        rec.ended_at = time.time()
        self._close_iteration(rec)

    # ------------------------------------------------------------------ #
    # Phases: implement (with repair)
    # ------------------------------------------------------------------ #

    def _implement(self, n: int, plan: str, rec: IterationRecord):
        resp = self._agent(
            n, "B_implement", tier_for_phase("B_implement"),
            prompts.implement_prompt(self.contract, plan, {}))
        source = resp.get("source")
        language = resp.get("language") or self.contract.language
        if not source:
            raise PipelineError("B_implement returned no candidate source")
        build = self._build_and_check(source, language)
        return source, language, build

    def _build_and_check(self, source: str, language: str):
        kernel_dir = self.workspace.root / "candidates"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        return self.backend.build(source, language, self.contract, kernel_dir)

    # ------------------------------------------------------------------ #
    # Promotion + ledger
    # ------------------------------------------------------------------ #

    def _should_promote(self, new_perf: Dict[str, PerfResult]):
        noise = self.cfg.noise_factor
        reg = self.cfg.regression_factor
        metrics: Dict[str, CaseMetric] = {}
        promoted = False
        for cid, r in new_perf.items():
            incumbent = self._champ_perf.get(cid)
            if incumbent is None:
                speedup = 1.0
                promoted = True
            else:
                speedup = incumbent / r.latency_ns if r.latency_ns else 1.0
                if r.latency_ns < incumbent * (1 - noise):
                    promoted = True
                if r.latency_ns > incumbent * (1 + reg):
                    return {}, False   # regression anywhere kills the promotion
            metrics[cid] = CaseMetric(latency_ns=r.latency_ns, speedup=speedup)
        return metrics, promoted

    def _promote(self, build, metrics: Dict[str, CaseMetric], parent: Optional[LedgerEntry]):
        nxt_iter = 0 if parent is None else parent.iteration + 1
        entry = LedgerEntry(
            iteration=nxt_iter,
            kernel_digest=build.digest,
            language=build.language,
            contract_digest=self.oracle.contract["name"],
            parent_iteration=parent.iteration if parent else None,
            case_metrics=metrics,
            conformance_digest=build.digest,
        )
        self.ledger.append(entry)
        self._champ = entry
        self._champ_perf = {cid: m.latency_ns for cid, m in metrics.items()}
        self.store.append_timeline("promote", {
            "iteration": nxt_iter,
            "kernel_digest": build.digest,
            "speedup": {c: m.speedup for c, m in metrics.items()},
        })

    def _load_champion(self) -> None:
        champ = self.ledger.current_champion()
        if champ is None:
            raise PipelineError("no champion to resume from")
        self._champ = champ
        self._champ_perf = {cid: m.latency_ns for cid, m in champ.case_metrics.items()}

    # ------------------------------------------------------------------ #
    # Agent / bookkeeping
    # ------------------------------------------------------------------ #

    def _agent(self, n: int, phase: str, tier: str, prompt: str) -> Dict[str, Any]:
        iter_dir = self.workspace.iter_dir(n)
        iter_dir.mkdir(parents=True, exist_ok=True)
        self.store.append_timeline("agent_launch", {
            "phase": phase, "tier": tier, "iteration": n,
        })
        resp = self.agent_runner(phase, tier, prompt, iter_dir, n)
        if not isinstance(resp, dict):
            raise PipelineError(f"{phase} agent returned non-dict: {resp!r}")
        return resp

    def _set_phase(self, n: int, phase: str) -> None:
        self.store.update_run(current_iteration=n, current_phase=phase)
        self.store.append_timeline("phase_start", {"iteration": n, "phase": phase})

    def _close_iteration(self, rec: IterationRecord) -> None:
        if not rec.ended_at:
            rec.ended_at = time.time()
        if rec.status == "running":
            rec.status = "success"
        self.store.write_iteration(rec.iteration, rec.as_dict())

    def _finish(self) -> None:
        self.store.update_run(finished=True, final_status="success",
                              current_phase="finished")
        self.store.append_timeline("run_finish", {
            "current_iteration": self.store.load_run().current_iteration,
        })


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _conformance_summary(report: ConformanceReport) -> str:
    return ", ".join(
        f"{r.case_id}={('pass' if r.passed else f'abs{r.max_abs_err:.2g}/rel{r.max_rel_err:.2g}')}"
        for r in report.results)


def _conformance_failures(report: ConformanceReport) -> List[str]:
    return [
        f"{r.case_id}: {r.detail or 'failed'}"
        for r in report.results if not r.passed
    ]


def _perf_dict(perf: Dict[str, PerfResult]) -> Dict[str, Any]:
    return {cid: {"latency_ns": r.latency_ns, "detail": r.detail}
            for cid, r in perf.items()}


__all__ = ["PipelineError", "PipelineConfig", "Backend", "AgentRunner", "Pipeline"]
