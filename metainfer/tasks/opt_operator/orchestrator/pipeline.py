"""Pool-evolution optimization loop (OPT_KERNEL_SPEC FR-4/5/6/7).

The pipeline wires the contract, frozen oracle, GPU pool, kernel pool, twin
harnesses (correctness + benchmark), backend and an agent runner into a
deterministic *pool-evolution* loop::

    harness_setup (genesis + adversarial harness self-review)
        -> select_kernel (quality-weighted, seeded) -> optimize
            -> verify (twin harnesses) -> [fail] -> repair (<= max_repairs)
                -> [admit_to_pool | discarded]
        -> select_kernel -> ... -> finished

Unlike the old single-champion A…F chain, kernels accumulate in a **pool**
(:class:`KernelPool`, authoritative ``kernel_pool.jsonl``). Every admitted
kernel carries its own benchmark evidence; the champion, per-kernel speedups,
quality scores and lineage are all *derived* from the pool at read time. Each
round picks a kernel to improve by quality-weighted probability (seeded and
reproducible), so high-quality kernels are improved most often but exploration
is never fully starved.

An ``optimize`` round never crashes the whole run: a candidate that yields no
source, fails to build/launch, or still fails correctness after ``max_repairs``
repairs is recorded as ``failed`` / ``discarded`` and the loop moves on to the
next selected kernel.

The ``Backend`` and ``agent_runner`` are injected so the loop is testable with
pure-Python fakes (no GPU, no numpy, no LLM).
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from metainfer.orchestrator.iteration import IterationWorkspace
from metainfer.orchestrator.state import StateStore

from . import prompts
from .adversarial import review_benchmark, review_correctness
from .conformance import ConformanceReport
from .contract import OperatorContract
from .harness import BenchmarkHarness, CorrectnessHarness
from .iteration_record import IterationRecord
from .ledger import ChampionLedger
from .oracle import FrozenOracle
from .phases import tier_for_phase
from .pool import KernelPool, PoolEntry, PoolError
from .profiler import PerfResult


class PipelineError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Config + injectable seams
# --------------------------------------------------------------------------- #

@dataclass
class PipelineConfig:
    max_iterations: int = 20       # outer round budget
    max_repairs: int = 3           # verify-correctness repair attempts per round
    weight_power: float = 1.0      # selection weight ~ quality^weight_power
    score_gate: float = 1.0        # admission: quality >= gate vs baseline
    sample_seed: Optional[int] = None   # fixed -> reproducible sampling
    lease_timeout_s: float = 300.0
    warmup: int = 2                # benchmark harness口径 (recorded as meta)
    reps: int = 10
    statistic: str = "median"      # "median" | "mean"
    job_id: str = "opt_operator"


class Backend(Protocol):
    """Build / conformance / profile seam. Production impl wires build.py +
    kernel_adapter.py + conformance.py + profiler.py; tests inject fakes.

    A backend MAY also expose ``oracle_outputs(contract, oracle, job_id)``
    returning per-case oracle outputs; when present the pipeline runs the
    adversarial correctness review at harness_setup (negative-case self-proof).
    Its absence only skips that optional review, never fails a run.
    """

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
        # The authoritative pool is behind the ledger facade (pool file).
        self._pool: KernelPool = ledger.pool
        # Twin harnesses carry the reproducibility口径 for every verification.
        self._corr = CorrectnessHarness(contract, oracle)
        self._bench = BenchmarkHarness(contract, warmup=self.cfg.warmup,
                                       reps=self.cfg.reps,
                                       statistic=self.cfg.statistic)
        seed = self.cfg.sample_seed
        if seed is None:
            seed = int(self.oracle.digest[:8], 16)
        self._rng = random.Random(seed)
        self._seed = seed
        self.done = False

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self, *, is_resume: bool = False) -> None:
        run = self.store.load_run()
        if self._pool.read_all():
            # Pool already has genesis + admissions (restart / resume): keep the
            # champion state and continue from the last completed round. Only the
            # *pool file* is authoritative here — never re-run harness_setup.
            self.done = run.finished
            start_round = run.current_iteration
            if self.done:
                self._finish()
                return
        else:
            self._harness_setup()
            start_round = 0

        n = start_round + 1
        while n <= self.cfg.max_iterations:
            self._run_round(n)
            if self.done:
                break
            n += 1
        self._finish()

    # ------------------------------------------------------------------ #
    # harness_setup — genesis + adversarial harness self-review (once)
    # ------------------------------------------------------------------ #

    def _harness_setup(self) -> None:
        self._set_phase(0, "harness_setup")
        source, language = self.initial_source, self.initial_language
        if not source:
            # Mode B without a library baseline: strong agent generates a naive
            # correct kernel, which we then build + certify into the pool.
            resp = self._agent(0, "harness_setup",
                               tier_for_phase("harness_setup"),
                               prompts.baseline_prompt(self.contract,
                                                       self.oracle, {}))
            source = resp.get("source")
            language = resp.get("language") or self.contract.language
            if not source:
                raise PipelineError("harness_setup agent returned no baseline source")

        build = self._build_and_check(source, language)
        report = self.backend.conformance(self.contract, self.oracle, build,
                                          self.cfg.job_id)
        if not report.passed:
            raise PipelineError(
                f"baseline failed conformance: {_conformance_summary(report)}")

        perf = self.backend.profile(self.contract, build, self.cfg.job_id,
                                    self.cfg.reps)
        genesis = self._admit(
            PoolEntry(
                iteration=0,
                kernel_digest=build.digest,
                language=build.language,
                contract_digest=self.contract.name,
                parent_iteration=None,
                case_latency_ns={cid: p.latency_ns for cid, p in perf.items()},
                source_path=build.artifact,
                conformance_digest=build.digest,
            ),
            note="genesis baseline",
        )
        self._bench.baseline_digest = genesis.kernel_digest

        reviews = self._run_harness_reviews()
        self._close_iteration(IterationRecord(
            iteration=0, phase="harness_setup", status="success",
            candidate_source=source, candidate_language=language,
            candidate_digest=build.digest, admitted=True, outcome="admitted",
            conformance=report.as_dict(), perf=_perf_dict(perf),
            correctness_meta=self._corr.meta,
            benchmark_meta=self._bench.meta, reviews=reviews,
            notes=[f"seed={self._seed}", f"genesis digest={build.digest[:12]}…"],
        ))

    def _run_harness_reviews(self) -> Dict[str, Any]:
        """Snapshot the twin-harness self-review into a leaves-on-record dict.

        The benchmark review (methodology authority) always runs from the harness
        meta. The adversarial correctness review needs per-case oracle *outputs*;
        it runs only when the backend can produce them (production GPU path),
        otherwise it is recorded as skipped so the loop stays testable.
        """
        bench_review = review_benchmark(self._bench.meta,
                                        self._corr.shape_ids).as_dict()
        oracle_outputs = getattr(self.backend, "oracle_outputs", None)
        if oracle_outputs is None:
            corr_review = {
                "kind": "correctness",
                "passed": None,
                "checks": [{"name": "oracle_outputs", "passed": False,
                            "detail": "backend lacks oracle_outputs; "
                                      "adversarial review skipped"}],
                "negative_evidence": [],
                "message": "skipped (no oracle outputs capability)",
            }
        else:
            try:
                outputs = oracle_outputs(self.contract, self.oracle,
                                         self.cfg.job_id)
                corr_review = review_correctness(self.contract, outputs).as_dict()
            except Exception as exc:  # noqa: BLE001 — a review failure is recorded, not fatal
                corr_review = {"kind": "correctness", "passed": False,
                               "checks": [], "negative_evidence": [],
                               "message": f"adversarial review raised: {exc}"}
        reviews = {"correctness": corr_review, "benchmark": bench_review}
        self.store.append_timeline("harness_review", {
            "iteration": 0,
            "correctness_passed": corr_review.get("passed"),
            "benchmark_passed": bench_review.get("passed"),
        })
        return reviews

    # ------------------------------------------------------------------ #
    # One optimization round: select -> optimize -> verify(+repair) -> settle
    # ------------------------------------------------------------------ #

    def _run_round(self, n: int) -> None:
        rec = IterationRecord(iteration=n, phase="optimize", started_at=time.time())

        # select_kernel — seeded quality-weighted sample from the pool.
        self._set_phase(n, "select_kernel")
        base = self._pool.sample_kernel(self._rng, weight_power=self.cfg.weight_power)
        if base is None:
            # A fresh pool has genesis, so this only fires on a corrupt/empty
            # store — treat as a hard invariant, not a recoverable round.
            raise PipelineError("select_kernel: pool is empty")
        rec.selected_iteration = base.iteration
        rec.selected_digest = base.kernel_digest
        rec.notes.append(
            f"selected pool kernel iteration={base.iteration} "
            f"digest={base.kernel_digest[:12]}…")

        # optimize — agent improves the selected kernel into a candidate.
        self._set_phase(n, "optimize")
        base_source = _read_source(base)
        base_lat = self._pool.rep_latency_for(base)
        opt = self._agent(
            n, "optimize", tier_for_phase("optimize"),
            prompts.optimize_prompt(
                self.contract, base_source, base_lat,
                self._pool.quality(base),
                {"selected_iteration": base.iteration}))
        source = opt.get("source")
        language = opt.get("language") or self.contract.language
        if not source:
            self._settle(rec, n, outcome="failed", note="optimize returned no source")
            return
        rec.candidate_source = source
        rec.candidate_language = language

        # verify (correctness harness) with bounded repair; returns a build only
        # if the candidate is conformance-clean, else None.
        result = self._verify_and_repair(n, rec, source, language)
        if result is None:
            self._settle(rec, n, outcome="discarded",
                         note="correctness failed after repairs / no build")
            return
        build, report = result
        rec.candidate_digest = build.digest
        rec.conformance = report.as_dict()

        # benchmark harness — profile every case (same shape set as correctness).
        try:
            perf = self._bench.run(self.backend, build, self.cfg.job_id)
        except Exception as exc:  # noqa: BLE001 — a profiling failure discards, never crashes
            rec.notes.append(f"benchmark failed: {exc}")
            self._settle(rec, n, outcome="discarded", note="benchmark failed")
            return
        rec.perf = _perf_dict(perf)
        lat = {cid: p.latency_ns for cid, p in perf.items()}

        # Admission gate — candidate is correct and not a gross regression vs the
        # baseline. Per-shape specialization is allowed (champion is derived); we
        # only keep a *representative* floor so a globally-worse kernel can't flood
        # the pool.
        candidate = PoolEntry(
            iteration=n,
            kernel_digest=build.digest,
            language=build.language,
            contract_digest=self.contract.name,
            parent_iteration=base.iteration,
            case_latency_ns=lat,
            source_path=build.artifact,
            conformance_digest=build.digest,
        )
        quality = self._pool.quality(candidate)
        rec.quality = quality
        rec.speedup_vs_baseline = quality
        if quality >= self.cfg.score_gate:
            self._set_phase(n, "admit_to_pool")
            admitted = self._admit(candidate,
                                   note=f"round {n} (parent iter {base.iteration})")
            rec.admitted = True
            rec.outcome = "admitted"
            self.store.append_timeline("admit", {
                "iteration": n, "kernel_digest": build.digest,
                "parent_iteration": base.iteration,
                "quality": round(quality, 4),
            })
        else:
            self._set_phase(n, "discarded")
            rec.outcome = "discarded"
            rec.notes.append(f"quality {quality:.3f} < gate {self.cfg.score_gate}")
            self.store.append_timeline("discarded", {
                "iteration": n, "kernel_digest": build.digest,
                "quality": round(quality, 4), "reason": "below_score_gate",
            })
        rec.status = "success"
        rec.ended_at = time.time()
        self._close_iteration(rec)

    def _verify_and_repair(self, n: int, rec: IterationRecord, source: str,
                           language: str):
        """Drive a candidate to conformance-clean, repairing up to max_repairs.

        Returns ``(build, report)`` when conformance passes; None when the
        candidate could not be built/launched or is still failing after repairs.
        Never raises: bad candidates discard the round, not the run.
        """
        self._set_phase(n, "verify")
        try:
            build = self._build_and_check(source, language)
            report = self.backend.conformance(self.contract, self.oracle,
                                              build, self.cfg.job_id)
        except Exception as exc:  # noqa: BLE001
            rec.notes.append(f"verify build/launch failed: {exc}")
            return None
        failures = _conformance_failures(report)
        repairs = 0
        while not report.passed and repairs < self.cfg.max_repairs:
            repairs += 1
            self._set_phase(n, "repair")
            rec.repairs = repairs
            try:
                resp = self._agent(
                    n, "repair", tier_for_phase("repair"),
                    prompts.repair_prompt(self.contract, failures, {}))
                source = resp.get("source") or source
                language = resp.get("language") or language
                if not source:
                    rec.notes.append("repair returned no source")
                    return None
                build = self._build_and_check(source, language)
                report = self.backend.conformance(self.contract, self.oracle,
                                                  build, self.cfg.job_id)
                failures = _conformance_failures(report)
            except Exception as exc:  # noqa: BLE001
                rec.notes.append(f"repair {repairs} failed: {exc}")
                return None
        if not report.passed:
            return None
        return build, report

    def _settle(self, rec: IterationRecord, n: int, *, outcome: str,
                note: str) -> None:
        """Close a round that produced no admitted candidate (failed/discarded)."""
        if outcome == "failed":
            rec.status = "failed"
            rec.phase = "optimize"
        else:
            rec.status = "success"
            rec.phase = "discarded"
        rec.outcome = outcome
        rec.notes.append(note)
        self.store.append_timeline(outcome, {
            "iteration": n, "reason": note,
        })
        rec.ended_at = time.time()
        self._close_iteration(rec)

    # ------------------------------------------------------------------ #
    # Pool admission / helpers
    # ------------------------------------------------------------------ #

    def _admit(self, entry: PoolEntry, *, note: str) -> PoolEntry:
        try:
            return self._pool.admit(entry)
        except PoolError as exc:
            raise PipelineError(f"pool admit failed: {exc}") from exc

    def _build_and_check(self, source: str, language: str):
        kernel_dir = self.workspace.root / "candidates"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        return self.backend.build(source, language, self.contract, kernel_dir)

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

def _read_source(entry: PoolEntry) -> str:
    """Best-effort read of an admitted kernel's staged source for re-optimization."""
    if entry.source_path:
        try:
            p = Path(entry.source_path)
            if p.is_file():
                return p.read_text(encoding="utf-8")
        except OSError:
            pass
    return ""   # caller prompts with "<source unavailable>" + digest


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
