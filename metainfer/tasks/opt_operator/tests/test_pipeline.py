"""Tests for the optimization loop (S_baseline + A→…→F) using fake backend/agent."""

from __future__ import annotations

from pathlib import Path

import pytest

from metainfer.orchestrator.iteration import IterationWorkspace
from metainfer.orchestrator.state import StateStore

from metainfer.tasks.opt_operator.orchestrator.build import BuildResult, kernel_digest
from metainfer.tasks.opt_operator.orchestrator.conformance import (
    CaseResult,
    ConformanceReport,
)
from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator.ledger import ChampionLedger
from metainfer.tasks.opt_operator.orchestrator.oracle import freeze_reference
from metainfer.tasks.opt_operator.orchestrator.pipeline import Pipeline, PipelineConfig
from metainfer.tasks.opt_operator.orchestrator.profiler import PerfResult
from metainfer.tasks.opt_operator.tests._helpers import contract_dict


def make_contract(**mut):
    return OperatorContract.load(contract_dict(**mut))


def make_oracle(contract):
    return freeze_reference("RMSNorm", contract, "def forward(**t): return t", "generated")


class FakeBackend:
    """Scripted backend: conformance passes, latency improves per profile call."""

    def __init__(self, *, baseline_passed=True, candidate_passed=True,
                 latency_curve=None):
        self.baseline_passed = baseline_passed
        self.candidate_passed = candidate_passed
        self.latency_curve = latency_curve or [100.0, 80.0, 60.0]
        self._pi = 0
        self.builds = []
        self.conformance_calls = 0
        self.profile_calls = 0

    def build(self, source, language, contract, kernel_dir):
        r = BuildResult(language=language, artifact=str(kernel_dir / "k"),
                        workspace_dir=kernel_dir,
                        digest=kernel_digest(source, language))
        self.builds.append(source)
        return r

    def conformance(self, contract, oracle, build, job_id):
        self.conformance_calls += 1
        passed = self.baseline_passed if self.conformance_calls == 1 \
            else self.candidate_passed
        results = [CaseResult(c.id, passed, 0.0, 0.0)
                   for c in contract.generate_cases()]
        return ConformanceReport(passed, results, contract.name)

    def profile(self, contract, build, job_id, reps):
        self.profile_calls += 1
        lat = self.latency_curve[min(self._pi, len(self.latency_curve) - 1)]
        self._pi += 1
        return {c.id: PerfResult(c.id, lat) for c in contract.generate_cases()}


def make_runner(stop_after_cycle=2):
    calls = []

    def runner(phase, tier, prompt, iter_dir, n):
        calls.append((phase, tier, n))
        if phase == "S_baseline":
            return {"language": "triton", "source": "// baseline"}
        if phase == "A_plan":
            return {"approach": "tile inner loop", "detail": "x", "done": False}
        if phase == "B_implement":
            return {"language": "triton", "source": f"// cand n{n}"}
        if phase == "D_review":
            return {"guidance": "looks correct"}
        if phase == "F_perf_plan":
            return {"next_plan": "keep going", "done": n >= stop_after_cycle}
        return {}

    return runner, calls


def build_pipeline(tmp_path, *, contract=None, backend=None, runner=None,
                   initial_source=None, initial_language=None, cfg=None):
    contract = contract or make_contract(shapes={"B": 1, "S": 8, "H": 4})
    state_dir = tmp_path / "state"
    store = StateStore(state_dir)
    store.init_or_resume("task")
    workspace = IterationWorkspace(tmp_path / "ws", tmp_path / "logs")
    oracle = make_oracle(contract)
    ledger = ChampionLedger(tmp_path / "ledger.jsonl")
    runner_fn, calls = runner or make_runner()
    pipe = Pipeline(
        store=store, workspace=workspace, backend=backend or FakeBackend(),
        agent_runner=runner_fn, ledger=ledger, contract=contract, oracle=oracle,
        initial_source=initial_source, initial_language=initial_language,
        cfg=cfg or PipelineConfig(max_iterations=5),
    )
    return pipe, store, ledger, calls


def test_s_baseline_promotes_genesis(tmp_path):
    pipe, store, ledger, calls = build_pipeline(tmp_path, initial_source="// src",
                                                initial_language="triton")
    pipe.run()
    genesis = ledger.lineage()[0]
    assert genesis.iteration == 0
    assert genesis.parent_iteration is None
    assert genesis.kernel_digest == kernel_digest("// src", "triton")
    # S_baseline certified a provided source -> no S_baseline agent call
    assert not [c for c in calls if c[0] == "S_baseline"]


def test_s_baseline_generates_when_no_source(tmp_path):
    pipe, store, ledger, calls = build_pipeline(tmp_path)
    pipe.run()
    genesis = ledger.lineage()[0]
    assert genesis.kernel_digest == kernel_digest("// baseline", "triton")
    assert any(c[0] == "S_baseline" for c in calls)


def test_tiering_strong_vs_cheap(tmp_path):
    pipe, store, ledger, calls = build_pipeline(tmp_path, initial_source="// src",
                                                initial_language="triton")
    pipe.run()
    for phase, tier, _ in calls:
        if phase in ("A_plan", "D_review", "F_perf_plan"):
            assert tier == "strong", (phase, tier)
        elif phase == "B_implement":
            assert tier == "cheap", (phase, tier)


def test_promotion_and_regression(tmp_path):
    # Latency improves then regresses on the second cycle -> no second promotion.
    backend = FakeBackend(latency_curve=[100.0, 80.0, 120.0])
    runner, calls = make_runner(stop_after_cycle=3)
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, backend=backend, runner=(runner, calls), initial_source="// src",
        initial_language="triton", cfg=PipelineConfig(max_iterations=5))
    pipe.run()
    chain = ledger.lineage()
    # genesis + cycle-1 promotion (80 < 100); cycle-2 regresses (120 > 80) -> no promote
    assert len(chain) == 2
    assert chain[0].iteration == 0
    assert chain[1].iteration == 1


def test_conformance_fail_blocks_promotion(tmp_path):
    backend = FakeBackend(candidate_passed=False)
    runner, calls = make_runner(stop_after_cycle=2)
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, backend=backend, runner=(runner, calls), initial_source="// src",
        initial_language="triton", cfg=PipelineConfig(max_iterations=3))
    pipe.run()
    # genesis still promotes (baseline passed); candidate never promotes
    assert len(ledger.lineage()) == 1
    # repair agent should have been invoked (max_repairs times per cycle)
    repairs = [c for c in calls if c[0] == "B_implement"]
    assert len(repairs) > 0


def test_baseline_conformance_failure_raises(tmp_path):
    backend = FakeBackend(baseline_passed=False)
    runner_fn, calls = make_runner()
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, backend=backend, runner=(runner_fn, calls), cfg=PipelineConfig())
    with pytest.raises(Exception):
        pipe.run()


def test_finishes_and_sets_run_status(tmp_path):
    pipe, store, ledger, calls = build_pipeline(tmp_path, initial_source="// src",
                                                initial_language="triton")
    pipe.run()
    run = store.load_run()
    assert run.finished is True
    assert run.final_status == "success"
    assert run.current_phase == "finished"
