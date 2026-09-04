"""Tests for the pool-evolution loop using fake backend/agent.

The loop runs harness_setup (genesis into the pool) then rounds of
select_kernel -> optimize -> verify(+repair) -> admit | discarded. A bad
candidate (no source / no build / launch crash / still-wrong after repairs) must
discard its round, never crash the run.
"""

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
from metainfer.tasks.opt_operator.orchestrator.pool import KernelPool
from metainfer.tasks.opt_operator.orchestrator.profiler import PerfResult
from metainfer.tasks.opt_operator.tests._helpers import contract_dict


def make_contract(**mut):
    return OperatorContract.load(contract_dict(**mut))


def make_oracle(contract):
    return freeze_reference("RMSNorm", contract, "def forward(**t): return t", "generated")


class FakeBackend:
    """Scripted backend: conformance passes; latency improves per profile call."""

    def __init__(self, *, baseline_passed=True, candidate_passed=True,
                 crash_candidate_launch=False, latency_curve=None):
        self.baseline_passed = baseline_passed
        self.candidate_passed = candidate_passed
        self.crash_candidate_launch = crash_candidate_launch
        self.latency_curve = latency_curve or [100.0, 80.0, 60.0, 45.0]
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
        is_baseline = self.conformance_calls == 1
        if not is_baseline and self.crash_candidate_launch:
            raise RuntimeError("candidate failed to launch")
        passed = self.baseline_passed if is_baseline else self.candidate_passed
        results = [CaseResult(c.id, passed, 0.0, 0.0)
                   for c in contract.generate_cases()]
        return ConformanceReport(passed, results, contract.name)

    def profile(self, contract, build, job_id, reps):
        self.profile_calls += 1
        lat = self.latency_curve[min(self._pi, len(self.latency_curve) - 1)]
        self._pi += 1
        return {c.id: PerfResult(c.id, lat) for c in contract.generate_cases()}


def make_runner(calls=None):
    """Agent runner keyed on the new phase names."""
    calls = [] if calls is None else calls

    def runner(phase, tier, prompt, iter_dir, n):
        calls.append((phase, tier, n))
        if phase == "harness_setup":
            return {"language": "triton", "source": "// baseline"}
        if phase == "optimize":
            return {"language": "triton", "source": f"// cand n{n}"}
        if phase == "repair":
            return {"language": "triton", "source": f"// repaired n{n}"}
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
    ledger = ChampionLedger(state_dir / "kernel_pool.jsonl")
    runner_fn, calls = runner or make_runner()
    pipe = Pipeline(
        store=store, workspace=workspace, backend=backend or FakeBackend(),
        agent_runner=runner_fn, ledger=ledger, contract=contract, oracle=oracle,
        initial_source=initial_source, initial_language=initial_language,
        cfg=cfg or PipelineConfig(max_iterations=5),
    )
    return pipe, store, ledger, calls


def _pool(ledger):
    return KernelPool(ledger.pool.path)


# --------------------------------------------------------------------------- #
# harness_setup / genesis
# --------------------------------------------------------------------------- #

def test_harness_setup_admits_genesis(tmp_path):
    # Zero improvement rounds: only harness_setup (genesis admission) runs.
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=0))
    pipe.run()
    pool = _pool(ledger)
    entries = pool.read_all()
    assert len(entries) == 1
    genesis = entries[0]
    assert genesis.iteration == 0
    assert genesis.parent_iteration is None
    assert genesis.kernel_digest == kernel_digest("// src", "triton")
    # provided source -> no harness_setup baseline agent call
    assert not [c for c in calls if c[0] == "harness_setup" and c[2] == 0]
    run = store.load_run()
    assert run.finished is True
    assert run.final_status == "success"
    assert run.current_phase == "finished"


def test_harness_setup_generates_baseline_when_no_source(tmp_path):
    pipe, store, ledger, calls = build_pipeline(tmp_path)
    pipe.run()
    genesis = _pool(ledger).read_all()[0]
    assert genesis.kernel_digest == kernel_digest("// baseline", "triton")
    assert any(c[0] == "harness_setup" for c in calls)


def test_baseline_conformance_failure_raises(tmp_path):
    backend = FakeBackend(baseline_passed=False)
    runner_fn, calls = make_runner()
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, backend=backend, runner=(runner_fn, calls), cfg=PipelineConfig())
    with pytest.raises(Exception):
        pipe.run()


# --------------------------------------------------------------------------- #
# rounds: admit / discard / repair
# --------------------------------------------------------------------------- #

def test_rounds_admit_improving_candidates(tmp_path):
    # latency improves 100 -> 80 -> 60 -> 45, each above the gate -> all admitted.
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=3))
    pipe.run()
    pool = _pool(ledger)
    entries = pool.read_all()
    # genesis + one admission per round
    assert len(entries) == 4
    assert [e.iteration for e in entries] == [0, 1, 2, 3]
    # each admitted candidate's parent is the kernel selected that round (>=0)
    assert all(e.parent_iteration is not None for e in entries[1:])
    # champion is derived: the lowest representative latency (last admitted)
    champ = pool.champion()
    assert champ.iteration == 3
    iters = store.load_all_iterations()
    # genesis (iter 0) + one improving admission per round => 4 admitted records
    assert sum(1 for i in iters if i.get("outcome") == "admitted") == 4


def test_conformance_fail_repairs_then_discards(tmp_path):
    # A single round whose candidate can never pass correctness: repaired up to
    # max_repairs, then discarded (the run survives and genesis stays champion).
    backend = FakeBackend(candidate_passed=False)
    runner, calls = make_runner()
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, backend=backend, runner=(runner, calls),
        initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=1, max_repairs=3))
    pipe.run()
    pool = _pool(ledger)
    # genesis admitted; the failing candidate never enters the pool
    assert len(pool.read_all()) == 1
    # repair agent invoked on the failing candidate (up to max_repairs)
    repairs = [c for c in calls if c[0] == "repair"]
    assert 1 <= len(repairs) <= 3
    iters = store.load_all_iterations()
    assert any(i.get("outcome") == "discarded" for i in iters)


def test_below_gate_candidate_is_discarded(tmp_path):
    # genesis 100; candidate regresses to 200 -> quality 0.5 < gate 1.0 -> discard.
    backend = FakeBackend(latency_curve=[100.0, 200.0, 200.0])
    runner, calls = make_runner()
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, backend=backend, runner=(runner, calls),
        initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=2))
    pipe.run()
    pool = _pool(ledger)
    entries = pool.read_all()
    assert len(entries) == 1          # only genesis ever admitted
    assert entries[0].iteration == 0
    iters = store.load_all_iterations()
    assert any(i.get("outcome") == "discarded" for i in iters)


def test_optimize_no_source_skips_gracefully(tmp_path):
    calls = []

    def runner(phase, tier, prompt, iter_dir, n):
        calls.append((phase, tier, n))
        if phase == "harness_setup":
            return {"language": "triton", "source": "// baseline"}
        if phase == "optimize":
            return {}                # no candidate source
        return {}

    pipe, store, ledger, _ = build_pipeline(
        tmp_path, runner=(runner, calls), cfg=PipelineConfig(max_iterations=3))
    pipe.run()                       # must not raise
    pool = _pool(ledger)
    assert len(pool.read_all()) == 1     # genesis champion kept
    assert store.load_run().finished
    iters = store.load_all_iterations()
    assert len(iters) == 4               # harness_setup + 3 failed rounds
    assert sum(1 for i in iters if i.get("outcome") == "failed") == 3
    # the failed path never reached the benchmark harness
    assert not [c for c in calls if c[0] in ("verify", "repair")]


def test_conformance_launch_crash_skips_gracefully(tmp_path):
    backend = FakeBackend(crash_candidate_launch=True)
    runner, calls = make_runner()
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, backend=backend, runner=(runner, calls),
        initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=3))
    pipe.run()                       # must not raise
    pool = _pool(ledger)
    assert len(pool.read_all()) == 1     # only genesis
    assert store.load_run().finished


def test_tiering_strong_setup_cheap_rounds(tmp_path):
    pipe, store, ledger, calls = build_pipeline(
        tmp_path, initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=2))
    pipe.run()
    for phase, tier, _ in calls:
        if phase == "harness_setup":
            assert tier == "strong", (phase, tier)
        elif phase in ("select_kernel", "optimize", "verify", "repair"):
            assert tier == "cheap", (phase, tier)


# --------------------------------------------------------------------------- #
# resume / reproducibility
# --------------------------------------------------------------------------- #

def test_resume_does_not_rerun_harness_setup(tmp_path):
    # Run 1 admits genesis + round 1.
    state_dir = tmp_path / "state"
    calls1 = []
    runner1, _ = make_runner(calls1)
    pipe1, store, ledger, calls1 = build_pipeline(
        tmp_path, runner=(runner1, calls1), initial_source="// src",
        initial_language="triton", cfg=PipelineConfig(max_iterations=2))
    pipe1.run()
    pool_before = len(_pool(ledger).read_all())
    assert store.load_run().finished is True

    # A second pipeline over the SAME pool, already finished -> idempotent resume.
    ledger2 = ChampionLedger(state_dir / "kernel_pool.jsonl")
    runner2, calls2 = make_runner()
    pipe2 = Pipeline(
        store=store, workspace=IterationWorkspace(tmp_path / "ws", tmp_path / "logs"),
        backend=FakeBackend(), agent_runner=runner2, ledger=ledger2,
        contract=make_contract(shapes={"B": 1, "S": 8, "H": 4}),
        oracle=make_oracle(make_contract(shapes={"B": 1, "S": 8, "H": 4})),
        initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=5))
    pipe2.run(is_resume=True)
    # finished run is not extended; pool unchanged; harness_setup not re-run.
    assert len(_pool(ledger2).read_all()) == pool_before
    assert not [c for c in calls2 if c[0] == "harness_setup"]
    assert store.load_run().finished is True


def _decision_fingerprint(ledger):
    """Pool content that must be seed-reproducible — excludes wall-clock
    admitted_at and per-run workspace source_path (both intended nondeterminism
    embedded in the chain digest), keeps the actual evolution decision."""
    return [
        (e.iteration, e.kernel_digest, e.parent_iteration, e.note,
         tuple(sorted(e.case_latency_ns.items())))
        for e in _pool(ledger).read_all()
    ]


def test_fixed_seed_is_reproducible(tmp_path):
    out = []
    for i in range(2):
        pipe, store, ledger, calls = build_pipeline(
            tmp_path / f"run{i}",
            initial_source="// src", initial_language="triton",
            cfg=PipelineConfig(max_iterations=4, sample_seed=12345))
        pipe.run()
        out.append(_decision_fingerprint(ledger))
    # Same seed -> the same kernels are selected/admitted in the same order.
    assert out[0] == out[1]


def test_resume_continues_from_pool(tmp_path):
    # Run 1 admits genesis (iter 0) + round 1 (iter 1) then stops.
    state_dir = tmp_path / "state"
    calls1 = []
    runner1, _ = make_runner(calls1)
    pipe1, store, ledger, calls1 = build_pipeline(
        tmp_path, runner=(runner1, calls1), initial_source="// src",
        initial_language="triton", cfg=PipelineConfig(max_iterations=1))
    pipe1.run()
    assert [e.iteration for e in _pool(ledger).read_all()] == [0, 1]

    # Simulate an interrupted run: not finished, still at round 1.
    store.update_run(finished=False, current_phase="finished", current_iteration=1)

    # Resume with a fresh Pipeline over the SAME pool file -> continue from round 2.
    ledger2 = ChampionLedger(state_dir / "kernel_pool.jsonl")
    calls2 = []
    runner2, _ = make_runner(calls2)
    pipe2 = Pipeline(
        store=store, workspace=IterationWorkspace(tmp_path / "ws", tmp_path / "logs"),
        backend=FakeBackend(), agent_runner=runner2, ledger=ledger2,
        contract=make_contract(shapes={"B": 1, "S": 8, "H": 4}),
        oracle=make_oracle(make_contract(shapes={"B": 1, "S": 8, "H": 4})),
        initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=3))
    pipe2.run(is_resume=True)
    # rounds 2 and 3 added; harness_setup NOT re-run (no baseline agent call).
    assert [e.iteration for e in _pool(ledger2).read_all()] == [0, 1, 2, 3]
    assert not [c for c in calls2 if c[0] == "harness_setup"]
    assert store.load_run().finished is True
