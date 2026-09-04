"""Tests for the twin harness objects + their recorded口径 metadata."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator.harness import (
    BenchmarkHarness,
    CorrectnessHarness,
    HarnessError,
)
from metainfer.tasks.opt_operator.tests._helpers import contract_dict


def contract(**mut):
    return OperatorContract.load(contract_dict(**mut))


def oracle(origin="generated", digest="abc123"):
    return SimpleNamespace(origin=origin, digest=digest)


class _Sentinel:
    pass


class StubBackend:
    """Records the call and returns a sentinel so we can assert delegation."""

    def __init__(self):
        self.sentinel = _Sentinel()
        self.conformance_calls = 0
        self.profile_calls = 0

    def conformance(self, contract, oracle, build, job_id):
        self.conformance_calls += 1
        return self.sentinel

    def profile(self, contract, build, job_id, reps):
        self.profile_calls += 1
        self.profile_reps = reps
        return self.sentinel


def test_correctness_meta_records_oracle_and_shapes():
    c = contract(shapes={"B": 1, "S": [8, 16], "H": 4})
    h = CorrectnessHarness(c, oracle(origin="library", digest="d9"))
    m = h.meta
    assert m["kind"] == "correctness"
    assert m["shape_count"] == 2
    assert m["oracle_origin"] == "library"
    assert m["oracle_digest"] == "d9"
    assert m["numerics"]["abs_tol"] == 1e-3
    assert m["numerics"]["rel_tol"] == 1e-2


def test_benchmark_meta_records_methodology():
    c = contract(shapes={"B": 1, "S": [8, 16], "H": [4, 8]})
    h = BenchmarkHarness(c, baseline_digest="g0", warmup=3, reps=25,
                         statistic="median")
    m = h.meta
    assert m["kind"] == "benchmark"
    assert m["shape_count"] == 4
    assert m["warmup"] == 3
    assert m["reps"] == 25
    assert m["statistic"] == "median"
    assert m["baseline_digest"] == "g0"


def test_benchmark_defaults():
    h = BenchmarkHarness(contract(shapes={"B": 1, "S": 8, "H": 4}))
    assert h.meta["warmup"] == 2
    assert h.meta["reps"] == 10
    assert h.meta["statistic"] == "median"


def test_benchmark_rejects_unknown_statistic():
    with pytest.raises(HarnessError):
        BenchmarkHarness(contract(shapes={"B": 1, "S": 8, "H": 4}),
                         statistic="mode")


def test_correctness_run_delegates_to_backend():
    c = contract()
    h = CorrectnessHarness(c, oracle())
    backend = StubBackend()
    out = h.run(backend, build="b", job_id="j")
    assert out is backend.sentinel
    assert backend.conformance_calls == 1


def test_benchmark_run_delegates_with_reps():
    c = contract()
    h = BenchmarkHarness(c, reps=17)
    backend = StubBackend()
    out = h.run(backend, build="b", job_id="j")
    assert out is backend.sentinel
    assert backend.profile_calls == 1
    assert backend.profile_reps == 17
