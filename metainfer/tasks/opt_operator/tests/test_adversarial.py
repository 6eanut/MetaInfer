"""Tests for adversarial harness review + negative-case evidence (FR-2/3)."""

from __future__ import annotations

from metainfer.tasks.opt_operator.orchestrator.adversarial import (
    review_benchmark,
    review_correctness,
)
from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.tests._helpers import _fill, contract_dict


def contract(**mut):
    return OperatorContract.load(contract_dict(**mut))


def oracle_outputs(contract, base=1.0):
    out = {}
    for case in contract.generate_cases():
        out[case.id] = {
            t.name: _fill(t.resolved_shape(case.dims), base=base)
            for t in contract.outputs
        }
    return out


# --------------------------------------------------------------------------- #
# Correctness review
# --------------------------------------------------------------------------- #

def test_correctness_review_passes_on_honest_oracle():
    c = contract(shapes={"B": 1, "S": [8, 16], "H": 4})
    review = review_correctness(c, oracle_outputs(c))
    assert review.passed
    assert review.kind == "correctness"
    # at least one concrete negative case is left on the record
    assert review.negative_evidence
    assert all(e.harness_caught for e in review.negative_evidence)


def test_correctness_review_negative_evidence_kinds():
    c = contract(shapes={"B": 1, "S": [8, 16], "H": 4})
    review = review_correctness(c, oracle_outputs(c))
    names = {e.name for e in review.negative_evidence}
    assert {"element_bias", "one_case_scaled", "wrong_shape"}.issubset(names)
    assert "missing_case" in names


def test_correctness_review_catches_lenient_gate():
    # Tolerances so large that a +1000 element bias / 100x scale is "within tol":
    # the gate would accept a wrong kernel — the adversarial review must FAIL it.
    c = contract(shapes={"B": 1, "S": 8, "H": 4},
                 numerics={"abs_tol": 1e12, "rel_tol": 1e12})
    review = review_correctness(c, oracle_outputs(c))
    assert not review.passed
    assert any(not e.harness_caught for e in review.negative_evidence)


def test_correctness_review_empty_outputs_fails():
    c = contract(shapes={"B": 1, "S": 8, "H": 4})
    review = review_correctness(c, {})
    assert not review.passed


# --------------------------------------------------------------------------- #
# Benchmark review (methodology authority)
# --------------------------------------------------------------------------- #

def valid_meta(**mut):
    m = {
        "kind": "benchmark",
        "shape_ids": ["c0", "c1"],
        "warmup": 2,
        "reps": 10,
        "statistic": "median",
        "baseline_digest": "g0",
    }
    m.update(mut)
    return m


def test_benchmark_review_passes_valid_meta():
    review = review_benchmark(valid_meta(), correctness_shape_ids=["c0", "c1"])
    assert review.passed


def test_benchmark_review_fails_without_reps():
    review = review_benchmark(valid_meta(reps=0), correctness_shape_ids=["c0", "c1"])
    assert not review.passed
    assert any(c.name == "reps_specified" and not c.passed for c in review.checks)


def test_benchmark_review_fails_shape_misalignment():
    review = review_benchmark(valid_meta(shape_ids=["c0"]),
                              correctness_shape_ids=["c0", "c1"])
    assert not review.passed
    assert any(c.name == "shape_aligned_with_correctness" and not c.passed
               for c in review.checks)


def test_benchmark_review_fails_without_baseline():
    review = review_benchmark(valid_meta(baseline_digest=None),
                              correctness_shape_ids=["c0", "c1"])
    assert not review.passed
    assert any(c.name == "baseline_defined" and not c.passed for c in review.checks)
