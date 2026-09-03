"""Tests for the numerics-conformance gate (pure-python fake outputs)."""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.conformance import conformance_report
from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator._compare import within_tol
from metainfer.tasks.opt_operator.tests._helpers import _fill, contract_dict


def make_contract(**mut):
    return OperatorContract.load(contract_dict(**mut))


def oracle_outputs(contract, base=1.0):
    out = {}
    for case in contract.generate_cases():
        out[case.id] = {
            t.name: _fill(t.resolved_shape(case.dims), base=base)
            for t in contract.outputs
        }
    return out


def candidate_like(oracle, scale=1.0):
    # candidate = oracle * scale (plus a tiny shift), within/without tolerance.
    cand = {}
    for cid, tensors in oracle.items():
        cand[cid] = {k: _scale(v, scale) for k, v in tensors.items()}
    return cand


def _scale(v, scale):
    if isinstance(v, list):
        return [_scale(x, scale) for x in v]
    return v * scale


def test_all_pass_within_tolerance():
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4}, numerics={"abs_tol": 1e-3, "rel_tol": 1e-2})
    oracle = oracle_outputs(c)
    cand = candidate_like(oracle, scale=1.0 + 1e-6)
    report = conformance_report(c, oracle, cand)
    assert report.passed
    assert all(r.passed for r in report.results)
    assert len(report.results) == 1


def test_fails_when_out_of_tolerance():
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4}, numerics={"abs_tol": 1e-3, "rel_tol": 1e-2})
    oracle = oracle_outputs(c)
    cand = candidate_like(oracle, scale=2.0)  # rel error 100% >> 1%
    report = conformance_report(c, oracle, cand)
    assert not report.passed
    r = report.result(list(oracle)[0])
    assert r.max_rel_err > 1e-2


def test_missing_candidate_case_fails():
    c = make_contract(shapes={"B": 1, "S": [8, 16], "H": 4})
    oracle = oracle_outputs(c)
    cand = {list(oracle)[0]: oracle[list(oracle)[0]]}  # only one of two cases
    report = conformance_report(c, oracle, cand)
    assert not report.passed


def test_missing_output_key_fails():
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4})
    oracle = oracle_outputs(c)
    cand = candidate_like(oracle)
    for cid in cand:
        del cand[cid]["Y"]
        break
    report = conformance_report(c, oracle, cand)
    assert not report.passed


def test_wrong_shape_fails():
    c = make_contract(shapes={"B": 1, "S": 8, "H": 4})
    oracle = oracle_outputs(c)
    cand = candidate_like(oracle)
    for cid in cand:
        cand[cid]["Y"] = [1.0, 2.0, 3.0]  # wrong rank
        break
    report = conformance_report(c, oracle, cand)
    assert not report.passed


def test_multi_case_all_must_pass():
    c = make_contract(shapes={"B": 1, "S": [8, 16], "H": [4, 8]})  # 4 cases
    oracle = oracle_outputs(c)
    cand = candidate_like(oracle)
    # Break exactly one case.
    one = list(cand)[0]
    cand[one] = {k: _scale(v, 5.0) for k, v in cand[one].items()}
    report = conformance_report(c, oracle, cand)
    assert not report.passed
    assert sum(1 for r in report.results if r.passed) == 3


# --- Per-element gate semantics ----------------------------------------------
# A case must FAIL only when a *single element* violates both abs_tol AND
# rel_tol. The global max-abs and global max-rel can land on different elements
# (a large-magnitude element with a big abs error, plus a near-zero element with
# a big *relative* error, e.g. fp16-subnormal noise). Coupling the two global
# maxima wrongly rejects such a correct candidate. These tests pin that fix.

def test_within_tol_passes_when_maxima_on_different_elements():
    # element0: abs err 30 (>abs_tol 10) but rel 0.03 (<rel_tol 0.05) -> ok
    # element1: abs err 0.001 (<abs_tol 10) but rel 1.0 (>rel_tol)    -> ok
    # (rel_tol violation only counts if abs ALSO exceeds abs_tol)
    assert within_tol([1030.0, 0.002], [1000.0, 0.001], abs_tol=10.0, rel_tol=0.05)


def test_within_tol_fails_when_single_element_violates_both():
    # element0: abs err 300 (>10) AND rel 0.3 (>0.05) on the SAME element.
    assert not within_tol([1300.0, 0.002], [1000.0, 0.001], abs_tol=10.0, rel_tol=0.05)


def test_conformance_passes_when_maxima_on_different_elements():
    c = make_contract(shapes={"B": 1, "S": 1, "H": 2},
                      numerics={"abs_tol": 10.0, "rel_tol": 0.05})
    case = list(c.generate_cases())[0]
    oracle = {case.id: {"Y": [[[1000.0, 0.001]]]}}
    cand = {case.id: {"Y": [[[1030.0, 0.002]]]}}
    report = conformance_report(c, oracle, cand)
    assert report.passed
    r = report.result(case.id)
    # The report still surfaces the global maxima for visibility...
    assert r.max_abs_err > 10.0
    assert r.max_rel_err > 0.05
    # ...but no single element violates both tolerances, so the case passes.
    assert r.passed
    assert r.detail == ""


def test_conformance_fails_when_single_element_violates_both():
    c = make_contract(shapes={"B": 1, "S": 1, "H": 2},
                      numerics={"abs_tol": 10.0, "rel_tol": 0.05})
    case = list(c.generate_cases())[0]
    oracle = {case.id: {"Y": [[[1000.0, 0.001]]]}}
    cand = {case.id: {"Y": [[[1300.0, 0.002]]]}}  # element0: abs 300 & rel 0.3
    report = conformance_report(c, oracle, cand)
    assert not report.passed
    r = report.result(case.id)
    assert not r.passed
    assert r.detail != ""
