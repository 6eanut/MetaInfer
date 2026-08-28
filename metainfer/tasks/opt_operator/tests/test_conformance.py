"""Tests for the numerics-conformance gate (pure-python fake outputs)."""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.conformance import conformance_report
from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
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
