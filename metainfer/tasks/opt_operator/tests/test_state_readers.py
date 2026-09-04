"""Tests for the opt_operator web state readers."""

from __future__ import annotations

import json

import pytest

from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator.ledger import (
    CaseMetric,
    ChampionLedger,
    LedgerEntry,
)
from metainfer.tasks.opt_operator.orchestrator.oracle import freeze_reference, write_oracle_artifacts
from metainfer.tasks.opt_operator.server import _state_readers
from metainfer.tasks.opt_operator.tests._helpers import contract_dict


def _contract():
    return OperatorContract.load(contract_dict(shapes={"B": 1, "S": 8, "H": 4}))


def _seed_ledger(state_dir):
    # Authoritative pool file; ChampionLedger is the derived lineage view.
    ledger = ChampionLedger(state_dir / "kernel_pool.jsonl")
    ledger.append(LedgerEntry(
        iteration=0, kernel_digest="k0", language="triton", contract_digest="RMSNorm",
        parent_iteration=None,
        case_metrics={"S8H4": CaseMetric(latency_ns=100.0, speedup=1.0)}))
    ledger.append(LedgerEntry(
        iteration=1, kernel_digest="k1", language="triton", contract_digest="RMSNorm",
        parent_iteration=0,
        case_metrics={"S8H4": CaseMetric(latency_ns=80.0, speedup=1.25)}))
    return ledger


def _seed_state_dir(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.joinpath("run.json").write_text(json.dumps({
        "task_id": "t", "current_phase": "verify",
        "current_iteration": 2, "finished": False,
    }), encoding="utf-8")
    _seed_ledger(state_dir)
    (state_dir / "iterations").mkdir()
    (state_dir / "iterations" / "002.json").write_text(json.dumps({
        "iteration": 2, "phase": "admit_to_pool", "status": "success",
        "outcome": "admitted", "admitted": True, "candidate_language": "triton",
        "conformance": {"passed": True, "contract_name": "RMSNorm", "results": []},
        "perf": {"S8H4": {"latency_ns": 80.0}},
    }), encoding="utf-8")
    # frozen oracle artifact
    contract = _contract()
    oracle = freeze_reference("RMSNorm", contract, "def forward(t): return t", "generated")
    write_oracle_artifacts(state_dir / "system_oracle" / "run1", oracle)
    return state_dir


def test_read_run(tmp_path):
    state_dir = _seed_state_dir(tmp_path)
    run = _state_readers.read_run(state_dir)
    assert run["current_phase"] == "verify"
    assert run["current_iteration"] == 2


def test_read_state_graph(tmp_path):
    state_dir = _seed_state_dir(tmp_path)
    g = _state_readers.read_state_graph(state_dir)
    assert g["current"] == "verify"
    by_id = {n["id"]: n for n in g["nodes"]}
    assert by_id["verify"]["state"] == "current"
    assert by_id["harness_setup"]["state"] == "done"
    assert by_id["finished"]["state"] == "pending"
    assert g["active_edge"]["to"] == "verify"


def test_read_lineage(tmp_path):
    state_dir = _seed_state_dir(tmp_path)
    lineage = _state_readers.read_lineage(state_dir)
    assert [e["iteration"] for e in lineage] == [0, 1]
    # genesis best latency 100; champion 80 -> speedup 1.25
    assert lineage[-1]["speedup_vs_genesis"] == 1.25
    assert lineage[0]["speedup_vs_genesis"] is None


def test_read_overview(tmp_path):
    state_dir = _seed_state_dir(tmp_path)
    ov = _state_readers.read_overview(state_dir)
    assert ov["run"]["current_phase"] == "verify"
    assert ov["reference"]["origin"] == "generated"
    assert ov["reference"]["op_id"] == "RMSNorm"
    assert ov["summary"]["promotions"] == 1
    assert ov["summary"]["speedup_vs_genesis"] == 1.25
    assert isinstance(ov["gpu_pool"], list)


def test_read_iterations(tmp_path):
    state_dir = _seed_state_dir(tmp_path)
    iters = _state_readers.read_iterations(state_dir)
    assert len(iters) == 1
    assert iters[0]["iteration"] == 2


def test_read_conformance(tmp_path):
    state_dir = _seed_state_dir(tmp_path)
    conf = _state_readers.read_conformance(state_dir, 2)
    assert conf["admitted"] is True
    assert conf["outcome"] == "admitted"
    assert conf["conformance"]["passed"] is True
    assert _state_readers.read_conformance(state_dir, 99) is None


def test_overview_missing_run(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    ov = _state_readers.read_overview(state_dir)
    assert ov["run"]["current_phase"] == "idle"
    assert ov["lineage"] == []
    assert ov["reference"] == {}


def test_read_pool_lists_all_kernels_with_derived_quality(tmp_path):
    state_dir = _seed_state_dir(tmp_path)  # pool: k0 (100ns genesis), k1 (80ns)
    pl = _state_readers.read_pool(state_dir)
    assert pl["pool_size"] == 2
    assert pl["baseline_rep_latency_ns"] == pytest.approx(100.0)
    rows = {r["iteration"]: r for r in pl["pool"]}
    assert rows[0]["is_champion"] is False
    assert rows[0]["quality"] == pytest.approx(1.0)   # genesis == baseline
    assert rows[1]["quality"] == pytest.approx(1.25)  # 100 / 80
    assert rows[1]["speedup_vs_baseline"] == pytest.approx(1.25)
    assert rows[1]["is_champion"] is True
    assert rows[1]["on_lineage"] is True
    # top-quality first
    assert pl["pool"][0]["iteration"] == 1


def test_read_pool_empty(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    pl = _state_readers.read_pool(state_dir)
    assert pl["pool"] == []
    assert pl["pool_size"] == 0


def test_read_harness_reviews_aggregates_negative_evidence(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "iterations").mkdir()
    (state_dir / "iterations" / "000.json").write_text(json.dumps({
        "iteration": 0, "phase": "harness_setup", "status": "success",
        "reviews": {
            "correctness": {
                "passed": True, "kind": "correctness", "message": "",
                "checks": [{"name": "negative_cases_caught", "passed": True}],
                "negative_evidence": [
                    {"name": "element_bias", "constructed": "biased one element",
                     "harness_caught": True, "detail": ""},
                    {"name": "one_case_scaled", "constructed": "scaled a case",
                     "harness_caught": False, "detail": "gate MISSED it"},
                ],
            },
            "benchmark": {
                "passed": True, "kind": "benchmark", "message": "",
                "checks": [{"name": "warmup_specified", "passed": True,
                            "detail": "warmup=3"}],
            },
        },
    }), encoding="utf-8")
    h = _state_readers.read_harness_reviews(state_dir)
    assert h["present"] is True
    assert h["iteration"] == 0
    assert h["correctness"]["negatives_total"] == 2
    assert h["correctness"]["negatives_caught"] == 1
    assert h["correctness"]["passed"] is True
    assert h["benchmark"]["passed"] is True
    assert h["benchmark"]["checks"][0]["detail"] == "warmup=3"


def test_read_harness_reviews_absent(tmp_path):
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    h = _state_readers.read_harness_reviews(state_dir)
    assert h["present"] is False
