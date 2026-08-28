"""Route tests for the opt_operator server (overview/lineage/iterations)."""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator.ledger import (
    CaseMetric,
    ChampionLedger,
    LedgerEntry,
)
from metainfer.tasks.opt_operator.orchestrator.oracle import freeze_reference, write_oracle_artifacts
from metainfer.tasks.opt_operator.server import routes as routes_mod
from metainfer.tasks.opt_operator.tests._helpers import contract_dict


def _seed(state_dir):
    state_dir.mkdir(parents=True, exist_ok=True)
    state_dir.joinpath("run.json").write_text(json.dumps({
        "task_id": "t", "current_phase": "A_plan", "current_iteration": 1,
        "finished": False,
    }), encoding="utf-8")
    ledger = ChampionLedger(state_dir / "champion_ledger.jsonl")
    ledger.append(LedgerEntry(
        iteration=0, kernel_digest="k0", language="triton", contract_digest="RMSNorm",
        parent_iteration=None, case_metrics={"c": CaseMetric(latency_ns=100.0)}))
    contract = OperatorContract.load(contract_dict(shapes={"B": 1, "S": 8, "H": 4}))
    oracle = freeze_reference("RMSNorm", contract, "def forward(t): return t", "library")
    write_oracle_artifacts(state_dir / "system_oracle" / "run1", oracle)
    (state_dir / "iterations").mkdir()
    (state_dir / "iterations" / "001.json").write_text(json.dumps({
        "iteration": 1, "phase": "A_plan", "status": "success", "promoted": False,
    }), encoding="utf-8")


@pytest.fixture
def seeded_state_dir(tmp_path):
    state_dir = tmp_path / "state"
    _seed(state_dir)
    return state_dir


@pytest.fixture
def client(monkeypatch, seeded_state_dir):
    from types import SimpleNamespace
    from metainfer.tasks.opt_operator.server._qa import CONFIG as QA_CONFIG
    plugin = SimpleNamespace(type="opt-operator", qa_config=QA_CONFIG)
    app = FastAPI()
    app.include_router(routes_mod.build_router(plugin), prefix="/api/opt-operator/{task_id}")
    monkeypatch.setattr(routes_mod, "task_or_404", lambda tid: {"id": tid, "type": "opt-operator"})
    monkeypatch.setattr(routes_mod, "require_task_type", lambda e, t: None)
    monkeypatch.setattr(routes_mod, "state_dir_for", lambda e: seeded_state_dir)
    return TestClient(app)


def test_overview(client):
    r = client.get("/api/opt-operator/t/overview")
    assert r.status_code == 200
    data = r.json()
    assert data["run"]["current_phase"] == "A_plan"
    assert data["reference"]["origin"] == "library"
    assert len(data["lineage"]) == 1
    assert data["summary"]["promotions"] == 0


def test_state_graph(client):
    r = client.get("/api/opt-operator/t/state-graph")
    assert r.status_code == 200
    data = r.json()
    assert data["current"] == "A_plan"


def test_lineage(client):
    r = client.get("/api/opt-operator/t/lineage")
    assert r.status_code == 200
    assert len(r.json()) == 1


def test_iterations(client):
    r = client.get("/api/opt-operator/t/iterations")
    assert r.status_code == 200
    iters = r.json()
    assert len(iters) == 1
    assert iters[0]["iteration"] == 1


def test_iteration_detail(client):
    r = client.get("/api/opt-operator/t/iterations/1")
    assert r.status_code == 200
    assert r.json()["iteration"] == 1
    assert client.get("/api/opt-operator/t/iterations/99").status_code == 404
