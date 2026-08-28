"""Tests for oracle freeze / digest / cold-restart (no reference execution)."""

from __future__ import annotations

import json

import pytest

from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator.oracle import (
    FrozenOracle,
    OracleError,
    digest_reference,
    freeze_reference,
    load_oracle,
    write_oracle_artifacts,
)
from metainfer.tasks.opt_operator.tests._helpers import contract_dict


SRC = "def forward(**tensors):\n    return {k: v for k, v in tensors.items()}\n"


@pytest.fixture()
def contract() -> OperatorContract:
    return OperatorContract.load(contract_dict(shapes={"B": 1, "S": 8, "H": 4}))


def test_digest_is_stable_and_content_sensitive():
    d1 = digest_reference("rmsnorm", contract_dict(), SRC)
    d2 = digest_reference("rmsnorm", contract_dict(), SRC)
    assert d1 == d2
    assert d1 != digest_reference("rmsnorm", contract_dict(), SRC + "# x")
    assert d1 != digest_reference("rmsnorm2", contract_dict(), SRC)


def test_freeze_builds_consistent_oracle(contract):
    o = freeze_reference("rmsnorm", contract, SRC, "user")
    assert o.origin == "user"
    assert o.digest == digest_reference("rmsnorm", contract.to_dict(), SRC)
    assert o.contract == contract.to_dict()


def test_frozen_oracle_rejects_digest_mismatch(contract):
    with pytest.raises(OracleError):
        FrozenOracle(
            op_id="rmsnorm",
            contract=contract.to_dict(),
            reference_source=SRC,
            origin="user",
            digest="0" * 64,
            created_at=1.0,
        )


def test_round_trip_and_cold_restart(tmp_path, contract):
    oracle = freeze_reference("rmsnorm", contract, SRC, "library")
    write_oracle_artifacts(tmp_path, oracle)

    reloaded = load_oracle(tmp_path)
    assert reloaded is not None
    assert reloaded.digest == oracle.digest
    assert reloaded.contract == oracle.contract
    assert reloaded.reference_source == SRC
    assert reloaded.origin == "library"


def test_load_oracle_none_when_missing(tmp_path):
    assert load_oracle(tmp_path / "nope") is None


def test_load_oracle_detects_tampering(tmp_path, contract):
    oracle = freeze_reference("rmsnorm", contract, SRC, "user")
    write_oracle_artifacts(tmp_path, oracle)
    # Corrupt the authoritative descriptor so the digest no longer matches.
    path = tmp_path / "oracle.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["digest"] = "f" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(OracleError):
        load_oracle(tmp_path)
