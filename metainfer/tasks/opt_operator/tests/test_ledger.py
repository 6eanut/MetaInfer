"""Tests for the append-only champion lineage ledger."""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.ledger import (
    ChampionLedger,
    CaseMetric,
    LedgerError,
    LedgerEntry,
)


def entry(iteration, kernel, parent=None, latency=10.0, speedup=1.0):
    return LedgerEntry(
        iteration=iteration,
        kernel_digest=kernel,
        language="triton",
        contract_digest="c0",
        parent_iteration=parent,
        case_metrics={"S2048H128": CaseMetric(latency_ns=latency, speedup=speedup)},
        report_digest="r",
        conformance_digest="conf",
    )


def test_append_then_replay(tmp_path):
    ledger = ChampionLedger(tmp_path / "ledger.jsonl")
    ledger.append(entry(1, "k1", parent=None))
    ledger.append(entry(2, "k2", parent=1, speedup=1.5))

    entries = ledger.read_all()
    assert len(entries) == 2
    assert entries[1].prev_digest == entries[0].digest
    assert entries[1].kernel_digest == "k2"


def test_cold_restart_replay(tmp_path):
    ledger = ChampionLedger(tmp_path / "ledger.jsonl")
    ledger.append(entry(1, "k1"))
    ledger.append(entry(2, "k2", parent=1, speedup=1.2))

    fresh = ChampionLedger(tmp_path / "ledger.jsonl")
    champ = fresh.current_champion()
    assert champ is not None and champ.iteration == 2
    assert champ.parent_iteration == 1


def test_lineage_order(tmp_path):
    ledger = ChampionLedger(tmp_path / "ledger.jsonl")
    ledger.append(entry(1, "k1", parent=None))
    ledger.append(entry(2, "k2", parent=1, speedup=1.5))
    ledger.append(entry(3, "k3", parent=2, speedup=2.0))

    chain = ledger.lineage()
    assert [e.iteration for e in chain] == [1, 2, 3]
    assert chain[0].kernel_digest == "k1"
    assert chain[-1].kernel_digest == "k3"


def test_detects_tampered_entry(tmp_path):
    ledger = ChampionLedger(tmp_path / "ledger.jsonl")
    ledger.append(entry(1, "k1"))
    ledger.append(entry(2, "k2", parent=1))

    # Corrupt the first entry's kernel_digest in-place (keep JSON valid).
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    import json as _json
    data = _json.loads(lines[0])
    data["kernel_digest"] = "EVIL"
    lines[0] = _json.dumps(data)
    (tmp_path / "ledger.jsonl").write_text("\n".join(lines) + "\n")

    with pytest.raises(LedgerError):
        ledger.read_all()


def test_empty_ledger(tmp_path):
    ledger = ChampionLedger(tmp_path / "ledger.jsonl")
    assert ledger.current_champion() is None
    assert ledger.lineage() == []
