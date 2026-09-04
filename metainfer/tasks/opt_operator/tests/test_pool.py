"""Tests for the authoritative kernel pool + derived champion/lineage/sampling."""

from __future__ import annotations

import json
import random

import pytest

from metainfer.tasks.opt_operator.orchestrator.pool import (
    KernelPool,
    PoolEntry,
    PoolError,
    compute_digest,
)


def pe(iteration, kernel, parent=None, latency=10.0, cases=("c0",)):
    return PoolEntry(
        iteration=iteration,
        kernel_digest=kernel,
        language="triton",
        contract_digest="ct",
        parent_iteration=parent,
        case_latency_ns={c: latency for c in cases},
    )


def admit(pool, entry):
    return pool.admit(entry)


def test_append_then_replay(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "k1"))
    admit(pool, pe(2, "k2", parent=1, latency=5.0))

    entries = pool.read_all()
    assert len(entries) == 2
    assert entries[1].prev_digest == entries[0].digest
    assert entries[1].kernel_digest == "k2"
    assert entries[1].case_latency_ns["c0"] == 5.0


def test_cold_restart_derives_champion(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "baseline", latency=10.0))
    admit(pool, pe(2, "better", parent=1, latency=4.0))
    # a slower, non-improving kernel is still admitted (pool holds many kernels)
    admit(pool, pe(3, "slower", parent=1, latency=8.0))

    fresh = KernelPool(tmp_path / "pool.jsonl")
    champ = fresh.champion()
    assert champ is not None and champ.kernel_digest == "better"
    # champion is the min representative latency, not merely the last admitted
    assert champ.iteration == 2


def test_champion_tie_resolves_to_most_recent(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "k1", latency=6.0))
    admit(pool, pe(2, "k2", parent=1, latency=6.0))
    champ = pool.champion()
    assert champ.kernel_digest == "k2"  # tie -> later wins


def test_baseline_and_quality(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "genesis", latency=10.0))
    admit(pool, pe(2, "half", parent=1, latency=5.0))
    admit(pool, pe(3, "quarter", parent=1, latency=2.5))
    base = pool.baseline()
    assert base.kernel_digest == "genesis"
    by_kernel = {e.kernel_digest: e for e in pool.read_all()}
    assert pool.quality(by_kernel["genesis"], base) == pytest.approx(1.0)
    assert pool.quality(by_kernel["half"], base) == pytest.approx(2.0)
    assert pool.quality(by_kernel["quarter"], base) == pytest.approx(4.0)


def test_speedup_vs_baseline_derived(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "genesis", latency=8.0))
    by_kernel = {e.kernel_digest: e for e in pool.read_all()}
    assert pool.speedup_vs_baseline(by_kernel["genesis"]) == pytest.approx(1.0)


def test_lineage_walks_parents_from_champion(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "genesis", latency=10.0))
    admit(pool, pe(2, "a", parent=1, latency=7.0))
    admit(pool, pe(3, "b", parent=2, latency=5.0))
    admit(pool, pe(4, "c", parent=1, latency=4.0))  # best but forks from genesis
    chain = pool.lineage()
    assert [e.iteration for e in chain] == [1, 4]
    assert chain[0].kernel_digest == "genesis"
    assert chain[-1].kernel_digest == "c"


def test_detects_tampered_entry(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "k1"))
    admit(pool, pe(2, "k2", parent=1))
    lines = (tmp_path / "pool.jsonl").read_text().splitlines()
    data = json.loads(lines[0])
    data["kernel_digest"] = "EVIL"
    lines[0] = json.dumps(data)
    (tmp_path / "pool.jsonl").write_text("\n".join(lines) + "\n")
    with pytest.raises(PoolError):
        pool.read_all()


def test_empty_pool(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    assert pool.baseline() is None
    assert pool.champion() is None
    assert pool.lineage() == []
    assert pool.sample_kernel(random.Random(1)) is None


def test_seeded_sampling_is_reproducible(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "genesis", latency=10.0))
    admit(pool, pe(2, "faster", parent=1, latency=2.0))
    admit(pool, pe(3, "slowest", parent=1, latency=50.0))
    seq_a = [pool.sample_kernel(random.Random(123)).kernel_digest for _ in range(40)]
    seq_b = [pool.sample_kernel(random.Random(123)).kernel_digest for _ in range(40)]
    assert seq_a == seq_b


def test_sampling_prefers_higher_quality(tmp_path):
    pool = KernelPool(tmp_path / "pool.jsonl")
    admit(pool, pe(1, "genesis", latency=10.0))
    admit(pool, pe(2, "excellent", parent=1, latency=1.0))
    admit(pool, pe(3, "decent", parent=1, latency=5.0))
    rng = random.Random(7)
    counts = {"genesis": 0, "excellent": 0, "decent": 0}
    for _ in range(2000):
        counts[pool.sample_kernel(rng).kernel_digest] += 1
    # excellent has the highest weight -> clearly the most frequent
    assert counts["excellent"] > counts["decent"]
    assert counts["decent"] > counts["genesis"]
