"""Tests for metainfer.cluster.worker_registry."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from metainfer.cluster import paths, worker_registry


def test_register_then_read_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    # Recreate paths module dir state under the new root
    topo = {0: {"uuid": "GPU-aaa", "name": "A100", "total_memory_mib": 40960, "pci_id": "0:1"}}
    rec = worker_registry.register_worker(
        node_id="worker-a",
        ip="10.0.0.1",
        hostname="worker-a.local",
        mac="aa:bb:cc:dd:ee:ff",
        gpu_topology=topo,
    )
    assert rec.boot_id

    out = worker_registry.read_worker("worker-a")
    assert out is not None
    assert out.ip == "10.0.0.1"
    assert out.gpu_topology[0]["uuid"] == "GPU-aaa"


def test_list_workers_deterministic_order(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    for nid in ["c-node", "a-node", "b-node"]:
        worker_registry.register_worker(nid, "ip", "h", "m", {})
    ids = [w.node_id for w in worker_registry.list_workers()]
    assert ids == ["a-node", "b-node", "c-node"]


def test_is_worker_alive_fresh_vs_stale(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    worker_registry.register_worker("w1", "ip", "h", "m", {})
    assert worker_registry.is_worker_alive("w1") is True

    # Backdate heartbeat to simulate stale worker
    hb = paths.worker_heartbeat("w1")
    old = time.time() - 120
    os.utime(hb, (old, old))
    assert worker_registry.is_worker_alive("w1") is False


def test_is_worker_alive_missing_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    assert worker_registry.is_worker_alive("nope") is False


def test_reregister_overwrites_old_record(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    worker_registry.register_worker("w1", "old-ip", "h", "m", {0: {"uuid": "old"}})
    first = worker_registry.read_worker("w1")
    assert first.ip == "old-ip"

    # Re-register with new IP + topology
    worker_registry.register_worker("w1", "new-ip", "h", "m", {0: {"uuid": "new"}, 1: {"uuid": "new2"}})
    second = worker_registry.read_worker("w1")
    assert second.ip == "new-ip"
    assert second.boot_id != first.boot_id, "re-registration must mint a fresh boot_id"
    assert set(second.gpu_topology.keys()) == {0, 1}


def test_cold_restart_invariant(tmp_path: Path, monkeypatch) -> None:
    """WebUI restart: no in-memory state. list_workers() must return same data."""
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    worker_registry.register_worker("w1", "ip1", "h1", "m1", {0: {"uuid": "u1"}})
    worker_registry.register_worker("w2", "ip2", "h2", "m2", {})

    # Snapshot
    snap1 = [w.to_dict() for w in worker_registry.list_workers()]

    # Simulate webui restart: drop any caches and re-read
    # (No caches exist in our impl — this is a contract assertion.)
    snap2 = [w.to_dict() for w in worker_registry.list_workers()]

    assert snap1 == snap2
    assert {w["node_id"] for w in snap2} == {"w1", "w2"}


def test_read_worker_corrupt_returns_none(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    rec_path = paths.worker_record("w1")
    rec_path.parent.mkdir(parents=True, exist_ok=True)
    rec_path.write_text("{broken")
    assert worker_registry.read_worker("w1") is None


def test_gpu_topology_int_keys_preserved_through_json_round_trip(
    tmp_path: Path, monkeypatch
) -> None:
    """GPU indices must survive json string-key encoding back to int."""
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    topo = {0: {"uuid": "a"}, 7: {"uuid": "b"}}
    worker_registry.register_worker("w", "ip", "h", "m", topo)
    # Inspect raw JSON to confirm string keys on disk
    on_disk = json.loads(paths.worker_record("w").read_text())
    assert set(on_disk["gpu_topology"].keys()) == {"0", "7"}
    # And reader restores int keys
    rec = worker_registry.read_worker("w")
    assert set(rec.gpu_topology.keys()) == {0, 7}


def test_register_rejects_hostname_drift(tmp_path: Path, monkeypatch) -> None:
    """Regression: a daemon with a mis-set METAINFER_NODE_ID must NOT clobber
    the real worker's record. Previously register_worker was unconditional
    overwrite — a daemon on host B with METAINFER_NODE_ID=A would silently
    overwrite A's hostname/IP/MAC with B's, corrupting PP2 rendezvous and
    scoreboard slot ownership. The fix: refuse the second registration."""
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    # Legit first registration: worker21 on host worker21
    worker_registry.register_worker(
        "worker21", ip="10.18.17.66", hostname="worker21",
        mac="aa:bb:cc:dd:ee:66", gpu_topology={},
    )
    # Imposter: a daemon on host worker25 self-identifies as worker21
    with pytest.raises(worker_registry.WorkerIdentityConflict) as ei:
        worker_registry.register_worker(
            "worker21", ip="10.18.17.73", hostname="worker25",
            mac="aa:bb:cc:dd:ee:73", gpu_topology={},
        )
    assert "worker21" in str(ei.value)
    assert "worker25" in str(ei.value)
    # SSOT preserved: worker21.json still says worker21/10.18.17.66
    rec = worker_registry.read_worker("worker21")
    assert rec.hostname == "worker21"
    assert rec.ip == "10.18.17.66"
    assert rec.mac == "aa:bb:cc:dd:ee:66"
    # Conflict forensics written for operator audit
    conflicts = list(paths.workers_dir().glob("worker21.conflict.*.json"))
    assert len(conflicts) == 1
    payload = json.loads(conflicts[0].read_text())
    assert payload["existing_record"]["hostname"] == "worker21"
    assert payload["attempted_overwrite"]["hostname"] == "worker25"


def test_register_allows_cold_restart_same_hostname(tmp_path: Path, monkeypatch) -> None:
    """Cold restart on the same box is the legit re-registration path — must
    not be blocked by the identity-drift check."""
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    worker_registry.register_worker(
        "worker21", ip="10.18.17.66", hostname="worker21",
        mac="aa:bb:cc:dd:ee:66", gpu_topology={0: {"uuid": "x"}},
    )
    # Daemon restarts on same box: same hostname, new boot_id
    worker_registry.register_worker(
        "worker21", ip="10.18.17.66", hostname="worker21",
        mac="aa:bb:cc:dd:ee:66", gpu_topology={0: {"uuid": "x"}},
    )
    rec = worker_registry.read_worker("worker21")
    assert rec.hostname == "worker21"


def test_register_allows_explicit_rehome_after_json_delete(
    tmp_path: Path, monkeypatch
) -> None:
    """If a box is genuinely re-homed (new hostname for the same node_id),
    the operator path is: delete the JSON, then re-register. This keeps the
    re-homing visible in filesystem history rather than silent."""
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    worker_registry.register_worker(
        "worker21", ip="10.18.17.66", hostname="worker21",
        mac="aa:bb:cc:dd:ee:66", gpu_topology={},
    )
    # Explicit operator action: clear the record
    paths.worker_record("worker21").unlink()
    # Now re-registration with a different hostname is allowed
    worker_registry.register_worker(
        "worker21", ip="10.18.17.99", hostname="worker21-new",
        mac="aa:bb:cc:dd:ee:99", gpu_topology={},
    )
    rec = worker_registry.read_worker("worker21")
    assert rec.hostname == "worker21-new"
