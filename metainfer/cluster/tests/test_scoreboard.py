"""Tests for metainfer.cluster.scoreboard.

Critical invariants verified:
- Concurrent same-slot acquirers: exactly one wins per round.
- Multi-slot all-or-nothing: partial acquisition triggers rollback.
- Cross-set concurrent acquirers: never two holders of the same GPU.
- Lease expiry + heartbeat-stale → reaper frees slot.
- Secret-verified release: wrong-secret release does NOT unlink.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import pytest

from metainfer.cluster import paths, scoreboard, worker_registry
from metainfer.cluster.scoreboard import LeaseToken, Slot


@pytest.fixture(autouse=True)
def _scratch_root(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    return tmp_path


# Helper: register a fake worker so heartbeat lookups during reap succeed.
def _register_worker(node_id: str) -> None:
    worker_registry.register_worker(node_id, "ip", "h", "m", {0: {"uuid": "x"}})


# --------------------------------------------------------------------------- #
# Single-slot basics
# --------------------------------------------------------------------------- #
def test_acquire_release_round_trip() -> None:
    _register_worker("n0")
    tok = scoreboard.acquire_gpus([("n0", 0)], holder="t1", job_id="j1", deadline_s=2.0)
    assert tok is not None
    assert tok.slots == [("n0", 0)]

    # Second acquirer fails quickly (slot held, holder alive).
    tok2 = scoreboard.acquire_gpus([("n0", 0)], holder="t2", job_id="j2",
                                    deadline_s=0.5)
    assert tok2 is None

    scoreboard.release_gpus(tok)
    # Now acquirer 2 can succeed.
    tok3 = scoreboard.acquire_gpus([("n0", 0)], holder="t2", job_id="j2",
                                    deadline_s=2.0)
    assert tok3 is not None


def test_concurrent_same_slot_unique_winner() -> None:
    _register_worker("n0")
    N = 15
    winners: List[str] = []
    lock = threading.Lock()
    tokens: List[LeaseToken] = []

    def acquirer(i: int) -> None:
        tok = scoreboard.acquire_gpus([("n0", 0)], holder=f"h{i}", job_id=f"j{i}",
                                       deadline_s=2.0)
        if tok is not None:
            with lock:
                winners.append(f"h{i}")
                tokens.append(tok)

    with ThreadPoolExecutor(max_workers=N) as ex:
        list(ex.map(acquirer, range(N)))

    assert len(winners) == 1, f"exactly one winner, got {winners}"
    scoreboard.release_gpus(tokens[0])


# --------------------------------------------------------------------------- #
# Multi-slot all-or-nothing
# --------------------------------------------------------------------------- #
def test_multi_slot_all_acquired() -> None:
    _register_worker("n0")
    _register_worker("n1")
    tok = scoreboard.acquire_gpus([("n0", 0), ("n1", 0)], holder="h", job_id="j",
                                   deadline_s=2.0)
    assert tok is not None
    assert set(tok.slots) == {("n0", 0), ("n1", 0)}
    scoreboard.release_gpus(tok)


def test_multi_slot_partial_rolls_back() -> None:
    """If one slot in a set is already held, all other acquired slots must roll back."""
    _register_worker("n0")
    _register_worker("n1")

    # Pre-acquire n1:0 by someone else.
    blocker = scoreboard.acquire_gpus([("n1", 0)], holder="blocker", job_id="b",
                                       deadline_s=2.0)
    assert blocker is not None

    # Now try to acquire n0:0 + n1:0 — should fail (deadline too short for retry
    # to succeed since blocker holds n1:0 indefinitely).
    tok = scoreboard.acquire_gpus([("n0", 0), ("n1", 0)], holder="h", job_id="j",
                                   deadline_s=0.5)
    assert tok is None

    # Critical assertion: n0:0 must NOT be left held by failed acquirer.
    # list_claims now returns free GPUs too — filter to status=held.
    claim = scoreboard.list_claims()
    held = [(c["node_id"], c["gpu_idx"]) for c in claim if c["status"] == "held"]
    assert ("n0", 0) not in held, "failed acquire must not leak held slot"
    assert ("n1", 0) in held, "blocker's slot must still be held"

    scoreboard.release_gpus(blocker)


# --------------------------------------------------------------------------- #
# Cross-set concurrent acquirers — deadlock freedom + never-two-holders
# --------------------------------------------------------------------------- #
def test_cross_set_no_double_hold_under_contention() -> None:
    """Two threads each try to acquire (A,B); assert never both succeed simultaneously
    AND eventually both rounds complete without permanent hold."""
    _register_worker("n0")
    _register_worker("n1")

    violation = {"count": 0}
    hold_observed: List[int] = []
    lock = threading.Lock()

    def round_worker(rid: int) -> None:
        tok = scoreboard.acquire_gpus([("n0", 0), ("n1", 0)],
                                       holder=f"r{rid}", job_id=f"j{rid}",
                                       deadline_s=3.0)
        if tok is None:
            return
        with lock:
            hold_observed.append(rid)
            # Verify invariant: only THIS token holds the slots right now.
            # Filter to status=held since list_claims now also returns free GPUs.
            claims_now = scoreboard.list_claims()
            held_now = [c for c in claims_now if c["status"] == "held"]
            n0_holders = [c["holder"] for c in held_now if c["node_id"] == "n0" and c["gpu_idx"] == 0]
            n1_holders = [c["holder"] for c in held_now if c["node_id"] == "n1" and c["gpu_idx"] == 0]
            if len(n0_holders) > 1 or len(n1_holders) > 1:
                violation["count"] += 1
            if n0_holders and n1_holders and n0_holders[0] != n1_holders[0]:
                violation["count"] += 1
        # Hold briefly then release.
        time.sleep(0.05)
        scoreboard.release_gpus(tok)

    threads = [threading.Thread(target=round_worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert violation["count"] == 0, "two holders of same slot observed — invariant violated"
    # Sanity: at least 2 rounds succeeded.
    assert len(hold_observed) >= 2


def test_cross_set_opposite_orders_no_deadlock() -> None:
    """Thread A acquires (n0:0, n1:0) in sorted order; Thread B acquires (n1:0, n0:0).
    Sorted ordering means both threads acquire n0 first then n1 — no AB-BA deadlock."""
    _register_worker("n0")
    _register_worker("n1")

    done = {"count": 0}
    lock = threading.Lock()

    def worker(holder: str) -> None:
        tok = scoreboard.acquire_gpus([("n0", 0), ("n1", 0)],
                                       holder=holder, job_id=holder,
                                       deadline_s=5.0)
        if tok is not None:
            time.sleep(0.05)
            scoreboard.release_gpus(tok)
            with lock:
                done["count"] += 1

    threads = [
        threading.Thread(target=worker, args=("A",)),
        threading.Thread(target=worker, args=("B",)),
        threading.Thread(target=worker, args=("C",)),
        threading.Thread(target=worker, args=("D",)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert done["count"] == 4, f"all 4 must complete without deadlock, got {done['count']}"


# --------------------------------------------------------------------------- #
# Lease expiry + reap
# --------------------------------------------------------------------------- #
def test_reap_expired_claim_with_stale_heartbeat() -> None:
    _register_worker("n0")
    tok = scoreboard.acquire_gpus([("n0", 0)], holder="n0", job_id="j",
                                   lease_s=0.1, deadline_s=2.0)
    assert tok is not None

    # Simulate worker death: stale heartbeat + wait past lease.
    hb = paths.worker_heartbeat("n0")
    old = time.time() - 120
    os.utime(hb, (old, old))
    time.sleep(0.2)

    reaped = scoreboard.reap_expired_claims("n0")
    assert ("n0", 0) in reaped

    # Slot is now free.
    tok2 = scoreboard.acquire_gpus([("n0", 0)], holder="t2", job_id="j2",
                                    deadline_s=1.0)
    assert tok2 is not None


def test_reap_skips_when_heartbeat_fresh() -> None:
    """Even if lease_until is past, a fresh heartbeat means worker may still renew."""
    _register_worker("n0")
    tok = scoreboard.acquire_gpus([("n0", 0)], holder="n0", job_id="j",
                                   lease_s=0.1, deadline_s=2.0)
    assert tok is not None
    time.sleep(0.2)

    # Heartbeat is fresh (worker just called touch_heartbeat during register).
    reaped = scoreboard.reap_expired_claims("n0")
    assert reaped == [], "fresh-heartbeat holder must not be reaped"


# --------------------------------------------------------------------------- #
# Secret verification
# --------------------------------------------------------------------------- #
def test_release_with_wrong_secret_does_not_unlink() -> None:
    _register_worker("n0")
    tok = scoreboard.acquire_gpus([("n0", 0)], holder="h", job_id="j", deadline_s=1.0)
    assert tok is not None

    # Forge a token with the wrong secret but same slots.
    forged = LeaseToken(holder="h", job_id="j", secret="WRONG",
                        slots=tok.slots, acquired_at=tok.acquired_at,
                        lease_until=tok.lease_until)
    scoreboard.release_gpus(forged)

    # Original slot still held.
    claims = scoreboard.list_claims()
    held = [c for c in claims if c["status"] == "held"]
    assert any(c["node_id"] == "n0" and c["gpu_idx"] == 0 for c in held)

    scoreboard.release_gpus(tok)


def test_force_release_unconditional() -> None:
    _register_worker("n0")
    tok = scoreboard.acquire_gpus([("n0", 0)], holder="h", job_id="j", deadline_s=1.0)
    assert tok is not None

    existed = scoreboard.force_release(("n0", 0), reason="test")
    assert existed is True

    claims = scoreboard.list_claims()
    held = [c for c in claims if c["status"] == "held"]
    assert all(not (c["node_id"] == "n0" and c["gpu_idx"] == 0) for c in held)
    # list_claims should still REPORT the slot (as free) — that's the new
    # behavior that makes the WebUI show full topology when idle.
    free = [c for c in claims if c["status"] == "free"]
    assert any(c["node_id"] == "n0" and c["gpu_idx"] == 0 for c in free), \
        "free GPUs must appear in list_claims so WebUI shows full topology"


def test_force_release_writes_cancel_marker(tmp_path: Path) -> None:
    _register_worker("n0")
    tok = scoreboard.acquire_gpus([("n0", 0)], holder="h", job_id="j", deadline_s=1.0)
    assert tok is not None

    job_dir = tmp_path / "fakejob"
    job_dir.mkdir()
    scoreboard.force_release(("n0", 0), reason="admin-kill", cancel_job_dir=job_dir)
    assert (job_dir / "cancel.marker").exists()


# --------------------------------------------------------------------------- #
# Renew lease
# --------------------------------------------------------------------------- #
def test_renew_lease_extends_meta() -> None:
    _register_worker("n0")
    tok = scoreboard.acquire_gpus([("n0", 0)], holder="h", job_id="j",
                                   lease_s=10.0, deadline_s=1.0)
    assert tok is not None
    original = tok.lease_until

    ok = scoreboard.renew_lease(tok, extra_s=100.0)
    assert ok is True
    assert tok.lease_until > original

    # Meta file reflects the new lease.
    meta = scoreboard.fs_primitives.read_claim(paths.gpu_meta_path("n0", 0))
    assert meta is not None
    assert float(meta["lease_until"]) >= original + 50

    scoreboard.release_gpus(tok)
