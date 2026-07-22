"""Unit tests for metainfer.cluster.fs_primitives.

Covers:
- atomic_write_text / atomic_write_json basic round-trip
- link_claim race: N threads, exactly one winner
- read_claim tolerates missing / corrupt files
- break_claim with and without expected_secret
- touch_heartbeat / is_stale_heartbeat semantics
"""

from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from metainfer.cluster.fs_primitives import (
    atomic_write_json,
    atomic_write_text,
    break_claim,
    generate_secret,
    is_stale_heartbeat,
    link_claim,
    read_claim,
    touch_heartbeat,
)


# --------------------------------------------------------------------------- #
# atomic_write_text / atomic_write_json
# --------------------------------------------------------------------------- #
def test_atomic_write_text_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "f.txt"
    atomic_write_text(p, "hello\nworld\n")
    assert p.read_text() == "hello\nworld\n"


def test_atomic_write_json_round_trip(tmp_path: Path) -> None:
    p = tmp_path / "f.json"
    obj = {"b": 2, "a": 1, "nested": {"x": [1, 2, 3]}}
    atomic_write_json(p, obj)
    assert json.loads(p.read_text()) == obj


def test_atomic_write_concurrent_writers_no_collision(tmp_path: Path) -> None:
    """N threads writing the same path with different content — each write must succeed
    (no tmp-file collision); final content is one of the inputs."""
    p = tmp_path / "out.json"
    N = 20

    def writer(i: int) -> None:
        atomic_write_json(p, {"i": i})

    with ThreadPoolExecutor(max_workers=N) as ex:
        list(ex.map(writer, range(N)))

    final = json.loads(p.read_text())
    assert final["i"] in set(range(N))
    # No leftover tmp files
    assert not any(c.name.startswith(".out.json.") for c in tmp_path.iterdir())


# --------------------------------------------------------------------------- #
# link_claim race
# --------------------------------------------------------------------------- #
def test_link_claim_first_call_wins_second_returns_false(tmp_path: Path) -> None:
    target = tmp_path / "claim.json"
    payload_a = {"holder": "A", "secret": "sa"}
    payload_b = {"holder": "B", "secret": "sb"}

    assert link_claim(target, payload_a) is True
    assert link_claim(target, payload_b) is False

    # Original payload preserved (B did not overwrite)
    assert read_claim(target) == payload_a


def test_link_claim_concurrent_unique_winner(tmp_path: Path) -> None:
    """N threads link_claim the same target — exactly one wins."""
    target = tmp_path / "race.json"
    N = 30
    results: list[bool] = []
    lock = threading.Lock()

    def claim(i: int) -> None:
        ok = link_claim(target, {"holder": f"h{i}", "secret": generate_secret()})
        with lock:
            results.append(ok)

    with ThreadPoolExecutor(max_workers=N) as ex:
        list(ex.map(claim, range(N)))

    assert sum(results) == 1, f"expected exactly 1 winner, got {sum(results)}"


def test_link_claim_cleans_up_tmp_files(tmp_path: Path) -> None:
    target = tmp_path / "claim.json"
    link_claim(target, {"holder": "h", "secret": "s"})
    # No tmp files leftover in parent dir
    leftover = [c for c in tmp_path.iterdir() if c.name.startswith(".claim.json.")]
    assert leftover == []


# --------------------------------------------------------------------------- #
# read_claim robustness
# --------------------------------------------------------------------------- #
def test_read_claim_missing_returns_none(tmp_path: Path) -> None:
    assert read_claim(tmp_path / "nope.json") is None


def test_read_claim_corrupt_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    assert read_claim(p) is None


def test_read_claim_valid_payload(tmp_path: Path) -> None:
    p = tmp_path / "ok.json"
    atomic_write_json(p, {"k": "v"})
    assert read_claim(p) == {"k": "v"}


# --------------------------------------------------------------------------- #
# break_claim
# --------------------------------------------------------------------------- #
def test_break_claim_no_secret_unconditional(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    link_claim(p, {"secret": "abc"})
    assert break_claim(p) is True
    assert not p.exists()


def test_break_claim_no_secret_idempotent(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    # File never existed
    assert break_claim(p) is True


def test_break_claim_correct_secret(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    link_claim(p, {"secret": "abc"})
    assert break_claim(p, expected_secret="abc") is True
    assert not p.exists()


def test_break_claim_wrong_secret_no_unlink(tmp_path: Path) -> None:
    p = tmp_path / "c.json"
    link_claim(p, {"secret": "abc"})
    assert break_claim(p, expected_secret="WRONG") is False
    assert p.exists(), "claim must survive wrong-secret release attempt"


def test_break_claim_wrong_secret_missing_file_returns_true(tmp_path: Path) -> None:
    # Already released — caller's secret is irrelevant.
    p = tmp_path / "ghost.json"
    assert break_claim(p, expected_secret="anything") is True


# --------------------------------------------------------------------------- #
# heartbeat
# --------------------------------------------------------------------------- #
def test_touch_heartbeat_creates_then_updates(tmp_path: Path) -> None:
    h = tmp_path / "hb"
    assert not h.exists()
    touch_heartbeat(h)
    assert h.exists()
    m1 = h.stat().st_mtime

    time.sleep(0.05)
    touch_heartbeat(h)
    m2 = h.stat().st_mtime
    assert m2 > m1, "second touch must advance mtime"


def test_is_stale_heartbeat_missing_is_stale(tmp_path: Path) -> None:
    assert is_stale_heartbeat(tmp_path / "nope") is True


def test_is_stale_heartbeat_fresh_is_not_stale(tmp_path: Path) -> None:
    h = tmp_path / "hb"
    touch_heartbeat(h)
    assert is_stale_heartbeat(h, stale_after_s=60) is False


def test_is_stale_heartbeat_old_is_stale(tmp_path: Path) -> None:
    h = tmp_path / "hb"
    touch_heartbeat(h)
    # Backdate mtime by 120s
    old = time.time() - 120
    os.utime(h, (old, old))
    assert is_stale_heartbeat(h, stale_after_s=60) is True
    # Boundary: stale_after_s=200 should still consider it fresh
    assert is_stale_heartbeat(h, stale_after_s=200) is False


def test_generate_secret_unique() -> None:
    secrets = {generate_secret() for _ in range(100)}
    assert len(secrets) == 100
