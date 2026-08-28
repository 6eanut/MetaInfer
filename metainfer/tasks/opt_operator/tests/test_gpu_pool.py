"""Tests for the idle-dispatch GPU pool (injected topology/scoreboard)."""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.gpu_pool import GpuLease, GpuPool, GpuPoolError


def make_token(gpu_idx, secret="s"):
    from metainfer.cluster.scoreboard import LeaseToken
    return LeaseToken(holder="h", job_id="j", secret=secret,
                      slots=[("node", gpu_idx)], acquired_at=1.0, lease_until=2.0)


class FakeLease:
    """Simulates the scoreboard: tracks held slots per gpu_idx."""

    def __init__(self, free: list, held: list):
        self.free = list(free)      # gpu idxs that are currently free
        self.held = list(held)      # gpu idxs held by others (live tenants)
        self.released = []

    def acquire(self, slots, holder, job_id, lease_s, deadline_s):
        idx = slots[0][1]
        if idx in self.held:
            return None
        if idx in self.free:
            self.free.remove(idx)
            self.held.append(idx)
            return make_token(idx)
        return None

    def release(self, token):
        for slot in token.slots:
            self.released.append(slot[1])
            self.held.remove(slot[1])
            self.free.append(slot[1])


def make_pool(fake, node="node", poll_s=0.001):
    return GpuPool(
        node_id=node,
        holder="orch",
        discover=lambda: {i: {} for i in (fake.free + fake.held)},
        acquire=fake.acquire,
        release=fake.release,
        poll_s=poll_s,
        slot_deadline_s=0.001,
    )


def test_acquire_idle_gpu():
    fake = FakeLease(free=[0, 1, 2], held=[])
    pool = make_pool(fake)
    lease = pool.acquire_one("job", timeout_s=5.0)
    assert lease.gpu_idx in (0, 1, 2)
    assert lease.env["HIP_VISIBLE_DEVICES"] == str(lease.gpu_idx)
    assert lease.gpu_idx in fake.held
    # release
    lease.release()
    assert lease.gpu_idx in fake.free


def test_acquire_context_manager_releases():
    fake = FakeLease(free=[0], held=[])
    pool = make_pool(fake)
    with pool.acquire_one("job", timeout_s=5.0) as lease:
        assert lease.gpu_idx in fake.held
    assert lease.gpu_idx in fake.free


def test_waits_for_idle_gpu():
    # gpu 0 busy at first; becomes free after a couple polls.
    fake = FakeLease(free=[], held=[0])
    call = {"n": 0}
    original_acquire = fake.acquire

    def flaky(slots, holder, job_id, lease_s, deadline_s):
        call["n"] += 1
        if call["n"] >= 3:
            fake.held.remove(0)
            fake.free.append(0)
        return original_acquire(slots, holder, job_id, lease_s, deadline_s)

    pool = GpuPool(
        node_id="node", holder="orch",
        discover=lambda: {0: {}}, acquire=flaky, release=fake.release,
        poll_s=0.005, slot_deadline_s=0.001,
    )
    lease = pool.acquire_one("job", timeout_s=5.0)
    assert lease.gpu_idx == 0
    assert call["n"] >= 3


def test_timeout_raises():
    fake = FakeLease(free=[], held=[0, 1])
    pool = make_pool(fake, poll_s=0.01)
    with pytest.raises(GpuPoolError):
        pool.acquire_one("job", timeout_s=0.03)


def test_no_gpus_falls_back_to_single_slot():
    # Topology is undetectable ({}), but slot 0 is acquirable — fall back to it.
    fake = FakeLease(free=[0], held=[])
    pool = GpuPool(
        node_id="node", holder="orch",
        discover=lambda: {},  # undetectable topology
        acquire=fake.acquire,
        release=fake.release,
        poll_s=0.005,
        slot_deadline_s=0.001,
    )
    lease = pool.acquire_one("job", timeout_s=5.0)
    assert lease.gpu_idx == 0


def test_lease_is_idempotent_release():
    fake = FakeLease(free=[0], held=[])
    pool = make_pool(fake)
    lease = pool.acquire_one("job", timeout_s=5.0)
    lease.release()
    lease.release()  # second release is a no-op
    assert lease.gpu_idx in fake.free
