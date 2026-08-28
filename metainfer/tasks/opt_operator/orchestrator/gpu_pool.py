"""GPU pool — idle-dispatch of validation tasks across the node's GPUs.

Per plan §8: *look at which GPU is idle, dispatch the correctness/perf task onto it,
and if none is idle, wait* (NOT pre-sharding the case matrix). This module:

1. Discovers the node's GPUs via :func:`cluster.topology.detect_gpu_topology`.
2. Acquires a single idle slot via :func:`cluster.scoreboard.acquire_gpus`
   (os.link-based, respects others' claims, never double-claims). If every slot is
   busy, it polls until one frees up or the deadline elapses.
3. Returns a :class:`GpuLease` that isolates the GPU via ``*_VISIBLE_DEVICES`` and
   releases the slot on ``release()`` / context exit.

**Lease ownership** (CLAUDE.md invariant): the orchestrator owns the token and must
release it in a ``finally``. The worker / profiler never calls ``release_gpus``.

Discovery and acquisition are injectable so tests can drive them without real GPUs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from metainfer.cluster import scoreboard, topology
from metainfer.cluster.scoreboard import LeaseToken


class GpuPoolError(RuntimeError):
    """Could not acquire any GPU within the deadline."""


Slot = Tuple[str, int]


def default_discover() -> Dict[int, Dict[str, Any]]:
    return topology.detect_gpu_topology()


def default_acquire(slots: Sequence[Slot], holder: str, job_id: str,
                    lease_s: float, deadline_s: float) -> Optional[LeaseToken]:
    return scoreboard.acquire_gpus(list(slots), holder, job_id,
                                   lease_s=lease_s, deadline_s=deadline_s)


def default_release(token: LeaseToken) -> None:
    scoreboard.release_gpus(token)


@dataclass(frozen=True)
class GpuLease:
    """A held GPU slot. Release exactly once (orchestrator-owned)."""

    node_id: str
    gpu_idx: int
    token: LeaseToken
    _release: Callable[[LeaseToken], None] = field(repr=False)

    def __enter__(self) -> "GpuLease":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.release()

    def release(self) -> None:
        if self.token.slots:
            self._release(self.token)
            object.__setattr__(self, "token", LeaseToken(
                holder=self.token.holder, job_id=self.token.job_id,
                secret=self.token.secret, slots=[], acquired_at=self.token.acquired_at,
                lease_until=self.token.lease_until,
            ))

    @property
    def env(self) -> Dict[str, str]:
        """Environment that isolates this slot (HIP/CUDA visibility)."""
        return {
            "HIP_VISIBLE_DEVICES": str(self.gpu_idx),
            "CUDA_VISIBLE_DEVICES": str(self.gpu_idx),
        }


class GpuPool:
    """Idle-dispatch GPU pool for one node."""

    def __init__(
        self,
        *,
        node_id: str,
        holder: str,
        discover: Optional[Callable[[], Dict[int, Dict[str, Any]]]] = None,
        acquire: Optional[Callable[[Sequence[Slot], str, str, float, float], Optional[LeaseToken]]] = None,
        release: Optional[Callable[[LeaseToken], None]] = None,
        lease_s: float = 1800.0,
        slot_deadline_s: float = 5.0,
        poll_s: float = 2.0,
    ) -> None:
        self.node_id = node_id
        self.holder = holder
        self._discover = discover or default_discover
        self._acquire = acquire or default_acquire
        self._release = release or default_release
        self.lease_s = lease_s
        self.slot_deadline_s = slot_deadline_s
        self.poll_s = poll_s

    def discover_gpus(self) -> List[int]:
        topo = self._discover()
        return sorted(int(k) for k in topo)

    def acquire_one(self, job_id: str, timeout_s: float) -> GpuLease:
        """Wait for any idle GPU and lease it. Raises :class:`GpuPoolError` on timeout."""
        gpus = self.discover_gpus()
        if not gpus:
            gpus = [0]  # fall back to a single nominal slot when undetectable
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            for gi in gpus:
                token = self._acquire([(self.node_id, gi)], self.holder, job_id,
                                      self.lease_s, self.slot_deadline_s)
                if token is not None and token.slots:
                    return GpuLease(self.node_id, gi, token, self._release)
            time.sleep(self.poll_s)
        raise GpuPoolError(
            f"no GPU free on {self.node_id!r} within {timeout_s:.0f}s "
            f"(holder={self.holder!r}, job={job_id!r})"
        )


def list_occupancy(node_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Thin view over the shared scoreboard for the WebUI GPU-pool panel."""
    return scoreboard.list_claims(node_id)


__all__ = [
    "GpuPoolError",
    "GpuLease",
    "GpuPool",
    "default_discover",
    "default_acquire",
    "default_release",
    "list_occupancy",
]
