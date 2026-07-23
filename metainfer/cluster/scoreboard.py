"""GPU-level scoreboard — atomic all-or-nothing acquisition across nodes.

Authority sources:

- ``cluster/scoreboard/<node_id>/gpu-<i>.claim`` — the mutex itself. Created
  by ``link_claim`` (atomic on NFS). Carries ``{holder, job_id, acquired_at,
  lease_until, secret}``. **Content is immutable after link** — to extend the
  lease, rewrite the sibling ``.meta.json`` (see :func:`renew_lease`).

- ``cluster/scoreboard/<node_id>/gpu-<i>.meta.json`` — derived/mutable holder
  metadata, kept in sync with the claim file by ``acquire``/``renew``/``release``
  /``reap``. Readers wanting "who holds this slot now" should prefer the claim
  file (SSOT) and use meta.json only for fields the claim cannot update
  (``lease_until`` post-renewal).

Design decisions (see plan file):

- **All-or-nothing with sorted ordering** (decision 2). Acquirer sorts requested
  ``(node_id, gpu_idx)`` pairs deterministically, links them one by one. Any
  failure → release all previously acquired in this attempt → backoff with
  jitter → retry until success or deadline. **Provably deadlock-free**: holders
  never wait while holding.

- **Lease token with secret** (decision 3). ``acquire_gpus`` returns a
  ``LeaseToken`` containing the secret that was written into each claim file.
  ``release_gpus(token)`` verifies each claim's secret matches before unlinking.

- **Single reap path** (decision 4, mirrors CLAUDE.md launcher invariant).
  ``reap_expired_claims`` and ``force_release`` share the same internal
  ``_break_claim_unconditional`` — there is no second reaper.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from . import fs_primitives
from . import paths
from .paths import gpu_claim_path, gpu_meta_path


DEFAULT_LEASE_S = 1800.0       # 30 min
DEFAULT_DEADLINE_S = 60.0      # how long acquire_gpus retries before giving up
DEFAULT_BACKOFF_MIN_S = 0.05
DEFAULT_BACKOFF_MAX_S = 0.5


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
Slot = Tuple[str, int]  # (node_id, gpu_idx)


@dataclass
class LeaseToken:
    """Returned by :func:`acquire_gpus`. Caller must present this to release."""
    holder: str
    job_id: str
    secret: str
    slots: List[Slot] = field(default_factory=list)
    acquired_at: float = 0.0
    lease_until: float = 0.0


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _build_claim_payload(holder: str, job_id: str, secret: str,
                         acquired_at: float, lease_until: float) -> Dict[str, object]:
    return {
        "holder": holder,
        "job_id": job_id,
        "secret": secret,
        "acquired_at": acquired_at,
        "lease_until": lease_until,
    }


def _write_meta(slot: Slot, payload: Dict[str, object]) -> None:
    fs_primitives.atomic_write_json(gpu_meta_path(slot[0], slot[1]), payload)


def _try_acquire_one(slot: Slot, payload: Dict[str, object]) -> bool:
    """Attempt to link-claim a single slot. Returns True on win."""
    return fs_primitives.link_claim(gpu_claim_path(slot[0], slot[1]), payload)


def _break_claim_unconditional(slot: Slot) -> None:
    """Single reap path. Used by release (with secret check done by caller),
    force_release, and reap_expired_claims. Idempotent."""
    fs_primitives.break_claim(gpu_claim_path(slot[0], slot[1]))
    # Best-effort meta cleanup
    try:
        gpu_meta_path(slot[0], slot[1]).unlink()
    except FileNotFoundError:
        pass
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def acquire_gpus(
    slots: Iterable[Slot],
    holder: str,
    job_id: str,
    lease_s: float = DEFAULT_LEASE_S,
    deadline_s: float = DEFAULT_DEADLINE_S,
) -> Optional[LeaseToken]:
    """Acquire a set of GPU slots atomically (all-or-nothing).

    Returns a :class:`LeaseToken` on success, or None if the deadline elapsed
    before all slots could be acquired simultaneously.

    Algorithm:
      1. Sort slots by (node_id, gpu_idx) for deterministic order.
      2. Loop until deadline:
         a. Mint a fresh secret, write the same payload to each slot via
            ``link_claim``. Track which we won.
         b. If a slot is already held, attempt stale-reap: if the existing
            claim's lease_until has passed AND the holder's heartbeat is stale,
            break it and retry the link.
         c. If a slot is held by a live holder, release all slots we acquired
            in this attempt (rollback), back off, retry.
      3. On success, return LeaseToken with the minted secret.

    Deadlock-freedom argument: we never block while holding slots. Every failed
    attempt fully releases before sleeping. Sorted ordering prevents cyclic
    wait across multiple concurrent acquirers.
    """
    sorted_slots: List[Slot] = sorted(set(slots))
    if not sorted_slots:
        # Vacuous success — return an empty token.
        return LeaseToken(holder=holder, job_id=job_id,
                          secret=fs_primitives.generate_secret(),
                          slots=[], acquired_at=time.time(),
                          lease_until=time.time() + lease_s)

    deadline = time.time() + deadline_s
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        secret = fs_primitives.generate_secret()
        now = time.time()
        lease_until = now + lease_s
        payload = _build_claim_payload(holder, job_id, secret, now, lease_until)

        acquired: List[Slot] = []
        rolled_back = False
        for slot in sorted_slots:
            if _try_acquire_one(slot, payload):
                acquired.append(slot)
                _write_meta(slot, payload)
                continue
            # Slot is held — try stale-reap, then retry the link once.
            if _maybe_reap_stale(slot):
                if _try_acquire_one(slot, payload):
                    acquired.append(slot)
                    _write_meta(slot, payload)
                    continue
            # Held by a live tenant — rollback this attempt.
            for s in acquired:
                _break_claim_unconditional(s)
            rolled_back = True
            break

        if not rolled_back:
            # All slots acquired.
            return LeaseToken(holder=holder, job_id=job_id, secret=secret,
                              slots=acquired, acquired_at=now, lease_until=lease_until)

        # Backoff with jitter before retry.
        time.sleep(random.uniform(DEFAULT_BACKOFF_MIN_S, DEFAULT_BACKOFF_MAX_S))

    return None


def release_gpus(token: LeaseToken) -> None:
    """Release all slots held by ``token``. Verifies secret before unlinking.

    If a slot has been reaped-and-reacquired by someone else (different secret),
    we silently skip it — it's no longer ours.
    """
    for slot in token.slots:
        claim = fs_primitives.read_claim(gpu_claim_path(slot[0], slot[1]))
        if claim is None:
            # Already released or reaped.
            continue
        if claim.get("secret") != token.secret:
            # Slot was reaped and reacquired by someone else — not ours anymore.
            continue
        _break_claim_unconditional(slot)


def renew_lease(token: LeaseToken, extra_s: float = DEFAULT_LEASE_S) -> bool:
    """Extend the lease on all slots in ``token``.

    Because claim-file content is immutable post-link, we update the sibling
    ``.meta.json``'s ``lease_until``. The claim file's ``lease_until`` stays at
    the original value; the reaper treats meta.json as the live source of truth
    when present.

    Returns True if all slots still belong to ``token`` (secret match), False
    if any slot was lost (caller should treat as failure and stop relying on
    those slots).
    """
    now = time.time()
    new_lease = now + extra_s
    all_ok = True
    for slot in token.slots:
        claim = fs_primitives.read_claim(gpu_claim_path(slot[0], slot[1]))
        if claim is None or claim.get("secret") != token.secret:
            all_ok = False
            continue
        # Update meta.json with extended lease.
        meta = dict(claim)
        meta["lease_until"] = new_lease
        meta["renewed_at"] = now
        _write_meta(slot, meta)
    if all_ok:
        token.lease_until = new_lease
    return all_ok


def force_release(slot: Slot, reason: str = "admin-kill",
                  cancel_job_dir: Optional[Path] = None) -> bool:
    """Single reap path: unconditionally remove a claim, regardless of secret.

    Optionally writes a ``cancel.marker`` into ``cancel_job_dir`` so the
    associated worker subprocess can SIGTERM its child.

    Returns True if the claim existed and was removed.
    """
    claim_path = gpu_claim_path(slot[0], slot[1])
    existed = claim_path.exists()
    claim = fs_primitives.read_claim(claim_path)
    _break_claim_unconditional(slot)
    if cancel_job_dir is not None and claim is not None:
        marker = cancel_job_dir / "cancel.marker"
        try:
            fs_primitives.atomic_write_text(
                marker,
                f'{{"reason":"{reason}","ts":{time.time():.3f}}}\n',
            )
        except OSError:
            pass
    return existed


def reap_expired_claims(node_id: str) -> List[Slot]:
    """Reap all expired claims on ``node_id``. Returns the reaped slots.

    A claim is reaped iff:
      - its lease_until (from meta.json if present, else claim file) has passed, AND
      - the holder's heartbeat is stale (worker is presumed dead).

    If lease_until has passed but the worker heartbeat is fresh, the claim is
    NOT reaped (the worker may be mid-renew and just hasn't called renew_lease
    yet — see CLAUDE.md single-reap-path invariant).
    """
    from . import worker_registry

    reaped: List[Slot] = []
    d = paths.scoreboard_dir(node_id)
    for entry in list(d.iterdir()):
        if not entry.name.startswith("gpu-") or not entry.name.endswith(".claim"):
            continue
        try:
            idx = int(entry.name[len("gpu-"):-len(".claim")])
        except ValueError:
            continue
        slot: Slot = (node_id, idx)
        claim = fs_primitives.read_claim(entry)
        if claim is None:
            # Claim vanished during scan; cleanup any stray meta.
            _break_claim_unconditional(slot)
            continue
        # Determine effective lease_until: prefer meta.json (post-renewal) over claim.
        meta = fs_primitives.read_claim(gpu_meta_path(node_id, idx)) or {}
        lease_until = float(meta.get("lease_until", claim.get("lease_until", 0)))
        if time.time() < lease_until:
            continue
        # Lease expired — also require heartbeat to be stale.
        holder = str(claim.get("holder", ""))
        if holder and not fs_primitives.is_stale_heartbeat(
            paths.worker_heartbeat(holder),
            stale_after_s=worker_registry.STALE_AFTER_S,
        ):
            continue
        _break_claim_unconditional(slot)
        reaped.append(slot)
    return reaped


def list_claims(node_id: Optional[str] = None) -> List[Dict[str, object]]:
    """Snapshot of every GPU in the cluster, free OR held.

    Joins the worker topology (``cluster/workers/<id>.json::gpu_topology``)
    with current claim files. Every GPU the cluster knows about produces
    one row, with ``status="free"`` or ``status="held"``. Free GPUs carry
    empty ``holder``/``job_id``; held GPUs carry the claim metadata plus
    lease remaining.

    If ``node_id`` is given, restrict to that node. GPUs referenced by a
    claim file but missing from the worker's topology (e.g. worker record
    deleted mid-run) are still reported so leases aren't silently lost.
    """
    # Lazy import to avoid circular dependency at module load.
    from . import worker_registry

    now = time.time()

    # Collect the set of nodes to report. We union:
    #   (a) worker records (canonical GPU topology source), AND
    #   (b) scoreboard dirs that have claim files (covers the case where
    #       a worker record was deleted but a claim is still live).
    nodes_topology: Dict[str, Dict[int, Dict[str, object]]] = {}
    for w in worker_registry.list_workers():
        if node_id is not None and w.node_id != node_id:
            continue
        nodes_topology[w.node_id] = {
            int(k): v for k, v in (w.gpu_topology or {}).items()
        }
    try:
        for entry in paths.scoreboard_root().iterdir():
            if not entry.is_dir():
                continue
            nid = entry.name
            if node_id is not None and nid != node_id:
                continue
            nodes_topology.setdefault(nid, {})
    except OSError:
        pass

    # Collect claim data per (node_id, gpu_idx).
    claims_by_slot: Dict[Tuple[str, int], Dict[str, object]] = {}
    for nid in nodes_topology:
        d = paths.scoreboard_dir(nid)
        try:
            entries = list(d.iterdir())
        except OSError:
            entries = []
        for entry in entries:
            if not entry.name.startswith("gpu-") or not entry.name.endswith(".claim"):
                continue
            try:
                idx = int(entry.name[len("gpu-"):-len(".claim")])
            except ValueError:
                continue
            claim = fs_primitives.read_claim(entry)
            if claim is None:
                continue
            claims_by_slot[(nid, idx)] = claim

    # Emit one row per known GPU, plus rows for claim files whose GPU idx
    # isn't in the topology (orphaned claim — shouldn't happen normally
    # but must be surfaced so it can be released).
    out: List[Dict[str, object]] = []
    for nid in sorted(nodes_topology):
        topo = nodes_topology[nid]
        all_idxs = set(topo.keys()) | {
            idx for (n, idx) in claims_by_slot if n == nid
        }
        for idx in sorted(all_idxs):
            claim = claims_by_slot.get((nid, idx))
            topo_entry = topo.get(idx, {}) or {}
            if claim is None:
                out.append({
                    "node_id": nid,
                    "gpu_idx": idx,
                    "status": "free",
                    "gpu_name": topo_entry.get("name", ""),
                    "total_memory_mib": topo_entry.get("total_memory_mib", 0),
                    "holder": "",
                    "job_id": "",
                    "acquired_at": 0,
                    "acquired_ago_s": 0,
                    "lease_until": 0,
                    "lease_remaining_s": 0,
                })
                continue
            meta = fs_primitives.read_claim(gpu_meta_path(nid, idx)) or {}
            lease_until = float(meta.get("lease_until", claim.get("lease_until", 0)))
            out.append({
                "node_id": nid,
                "gpu_idx": idx,
                "status": "held",
                "gpu_name": topo_entry.get("name", ""),
                "total_memory_mib": topo_entry.get("total_memory_mib", 0),
                "holder": claim.get("holder", ""),
                "job_id": claim.get("job_id", ""),
                "acquired_at": float(claim.get("acquired_at", 0)),
                "acquired_ago_s": max(0.0, now - float(claim.get("acquired_at", 0))),
                "lease_until": lease_until,
                "lease_remaining_s": max(0.0, lease_until - now),
            })
    return out


def _maybe_reap_stale(slot: Slot) -> bool:
    """If the existing claim on ``slot`` is past lease AND holder heartbeat is
    stale, break it. Returns True if the slot is now free.

    This is the inline stale-reap path used during acquire_gpus to avoid
    waiting forever for a dead holder.
    """
    claim = fs_primitives.read_claim(gpu_claim_path(slot[0], slot[1]))
    if claim is None:
        return True
    meta = fs_primitives.read_claim(gpu_meta_path(slot[0], slot[1])) or {}
    lease_until = float(meta.get("lease_until", claim.get("lease_until", 0)))
    if time.time() < lease_until:
        return False
    holder = str(claim.get("holder", ""))
    if holder:
        from . import worker_registry
        if not fs_primitives.is_stale_heartbeat(
            paths.worker_heartbeat(holder),
            stale_after_s=worker_registry.STALE_AFTER_S,
        ):
            return False
    _break_claim_unconditional(slot)
    return True
