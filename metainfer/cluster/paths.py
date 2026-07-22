"""Filesystem paths for the cluster-wide coordination layer.

Lives parallel to ``<root>/nodes/`` because it is cross-host coordination state
(not any single node's bookkeeping). MUST be on the same NFS mount as ``nodes/``
— the hardlink-claim primitives in :mod:`metainfer.cluster.fs_primitives` only
work within a single filesystem.

Layout::

    $METAINFER_ROOT/
    ├── nodes/<node_id>/...            (existing — per-node state, unchanged)
    └── cluster/                       (NEW — shared coordination state)
        ├── workers/<node_id>.json             worker identity + GPU topology (SSOT)
        ├── workers/<node_id>.heartbeat        mtime-only liveness signal
        ├── scoreboard/<node_id>/
        │   ├── gpu-<i>.claim                  hardlink-claim file = GPU slot mutex
        │   └── gpu-<i>.meta.json              derived holder metadata (mutable)
        ├── inbox/<worker_node_id>/<job_id>/   submitted jobs (orchestrator-owned)
        │   ├── job.json
        │   ├── script.sh | prompt.txt
        │   ├── stdout.log, stderr.log         (worker appends live)
        │   ├── status.json
        │   └── cancel.marker                  (force-kill signal)
        └── replies/<orchestrator_node_id>/<job_id>.result.json
"""

from __future__ import annotations

from pathlib import Path

from metainfer.server.paths import root_dir


def cluster_dir() -> Path:
    """``<root>/cluster/``. Cross-host coordination root. Created on first access."""
    p = root_dir() / "cluster"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# Worker registry
# --------------------------------------------------------------------------- #
def workers_dir() -> Path:
    p = cluster_dir() / "workers"
    p.mkdir(parents=True, exist_ok=True)
    return p


def worker_record(node_id: str) -> Path:
    """``workers/<node_id>.json`` — authoritative worker record."""
    return workers_dir() / f"{node_id}.json"


def worker_heartbeat(node_id: str) -> Path:
    """``workers/<node_id>.heartbeat`` — mtime-only liveness file."""
    return workers_dir() / f"{node_id}.heartbeat"


# --------------------------------------------------------------------------- #
# Scoreboard
# --------------------------------------------------------------------------- #
def scoreboard_root() -> Path:
    p = cluster_dir() / "scoreboard"
    p.mkdir(parents=True, exist_ok=True)
    return p


def scoreboard_dir(node_id: str) -> Path:
    """Per-node directory holding ``gpu-<i>.claim`` files."""
    p = scoreboard_root() / node_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def gpu_claim_path(node_id: str, gpu_idx: int) -> Path:
    """``scoreboard/<node_id>/gpu-<i>.claim`` — the slot mutex (hardlink target)."""
    return scoreboard_dir(node_id) / f"gpu-{gpu_idx}.claim"


def gpu_meta_path(node_id: str, gpu_idx: int) -> Path:
    """``scoreboard/<node_id>/gpu-<i>.meta.json`` — mutable holder metadata.

    Distinct from ``.claim`` (immutable post-link): ``renew_lease`` rewrites
    this file under flock to extend ``lease_until``.
    """
    return scoreboard_dir(node_id) / f"gpu-{gpu_idx}.meta.json"


# --------------------------------------------------------------------------- #
# Message queue
# --------------------------------------------------------------------------- #
def inbox_root() -> Path:
    p = cluster_dir() / "inbox"
    p.mkdir(parents=True, exist_ok=True)
    return p


def inbox_dir(worker_node_id: str) -> Path:
    """All jobs targeted at ``worker_node_id``. Worker polls here."""
    p = inbox_root() / worker_node_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def job_dir(worker_node_id: str, job_id: str) -> Path:
    """A single submitted job's directory."""
    return inbox_dir(worker_node_id) / job_id


def job_claimed_marker(job_dir_path: Path) -> Path:
    """``<job_dir>/claimed`` — hardlink marker indicating the job has been consumed."""
    return job_dir_path / "claimed"


def job_cancel_marker(job_dir_path: Path) -> Path:
    """``<job_dir>/cancel.marker`` — written by scoreboard force-release."""
    return job_dir_path / "cancel.marker"


def replies_root() -> Path:
    p = cluster_dir() / "replies"
    p.mkdir(parents=True, exist_ok=True)
    return p


def replies_dir(orchestrator_node_id: str) -> Path:
    """All results addressed back to orchestrator ``orchestrator_node_id``."""
    p = replies_root() / orchestrator_node_id
    p.mkdir(parents=True, exist_ok=True)
    return p


def result_path(orchestrator_node_id: str, job_id: str) -> Path:
    return replies_dir(orchestrator_node_id) / f"{job_id}.result.json"
