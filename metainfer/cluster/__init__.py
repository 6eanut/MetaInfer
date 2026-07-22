"""Cluster-wide coordination layer.

Authority sources (see CLAUDE.md SSOT rules):

- ``cluster/workers/<node_id>.json`` — worker identity + GPU topology (SSOT)
- ``cluster/workers/<node_id>.heartbeat`` — liveness signal (mtime)
- ``cluster/scoreboard/<node_id>/gpu-<i>.claim`` — GPU slot mutex (hardlink claim)
- ``cluster/scoreboard/<node_id>/gpu-<i>.meta.json`` — derived holder metadata
- ``cluster/inbox/<worker_id>/<job_id>/`` — submitted jobs (owned by orchestrator, consumed by worker)
- ``cluster/replies/<orch_id>/<job_id>.result.json`` — job results (owned by worker)

All cross-host coordination MUST go through primitives in :mod:`metainfer.cluster.fs_primitives`.
"""

from .fs_primitives import (
    atomic_write_json,
    atomic_write_text,
    link_claim,
    read_claim,
    break_claim,
    touch_heartbeat,
    is_stale_heartbeat,
)

__all__ = [
    "atomic_write_json",
    "atomic_write_text",
    "link_claim",
    "read_claim",
    "break_claim",
    "touch_heartbeat",
    "is_stale_heartbeat",
]
