"""Worker registry — authoritative list of compute nodes available for remote jobs.

Authority sources:

- ``cluster/workers/<node_id>.json`` — worker identity + GPU topology. Written by
  the worker process on startup via :func:`register_worker`. Re-registration
  overwrites the prior record (cold-start safe). **Never rewritten by heartbeat.**

- ``cluster/workers/<node_id>.heartbeat`` — mtime-only liveness file. Touched by
  the worker every ``HEARTBEAT_INTERVAL_S`` seconds. Readers use
  :func:`metainfer.cluster.fs_primitives.is_stale_heartbeat` to decide liveness.

Derived state (computed at read time, not persisted):

- ``is_worker_alive(node_id)`` — combines record existence with heartbeat freshness.

This module is symmetric: webui server (for listing), workers (for registration),
and orchestrators (for host lookups) all read from the same files.
"""

from __future__ import annotations

import socket
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from . import fs_primitives
from . import paths


# Heartbeat interval: how often the worker touches its heartbeat file. Liveness
# readers should treat anything older than 4x this as stale (covers GC pauses,
# NFS hiccups, etc).
HEARTBEAT_INTERVAL_S = 15.0
STALE_AFTER_S = 60.0


@dataclass
class WorkerRecord:
    """In-memory representation of ``cluster/workers/<node_id>.json``."""
    node_id: str
    ip: str
    hostname: str
    mac: str
    gpu_topology: Dict[int, Dict[str, object]] = field(default_factory=dict)
    registered_at: float = 0.0
    boot_id: str = ""

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a JSON-friendly dict (gpu_topology keys cast to str)."""
        # JSON keys must be strings — gpu_topology dict has int keys.
        topo_serializable = {str(k): v for k, v in self.gpu_topology.items()}
        return {
            "node_id": self.node_id,
            "ip": self.ip,
            "hostname": self.hostname,
            "mac": self.mac,
            "gpu_topology": topo_serializable,
            "registered_at": self.registered_at,
            "boot_id": self.boot_id,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "WorkerRecord":
        """Inverse of to_dict; casts gpu_topology keys back to int."""
        raw_topo = d.get("gpu_topology", {}) or {}
        topo: Dict[int, Dict[str, object]] = {}
        if isinstance(raw_topo, dict):
            for k, v in raw_topo.items():
                try:
                    topo[int(k)] = dict(v) if isinstance(v, dict) else {}
                except (ValueError, TypeError):
                    continue
        return cls(
            node_id=str(d.get("node_id", "")),
            ip=str(d.get("ip", "")),
            hostname=str(d.get("hostname", "")),
            mac=str(d.get("mac", "")),
            gpu_topology=topo,
            registered_at=float(d.get("registered_at", 0.0)),
            boot_id=str(d.get("boot_id", "")),
        )


class WorkerIdentityConflict(Exception):
    """Raised when a registration would change the physical identity of an
    existing worker record.

    Per SSOT, ``workers/<node_id>.json`` is authoritative for "which physical
    box is worker N". A daemon that shows up with ``METAINFER_NODE_ID=worker21``
    but ``hostname=worker25`` is either misconfigured or actively spoofing —
    either way, letting it overwrite the record would corrupt the cluster map
    (e.g. PP2 ranks would resolve to the wrong IP and rendezvous on the wrong
    host).

    The caller (worker daemon) MUST treat this as a fatal startup error: log
    the conflict, refuse to enter the main loop, and exit non-zero. Recovery
    requires an operator decision — either fix the new daemon's env (unset the
    wrong ``METAINFER_NODE_ID``), or explicitly clear the existing record if
    the box genuinely re-homed.
    """


def _write_conflict_file(node_id: str, existing: WorkerRecord,
                          attempted: Dict[str, object]) -> None:
    """Drop a sidecar ``.conflict.<ts>.json`` recording the rejected overwrite.

    The conflict file is for forensic value only — it is never read back by
    any runtime path. SSOT for the live record stays at ``<node_id>.json``.
    """
    import json as _json
    ts = int(time.time())
    path = paths.workers_dir() / f"{node_id}.conflict.{ts}.json"
    payload = {
        "existing_record": existing.to_dict(),
        "attempted_overwrite": attempted,
        "rejected_at": ts,
        "reason": (
            "register_worker refused to overwrite an existing worker record "
            "with a different physical identity (hostname mismatch). If this "
            "is a legitimate re-homing, delete workers/<node_id>.json first."
        ),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
    except OSError:
        pass  # best-effort forensics; never block the reject path


def register_worker(
    node_id: str,
    ip: str,
    hostname: str,
    mac: str,
    gpu_topology: Dict[int, Dict[str, object]],
) -> WorkerRecord:
    """Register (or re-register) a worker node.

    Writes ``workers/<node_id>.json`` atomically and touches the initial
    heartbeat. Safe to call repeatedly (cold restart path); each call generates
    a fresh ``boot_id`` to disambiguate sessions.

    The JSON is the SSOT for identity + topology. Subsequent heartbeat touches
    update only the ``.heartbeat`` file's mtime — never the JSON.

    Identity-drift rejection: if ``workers/<node_id>.json`` already exists and
    its ``hostname`` differs from the registering daemon's reported hostname,
    the registration is refused (see :class:`WorkerIdentityConflict`). This
    stops a daemon with a mis-set ``METAINFER_NODE_ID`` from clobbering a
    real worker's record with another box's hostname/IP/MAC, which would
    silently corrupt PP2 rendezvous and scoreboard slot ownership.

    The one legitimate re-homing path is explicit: delete the JSON first,
    then register fresh. This makes the operator's intent visible in the
    filesystem history instead of being silent overwrite-by-default.
    """
    existing = read_worker(node_id)
    if (existing is not None
            and existing.hostname
            and hostname
            and existing.hostname != hostname):
        _write_conflict_file(node_id, existing, {
            "node_id": node_id,
            "ip": ip,
            "hostname": hostname,
            "mac": mac,
        })
        raise WorkerIdentityConflict(
            f"refusing to re-register worker {node_id!r}: existing record "
            f"has hostname={existing.hostname!r} (ip={existing.ip!r}), but "
            f"this daemon reports hostname={hostname!r} (ip={ip!r}). "
            f"A worker's physical identity (hostname/IP/MAC) must not change "
            f"between re-registrations. Either fix the new daemon's "
            f"METAINFER_NODE_ID / --hostname, or explicitly delete "
            f"{paths.worker_record(node_id)} to re-home this node_id. "
            f"Conflict details written to workers/{node_id}.conflict.*.json."
        )

    record = WorkerRecord(
        node_id=node_id,
        ip=ip,
        hostname=hostname,
        mac=mac,
        gpu_topology=gpu_topology,
        registered_at=time.time(),
        boot_id=uuid.uuid4().hex,
    )
    fs_primitives.atomic_write_json(paths.worker_record(node_id), record.to_dict())
    fs_primitives.touch_heartbeat(paths.worker_heartbeat(node_id))
    return record


def read_worker(node_id: str) -> Optional[WorkerRecord]:
    """Read one worker's record. Returns None if missing or corrupt."""
    p = paths.worker_record(node_id)
    data = fs_primitives.read_claim(p)  # read_claim is generic JSON read
    if data is None:
        return None
    try:
        return WorkerRecord.from_dict(data)
    except (TypeError, ValueError):
        return None


def list_workers() -> List[WorkerRecord]:
    """List all registered workers (alive and dead)."""
    out: List[WorkerRecord] = []
    d = paths.workers_dir()
    for entry in d.iterdir():
        if not entry.name.endswith(".json"):
            continue
        rec = read_worker(entry.stem)
        if rec is not None:
            out.append(rec)
    # Deterministic order for stable UI / test output.
    out.sort(key=lambda r: r.node_id)
    return out


def is_worker_alive(node_id: str, stale_after_s: float = STALE_AFTER_S) -> bool:
    """A worker is "alive" iff its record exists AND its heartbeat is fresh."""
    if read_worker(node_id) is None:
        return False
    return not fs_primitives.is_stale_heartbeat(
        paths.worker_heartbeat(node_id), stale_after_s=stale_after_s
    )


def touch_heartbeat(node_id: str) -> None:
    """Convenience wrapper for the worker daemon's main loop."""
    fs_primitives.touch_heartbeat(paths.worker_heartbeat(node_id))


def detect_local_ip() -> str:
    """Best-effort primary outbound IP. Used by worker if --ip not given."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # Doesn't actually connect, just picks the routing entry.
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"


def detect_local_mac() -> str:
    """Best-effort primary MAC address. Empty string if unavailable."""
    import uuid as _uuid
    try:
        return _uuid.getnode().to_bytes(6, "big").hex(":")
    except (AttributeError, TypeError):
        return ""
