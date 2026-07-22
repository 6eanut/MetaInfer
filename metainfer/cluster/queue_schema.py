"""Job spec for the cluster message queue.

A JobSpec describes a unit of work that an orchestrator submits to a remote
worker. It is serialized to ``cluster/inbox/<worker_node_id>/<job_id>/job.json``.

Two job types:
- ``script``: a bash script body. Worker runs ``bash script.sh``.
- ``agent``: a prompt body. Worker runs ``ccb`` (claude-code-binary) with the
  prompt on stdin.

Required fields are validated at submit time. ``reply_to`` is filled in by
:func:`metainfer.cluster.mqueue.submit_job` based on the submitter's node id.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Status enum (string literal for JSON friendliness)
# --------------------------------------------------------------------------- #
STATUS_PENDING = "pending"          # submitted, not yet claimed
STATUS_INFLIGHT = "inflight"        # worker has claimed, child process running
STATUS_DONE = "done"                # child exited normally (exit_code may still be non-zero)
STATUS_TIMEOUT = "timeout"          # worker killed child after timeout_s
STATUS_CANCELLED = "cancelled"      # force-release wrote cancel.marker
STATUS_WORKER_DEAD = "worker_dead"  # orchestrator-side reap detected worker heartbeat gone
STATUS_FAILED = "failed"            # worker could not execute (e.g. bad script body)


@dataclass
class JobSpec:
    """In-memory representation of ``<job_dir>/job.json``."""
    job_id: str = ""
    type: str = "script"             # "script" | "agent"
    worker_node_id: str = ""
    gpu_slots: List[Tuple[str, int]] = field(default_factory=list)
    timeout_s: float = 1800.0
    env: Dict[str, str] = field(default_factory=dict)
    cwd: str = ""
    submitter: str = ""              # orchestrator node_id
    submitted_at: float = 0.0
    reply_to: str = ""               # absolute path to result.json (filled by submit_job)
    script_body: str = ""            # for type=script
    prompt_body: str = ""            # for type=agent
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "type": self.type,
            "worker_node_id": self.worker_node_id,
            "gpu_slots": [[n, i] for n, i in self.gpu_slots],
            "timeout_s": self.timeout_s,
            "env": self.env,
            "cwd": self.cwd,
            "submitter": self.submitter,
            "submitted_at": self.submitted_at,
            "reply_to": self.reply_to,
            "script_body": self.script_body,
            "prompt_body": self.prompt_body,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobSpec":
        slots_raw = d.get("gpu_slots") or []
        slots: List[Tuple[str, int]] = []
        for s in slots_raw:
            if isinstance(s, (list, tuple)) and len(s) == 2:
                slots.append((str(s[0]), int(s[1])))
        return cls(
            job_id=str(d.get("job_id", "")),
            type=str(d.get("type", "script")),
            worker_node_id=str(d.get("worker_node_id", "")),
            gpu_slots=slots,
            timeout_s=float(d.get("timeout_s", 1800.0)),
            env={str(k): str(v) for k, v in (d.get("env") or {}).items()},
            cwd=str(d.get("cwd", "")),
            submitter=str(d.get("submitter", "")),
            submitted_at=float(d.get("submitted_at", 0.0)),
            reply_to=str(d.get("reply_to", "")),
            script_body=str(d.get("script_body", "")),
            prompt_body=str(d.get("prompt_body", "")),
            meta=dict(d.get("meta") or {}),
        )

    def validate(self) -> None:
        if self.type not in ("script", "agent"):
            raise ValueError(f"invalid job type: {self.type!r}")
        if self.type == "script" and not self.script_body.strip():
            raise ValueError("script job requires non-empty script_body")
        if self.type == "agent" and not self.prompt_body.strip():
            raise ValueError("agent job requires non-empty prompt_body")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")


@dataclass
class JobHandle:
    """Worker-side handle to a claimed job."""
    spec: JobSpec
    job_dir: str
    claimed_at: float
    worker_pid: int


@dataclass
class JobResult:
    status: str
    exit_code: Optional[int] = None
    signal: Optional[int] = None
    duration_s: float = 0.0
    error: str = ""
    job_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "signal": self.signal,
            "duration_s": self.duration_s,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "JobResult":
        return cls(
            job_id=str(d.get("job_id", "")),
            status=str(d.get("status", "")),
            exit_code=int(d["exit_code"]) if d.get("exit_code") is not None else None,
            signal=int(d["signal"]) if d.get("signal") is not None else None,
            duration_s=float(d.get("duration_s", 0.0)),
            error=str(d.get("error", "")),
        )


def new_job_id() -> str:
    return uuid.uuid4().hex


def now_ts() -> float:
    return time.time()
