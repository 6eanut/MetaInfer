"""Iteration record schema for the opt_operator run.

One record is persisted per phase-step via :meth:`StateStore.write_iteration`
(``iterations/<NNN>.json``). The shell treats these as opaque dicts; only this
task's ``_state_readers.py`` and detail view know the schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IterationRecord:
    iteration: int
    phase: str                        # which phase produced this record
    status: str = "running"           # "running" | "success" | "failed"
    started_at: float = 0.0
    ended_at: float = 0.0
    notes: List[str] = field(default_factory=list)
    # task-specific evidence -------------------------------------------------
    plan: Optional[Dict[str, Any]] = None
    candidate_source: Optional[str] = None
    candidate_language: Optional[str] = None
    candidate_digest: Optional[str] = None
    conformance: Optional[Dict[str, Any]] = None   # ConformanceReport.as_dict()
    perf: Optional[Dict[str, Any]] = None          # case_id -> {latency_ns, speedup}
    promoted: bool = False
    guidance: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["IterationRecord"]
