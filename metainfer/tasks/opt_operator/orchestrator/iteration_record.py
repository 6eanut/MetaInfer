"""Iteration record schema for the opt_operator run.

One record is persisted per **round** via :meth:`StateStore.write_iteration`
(``iterations/<NNN>.json``). ``NNN`` is the outer round counter; harness_setup
writes iteration 0 and each optimization round writes its own number. The shell
treats these as opaque dicts; only this task's ``_state_readers.py`` and detail
view know the schema.

A round's fields describe the pool-evolution evidence an auditor needs: which
pool kernel was selected, what candidate was produced, the correctness verdict,
the benchmark latencies, and the round's settled outcome (admitted / discarded /
failed) plus the harness口径 snapshots that back each number.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IterationRecord:
    iteration: int
    phase: str                        # final phase of this round / record
    status: str = "running"           # "running" | "success" | "failed"
    started_at: float = 0.0
    ended_at: float = 0.0
    notes: List[str] = field(default_factory=list)
    outcome: str = ""                 # "admitted" | "discarded" | "failed"
    admitted: bool = False
    # selection context ------------------------------------------------------
    selected_iteration: Optional[int] = None   # pool kernel chosen this round
    selected_digest: Optional[str] = None
    # candidate + verification evidence --------------------------------------
    candidate_source: Optional[str] = None
    candidate_language: Optional[str] = None
    candidate_digest: Optional[str] = None
    repairs: int = 0
    conformance: Optional[Dict[str, Any]] = None   # ConformanceReport.as_dict()
    perf: Optional[Dict[str, Any]] = None          # case_id -> {latency_ns}
    quality: Optional[float] = None                # rep_latency vs baseline
    speedup_vs_baseline: Optional[float] = None
    # harness口径 snapshots (why the numbers are believable) -----------------
    correctness_meta: Optional[Dict[str, Any]] = None
    benchmark_meta: Optional[Dict[str, Any]] = None
    reviews: Optional[Dict[str, Any]] = None       # harness review conclusions
    guidance: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


__all__ = ["IterationRecord"]
