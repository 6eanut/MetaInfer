"""Iteration record schema for port-model.

An "iteration" here is the running index of one full pass through the
state machine. Each phase records what it produced (artifacts) and its
outcome; the record is appended to / replaced in
``iterations/<NNN>.json`` after each phase completes.

The WebUI's "iterations" panel renders this directly.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class PhaseRecord:
    phase: str
    outcome: Optional[str] = None
    started_at: float = 0.0
    ended_at: float = 0.0
    duration_s: float = 0.0
    agent_name: Optional[str] = None
    summary_path: Optional[str] = None
    summary_excerpt: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    error: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IterationRecord:
    iteration: int
    goal: str = ""
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    duration_s: Optional[float] = None
    status: str = "running"  # "running" | "success" | "failed" | "aborted"
    phases: Dict[str, PhaseRecord] = field(default_factory=dict)
    # Final-phase artifacts the WebUI cares about:
    p1_artifact: Optional[str] = None       # p1_weight_analysis.md
    p2_artifacts: List[str] = field(default_factory=list)  # per ref source
    p3_artifact: Optional[str] = None       # p3_consolidated_spec.md
    p4_artifact: Optional[str] = None       # p4 run.py dir
    p5_verdict: Optional[Dict[str, Any]] = None
    p5_dumps_dir: Optional[str] = None
    p6_iterations: List[Dict[str, Any]] = field(default_factory=list)
    p6_commit_shas: List[str] = field(default_factory=list)
    final_status: Optional[str] = None
    failure_reason: Optional[str] = None

    def upsert_phase(self, ph: PhaseRecord) -> None:
        self.phases[ph.phase] = ph

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d


def write_iteration(state_dir: Path, rec: IterationRecord) -> Path:
    """Atomically write the iteration record to ``iterations/<NNN>.json``."""
    iters_dir = state_dir / "iterations"
    iters_dir.mkdir(parents=True, exist_ok=True)
    path = iters_dir / f"{rec.iteration:03d}.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def read_iteration(state_dir: Path, n: int) -> Optional[IterationRecord]:
    path = state_dir / "iterations" / f"{n:03d}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    # Re-hydrate PhaseRecord sub-dicts.
    phases_data = data.pop("phases", {}) or {}
    phases = {k: PhaseRecord(**v) for k, v in phases_data.items() if isinstance(v, dict)}
    return IterationRecord(**data, phases=phases) if phases else IterationRecord(**data)


def next_iteration_number(state_dir: Path) -> int:
    """Return the next iteration number (1-based) by scanning iterations/."""
    iters_dir = state_dir / "iterations"
    if not iters_dir.is_dir():
        return 1
    existing = []
    for p in iters_dir.glob("*.json"):
        try:
            existing.append(int(p.stem))
        except ValueError:
            continue
    return (max(existing) + 1) if existing else 1


def read_summary_excerpt(path: Optional[str], max_chars: int = 400) -> Optional[str]:
    """Return the first paragraph of a summary.md for the UI."""
    if not path:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    return text[:max_chars]
