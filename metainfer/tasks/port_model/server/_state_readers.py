"""Read-only access to port-model on-disk state.

Called by the API routes. All reads are defensive: missing files
return None / empty defaults rather than raising.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.server import state_reader as _sr


def _load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    """Return all iteration records sorted by iteration number."""
    iters_dir = state_dir / "iterations"
    if not iters_dir.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(iters_dir.glob("*.json")):
        data = _load_json(p, None)
        if data is not None:
            data.setdefault("iteration", int(p.stem))
            out.append(data)
    return out


def read_iteration(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "iterations" / f"{n:03d}.json", None)


def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    """Build the state-graph payload from run.json + phases."""
    run = _sr.read_run(state_dir)
    try:
        from ..orchestrator.phases import graph_payload
    except ImportError:
        return {"current": "idle", "nodes": [], "edges": []}
    return graph_payload(
        current=run.get("current_phase", "idle"),
        last_outcome=run.get("last_outcome"),
        last_label=run.get("last_transition_label"),
    )


def read_memory_markdown(workspace_dir: Path, name: str) -> Optional[str]:
    """Read a memory/<name>.md file (canonical consolidated artifacts)."""
    path = workspace_dir / "memory" / f"{name}.md"
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def read_phase_summary(workspace_dir: Path, phase: str) -> Optional[str]:
    """Read a phase workdir's summary.md."""
    phase_dirs = {
        "P1_weight_analysis": "p1",
        "P2_framework_analysis": "p2",
        "P3_architect_review": "p3",
        "P4_minimal_framework": "p4",
        "P5_verify_minimal": "p5",
        "P6_port_engine": "p6",
    }
    sub = phase_dirs.get(phase)
    if sub is None:
        return None
    summary_path = workspace_dir / sub / "summary.md"
    if not summary_path.is_file():
        return None
    try:
        return summary_path.read_text(encoding="utf-8")
    except OSError:
        return None


def read_p5_dumps_index(workspace_dir: Path) -> Dict[str, Any]:
    """List hidden_state dumps produced by the minimal framework."""
    dumps_dir = workspace_dir / "dumps"
    if not dumps_dir.is_dir():
        return {"configured": False, "dumps": []}
    dumps = sorted(p.name for p in dumps_dir.glob("layer_*.npy"))
    return {"configured": True, "dumps_dir": str(dumps_dir), "dumps": dumps}


def read_p6_iterations(workspace_dir: Path) -> List[Dict[str, Any]]:
    """List P6 attempt dirs + their verdict JSONs."""
    p6_root = workspace_dir / "p6"
    if not p6_root.is_dir():
        return []
    out: List[Dict[str, Any]] = []
    for iter_dir in sorted(p6_root.glob("iter_*")):
        verdict_files = sorted(iter_dir.glob("verdict_*.json"))
        verdict: Dict[str, Any] = {}
        if verdict_files:
            try:
                verdict = json.loads(verdict_files[-1].read_text(encoding="utf-8"))
            except (ValueError, OSError):
                verdict = {}
        commit_files = sorted(iter_dir.glob("commit_*.txt"))
        commit_sha = None
        if commit_files:
            try:
                commit_sha = commit_files[-1].read_text(encoding="utf-8").strip()
            except OSError:
                commit_sha = None
        summary_path = iter_dir / "summary.md"
        summary = None
        if summary_path.is_file():
            try:
                summary = summary_path.read_text(encoding="utf-8")
            except OSError:
                summary = None
        out.append({
            "dir": str(iter_dir),
            "name": iter_dir.name,
            "verdict": verdict,
            "commit_sha": commit_sha,
            "summary": summary,
        })
    return out
