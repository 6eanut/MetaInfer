"""State-dir readers for the opt_operator task type.

Backs the overview (macro: phase graph + champion lineage + reference origin +
GPU pool + summary) and the per-iteration drill-in (conformance + latency). Reads
only — never writes — and never caches; everything is derived from the SSOT files
(run.json, champion_ledger.jsonl, system_oracle/*/oracle.json, iterations/*.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..orchestrator import phases as _phases
from ..orchestrator.ledger import ChampionLedger


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def read_run(state_dir: Path) -> Dict[str, Any]:
    return _load_json(state_dir / "run.json", {}) or {}


def read_state_graph(state_dir: Path) -> Dict[str, Any]:
    run = read_run(state_dir)
    return _phases.graph_payload(
        run.get("current_phase", "idle"),
        run.get("last_outcome"),
        run.get("last_transition_label"),
    )


def _oracle_info(state_dir: Path) -> Dict[str, Any]:
    hits = sorted((state_dir / "system_oracle").glob("*/oracle.json"))
    if not hits:
        return {}
    data = _load_json(hits[-1], {}) or {}
    return {
        "op_id": data.get("op_id"),
        "origin": data.get("origin"),
        "digest": data.get("digest"),
        "created_at": data.get("created_at"),
    }


def read_lineage(state_dir: Path) -> List[Dict[str, Any]]:
    ledger = ChampionLedger(state_dir / "champion_ledger.jsonl")
    try:
        entries = ledger.lineage()
    except Exception:  # noqa: BLE001 — a corrupt ledger shouldn't 500 the overview
        return []
    out: List[Dict[str, Any]] = []
    genesis_latency = None
    for i, e in enumerate(entries):
        latencies = [m.latency_ns for m in e.case_metrics.values() if m.latency_ns]
        avg = (sum(latencies) / len(latencies)) if latencies else None
        best = min(latencies) if latencies else None
        if i == 0:
            genesis_latency = best  # the reference point itself
            speedup = None
        else:
            speedup = (genesis_latency / best) if (best and genesis_latency) else None
        out.append({
            "iteration": e.iteration,
            "kernel_digest": e.kernel_digest,
            "language": e.language,
            "parent_iteration": e.parent_iteration,
            "avg_latency_ns": avg,
            "best_latency_ns": best,
            "speedup_vs_genesis": speedup,
            "case_count": len(e.case_metrics),
        })
    return out


def _gpu_pool(state_dir: Path) -> List[Dict[str, Any]]:
    try:
        from metainfer.cluster import scoreboard
        import os
        return scoreboard.list_claims(node_id=os.environ.get("METAINFER_NODE_ID"))
    except Exception:  # noqa: BLE001 — cluster unavailable is not fatal for the UI
        return []


def read_overview(state_dir: Path) -> Dict[str, Any]:
    run = read_run(state_dir)
    lineage = read_lineage(state_dir)
    champion = lineage[-1] if lineage else None
    genesis = lineage[0] if lineage else None
    return {
        "run": {
            "current_phase": run.get("current_phase", "idle"),
            "current_iteration": run.get("current_iteration", 0),
            "finished": run.get("finished", False),
            "final_status": run.get("final_status"),
        },
        "state_graph": read_state_graph(state_dir),
        "lineage": lineage,
        "reference": _oracle_info(state_dir),
        "gpu_pool": _gpu_pool(state_dir),
        "summary": {
            "champion": champion,
            "genesis": genesis,
            "promotions": len(lineage) - 1 if lineage else 0,
            "speedup_vs_genesis": champion.get("speedup_vs_genesis") if champion else None,
        },
    }


def read_iterations(state_dir: Path) -> List[Dict[str, Any]]:
    iters_dir = state_dir / "iterations"
    if not iters_dir.exists():
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(iters_dir.glob("*.json")):
        data = _load_json(p, None)
        if data is not None:
            out.append(data)
    return out


def read_iteration(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    return _load_json(state_dir / "iterations" / f"{n:03d}.json", None)


def read_conformance(state_dir: Path, n: int) -> Optional[Dict[str, Any]]:
    rec = read_iteration(state_dir, n)
    if rec is None:
        return None
    return {
        "iteration": n,
        "conformance": rec.get("conformance"),
        "perf": rec.get("perf"),
        "promoted": rec.get("promoted", False),
        "candidate_digest": rec.get("candidate_digest"),
    }


__all__ = ["read_run", "read_state_graph", "read_lineage", "read_overview",
           "read_iterations", "read_iteration", "read_conformance"]
