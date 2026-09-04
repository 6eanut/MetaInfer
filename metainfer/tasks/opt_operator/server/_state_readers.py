"""State-dir readers for the opt_operator task type.

Backs the overview (macro: phase graph + champion lineage + reference origin +
GPU pool + summary) and the per-iteration drill-in (conformance + latency). Reads
only — never writes — and never caches; everything is derived from the SSOT files
(run.json, kernel_pool.jsonl, system_oracle/*/oracle.json, iterations/*.json).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..orchestrator import phases as _phases
from ..orchestrator.ledger import ChampionLedger
from ..orchestrator.pool import KernelPool, rep_latency


def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return default


def _iso(ts: float) -> Optional[str]:
    if not ts:
        return None
    try:
        import datetime as _dt
        return _dt.datetime.fromtimestamp(
            ts, _dt.timezone.utc).isoformat(timespec="seconds")
    except Exception:  # noqa: BLE001 — a bad ts shouldn't 500 the overview
        return None


def _iteration_index(state_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Map iteration number -> its persisted record (iterations/<NNN>.json)."""
    iters_dir = state_dir / "iterations"
    if not iters_dir.exists():
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for p in sorted(iters_dir.glob("*.json")):
        data = _load_json(p, None)
        it = data.get("iteration") if isinstance(data, dict) else None
        if isinstance(it, int) and data is not None:
            out[it] = data
    return out


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
    # Authoritative kernel pool; ChampionLedger is the derived lineage view.
    ledger = ChampionLedger(state_dir / "kernel_pool.jsonl")
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


def read_pool(state_dir: Path) -> Dict[str, Any]:
    """Full pool view — every admitted kernel, not just the champion's ancestry.

    Each row carries its derived quality / speedup-vs-baseline plus the round's
    benchmark 口径 (statistic/reps/warmup/shape count) so a score is never a
    bare number: the harness that produced it is annotated alongside. All
    derived at read time from kernel_pool.jsonl + iterations/*.json.
    """
    pool = KernelPool(state_dir / "kernel_pool.jsonl")
    try:
        entries = pool.read_all()
    except Exception:  # noqa: BLE001 — a corrupt pool shouldn't 500 the UI
        return {"pool": [], "baseline_rep_latency_ns": None, "pool_size": 0}
    if not entries:
        return {"pool": [], "baseline_rep_latency_ns": None, "pool_size": 0}
    recs = _iteration_index(state_dir)
    baseline = pool.baseline()
    baseline_rep = rep_latency(baseline) if baseline else None
    champ = pool.champion()
    champ_iter = champ.iteration if champ else None
    lineage_iters = {e.iteration for e in pool.lineage()}
    rows: List[Dict[str, Any]] = []
    for e in entries:
        rec = recs.get(e.iteration) or {}
        bmeta = rec.get("benchmark_meta") or {}
        quality = pool.quality(e)
        rows.append({
            "iteration": e.iteration,
            "kernel_digest": (e.kernel_digest or "")[:16],
            "language": e.language,
            "parent_iteration": e.parent_iteration,
            "admitted_at": _iso(e.admitted_at),
            "note": e.note,
            "case_count": len(e.case_latency_ns),
            "rep_latency_ns": rep_latency(e),
            "quality": quality,
            "speedup_vs_baseline": quality,
            "is_champion": e.iteration == champ_iter,
            "on_lineage": e.iteration in lineage_iters,
            # benchmark harness 口径 for this kernel's score
            "statistic": bmeta.get("statistic"),
            "reps": bmeta.get("reps"),
            "warmup": bmeta.get("warmup"),
            "shape_count": len(bmeta.get("shape_ids") or []),
        })
    # Top-quality first for the pool-top view; stable on iteration ties.
    rows.sort(key=lambda r: (-(r["quality"] or 0.0), r["iteration"]))
    return {"pool": rows, "baseline_rep_latency_ns": baseline_rep,
            "pool_size": len(rows)}


def read_harness_reviews(state_dir: Path) -> Dict[str, Any]:
    """The harness self-proof (FR-2/3): did the correctness gate catch every
    adversarially-wrong candidate, and is the benchmark methodology sound?

    Surfaces the harness_setup review record (the first round record carrying a
    ``reviews`` dict). Negative-case evidence is the *trust* story — a gate that
    flagged every injected error is one we believe. Caliber checks (shape set,
    warmup, reps, statistic, baseline) are listed item-by-item.
    """
    recs = _iteration_index(state_dir)
    setup_rec: Optional[Dict[str, Any]] = None
    for n in sorted(recs):
        if recs[n].get("reviews"):
            setup_rec = recs[n]
            break
    if not setup_rec:
        return {"present": False}
    reviews = setup_rec.get("reviews") or {}
    corr = reviews.get("correctness") or {}
    bench = reviews.get("benchmark") or {}
    negatives = corr.get("negative_evidence") or []
    return {
        "present": True,
        "iteration": setup_rec.get("iteration"),
        "correctness": {
            "passed": corr.get("passed"),
            "message": corr.get("message"),
            "checks": corr.get("checks") or [],
            "negatives_total": len(negatives),
            "negatives_caught": sum(1 for e in negatives
                                    if e.get("harness_caught")),
            "negative_evidence": negatives,
        },
        "benchmark": {
            "passed": bench.get("passed"),
            "message": bench.get("message"),
            "checks": bench.get("checks") or [],
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
        "outcome": rec.get("outcome", ""),
        "admitted": rec.get("admitted", False),
        "candidate_digest": rec.get("candidate_digest"),
    }


__all__ = ["read_run", "read_state_graph", "read_lineage", "read_overview",
           "read_pool", "read_harness_reviews", "read_iterations",
           "read_iteration", "read_conformance"]
