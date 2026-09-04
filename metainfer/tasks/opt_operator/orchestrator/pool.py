"""Kernel pool — the authoritative append-only store of *admitted* kernels.

This is the **single source of truth** for every kernel that passed the
correctness harness and met the admission score gate (see OPT_KERNEL_SPEC
FR-4). Each admitted kernel carries its own benchmark evidence (per-case
latency) so that scores, the champion, per-kernel speedups and the lineage
graph are all **derived** from this file at read time — never dual-written.

File layout (``kernel_pool.jsonl``): one JSON object per line, chained with a
SHA-256 digest over the previous line's digest + this line's payload, so the
history is immutable and tamper-evident (mirrors the old ledger invariant).

What is authoritative here (stored):
    iteration, kernel_digest, language, contract_digest, parent_iteration,
    case_metrics {case_id: latency_ns}, source_path, admitted_at, note,
    conformance_digest, perf_report_digest, prev_digest, digest

What is **derived** (never stored here — recomputed on read):
    per-case / overall speedup vs baseline, representative latency, quality
    score, champion, lineage, weighting for selection.

The :class:`KernelPool` also implements weighted-probability selection
(:meth:`sample_kernel`) with a caller-supplied ``random.Random`` so a fixed
seed makes a run reproducible/replayable.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class PoolError(ValueError):
    """Corrupt or misconfigured kernel pool."""


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PoolEntry:
    """One admitted kernel with its benchmark evidence."""

    iteration: int
    kernel_digest: str
    language: str
    contract_digest: str
    parent_iteration: Optional[int]
    case_latency_ns: Dict[str, float]       # case_id -> measured latency
    source_path: str = ""
    admitted_at: float = 0.0
    note: str = ""
    conformance_digest: str = ""
    perf_report_digest: str = ""
    prev_digest: Optional[str] = None
    digest: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "PoolEntry":
        if not isinstance(data, dict):
            raise PoolError("PoolEntry must be a mapping")
        try:
            return cls(
                iteration=int(data["iteration"]),
                kernel_digest=str(data["kernel_digest"]),
                language=str(data["language"]),
                contract_digest=str(data["contract_digest"]),
                parent_iteration=data.get("parent_iteration"),
                case_latency_ns={
                    str(k): float(v)
                    for k, v in (data.get("case_latency_ns") or {}).items()
                },
                source_path=str(data.get("source_path") or ""),
                admitted_at=float(data.get("admitted_at") or 0.0),
                note=str(data.get("note") or ""),
                conformance_digest=str(data.get("conformance_digest") or ""),
                perf_report_digest=str(data.get("perf_report_digest") or ""),
                prev_digest=data.get("prev_digest"),
                digest=str(data.get("digest") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PoolError(f"bad PoolEntry: {exc}") from exc


# --------------------------------------------------------------------------- #
# Chained digest + representative metrics
# --------------------------------------------------------------------------- #

def _chain_digest(prev: Optional[str], payload: str) -> str:
    blob = (prev or "") + "\n" + payload
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _payload(entry: PoolEntry) -> str:
    return json.dumps({
        "iteration": entry.iteration,
        "kernel_digest": entry.kernel_digest,
        "language": entry.language,
        "contract_digest": entry.contract_digest,
        "parent_iteration": entry.parent_iteration,
        "case_latency_ns": entry.case_latency_ns,
        "source_path": entry.source_path,
        "admitted_at": entry.admitted_at,
        "note": entry.note,
        "conformance_digest": entry.conformance_digest,
        "perf_report_digest": entry.perf_report_digest,
    }, sort_keys=True)


def compute_digest(entry: PoolEntry) -> str:
    """The entry's digest given its fields (before ``digest`` is set)."""
    return _chain_digest(entry.prev_digest, _payload(entry))


def rep_latency(entry: PoolEntry) -> float:
    """Representative latency for an entry over its measured cases.

    Default: geometric mean of per-case latencies (lower = better). A single
    measured case therefore reports that case's latency. Callers that want a
    targeted-shape metric (single shape) can pass a subset of cases into
    :meth:`KernelPool.rep_latency_for`.
    """
    return _geomean(list(entry.case_latency_ns.values()))


def _geomean(values: List[float]) -> float:
    if not values:
        return float("inf")
    logsum = sum(math.log(max(v, 1e-12)) for v in values)
    return math.exp(logsum / len(values))


# --------------------------------------------------------------------------- #
# Pool (authoritative store)
# --------------------------------------------------------------------------- #

class KernelPool:
    """Append-only, chain-verified store of admitted kernels in a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    # -- read ------------------------------------------------------------ #

    def read_all(self) -> List[PoolEntry]:
        if not self.path.exists():
            return []
        out: List[PoolEntry] = []
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = PoolEntry.from_dict(json.loads(line))
                expect = compute_digest(entry)
                if expect != entry.digest:
                    raise PoolError(
                        f"pool corruption at line {lineno}: digest mismatch")
                if (entry.prev_digest or "") != (out[-1].digest if out else ""):
                    raise PoolError(
                        f"pool corruption at line {lineno}: broken chain")
                out.append(entry)
        return out

    def current_pool(self) -> List[PoolEntry]:
        """All admitted kernels (alias of :meth:`read_all`)."""
        return self.read_all()

    def baseline(self) -> Optional[PoolEntry]:
        """The genesis / baseline kernel (speedup denominator)."""
        entries = self.read_all()
        return entries[0] if entries else None

    def champion(self) -> Optional[PoolEntry]:
        """Derived champion: the admitted kernel with the lowest representative
        latency; ties resolve to the most recently admitted."""
        entries = self.read_all()
        if not entries:
            return None
        best: Optional[PoolEntry] = None
        best_lat = float("inf")
        for e in entries:
            lat = rep_latency(e)
            # `<=` so a tie picks the later entry (most recent best).
            if lat <= best_lat:
                best, best_lat = e, lat
        return best

    def rep_latency_for(self, entry: PoolEntry,
                        cases: Optional[List[str]] = None) -> float:
        """Representative latency restricted to ``cases`` (targeted shape set),
        or all measured cases when ``cases`` is None."""
        if cases:
            lat = [entry.case_latency_ns[c] for c in cases
                   if c in entry.case_latency_ns]
            if not lat:
                return float("inf")
            return _geomean(lat)
        return rep_latency(entry)

    def quality(self, entry: PoolEntry, baseline: Optional[PoolEntry] = None,
                cases: Optional[List[str]] = None) -> float:
        """Higher-is-better quality = baseline_rep_latency / this rep_latency.

        Baseline (or genesis) has quality 1.0; a kernel half the baseline's
        representative latency has quality 2.0. Falls back to genesis when
        ``baseline`` is None."""
        base = baseline if baseline is not None else self.baseline()
        if base is None:
            return 1.0
        base_lat = self.rep_latency_for(base, cases)
        this_lat = self.rep_latency_for(entry, cases)
        if this_lat in (0.0, float("inf")) or not math.isfinite(this_lat):
            return 0.0
        return base_lat / this_lat

    def speedup_vs_baseline(self, entry: PoolEntry,
                            cases: Optional[List[str]] = None) -> float:
        return self.quality(entry, cases=cases)

    def lineage(self) -> List[PoolEntry]:
        """Derived lineage: the champion's ancestry (genesis -> champion) via
        ``parent_iteration`` links."""
        champ = self.champion()
        if champ is None:
            return []
        by_iter = {e.iteration: e for e in self.read_all()}
        chain: List[PoolEntry] = []
        seen: set = set()
        cur: Optional[PoolEntry] = champ
        while cur is not None and cur.iteration not in seen:
            seen.add(cur.iteration)
            chain.append(cur)
            cur = by_iter.get(cur.parent_iteration) if cur.parent_iteration is not None else None
        return list(reversed(chain))

    # -- write ----------------------------------------------------------- #

    def admit(self, entry: PoolEntry) -> PoolEntry:
        """Append one admitted kernel. Returns the finalized entry (digest set)."""
        entries = self.read_all()
        prev = entries[-1].digest if entries else None
        finalized = PoolEntry(
            iteration=entry.iteration,
            kernel_digest=entry.kernel_digest,
            language=entry.language,
            contract_digest=entry.contract_digest,
            parent_iteration=entry.parent_iteration,
            case_latency_ns=entry.case_latency_ns,
            source_path=entry.source_path,
            admitted_at=entry.admitted_at or time.time(),
            note=entry.note,
            conformance_digest=entry.conformance_digest,
            perf_report_digest=entry.perf_report_digest,
            prev_digest=prev,
            digest="",
        )
        final_digest = compute_digest(finalized)
        finalized = PoolEntry(
            **{**finalized.as_dict(), "digest": final_digest})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(finalized.as_dict()) + "\n")
        return finalized

    # -- weighted-probability selection -------------------------------- #

    def sample_kernel(self, rng, cases: Optional[List[str]] = None,
                      weight_power: float = 1.0) -> Optional[PoolEntry]:
        """Pick an admitted kernel by quality-weighted probability.

        Weight ∝ ``quality^weight_power``: higher-quality kernels are more
        likely to be selected for the next optimization round. ``rng`` is a
        ``random.Random`` so a fixed seed makes selection reproducible. Returns
        None on an empty pool.
        """
        entries = self.read_all()
        if not entries:
            return None
        weights: List[float] = []
        for e in entries:
            q = self.quality(e, cases=cases)
            weights.append(max(0.0, q) ** weight_power if q > 0 else 0.0)
        total = sum(weights)
        if total <= 0 or not math.isfinite(total):
            # All degenerate (e.g. empty metric) — fall back to uniform so a
            # misconfigured metric never wedges selection.
            return rng.choice(entries)
        r = rng.random() * total
        acc = 0.0
        for e, w in zip(entries, weights):
            acc += w
            if r <= acc:
                return e
        return entries[-1]


__all__ = ["PoolError", "PoolEntry", "KernelPool", "compute_digest",
           "rep_latency"]
