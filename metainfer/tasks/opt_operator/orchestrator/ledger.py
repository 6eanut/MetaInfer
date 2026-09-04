"""Champion lineage view over the authoritative kernel pool (derived).

**SSOT note:** the append-only authority for admitted kernels is
``kernel_pool.jsonl`` written by :class:`pool.KernelPool` (see OPT_KERNEL_SPEC
FR-4 / §4). The pool records *every* admitted kernel with its benchmark
evidence; the "champion chain" that used to be a standalone ledger is now a
**derived view** over that pool. This module keeps the historical
:class:`ChampionLedger` name and method surface (``append`` /
``read_all`` / ``current_champion`` / ``lineage``) as a thin facade so callers
and the chained-digest tamper guarantee are preserved, but the underlying file
is the pool and champion/lineage/speedups are recomputed at read time — nothing
is dual-written.

:class:`LedgerEntry` / :class:`CaseMetric` are retained as the compatibility
shape returned to older callers (notably the pipeline's incumbent tracking).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .pool import KernelPool, PoolEntry, PoolError


class LedgerError(ValueError):
    pass


@dataclass(frozen=True)
class CaseMetric:
    latency_ns: float
    speedup: Optional[float] = None    # derived vs the kernel's parent

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LedgerEntry:
    iteration: int
    kernel_digest: str
    language: str
    contract_digest: str
    parent_iteration: Optional[int]
    case_metrics: Dict[str, CaseMetric]   # case_id -> metric
    report_digest: str = ""
    conformance_digest: str = ""
    prev_digest: Optional[str] = None
    digest: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "LedgerEntry":
        if not isinstance(data, dict):
            raise LedgerError("LedgerEntry must be a mapping")
        try:
            return cls(
                iteration=int(data["iteration"]),
                kernel_digest=str(data["kernel_digest"]),
                language=str(data["language"]),
                contract_digest=str(data["contract_digest"]),
                parent_iteration=data.get("parent_iteration"),
                case_metrics={
                    k: CaseMetric(latency_ns=float(v["latency_ns"]),
                                  speedup=v.get("speedup"))
                    for k, v in (data.get("case_metrics") or {}).items()
                },
                report_digest=str(data.get("report_digest") or ""),
                conformance_digest=str(data.get("conformance_digest") or ""),
                prev_digest=data.get("prev_digest"),
                digest=str(data.get("digest") or ""),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerError(f"bad LedgerEntry: {exc}") from exc


# --------------------------------------------------------------------------- #
# Pool <-> Ledger conversion
# --------------------------------------------------------------------------- #

def _to_pool(entry: "LedgerEntry") -> PoolEntry:
    return PoolEntry(
        iteration=entry.iteration,
        kernel_digest=entry.kernel_digest,
        language=entry.language,
        contract_digest=entry.contract_digest,
        parent_iteration=entry.parent_iteration,
        case_latency_ns={k: m.latency_ns for k, m in entry.case_metrics.items()},
        conformance_digest=entry.conformance_digest,
        perf_report_digest=entry.report_digest,
        prev_digest=None,
        digest="",
    )


def _from_pool(entry: PoolEntry,
               parent_lat: Optional[Dict[str, float]]) -> "LedgerEntry":
    metrics: Dict[str, CaseMetric] = {}
    for cid, lat in entry.case_latency_ns.items():
        speedup = None
        if parent_lat and cid in parent_lat and parent_lat[cid]:
            speedup = parent_lat[cid] / lat if lat else None
        metrics[cid] = CaseMetric(latency_ns=lat, speedup=speedup)
    return LedgerEntry(
        iteration=entry.iteration,
        kernel_digest=entry.kernel_digest,
        language=entry.language,
        contract_digest=entry.contract_digest,
        parent_iteration=entry.parent_iteration,
        case_metrics=metrics,
        report_digest=entry.perf_report_digest,
        conformance_digest=entry.conformance_digest,
        prev_digest=entry.prev_digest,
        digest=entry.digest,
    )


# --------------------------------------------------------------------------- #
# ChampionLedger facade
# --------------------------------------------------------------------------- #

class ChampionLedger:
    """Derived champion/lineage view over the authoritative kernel pool.

    Construct with the **pool file path** (``kernel_pool.jsonl``). Appends are
    delegated to :class:`KernelPool.admit` (the single write path).
    """

    def __init__(self, path: Path) -> None:
        self._pool = KernelPool(Path(path))

    @property
    def pool(self) -> KernelPool:
        """Expose the underlying authoritative store for pool-aware callers."""
        return self._pool

    def read_all(self) -> List[LedgerEntry]:
        try:
            entries = self._pool.read_all()
        except PoolError as exc:
            raise LedgerError(str(exc)) from exc
        return _convert_chain(entries)

    def current_champion(self) -> Optional[LedgerEntry]:
        try:
            champ = self._pool.champion()
        except PoolError as exc:
            raise LedgerError(str(exc)) from exc
        if champ is None:
            return None
        return _from_pool(champ, _parent_latency(self._pool, champ))

    def lineage(self) -> List[LedgerEntry]:
        try:
            chain = self._pool.lineage()
        except PoolError as exc:
            raise LedgerError(str(exc)) from exc
        return _convert_chain(chain)

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        """Admit one kernel into the pool. Returns the finalized LedgerEntry."""
        finalized = self._pool.admit(_to_pool(entry))
        return _from_pool(finalized, _parent_latency(self._pool, finalized))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _convert_chain(entries: List[PoolEntry]) -> List[LedgerEntry]:
    by_iter = {e.iteration: e for e in entries}
    parent_lat: Dict[int, Dict[str, float]] = {}
    for e in entries:
        if e.parent_iteration is not None and e.parent_iteration in by_iter:
            parent_lat[e.iteration] = by_iter[e.parent_iteration].case_latency_ns
    return [_from_pool(e, parent_lat.get(e.iteration)) for e in entries]


def _parent_latency(pool: KernelPool, entry: PoolEntry):
    if entry.parent_iteration is None:
        return None
    try:
        parent = next(e for e in pool.read_all()
                      if e.iteration == entry.parent_iteration)
        return parent.case_latency_ns
    except StopIteration:
        return None


__all__ = ["LedgerError", "CaseMetric", "LedgerEntry", "ChampionLedger"]
