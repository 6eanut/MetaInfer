"""Append-only champion lineage ledger (chained digests).

The ledger records every **promotion** of a candidate to champion for one run.
Entries are appended to a JSONL file and cryptographically chained (each entry's
digest covers the previous entry's digest + its own content), so:

- **Append-only** — a run's history is immutable; cold restart replays the file.
- **Traceable** — the current champion and its full lineage (parent chain) are
  derived purely from the file, no secondary store (SSOT).

Each entry also links the evidence it was promoted on (per-case latency/speedup,
report + conformance digests) so the WebUI lineage curve and per-iteration speedup
are read straight from here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class LedgerError(ValueError):
    pass


@dataclass(frozen=True)
class CaseMetric:
    latency_ns: float
    speedup: Optional[float] = None    # vs the prior champion / baseline

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
        # asdict recursively converts the nested CaseMetric dataclasses too.
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


def _chain_digest(prev: Optional[str], payload: str) -> str:
    blob = (prev or "") + "\n" + payload
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _payload(entry: "LedgerEntry") -> str:
    return json.dumps({
        "iteration": entry.iteration,
        "kernel_digest": entry.kernel_digest,
        "language": entry.language,
        "contract_digest": entry.contract_digest,
        "parent_iteration": entry.parent_iteration,
        "case_metrics": {k: v.as_dict() for k, v in entry.case_metrics.items()},
        "report_digest": entry.report_digest,
        "conformance_digest": entry.conformance_digest,
    }, sort_keys=True)


def compute_digest(entry: "LedgerEntry") -> str:
    """The entry's digest given its fields (before ``digest`` is set)."""
    return _chain_digest(entry.prev_digest, _payload(entry))


class ChampionLedger:
    """Append-only lineage ledger backed by a JSONL file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read_all(self) -> List[LedgerEntry]:
        if not self.path.exists():
            return []
        out: List[LedgerEntry] = []
        with open(self.path, encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                entry = LedgerEntry.from_dict(json.loads(line))
                # verify chain integrity
                expect = compute_digest(entry)
                if expect != entry.digest:
                    raise LedgerError(f"ledger corruption at line {lineno}: digest mismatch")
                if (entry.prev_digest or "") != (out[-1].digest if out else ""):
                    raise LedgerError(f"ledger corruption at line {lineno}: broken chain")
                out.append(entry)
        return out

    def current_champion(self) -> Optional[LedgerEntry]:
        entries = self.read_all()
        return entries[-1] if entries else None

    def lineage(self) -> List[LedgerEntry]:
        """Champion lineage from genesis to current (via parent_iteration chain)."""
        entries = self.read_all()
        by_iter = {e.iteration: e for e in entries}
        # find genesis
        start = self.current_champion()
        if start is None:
            return []
        chain: List[LedgerEntry] = []
        seen = set()
        cur: Optional[LedgerEntry] = start
        while cur is not None and cur.iteration not in seen:
            seen.add(cur.iteration)
            chain.append(cur)
            pid = cur.parent_iteration
            cur = by_iter.get(pid)
        return list(reversed(chain))

    def append(self, entry: LedgerEntry) -> LedgerEntry:
        """Append one promotion. Returns the finalized entry (with digest set)."""
        entries = self.read_all()
        prev = entries[-1].digest if entries else None
        finalized = LedgerEntry(
            iteration=entry.iteration,
            kernel_digest=entry.kernel_digest,
            language=entry.language,
            contract_digest=entry.contract_digest,
            parent_iteration=entry.parent_iteration,
            case_metrics=entry.case_metrics,
            report_digest=entry.report_digest,
            conformance_digest=entry.conformance_digest,
            prev_digest=prev,
            digest="",
        )
        final_digest = compute_digest(finalized)
        finalized = LedgerEntry(
            **{**finalized.as_dict(), "digest": final_digest}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(finalized.as_dict()) + "\n")
        return finalized


__all__ = ["LedgerError", "CaseMetric", "LedgerEntry", "ChampionLedger",
           "compute_digest"]
