"""Shared, versioned, auto-reviewed operator reference library.

This is the **single source of truth** for correctness references across runs
(see plan §2). Correctness-reference resolution order:

    1. User provides a reference this run          -> freeze it, use directly.
    2. Library hit for the operator                -> freeze its reference, use.
    3. Neither                                     -> generate a reference, run the
                                                       **full-auto strict review gate**;
                                                       on pass atomically admit to the
                                                       library AND freeze for this run;
                                                       on fail freeze per-run only (never
                                                       admitted).

The library lives in a shared runtime dir
``$METAINFER_ROOT/opt_operator/reference_library/<op_id>/`` with:

    contract.yaml      # OperatorContract dict
    reference.py       # forward(**tensors) -> dict[str, np.ndarray]
    baseline.<ext>     # optional naive-correct HIP/Triton baseline (mode-B champion)
    REVIEW.md          # human-readable record of the admission review
    meta.json          # op_id, digest, admitted_at, review report digest

Admission is **atomic + single-writer** using ``os.link``-based claims
(``fs_primitives.link_claim``) — safe across hosts on NFS. Two writers racing to
admit the same op: exactly one wins the lock, writes, releases; the other either
re-checks (already admitted) or fails cleanly.

This module is pure logic. Reference *execution* is delegated to an injectable
:class:`oracle.ReferenceExecutor` so the review gate is unit-testable without
numpy/torch.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.cluster.fs_primitives import atomic_write_text

from .contract import OperatorContract
from .oracle import (
    FrozenOracle,
    ReferenceExecutor,
    digest_reference,
    freeze_reference,
    load_oracle,
    write_oracle_artifacts,
)


class ReferenceLibError(ValueError):
    """Library misconfiguration or a failed admission."""


# --------------------------------------------------------------------------- #
# Structured review report
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReferenceCheck:
    name: str
    passed: bool
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReviewReport:
    passed: bool
    checks: List[ReferenceCheck]
    message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [c.as_dict() for c in self.checks],
            "message": self.message,
        }

    def check(self, name: str) -> Optional[ReferenceCheck]:
        for c in self.checks:
            if c.name == name:
                return c
        return None


# --------------------------------------------------------------------------- #
# Reference entry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ReferenceEntry:
    """A correctness reference (candidate or admitted) for one operator."""

    op_id: str
    contract: Dict[str, Any]
    reference_source: str
    baseline_language: Optional[str] = None   # "hip" | "triton"
    baseline_source: Optional[str] = None
    digest: str = ""

    def __post_init__(self) -> None:
        if not self.digest:
            object.__setattr__(
                self, "digest",
                digest_reference(self.op_id, self.contract, self.reference_source),
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "ReferenceEntry":
        if not isinstance(data, dict):
            raise ReferenceLibError("ReferenceEntry must be a mapping")
        try:
            return cls(
                op_id=str(data["op_id"]),
                contract=dict(data["contract"]),
                reference_source=str(data["reference_source"]),
                baseline_language=data.get("baseline_language"),
                baseline_source=data.get("baseline_source"),
                digest=str(data.get("digest") or ""),
            )
        except KeyError as exc:
            raise ReferenceLibError(f"ReferenceEntry missing field {exc}") from exc


# --------------------------------------------------------------------------- #
# Generic numerical comparison (works on numpy arrays OR nested python lists)
# --------------------------------------------------------------------------- #

from ._compare import allclose as _allclose, is_finite as _is_finite, shape_of as _shape_of


# --------------------------------------------------------------------------- #
# Full-auto strict review gate
# --------------------------------------------------------------------------- #

def _run_cases(
    contract: OperatorContract, executor: ReferenceExecutor, reference_source: str,
):
    """Yield (case, outputs) for every case in the contract's shape-sweep."""
    for case in contract.generate_cases():
        outputs = executor.run(reference_source, contract, case.dims)
        yield case, outputs


def review_reference(
    contract: OperatorContract, executor: ReferenceExecutor, reference_source: str,
) -> ReviewReport:
    """Run the full-auto strict review gate over the contract's case matrix.

    Checks:
      1. ``deterministic`` — running the reference twice yields identical outputs.
      2. ``shapes_match``   — output tensor shapes equal the contract's declared
                             output shapes for every case.
      3. ``finite``         — every output element is finite (no NaN/Inf).
      4. ``all_cases_pass`` — the reference runs without error across the sweep.

    A reference only passes if **every** check passes on **every** case.
    """
    checks: List[ReferenceCheck] = []
    cases = list(contract.generate_cases())
    if not cases:
        return ReviewReport(False, [
            ReferenceCheck("case_matrix", False, "contract produced no cases"),
        ])

    # 1. determinism (run first case twice)
    first = cases[0]
    try:
        o1 = executor.run(reference_source, contract, first.dims)
        o2 = executor.run(reference_source, contract, first.dims)
    except Exception as exc:  # noqa: BLE001 — a crash is a failed check
        return ReviewReport(False, [
            ReferenceCheck("runs_without_error", False, f"first case crashed: {exc}"),
        ], message="reference raised on first case")
    if set(o1) != set(o2):
        return ReviewReport(False, [
            ReferenceCheck("deterministic", False, "output keys differ between runs"),
        ])
    det_ok = all(_allclose(o1[k], o2[k]) for k in o1)
    checks.append(ReferenceCheck("deterministic", det_ok,
                                 "outputs identical across runs" if det_ok
                                 else "outputs differ between runs"))

    # 2/3/4 across all cases
    shapes_ok, finite_ok, run_ok = True, True, True
    shape_detail, run_detail = "", ""
    for case, outputs in _run_cases(contract, executor, reference_source):
        # shapes match contract outputs
        for t in contract.outputs:
            got = _shape_of(outputs.get(t.name))
            expect = t.resolved_shape(case.dims)
            if got != expect:
                shapes_ok = False
                shape_detail = f"{t.name}: got {got} want {expect}"
        # finite
        for k, v in outputs.items():
            if not _is_finite(v):
                finite_ok = False

    checks.append(ReferenceCheck("shapes_match", shapes_ok,
                                 shape_detail or f"all {len(cases)} cases match"))
    checks.append(ReferenceCheck("finite", finite_ok,
                                 "all outputs finite" if finite_ok else "non-finite output found"))
    checks.append(ReferenceCheck("all_cases_pass", run_ok,
                                 run_detail or f"all {len(cases)} cases ran"))

    passed = all(c.passed for c in checks)
    return ReviewReport(passed, checks, message="" if passed else "review gate failed")


# --------------------------------------------------------------------------- #
# Reference library (shared dir + atomic admission)
# --------------------------------------------------------------------------- #

def make_reference_entry(
    contract: OperatorContract, reference_source: str, **kw: Any,
) -> ReferenceEntry:
    """Build a :class:`ReferenceEntry` from a contract + reference source."""
    return ReferenceEntry(
        op_id=op_id_for(contract),
        contract=contract.to_dict(),
        reference_source=reference_source,
        **kw,
    )


def op_id_for(contract: OperatorContract) -> str:
    """Stable library id derived from the contract name (slugified)."""
    name = contract.name.strip()
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", name).strip("_").lower()
    return slug or "op"


class ReferenceLibrary:
    """Manages the shared, versioned operator reference library."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    # -- layout ----------------------------------------------------------- #

    def entry_dir(self, op_id: str) -> Path:
        return self.root / op_id

    def _meta_path(self, op_id: str) -> Path:
        return self.entry_dir(op_id) / "meta.json"

    # -- queries ---------------------------------------------------------- #

    def find(self, op_id: str) -> Optional[ReferenceEntry]:
        meta = self._meta_path(op_id)
        if not meta.exists():
            return None
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        entry = ReferenceEntry.from_dict(data)
        # integrity: the on-disk reference.py must hash to the recorded digest
        ref_path = self.entry_dir(op_id) / "reference.py"
        if ref_path.exists():
            on_disk = ref_path.read_text(encoding="utf-8")
            if digest_reference(op_id, entry.contract, on_disk) != entry.digest:
                return None
        return entry

    def list_op_ids(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )

    # -- admission (atomic, single-writer) -------------------------------- #

    def admit(self, entry: ReferenceEntry, report: ReviewReport) -> bool:
        """Atomically write ``entry`` into the library under an os.link claim.

        Returns ``True`` if this caller admitted it; ``False`` if a concurrent
        writer already holds the lock (caller should re-check / skip). Raises
        :class:`ReferenceLibError` on malformed input or write failure.
        """
        if not report.passed:
            raise ReferenceLibError("refusing to admit an entry that failed review")
        lock_path = self.entry_dir(entry.op_id).with_name(f".{entry.op_id}.lock")

        from metainfer.cluster.fs_primitives import link_claim, break_claim
        if not link_claim(lock_path, {"holder": "reference_lib", "at": time.time()}):
            return False  # concurrent writer won; caller re-checks find()
        try:
            self._write_entry(entry, report)
            return True
        finally:
            break_claim(lock_path)

    def _write_entry(self, entry: ReferenceEntry, report: ReviewReport) -> None:
        d = self.entry_dir(entry.op_id)
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "op_id": entry.op_id,
            "contract": entry.contract,
            "reference_source": entry.reference_source,
            "baseline_language": entry.baseline_language,
            "baseline_source": entry.baseline_source,
            "digest": entry.digest,
            "admitted_at": time.time(),
            "review": report.as_dict(),
        }
        atomic_write_text(self._meta_path(entry.op_id),
                          json.dumps(meta, indent=2, sort_keys=True))
        atomic_write_text(d / "contract.yaml",
                          __import__("yaml").safe_dump(entry.contract, sort_keys=False))
        atomic_write_text(d / "reference.py", entry.reference_source)
        if entry.baseline_source:
            ext = "hip" if entry.baseline_language == "hip" else "triton"
            atomic_write_text(d / f"baseline.{ext}", entry.baseline_source)
        atomic_write_text(d / "REVIEW.md", _review_md(entry, report))

    # -- resolution (SSOT order + per-run freeze) ------------------------- #

    def resolve(
        self,
        contract: OperatorContract,
        executor: ReferenceExecutor,
        *,
        system_oracle_dir: Path,
        run_id: str,
        user_reference: Optional[str] = None,
        generated_reference: Optional[str] = None,
        baseline_language: Optional[str] = None,
        baseline_source: Optional[str] = None,
    ) -> FrozenOracle:
        """Resolve the oracle for this run following the SSOT resolution order.

        Returns a per-run :class:`FrozenOracle` and writes its artifacts into
        ``system_oracle_dir/<run_id>/``. The returned oracle is what conformance
        compares against — immutable for the run.

        ``generated_reference`` is the LLM-produced reference to review when no
        user/library reference exists. If the review passes it is atomically
        admitted to the library (best-effort) and frozen; if it fails it is frozen
        per-run as ``generated_unadmitted`` (never admitted).
        """
        op_id = op_id_for(contract)
        origin: str
        source: str

        if user_reference is not None:
            source, origin = user_reference, "user"
        else:
            hit = self.find(op_id)
            if hit is not None:
                source, origin = hit.reference_source, "library"
            elif generated_reference is not None:
                entry = ReferenceEntry(
                    op_id=op_id,
                    contract=contract.to_dict(),
                    reference_source=generated_reference,
                    baseline_language=baseline_language,
                    baseline_source=baseline_source,
                )
                report = review_reference(contract, executor, generated_reference)
                if report.passed:
                    try:
                        self.admit(entry, report)
                    except ReferenceLibError:
                        pass  # freeze per-run regardless; admission is best-effort
                    source, origin = generated_reference, "generated"
                else:
                    source, origin = generated_reference, "generated_unadmitted"
            else:
                raise ReferenceLibError(
                    f"no reference available for {op_id!r} (no user ref, no library "
                    "hit, and no generated reference supplied)"
                )

        oracle = freeze_reference(op_id, contract, source, origin)
        run_oracle_dir = Path(system_oracle_dir) / run_id
        write_oracle_artifacts(run_oracle_dir, oracle)
        return oracle

    def reload(self, system_oracle_dir: Path, run_id: str) -> Optional[FrozenOracle]:
        """Cold-restart: reload this run's frozen oracle if it exists."""
        return load_oracle(Path(system_oracle_dir) / run_id)


def _review_md(entry: ReferenceEntry, report: ReviewReport) -> str:
    lines = [
        f"# Review: {entry.op_id}",
        f"digest: {entry.digest[:16]}",
        "",
        f"passed: {report.passed}",
        "",
    ]
    for c in report.checks:
        lines.append(f"- [{'x' if c.passed else ' '}] {c.name}: {c.detail}")
    if report.message:
        lines.append("")
        lines.append(f"message: {report.message}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ReferenceLibError",
    "ReferenceCheck",
    "ReviewReport",
    "ReferenceEntry",
    "make_reference_entry",
    "op_id_for",
    "ReferenceLibrary",
    "review_reference",
]
