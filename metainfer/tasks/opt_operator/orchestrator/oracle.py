"""Reference oracle — load, run, freeze, and cold-restart a correctness reference.

The oracle is the **immutable** numerical reference the conformance gate compares
candidates against. Once a reference is chosen for a run, its source + contract are
**frozen** (SHA-256 digest) into the run's ``system_oracle/`` dir, so later changes
to the shared reference library can never alter how a historical run was validated.

Two concerns live here:

1. **Pure serialization** (:class:`FrozenOracle`, :func:`freeze_reference`,
   :func:`write_oracle_artifacts`, :func:`load_oracle`) — no numpy, directly
   testable.
2. **Reference execution** (:class:`ReferenceExecutor` + a lazy numpy impl) — runs
   a reference ``forward(...)`` over concrete dims to produce oracle outputs. This is
   injectable so tests can supply a pure-Python fake instead of numpy/torch.

The reference source protocol (a ``reference.py`` in the library / user submission):

    def forward(**tensors) -> dict[str, np.ndarray]:
        \"\"\"tensors keyed by contract *input* name; returns outputs keyed by
        contract *output* name.\"\"\"

This module is pure logic: no filesystem coupling beyond the explicit artifact
paths passed in, no LLM, no subprocesses.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import yaml

from .contract import ContractError, OperatorContract


class OracleError(ValueError):
    """Reference is missing, malformed, or its digest does not match."""


# --------------------------------------------------------------------------- #
# Digesting / freezing
# --------------------------------------------------------------------------- #

def digest_reference(
    op_id: str, contract_dict: Dict[str, Any], reference_source: str,
) -> str:
    """SHA-256 over a canonical serialization of (op_id, contract, source).

    ``contract_dict`` is digested in sorted-key JSON form so the digest is
    independent of insertion order and matches across YAML round-trips.
    """
    blob = (
        json.dumps({"op_id": op_id, "contract": contract_dict}, sort_keys=True)
        + "\n"
        + reference_source
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenOracle:
    """The frozen correctness reference for one run.

    ``origin`` records how the reference was obtained (SSOT for provenance):
      - ``"user"``                 — user supplied it this run
      - ``"library"``              — hit in the shared reference library
      - ``"generated"``            — generated + passed full-auto review, admitted
      - ``"generated_unadmitted"`` — generated, passed review, but NOT admitted
                                     (library admission failed / declined) — frozen
                                     per-run only
    """

    op_id: str
    contract: Dict[str, Any]
    reference_source: str
    origin: str
    digest: str
    created_at: float

    def __post_init__(self) -> None:
        # Validate structure / consistency at construction time.
        expect = digest_reference(self.op_id, self.contract, self.reference_source)
        if self.digest != expect:
            raise OracleError(
                f"FrozenOracle digest mismatch: {self.digest} != {expect}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Any) -> "FrozenOracle":
        if not isinstance(data, dict):
            raise OracleError("FrozenOracle must be a mapping")
        try:
            return cls(
                op_id=str(data["op_id"]),
                contract=dict(data["contract"]),
                reference_source=str(data["reference_source"]),
                origin=str(data["origin"]),
                digest=str(data["digest"]),
                created_at=float(data["created_at"]),
            )
        except KeyError as exc:
            raise OracleError(f"FrozenOracle missing field {exc}") from exc


def freeze_reference(
    op_id: str, contract: OperatorContract, reference_source: str, origin: str,
) -> FrozenOracle:
    """Build a frozen oracle from a contract + reference source."""
    return FrozenOracle(
        op_id=op_id,
        contract=contract.to_dict(),
        reference_source=reference_source,
        origin=origin,
        digest=digest_reference(op_id, contract.to_dict(), reference_source),
        created_at=time.time(),
    )


# --------------------------------------------------------------------------- #
# Per-run artifact writing / cold restart
# --------------------------------------------------------------------------- #

def write_oracle_artifacts(oracle_dir: Path, oracle: FrozenOracle) -> None:
    """Write ``contract.yaml`` + ``reference.py`` + ``oracle.json`` atomically.

    ``oracle.json`` is the **authoritative** descriptor; the other two files are
    readable copies for inspection/tooling. On cold restart only ``oracle.json``
    is read; its digest is verified against the on-disk source.
    """
    oracle_dir = Path(oracle_dir)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    from metainfer.cluster.fs_primitives import atomic_write_text

    atomic_write_text(oracle_dir / "contract.yaml", yaml.safe_dump(oracle.contract, sort_keys=False))
    atomic_write_text(oracle_dir / "reference.py", oracle.reference_source)
    atomic_write_text(oracle_dir / "oracle.json", json.dumps(oracle.to_dict(), indent=2, sort_keys=True))


def load_oracle(oracle_dir: Path, op_id: Optional[str] = None) -> Optional[FrozenOracle]:
    """Cold-restart: reload a frozen oracle from ``oracle_dir``.

    Returns ``None`` if no oracle exists there. Raises :class:`OracleError` if the
    on-disk ``oracle.json`` digest does not match the stored reference source /
    contract (indicating the dir was tampered with or partially written).
    """
    oracle_dir = Path(oracle_dir)
    path = oracle_dir / "oracle.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleError(f"unreadable oracle.json: {exc}") from exc
    oracle = FrozenOracle.from_dict(data)
    if op_id is not None and oracle.op_id != op_id:
        raise OracleError(f"oracle op_id {oracle.op_id} != expected {op_id}")
    return oracle


# --------------------------------------------------------------------------- #
# Reference execution (injectable; numpy is lazy)
# --------------------------------------------------------------------------- #

class ReferenceExecutor(Protocol):
    """Runs a reference source over concrete dims -> dict of output values."""

    def generate_inputs(self, contract: OperatorContract, dims: Dict[str, int]) -> Dict[str, Any]:
        """Deterministically seeded random inputs keyed by contract input name."""
        ...

    def run(
        self, reference_source: str, contract: OperatorContract, dims: Dict[str, int],
    ) -> Dict[str, Any]:
        """Execute ``forward`` and return outputs keyed by contract output name."""
        ...


class NumpyReferenceExecutor:
    """Production executor backed by numpy (imported lazily so this module stays
    importable where numpy is unavailable, e.g. unit tests)."""

    # contract dtype -> numpy dtype + scale used for deterministic fills
    _DTYPES = {
        "fp16": ("float16", 1.0), "bf16": ("float16", 1.0),
        "fp32": ("float32", 1.0), "fp64": ("float64", 1.0),
        "int8": ("int8", 1.0), "int4": ("int8", 1.0), "fp8": ("float16", 1.0),
    }

    def _np(self):
        try:
            import numpy as np  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover — env-specific
            raise OracleError("numpy is required to run a numpy reference") from exc
        return np

    def generate_inputs(self, contract: OperatorContract, dims: Dict[str, int]) -> Dict[str, Any]:
        np = self._np()
        rng = np.random.default_rng(abs(hash(frozenset(dims.items()))) % (2 ** 32))
        out: Dict[str, Any] = {}
        for t in contract.inputs:
            shape = t.resolved_shape(dims)
            dtype, scale = self._DTYPES.get(t.dtype, ("float32", 1.0))
            arr = (rng.standard_normal(shape) * scale).astype(dtype)
            out[t.name] = arr
        return out

    def run(
        self, reference_source: str, contract: OperatorContract, dims: Dict[str, int],
    ) -> Dict[str, Any]:
        namespace: Dict[str, Any] = {}
        exec(reference_source, namespace)  # noqa: S102 — the reference source is user/LLM authored
        if "forward" not in namespace:
            raise OracleError("reference must define forward(**tensors)")
        inputs = self.generate_inputs(contract, dims)
        outputs = namespace["forward"](**inputs)
        if not isinstance(outputs, dict):
            raise OracleError("reference forward must return a dict keyed by output name")
        return outputs


def generate_inputs_for(contract: OperatorContract, dims: Dict[str, int]) -> Dict[str, Any]:
    """Convenience wrapper for callers that just want synthetic inputs."""
    return NumpyReferenceExecutor().generate_inputs(contract, dims)


__all__ = [
    "OracleError",
    "digest_reference",
    "FrozenOracle",
    "freeze_reference",
    "write_oracle_artifacts",
    "load_oracle",
    "ReferenceExecutor",
    "NumpyReferenceExecutor",
    "generate_inputs_for",
]
