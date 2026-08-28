"""OperatorContract — typed, operator-agnostic description of one operator.

This is the single source of truth that drives the whole ``opt_operator``
task. A contract describes an operator's tensor signature via *symbolic shape
expressions*, a shape-sweep DSL that expands into a correctness + benchmark
case matrix for **any** operator (no hardcoded GEMM/attention semantics).

Shape arithmetic is parsed by a small recursive-descent evaluator
(:func:`_eval_expr`) — deliberately NOT ``eval``, which is an arbitrary-code
execution vector even when ``__builtins__`` is stripped (attribute chains like
``().__class__.__bases__`` still escape). The parser accepts only integer/float
literals, symbolic dims, ``+ - * // % **`` and parentheses.

This module is pure logic: no filesystem, no LLM, no subprocesses.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


class ContractError(ValueError):
    """Invalid OperatorContract — fail fast at task creation, not mid-run."""


LANGUAGES = ("hip", "triton")
SHAPE_MODES = ("targeted", "general")
DTYPES = ("fp16", "bf16", "fp32", "fp64", "int8", "int4", "fp8")

# bytes per element for the common dtypes (used only for buffer sizing hints /
# optional metric fallback; not authoritative — the harness may override).
_DTYPE_BYTES = {
    "fp16": 2, "bf16": 2, "fp32": 4, "fp64": 8,
    "int8": 1, "int4": 1, "fp8": 1,
}

_DIM_NAME = r"[A-Za-z_][A-Za-z0-9_]*"


# --------------------------------------------------------------------------- #
# Safe shape-expression evaluator
# --------------------------------------------------------------------------- #


class _ExprError(ValueError):
    pass


class _ExprParser:
    """Recursive-descent parser for shape/dim arithmetic.

    Grammar (all values are numbers; results may be int or float):
        expr    := term (('+' | '-') term)*
        term    := factor (('*' | '//' | '%') factor)*
        factor  := unary ('**' factor)?
        unary   := ('-' | '+') unary | primary
        primary := NUMBER | NAME | '(' expr ')'
    """

    def __init__(self, text: str, dims: Dict[str, int]) -> None:
        self.text = text
        self.dims = dims
        self.pos = 0
        self.n = len(text)

    def parse(self) -> Union[int, float]:
        value = self._expr()
        self._skip_ws()
        if self.pos != self.n:
            raise _ExprError(f"unexpected trailing '{self.text[self.pos:]}'")
        return value

    def _skip_ws(self) -> None:
        while self.pos < self.n and self.text[self.pos] in " \t":
            self.pos += 1

    def _peek(self) -> str:
        self._skip_ws()
        return self.text[self.pos] if self.pos < self.n else ""

    def _op(self, n: int = 1) -> str:
        """Next ``n`` non-whitespace chars (shorter at EOF)."""
        self._skip_ws()
        return self.text[self.pos:self.pos + n]

    def _expr(self) -> Union[int, float]:
        value = self._term()
        while True:
            op = self._op()
            if op == "+" or op == "-":
                self.pos += 1
                rhs = self._term()
                value = value + rhs if op == "+" else value - rhs
            else:
                return value

    def _term(self) -> Union[int, float]:
        value = self._factor()
        while True:
            op = self._op()
            if op == "*":
                self.pos += 1
                value = value * self._factor()
            elif op == "/" and self._op(2) == "//":
                self.pos += 2
                value = int(value) // int(self._factor())
            elif op == "%":
                self.pos += 1
                value = int(value) % int(self._factor())
            else:
                return value

    def _factor(self) -> Union[int, float]:
        value = self._unary()
        if self._op(2) == "**":
            self.pos += 2
            exp = self._factor()
            value = value ** exp
        return value

    def _unary(self) -> Union[int, float]:
        op = self._op()
        if op == "+" or op == "-":
            self.pos += 1
            v = self._unary()
            return v if op == "+" else -v
        return self._primary()

    def _primary(self) -> Union[int, float]:
        self._skip_ws()
        if self.pos >= self.n:
            raise _ExprError("unexpected end of expression")
        ch = self.text[self.pos]
        if ch.isdigit() or (ch == "." and self.pos + 1 < self.n and self.text[self.pos + 1].isdigit()):
            return self._number()
        if ch.isalpha() or ch == "_":
            return self._name()
        if ch == "(":
            self.pos += 1
            value = self._expr()
            self._skip_ws()
            if self.pos >= self.n or self.text[self.pos] != ")":
                raise _ExprError("missing ')'")
            self.pos += 1
            return value
        raise _ExprError(f"unexpected character {ch!r}")

    def _number(self) -> Union[int, float]:
        start = self.pos
        while self.pos < self.n and (self.text[self.pos].isdigit() or self.text[self.pos] == "."):
            self.pos += 1
        raw = self.text[start:self.pos]
        if raw.count(".") > 1:
            raise _ExprError(f"malformed number {raw!r}")
        return float(raw) if "." in raw else int(raw)

    def _name(self) -> int:
        start = self.pos
        while self.pos < self.n and (self.text[self.pos].isalnum() or self.text[self.pos] == "_"):
            self.pos += 1
        name = self.text[start:self.pos]
        if name not in self.dims:
            raise _ExprError(f"unknown symbolic dim {name!r}")
        return int(self.dims[name])


def eval_expr(expr: str, dims: Dict[str, int]) -> Union[int, float]:
    """Evaluate a shape/dim expression over ``dims`` (safe, no ``eval``).

    Raises :class:`ContractError` on any malformed expression — the public
    error type for this module.
    """
    try:
        return _ExprParser(expr, dims).parse()
    except _ExprError as exc:
        raise ContractError(f"bad expression {expr!r}: {exc}") from exc


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TensorSpec:
    name: str
    dtype: str
    shape: List[str]          # symbolic shape expressions, e.g. ["B", "S", "H"]
    role: str = "input"       # "input" | "output"

    @classmethod
    def from_dict(cls, data: Any, *, role: str) -> "TensorSpec":
        if not isinstance(data, dict):
            raise ContractError("tensor spec must be a mapping")
        name = str(data.get("name") or "").strip()
        dtype = str(data.get("dtype") or "").strip().lower()
        shape = data.get("shape")
        if not name:
            raise ContractError("tensor spec requires a non-empty name")
        if dtype not in DTYPES:
            raise ContractError(f"tensor {name!r}: unsupported dtype {dtype!r}")
        if not isinstance(shape, list) or not shape:
            raise ContractError(f"tensor {name!r}: shape must be a non-empty expression list")
        if not all(isinstance(s, str) and s.strip() for s in shape):
            raise ContractError(f"tensor {name!r}: shape entries must be non-empty strings")
        return cls(name=name, dtype=dtype, shape=list(shape), role=role)

    def resolved_shape(self, dims: Dict[str, int]) -> List[int]:
        out: List[int] = []
        for s in self.shape:
            try:
                v = eval_expr(s, dims)
            except _ExprError as exc:
                raise ContractError(
                    f"tensor {self.name!r}: bad shape expr {s!r}: {exc}"
                ) from exc
            if isinstance(v, float) or int(v) != v or int(v) <= 0:
                raise ContractError(
                    f"tensor {self.name!r}: shape expr {s!r} resolved to non-positive "
                    f"integer {v}"
                )
            out.append(int(v))
        return out

    def numel(self, dims: Dict[str, int]) -> int:
        n = 1
        for v in self.resolved_shape(dims):
            n *= v
        return n

    def bytes_per_elem(self) -> int:
        return _DTYPE_BYTES.get(self.dtype, 4)


@dataclass(frozen=True)
class CaseSpec:
    """One concrete correctness/benchmark case (fully-resolved shapes)."""
    id: str
    dims: Dict[str, int]                      # symbolic dim -> value
    shapes: Dict[str, List[int]]              # tensor name -> concrete shape
    flops: Optional[float] = None
    bytes: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OperatorContract:
    name: str
    entrypoint: str
    language: str                             # hip | triton
    shape_mode: str                           # targeted | general
    inputs: List[TensorSpec]
    outputs: List[TensorSpec]
    shapes: Dict[str, Any]                    # sym dim -> int | list[int] (sweep)
    numerics: Dict[str, float]                # abs_tol, rel_tol
    constraints: str = ""
    target_shapes: List[Dict[str, int]] = field(default_factory=list)
    metric: Optional[Dict[str, str]] = None   # optional flops/bytes exprs

    # -- construction / validation ---------------------------------------- #

    @classmethod
    def load(cls, data: Any) -> "OperatorContract":
        if isinstance(data, str):
            try:
                data = yaml.safe_load(data)
            except yaml.YAMLError as exc:
                raise ContractError(f"contract is not valid YAML: {exc}") from exc
        if not isinstance(data, dict):
            raise ContractError("OperatorContract must be a mapping")

        name = str(data.get("name") or "").strip()
        if not name:
            raise ContractError("contract requires a non-empty name")
        entrypoint = str(data.get("entrypoint") or "").strip()
        if not entrypoint:
            raise ContractError("contract requires a non-empty entrypoint")
        language = str(data.get("language") or "").strip().lower()
        if language not in LANGUAGES:
            raise ContractError(f"language must be one of {LANGUAGES}")
        shape_mode = str(data.get("shape_mode") or "general").strip().lower()
        if shape_mode not in SHAPE_MODES:
            raise ContractError(f"shape_mode must be one of {SHAPE_MODES}")

        inputs_raw = data.get("inputs")
        outputs_raw = data.get("outputs")
        if not isinstance(inputs_raw, list) or not inputs_raw:
            raise ContractError("inputs must be a non-empty list")
        if not isinstance(outputs_raw, list) or not outputs_raw:
            raise ContractError("outputs must be a non-empty list")
        inputs = [TensorSpec.from_dict(t, role="input") for t in inputs_raw]
        outputs = [TensorSpec.from_dict(t, role="output") for t in outputs_raw]

        # unique tensor names across inputs+outputs
        names = [t.name for t in inputs] + [t.name for t in outputs]
        if len(set(names)) != len(names):
            raise ContractError("tensor names must be unique across inputs/outputs")

        shapes = data.get("shapes")
        if not isinstance(shapes, dict) or not shapes:
            raise ContractError("shapes must be a non-empty mapping of symbolic dim -> value(s)")
        shapes = _validate_shapes(shapes)

        numerics = _validate_numerics(data.get("numerics"))

        constraints = str(data.get("constraints") or "").strip()

        target_shapes = _validate_target_shapes(data.get("target_shapes"), shape_mode)

        metric = _validate_metric(data.get("metric"))

        return cls(
            name=name,
            entrypoint=entrypoint,
            language=language,
            shape_mode=shape_mode,
            inputs=inputs,
            outputs=outputs,
            shapes=shapes,
            numerics=numerics,
            constraints=constraints,
            target_shapes=target_shapes,
            metric=metric,
        )

    # -- serialization ----------------------------------------------------- #

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "entrypoint": self.entrypoint,
            "language": self.language,
            "shape_mode": self.shape_mode,
            "inputs": [asdict(t) for t in self.inputs],
            "outputs": [asdict(t) for t in self.outputs],
            "shapes": self.shapes,
            "numerics": self.numerics,
            "constraints": self.constraints,
            "target_shapes": self.target_shapes,
            "metric": self.metric,
        }

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    def from_dict_like(self, data: Dict[str, Any]) -> "OperatorContract":
        return OperatorContract(**data)

    # -- derived ----------------------------------------------------------- #

    @property
    def all_tensors(self) -> List[TensorSpec]:
        return list(self.inputs) + list(self.outputs)

    @property
    def io_names(self) -> List[str]:
        return [t.name for t in self.all_tensors]

    def tensor_by_name(self, name: str) -> Optional[TensorSpec]:
        for t in self.all_tensors:
            if t.name == name:
                return t
        return None

    # -- case matrix ------------------------------------------------------- #

    def generate_cases(self) -> List[CaseSpec]:
        """Expand the shape-sweep DSL into concrete correctness/benchmark cases.

        - ``general``: cartesian product over sweep dims (dims whose value is a
          list); scalar dims are fixed.
        - ``targeted``: one case per entry in ``target_shapes``; if empty,
          fall back to a single point derived from ``shapes`` (first value per
          dim).
        """
        if self.shape_mode == "targeted":
            dim_maps = self._target_dim_maps()
        else:
            dim_maps = self._general_dim_maps()
        cases: List[CaseSpec] = []
        for dims in dim_maps:
            shapes = {t.name: t.resolved_shape(dims) for t in self.all_tensors}
            flops, transferred = self._metrics(dims, shapes)
            cases.append(CaseSpec(
                id=_case_id(dims),
                dims=dict(dims),
                shapes=shapes,
                flops=flops,
                bytes=transferred,
            ))
        return cases

    def _general_dim_maps(self) -> List[Dict[str, int]]:
        sweep: Dict[str, List[int]] = {}
        fixed: Dict[str, int] = {}
        for dim, value in self.shapes.items():
            if isinstance(value, list):
                sweep[dim] = value
            else:
                fixed[dim] = value
        if not sweep:
            return [dict(fixed)]
        keys = list(sweep.keys())
        out: List[Dict[str, int]] = []
        for combo in itertools.product(*(sweep[k] for k in keys)):
            dm = dict(fixed)
            for k, v in zip(keys, combo):
                dm[k] = v
            out.append(dm)
        return out

    def _target_dim_maps(self) -> List[Dict[str, int]]:
        if self.target_shapes:
            return [dict(t) for t in self.target_shapes]
        # Fall back to a single point: first scalar or first of each list.
        single: Dict[str, int] = {}
        for dim, value in self.shapes.items():
            single[dim] = value[0] if isinstance(value, list) else value
        return [single]

    def _metrics(
        self, dims: Dict[str, int], shapes: Dict[str, List[int]]
    ) -> Tuple[Optional[float], Optional[float]]:
        if not self.metric:
            return None, None
        flops: Optional[float] = None
        transferred: Optional[float] = None
        if "flops" in self.metric:
            flops = float(_eval_metric(self.metric["flops"], dims))
        if "bytes" in self.metric:
            transferred = float(_eval_metric(self.metric["bytes"], dims))
        return flops, transferred


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #


def _validate_shapes(shapes: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dim, value in shapes.items():
        dim = str(dim).strip()
        if not dim:
            raise ContractError("symbolic dim name must be non-empty")
        values: List[int] = []
        if isinstance(value, list):
            if not value:
                raise ContractError(f"dim {dim!r} sweep must be a non-empty list")
            for v in value:
                values.append(_positive_int(v, f"dim {dim!r}"))
        else:
            values.append(_positive_int(value, f"dim {dim!r}"))
        if len(values) == 1:
            out[dim] = values[0]
        else:
            if len(set(values)) != len(values):
                raise ContractError(f"dim {dim!r} sweep contains duplicate values")
            out[dim] = values
    return out


def _validate_numerics(raw: Any) -> Dict[str, float]:
    if raw is None:
        return {"abs_tol": 1e-3, "rel_tol": 1e-2}
    if not isinstance(raw, dict):
        raise ContractError("numerics must be a mapping with abs_tol / rel_tol")
    out: Dict[str, float] = {}
    for key in ("abs_tol", "rel_tol"):
        default = 1e-3 if key == "abs_tol" else 1e-2
        val = raw.get(key, default)
        try:
            val = float(val)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"numerics.{key} must be a finite number") from exc
        if val < 0:
            raise ContractError(f"numerics.{key} must be >= 0")
        out[key] = val
    return out


def _validate_target_shapes(raw: Any, shape_mode: str) -> List[Dict[str, int]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ContractError("target_shapes must be a list of dim-value mappings")
    out: List[Dict[str, int]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise ContractError("each target_shape must be a dim-value mapping")
        dm: Dict[str, int] = {}
        for dim, value in item.items():
            dm[str(dim)] = _positive_int(value, f"target_shape dim {dim!r}")
        out.append(dm)
    if out and shape_mode != "targeted":
        raise ContractError("target_shapes is only valid when shape_mode=targeted")
    return out


def _validate_metric(raw: Any) -> Optional[Dict[str, str]]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ContractError("metric must be a mapping")
    out: Dict[str, str] = {}
    for key in ("flops", "bytes"):
        if key in raw:
            val = str(raw[key]).strip()
            if not val:
                raise ContractError(f"metric.{key} must be a non-empty expression")
            out[key] = val
    return out or None


def _positive_int(value: Any, label: str) -> int:
    try:
        v = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} must be a positive integer") from exc
    if v <= 0:
        raise ContractError(f"{label} must be a positive integer")
    return v


def _eval_metric(expr: str, dims: Dict[str, int]) -> Union[int, float]:
    try:
        return _ExprParser(expr, dims).parse()
    except _ExprError as exc:
        raise ContractError(f"bad metric expression {expr!r}: {exc}") from exc


def _case_id(dims: Dict[str, int]) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(dims.items()))


__all__ = [
    "ContractError",
    "LANGUAGES",
    "SHAPE_MODES",
    "DTYPES",
    "TensorSpec",
    "CaseSpec",
    "OperatorContract",
    "eval_expr",
]
