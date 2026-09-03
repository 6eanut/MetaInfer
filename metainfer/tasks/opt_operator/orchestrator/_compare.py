"""Shared generic numerical comparison helpers.

Work on numpy arrays **or** nested python sequences (tests use the latter so the
numerics gates are unit-testable without numpy). Used by both the reference
review gate (:mod:`reference_lib`) and the conformance gate (:mod:`conformance`)
so there is a single definition of "close enough".
"""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple


def to_flat_scalars(value: Any) -> List[float]:
    """Recursively flatten a numpy array / nested sequence / scalar into floats."""
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        value = tolist()
    if isinstance(value, (list, tuple)):
        out: List[float] = []
        for item in value:
            out.extend(to_flat_scalars(item))
        return out
    if isinstance(value, (int, float)):
        return [float(value)]
    return [0.0]


def shape_of(value: Any) -> Optional[List[int]]:
    """Structural shape of a numpy array OR a nested python sequence."""
    if value is None:
        return None
    arr_shape = getattr(value, "shape", None)
    if arr_shape is not None:
        return [int(x) for x in arr_shape]
    shape: List[int] = []
    cur = value
    while isinstance(cur, (list, tuple)):
        shape.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return shape or None


def is_finite(value: Any) -> bool:
    for x in to_flat_scalars(value):
        if not math.isfinite(x):
            return False
    return True


def allclose(a: Any, b: Any, rtol: float = 1e-3, atol: float = 1e-5) -> bool:
    fa, fb = to_flat_scalars(a), to_flat_scalars(b)
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb):
        if math.isnan(x) or math.isnan(y):
            return False
        if not math.isclose(x, y, rel_tol=rtol, abs_tol=atol):
            return False
    return True


def within_tol(a: Any, b: Any, abs_tol: float, rel_tol: float) -> bool:
    """True iff *every* element of ``a`` is within tolerance of ``b``.

    Per-element numpy-style semantics: an element passes unless its absolute
    error exceeds ``abs_tol`` **and** its relative error exceeds ``rel_tol``
    simultaneously. The case passes only if no element violates both.

    This is deliberately *not* "global max-abs and global max-rel both exceed":
    those two maxima can land on different elements (a large-magnitude element
    with a big absolute error plus a near-zero element with a big *relative*
    error), which would wrongly reject a correct candidate.
    """
    fa, fb = to_flat_scalars(a), to_flat_scalars(b)
    if len(fa) != len(fb):
        return False
    for x, y in zip(fa, fb):
        if math.isnan(x) or math.isnan(y):
            return False
        abs_err = abs(x - y)
        if abs_err > abs_tol and abs_err > rel_tol * abs(y):
            return False
    return True


def max_abs_rel_error(a: Any, b: Any) -> Tuple[float, float]:
    """Return (max_abs, max_rel) elementwise errors between ``a`` and ``b``.

    Shape mismatch yields ``(inf, inf)``. NaNs count as infinite error.
    ``rel`` is relative to ``b`` (the reference), guarding against div-by-zero.
    """
    fa, fb = to_flat_scalars(a), to_flat_scalars(b)
    if len(fa) != len(fb):
        return math.inf, math.inf
    max_abs = 0.0
    max_rel = 0.0
    for x, y in zip(fa, fb):
        if math.isnan(x) or math.isnan(y):
            return math.inf, math.inf
        abs_err = abs(x - y)
        max_abs = max(max_abs, abs_err)
        denom = abs(y)
        if denom == 0.0:
            rel = 0.0 if abs_err == 0.0 else math.inf
        else:
            rel = abs_err / denom
        max_rel = max(max_rel, rel)
    return max_abs, max_rel
