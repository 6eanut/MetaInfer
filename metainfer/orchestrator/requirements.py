"""Helpers for reading ``requirements.json`` consistently across task
orchestrators.

Single source of truth for the requirements.json schema
-------------------------------------------------------

The WebUI's ``create_task`` endpoint writes a **flat** JSON object — the
form answers are spread to the top level alongside ``task_id`` /
``task_type`` / ``label`` / ``raw_request``:

    {
      "task_id": "...",
      "task_type": "...",
      "label": "...",
      "raw_request": "...",
      "target_model": "...",       ← was in form "answers"
      "max_iterations": "50",      ← was in form "answers"
      ...
    }

There is NO ``answers`` or ``form`` sub-key. Some early code and test
fixtures wrote a nested ``{"form": {...}}`` or ``{"answers": {...}}``
shape; that is legacy and not produced by the WebUI anymore.

To avoid every orchestrator re-implementing the same "flat key first,
fall back to nested for legacy" logic (which drifts — see the
``token_budget`` cascade bug for how that pattern burns), all readers
should go through :func:`req_field`.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


# Keys that the orchestrator framework itself writes — task code should
# treat these as opaque identity/metadata, not as part of the form.
RESERVED_KEYS = frozenset({"task_id", "task_type", "label", "raw_request"})


def req_field(
    req: Dict[str, Any], key: str, default: Any = None,
) -> Any:
    """Read one field from ``requirements.json`` — single source of truth.

    Resolution order:
      1. Top-level flat key (canonical — what the WebUI writes).
      2. ``req["form"][key]`` (legacy nested form).
      3. ``req["answers"][key]`` (legacy nested answers).
      4. ``default``.

    The legacy nested fallbacks exist ONLY for backward compatibility
    with old task files on disk / hand-written test fixtures. Production
    WebUI output is flat, so resolution stops at step 1 in practice.

    Always use this helper instead of hand-writing
    ``req.get("x") or (req.get("answers") or {}).get("x")`` — that
    pattern got duplicated across 12+ call sites, each with subtly
    different null-handling, and was the root cause of at least one
    "limit silently lost" bug. Centralizing the read keeps the schema
    honest.
    """
    if not isinstance(req, dict):
        return default
    if key in req:
        return req[key]
    for ns in ("form", "answers"):
        bucket = req.get(ns)
        if isinstance(bucket, dict) and key in bucket:
            return bucket[key]
    return default


def req_field_int(
    req: Dict[str, Any], key: str, default: Optional[int] = None,
) -> Optional[int]:
    """Like :func:`req_field` but coerces to int. Returns ``default``
    on missing/unparseable values. Useful for fields like
    ``max_iterations`` that the form may emit as a string."""
    v = req_field(req, key, None)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def req_field_float(
    req: Dict[str, Any], key: str, default: Optional[float] = None,
) -> Optional[float]:
    """Like :func:`req_field` but coerces to float."""
    v = req_field(req, key, None)
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default
