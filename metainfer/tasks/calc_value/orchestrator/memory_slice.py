"""Per-node memory slicing for calc_value prompts.

The full ``step1/memory.json`` is ~70KB (~20K tokens) and was previously
injected verbatim into every per-node validator / writer prompt — even
though each agent only reasons about ONE node. This module produces a
~2-3KB slice containing just what a per-node agent needs:

  - ``architecture_summary`` (global config: hidden_size, num_layers,
    vocab_size, ...) — every node needs this for shape resolution.
  - The single ``operator_calls`` entry whose ``node_id_hint`` matches
    the target node's id (or its section id when the node itself has
    no direct hint).
  - Any ``uncertainties`` entries that reference the matching operator.
  - ``quantization_load`` and ``tp_behavior`` (small, always relevant
    for shape reconciliation).

Result: ~95% prompt-size reduction for per-node agents. The orchestrator
still passes the full memory.json to step-level agents (graph_builder,
graph_fixer) that need the global view.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _normalize_node_id(node_id: str) -> str:
    """Lowercase + collapse underscores/dashes for fuzzy matching."""
    if not node_id:
        return ""
    s = node_id.lower()
    # Strip trailing index suffixes like "_0", "_7" so "attn_mqa_7"
    # matches the hint "attn_mqa".
    parts = s.split("_")
    while parts and parts[-1].isdigit():
        parts.pop()
    return "_".join(parts)


def _match_operator_call(
    op_calls: List[Dict[str, Any]], node_id: str, node_purpose: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Find the operator_calls entry that best matches the target node.

    Matching precedence:
      1. Exact ``node_id_hint`` equality (normalized).
      2. ``node_id_hint`` is a substring of node_id (or vice versa).
      3. ``purpose`` substring overlap (last resort — keeps the slice
         non-empty even when the graph builder minted a fresh id).
    """
    if not op_calls:
        return None
    norm_target = _normalize_node_id(node_id)
    # Pass 1: exact match on normalized id.
    for c in op_calls:
        h = _normalize_node_id(c.get("node_id_hint") or "")
        if h and h == norm_target:
            return c
    # Pass 2: substring containment either direction.
    for c in op_calls:
        h = _normalize_node_id(c.get("node_id_hint") or "")
        if not h or not norm_target:
            continue
        if h in norm_target or norm_target in h:
            return c
    # Pass 3: purpose overlap.
    tgt_p = (node_purpose or "").lower()
    if tgt_p:
        for c in op_calls:
            cp = (c.get("purpose") or "").lower()
            if cp and (cp in tgt_p or tgt_p in cp):
                return c
    return None


def slice_memory_for_node(
    memory: Dict[str, Any],
    node_id: str,
    node_purpose: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a slimmed dict containing only the memory entries relevant
    to a single node.

    Always includes ``architecture_summary`` (small, every node needs it
    for shape symbols). Includes the best-matching ``operator_calls``
    entry if one exists. Includes any ``uncertainties`` that mention
    the matched operator. Includes ``quantization_load`` and
    ``tp_behavior`` (small global context).

    The returned dict serializes to ~2-3KB JSON — vs ~70KB for the
    full memory — so per-node agent prompts drop by ~95%.
    """
    if not isinstance(memory, dict):
        return {}

    out: Dict[str, Any] = {}

    # Always include global config — every per-node agent needs it to
    # resolve shape symbols.
    arch = memory.get("architecture_summary")
    if isinstance(arch, dict):
        out["architecture_summary"] = arch

    # Match the operator_calls entry for this node.
    op_calls = memory.get("operator_calls") or []
    if not isinstance(op_calls, list):
        op_calls = []
    match = _match_operator_call(op_calls, node_id, node_purpose)
    if match is not None:
        out["operator_call"] = match
        # Surface any uncertainty involving the same operator.
        op_name = (match.get("op") or "").lower()
        node_hint = _normalize_node_id(match.get("node_id_hint") or node_id)
        related_uncert: List[Dict[str, Any]] = []
        for u in memory.get("uncertainties") or []:
            if not isinstance(u, dict):
                continue
            blob = json.dumps(u, ensure_ascii=False).lower()
            if (op_name and op_name in blob) or (
                node_hint and node_hint in blob
            ):
                related_uncert.append(u)
        if related_uncert:
            out["uncertainties"] = related_uncert

    # Small global sections, always relevant for shape reconciliation.
    ql = memory.get("quantization_load")
    if isinstance(ql, dict) and ql:
        out["quantization_load"] = ql
    tp = memory.get("tp_behavior")
    if isinstance(tp, dict) and tp:
        # Trim tp_behavior to just summary + nonsplit_weights keys —
        # split_weights can be a long per-layer listing.
        slim_tp = {
            "summary": tp.get("summary"),
            "nonsplit_weights": tp.get("nonsplit_weights"),
        }
        if any(slim_tp.values()):
            out["tp_behavior"] = slim_tp

    return out


def slice_memory_for_node_as_json(
    memory: Dict[str, Any],
    node_id: str,
    node_purpose: Optional[str] = None,
) -> str:
    """Convenience: slice + json.dumps with indent=2."""
    return json.dumps(
        slice_memory_for_node(memory, node_id, node_purpose),
        indent=2, ensure_ascii=False,
    )
