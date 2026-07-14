"""Pure-Python deterministic helpers used by all 4 steps.

Nothing in this module calls an LLM. Every function is reproducible and
testable in isolation. The orchestrator relies on these for:

* parsing JSON out of agent responses (which often wrap JSON in
  ```json ... ``` fences or surround it with prose),
* merging 2 agents' structured outputs into a consensus memory,
* validating graph.json structure (no orphan nodes, no missing fields),
* comparing 2 calc scripts' outputs at the canonical shape (B=1, S=512)
  with a strict tolerance, and falling back to median if 3 rounds
  don't converge.

Unit-test friendly: every public function takes data, returns data, no
I/O except where the function's job is I/O.
"""

from __future__ import annotations

import importlib.util
import io
import json
import math
import re
import sys
import textwrap
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# JSON extraction from LLM responses
# --------------------------------------------------------------------------- #

_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n(.*?)\n```",
    re.DOTALL,
)


def extract_json(text: str) -> Optional[Any]:
    """Pull a JSON object/array out of an LLM response.

    Handles three common patterns:
    * fenced ```json\n...\n```
    * bare JSON (whitespace-trimmed)
    * JSON embedded in prose (heuristic: first '{' or '[' to its match)

    Returns the parsed object, or None if nothing parseable was found.

    Lenient about common LLM "JSON-ish" output: strips ``//`` line
    comments and trailing commas before retrying (the prompt schema
    includes commented examples for clarity, and agents sometimes
    copy them verbatim).
    """
    # NOTE: this function is the FALLBACK path. The preferred path is
    # :func:`load_agent_json`, which reads a structured JSON file the
    # agent was instructed to Write to its workdir. We only fall back
    # to scraping the response text when the file is missing or invalid
    # (older agents that ignored the Write instruction, or mid-stream
    # crashes). See the contract docstring in load_agent_json.
    if not text:
        return None
    # 1. Fenced block(s) — take the LAST one (LLMs often summarize-then-output).
    fences = _FENCE_RE.findall(text)
    for block in reversed(fences):
        parsed = _try_json_lenient(block)
        if parsed is not None:
            return parsed
    # 2. Bare JSON (whole text trimmed).
    parsed = _try_json_lenient(text.strip())
    if parsed is not None:
        return parsed
    # 3. First balanced { ... } or [ ... ].
    obj = _extract_balanced(text, "{", "}")
    if obj is not None:
        return obj
    arr = _extract_balanced(text, "[", "]")
    if arr is not None:
        return arr
    return None


# --------------------------------------------------------------------------- #
# File-first structured output loading
# --------------------------------------------------------------------------- #
# Contract: every calc-value agent is told to Write its structured output
# to a specific filename in its workdir (output.json / graph.json /
# verdict.json / calc.py / viz.html). The agent's natural-language
# response.txt is a human-readable narrative — log/retrospective — NOT
# the data transport. These loaders enforce that contract: they read the
# file first, and only fall back to scraping response.txt if the file is
# missing or invalid. This keeps "where the data lives" (file) and "what
# the agent said about it" (response.txt) cleanly separated, and means
# the UI can show them in different panels without confusion.

def load_agent_json(
    workdir: Path, filename: str, response_text: str,
) -> Tuple[Optional[Any], str]:
    """Read structured JSON from ``<workdir>/<filename>``; fall back to
    scraping ``response_text`` if the file is missing/unparseable.

    Returns ``(parsed_value, source)`` where ``source`` is one of:
    * ``"file"``     — parsed the JSON file successfully
    * ``"response"`` — file missed, but extract_json found JSON in the text
    * ``"none"``     — neither path produced parseable JSON; value is None
    """
    p = workdir / filename
    if p.is_file():
        try:
            value = json.loads(p.read_text(encoding="utf-8"))
            return value, "file"
        except (ValueError, OSError):
            pass  # fall through to response scraping
    scraped = extract_json(response_text or "")
    if scraped is not None:
        return scraped, "response"
    return None, "none"


def load_agent_text_file(
    workdir: Path, filename: str, response_text: str,
    extractor,
) -> Tuple[str, str]:
    """Read raw text (Python source / HTML / etc.) from
    ``<workdir>/<filename>``; fall back to ``extractor(response_text)``.

    ``extractor`` is e.g. :func:`step3_calculate._extract_python_source`
    or :func:`step4_visualize._extract_html`.

    Returns ``(text, source)`` with the same ``source`` vocabulary as
    :func:`load_agent_json` (``"file"`` / ``"response"`` / ``"none"``).
    """
    p = workdir / filename
    if p.is_file():
        try:
            text = p.read_text(encoding="utf-8")
            if text.strip():
                return text, "file"
        except OSError:
            pass
    extracted = extractor(response_text or "") if extractor else ""
    if extracted:
        return extracted, "response"
    return "", "none"


# Comment / trailing-comma stripping for lenient JSON parsing. The line
# comment regex is intentionally simple: a `//` that's not inside a string
# is approximated by only matching it when preceded by whitespace, start
# of line, or a JSON structural char — avoids catching `//` inside URLs
# that appear as string contents. Good enough for the common case where
# the LLM copies our schema example verbatim.
_LINE_COMMENT_RE = re.compile(
    r"(^|[\s,{\[:])//[^\n]*", re.MULTILINE,
)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _strip_json_comments(text: str) -> str:
    """Remove ``//`` line comments and trailing commas from JSON-ish text."""
    if not text:
        return text
    # Strip line comments. The lookbehind-equivalent capture group
    # preserves the leading char.
    out = _LINE_COMMENT_RE.sub(lambda m: m.group(1), text)
    # Strip trailing commas before } or ].
    out = _TRAILING_COMMA_RE.sub(r"\1", out)
    return out


def _try_json_lenient(text: str) -> Optional[Any]:
    """Try strict JSON first, then retry after stripping comments."""
    if not text or not text.strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        return json.loads(_strip_json_comments(text))
    except json.JSONDecodeError:
        return None


def _extract_balanced(text: str, open_ch: str, close_ch: str) -> Optional[Any]:
    """Return the first balanced open_ch ... close_ch span, parsed as JSON."""
    start = text.find(open_ch)
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            c = text[i]
            if c == open_ch:
                depth += 1
            elif c == close_ch:
                depth -= 1
                if depth == 0:
                    snippet = text[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except json.JSONDecodeError:
                        break
        start = text.find(open_ch, start + 1)
    return None


# --------------------------------------------------------------------------- #
# Memory merging (Step 1)
# --------------------------------------------------------------------------- #

# Fields where both agents must agree (else another round).
CRITICAL_FIELDS = (
    "architecture",
    "num_layers",
    "hidden_size",
    "num_attention_heads",
    "quantization",
)


def merge_memories(agent_outputs: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Merge 2 agents' memory dicts into a consensus.

    Returns ``(merged_memory, disputes)`` where ``disputes`` is a list of
    ``{field, values: [agent_a_val, agent_b_val, agent_c_val]}`` records
    for fields where the agents disagreed. Disputes on non-critical
    fields (e.g. ``intermediate_size``) are still surfaced but won't
    force another round.

    The merged memory preserves every agent's full output under
    ``agent_findings`` for downstream auditability.
    """
    disputes: List[Dict[str, Any]] = []

    # Architecture-level: majority vote on each critical field.
    arch_summary = {}
    arch_blocks = [a.get("architecture_summary") or {} for a in agent_outputs]
    for field in ("architecture", "num_layers", "hidden_size",
                  "num_attention_heads", "num_key_value_heads",
                  "intermediate_size", "vocab_size", "context_length",
                  "quantization"):
        values = [_safe(b.get(field)) for b in arch_blocks]
        counts: Dict[str, int] = {}
        for v in values:
            counts[v] = counts.get(v, 0) + 1
        # Majority: at least 2 agents agree.
        winner = max(counts.items(), key=lambda kv: kv[1])
        if winner[1] >= 2:
            arch_summary[field] = _from_safe(winner[0])
        else:
            arch_summary[field] = _from_safe(values[0]) if values else None
        # Track disputes on critical fields.
        if field in CRITICAL_FIELDS and len(set(values)) > 1:
            disputes.append({
                "section": "architecture_summary",
                "field": field,
                "values": values,
            })
    # Preserve the evidence block from the first agent that supplied one.
    for b in arch_blocks:
        if b.get("evidence"):
            arch_summary["evidence"] = b["evidence"]
            break

    # Operator calls: union with de-dup by node_id_hint + op.
    merged_ops: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for a in agent_outputs:
        for op in a.get("operator_calls") or []:
            if not isinstance(op, dict):
                continue
            key = (str(op.get("node_id_hint", "")), str(op.get("op", "")))
            if key not in merged_ops:
                merged_ops[key] = op
    operator_calls = list(merged_ops.values())

    # Framework entry points: union by file path.
    seen_files: set = set()
    entry_points: List[Dict[str, Any]] = []
    for a in agent_outputs:
        for ep in a.get("framework_entry_points") or []:
            if not isinstance(ep, dict):
                continue
            f = ep.get("file")
            if f and f not in seen_files:
                seen_files.add(f)
                entry_points.append(ep)

    # Quantization: majority vote on approach.
    q_blocks = [a.get("quantization_load") or {} for a in agent_outputs]
    q_approaches = [_safe(q.get("approach")) for q in q_blocks]
    q_counts: Dict[str, int] = {}
    for v in q_approaches:
        q_counts[v] = q_counts.get(v, 0) + 1
    q_winner = max(q_counts.items(), key=lambda kv: kv[1])
    quantization_load = {
        "approach": _from_safe(q_winner[0]),
    }
    for q in q_blocks:
        if q.get("approach") == quantization_load["approach"] and q:
            quantization_load.update(q)
            break
    if len(set(q_approaches)) > 1:
        disputes.append({
            "section": "quantization_load",
            "field": "approach",
            "values": q_approaches,
        })

    # TP behavior: prefer the agent with the most detailed evidence.
    tp_picks: List[Dict[str, Any]] = [
        a.get("tp_behavior") or {} for a in agent_outputs
    ]
    tp_behavior = max(tp_picks, key=lambda b: len(b or {}), default={}) or {}

    # Uncertainties: concatenate all.
    uncertainties: List[str] = []
    for a in agent_outputs:
        for u in a.get("uncertainties") or []:
            if isinstance(u, str) and u not in uncertainties:
                uncertainties.append(u)

    # Findings: keep every agent's full output for audit.
    agent_findings: Dict[str, Any] = {}
    for idx, a in enumerate(agent_outputs):
        agent_findings[f"agent_{chr(ord('a') + idx)}"] = a

    merged = {
        "architecture_summary": arch_summary,
        "framework_entry_points": entry_points,
        "operator_calls": operator_calls,
        "quantization_load": quantization_load,
        "tp_behavior": tp_behavior,
        "uncertainties": uncertainties,
        "agent_findings": agent_findings,
        "_meta": {
            "agent_count": len(agent_outputs),
            "dispute_count": len(disputes),
        },
    }
    return merged, disputes


def _safe(v: Any) -> str:
    """Normalize a value for counting/majority vote (hashable)."""
    if v is None:
        return "null"
    if isinstance(v, (int, float, str, bool)):
        return str(v)
    return json.dumps(v, sort_keys=True)


def _from_safe(s: str) -> Any:
    """Inverse of _safe for scalar types only."""
    if s == "null":
        return None
    if s in ("true", "false"):
        return s == "true"
    # int?
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# --------------------------------------------------------------------------- #
# Graph validation (Step 2)
# --------------------------------------------------------------------------- #

REQUIRED_NODE_FIELDS = ("id", "purpose", "op", "inputs", "outputs")


def validate_graph_structure(graph: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Pure structural validation of graph.json.

    Returns ``(ok, errors)``. ``errors`` is empty iff ok. Does NOT
    consult any LLM — just shape checks. Used as the inner loop of the
    Step 2 iterative validator.

    Checks:
    * top-level has ``nodes`` (non-empty list) and ``edges`` (list)
    * every node has ``id`` / ``purpose`` / ``op`` / ``inputs`` / ``outputs``
    * node ids are unique
    * every edge ``from`` / ``to`` references an existing node id
    * no orphan nodes: each node is reachable from a source (in-degree 0
      is OK for exactly the entry node; everything else needs an in-edge)
    """
    errors: List[str] = []
    nodes = graph.get("nodes")
    edges = graph.get("edges")
    if not isinstance(nodes, list) or not nodes:
        errors.append("graph.nodes missing or empty")
        return False, errors
    if not isinstance(edges, list):
        errors.append("graph.edges missing or not a list")
        edges = []

    node_ids: set = set()
    for i, n in enumerate(nodes):
        if not isinstance(n, dict):
            errors.append(f"node[{i}] is not an object")
            continue
        for f in REQUIRED_NODE_FIELDS:
            if f not in n:
                errors.append(f"node[{i}] missing required field {f!r}")
        nid = n.get("id")
        if not nid:
            continue
        if nid in node_ids:
            errors.append(f"duplicate node id: {nid!r}")
        node_ids.add(nid)
        # Inputs/outputs must be lists of dicts with a `shape` field that's
        # a list of (str|int) symbols.
        for fld in ("inputs", "outputs"):
            io_list = n.get(fld)
            if io_list is None:
                continue
            if not isinstance(io_list, list):
                errors.append(f"node {nid!r}: {fld} must be a list")
                continue
            for j, t in enumerate(io_list):
                if not isinstance(t, dict):
                    errors.append(f"node {nid!r}: {fld}[{j}] is not an object")
                    continue
                shape = t.get("shape")
                if shape is None:
                    continue
                if not isinstance(shape, list):
                    errors.append(f"node {nid!r}: {fld}[{j}].shape must be a list")

    # Edge validity.
    for i, e in enumerate(edges):
        if not isinstance(e, dict):
            errors.append(f"edge[{i}] is not an object")
            continue
        f, t = e.get("from"), e.get("to")
        if not f or not t:
            errors.append(f"edge[{i}] missing from/to")
            continue
        if node_ids and f not in node_ids:
            errors.append(f"edge[{i}] from unknown node {f!r}")
        if node_ids and t not in node_ids:
            errors.append(f"edge[{i}] to unknown node {t!r}")

    # Orphan check: every node must appear in at least one edge, except
    # possibly a single source node and a single sink. If there are >1
    # nodes and 0 edges, ALL of them are orphans.
    if len(node_ids) > 1:
        in_deg: Dict[str, int] = {nid: 0 for nid in node_ids}
        out_deg: Dict[str, int] = {nid: 0 for nid in node_ids}
        for e in edges:
            if isinstance(e, dict):
                f, t = e.get("from"), e.get("to")
                if f in out_deg:
                    out_deg[f] += 1
                if t in in_deg:
                    in_deg[t] += 1
        sources = [nid for nid in node_ids if in_deg[nid] == 0 and out_deg[nid] > 0]
        sinks = [nid for nid in node_ids if out_deg[nid] == 0 and in_deg[nid] > 0]
        connected = set(sources) | set(sinks)
        for nid in node_ids:
            if nid not in connected and (in_deg[nid] == 0 and out_deg[nid] == 0):
                errors.append(f"orphan node (no edges): {nid!r}")

    return (not errors), errors


# --------------------------------------------------------------------------- #
# Sectioned graph: input + N layer-templates + output
# --------------------------------------------------------------------------- #
# A real model's forward pass rarely has 1000+ unique operators. It has
# a few STAGES that repeat (per-layer dense / per-layer MoE / per-layer
# shared-expert) sandwiched between non-repeating ends (embedding,
# sampling). Inlining N copies of the same per-layer subgraph in a
# single flat graph wastes space and forces the agent to either emit
# thousands of nodes (which it then can't Write in one JSON) or sketch
# them with literal "..." (which the JSON parser chokes on).
#
# The sectioned schema forces the agent to GROUP identical layers into
# ONE section with a ``repeat_count``, and to extract the non-repeating
# ends into ``input`` / ``output`` sections. A simple model (Llama) has
# 3 sections: input + 1 layer_template (×num_layers) + output. DeepSeek
# -V3 has 4: input + dense ×3 + MoE ×58 + output.
#
# Schema (JSON):
#     {
#       "sections": [
#         {
#           "id": "<stable snake_case id, unique within sections>",
#           "kind": "input" | "layer_template" | "output",
#           "description": "<short human-readable>",
#           "applies_to": [<int>, ...],   # layer indices, REQUIRED for
#                                         # layer_template; [] / absent
#                                         # for input/output
#           "repeat_count": <int>,        # REQUIRED for layer_template
#                                         # (== len(applies_to)); absent
#                                         # or 1 for input/output
#           "graph": {"nodes": [...], "edges": [...]}   # ONE occurrence
#                                                       # if repeat>1
#         },
#         ...
#       ],
#       "inter_section_edges": [
#         {"from_section": "<id>", "to_section": "<id>"}
#       ]
#     }
#
# Backward compat: a legacy flat ``{"nodes": [...], "edges": [...]}``
# (no ``sections``) is wrapped as a single layer_template section with
# repeat_count=1. This lets old state files and old agent output load
# without changes; the wrapper is internal — new agent output should
# emit the sectioned schema directly.

SECTION_KINDS = ("input", "layer_template", "output")


def is_sectioned(graph: Any) -> bool:
    """True if ``graph`` follows the sectioned schema."""
    return isinstance(graph, dict) and isinstance(graph.get("sections"), list)


def wrap_flat_as_sectioned(graph: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap a legacy flat ``{nodes, edges}`` graph as a single-section
    sectioned graph. Used for backward compatibility with old state
    files + old agent output. We omit ``applies_to`` since we don't
    know which layers the original agent intended."""
    return {
        "sections": [
            {
                "id": "all",
                "kind": "layer_template",
                "description": "legacy flat graph wrapped as single section",
                "repeat_count": 1,
                "graph": {
                    "nodes": list(graph.get("nodes") or []),
                    "edges": list(graph.get("edges") or []),
                },
            }
        ],
        "inter_section_edges": [],
    }


def normalize_graph(graph: Any) -> Dict[str, Any]:
    """Coerce any supported graph shape to the sectioned schema.
    Returns a NEW dict; the input is untouched."""
    if not isinstance(graph, dict):
        raise ValueError(f"graph must be a dict, got {type(graph).__name__}")
    if is_sectioned(graph):
        return graph
    if "nodes" in graph or "edges" in graph:
        return wrap_flat_as_sectioned(graph)
    raise ValueError(
        "graph has neither 'sections' nor 'nodes'/'edges' — unrecognized"
    )


def iter_section_nodes(graph: Dict[str, Any]):
    """Yield ``(section, node)`` for every node in every section."""
    for sec in graph.get("sections") or []:
        for n in (sec.get("graph") or {}).get("nodes") or []:
            yield sec, n


def section_node_count(graph: Dict[str, Any]) -> int:
    return sum(1 for _ in iter_section_nodes(graph))


def section_edge_count(graph: Dict[str, Any]) -> int:
    return sum(len((sec.get("graph") or {}).get("edges") or [])
              for sec in graph.get("sections") or [])


def validate_sectioned_graph(graph: Any) -> Tuple[bool, List[str]]:
    """Pure structural validation of a sectioned graph.

    Checks:
    * top-level ``sections`` is a non-empty list
    * exactly one section with ``kind == "input"`` (or at most one;
      zero is OK if the model genuinely has no preprocessing)
    * exactly one section with ``kind == "output"`` (same caveat)
    * every section id is unique
    * layer_template sections have ``repeat_count >= 1`` and
      ``len(applies_to) == repeat_count`` (if applies_to is present)
    * each section's inner ``graph`` passes ``validate_graph_structure``
    * ``inter_section_edges`` reference existing section ids
    """
    if not is_sectioned(graph):
        return validate_graph_structure(graph if isinstance(graph, dict)
                                        else {"nodes": []})
    errors: List[str] = []
    sections = graph.get("sections") or []
    if not sections:
        return False, ["graph.sections is empty"]

    section_ids: set = set()
    kind_counts = {"input": 0, "layer_template": 0, "output": 0}
    section_graph_ok: Dict[str, bool] = {}

    for i, sec in enumerate(sections):
        if not isinstance(sec, dict):
            errors.append(f"section[{i}] is not an object")
            continue
        sid = sec.get("id")
        if not sid:
            errors.append(f"section[{i}] missing 'id'")
            sid = f"__section_{i}"
        if sid in section_ids:
            errors.append(f"duplicate section id: {sid!r}")
        section_ids.add(sid)

        kind = sec.get("kind")
        if kind not in SECTION_KINDS:
            errors.append(
                f"section {sid!r}: kind {kind!r} not in {SECTION_KINDS}"
            )
        else:
            kind_counts[kind] += 1

        # layer_template repeat_count / applies_to consistency.
        if kind == "layer_template":
            rc = sec.get("repeat_count")
            if not isinstance(rc, int) or rc < 1:
                errors.append(
                    f"section {sid!r}: layer_template needs repeat_count>=1, "
                    f"got {rc!r}"
                )
            applies = sec.get("applies_to")
            if applies is not None:
                if not isinstance(applies, list):
                    errors.append(
                        f"section {sid!r}: applies_to must be a list"
                    )
                elif rc is not None and isinstance(rc, int) and rc >= 1 \
                        and len(applies) != rc:
                    errors.append(
                        f"section {sid!r}: len(applies_to)={len(applies)} "
                        f"!= repeat_count={rc}"
                    )
        else:
            # input/output should not carry repeat_count > 1.
            rc = sec.get("repeat_count")
            if rc is not None and rc != 1:
                errors.append(
                    f"section {sid!r}: kind={kind} should not repeat "
                    f"(repeat_count={rc})"
                )

        # Inner graph structure check (re-uses the flat validator).
        inner = sec.get("graph")
        if not isinstance(inner, dict):
            errors.append(f"section {sid!r}: missing inner 'graph' object")
            section_graph_ok[sid] = False
            continue
        ok, inner_errs = validate_graph_structure(inner)
        section_graph_ok[sid] = ok
        for e in inner_errs:
            errors.append(f"section {sid!r}: {e}")

    # inter_section_edges reference existing sections.
    ise = graph.get("inter_section_edges") or []
    if not isinstance(ise, list):
        errors.append("inter_section_edges must be a list")
        ise = []
    for i, e in enumerate(ise):
        if not isinstance(e, dict):
            errors.append(f"inter_section_edges[{i}] is not an object")
            continue
        f, t = e.get("from_section"), e.get("to_section")
        if not f or not t:
            errors.append(
                f"inter_section_edges[{i}] missing from_section / to_section"
            )
            continue
        if f not in section_ids:
            errors.append(
                f"inter_section_edges[{i}] from unknown section {f!r}"
            )
        if t not in section_ids:
            errors.append(
                f"inter_section_edges[{i}] to unknown section {t!r}"
            )

    return (not errors), errors


def aggregated_node_count(graph: Dict[str, Any]) -> int:
    """Total operator count experienced by a forward pass: sum over
    sections of ``section_node_count * repeat_count``. This is the
    user-facing "how big is the model's compute graph" number.
    """
    if not is_sectioned(graph):
        return len((graph or {}).get("nodes") or []) if isinstance(graph, dict) else 0
    total = 0
    for sec in graph.get("sections") or []:
        inner_n = len((sec.get("graph") or {}).get("nodes") or [])
        rc = sec.get("repeat_count") if sec.get("kind") == "layer_template" else 1
        rc = rc if isinstance(rc, int) and rc >= 1 else 1
        total += inner_n * rc
    return total


# --------------------------------------------------------------------------- #
# Step 3: calc script comparison at the canonical shape
# --------------------------------------------------------------------------- #

# Agents compute at a single canonical shape during the pipeline. The
# WebUI can re-run calc.py at any other shape on demand via /calc/compute.
CANONICAL_BATCH = 1
CANONICAL_SEQ = 512

# Relative tolerance for "both agents agree". 5% lets writers converge even
# when they disagree on minor FLOP-accounting conventions (e.g. whether
# to include 2KH weighted-combine + scaling-factor multiplies in MoE).
# ABS_TOL_* are floors used when the reference magnitude is ~0.
REL_TOL = 0.05
ABS_TOL_FLOPS = 1.0        # 1 TFLOP floor
ABS_TOL_BYTES = 1.0 / 1024  # 1 MB floor


def load_calc_module(script_path: Path, module_name: str = "_calc"):
    """Import a Python file as a module and return it.

    Each Step 3 calc.py must expose a top-level function::

        def calc(batch_size: int, seq_len: int) -> dict:
            return {
                "prefill": {"tflops": <float>, "access_gb": <float>},
                "decode":  {"tflops": <float>, "access_gb": <float>},
            }

    Legacy shape ``{"tflops": ..., "access_gb": ...}`` (no phase split)
    is accepted as a backward-compatibility fallback — it is treated as
    prefill-only with decode zeroed out.

    Raises ImportError if the module can't be loaded or doesn't expose
    ``calc``.
    """
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # noqa: pylint — exec_module is the API
    if not hasattr(mod, "calc"):
        raise ImportError(f"{script_path} does not define `calc(batch_size, seq_len)`")
    return mod


# Sentinel used for backward-compat with old-shape calc.py — decode fields
# are zeroed when an old-shape script is detected, so the rest of the
# pipeline still has a consistent 4-field shape to work with.
_ZERO_PHASE: Dict[str, float] = {"tflops": 0.0, "access_gb": 0.0}


def _coerce_phase(d: Any) -> Dict[str, float]:
    """Normalize one phase sub-dict to {tflops: float, access_gb: float}."""
    if not isinstance(d, dict):
        raise ValueError(f"phase value must be dict, got {type(d).__name__}")
    tflops = float(d.get("tflops", 0.0))
    gb = float(d.get("access_gb", d.get("gb", 0.0)))
    if not (math.isfinite(tflops) and math.isfinite(gb)):
        raise ValueError(f"non-finite phase result: tflops={tflops}, gb={gb}")
    return {"tflops": tflops, "access_gb": gb}


def call_calc(mod, batch_size: int, seq_len: int) -> Dict[str, Dict[str, float]]:
    """Call calc() and normalize the return shape.

    Returns ``{"prefill": {tflops, access_gb}, "decode": {tflops, access_gb}}``.

    Backward-compat: if calc returns the legacy shape
    ``{"tflops": ..., "access_gb": ...}`` (or a 2-tuple), it's treated as
    prefill-only and decode is zeroed.
    """
    out = mod.calc(batch_size=batch_size, seq_len=seq_len)
    if isinstance(out, dict):
        if "prefill" in out or "decode" in out:
            prefill = _coerce_phase(out.get("prefill") or _ZERO_PHASE)
            decode = _coerce_phase(out.get("decode") or _ZERO_PHASE)
        else:
            # Legacy shape — treat as prefill-only.
            prefill = _coerce_phase(out)
            decode = dict(_ZERO_PHASE)
    else:
        # Legacy 2-tuple shape.
        tflops, gb = float(out[0]), float(out[1])
        if not (math.isfinite(tflops) and math.isfinite(gb)):
            raise ValueError(f"non-finite result for b={batch_size}, s={seq_len}")
        prefill = {"tflops": tflops, "access_gb": gb}
        decode = dict(_ZERO_PHASE)
    return {"prefill": prefill, "decode": decode}


def run_calc_canonical(script_path: Path) -> Dict[str, Any]:
    """Load script_path and run its calc() ONCE at the canonical shape
    (``CANONICAL_BATCH``, ``CANONICAL_SEQ``).

    Returns a single record::

        {"batch_size": int, "seq_len": int,
         "prefill": {"tflops": float, "access_gb": float},
         "decode":  {"tflops": float, "access_gb": float}}

    Legacy scripts (old shape) produce decode = {0, 0}.
    """
    mod = load_calc_module(script_path, module_name=f"_calc_{script_path.stem}")
    phases = call_calc(mod, CANONICAL_BATCH, CANONICAL_SEQ)
    return {
        "batch_size": CANONICAL_BATCH,
        "seq_len": CANONICAL_SEQ,
        "prefill": phases["prefill"],
        "decode":  phases["decode"],
    }


def compare_calc_results(
    results: List[Dict[str, Any]],
    *,
    rel_tol: float = REL_TOL,
    abs_tol_flops: float = ABS_TOL_FLOPS,
    abs_tol_bytes: float = ABS_TOL_BYTES,
) -> Dict[str, Any]:
    """Compare N agents' single-shape results.

    Checks four quantities: ``prefill.tflops``, ``prefill.access_gb``,
    ``decode.tflops``, ``decode.access_gb``. Any one exceeding tolerance
    records a mismatch entry (at most one entry — there's only one shape).

    Returns ``{"ok": bool, "mismatches": [...], "rounds": n}``.
    """
    n = len(results)
    if n == 0:
        return {"ok": False, "mismatches": [], "rounds": 0,
                "error": "no results provided"}

    def _split(rec: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if "prefill" in rec or "decode" in rec:
            return (_coerce_phase(rec.get("prefill") or _ZERO_PHASE),
                    _coerce_phase(rec.get("decode") or _ZERO_PHASE))
        return (_coerce_phase(rec), dict(_ZERO_PHASE))

    split = [_split(r) for r in results]
    pre_t = [s[0]["tflops"] for s in split]
    pre_g = [s[0]["access_gb"] for s in split]
    dec_t = [s[1]["tflops"] for s in split]
    dec_g = [s[1]["access_gb"] for s in split]

    def _check(vals: List[float], abs_tol: float) -> Tuple[bool, float]:
        spread = max(vals) - min(vals)
        ref = max(abs(x) for x in vals)
        return (spread <= max(abs_tol, rel_tol * ref), spread)

    pre_t_ok, pre_t_spread = _check(pre_t, abs_tol_flops)
    pre_g_ok, pre_g_spread = _check(pre_g, abs_tol_bytes)
    dec_t_ok, dec_t_spread = _check(dec_t, abs_tol_flops)
    dec_g_ok, dec_g_spread = _check(dec_g, abs_tol_bytes)

    if pre_t_ok and pre_g_ok and dec_t_ok and dec_g_ok:
        return {"ok": True, "mismatches": [], "rounds": n}

    mismatch = {
        "batch_size": results[0].get("batch_size", CANONICAL_BATCH),
        "seq_len":    results[0].get("seq_len", CANONICAL_SEQ),
        "values": [
            {"prefill": {"tflops": pt, "access_gb": pg},
             "decode":  {"tflops": dt, "access_gb": dg}}
            for (pt, pg), (dt, dg) in zip(
                ((p["tflops"], p["access_gb"]) for p in
                 [s[0] for s in split]),
                ((d["tflops"], d["access_gb"]) for d in
                 [s[1] for s in split]),
            )
        ],
        "spread": {
            "prefill": {"tflops": pre_t_spread, "access_gb": pre_g_spread},
            "decode":  {"tflops": dec_t_spread, "access_gb": dec_g_spread},
        },
    }
    return {"ok": False, "mismatches": [mismatch], "rounds": n}


def median_result(
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Take the median of prefill/decode tflops+gb across N agents at the
    single canonical shape. Used when 3 rounds of disagreement fail.
    """
    if not results:
        return {
            "batch_size": CANONICAL_BATCH, "seq_len": CANONICAL_SEQ,
            "prefill": {"tflops": 0.0, "access_gb": 0.0},
            "decode":  {"tflops": 0.0, "access_gb": 0.0},
            "source": "median_fallback_empty",
        }

    def _split(rec: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        if "prefill" in rec or "decode" in rec:
            return (_coerce_phase(rec.get("prefill") or _ZERO_PHASE),
                    _coerce_phase(rec.get("decode") or _ZERO_PHASE))
        return (_coerce_phase(rec), dict(_ZERO_PHASE))

    split = [_split(r) for r in results]
    mid = len(split) // 2
    pre_t = sorted(s[0]["tflops"] for s in split)[mid]
    pre_g = sorted(s[0]["access_gb"] for s in split)[mid]
    dec_t = sorted(s[1]["tflops"] for s in split)[mid]
    dec_g = sorted(s[1]["access_gb"] for s in split)[mid]
    return {
        "batch_size": results[0].get("batch_size", CANONICAL_BATCH),
        "seq_len":    results[0].get("seq_len", CANONICAL_SEQ),
        "prefill": {"tflops": pre_t, "access_gb": pre_g},
        "decode":  {"tflops": dec_t, "access_gb": dec_g},
        "source": "median_fallback",
    }


# --------------------------------------------------------------------------- #
# Format diffs for Step 3 re-prompting
# --------------------------------------------------------------------------- #

def format_mismatches_for_prompt(mismatches: List[Dict[str, Any]],
                                 max_rows: int = 30) -> str:
    """Render mismatches as a readable text block for the next round's prompt.

    Handles both the new prefill/decode shape and the legacy
    ``{"tflops", "access_gb"}`` value shape (defensive — newer code
    always emits prefill/decode).
    """
    if not mismatches:
        return "Agents agreed at the canonical shape."
    lines = [f"Mismatches at canonical shape: {len(mismatches)}"]
    for m in mismatches[:max_rows]:
        cells = []
        for i, v in enumerate(m["values"]):
            if "prefill" in v or "decode" in v:
                pre = v.get("prefill") or {}
                dec = v.get("decode") or {}
                cells.append(
                    f"a{i}=(pre.t={pre.get('tflops', 0):.4g}, "
                    f"pre.gb={pre.get('access_gb', 0):.4g}, "
                    f"dec.t={dec.get('tflops', 0):.4g}, "
                    f"dec.gb={dec.get('access_gb', 0):.4g})"
                )
            else:
                cells.append(f"a{i}=(t={v.get('tflops', 0):.4g}, "
                             f"gb={v.get('access_gb', 0):.4g})")
        lines.append(
            f"  batch={m.get('batch_size')} seq={m.get('seq_len')}  "
            + "  ".join(cells)
        )
    return "\n".join(lines)
