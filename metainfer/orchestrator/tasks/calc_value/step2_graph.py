"""Step 2: build graph.json + iteratively validate every node.

Pipeline::

    build  →  structure-check  →  per-node LLM validation  →  fix  →  loop

* One agent builds the initial graph from memory.json.
* ``deterministic.validate_graph_structure`` checks shape/edges.
* Per-node validation: ONE agent per node, all in parallel. Each agent
  gets the node + its neighbors + the memory and returns a pass/reject
  verdict.
* If any node rejects, one fix-agent rewrites graph.json to address
  all failures. Re-run structure check + per-node validation.
* Up to 3 rounds; after that the graph (with flagged nodes noted) is
  accepted and Step 3 proceeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...subagent_manager import AgentSpec
from . import deterministic as det
from . import prompts as P


MAX_ROUNDS = 3
PER_AGENT_TIMEOUT_S = 1800
VALIDATE_TIMEOUT_S = 600


def _format_env_block(env_vars: str) -> str:
    if not env_vars:
        return "(none)"
    return "\n".join(f"  {ln}" for ln in env_vars.splitlines() if ln.strip())


def _common_format(req: Dict[str, Any], memory_json: str) -> Dict[str, str]:
    return {
        "model_dir": req["model_dir"],
        "framework_dir": req["framework_source_dir"],
        "memory_json": memory_json,
        "cmdline": req.get("cmdline_args") or "(none)",
        "env_block": _format_env_block(req.get("env_vars") or ""),
        "readonly": P.READONLY_WARNING.format(
            model_dir=req["model_dir"],
            framework_dir=req["framework_source_dir"],
        ),
        "output_schema": P.STEP2_OUTPUT_SCHEMA,
    }


def _write_prompt(workdir: Path, name: str, text: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / f"{name}.prompt.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _build_graph(manager, round_dir: Path, common: Dict[str, str]) -> Dict[str, Any]:
    """Run 1 agent to produce graph.json. Returns parsed graph dict."""
    name = "graph_builder"
    workdir = round_dir / name
    log_dir = round_dir / "logs" / name
    text = P.STEP2_BUILD_PROMPT.format(**common)
    prompt_file = _write_prompt(workdir, name, text)
    spec = AgentSpec(
        name=name, role="graph_builder",
        prompt_file=prompt_file, workdir=workdir, log_dir=log_dir,
        timeout_s=PER_AGENT_TIMEOUT_S, stuck_timeout_s=900, max_retries=2,
    )
    manager.launch(spec)
    return _extract_graph_from_result(manager, spec, workdir)


def _fix_graph(manager, round_dir: Path, common: Dict[str, str],
               graph: Dict[str, Any], verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Run 1 agent to repair graph.json. Returns parsed graph dict."""
    name = "graph_fixer"
    workdir = round_dir / name
    log_dir = round_dir / "logs" / name
    verdicts_text = json.dumps(verdicts, indent=2, ensure_ascii=False)
    text = P.STEP2_FIX_PROMPT.format(
        **common,
        graph_json=json.dumps(graph, indent=2, ensure_ascii=False),
        verdicts=verdicts_text,
    )
    prompt_file = _write_prompt(workdir, name, text)
    spec = AgentSpec(
        name=name, role="graph_fixer",
        prompt_file=prompt_file, workdir=workdir, log_dir=log_dir,
        timeout_s=PER_AGENT_TIMEOUT_S, stuck_timeout_s=900, max_retries=2,
    )
    manager.launch(spec)
    return _extract_graph_from_result(manager, spec, workdir)


def _extract_graph_from_result(manager, spec, workdir: Path) -> Dict[str, Any]:
    """Pull the sectioned graph dict out of the agent's workdir.

    File-first: reads ``workdir/graph.json`` (the agent was instructed to
    Write it). Falls back to scraping ``response.txt`` only if the file
    is missing or unparseable. Accepts both the sectioned schema
    (``{"sections": [...]}``) and legacy flat ``{"nodes": [...],
    "edges": [...]}`` — the latter is normalized via
    :func:`det.normalize_graph` so downstream code only sees one shape.
    """
    result = manager.result(spec.name)
    if result is None or not result.success:
        raise RuntimeError(
            f"graph agent {spec.name} failed: "
            f"{result.error if result else 'no result'}"
        )
    text = result.final_text or ""
    (workdir / "response.txt").write_text(text, encoding="utf-8")
    parsed, source = det.load_agent_json(workdir, "graph.json", text)
    if not isinstance(parsed, dict) or (
        "sections" not in parsed and "nodes" not in parsed
    ):
        raise RuntimeError(
            f"graph agent {spec.name} did not produce a graph dict "
            f"(checked graph.json and response.txt; source={source})"
        )
    # Normalize legacy flat shape → single-section sectioned graph so
    # the rest of S2/S3/S4 only has one shape to deal with.
    return det.normalize_graph(parsed)


def _neighbors_for(
    node_id: str, section_graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Return the upstream + downstream node dicts adjacent to node_id
    WITHIN its own section. Cross-section neighbors are the
    orchestrator's job (via ``inter_section_edges``) and are not
    surfaced to per-node validators."""
    ids_up: List[str] = []
    ids_down: List[str] = []
    for e in section_graph.get("edges") or []:
        if not isinstance(e, dict):
            continue
        if e.get("to") == node_id:
            ids_up.append(e.get("from"))
        if e.get("from") == node_id:
            ids_down.append(e.get("to"))
    by_id = {n.get("id"): n for n in section_graph.get("nodes") or []
             if isinstance(n, dict) and n.get("id")}
    return [by_id[i] for i in ids_up + ids_down if i in by_id]


def _safe_validator_name(s: str) -> str:
    """Make a filesystem-safe validator name from a section id +
    node id. The SubAgentManager requires globally-unique agent names."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:80] or "node"


def _validate_nodes_parallel(
    manager, round_dir: Path, common: Dict[str, str],
    graph: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Spawn ONE validator per (section, node) pair, all in parallel.

    The validator prompt receives the node's section context (kind,
    repeat_count, applies_to) so it can reason about whether the
    node's shapes are right for what it represents.
    """
    targets: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []
    # (section, node, index_within_section)
    for sec in graph.get("sections") or []:
        sec_graph = sec.get("graph") or {}
        for i, node in enumerate(sec_graph.get("nodes") or []):
            if isinstance(node, dict):
                targets.append((sec, node, i))
    if not targets:
        return []

    specs: List[AgentSpec] = []
    # Track the (section, node_id, idx) for each spec so we can stamp
    # the verdict back with full context.
    spec_meta: List[Tuple[str, str, int]] = []
    for sec, node, idx in targets:
        sid = sec.get("id", "section")
        nid = node.get("id") or f"node_{idx}"
        name = _safe_validator_name(f"validator_{sid}_{idx}_{nid}")
        workdir = round_dir / name
        log_dir = round_dir / "logs" / name
        section_ctx = {
            "id": sid,
            "kind": sec.get("kind"),
            "repeat_count": sec.get("repeat_count"),
            "applies_to": sec.get("applies_to"),
            "description": sec.get("description"),
        }
        text = P.STEP2_VALIDATE_NODE_PROMPT.format(
            **common,
            node_json=json.dumps(node, indent=2, ensure_ascii=False),
            neighbors_json=json.dumps(_neighbors_for(nid, sec_graph),
                                      indent=2, ensure_ascii=False),
            section_json=json.dumps(section_ctx, indent=2,
                                    ensure_ascii=False),
        )
        prompt_file = _write_prompt(workdir, name, text)
        specs.append(AgentSpec(
            name=name, role="node_validator",
            prompt_file=prompt_file, workdir=workdir, log_dir=log_dir,
            timeout_s=VALIDATE_TIMEOUT_S, stuck_timeout_s=300, max_retries=2,
        ))
        spec_meta.append((sid, nid, idx))

    threads = [manager.launch_async(s) for s in specs]
    for t in threads:
        t.join()

    verdicts: List[Dict[str, Any]] = []
    for spec, (sid, nid, idx) in zip(specs, spec_meta):
        result = manager.result(spec.name)
        text = (result.final_text if result else "") or ""
        (spec.workdir / "response.txt").write_text(text, encoding="utf-8")
        parsed, _ = det.load_agent_json(spec.workdir, "verdict.json", text)
        if not isinstance(parsed, dict):
            verdicts.append({
                "section_id": sid,
                "node_id": nid,
                "verdict": "reject",
                "reason": "validator returned unparseable response",
                "suggested_fix": None,
                "_parse_failed": True,
            })
            continue
        parsed.setdefault("section_id", sid)
        parsed.setdefault("node_id", nid)
        verdicts.append(parsed)
    return verdicts


def run_step2_graph(
    *,
    req: Dict[str, Any],
    store,
    manager,
    paths: Dict[str, Path],
    memory_path: Path,
) -> Path:
    """Build + validate graph.json. Returns its path."""
    memory_json = memory_path.read_text(encoding="utf-8")
    common = _common_format(req, memory_json)

    step2_dir = paths["step2_dir"]
    rounds_root = step2_dir / "rounds"
    rounds_root.mkdir(parents=True, exist_ok=True)

    # Round 0: build.
    t0 = time.time()
    store.append_timeline("calc_value.s2.build.start", {})
    graph = _build_graph(manager, rounds_root / "00_build", common)
    (rounds_root / "00_build" / "graph.json").write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    store.append_timeline(
        "calc_value.s2.build.done",
        {"elapsed_s": round(time.time() - t0, 1),
         "sections": len(graph.get("sections") or []),
         "template_nodes": det.section_node_count(graph),
         "aggregated_nodes": det.aggregated_node_count(graph)},
    )

    # Iterative validate + fix.
    for round_idx in range(MAX_ROUNDS):
        # Structure check first (cheap, no LLM).
        ok, errs = det.validate_sectioned_graph(graph)
        if not ok:
            store.append_timeline(
                "calc_value.s2.structure_check.failed",
                {"round": round_idx, "errors": errs[:5]},
            )
            # Try a fix agent — pass the structural errors as the verdicts.
            verdicts = [{
                "node_id": "_structure",
                "verdict": "reject",
                "reason": "; ".join(errs),
                "suggested_fix": "fix the structural errors and re-emit graph.json",
            }]
            round_dir = rounds_root / f"{round_idx + 1:02d}_fix_struct"
            graph = _fix_graph(manager, round_dir, common, graph, verdicts)
            (round_dir / "graph.json").write_text(
                json.dumps(graph, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            continue

        # Per-node LLM validation (parallel).
        t0 = time.time()
        round_dir = rounds_root / f"{round_idx + 1:02d}_validate"
        verdicts = _validate_nodes_parallel(manager, round_dir, common, graph)
        (round_dir / "verdicts.json").write_text(
            json.dumps(verdicts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        n_reject = sum(1 for v in verdicts if v.get("verdict") == "reject")
        n_pass = sum(1 for v in verdicts if v.get("verdict") == "pass")
        store.append_timeline(
            "calc_value.s2.validate.done",
            {"round": round_idx, "pass": n_pass, "reject": n_reject,
             "elapsed_s": round(time.time() - t0, 1)},
        )

        if n_reject == 0:
            # All pass — done.
            break

        if round_idx == MAX_ROUNDS - 1:
            store.append_timeline(
                "calc_value.s2.did_not_converge",
                {"round": round_idx, "remaining_rejects": n_reject},
            )
            print(f"[calc-value.S2] {n_reject} node(s) still rejected after "
                  f"{MAX_ROUNDS} rounds; accepting graph with flags.",
                  flush=True)
            break

        # Fix round.
        round_dir = rounds_root / f"{round_idx + 2:02d}_fix"
        graph = _fix_graph(manager, round_dir, common, graph, verdicts)
        (round_dir / "graph.json").write_text(
            json.dumps(graph, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # Final graph.json
    final_path = step2_dir / "graph.json"
    final_path.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return final_path
