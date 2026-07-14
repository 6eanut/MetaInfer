"""Step 2: build graph.json + iteratively validate every node.

Pipeline::

    build  →  structure-check  →  per-node LLM validation  →  fix  →  loop

* One agent builds the initial graph from memory.json.
* ``deterministic.validate_graph_structure`` checks shape/edges.
* Per-node validation runs through an :class:`AgentPool` with N=3
  workers. Each worker holds ONE ccb session: the first node it
  validates seeds the session (memory slice + section context), and
  every subsequent node resumes that session so the model wakes up
  with the primer + earlier verdicts cached. Per-node prompts shrink
  from ~80KB (full memory) to ~2KB (memory slice) + ~1KB (node delta)
  on resumed turns.
* Fix rounds re-validate ONLY changed nodes + their in-section
  neighbors (shape reconciliation). Unchanged nodes inherit the
  previous verdict — no reason to re-pay the LLM.
* Up to 3 rounds; after that the graph (with flagged nodes noted) is
  accepted and Step 3 proceeds.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from metainfer.orchestrator.agent_pool import AgentPool, PoolTask, PoolTaskResult
from metainfer.orchestrator.subagent_manager import AgentSpec
from . import deterministic as det
from . import prompts as P
from .memory_slice import slice_memory_for_node_as_json


MAX_ROUNDS = 3
PER_AGENT_TIMEOUT_S = 1800
VALIDATE_TIMEOUT_S = 600
# Pool size for per-node validators. N=3 trades peak concurrency for
# much higher context reuse: with 43 nodes, each worker handles ~14
# turns on a single primer — turns 2-14 hit cache for the boilerplate
# and only pay for the new node JSON.
VALIDATOR_POOL_SIZE = 3


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
    memory: Dict[str, Any],
    only_targets: Optional[Set[Tuple[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Run per-node validators through the :class:`AgentPool`.

    The pool keeps N=VALIDATOR_POOL_SIZE worker sessions alive across
    nodes — the first node each worker handles seeds the session
    (memory slice + section context + readonly warning + output
    schema) and every subsequent node resumes that session. Resume
    turns hit ccb's prompt cache for the boilerplate, paying only
    for the new node JSON delta. Net effect on a 43-node graph:
    ~3 primer-turns + ~40 resume-turns, vs the old 43 cold-start
    agents re-reading the full memory each time.

    Memory slicing: each validator only sees the operator_calls /
    uncertainties relevant to ITS node (see :mod:`memory_slice`),
    shrinking the per-node prompt from ~80KB to ~3KB.

    Args:
        only_targets: If supplied, validate ONLY the (section_id,
            node_id) pairs in this set + their in-section neighbors.
            Used by fix rounds — unchanged nodes inherit the prior
            verdict. ``None`` means "validate every node in the graph"
            (used for the first round).

    Returns:
        Verdicts as a list of dicts (same shape as before). Each dict
        has ``section_id`` / ``node_id`` / ``verdict`` / ``reason`` /
        ``suggested_fix`` populated.
    """
    targets: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []
    # (section, node, index_within_section)
    for sec in graph.get("sections") or []:
        sec_graph = sec.get("graph") or {}
        sid = sec.get("id", "section")
        for i, node in enumerate(sec_graph.get("nodes") or []):
            if not isinstance(node, dict):
                continue
            nid = node.get("id") or f"node_{i}"
            if only_targets is not None and (sid, nid) not in only_targets:
                continue
            targets.append((sec, node, i))
    if not targets:
        return []

    # Build the pool task list. Each task has its OWN workdir so its
    # verdict.json doesn't collide with another task's — session state
    # lives in ccb's own storage keyed by session_id, not in workdir.
    pool = AgentPool(
        manager,
        n_workers=VALIDATOR_POOL_SIZE,
        log_dir=round_dir / "logs" / "pool",
        role="node_validator",
        name_prefix=f"validator_{round_dir.name}",
        timeout_s=VALIDATE_TIMEOUT_S,
        stuck_timeout_s=300,
        max_retries=2,
    )
    tasks: List[PoolTask] = []
    meta: List[Tuple[str, str, int]] = []  # parallel to tasks
    for sec, node, idx in targets:
        sid = sec.get("id", "section")
        nid = node.get("id") or f"node_{idx}"
        sec_graph = sec.get("graph") or {}
        section_ctx = {
            "id": sid,
            "kind": sec.get("kind"),
            "repeat_count": sec.get("repeat_count"),
            "applies_to": sec.get("applies_to"),
            "description": sec.get("description"),
        }
        memory_json = slice_memory_for_node_as_json(memory, nid,
                                                    node.get("purpose"))
        # ``common`` already carries a full-memory entry under the same
        # key — replace it with the slice so the prompt format sees a
        # single value.
        local_common = dict(common)
        local_common["memory_json"] = memory_json
        text = P.STEP2_VALIDATE_NODE_PROMPT.format(
            **local_common,
            node_json=json.dumps(node, indent=2, ensure_ascii=False),
            neighbors_json=json.dumps(_neighbors_for(nid, sec_graph),
                                      indent=2, ensure_ascii=False),
            section_json=json.dumps(section_ctx, indent=2,
                                    ensure_ascii=False),
        )
        workdir = round_dir / "tasks" / _safe_validator_name(f"{sid}_{idx}_{nid}")
        tasks.append(PoolTask(
            key=f"{sid}::{nid}",
            prompt=text,
            workdir=workdir,
            name=_safe_validator_name(f"validator_{sid}_{idx}_{nid}"),
        ))
        meta.append((sid, nid, idx))

    pool_results: List[PoolTaskResult] = pool.run(tasks)

    verdicts: List[Dict[str, Any]] = []
    for pr, (sid, nid, idx) in zip(pool_results, meta):
        # Each pool task had its own workdir — pool.run returned the
        # result keyed back to the task. Save the raw response text
        # for forensics, then parse verdict.json.
        task_workdir = tasks[meta.index((sid, nid, idx))].workdir
        (task_workdir / "response.txt").write_text(pr.final_text,
                                                   encoding="utf-8")
        parsed, _ = det.load_agent_json(task_workdir, "verdict.json",
                                        pr.final_text)
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


def _expand_to_neighbors(
    graph: Dict[str, Any], seeds: Set[Tuple[str, str]],
) -> Set[Tuple[str, str]]:
    """Given a set of (section_id, node_id) seeds, return seeds ∪
    their in-section neighbors. After a fix round, changed nodes can
    break shape consistency on adjacent nodes — so we re-validate
    those too. Cross-section neighbors are NOT included (those are
    the orchestrator's responsibility via inter_section_edges)."""
    if not seeds:
        return set()
    out = set(seeds)
    by_section: Dict[str, List[str]] = {}
    section_node_map: Dict[str, Dict[str, Dict]] = {}
    for sec in graph.get("sections") or []:
        sid = sec.get("id")
        sec_graph = sec.get("graph") or {}
        nodes = [n for n in (sec_graph.get("nodes") or [])
                 if isinstance(n, dict) and n.get("id")]
        section_node_map[sid] = {n["id"]: n for n in nodes}
        for n in nodes:
            by_section.setdefault(sid, []).append(n["id"])
    for sid, nid in list(seeds):
        # Use a sentinel default — the seed may reference a section that
        # was deleted or renamed by a fix round between the diff and
        # this expansion (fix agents are free to restructure the graph).
        # An orphaned seed just contributes no neighbors.
        matching = [s for s in graph.get("sections") or []
                    if s.get("id") == sid]
        if not matching:
            continue
        sec_graph = matching[0].get("graph") or {}
        for e in sec_graph.get("edges") or []:
            if not isinstance(e, dict):
                continue
            if e.get("from") == nid and e.get("to"):
                out.add((sid, e["to"]))
            if e.get("to") == nid and e.get("from"):
                out.add((sid, e["from"]))
    return out


def _diff_nodes(
    old_graph: Dict[str, Any], new_graph: Dict[str, Any],
) -> Set[Tuple[str, str]]:
    """Return the set of (section_id, node_id) pairs whose node JSON
    differs between the two graphs, plus nodes that only exist in one
    of them. Used after a fix round to drive incremental validation."""
    old_by_sec: Dict[str, Dict[str, Dict]] = {}
    for sec in old_graph.get("sections") or []:
        sid = sec.get("id")
        sec_graph = sec.get("graph") or {}
        old_by_sec[sid] = {n["id"]: n
                           for n in (sec_graph.get("nodes") or [])
                           if isinstance(n, dict) and n.get("id")}
    new_by_sec: Dict[str, Dict[str, Dict]] = {}
    for sec in new_graph.get("sections") or []:
        sid = sec.get("id")
        sec_graph = sec.get("graph") or {}
        new_by_sec[sid] = {n["id"]: n
                           for n in (sec_graph.get("nodes") or [])
                           if isinstance(n, dict) and n.get("id")}

    changed: Set[Tuple[str, str]] = set()
    all_sids = set(old_by_sec) | set(new_by_sec)
    for sid in all_sids:
        old_nodes = old_by_sec.get(sid, {})
        new_nodes = new_by_sec.get(sid, {})
        for nid in set(old_nodes) | set(new_nodes):
            if old_nodes.get(nid) != new_nodes.get(nid):
                changed.add((sid, nid))
    return changed


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
    memory = json.loads(memory_json)
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

    # Carry forward verdicts from earlier rounds so we don't re-pay
    # the LLM on nodes that haven't changed. Keyed by (section_id, node_id).
    prior_verdicts: Dict[Tuple[str, str], Dict[str, Any]] = {}

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
            verdicts_payload = [{
                "node_id": "_structure",
                "verdict": "reject",
                "reason": "; ".join(errs),
                "suggested_fix": "fix the structural errors and re-emit graph.json",
            }]
            round_dir = rounds_root / f"{round_idx + 1:02d}_fix_struct"
            prev_graph = graph
            graph = _fix_graph(manager, round_dir, common, graph,
                               verdicts_payload)
            (round_dir / "graph.json").write_text(
                json.dumps(graph, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Structural fix likely touched many nodes → drop carry-forward.
            prior_verdicts = {}
            continue

        # Per-node LLM validation. Round 0 = full graph; later rounds =
        # only changed nodes + neighbors (incremental). prior_verdicts
        # supplies the unchanged nodes' verdicts directly.
        t0 = time.time()
        round_dir = rounds_root / f"{round_idx + 1:02d}_validate"
        if round_idx == 0 or not prior_verdicts:
            # First full pass — validate every node.
            only_targets = None
            new_verdicts = _validate_nodes_parallel(
                manager, round_dir, common, graph, memory, only_targets,
            )
        else:
            # Incremental: only changed nodes + their in-section neighbors.
            changed = _diff_nodes(prev_graph, graph)
            targets = _expand_to_neighbors(graph, changed)
            store.append_timeline(
                "calc_value.s2.validate.incremental",
                {"round": round_idx, "changed": len(changed),
                 "with_neighbors": len(targets),
                 "carried_forward": len(prior_verdicts) - len(targets)},
            )
            new_verdicts = _validate_nodes_parallel(
                manager, round_dir, common, graph, memory,
                only_targets=targets,
            )
            # Merge: carry forward unchanged nodes' verdicts, overwrite
            # any node that was just re-validated.
            merged: Dict[Tuple[str, str], Dict[str, Any]] = dict(prior_verdicts)
            for v in new_verdicts:
                merged[(v.get("section_id"), v.get("node_id"))] = v
            new_verdicts = list(merged.values())
        (round_dir / "verdicts.json").write_text(
            json.dumps(new_verdicts, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Refresh carry-forward state.
        prior_verdicts = {
            (v.get("section_id"), v.get("node_id")): v for v in new_verdicts
            if v.get("section_id") and v.get("node_id")
        }
        n_reject = sum(1 for v in new_verdicts if v.get("verdict") == "reject")
        n_pass = sum(1 for v in new_verdicts if v.get("verdict") == "pass")
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
        prev_graph = graph
        graph = _fix_graph(manager, round_dir, common, graph, new_verdicts)
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
