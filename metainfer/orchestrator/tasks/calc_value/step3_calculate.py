"""Step 3: per-node FLOPs / mem-traffic calc with 3-agent agreement.

Algorithm (per node):

1. Spawn 3 INDEPENDENT writer agents (different prompts / angles) that
   each produce a ``calc.py`` file defining ``calc(batch_size, seq_len)
   -> {"tflops": float, "access_gb": float}``.
2. Deterministic comparator (:func:`deterministic.compare_calc_grids`)
   runs each script on the 42-combo cartesian product
   (seq_len x batch_size) and checks agreement within rel-tol 1e-6.
3. If all 42 combos agree across 3 writers → accept that node.
4. Otherwise, re-spawn the disagreeing writers with the diff feedback
   (their previous script + the numeric mismatches, NOT the other
   writers' source code). Iterate.
5. Hard cap 15 rounds per node. After cap → take median of the 3
   outputs for each combo, mark ``approximate: true`` in the meta file,
   move on.

Output layout::

    step3/
    ├── rounds/<node_id>/<round_idx>/{writer_0, writer_1, writer_2}/calc.py
    ├── rounds/<node_id>/<round_idx>/grid_<i>.json
    ├── rounds/<node_id>/<round_idx>/comparison.json
    └── final/
        ├── <node_id>.py          # final winning calc.py
        └── <node_id>.meta.json   # {approximate, source_agent, rounds, ...}
"""

from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...subagent_manager import AgentSpec
from . import deterministic as det
from . import prompts as P


MAX_ROUNDS_PER_NODE = 15
PER_AGENT_TIMEOUT_S = 900  # 15 min per writer
WRITER_COUNT = 3


def _format_env_block(env_vars: str) -> str:
    if not env_vars:
        return "(none)"
    return "\n".join(f"  {ln}" for ln in env_vars.splitlines() if ln.strip())


def _safe_node_id(node_id: str) -> str:
    """Filesystem-safe node id (used for dir names)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", node_id)[:80] or "node"


def _write_prompt(workdir: Path, name: str, text: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / f"{name}.prompt.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _extract_python_source(text: str) -> str:
    """Pull a Python code block out of an LLM response.

    Handles:
    * ```python\n...\n``` fenced
    * ```\n...\n``` bare-fenced (only if contents look like Python)
    * bare source (whole text starts with `def` / `import` / comment)
    """
    if not text:
        return ""
    # Fenced python first.
    m = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1)
    # Bare fenced: contents must contain 'def calc'.
    for m in re.finditer(r"```\s*\n(.*?)\n```", text, re.DOTALL):
        if "def calc" in m.group(1):
            return m.group(1)
    # Bare: full text contains def calc.
    if "def calc" in text:
        return text.strip()
    return ""


def _spawn_writers(
    manager,
    work_root: Path,
    node: Dict[str, Any],
    memory_json: str,
    req: Dict[str, Any],
    prev_scripts: Optional[List[Optional[str]]] = None,
    mismatches: Optional[List[Dict[str, Any]]] = None,
) -> List[Tuple[AgentSpec, str]]:
    """Launch 3 writers in parallel. Returns list of (spec, source_text).

    On round 0: all 3 use the angle-specific writer prompts.
    On round >=1: only WRITER_COUNT writers re-launch, each using the
    STEP3_FIX_PROMPT with their own previous script.
    """
    node_json = json.dumps(node, indent=2, ensure_ascii=False)
    readonly = P.READONLY_WARNING.format(
        model_dir=req["model_dir"],
        framework_dir=req["framework_source_dir"],
    )
    calc_contract = P.STEP3_CALC_FUNC_CONTRACT
    common = {
        "node_json": node_json,
        "memory_json": memory_json,
        "readonly": readonly,
        "calc_contract": calc_contract,
    }

    nid = _safe_node_id(node.get("id", "node"))
    specs_and_sources: List[Tuple[AgentSpec, str]] = []

    threads: List = []
    spec_list: List[AgentSpec] = []

    for i in range(WRITER_COUNT):
        name = f"writer_{i}"
        workdir = work_root / name
        log_dir = work_root / "logs" / name
        if prev_scripts is None:
            text = P.STEP3_WRITER_PROMPTS[i].format(**common)
        else:
            prev = prev_scripts[i] if i < len(prev_scripts) else None
            if prev is None:
                # Writer that produced no script last round — use the
                # standard angle prompt to start fresh.
                text = P.STEP3_WRITER_PROMPTS[i].format(**common)
            else:
                text = P.STEP3_FIX_PROMPT.format(
                    **common,
                    your_script=prev,
                    mismatches=det.format_mismatches_for_prompt(mismatches or []),
                )
        prompt_file = _write_prompt(workdir, name, text)
        spec = AgentSpec(
            name=f"{nid}_{name}",  # globally unique for manager.result lookup
            role="calc_writer",
            prompt_file=prompt_file, workdir=workdir, log_dir=log_dir,
            timeout_s=PER_AGENT_TIMEOUT_S, stuck_timeout_s=300, max_retries=2,
        )
        spec_list.append(spec)

    # Parallel launch.
    for spec in spec_list:
        t = manager.launch_async(spec)
        threads.append(t)
    for t in threads:
        t.join()

    # Collect results.
    for spec in spec_list:
        result = manager.result(spec.name)
        if result is None or not result.success:
            err = result.error if result else "no result"
            print(f"[calc-value.S3] writer {spec.name} failed: {err}", flush=True)
            (spec.workdir / "error.txt").write_text(str(err), encoding="utf-8")
            specs_and_sources.append((spec, ""))
            continue
        text = result.final_text or ""
        (spec.workdir / "response.txt").write_text(text, encoding="utf-8")
        # File-first: prefer calc.py the agent Wrote directly; only fall
        # back to scraping response.txt if the file is missing/empty.
        src, source = det.load_agent_text_file(
            spec.workdir, "calc.py", text, _extract_python_source,
        )
        if src:
            if source == "response":
                # Agent inlined the source instead of Writing — preserve
                # what we scraped to the canonical filename so downstream
                # sees a consistent layout.
                (spec.workdir / "calc.py").write_text(src, encoding="utf-8")
        else:
            (spec.workdir / "parse_error.txt").write_text(
                "No `def calc` block found in calc.py or response.txt.\n"
                f"Response first 500 chars:\n{text[:500]}",
                encoding="utf-8",
            )
        specs_and_sources.append((spec, src))
    return specs_and_sources


def _run_one_calc(script_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Run a calc.py on the 42-combo grid. Returns None on script error."""
    try:
        return det.run_calc_on_grid(script_path)
    except Exception as exc:  # noqa: BLE001
        return None


def _process_one_node(
    *,
    node: Dict[str, Any],
    manager,
    step3_dir: Path,
    memory_json: str,
    req: Dict[str, Any],
    store,
    compound_id: Optional[str] = None,
) -> Tuple[Path, Dict[str, Any]]:
    """Run the 3-writer iterative convergence for ONE node.

    Returns (final_calc_script_path, meta). ``compound_id`` overrides
    the on-disk filename — used by the sectioned-graph flow so two
    sections with same-named nodes (e.g. ``input_norm`` in both dense
    and MoE templates) don't collide in ``final/``.
    """
    nid_raw = node.get("id", "node")
    nid = compound_id or _safe_node_id(nid_raw)
    node_root = step3_dir / "rounds" / nid
    node_root.mkdir(parents=True, exist_ok=True)
    final_dir = step3_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    prev_scripts: Optional[List[Optional[str]]] = None
    prev_mismatches: Optional[List[Dict[str, Any]]] = None
    last_grids: Optional[List[List[Dict[str, Any]]]] = None

    final_round = 0
    approximate = False

    for round_idx in range(MAX_ROUNDS_PER_NODE):
        round_dir = node_root / f"round_{round_idx:02d}"
        t0 = time.time()
        results = _spawn_writers(
            manager, round_dir, node, memory_json, req,
            prev_scripts=prev_scripts, mismatches=prev_mismatches,
        )
        scripts = [src for (_, src) in results]
        script_paths: List[Optional[Path]] = []
        for (spec, src) in results:
            if src:
                script_paths.append(spec.workdir / "calc.py")
            else:
                script_paths.append(None)

        # Run each script on the grid.
        grids: List[List[Dict[str, Any]]] = []
        usable_paths: List[Path] = []
        for p in script_paths:
            if p is None:
                continue
            grid = _run_one_calc(p)
            if grid is None:
                # Script failed at runtime.
                continue
            grids.append(grid)
            usable_paths.append(p)

        # Persist grids.
        for i, grid in enumerate(grids):
            (round_dir / f"grid_{i}.json").write_text(
                json.dumps(grid, indent=2, ensure_ascii=False), encoding="utf-8",
            )

        if len(grids) < WRITER_COUNT:
            store.append_timeline(
                "calc_value.s3.node.round.partial",
                {"node": nid, "round": round_idx,
                 "usable_scripts": len(grids),
                 "note": "some writer scripts failed; retrying"},
            )

        if len(grids) == 0:
            # No usable scripts this round. Give up after the cap.
            if round_idx == MAX_ROUNDS_PER_NODE - 1:
                approximate = True
                break
            prev_scripts = [s or None for s in scripts]
            prev_mismatches = []
            continue

        last_grids = grids
        comparison = det.compare_calc_grids(grids)
        (round_dir / "comparison.json").write_text(
            json.dumps(comparison, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        elapsed = time.time() - t0
        store.append_timeline(
            "calc_value.s3.node.round.done",
            {"node": nid, "round": round_idx,
             "ok": comparison["ok"],
             "mismatches": len(comparison.get("mismatches") or []),
             "elapsed_s": round(elapsed, 1)},
        )

        if comparison["ok"] and len(grids) >= WRITER_COUNT:
            # Converged. Accept the FIRST writer's script as canonical
            # (all 3 agree, so choice is arbitrary).
            final_script = usable_paths[0]
            final_round = round_idx
            (final_dir / f"{nid}.py").write_text(
                final_script.read_text(encoding="utf-8"), encoding="utf-8",
            )
            (final_dir / f"{nid}.meta.json").write_text(
                json.dumps({
                    "node_id": nid_raw,
                    "approximate": False,
                    "source_agent": "unanimous",
                    "rounds": round_idx + 1,
                    "writer_scripts": [str(p) for p in usable_paths],
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            return (final_dir / f"{nid}.py"), {"approximate": False,
                                                "rounds": round_idx + 1}

        if round_idx == MAX_ROUNDS_PER_NODE - 1:
            # Hard cap — median fallback.
            print(f"[calc-value.S3] node {nid} did not converge after "
                  f"{MAX_ROUNDS_PER_NODE} rounds; using median fallback.",
                  flush=True)
            approximate = True
            median_grid = det.median_fallback(grids)
            (round_dir / "median_fallback.json").write_text(
                json.dumps(median_grid, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Synthesize a "median" calc script that delegates to a small
            # lookup function — but since calc() must accept arbitrary
            # (batch_size, seq_len), we cannot use a lookup. Instead, pick
            # the writer whose grid is closest to the median on the most
            # combos.
            from statistics import median
            best_writer_idx = _pick_most_median_writer(grids, median_grid)
            final_script = usable_paths[best_writer_idx]
            (final_dir / f"{nid}.py").write_text(
                final_script.read_text(encoding="utf-8"), encoding="utf-8",
            )
            (final_dir / f"{nid}.meta.json").write_text(
                json.dumps({
                    "node_id": nid_raw,
                    "approximate": True,
                    "source_agent": f"median_fallback_from_writer_{best_writer_idx}",
                    "rounds": MAX_ROUNDS_PER_NODE,
                    "mismatch_count_at_cap": len(comparison.get("mismatches") or []),
                    "writer_scripts": [str(p) for p in usable_paths],
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            store.append_timeline(
                "calc_value.s3.node.median_fallback",
                {"node": nid,
                 "mismatches_at_cap": len(comparison.get("mismatches") or [])},
            )
            return (final_dir / f"{nid}.py"), {"approximate": True,
                                                "rounds": MAX_ROUNDS_PER_NODE}

        # Not converged, not at cap —> prepare diff feedback.
        prev_scripts = [s or None for s in scripts]
        prev_mismatches = comparison["mismatches"]
        final_round = round_idx

    # Unreachable: every loop branch either returns or hits the cap branch.
    # But just in case, write a degenerate result.
    print(f"[calc-value.S3] node {nid} exited loop unexpectedly", flush=True)
    fallback_script = (last_grids is not None and len(grids) > 0 and usable_paths)
    if fallback_script:
        (final_dir / f"{nid}.py").write_text(
            usable_paths[0].read_text(encoding="utf-8"), encoding="utf-8",
        )
    else:
        (final_dir / f"{nid}.py").write_text(
            "def calc(batch_size, seq_len):\n    return {'tflops': 0.0, 'access_gb': 0.0}\n",
            encoding="utf-8",
        )
    (final_dir / f"{nid}.meta.json").write_text(
        json.dumps({"node_id": nid_raw, "approximate": True,
                    "source_agent": "degenerate_fallback"},
                   indent=2),
        encoding="utf-8",
    )
    return (final_dir / f"{nid}.py"), {"approximate": True, "rounds": final_round + 1}


def _pick_most_median_writer(grids: List[List[Dict[str, Any]]],
                              median_grid: List[Dict[str, Any]]) -> int:
    """Pick the writer whose values are closest to the median grid overall."""
    if not grids:
        return 0
    best_idx = 0
    best_err = float("inf")
    for i, g in enumerate(grids):
        err = 0.0
        for r1, r2 in zip(g, median_grid):
            err += abs(r1["tflops"] - r2["tflops"])
            err += abs(r1["access_gb"] - r2["access_gb"])
        if err < best_err:
            best_err = err
            best_idx = i
    return best_idx


def run_step3_calculate(
    *,
    req: Dict[str, Any],
    store,
    manager,
    paths: Dict[str, Path],
    graph_path: Path,
) -> Path:
    """Iterate over every (section, node) in graph.json. Returns final/ dir.

    For each node inside each section we produce a calc.py that computes
    ONE occurrence's FLOPs / bytes (if the section is a ``layer_template``
    with ``repeat_count=N``, the calc result represents one of the N
    layers — the aggregator in S4 + /compute multiplies by N). Meta
    files are stamped with section context (section_id, kind,
    repeat_count) so downstream can find them and aggregate correctly.
    """
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    graph = det.normalize_graph(raw)
    memory_path = paths["step1_dir"] / "memory.json"
    memory_json = memory_path.read_text(encoding="utf-8") \
        if memory_path.exists() else "{}"

    step3_dir = paths["step3_dir"]
    final_dir = step3_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    # Flatten (section, node) targets so we still get deterministic order.
    targets: List[Tuple[Dict[str, Any], Dict[str, Any], int]] = []
    for sec in graph.get("sections") or []:
        for i, node in enumerate((sec.get("graph") or {}).get("nodes") or []):
            if isinstance(node, dict):
                targets.append((sec, node, i))
    total = len(targets)
    store.append_timeline(
        "calc_value.s3.start",
        {"nodes": total,
         "aggregated_nodes": det.aggregated_node_count(graph),
         "sections": len(graph.get("sections") or [])},
    )

    summary = []
    for i, (sec, node, idx_in_sec) in enumerate(targets):
        nid_raw = node.get("id", f"node_{idx_in_sec}")
        sid = sec.get("id", "section")
        # Globally-unique calc filename = section_id + node_id, so two
        # sections with a same-named node (e.g. "input_norm" appearing
        # in both dense + MoE templates) don't collide.
        compound = f"{sid}__{nid_raw}"
        safe = _safe_node_id(compound)
        t0 = time.time()
        try:
            script_path, meta = _process_one_node(
                node=node, manager=manager, step3_dir=step3_dir,
                memory_json=memory_json, req=req, store=store,
                compound_id=safe,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[calc-value.S3] node {compound} crashed: {exc}\n{tb}",
                  flush=True)
            store.append_timeline(
                "calc_value.s3.node.crashed",
                {"node": nid_raw, "section_id": sid, "error": str(exc)},
            )
            (final_dir / f"{safe}.py").write_text(
                "def calc(batch_size, seq_len):\n"
                "    return {'tflops': 0.0, 'access_gb': 0.0}\n",
                encoding="utf-8",
            )
            (final_dir / f"{safe}.meta.json").write_text(
                json.dumps({"node_id": nid_raw, "section_id": sid,
                            "approximate": True,
                            "source_agent": "crash_fallback",
                            "error": str(exc)}, indent=2),
                encoding="utf-8",
            )
            meta = {"approximate": True, "crashed": True}
            script_path = final_dir / f"{safe}.py"
        # Stamp section context into meta so downstream aggregation
        # (S4 viz + /compute endpoint) can multiply by repeat_count.
        meta_path = final_dir / f"{safe}.meta.json"
        if meta_path.exists():
            try:
                existing = json.loads(meta_path.read_text(encoding="utf-8"))
            except ValueError:
                existing = {}
        else:
            existing = {}
        existing.update({
            "node_id": nid_raw,
            "section_id": sid,
            "section_kind": sec.get("kind"),
            "section_repeat_count": (
                sec.get("repeat_count") if sec.get("kind") == "layer_template"
                else 1
            ),
            "section_applies_to": sec.get("applies_to"),
            "compound_id": safe,
        })
        meta_path.write_text(
            json.dumps(existing, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        elapsed = time.time() - t0
        summary.append({
            "node_id": nid_raw, "section_id": sid,
            "compound_id": safe,
            "section_kind": sec.get("kind"),
            "repeat_count": existing["section_repeat_count"],
            "approximate": meta.get("approximate"),
            "rounds": meta.get("rounds"),
            "elapsed_s": round(elapsed, 1),
        })
        store.append_timeline(
            "calc_value.s3.node.done",
            {"node": nid_raw, "section_id": sid, "i": i, "total": total,
             "approximate": meta.get("approximate"),
             "rounds": meta.get("rounds"),
             "elapsed_s": round(elapsed, 1)},
        )

    (final_dir / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    store.append_timeline(
        "calc_value.s3.all_nodes_done",
        {"total": total,
         "approximate_count": sum(1 for s in summary if s["approximate"])},
    )
    return final_dir
