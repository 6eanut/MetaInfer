"""Step 0: rough single-pass estimate.

Single agent reads config.json, derives back-of-envelope formulas for
every standard LLM operator node, and writes one simplified calc.py per
node under ``step0/per_node/<compound>.py`` plus a ``rough_graph.json``
manifest. The orchestrator then runs each calc.py on the 42-combo grid
and aggregates results into ``rough_results.json`` for the WebUI.

Goal: get defensible numbers on screen within minutes, before the
detailed audit (S1+S2+S3) catches up. The detailed audit overwrites
these numbers in step3/final/.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...subagent_manager import AgentSpec
from . import deterministic as det
from . import prompts as P


PER_AGENT_TIMEOUT_S = 1800  # 30 min — single agent does everything


def _format_env_block(env_vars: str) -> str:
    if not env_vars:
        return "(none)"
    return "\n".join(f"  {ln}" for ln in env_vars.splitlines() if ln.strip())


def _safe_node_id(node_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", node_id)[:80] or "node"


def _write_prompt(workdir: Path, name: str, text: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / f"{name}.prompt.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _launch_rough_agent(
    *,
    manager,
    step0_dir: Path,
    req: Dict[str, Any],
) -> Path:
    """Launch the single rough-pass agent. Returns its workdir."""
    workdir = step0_dir / "agent_rough"
    log_dir = step0_dir / "logs"
    readonly = P.READONLY_WARNING.format(
        model_dir=req["model_dir"],
        framework_dir=req["framework_source_dir"],
    )
    calc_contract = P.STEP3_CALC_FUNC_CONTRACT
    text = P.STEP0_ROUGH_PROMPT.format(
        readonly=readonly,
        cmdline=req.get("cmdline_args") or "(none)",
        env_block=_format_env_block(req.get("env_vars") or ""),
        calc_contract=calc_contract,
        workdir=str(workdir),
    )
    prompt_file = _write_prompt(workdir, "rough", text)
    spec = AgentSpec(
        name="calc_value_s0_rough",
        role="calc_writer",
        prompt_file=prompt_file,
        workdir=workdir,
        log_dir=log_dir,
        timeout_s=PER_AGENT_TIMEOUT_S,
        stuck_timeout_s=600,
        max_retries=2,
    )
    t = manager.launch_async(spec)
    t.join()
    result = manager.result(spec.name)
    if result is None or not result.success:
        err = result.error if result else "no result"
        (workdir / "error.txt").write_text(str(err), encoding="utf-8")
        raise RuntimeError(f"rough agent failed: {err}")
    (workdir / "response.txt").write_text(result.final_text or "", encoding="utf-8")
    return workdir


def _load_rough_graph(workdir: Path) -> Dict[str, Any]:
    """Read rough_graph.json. Falls back to scraping per_node/ if missing."""
    graph_path = workdir / "rough_graph.json"
    if graph_path.exists():
        try:
            return json.loads(graph_path.read_text(encoding="utf-8"))
        except ValueError:
            pass
    # Fallback: synthesize a graph from per_node/ filenames.
    per_node = workdir / "per_node"
    nodes: List[Dict[str, Any]] = []
    sections_by_id: Dict[str, Dict[str, Any]] = {}
    if per_node.exists():
        for py in sorted(per_node.glob("*.py")):
            compound = py.stem
            if "__" in compound:
                sid, nid = compound.split("__", 1)
            else:
                sid, nid = "layer", compound
            sec = sections_by_id.setdefault(sid, {
                "id": sid, "kind": "layer_template", "repeat_count": 1,
                "graph": {"nodes": []},
            })
            sec["graph"]["nodes"].append({
                "id": nid, "op": "unknown", "compound": compound,
            })
    return {"sections": list(sections_by_id.values()), "fallback": True}


def _run_per_node_grid(
    workdir: Path,
    graph: Dict[str, Any],
    store,
) -> List[Dict[str, Any]]:
    """Run each per_node/<compound>.py on the 42-combo grid.

    Returns a list of result rows:
        [{compound, section_id, node_id, ok, grid, error, tflops_picked, gb_picked}]
    where tflops_picked/gb_picked are the values at (B=1, S=2048) for
    quick UI display. (B=1, S=2048) isn't in the standard 42-combo grid,
    so we ALSO run that specific combo separately.
    """
    per_node = workdir / "per_node"
    results: List[Dict[str, Any]] = []
    if not per_node.exists():
        store.append_timeline(
            "calc_value.s0.per_node_missing",
            {"workdir": str(workdir)},
        )
        return results

    # Build compound -> (section, node) index from the graph.
    compounds: Dict[str, Dict[str, str]] = {}
    for sec in graph.get("sections") or []:
        sid = sec.get("id", "section")
        for node in (sec.get("graph") or {}).get("nodes") or []:
            if not isinstance(node, dict):
                continue
            compound = node.get("compound") or f"{sid}__{node.get('id', 'node')}"
            compounds[_safe_node_id(compound)] = {
                "section_id": sid,
                "node_id": node.get("id", compound),
                "compound": compound,
                "section_kind": sec.get("kind"),
                "section_repeat_count": (
                    sec.get("repeat_count") if sec.get("kind") == "layer_template" else 1
                ),
            }

    # Include any per_node/*.py that the agent wrote even if not in graph.
    for py in sorted(per_node.glob("*.py")):
        compound = py.stem
        if compound not in compounds:
            sid, _, nid = compound.partition("__")
            compounds[compound] = {
                "section_id": sid or "layer",
                "node_id": nid or compound,
                "compound": compound,
                "section_kind": "layer_template",
                "section_repeat_count": 1,
            }

    for compound, meta in compounds.items():
        script = per_node / f"{compound}.py"
        row: Dict[str, Any] = {
            "compound": compound,
            "section_id": meta["section_id"],
            "node_id": meta["node_id"],
            "section_kind": meta["section_kind"],
            "section_repeat_count": meta["section_repeat_count"],
            "ok": False,
            "grid": [],
            "error": None,
            # Aliases kept for legacy clients — equal to prefill.tflops/gb.
            "tflops_picked": None,
            "gb_picked": None,
            # Authoritative phase-split values for the picked combo.
            "prefill": None,
            "decode": None,
        }
        try:
            grid = det.run_calc_on_grid(script)
            row["grid"] = grid
            row["ok"] = True
            # Pick a representative combo for quick display: B=1, S=512
            # (smallest non-trivial seq_len in the grid).
            for r in grid:
                if r["batch_size"] == 1 and r["seq_len"] == 512:
                    pre = r.get("prefill") or {}
                    dec = r.get("decode") or {}
                    row["prefill"] = {
                        "tflops": pre.get("tflops", 0.0),
                        "access_gb": pre.get("access_gb", 0.0),
                    }
                    row["decode"] = {
                        "tflops": dec.get("tflops", 0.0),
                        "access_gb": dec.get("access_gb", 0.0),
                    }
                    # Legacy aliases.
                    row["tflops_picked"] = pre.get("tflops", 0.0)
                    row["gb_picked"] = pre.get("access_gb", 0.0)
                    break
        except Exception as exc:  # noqa: BLE001
            row["error"] = f"{type(exc).__name__}: {exc}"
        results.append(row)
        store.append_timeline(
            "calc_value.s0.node.done",
            {"compound": compound,
             "section_id": meta["section_id"],
             "ok": row["ok"],
             "tflops_picked": row["tflops_picked"],
             "gb_picked": row["gb_picked"],
             "prefill": row["prefill"],
             "decode": row["decode"],
             "error": row["error"]},
        )
    return results


def run_step0_rough(
    *,
    req: Dict[str, Any],
    store,
    manager,
    paths: Dict[str, Path],
) -> Path:
    """Run Step 0: single-agent rough estimate.

    Returns the path to ``step0/rough_results.json``.
    """
    step0_dir = paths["step0_dir"]
    step0_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    store.append_timeline("calc_value.s0.start", {})

    try:
        workdir = _launch_rough_agent(manager=manager, step0_dir=step0_dir, req=req)
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        store.append_timeline(
            "calc_value.s0.failed",
            {"error": str(exc), "tb": tb},
        )
        # Write an empty rough_results.json so the UI can render the
        # "rough estimate unavailable" state and move on.
        empty = {
            "ok": False, "error": str(exc), "results": [], "graph": {"sections": []},
        }
        out = step0_dir / "rough_results.json"
        out.write_text(json.dumps(empty, indent=2, ensure_ascii=False), encoding="utf-8")
        return out

    graph = _load_rough_graph(workdir)
    (step0_dir / "rough_graph.json").write_text(
        json.dumps(graph, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    results = _run_per_node_grid(workdir, graph, store)

    elapsed = time.time() - t0
    out = {
        "ok": True,
        "elapsed_s": round(elapsed, 1),
        "graph": graph,
        "results": results,
        "summary": {
            "total_nodes": len(results),
            "ok_count": sum(1 for r in results if r["ok"]),
            "fail_count": sum(1 for r in results if not r["ok"]),
        },
    }
    out_path = step0_dir / "rough_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    store.append_timeline(
        "calc_value.s0.done",
        {"elapsed_s": round(elapsed, 1),
         "nodes": len(results),
         "ok": out["summary"]["ok_count"]},
    )
    return out_path
