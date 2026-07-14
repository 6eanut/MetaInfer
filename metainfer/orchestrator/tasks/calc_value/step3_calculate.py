"""Step 3: per-node FLOPs / mem-traffic calc with 2-angle agreement.

Algorithm (angle-serial, node-parallel, streaming):

1. For each round (up to MAX_ROUNDS_PER_NODE):
   - For each angle in [a, b] STRICTLY SERIALLY:
     - Launch ALL pending nodes IN PARALLEL (max_concurrent caps it).
     - As each agent finishes, run its calc.py once at the canonical
       shape, write the cell
       (calc.py / response.txt / result.json) + update _state.json +
       emit timeline. UI sees cells stream in.
   - After both angles done for this round: find nodes whose 2 angles
     disagree > REL_TOL. Those become the pending set for next round.
2. Converged nodes: pick angle a's script as canonical, write
   to final/<compound>.py + .meta.json.
3. Nodes still disputed after MAX_ROUNDS_PER_NODE: mark approximate,
   use the closer-to-median angle's script (same as before).

The WebUI re-runs final/<compound>.py on demand at arbitrary batch/seq
values via /calc/compute — we no longer precompute a 42-combo grid.

Output layout::

    step3/
    ├── cells/
    │   ├── _state.json                # UI-facing live state
    │   └── <compound>/
    │       ├── a/round_NN/{calc.py, response.txt, result.json}
    │       └── b/round_NN/...
    ├── final/
    │   ├── <compound>.py              # final winning calc.py
    │   └── <compound>.meta.json       # {approximate, source_agent, ...}
    └── _summary.json
"""

from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

from ...subagent_manager import AgentSpec
from . import deterministic as det
from . import prompts as P


MAX_ROUNDS_PER_NODE = 3
PER_AGENT_TIMEOUT_S = 900  # 15 min per writer
ANGLES = ("a", "b")  # maps to STEP3_WRITER_PROMPTS[0/1]


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #

def _safe_node_id(node_id: str) -> str:
    """Filesystem-safe node id (used for dir names)."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", node_id)[:80] or "node"


def _write_prompt(workdir: Path, name: str, text: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / f"{name}.prompt.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _extract_python_source(text: str) -> str:
    """Pull a Python code block out of an LLM response."""
    if not text:
        return ""
    m = re.search(r"```python\s*\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1)
    for m in re.finditer(r"```\s*\n(.*?)\n```", text, re.DOTALL):
        if "def calc" in m.group(1):
            return m.group(1)
    if "def calc" in text:
        return text.strip()
    return ""


# --------------------------------------------------------------------------- #
# Cell-state writer (atomic-ish, locked)
# --------------------------------------------------------------------------- #

class CellStateStore:
    """Thread-safe writer for step3/cells/_state.json.

    Holds the live grid of (compound × angle) cells that the WebUI reads
    to render the streaming audit table.
    """

    def __init__(self, state_path: Path):
        self.path = state_path
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self._doc = json.loads(self.path.read_text(encoding="utf-8"))
            except ValueError:
                self._doc = self._fresh_doc()
        else:
            self._doc = self._fresh_doc()

    @staticmethod
    def _fresh_doc() -> Dict[str, Any]:
        return {"round": 0, "updated_at": time.time(), "nodes": {}}

    def init_node(self, compound: str, *, node_id: str, section_id: str,
                  section_kind: Optional[str], section_repeat_count: int) -> None:
        with self._lock:
            if compound not in self._doc["nodes"]:
                self._doc["nodes"][compound] = {
                    "node_id": node_id,
                    "section_id": section_id,
                    "section_kind": section_kind,
                    "section_repeat_count": section_repeat_count,
                    "cells": {
                        a: {
                            # Legacy aliases — equal to prefill.tflops / prefill.access_gb.
                            "tflops": None, "gb": None,
                            # Authoritative phase-split values.
                            "prefill": None, "decode": None,
                            "round": None, "status": "pending",
                            "elapsed_s": None, "error": None,
                            "script_path": None,
                        }
                        for a in ANGLES
                    },
                    "converged": None,
                    "spread_pct": None,
                    "round": 0,
                }
                self._flush_locked()

    def update_cell(self, compound: str, angle: str, *,
                    prefill: Optional[Dict[str, float]],
                    decode: Optional[Dict[str, float]],
                    round_idx: int, status: str,
                    elapsed_s: Optional[float], error: Optional[str] = None,
                    script_path: Optional[str] = None) -> None:
        with self._lock:
            node = self._doc["nodes"].setdefault(compound, {
                "node_id": compound, "section_id": "", "cells": {a: {} for a in ANGLES},
            })
            pre = prefill or {"tflops": 0.0, "access_gb": 0.0}
            dec = decode  or {"tflops": 0.0, "access_gb": 0.0}
            node["cells"][angle] = {
                # Legacy aliases (prefill-derived).
                "tflops": pre.get("tflops"),
                "gb": pre.get("access_gb"),
                # Authoritative.
                "prefill": pre,
                "decode": dec,
                "round": round_idx,
                "status": status,
                "elapsed_s": elapsed_s,
                "error": error,
                "script_path": script_path,
            }
            # Recompute convergence if both angles have values.
            self._recompute_locked(compound)
            self._flush_locked()

    def mark_round(self, round_idx: int) -> None:
        with self._lock:
            self._doc["round"] = round_idx
            self._flush_locked()

    def _recompute_locked(self, compound: str) -> None:
        node = self._doc["nodes"].get(compound)
        if not node:
            return
        cells = node["cells"]
        # Convergence is based on prefill.tflops — same convention as the
        # 5% relative tolerance was originally calibrated for.
        vals = [cells[a] for a in ANGLES
                if isinstance(cells.get(a), dict)
                and cells[a].get("tflops") is not None]
        if len(vals) < len(ANGLES):
            node["converged"] = None
            node["spread_pct"] = None
            return
        tflops = [v["tflops"] for v in vals]
        if max(tflops) <= 0:
            spread_pct = 0.0
        else:
            spread_pct = (max(tflops) - min(tflops)) / max(abs(x) for x in tflops)
        node["spread_pct"] = round(spread_pct, 6)
        node["converged"] = spread_pct <= det.REL_TOL

    def _flush_locked(self) -> None:
        self._doc["updated_at"] = time.time()
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self._doc, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        tmp.replace(self.path)


# --------------------------------------------------------------------------- #
# Angle-stage: run one angle for all pending nodes in parallel
# --------------------------------------------------------------------------- #

def _run_angle_stage(
    *,
    manager,
    cells_root: Path,
    angle: str,
    angle_idx: int,
    pending: List[Dict[str, Any]],
    round_idx: int,
    memory_json: str,
    req: Dict[str, Any],
    prev_scripts_by_node: Dict[str, Optional[str]],
    mismatches_by_node: Dict[str, List[Dict[str, Any]]],
    cell_state: CellStateStore,
    store,
) -> None:
    """Launch all pending nodes for one angle in parallel.

    Blocks until all of this angle's cells finish. Each cell is written
    + timeline-emitted as its agent completes (polled every 1s).
    """
    # Launch all specs up front. manager.max_concurrent gates actual
    # parallelism (the launch_async thread waits on the semaphore
    # internally before invoking the subprocess).
    inflight: Dict[str, Tuple[Dict[str, Any], object, float]] = {}
    for node_meta in pending:
        compound = node_meta["compound"]
        prev = prev_scripts_by_node.get(compound)
        mm = mismatches_by_node.get(compound, [])

        # We can't run launch_async synchronously here because we need
        # the spec.name for later result lookup. Build spec inline.
        cell_dir = cells_root / compound / angle / f"round_{round_idx:02d}"
        cell_dir.mkdir(parents=True, exist_ok=True)
        workdir = cell_dir / "writer"
        log_dir = cell_dir / "logs"
        node_json = json.dumps(node_meta["node"], indent=2, ensure_ascii=False)
        readonly = P.READONLY_WARNING.format(
            model_dir=req["model_dir"],
            framework_dir=req["framework_source_dir"],
        )
        common = {
            "node_json": node_json,
            "memory_json": memory_json,
            "readonly": readonly,
            "calc_contract": P.STEP3_CALC_FUNC_CONTRACT,
        }
        if round_idx == 0 or prev is None:
            text = P.STEP3_WRITER_PROMPTS[angle_idx].format(**common)
        else:
            text = P.STEP3_FIX_PROMPT.format(
                **common,
                your_script=prev,
                mismatches=det.format_mismatches_for_prompt(mm),
            )
        prompt_file = _write_prompt(workdir, "writer", text)
        spec_name = f"s3_{compound}_{angle}_r{round_idx}"
        spec = AgentSpec(
            name=spec_name,
            role="calc_writer",
            prompt_file=prompt_file,
            workdir=workdir,
            log_dir=log_dir,
            timeout_s=PER_AGENT_TIMEOUT_S,
            stuck_timeout_s=300,
            max_retries=2,
        )
        t0 = time.time()
        thread = manager.launch_async(spec)
        inflight[spec.name] = (node_meta, thread, t0)

    # Poll until all done. As each finishes, process its cell.
    while inflight:
        done_names = [n for (n, (_, t, _)) in inflight.items() if not t.is_alive()]
        if not done_names:
            time.sleep(1.0)
            continue
        for name in done_names:
            node_meta, _thread, t0 = inflight.pop(name)
            compound = node_meta["compound"]
            elapsed = time.time() - t0
            cell_dir = cells_root / compound / angle / f"round_{round_idx:02d}"
            workdir = cell_dir / "writer"

            result = manager.result(name)
            status = "failed"
            calc_path_str: Optional[str] = None
            grid: Optional[List[Dict[str, Any]]] = None
            err: Optional[str] = None
            prefill_picked: Optional[Dict[str, float]] = None
            decode_picked: Optional[Dict[str, float]] = None

            if result is None or not result.success:
                err = result.error if result else "no result"
                (workdir / "error.txt").write_text(str(err), encoding="utf-8")
            else:
                text_out = result.final_text or ""
                (workdir / "response.txt").write_text(text_out, encoding="utf-8")
                src, _src_kind = det.load_agent_text_file(
                    workdir, "calc.py", text_out, _extract_python_source,
                )
                if not src:
                    (workdir / "parse_error.txt").write_text(
                        "No `def calc` block found.\n"
                        f"Response first 500 chars:\n{text_out[:500]}",
                        encoding="utf-8",
                    )
                    status = "no_source"
                    err = "no def calc in response"
                else:
                    calc_path = workdir / "calc.py"
                    if not calc_path.exists():
                        calc_path.write_text(src, encoding="utf-8")
                    try:
                        result_rec = det.run_calc_canonical(calc_path)
                        status = "ok"
                        calc_path_str = str(calc_path)
                        prefill_picked = {
                            "tflops": (result_rec.get("prefill") or {}).get("tflops", 0.0),
                            "access_gb": (result_rec.get("prefill") or {}).get("access_gb", 0.0),
                        }
                        decode_picked = {
                            "tflops": (result_rec.get("decode") or {}).get("tflops", 0.0),
                            "access_gb": (result_rec.get("decode") or {}).get("access_gb", 0.0),
                        }
                    except Exception as exc:  # noqa: BLE001
                        (workdir / "runtime_error.txt").write_text(
                            f"{type(exc).__name__}: {exc}", encoding="utf-8",
                        )
                        status = "runtime_error"
                        err = str(exc)

            if prefill_picked is not None:
                # Single-shape result (canonical B=1, S=512). The WebUI
                # re-runs calc.py at other shapes on demand via /calc/cells
                # or /calc/compute.
                result_rec = {
                    "batch_size": det.CANONICAL_BATCH,
                    "seq_len":    det.CANONICAL_SEQ,
                    "prefill": prefill_picked,
                    "decode":  decode_picked or {"tflops": 0.0, "access_gb": 0.0},
                }
                (cell_dir / "result.json").write_text(
                    json.dumps(result_rec, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

            cell_state.update_cell(
                compound, angle,
                prefill=prefill_picked, decode=decode_picked,
                round_idx=round_idx, status=status,
                elapsed_s=round(elapsed, 1),
                error=err, script_path=calc_path_str,
            )
            store.append_timeline(
                "calc_value.s3.cell.done",
                {"node": node_meta["node_id"],
                 "section_id": node_meta["section_id"],
                 "compound": compound,
                 "angle": angle, "round": round_idx,
                 "status": status,
                 "tflops": (prefill_picked or {}).get("tflops"),
                 "gb": (prefill_picked or {}).get("access_gb"),
                 "prefill": prefill_picked,
                 "decode": decode_picked,
                 "elapsed_s": round(elapsed, 1),
                 "error": err},
            )


# --------------------------------------------------------------------------- #
# Dispute detection
# --------------------------------------------------------------------------- #

def _find_disputed(
    *,
    pending: List[Dict[str, Any]],
    cells_root: Path,
    round_idx: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Optional[str]]]:
    """For each pending node, read back both angles' single-shape results
    from this round and check agreement. Returns:

      bad_pending           — nodes that still disagree > REL_TOL
      mismatches_by_node    — for each disputed node, the mismatch list
      prev_scripts_by_node  — each node's most-recent (any-angle) script path
    """
    bad: List[Dict[str, Any]] = []
    mismatches_by_node: Dict[str, List[Dict[str, Any]]] = {}
    prev_scripts_by_node: Dict[str, Optional[str]] = {}

    for node_meta in pending:
        compound = node_meta["compound"]
        results: List[Dict[str, Any]] = []
        script_for_node: Optional[str] = None
        ok = True
        for angle in ANGLES:
            cell_dir = cells_root / compound / angle / f"round_{round_idx:02d}"
            result_path = cell_dir / "result.json"
            calc_path = cell_dir / "writer" / "calc.py"
            if not result_path.exists():
                ok = False
                break
            try:
                r = json.loads(result_path.read_text(encoding="utf-8"))
            except ValueError:
                ok = False
                break
            results.append(r)
            if calc_path.exists() and script_for_node is None:
                script_for_node = calc_path.read_text(encoding="utf-8")
        prev_scripts_by_node[compound] = script_for_node
        if not ok or len(results) < len(ANGLES):
            bad.append(node_meta)
            mismatches_by_node[compound] = []
            continue
        comparison = det.compare_calc_results(results)
        if not comparison["ok"]:
            bad.append(node_meta)
            mismatches_by_node[compound] = comparison["mismatches"]
    return bad, mismatches_by_node, prev_scripts_by_node


# --------------------------------------------------------------------------- #
# Finalize: pick canonical script per node + write step3/final/
# --------------------------------------------------------------------------- #

def _finalize_node(
    *,
    compound: str,
    node_meta: Dict[str, Any],
    cells_root: Path,
    final_dir: Path,
    last_round: int,
    cell_state: CellStateStore,
) -> Tuple[Path, Dict[str, Any]]:
    """Pick the canonical script for a node and emit final/<compound>.py.

    Strategy: read both angles' scripts from the last completed round.
    If cell_state says converged → take writer_0 (arbitrary, all agree).
    Else → take the writer whose single-shape result is closest to the
    median (same idea as the legacy median fallback).
    """
    # Find the latest round where both angles produced a result.
    chosen_round = -1
    results_by_angle: Dict[str, Dict[str, Any]] = {}
    calc_by_angle: Dict[str, str] = {}
    for r in range(last_round, -1, -1):
        ok = True
        trial: Dict[str, Dict[str, Any]] = {}
        trial_calc: Dict[str, str] = {}
        for angle in ANGLES:
            result_path = cells_root / compound / angle / f"round_{r:02d}" / "result.json"
            calc_path = cells_root / compound / angle / f"round_{r:02d}" / "writer" / "calc.py"
            if not result_path.exists():
                ok = False
                break
            try:
                trial[angle] = json.loads(result_path.read_text(encoding="utf-8"))
            except ValueError:
                ok = False
                break
            if calc_path.exists():
                trial_calc[angle] = calc_path.read_text(encoding="utf-8")
        if ok and len(trial) == len(ANGLES):
            chosen_round = r
            results_by_angle = trial
            calc_by_angle = trial_calc
            break

    if chosen_round < 0:
        # No usable cells at all — degenerate fallback.
        script_text = ("def calc(batch_size, seq_len):\n"
                       "    return {'prefill': {'tflops': 0.0, 'access_gb': 0.0}, 'decode': {'tflops': 0.0, 'access_gb': 0.0}}\n")
        (final_dir / f"{compound}.py").write_text(script_text, encoding="utf-8")
        meta = {
            "node_id": node_meta["node_id"],
            "section_id": node_meta["section_id"],
            "compound_id": compound,
            "approximate": True,
            "source_agent": "degenerate_fallback",
            "rounds": last_round + 1,
        }
        (final_dir / f"{compound}.meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        return final_dir / f"{compound}.py", {"approximate": True,
                                               "rounds": last_round + 1}

    # Decide: converged or median fallback.
    comparison = det.compare_calc_results(list(results_by_angle.values()))
    node_state = cell_state._doc["nodes"].get(compound, {})
    converged = node_state.get("converged", False)

    if converged or comparison["ok"]:
        # All agree — pick angle a arbitrarily.
        chosen_angle = ANGLES[0]
        approximate = False
        source_agent = "unanimous"
    else:
        # Median fallback: pick the angle whose result is closest to the
        # per-phase median.
        median_rec = det.median_result(list(results_by_angle.values()))
        chosen_angle = _pick_most_median_angle(results_by_angle, median_rec)
        approximate = True
        source_agent = f"median_fallback_from_angle_{chosen_angle}"

    script_text = calc_by_angle.get(chosen_angle, "")
    if not script_text:
        # Edge case: angle had result.json but calc.py missing — fall back.
        script_text = ("def calc(batch_size, seq_len):\n"
                       "    return {'prefill': {'tflops': 0.0, 'access_gb': 0.0}, 'decode': {'tflops': 0.0, 'access_gb': 0.0}}\n")
        approximate = True
        source_agent = "degenerate_fallback_missing_script"

    (final_dir / f"{compound}.py").write_text(script_text, encoding="utf-8")
    meta = {
        "node_id": node_meta["node_id"],
        "section_id": node_meta["section_id"],
        "section_kind": node_meta.get("section_kind"),
        "section_repeat_count": node_meta.get("section_repeat_count", 1),
        "section_applies_to": node_meta.get("section_applies_to"),
        "compound_id": compound,
        "approximate": approximate,
        "source_agent": source_agent,
        "rounds": chosen_round + 1,
        "chosen_angle": chosen_angle,
        "writer_scripts": [
            str(cells_root / compound / a / f"round_{chosen_round:02d}" / "writer" / "calc.py")
            for a in ANGLES
        ],
        "mismatch_count_at_cap": len(comparison.get("mismatches") or []),
    }
    (final_dir / f"{compound}.meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return final_dir / f"{compound}.py", {
        "approximate": approximate, "rounds": chosen_round + 1,
    }


def _pick_most_median_angle(
    results_by_angle: Dict[str, Dict[str, Any]],
    median_rec: Dict[str, Any],
) -> str:
    if not results_by_angle:
        return ANGLES[0]

    def _pick(rec: Dict[str, Any], phase: str, field: str) -> float:
        """Extract a numeric field from a single-shape record."""
        if phase in rec:
            return (rec.get(phase) or {}).get(field, 0.0)
        # Legacy — only prefill.tflops / prefill.access_gb exist.
        if phase == "prefill":
            return rec.get(field, 0.0)
        return 0.0

    best_angle = ANGLES[0]
    best_err = float("inf")
    for angle, rec in results_by_angle.items():
        err = 0.0
        for phase in ("prefill", "decode"):
            err += abs(_pick(rec, phase, "tflops") - _pick(median_rec, phase, "tflops"))
            err += abs(_pick(rec, phase, "access_gb") - _pick(median_rec, phase, "access_gb"))
        if err < best_err:
            best_err = err
            best_angle = angle
    return best_angle


# --------------------------------------------------------------------------- #
# Top-level entry point (signature preserved for pipeline.py)
# --------------------------------------------------------------------------- #

def run_step3_calculate(
    *,
    req: Dict[str, Any],
    store,
    manager,
    paths: Dict[str, Path],
    graph_path: Path,
) -> Path:
    """Run the angle-serial / node-parallel / cell-streaming audit.

    Returns the final/ directory path.
    """
    raw = json.loads(graph_path.read_text(encoding="utf-8"))
    graph = det.normalize_graph(raw)
    memory_path = paths["step1_dir"] / "memory.json"
    memory_json = memory_path.read_text(encoding="utf-8") \
        if memory_path.exists() else "{}"

    step3_dir = paths["step3_dir"]
    cells_root = step3_dir / "cells"
    cells_root.mkdir(parents=True, exist_ok=True)
    final_dir = step3_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    cell_state = CellStateStore(cells_root / "_state.json")

    # Flatten (section, node) targets.
    targets: List[Dict[str, Any]] = []
    for sec in graph.get("sections") or []:
        sid = sec.get("id", "section")
        sec_kind = sec.get("kind")
        sec_rc = sec.get("repeat_count") if sec_kind == "layer_template" else 1
        for node in (sec.get("graph") or {}).get("nodes") or []:
            if not isinstance(node, dict):
                continue
            nid_raw = node.get("id", "node")
            compound_raw = f"{sid}__{nid_raw}"
            compound = _safe_node_id(compound_raw)
            targets.append({
                "node": node,
                "node_id": nid_raw,
                "section_id": sid,
                "section_kind": sec_kind,
                "section_repeat_count": sec_rc if isinstance(sec_rc, int) and sec_rc >= 1 else 1,
                "section_applies_to": sec.get("applies_to"),
                "compound": compound,
            })
            cell_state.init_node(
                compound, node_id=nid_raw, section_id=sid,
                section_kind=sec_kind,
                section_repeat_count=sec_rc if isinstance(sec_rc, int) and sec_rc >= 1 else 1,
            )

    total = len(targets)
    store.append_timeline(
        "calc_value.s3.start",
        {"nodes": total,
         "aggregated_nodes": det.aggregated_node_count(graph),
         "sections": len(graph.get("sections") or [])},
    )

    pending = list(targets)
    round_idx = 0
    last_round_completed = -1
    while pending and round_idx < MAX_ROUNDS_PER_NODE:
        cell_state.mark_round(round_idx)
        store.append_timeline(
            "calc_value.s3.round.start",
            {"round": round_idx, "pending": len(pending)},
        )
        # Build prev_scripts / mismatches for round >= 1 from previous round's cells.
        if round_idx == 0:
            prev_scripts: Dict[str, Optional[str]] = {m["compound"]: None for m in pending}
            mismatches: Dict[str, List[Dict[str, Any]]] = {m["compound"]: [] for m in pending}
        # else: prev_scripts / mismatches filled in by _find_disputed at the end of last iter

        for angle_idx, angle in enumerate(ANGLES):
            store.append_timeline(
                "calc_value.s3.angle.start",
                {"round": round_idx, "angle": angle, "pending": len(pending)},
            )
            _run_angle_stage(
                manager=manager, cells_root=cells_root,
                angle=angle, angle_idx=angle_idx,
                pending=pending, round_idx=round_idx,
                memory_json=memory_json, req=req,
                prev_scripts_by_node=prev_scripts,
                mismatches_by_node=mismatches,
                cell_state=cell_state, store=store,
            )
            store.append_timeline(
                "calc_value.s3.angle.done",
                {"round": round_idx, "angle": angle},
            )

        last_round_completed = round_idx
        # Check disputes.
        bad, mismatches, prev_scripts = _find_disputed(
            pending=pending, cells_root=cells_root, round_idx=round_idx,
        )
        store.append_timeline(
            "calc_value.s3.round.done",
            {"round": round_idx, "disputed": len(bad), "converged": len(pending) - len(bad)},
        )
        if not bad:
            break
        pending = bad
        round_idx += 1

    # Finalize every node.
    summary = []
    for node_meta in targets:
        compound = node_meta["compound"]
        try:
            _script_path, meta = _finalize_node(
                compound=compound, node_meta=node_meta,
                cells_root=cells_root, final_dir=final_dir,
                last_round=max(last_round_completed, 0),
                cell_state=cell_state,
            )
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"[calc-value.S3] finalize {compound} crashed: {exc}\n{tb}",
                  flush=True)
            store.append_timeline(
                "calc_value.s3.node.crashed",
                {"node": node_meta["node_id"], "section_id": node_meta["section_id"],
                 "error": str(exc)},
            )
            (final_dir / f"{compound}.py").write_text(
                "def calc(batch_size, seq_len):\n"
                "    return {'prefill': {'tflops': 0.0, 'access_gb': 0.0}, 'decode': {'tflops': 0.0, 'access_gb': 0.0}}\n",
                encoding="utf-8",
            )
            (final_dir / f"{compound}.meta.json").write_text(
                json.dumps({"node_id": node_meta["node_id"],
                            "section_id": node_meta["section_id"],
                            "compound_id": compound,
                            "approximate": True,
                            "source_agent": "crash_fallback",
                            "error": str(exc)}, indent=2),
                encoding="utf-8",
            )
            meta = {"approximate": True, "crashed": True, "rounds": 0}
        summary.append({
            "node_id": node_meta["node_id"],
            "section_id": node_meta["section_id"],
            "compound_id": compound,
            "section_kind": node_meta.get("section_kind"),
            "repeat_count": node_meta.get("section_repeat_count", 1),
            "approximate": meta.get("approximate"),
            "rounds": meta.get("rounds"),
        })
        store.append_timeline(
            "calc_value.s3.node.done",
            {"node": node_meta["node_id"], "section_id": node_meta["section_id"],
             "compound": compound,
             "approximate": meta.get("approximate"),
             "rounds": meta.get("rounds")},
        )

    (final_dir / "_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    store.append_timeline(
        "calc_value.s3.all_nodes_done",
        {"total": len(summary),
         "approximate_count": sum(1 for s in summary if s["approximate"])},
    )
    return final_dir
