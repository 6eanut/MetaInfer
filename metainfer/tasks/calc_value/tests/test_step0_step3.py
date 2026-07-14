"""Tests for calc_value step0_rough and step3_calculate with Mock agents.

These tests cover the per-node canonical-shape aggregation, the
CellStateStore convergence math, dispute detection, and finalize
(median-fallback). No real `claude` / `ccb` subprocess is ever spawned —
the MockAgentManager returns canned responses that include calc.py
source, which step3 writes to disk + runs through the deterministic
helpers.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest

from metainfer.tasks.calc_value.orchestrator import (
    deterministic as det,
)
from metainfer.tasks.calc_value.orchestrator import (
    step0_rough as s0,
)
from metainfer.tasks.calc_value.orchestrator import (
    step3_calculate as s3,
)
from metainfer.testing import FakeStore, MockAgentManager, write_calc_script


_CALC_TEMPLATE = """\
def calc(batch_size, seq_len):
    return {{
        "prefill": {{"tflops": {pre_t}, "access_gb": {pre_g}}},
        "decode":  {{"tflops": {dec_t}, "access_gb": {dec_g}}},
    }}
"""


def _calc_src(pre_t, pre_g, dec_t=0.0, dec_g=0.0) -> str:
    """A canned calc.py source the mock agent 'writes'."""
    return _CALC_TEMPLATE.format(pre_t=pre_t, pre_g=pre_g,
                                  dec_t=dec_t, dec_g=dec_g)


def _fenced(src: str) -> str:
    """Wrap a python block the way agents do — ```python ... ```."""
    return f"Here you go:\n```python\n{src}\n```\nDone."


# --------------------------------------------------------------------------- #
# step0_rough
# --------------------------------------------------------------------------- #

def test_load_rough_graph_reads_manifest(tmp_path):
    workdir = tmp_path / "agent_rough"
    workdir.mkdir()
    manifest = {
        "sections": [{
            "id": "layer", "kind": "layer_template", "repeat_count": 2,
            "graph": {"nodes": [{"id": "attn", "op": "attention"}]},
        }],
    }
    (workdir / "rough_graph.json").write_text(json.dumps(manifest))
    out = s0._load_rough_graph(workdir)
    assert out["sections"][0]["id"] == "layer"
    assert "fallback" not in out  # real manifest doesn't set fallback


def test_load_rough_graph_falls_back_to_per_node_filenames(tmp_path):
    """If the agent forgets rough_graph.json we synthesize a sectioned
    graph from per_node/*.py filenames."""
    workdir = tmp_path / "agent_rough"
    per_node = workdir / "per_node"
    per_node.mkdir(parents=True)
    # compound = "<section>__<node>.py"
    (per_node / "input__embedding.py").write_text("# placeholder")
    (per_node / "layer__attn.py").write_text("# placeholder")
    out = s0._load_rough_graph(workdir)
    assert out.get("fallback") is True
    section_ids = [s["id"] for s in out["sections"]]
    assert "input" in section_ids and "layer" in section_ids


def test_run_per_node_grid_canonical_shape(tmp_path):
    """The user's 42-combo-removal requirement: each per_node script runs
    ONCE at (B=1, S=512) and the row gets prefill/decode sub-dicts."""
    workdir = tmp_path / "agent_rough"
    per_node = workdir / "per_node"
    per_node.mkdir(parents=True)
    write_calc_script(per_node / "layer__attn.py",
                      prefill_tflops=100.0, prefill_gb=20.0,
                      decode_tflops=5.0, decode_gb=50.0)
    write_calc_script(per_node / "layer__mlp.py",
                      prefill_tflops=200.0, prefill_gb=40.0)
    graph = {"sections": [{
        "id": "layer", "kind": "layer_template", "repeat_count": 1,
        "graph": {"nodes": [
            {"id": "attn", "op": "attention", "compound": "layer__attn"},
            {"id": "mlp",  "op": "matmul",    "compound": "layer__mlp"},
        ]},
    }]}
    store = FakeStore()
    rows = s0._run_per_node_grid(workdir, graph, store)
    by_compound = {r["compound"]: r for r in rows}
    attn = by_compound["layer__attn"]
    assert attn["ok"] is True
    assert attn["prefill"] == {"tflops": 100.0, "access_gb": 20.0}
    assert attn["decode"] == {"tflops": 5.0, "access_gb": 50.0}
    # Legacy aliases stay in sync with prefill (per the contract).
    assert attn["tflops_picked"] == 100.0
    assert attn["gb_picked"] == 20.0
    # Timeline events fire per node.
    assert "calc_value.s0.node.done" in store.types()


def test_run_per_node_grid_records_runtime_error(tmp_path):
    workdir = tmp_path / "agent_rough"
    per_node = workdir / "per_node"
    per_node.mkdir(parents=True)
    # A calc.py that divides by zero.
    (per_node / "layer__bad.py").write_text(
        "def calc(batch_size, seq_len):\n"
        "    return {'prefill': {'tflops': 1/0, 'access_gb': 1.0}, 'decode': {'tflops': 0.0, 'access_gb': 0.0}}\n",
        encoding="utf-8",
    )
    graph = {"sections": [{
        "id": "layer", "kind": "layer_template", "repeat_count": 1,
        "graph": {"nodes": [{"id": "bad", "op": "x", "compound": "layer__bad"}]},
    }]}
    store = FakeStore()
    rows = s0._run_per_node_grid(workdir, graph, store)
    r = rows[0]
    assert r["ok"] is False
    assert r["error"] and "ZeroDivision" in r["error"]


def test_run_step0_rough_full_with_mock_agent(tmp_path):
    """End-to-end S0 with a MockAgentManager that writes two per_node
    scripts + a rough_graph.json manifest."""
    # Pre-create the artifacts the mock agent would have written.
    workdir_response = _fenced("# nothing — we materialize files below")

    # The mock agent's job: when launched, write per_node/<compound>.py
    # files + a rough_graph.json into its workdir, then return a final
    # text. We override launch_async's work via a side-effect hook.
    def response_fn(spec: Any) -> str:
        wd = Path(spec.workdir)
        per_node = wd / "per_node"
        per_node.mkdir(parents=True, exist_ok=True)
        write_calc_script(per_node / "layer__attn.py",
                          prefill_tflops=42.0, prefill_gb=9.0,
                          decode_tflops=2.0, decode_gb=18.0)
        (wd / "rough_graph.json").write_text(json.dumps({
            "sections": [{
                "id": "layer", "kind": "layer_template", "repeat_count": 1,
                "graph": {"nodes": [{"id": "attn", "op": "attention",
                                       "compound": "layer__attn"}]},
            }],
        }))
        return "done."

    manager = MockAgentManager(response_fn=response_fn)
    store = FakeStore()
    req = {
        "model_dir": "/tmp/x",
        "framework_source_dir": "/tmp/y",
        "cmdline_args": "",
        "env_vars": "",
    }
    paths = {"step0_dir": tmp_path / "step0"}
    out_path = s0.run_step0_rough(req=req, store=store, manager=manager, paths=paths)

    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["summary"]["ok_count"] == 1
    assert data["summary"]["total_nodes"] == 1
    r = data["results"][0]
    assert r["prefill"] == {"tflops": 42.0, "access_gb": 9.0}
    assert r["decode"] == {"tflops": 2.0, "access_gb": 18.0}
    assert "calc_value.s0.start" in store.types()
    assert "calc_value.s0.done" in store.types()


def test_run_step0_rough_handles_agent_failure(tmp_path):
    """When the rough agent fails, S0 should write an empty
    rough_results.json so the UI can render 'unavailable' and move on."""
    manager = MockAgentManager(failures={"calc_value_s0_rough": "boom"})
    store = FakeStore()
    req = {"model_dir": "/x", "framework_source_dir": "/y",
           "cmdline_args": "", "env_vars": ""}
    paths = {"step0_dir": tmp_path / "step0"}
    out_path = s0.run_step0_rough(req=req, store=store, manager=manager, paths=paths)
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert data["ok"] is False
    assert "boom" in data["error"]
    assert data["results"] == []
    assert "calc_value.s0.failed" in store.types()


# --------------------------------------------------------------------------- #
# step3_calculate — CellStateStore
# --------------------------------------------------------------------------- #

def test_cell_state_init_creates_node(tmp_path):
    cs = s3.CellStateStore(tmp_path / "_state.json")
    cs.init_node("layer__attn", node_id="attn", section_id="layer",
                 section_kind="layer_template", section_repeat_count=2)
    node = cs._doc["nodes"]["layer__attn"]
    assert node["section_id"] == "layer"
    assert node["section_repeat_count"] == 2
    assert set(node["cells"].keys()) == {"a", "b"}
    assert node["cells"]["a"]["status"] == "pending"


def test_cell_state_update_records_phases(tmp_path):
    cs = s3.CellStateStore(tmp_path / "_state.json")
    cs.init_node("n1", node_id="attn", section_id="layer",
                 section_kind="layer_template", section_repeat_count=1)
    cs.update_cell("n1", "a",
                   prefill={"tflops": 100.0, "access_gb": 10.0},
                   decode={"tflops": 1.0, "access_gb": 20.0},
                   round_idx=0, status="ok", elapsed_s=2.5)
    cell = cs._doc["nodes"]["n1"]["cells"]["a"]
    assert cell["prefill"] == {"tflops": 100.0, "access_gb": 10.0}
    assert cell["decode"] == {"tflops": 1.0, "access_gb": 20.0}
    # Legacy aliases.
    assert cell["tflops"] == 100.0
    assert cell["gb"] == 10.0
    assert cell["status"] == "ok"
    assert cell["round"] == 0


def test_cell_state_converges_when_both_angles_agree(tmp_path):
    """Convergence is judged on prefill.tflops within REL_TOL."""
    cs = s3.CellStateStore(tmp_path / "_state.json")
    cs.init_node("n1", node_id="attn", section_id="layer",
                 section_kind="layer_template", section_repeat_count=1)
    cs.update_cell("n1", "a", prefill={"tflops": 100.0, "access_gb": 10.0},
                   decode={"tflops": 0.0, "access_gb": 0.0},
                   round_idx=0, status="ok", elapsed_s=1.0)
    # After only one angle: no verdict yet.
    node = cs._doc["nodes"]["n1"]
    assert node["converged"] is None
    cs.update_cell("n1", "b", prefill={"tflops": 101.0, "access_gb": 10.5},
                   decode={"tflops": 0.0, "access_gb": 0.0},
                   round_idx=0, status="ok", elapsed_s=1.0)
    node = cs._doc["nodes"]["n1"]
    assert node["converged"] is True
    assert node["spread_pct"] is not None and node["spread_pct"] <= det.REL_TOL


def test_cell_state_disputed_when_angles_disagree(tmp_path):
    cs = s3.CellStateStore(tmp_path / "_state.json")
    cs.init_node("n1", node_id="attn", section_id="layer",
                 section_kind="layer_template", section_repeat_count=1)
    cs.update_cell("n1", "a", prefill={"tflops": 100.0, "access_gb": 10.0},
                   decode={"tflops": 0.0, "access_gb": 0.0},
                   round_idx=0, status="ok", elapsed_s=1.0)
    cs.update_cell("n1", "b", prefill={"tflops": 500.0, "access_gb": 10.0},
                   decode={"tflops": 0.0, "access_gb": 0.0},
                   round_idx=0, status="ok", elapsed_s=1.0)
    node = cs._doc["nodes"]["n1"]
    assert node["converged"] is False


def test_cell_state_persists_across_instances(tmp_path):
    """The file is the source of truth — re-opening should see the same state."""
    p = tmp_path / "_state.json"
    cs = s3.CellStateStore(p)
    cs.init_node("n1", node_id="attn", section_id="layer",
                 section_kind="layer_template", section_repeat_count=1)
    cs2 = s3.CellStateStore(p)
    assert "n1" in cs2._doc["nodes"]


# --------------------------------------------------------------------------- #
# step3_calculate — _pick_most_median_angle
# --------------------------------------------------------------------------- #

def test_pick_most_median_angle_picks_closer_one(tmp_path):
    results = {
        "a": {"prefill": {"tflops": 10.0, "access_gb": 1.0},
              "decode":  {"tflops": 0.0, "access_gb": 0.0}},
        "b": {"prefill": {"tflops": 30.0, "access_gb": 3.0},
              "decode":  {"tflops": 0.0, "access_gb": 0.0}},
    }
    median = {"prefill": {"tflops": 11.0, "access_gb": 1.0},
              "decode":  {"tflops": 0.0, "access_gb": 0.0}}
    # 'a' is much closer to the median than 'b'.
    assert s3._pick_most_median_angle(results, median) == "a"


def test_pick_most_median_angle_empty_returns_first(tmp_path):
    assert s3._pick_most_median_angle({}, {}) == s3.ANGLES[0]


# --------------------------------------------------------------------------- #
# step3_calculate — _find_disputed + _finalize_node
# --------------------------------------------------------------------------- #

def _setup_round(cells_root: Path, compound: str, angle: str, round_idx: int,
                 pre_t: float, pre_g: float, calc_src: str = "") -> None:
    """Materialize a fake completed cell with result.json + calc.py."""
    cell_dir = cells_root / compound / angle / f"round_{round_idx:02d}"
    writer = cell_dir / "writer"
    writer.mkdir(parents=True, exist_ok=True)
    (writer / "calc.py").write_text(calc_src or f"# calc for {compound}/{angle}\n",
                                     encoding="utf-8")
    (cell_dir / "result.json").write_text(json.dumps({
        "batch_size": 1, "seq_len": 512,
        "prefill": {"tflops": pre_t, "access_gb": pre_g},
        "decode":  {"tflops": 0.0, "access_gb": 0.0},
    }), encoding="utf-8")


def test_find_disputed_flags_disagreement(tmp_path):
    cells_root = tmp_path / "cells"
    pending = [{"compound": "n1"}]
    # Two angles disagree heavily on prefill.tflops.
    _setup_round(cells_root, "n1", "a", 0, 100.0, 10.0)
    _setup_round(cells_root, "n1", "b", 0, 500.0, 10.0)
    bad, mismatches, prev = s3._find_disputed(
        pending=pending, cells_root=cells_root, round_idx=0,
    )
    assert len(bad) == 1
    assert "n1" in mismatches and mismatches["n1"]
    # prev_scripts tracks both compounds.
    assert "n1" in prev


def test_find_disputed_marks_converged_clean(tmp_path):
    cells_root = tmp_path / "cells"
    pending = [{"compound": "n1"}]
    _setup_round(cells_root, "n1", "a", 0, 100.0, 10.0)
    _setup_round(cells_root, "n1", "b", 0, 101.0, 10.0)  # within tolerance
    bad, mismatches, _ = s3._find_disputed(
        pending=pending, cells_root=cells_root, round_idx=0,
    )
    assert bad == []
    # Converged nodes don't get added to mismatches_by_node (no entry).
    assert mismatches.get("n1", []) == []


def test_find_disputed_handles_missing_result_json(tmp_path):
    """If one angle didn't write result.json (e.g. agent failed), the
    node is flagged as bad with empty mismatches."""
    cells_root = tmp_path / "cells"
    pending = [{"compound": "n1"}]
    # Only one angle has a result.
    _setup_round(cells_root, "n1", "a", 0, 100.0, 10.0)
    bad, mismatches, _ = s3._find_disputed(
        pending=pending, cells_root=cells_root, round_idx=0,
    )
    assert len(bad) == 1
    assert mismatches["n1"] == []


def test_finalize_node_unanimous_picks_angle_a(tmp_path):
    cells_root = tmp_path / "cells"
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    _setup_round(cells_root, "n1", "a", 0, 100.0, 10.0,
                 calc_src="def calc(b, s):\n    return {}\n")
    _setup_round(cells_root, "n1", "b", 0, 101.0, 10.0,
                 calc_src="def calc(b, s):\n    return {}\n")
    cs = s3.CellStateStore(cells_root / "_state.json")
    cs.init_node("n1", node_id="attn", section_id="layer",
                 section_kind="layer_template", section_repeat_count=1)
    cs.update_cell("n1", "a", prefill={"tflops": 100.0, "access_gb": 10.0},
                   decode={"tflops": 0.0, "access_gb": 0.0},
                   round_idx=0, status="ok", elapsed_s=1.0)
    cs.update_cell("n1", "b", prefill={"tflops": 101.0, "access_gb": 10.0},
                   decode={"tflops": 0.0, "access_gb": 0.0},
                   round_idx=0, status="ok", elapsed_s=1.0)
    node_meta = {"node_id": "attn", "section_id": "layer",
                 "section_kind": "layer_template", "section_repeat_count": 1}
    script_path, summary = s3._finalize_node(
        compound="n1", node_meta=node_meta,
        cells_root=cells_root, final_dir=final_dir,
        last_round=0, cell_state=cs,
    )
    assert summary["approximate"] is False
    meta = json.loads((final_dir / "n1.meta.json").read_text())
    assert meta["source_agent"] == "unanimous"
    assert meta["chosen_angle"] == "a"
    # The final calc.py exists.
    assert (final_dir / "n1.py").exists()


def test_finalize_node_median_fallback_when_disputed(tmp_path):
    cells_root = tmp_path / "cells"
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    _setup_round(cells_root, "n1", "a", 0, 100.0, 10.0,
                 calc_src="def calc(b, s):\n    return {}\n")
    _setup_round(cells_root, "n1", "b", 0, 500.0, 10.0,
                 calc_src="def calc(b, s):\n    return {}\n")
    cs = s3.CellStateStore(cells_root / "_state.json")
    cs.init_node("n1", node_id="attn", section_id="layer",
                 section_kind="layer_template", section_repeat_count=1)
    cs.update_cell("n1", "a", prefill={"tflops": 100.0, "access_gb": 10.0},
                   decode={"tflops": 0.0, "access_gb": 0.0},
                   round_idx=0, status="ok", elapsed_s=1.0)
    cs.update_cell("n1", "b", prefill={"tflops": 500.0, "access_gb": 10.0},
                   decode={"tflops": 0.0, "access_gb": 0.0},
                   round_idx=0, status="ok", elapsed_s=1.0)
    node_meta = {"node_id": "attn", "section_id": "layer",
                 "section_kind": "layer_template", "section_repeat_count": 1}
    _, summary = s3._finalize_node(
        compound="n1", node_meta=node_meta,
        cells_root=cells_root, final_dir=final_dir,
        last_round=0, cell_state=cs,
    )
    assert summary["approximate"] is True
    meta = json.loads((final_dir / "n1.meta.json").read_text())
    assert meta["source_agent"].startswith("median_fallback_from_angle_")


def test_finalize_node_degenerate_when_no_cells(tmp_path):
    """No usable cells at all → emit a degenerate calc + mark approximate."""
    cells_root = tmp_path / "cells"
    cells_root.mkdir()
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    cs = s3.CellStateStore(cells_root / "_state.json")
    node_meta = {"node_id": "attn", "section_id": "layer",
                 "section_kind": "layer_template", "section_repeat_count": 1}
    _, summary = s3._finalize_node(
        compound="missing", node_meta=node_meta,
        cells_root=cells_root, final_dir=final_dir,
        last_round=0, cell_state=cs,
    )
    assert summary["approximate"] is True
    assert (final_dir / "missing.py").exists()
    meta = json.loads((final_dir / "missing.meta.json").read_text())
    assert meta["source_agent"] == "degenerate_fallback"


# --------------------------------------------------------------------------- #
# step3_calculate — top-level run with MockAgentManager
# --------------------------------------------------------------------------- #

def test_run_step3_two_angle_agreement_end_to_end(tmp_path):
    """Both angles produce the same calc.py → unanimous convergence."""
    # The mock agent returns a calc.py that always emits the same numbers.
    src = _calc_src(100.0, 10.0, dec_t=1.0, dec_g=20.0)

    def response_fn(spec: Any) -> str:
        return _fenced(src)

    manager = MockAgentManager(response_fn=response_fn)
    store = FakeStore()
    # Build a minimal graph (sectioned, single layer node).
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps({
        "sections": [{
            "id": "layer", "kind": "layer_template", "repeat_count": 1,
            "graph": {"nodes": [{
                "id": "attn", "purpose": "test", "op": "attention",
                "inputs": [], "outputs": [],
            }]},
        }],
    }))
    req = {"model_dir": "/tmp/x", "framework_source_dir": "/tmp/y"}
    paths = {
        "step1_dir": tmp_path / "step1",
        "step3_dir": tmp_path / "step3",
    }
    (tmp_path / "step1").mkdir()
    (tmp_path / "step1" / "memory.json").write_text("{}")
    final_dir = s3.run_step3_calculate(
        req=req, store=store, manager=manager, paths=paths,
        graph_path=graph_path,
    )
    # final/<compound>.py exists and matches what the agent wrote.
    assert (final_dir / "layer__attn.py").exists()
    meta = json.loads((final_dir / "layer__attn.meta.json").read_text())
    assert meta["approximate"] is False
    assert meta["source_agent"] == "unanimous"
    # result.json exists for both angles at round 0.
    cells_root = tmp_path / "step3" / "cells"
    for a in ("a", "b"):
        rj = cells_root / "layer__attn" / a / "round_00" / "result.json"
        assert rj.exists(), f"missing {rj}"
        rec = json.loads(rj.read_text())
        assert rec["prefill"]["tflops"] == 100.0
        assert rec["decode"]["tflops"] == 1.0
    # Timeline events.
    assert "calc_value.s3.start" in store.types()
    assert "calc_value.s3.cell.done" in store.types()
    assert "calc_value.s3.all_nodes_done" in store.types()


def test_run_step3_marks_approximate_when_angles_persistently_disagree(tmp_path):
    """When the two angles give wildly different numbers every round,
    after MAX_ROUNDS_PER_NODE the node is finalized as approximate."""
    # Use deterministic disagreement: angle a returns 100, angle b returns 500.
    def response_fn(spec: Any) -> str:
        # spec.name = "s3_<compound>_<angle>_r<round>"
        parts = spec.name.split("_")
        # Last meaningful token before round is the angle.
        # Find 'a' or 'b' anywhere in the name.
        if "_a_" in spec.name or spec.name.endswith("_a") or "_a_r" in spec.name:
            src = _calc_src(100.0, 10.0)
        else:
            src = _calc_src(500.0, 10.0)
        return _fenced(src)

    manager = MockAgentManager(response_fn=response_fn)
    store = FakeStore()
    graph_path = tmp_path / "graph.json"
    graph_path.write_text(json.dumps({
        "sections": [{
            "id": "layer", "kind": "layer_template", "repeat_count": 1,
            "graph": {"nodes": [{
                "id": "attn", "purpose": "t", "op": "attention",
                "inputs": [], "outputs": [],
            }]},
        }],
    }))
    req = {"model_dir": "/x", "framework_source_dir": "/y"}
    paths = {"step1_dir": tmp_path / "step1", "step3_dir": tmp_path / "step3"}
    (tmp_path / "step1").mkdir()
    (tmp_path / "step1" / "memory.json").write_text("{}")
    final_dir = s3.run_step3_calculate(
        req=req, store=store, manager=manager, paths=paths,
        graph_path=graph_path,
    )
    meta = json.loads((final_dir / "layer__attn.meta.json").read_text())
    assert meta["approximate"] is True
    assert meta["source_agent"].startswith("median_fallback_from_angle_")
    # MAX_ROUNDS_PER_NODE = 3 cap honored.
    assert meta["rounds"] <= s3.MAX_ROUNDS_PER_NODE


def test_max_rounds_cap_is_three():
    """The user's requirement: max 3 iterations cap for the 2-agent
    consistency loop."""
    assert s3.MAX_ROUNDS_PER_NODE == 3
    assert tuple(s3.ANGLES) == ("a", "b")
