"""Integration tests for step3_calculate with tiered angles + memory slicing.

Run directly::

    python metainfer/tasks/calc_value/tests/test_step3_tiered.py

Covers:
- Complexity classification: simple nodes (rmsnorm, residual add) → 1 angle,
  complex nodes (attention, mlp, linear, rope) → 2 angles.
- Simple nodes never spawn an agent for angle b.
- CellStateStore marks simple nodes converged as soon as angle a lands.
- _finalize_node picks the single angle for simple nodes without
  requiring median fallback.
- Each writer prompt contains a memory SLICE (no full memory dump).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from metainfer.tasks.calc_value.orchestrator import step3_calculate as s3
from metainfer.testing import FakeStore, MockAgentManager


def _make_calc_response(prefill_tflops: float, prefill_gb: float):
    """Response_fn that writes calc.py + returns text containing it.
    The calc.py uses the canonical schema."""
    src = (
        "def calc(batch_size, seq_len):\n"
        "    return {\n"
        f"        'prefill': {{'tflops': {prefill_tflops}, 'access_gb': {prefill_gb}}},\n"
        "        'decode': {'tflops': 1.0, 'access_gb': 2.0},\n"
        "    }\n"
    )
    def fn(spec):
        workdir = Path(spec.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "calc.py").write_text(src, encoding="utf-8")
        return f"wrote calc:\n```python\n{src}\n```\n"
    return fn


def _simple_graph() -> Dict[str, Any]:
    """A 3-node single-section graph: norm (simple), attn (complex), mlp (complex)."""
    return {
        "sections": [{
            "id": "layer",
            "kind": "layer_template",
            "repeat_count": 1,
            "description": "block",
            "graph": {
                "nodes": [
                    {"id": "norm", "op": "rmsnorm", "purpose": "norm"},
                    {"id": "attn", "op": "attention", "purpose": "attn"},
                    {"id": "mlp", "op": "mlp", "purpose": "ffn"},
                ],
                "edges": [],
            },
        }],
    }


def _toy_memory() -> Dict[str, Any]:
    return {
        "architecture_summary": {"hidden_size": 4096},
        "operator_calls": [
            {"node_id_hint": "norm", "op": "rmsnorm", "purpose": "norm"},
            {"node_id_hint": "attn", "op": "attention", "purpose": "attn"},
            {"node_id_hint": "mlp", "op": "mlp", "purpose": "ffn"},
            # 40 decoys to verify slicing.
            *[{"node_id_hint": f"decoy_{i}", "op": f"decoy_{i}",
               "purpose": "x"} for i in range(40)],
        ],
        "uncertainties": [],
        "quantization_load": {},
        "tp_behavior": {},
    }


def _write_graph(workdir: Path) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / "graph.json"
    p.write_text(json.dumps(_simple_graph()), encoding="utf-8")
    return p


def _setup_paths(tmp: Path):
    paths = {
        "step1_dir": tmp / "step1",
        "step3_dir": tmp / "step3",
    }
    paths["step1_dir"].mkdir(parents=True, exist_ok=True)
    (paths["step1_dir"] / "memory.json").write_text(
        json.dumps(_toy_memory()), encoding="utf-8",
    )
    return paths


def test_classify_complexity_picks_correct_tier():
    assert s3._classify_complexity({"op": "rmsnorm"}) is False
    assert s3._classify_complexity({"op": "add"}) is False
    assert s3._classify_complexity({"op": "silu"}) is False
    assert s3._classify_complexity({"op": "layernorm"}) is False
    assert s3._classify_complexity({"op": "attention"}) is True
    assert s3._classify_complexity({"op": "mlp"}) is True
    assert s3._classify_complexity({"op": "linear"}) is True
    assert s3._classify_complexity({"op": "rotary_embedding"}) is True
    assert s3._classify_complexity({"op": "quantize"}) is True


def test_simple_node_only_runs_one_angle():
    """A graph with only a simple node should spawn exactly 1 writer
    agent (angle a only) in round 0."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = _setup_paths(tmp)
        graph_path = tmp / "graph.json"
        graph_path.write_text(json.dumps({
            "sections": [{
                "id": "s", "kind": "block", "repeat_count": 1,
                "graph": {"nodes": [
                    {"id": "norm", "op": "rmsnorm"}
                ], "edges": []},
            }],
        }), encoding="utf-8")
        mgr = MockAgentManager(response_fn=_make_calc_response(10.0, 5.0))
        store = FakeStore()
        s3.run_step3_calculate(
            req={"model_dir": "/m", "framework_source_dir": "/f",
                 "cmdline_args": "", "env_vars": ""},
            store=store, manager=mgr, paths=paths, graph_path=graph_path,
        )
        # 1 spec launched (angle a only).
        assert len(mgr.launched_specs) == 1
        # Final script exists.
        final_script = paths["step3_dir"] / "final" / "s__norm.py"
        assert final_script.exists()
        # Meta marks it as non-approximate single-angle.
        meta = json.loads(
            (paths["step3_dir"] / "final" / "s__norm.meta.json").read_text()
        )
        assert meta["approximate"] is False
        assert meta["source_agent"] == "single_angle"
        assert meta["angles"] == ["a"]


def test_complex_nodes_spawn_two_angles():
    """A graph with 1 complex node should spawn 2 agents in round 0
    (angle a + angle b)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = _setup_paths(tmp)
        graph_path = tmp / "graph.json"
        graph_path.write_text(json.dumps({
            "sections": [{
                "id": "s", "kind": "block", "repeat_count": 1,
                "graph": {"nodes": [
                    {"id": "attn", "op": "attention"}
                ], "edges": []},
            }],
        }), encoding="utf-8")
        mgr = MockAgentManager(response_fn=_make_calc_response(10.0, 5.0))
        store = FakeStore()
        s3.run_step3_calculate(
            req={"model_dir": "/m", "framework_source_dir": "/f",
                 "cmdline_args": "", "env_vars": ""},
            store=store, manager=mgr, paths=paths, graph_path=graph_path,
        )
        # 2 specs launched: angle a + angle b.
        assert len(mgr.launched_specs) == 2
        names = sorted(s.name for s in mgr.launched_specs)
        assert any("_a_" in n for n in names)
        assert any("_b_" in n for n in names)


def test_mixed_graph_spawn_count():
    """1 simple + 1 complex node → 3 agents in round 0
    (norm:1, attn:2)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = _setup_paths(tmp)
        graph_path = tmp / "graph.json"
        graph_path.write_text(json.dumps({
            "sections": [{
                "id": "s", "kind": "block", "repeat_count": 1,
                "graph": {"nodes": [
                    {"id": "norm", "op": "rmsnorm"},
                    {"id": "attn", "op": "attention"},
                ], "edges": []},
            }],
        }), encoding="utf-8")
        mgr = MockAgentManager(response_fn=_make_calc_response(10.0, 5.0))
        store = FakeStore()
        s3.run_step3_calculate(
            req={"model_dir": "/m", "framework_source_dir": "/f",
                 "cmdline_args": "", "env_vars": ""},
            store=store, manager=mgr, paths=paths, graph_path=graph_path,
        )
        # Simple node spawns 1, complex spawns 2 → total 3.
        assert len(mgr.launched_specs) == 3
        # No "b" angle for norm.
        norm_b_specs = [s for s in mgr.launched_specs
                        if "s__norm" in s.name and "_b_" in s.name]
        assert norm_b_specs == []


def test_writer_prompt_contains_memory_slice_not_full():
    """Each writer's prompt should NOT include decoy ops from memory."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        paths = _setup_paths(tmp)
        graph_path = _write_graph(tmp)
        mgr = MockAgentManager(response_fn=_make_calc_response(10.0, 5.0))
        store = FakeStore()
        s3.run_step3_calculate(
            req={"model_dir": "/m", "framework_source_dir": "/f",
                 "cmdline_args": "", "env_vars": ""},
            store=store, manager=mgr, paths=paths, graph_path=graph_path,
        )
        # All prompt files must exclude the decoy ops.
        prompt_files = list(paths["step3_dir"].rglob("*.prompt.txt"))
        assert len(prompt_files) > 0
        for p in prompt_files:
            text = p.read_text(encoding="utf-8")
            assert "decoy_0" not in text
            assert "decoy_39" not in text


def test_simple_node_marks_converged_after_one_angle():
    """CellStateStore marks a simple node converged as soon as angle a
    has a value — no second angle needed."""
    import tempfile as tf2
    state_path = Path(tf2.mkdtemp()) / "_state.json"
    css = s3.CellStateStore(state_path)
    css.init_node("comp", node_id="n", section_id="s",
                  section_kind="block", section_repeat_count=1,
                  complex_node=False)
    # No values yet → not converged.
    assert css._doc["nodes"]["comp"]["converged"] is None
    # Add angle a.
    css.update_cell("comp", "a",
                    prefill={"tflops": 10.0, "access_gb": 5.0},
                    decode={"tflops": 1.0, "access_gb": 2.0},
                    round_idx=0, status="ok", elapsed_s=1.0)
    # Simple node → auto-converged.
    assert css._doc["nodes"]["comp"]["converged"] is True
    assert css._doc["nodes"]["comp"]["spread_pct"] == 0.0


def test_complex_node_requires_both_angles_for_convergence():
    """Complex node with only angle a → not yet converged (None)."""
    import tempfile as tf2
    state_path = Path(tf2.mkdtemp()) / "_state.json"
    css = s3.CellStateStore(state_path)
    css.init_node("comp", node_id="n", section_id="s",
                  section_kind="block", section_repeat_count=1,
                  complex_node=True)
    css.update_cell("comp", "a",
                    prefill={"tflops": 10.0, "access_gb": 5.0},
                    decode={"tflops": 1.0, "access_gb": 2.0},
                    round_idx=0, status="ok", elapsed_s=1.0)
    # One angle only → not yet converged (still waiting for b).
    assert css._doc["nodes"]["comp"]["converged"] is None
    # Add matching angle b → converged.
    css.update_cell("comp", "b",
                    prefill={"tflops": 10.5, "access_gb": 5.2},
                    decode={"tflops": 1.0, "access_gb": 2.0},
                    round_idx=0, status="ok", elapsed_s=1.0)
    assert css._doc["nodes"]["comp"]["converged"] is True


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            import traceback as tb
            print(f"FAIL  {fn.__name__}: {e!r}")
            tb.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
