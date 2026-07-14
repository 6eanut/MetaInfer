"""Integration tests for step2_graph with the new AgentPool.

Run directly::

    python metainfer/tasks/calc_value/tests/test_step2_pool.py

Covers:
- Validators go through AgentPool (3 worker sessions, round-robin)
- Memory slicing per validator (per-node prompts only get the slice)
- Incremental re-validation after fix: only changed nodes + neighbors
- Carry-forward verdicts for unchanged nodes
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from metainfer.tasks.calc_value.orchestrator import step2_graph as s2
from metainfer.testing import FakeStore, MockAgentManager


def _write_graph(workdir: Path, graph: Dict[str, Any]) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / "graph.json"
    p.write_text(json.dumps(graph, indent=2), encoding="utf-8")
    return p


def _simple_graph() -> Dict[str, Any]:
    """A 3-node single-section graph for test fixtures."""
    return {
        "sections": [{
            "id": "layer",
            "kind": "layer_template",
            "repeat_count": 2,
            "description": "transformer block",
            "graph": {
                "nodes": [
                    {"id": "attn", "op": "attention", "purpose": "attn"},
                    {"id": "mlp", "op": "mlp", "purpose": "ffn"},
                    {"id": "norm", "op": "rmsnorm", "purpose": "norm"},
                ],
                "edges": [
                    {"from": "attn", "to": "mlp"},
                    {"from": "mlp", "to": "norm"},
                ],
            },
        }],
    }


def _toy_memory() -> Dict[str, Any]:
    return {
        "architecture_summary": {"hidden_size": 4096, "num_layers": 32},
        "operator_calls": [
            {"node_id_hint": "attn", "op": "attention",
             "purpose": "multi-query attention"},
            {"node_id_hint": "mlp", "op": "mlp", "purpose": "ffn"},
            {"node_id_hint": "norm", "op": "rmsnorm", "purpose": "norm"},
        ],
        "uncertainties": [],
        "quantization_load": {},
        "tp_behavior": {},
    }


def _make_response_fn(verdict: str = "pass", reason: str = "ok"):
    """Returns a response_fn for MockAgentManager that writes a
    verdict.json file to the spec's workdir and returns final_text."""
    def fn(spec):
        verdict_obj = {
            "verdict": verdict,
            "reason": reason,
            "suggested_fix": None,
        }
        Path(spec.workdir).mkdir(parents=True, exist_ok=True)
        (Path(spec.workdir) / "verdict.json").write_text(
            json.dumps(verdict_obj), encoding="utf-8",
        )
        return f"validator done for {spec.name}"
    return fn


def _common(req_extra=None):
    req = {
        "model_dir": "/models/x",
        "framework_source_dir": "/src",
        "cmdline_args": "",
        "env_vars": "",
        **(req_extra or {}),
    }
    return req


def test_validate_nodes_uses_pool():
    """Smoke: _validate_nodes_parallel runs the full 3-node batch
    through the pool and returns 3 verdicts."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mgr = MockAgentManager(response_fn=_make_response_fn("pass"))
        store = FakeStore()
        round_dir = tmp / "round01_validate"
        common = s2._common_format(_common(), json.dumps(_toy_memory()))
        verdicts = s2._validate_nodes_parallel(
            mgr, round_dir, common, _simple_graph(), _toy_memory(),
            only_targets=None,
        )
        assert len(verdicts) == 3
        keys = {(v["section_id"], v["node_id"]) for v in verdicts}
        assert keys == {("layer", "attn"), ("layer", "mlp"), ("layer", "norm")}
        # All pass.
        assert all(v["verdict"] == "pass" for v in verdicts)
        # Pool left exactly 3 launches in the mock (one per task).
        assert len(mgr.launched_specs) == 3


def test_validate_nodes_increments_session_within_worker():
    """Turn 0 of each worker = fresh; turn 1+ = resume. With N=3 pool
    and 3 nodes, every task is turn 0 (no resumes). Scale to 6 nodes
    to exercise resume path."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Build a 6-node graph so each of 3 workers handles 2 turns.
        graph = {
            "sections": [{
                "id": "layer", "kind": "layer_template", "repeat_count": 1,
                "graph": {
                    "nodes": [
                        {"id": f"n{i}", "op": "add"} for i in range(6)
                    ],
                    "edges": [],
                },
            }],
        }
        mgr = MockAgentManager(response_fn=_make_response_fn("pass"))
        common = s2._common_format(_common(), json.dumps(_toy_memory()))
        verdicts = s2._validate_nodes_parallel(
            mgr, tmp / "round", common, graph, _toy_memory(),
        )
        assert len(verdicts) == 6

        # 6 specs launched total. 3 of them (turn 0 of each worker)
        # have resume_session_id=None. The other 3 (turn 1) have a
        # non-None resume_session_id.
        turn0 = [s for s in mgr.launched_specs if s.resume_session_id is None]
        turn1 = [s for s in mgr.launched_specs if s.resume_session_id is not None]
        assert len(turn0) == 3
        assert len(turn1) == 3
        # Each turn1's resume_session_id should match the session_id
        # synthesized for its worker's turn0 spec.
        turn0_sids = {s.name: f"mock-sess-{s.name}" for s in turn0}
        # Worker of turn1 == worker of some turn0 (round-robin assigns
        # i % 3). Verify the resume id is plausible — it must be a
        # mock-sess-<name> string.
        for s in turn1:
            assert s.resume_session_id is not None
            assert s.resume_session_id.startswith("mock-sess-")


def test_validate_nodes_uses_memory_slice():
    """The prompt passed to each validator should contain the memory
    SLICE (just the matching operator + global bits), NOT the full
    memory's operator_calls list."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Memory with 40 decoy operator_calls. The slice for "attn"
        # should only mention "attention" — none of the decoys.
        mem = _toy_memory()
        for i in range(40):
            mem["operator_calls"].append({
                "node_id_hint": f"decoy_{i}",
                "op": f"decoy_{i}",
                "purpose": f"decoy purpose {i}",
            })
        mgr = MockAgentManager(response_fn=_make_response_fn("pass"))
        common = s2._common_format(_common(), json.dumps(mem))
        s2._validate_nodes_parallel(
            mgr, tmp / "round", common, _simple_graph(), mem,
        )
        # Inspect the prompt file for the attn validator task.
        # Find the task workdir containing "attn".
        prompt_files = list((tmp / "round" / "tasks").glob("*/*.prompt.txt"))
        assert len(prompt_files) > 0
        attn_prompt = next(p for p in prompt_files if "attn" in p.name)
        text = attn_prompt.read_text(encoding="utf-8")
        # Must mention "attention" (its matched op).
        assert "attention" in text
        # Must NOT mention the decoy ops.
        assert "decoy_0" not in text
        assert "decoy_39" not in text


def test_diff_nodes_detects_changes():
    old = _simple_graph()
    new = json.loads(json.dumps(old))
    # Change the attn node only.
    new["sections"][0]["graph"]["nodes"][0]["op"] = "flash_attention"
    changed = s2._diff_nodes(old, new)
    assert ("layer", "attn") in changed
    assert ("layer", "mlp") not in changed
    assert ("layer", "norm") not in changed


def test_diff_nodes_detects_additions():
    old = _simple_graph()
    new = json.loads(json.dumps(old))
    new["sections"][0]["graph"]["nodes"].append(
        {"id": "residual", "op": "add"}
    )
    changed = s2._diff_nodes(old, new)
    assert ("layer", "residual") in changed


def test_expand_to_neighbors_includes_adjacent():
    """_expand_to_neighbors({(layer, attn)}) on the simple graph should
    also pick up mlp (attn's downstream neighbor)."""
    seeds = {("layer", "attn")}
    expanded = s2._expand_to_neighbors(_simple_graph(), seeds)
    assert ("layer", "attn") in expanded
    assert ("layer", "mlp") in expanded  # attn → mlp edge
    # norm is NOT adjacent to attn.
    assert ("layer", "norm") not in expanded


def test_expand_to_neighbors_empty():
    assert s2._expand_to_neighbors(_simple_graph(), set()) == set()


def test_expand_to_neighbors_handles_orphaned_section():
    """If a seed references a section that no longer exists in the
    graph (deleted/renamed by a fix round), _expand_to_neighbors must
    not raise — the orphaned seed just contributes no neighbors.
    Regression test for the StopIteration crash flagged by the verifier.
    """
    seeds = {("nonexistent_section", "some_node")}
    # Should not raise; orphaned seed is silently skipped.
    result = s2._expand_to_neighbors(_simple_graph(), seeds)
    assert result == seeds  # seed preserved, no neighbors added


def test_expand_to_neighbors_handles_orphaned_node():
    """Section exists but seed's node id is unknown — neighbors lookup
    just finds nothing."""
    seeds = {("layer", "ghost_node")}
    result = s2._expand_to_neighbors(_simple_graph(), seeds)
    # Seed preserved; no neighbor additions because ghost_node has no edges.
    assert ("layer", "ghost_node") in result


def test_incremental_after_section_deletion():
    """End-to-end regression: simulate a fix round that deletes a whole
    section, then ensure _diff_nodes + _expand_to_neighbors + the pool
    path doesn't crash on round >= 1.

    This was the exact failure mode the verifier caught: StopIteration
    in _expand_to_neighbors when called with orphaned seeds.
    """
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        old_graph = {
            "sections": [
                {"id": "keep", "kind": "block", "repeat_count": 1,
                 "graph": {"nodes": [{"id": "n0", "op": "add"}], "edges": []}},
                {"id": "drop", "kind": "block", "repeat_count": 1,
                 "graph": {"nodes": [{"id": "x", "op": "add"},
                                     {"id": "y", "op": "add"}],
                           "edges": [{"from": "x", "to": "y"}]}},
            ],
        }
        # Fix agent deletes the entire "drop" section.
        new_graph = {"sections": [old_graph["sections"][0]]}
        changed = s2._diff_nodes(old_graph, new_graph)
        # Both nodes of the dropped section register as changed.
        assert ("drop", "x") in changed
        assert ("drop", "y") in changed
        # This MUST not raise StopIteration.
        expanded = s2._expand_to_neighbors(new_graph, changed)
        # Orphaned seeds preserved, no neighbor additions possible.
        assert ("drop", "x") in expanded
        assert ("drop", "y") in expanded

        # And the pool call with these orphaned targets must also not
        # crash — it should just produce 2 verdicts (one per target).
        mgr = MockAgentManager(response_fn=_make_response_fn("pass"))
        common = s2._common_format(_common(), json.dumps(_toy_memory()))
        verdicts = s2._validate_nodes_parallel(
            mgr, tmp / "round", common, new_graph, _toy_memory(),
            only_targets=expanded,
        )
        # The "drop" section is gone from new_graph so no nodes match
        # those targets → empty verdicts (not a crash).
        assert verdicts == []


def test_only_targets_filters_nodes():
    """When only_targets is supplied, only matching nodes get
    validated."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mgr = MockAgentManager(response_fn=_make_response_fn("pass"))
        common = s2._common_format(_common(), json.dumps(_toy_memory()))
        # Only validate the attn node + its neighbors (mlp).
        targets = s2._expand_to_neighbors(_simple_graph(),
                                          {("layer", "attn")})
        verdicts = s2._validate_nodes_parallel(
            mgr, tmp / "round", common, _simple_graph(), _toy_memory(),
            only_targets=targets,
        )
        assert len(verdicts) == 2
        keys = {(v["section_id"], v["node_id"]) for v in verdicts}
        assert keys == {("layer", "attn"), ("layer", "mlp")}


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
