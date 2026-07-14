"""Unit tests for :mod:`metainfer.tasks.calc_value.orchestrator.memory_slice`.

Run directly::

    python metainfer/tasks/calc_value/tests/test_memory_slice.py
"""

from __future__ import annotations

import json

from metainfer.tasks.calc_value.orchestrator.memory_slice import (
    _normalize_node_id,
    slice_memory_for_node,
    slice_memory_for_node_as_json,
)


def _toy_memory():
    return {
        "architecture_summary": {
            "hidden_size": 4096,
            "num_layers": 32,
            "vocab_size": 32000,
        },
        "operator_calls": [
            {
                "node_id_hint": "attn_mqa",
                "op": "attention",
                "purpose": "multi-query attention",
            },
            {
                "node_id_hint": "mlp_0",
                "op": "mlp",
                "purpose": "feed-forward network",
            },
            {
                "node_id_hint": "embedding",
                "op": "embedding",
                "purpose": "token embedding lookup",
            },
        ],
        "uncertainties": [
            {
                "id": "u1",
                "op": "attention",
                "note": "is gqa reshaping correct?",
            },
            {
                "id": "u2",
                "op": "mlp",
                "note": "swiglu vs gelu",
            },
        ],
        "quantization_load": {"weight": "int8", "activations": "fp16"},
        "tp_behavior": {
            "summary": "tensor parallel 4",
            "nonsplit_weights": ["embedding", "lm_head"],
            "split_weights": [{"name": "wq", "axis": 0}],
        },
    }


def test_normalize_node_id_strips_trailing_digits():
    assert _normalize_node_id("attn_mqa_7") == "attn_mqa"
    assert _normalize_node_id("Attn_MQA_3") == "attn_mqa"
    assert _normalize_node_id("attn_mqa") == "attn_mqa"
    assert _normalize_node_id("") == ""
    assert _normalize_node_id("a_3_b_12") == "a_3_b"


def test_architecture_summary_always_present():
    mem = _toy_memory()
    out = slice_memory_for_node(mem, "nonexistent_node")
    assert "architecture_summary" in out
    assert out["architecture_summary"]["hidden_size"] == 4096


def test_operator_call_exact_match():
    mem = _toy_memory()
    out = slice_memory_for_node(mem, "attn_mqa_7")  # suffix-stripped exact match
    op = out.get("operator_call")
    assert op is not None
    assert op["op"] == "attention"
    assert op["node_id_hint"] == "attn_mqa"


def test_operator_call_substring_match():
    mem = _toy_memory()
    out = slice_memory_for_node(mem, "mlp_gate_proj_special")
    op = out.get("operator_call")
    assert op is not None
    assert op["op"] == "mlp"


def test_operator_call_purpose_fallback():
    """When the id doesn't match anything but purpose does, use purpose."""
    mem = _toy_memory()
    out = slice_memory_for_node(mem, "weird_id_xyz",
                                node_purpose="feed-forward network")
    op = out.get("operator_call")
    assert op is not None
    assert op["op"] == "mlp"


def test_related_uncertainties_attached():
    mem = _toy_memory()
    out = slice_memory_for_node(mem, "attn_mqa")
    assert "uncertainties" in out
    u_ops = {u["op"] for u in out["uncertainties"]}
    assert "attention" in u_ops
    # mlp uncertainty should NOT be included.
    assert "mlp" not in u_ops


def test_quantization_and_tp_always_present():
    mem = _toy_memory()
    out = slice_memory_for_node(mem, "any_node")
    assert out["quantization_load"]["weight"] == "int8"
    # tp_behavior should be slimmed to summary + nonsplit_weights only.
    assert "summary" in out["tp_behavior"]
    assert "nonsplit_weights" in out["tp_behavior"]
    assert "split_weights" not in out["tp_behavior"]


def test_no_operator_match_returns_minimal():
    mem = _toy_memory()
    out = slice_memory_for_node(mem, "totally_unknown_op")
    assert "operator_call" not in out
    assert "uncertainties" not in out
    # Architecture summary still present.
    assert "architecture_summary" in out


def test_slice_is_much_smaller_than_full():
    """The whole point of slicing — the JSON output must be tiny
    relative to the full memory."""
    mem = _toy_memory()
    full_size = len(json.dumps(mem, ensure_ascii=False))
    slice_size = len(slice_memory_for_node_as_json(mem, "attn_mqa"))
    assert slice_size < full_size
    # Even more so with the full memory scaled up to realistic size:
    big_mem = dict(mem)
    big_mem["operator_calls"] = [
        {"node_id_hint": f"node_{i}", "op": f"op_{i}",
         "purpose": "filler " * 50}
        for i in range(40)
    ]
    big_size = len(json.dumps(big_mem, ensure_ascii=False))
    out = slice_memory_for_node_as_json(big_mem, "attn_mqa")
    assert len(out) < big_size * 0.1, (
        f"slice should be <10% of full; got {len(out)} vs {big_size}")


def test_invalid_input_returns_empty():
    assert slice_memory_for_node(None, "x") == {}
    assert slice_memory_for_node("not a dict", "x") == {}
    assert slice_memory_for_node([], "x") == {}


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
            print(f"FAIL  {fn.__name__}: {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
