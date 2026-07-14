"""Unit tests for calc_value's deterministic helpers.

Covers: extract_json, load_calc_module, call_calc (incl. legacy shape
backward-compat), run_calc_canonical, compare_calc_results,
median_result, format_mismatches_for_prompt, normalize_graph,
validate_graph_structure (smoke).

Run: ``python -m pytest tests/test_calc_deterministic.py -v``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from metainfer.tasks.calc_value.orchestrator import deterministic as det
from metainfer.testing import write_calc_script


# --------------------------------------------------------------------------- #
# extract_json
# --------------------------------------------------------------------------- #

def test_extract_json_plain():
    assert det.extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = 'Here you go:\n```json\n{"x": 2}\n```\nthanks'
    assert det.extract_json(text) == {"x": 2}


def test_extract_json_fenced_no_lang():
    text = '```\n{"y": 3}\n```'
    assert det.extract_json(text) == {"y": 3}


def test_extract_json_with_prose():
    text = 'Result: {"k": "v"} was great'
    assert det.extract_json(text) == {"k": "v"}


def test_extract_json_invalid_returns_none():
    assert det.extract_json("not json at all") is None


def test_extract_json_empty_returns_none():
    assert det.extract_json("") is None


# --------------------------------------------------------------------------- #
# load_calc_module + call_calc
# --------------------------------------------------------------------------- #

def test_call_calc_new_shape():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.py"
        write_calc_script(p, prefill_tflops=10.0, prefill_gb=2.0,
                          decode_tflops=1.0, decode_gb=0.5)
        mod = det.load_calc_module(p, module_name="_t_new")
        out = det.call_calc(mod, batch_size=1, seq_len=512)
        assert out["prefill"] == {"tflops": 10.0, "access_gb": 2.0}
        assert out["decode"] == {"tflops": 1.0, "access_gb": 0.5}


def test_call_calc_legacy_dict_shape():
    """Legacy scripts returning ``{"tflops", "access_gb"}`` are treated
    as prefill-only (decode zeroed)."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.py"
        write_calc_script(p, prefill_tflops=7.0, prefill_gb=3.0,
                          legacy_shape=True)
        mod = det.load_calc_module(p, module_name="_t_legacy")
        out = det.call_calc(mod, batch_size=1, seq_len=512)
        assert out["prefill"] == {"tflops": 7.0, "access_gb": 3.0}
        assert out["decode"] == {"tflops": 0.0, "access_gb": 0.0}


def test_call_calc_legacy_tuple_shape():
    """Some early scripts returned ``(tflops, gb)`` 2-tuples."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.py"
        p.write_text(
            "def calc(batch_size, seq_len):\n"
            "    return (5.0, 1.0)\n",
            encoding="utf-8",
        )
        mod = det.load_calc_module(p, module_name="_t_tuple")
        out = det.call_calc(mod, batch_size=1, seq_len=1)
        assert out["prefill"] == {"tflops": 5.0, "access_gb": 1.0}
        assert out["decode"] == {"tflops": 0.0, "access_gb": 0.0}


def test_call_calc_non_finite_raises():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.py"
        p.write_text(
            "import math\n"
            "def calc(batch_size, seq_len):\n"
            "    return {'prefill': {'tflops': math.inf, 'access_gb': 1.0}, 'decode': {'tflops': 0.0, 'access_gb': 0.0}}\n",
            encoding="utf-8",
        )
        mod = det.load_calc_module(p, module_name="_t_inf")
        with pytest.raises(ValueError):
            det.call_calc(mod, batch_size=1, seq_len=1)


# --------------------------------------------------------------------------- #
# run_calc_canonical
# --------------------------------------------------------------------------- #

def test_run_calc_canonical_returns_single_record():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "c.py"
        write_calc_script(p, prefill_tflops=42.0, prefill_gb=9.0,
                          decode_tflops=3.0, decode_gb=4.0)
        rec = det.run_calc_canonical(p)
        assert rec["batch_size"] == det.CANONICAL_BATCH == 1
        assert rec["seq_len"] == det.CANONICAL_SEQ == 512
        assert rec["prefill"] == {"tflops": 42.0, "access_gb": 9.0}
        assert rec["decode"] == {"tflops": 3.0, "access_gb": 4.0}


def test_canonical_constants_are_b1_s512():
    """The user's requirement: canonical shape is (B=1, S=512). If this
    ever changes intentionally, update this test deliberately."""
    assert det.CANONICAL_BATCH == 1
    assert det.CANONICAL_SEQ == 512


# --------------------------------------------------------------------------- #
# compare_calc_results
# --------------------------------------------------------------------------- #

def _mkrec(pre_t, pre_g, dec_t=0.0, dec_g=0.0):
    return {
        "batch_size": 1, "seq_len": 512,
        "prefill": {"tflops": pre_t, "access_gb": pre_g},
        "decode":  {"tflops": dec_t, "access_gb": dec_g},
    }


def test_compare_results_agree():
    results = [_mkrec(10.0, 2.0), _mkrec(10.1, 2.0)]
    cmp = det.compare_calc_results(results)
    assert cmp["ok"] is True
    assert cmp["mismatches"] == []


def test_compare_results_disagree_prefill_tflops():
    results = [_mkrec(10.0, 2.0), _mkrec(20.0, 2.0)]  # 100% spread on prefill
    cmp = det.compare_calc_results(results)
    assert cmp["ok"] is False
    assert len(cmp["mismatches"]) == 1
    m = cmp["mismatches"][0]
    assert m["batch_size"] == 1 and m["seq_len"] == 512
    assert len(m["values"]) == 2
    assert m["spread"]["prefill"]["tflops"] == pytest.approx(10.0)


def test_compare_results_disagree_decode_access_gb():
    """Decode KV-cache read disagreement is the canonical disagreement
    case the audit is designed to surface."""
    results = [_mkrec(10.0, 2.0, dec_t=1.0, dec_g=0.5),
               _mkrec(10.0, 2.0, dec_t=1.0, dec_g=5.0)]
    cmp = det.compare_calc_results(results)
    assert cmp["ok"] is False
    m = cmp["mismatches"][0]
    assert m["spread"]["decode"]["access_gb"] == pytest.approx(4.5)


def test_compare_results_empty_input():
    cmp = det.compare_calc_results([])
    assert cmp["ok"] is False
    assert cmp["mismatches"] == []


def test_compare_results_single_input_is_ok():
    cmp = det.compare_calc_results([_mkrec(1.0, 1.0)])
    assert cmp["ok"] is True


def test_compare_results_within_tolerance():
    """5% rel tolerance is the documented contract."""
    results = [_mkrec(100.0, 2.0), _mkrec(104.0, 2.0)]  # 4% spread
    cmp = det.compare_calc_results(results)
    assert cmp["ok"] is True


def test_compare_results_accepts_legacy_shape():
    """Legacy records without prefill/decode keys are coerced."""
    legacy = [
        {"batch_size": 1, "seq_len": 512, "tflops": 10.0, "access_gb": 2.0},
        {"batch_size": 1, "seq_len": 512, "tflops": 10.1, "access_gb": 2.0},
    ]
    cmp = det.compare_calc_results(legacy)
    assert cmp["ok"] is True


# --------------------------------------------------------------------------- #
# median_result
# --------------------------------------------------------------------------- #

def test_median_result_two_agents():
    results = [_mkrec(10.0, 1.0), _mkrec(20.0, 2.0)]
    med = det.median_result(results)
    # Implementation: sorted(vals)[len//2] → upper of 2.
    assert med["prefill"]["tflops"] == 20.0
    assert med["prefill"]["access_gb"] == 2.0


def test_median_result_three_agents():
    results = [_mkrec(30.0, 3.0), _mkrec(10.0, 1.0), _mkrec(20.0, 2.0)]
    med = det.median_result(results)
    assert med["prefill"]["tflops"] == 20.0
    assert med["prefill"]["access_gb"] == 2.0


def test_median_result_empty():
    med = det.median_result([])
    assert med["prefill"] == {"tflops": 0.0, "access_gb": 0.0}
    assert med["decode"] == {"tflops": 0.0, "access_gb": 0.0}
    assert med["source"] == "median_fallback_empty"


def test_median_result_legacy_shape():
    legacy = [
        {"batch_size": 1, "seq_len": 512, "tflops": 10.0, "access_gb": 1.0},
        {"batch_size": 1, "seq_len": 512, "tflops": 30.0, "access_gb": 3.0},
    ]
    med = det.median_result(legacy)
    # len//2 = 1, sorted([10,30])[1] = 30.
    assert med["prefill"]["tflops"] == 30.0
    assert med["decode"]["tflops"] == 0.0  # legacy → decode zeroed


# --------------------------------------------------------------------------- #
# format_mismatches_for_prompt
# --------------------------------------------------------------------------- #

def test_format_mismatches_empty():
    out = det.format_mismatches_for_prompt([])
    assert "agreed" in out


def test_format_mismatches_new_shape():
    m = [{
        "batch_size": 1, "seq_len": 512,
        "values": [
            {"prefill": {"tflops": 10.0, "access_gb": 2.0},
             "decode":  {"tflops": 1.0, "access_gb": 0.5}},
            {"prefill": {"tflops": 20.0, "access_gb": 2.0},
             "decode":  {"tflops": 1.0, "access_gb": 0.5}},
        ],
        "spread": {
            "prefill": {"tflops": 10.0, "access_gb": 0.0},
            "decode":  {"tflops": 0.0, "access_gb": 0.0},
        },
    }]
    text = det.format_mismatches_for_prompt(m)
    assert "Mismatches" in text
    assert "pre.t=10" in text and "pre.t=20" in text


# --------------------------------------------------------------------------- #
# normalize_graph + validate_graph_structure smoke
# --------------------------------------------------------------------------- #

def test_normalize_graph_sectioned_passthrough():
    g = {"sections": [{"id": "x", "kind": "layer_template",
                        "graph": {"nodes": [], "edges": []}}]}
    out = det.normalize_graph(g)
    assert out is g  # sectioned graph returned as-is


def test_normalize_graph_wraps_flat():
    g = {"nodes": [{"id": "a"}], "edges": []}
    out = det.normalize_graph(g)
    assert det.is_sectioned(out)
    assert len(out["sections"]) == 1
    assert out["sections"][0]["graph"]["nodes"] == [{"id": "a"}]


def test_normalize_graph_rejects_garbage():
    with pytest.raises(ValueError):
        det.normalize_graph({"weird": True})


def test_validate_graph_structure_minimal_valid():
    g = {
        "nodes": [{
            "id": "n1", "purpose": "p", "op": "matmul",
            "inputs": [{"name": "x", "shape": ["B", "S"]}],
            "outputs": [{"name": "y", "shape": ["B", "S"]}],
        }],
        "edges": [],
    }
    ok, errs = det.validate_graph_structure(g)
    assert ok, errs


def test_validate_graph_structure_missing_node_field():
    g = {"nodes": [{"id": "n1"}], "edges": []}  # missing purpose/op/...
    ok, errs = det.validate_graph_structure(g)
    assert not ok
    assert any("purpose" in e for e in errs)


def test_validate_graph_structure_duplicate_id():
    g = {
        "nodes": [
            {"id": "n1", "purpose": "p", "op": "o", "inputs": [], "outputs": []},
            {"id": "n1", "purpose": "p", "op": "o", "inputs": [], "outputs": []},
        ],
        "edges": [],
    }
    ok, errs = det.validate_graph_structure(g)
    assert not ok
    assert any("duplicate" in e for e in errs)


def test_section_node_count():
    g = {"sections": [
        {"id": "a", "kind": "layer_template",
         "graph": {"nodes": [{}, {}, {}], "edges": []}},
        {"id": "b", "kind": "layer_template",
         "graph": {"nodes": [{}], "edges": []}},
    ]}
    assert det.section_node_count(g) == 4
