"""Tests for OperatorContract parsing/validation and shape-sweep DSL."""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.contract import (
    ContractError,
    OperatorContract,
    eval_expr,
)


RMSNORM_CONTRACT = {
    "name": "RMSNorm",
    "entrypoint": "rmsnorm_kernel",
    "language": "triton",
    "shape_mode": "general",
    "inputs": [
        {"name": "X", "dtype": "fp16", "shape": ["B", "S", "H"]},
        {"name": "W", "dtype": "fp16", "shape": ["H"]},
    ],
    "outputs": [
        {"name": "Y", "dtype": "fp16", "shape": ["B", "S", "H"]},
    ],
    "shapes": {"B": 1, "S": [2048, 8192], "H": [128, 512]},
    "numerics": {"abs_tol": 1e-3, "rel_tol": 1e-2},
    "constraints": "deterministic",
}


def test_general_sweep_cartesian_product():
    c = OperatorContract.load(RMSNORM_CONTRACT)
    cases = c.generate_cases()
    assert len(cases) == 2 * 2  # S x H
    dims = {(cs.dims["S"], cs.dims["H"]) for cs in cases}
    assert dims == {(2048, 128), (2048, 512), (8192, 128), (8192, 512)}
    # every case resolves all tensor shapes
    for cs in cases:
        assert cs.shapes["X"] == [cs.dims["B"], cs.dims["S"], cs.dims["H"]]
        assert cs.shapes["W"] == [cs.dims["H"]]
        assert cs.shapes["Y"] == cs.shapes["X"]
        assert cs.id


def test_general_single_point_no_sweep():
    data = dict(RMSNORM_CONTRACT)
    data["shapes"] = {"B": 1, "S": 2048, "H": 512}
    c = OperatorContract.load(data)
    cases = c.generate_cases()
    assert len(cases) == 1
    assert cases[0].shapes["X"] == [1, 2048, 512]


def test_targeted_uses_target_shapes():
    data = dict(RMSNORM_CONTRACT)
    data["shape_mode"] = "targeted"
    data["target_shapes"] = [
        {"B": 1, "S": 4096, "H": 1024},
        {"B": 2, "S": 1024, "H": 512},
    ]
    c = OperatorContract.load(data)
    cases = c.generate_cases()
    assert len(cases) == 2
    assert cases[0].shapes["X"] == [1, 4096, 1024]
    assert cases[1].shapes["X"] == [2, 1024, 512]


def test_targeted_falls_back_to_single_point():
    data = dict(RMSNORM_CONTRACT)
    data["shape_mode"] = "targeted"
    data["target_shapes"] = []
    c = OperatorContract.load(data)
    cases = c.generate_cases()
    assert len(cases) == 1
    assert cases[0].dims["S"] == 2048  # first of sweep


def test_shape_expression_arithmetic():
    data = dict(RMSNORM_CONTRACT)
    data["inputs"] = [
        {"name": "X", "dtype": "fp16", "shape": ["B", "S", "2*H"]},
    ]
    data["shapes"] = {"B": 1, "S": 4, "H": 256}
    c = OperatorContract.load(data)
    cases = c.generate_cases()
    assert cases[0].shapes["X"] == [1, 4, 512]


def test_metric_flops_and_bytes():
    data = dict(RMSNORM_CONTRACT)
    data["shapes"] = {"B": 1, "S": 8, "H": 64}
    data["metric"] = {
        "flops": "6 * B * S * H",
        "bytes": "2 * B * S * H + 2 * H",
    }
    c = OperatorContract.load(data)
    (case,) = c.generate_cases()
    assert case.flops == 6 * 1 * 8 * 64
    assert case.bytes == 2 * 1 * 8 * 64 + 2 * 64


def test_safe_eval_rejects_attribute_escape():
    # attribute chains must NOT work — eval_expr must be safe from RCE.
    with pytest.raises(ContractError):
        eval_expr("().__class__.__bases__[0].__subclasses__()", {})
    with pytest.raises(ContractError):
        eval_expr("__import__('os').system('echo pwned')", {})
    assert eval_expr("2 * 3 + 4", {}) == 10
    assert eval_expr("S // 8", {"S": 2048}) == 256
    assert eval_expr("H + 16", {"H": 128}) == 144


@pytest.mark.parametrize("mut,err", [
    ({"name": ""}, "name"),
    ({"language": "cuda"}, "language"),
    ({"shape_mode": "weird"}, "shape_mode"),
    ({"inputs": []}, "inputs"),
    ({"outputs": []}, "outputs"),
    ({"shapes": {}}, "shapes"),
    ({"shapes": {"B": 0}}, "positive integer"),
    ({"shapes": {"B": [-1]}}, "positive integer"),
    ({"numerics": {"abs_tol": -1}}, ">= 0"),
])
def test_invalid_contracts_rejected(mut, err):
    data = dict(RMSNORM_CONTRACT)
    for k, v in mut.items():
        data[k] = v
    with pytest.raises(ContractError) as excinfo:
        OperatorContract.load(data)
    assert err in str(excinfo.value)


def test_duplicate_tensor_names_rejected():
    data = dict(RMSNORM_CONTRACT)
    data["outputs"] = [{"name": "X", "dtype": "fp16", "shape": ["B", "S", "H"]}]
    with pytest.raises(ContractError):
        OperatorContract.load(data)


def test_yaml_round_trip():
    c = OperatorContract.load(RMSNORM_CONTRACT)
    c2 = OperatorContract.load(c.to_yaml())
    assert c2.to_dict() == c.to_dict()


def test_tensor_numel():
    c = OperatorContract.load(RMSNORM_CONTRACT)
    x = c.tensor_by_name("X")
    assert x is not None
    assert x.numel({"B": 1, "S": 2048, "H": 512}) == 1 * 2048 * 512
