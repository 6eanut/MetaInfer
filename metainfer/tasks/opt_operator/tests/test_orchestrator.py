"""Tests for orchestrator wiring helpers (contract/model-map/json/hardware/guidance)."""

from __future__ import annotations

import json

import pytest

from metainfer.tasks.opt_operator.orchestrator.guidance import GuidanceStore
from metainfer.tasks.opt_operator.orchestrator.hardware import hardware_profile, load_profiles
from metainfer.tasks.opt_operator.orchestrator.orchestrator import (
    OrchestratorError,
    _extract_json,
    _model_map,
    contract_from_req,
    kernel_source_from_req,
)
from metainfer.tasks.opt_operator.tests._helpers import contract_dict


# --------------------------------------------------------------------------- #
# contract_from_req
# --------------------------------------------------------------------------- #

def test_contract_from_inline_dict(tmp_path):
    req = {"operator_contract": contract_dict(name="InlineOp")}
    c = contract_from_req(req, tmp_path)
    assert c.name == "InlineOp"
    assert len(c.generate_cases()) > 0


def test_contract_from_yaml_string(tmp_path):
    import yaml
    req = {"operator_contract": yaml.safe_dump(contract_dict(name="YamlOp"))}
    c = contract_from_req(req, tmp_path)
    assert c.name == "YamlOp"


def test_contract_from_file(tmp_path):
    p = tmp_path / "contract.yaml"
    import yaml
    p.write_text(yaml.safe_dump(contract_dict(name="FileOp")), encoding="utf-8")
    req = {"operator_contract": str(p)}
    c = contract_from_req(req, tmp_path)
    assert c.name == "FileOp"


def test_contract_missing_raises(tmp_path):
    with pytest.raises(OrchestratorError):
        contract_from_req({}, tmp_path)


def test_contract_bad_raises(tmp_path):
    with pytest.raises(OrchestratorError):
        contract_from_req({"operator_contract": "not: [valid"}, tmp_path)


# --------------------------------------------------------------------------- #
# kernel_source_from_req
# --------------------------------------------------------------------------- #

def test_kernel_source_mode_source(tmp_path):
    src = tmp_path / "k.py"
    src.write_text("def k(): pass", encoding="utf-8")
    req = {"input_mode": "source", "kernel_source": str(src),
           "kernel_language": "triton"}
    source, language = kernel_source_from_req(req, tmp_path)
    assert language == "triton"
    assert "def k" in source


def test_kernel_source_mode_spec(tmp_path):
    req = {"input_mode": "spec"}
    assert kernel_source_from_req(req, tmp_path) == (None, None)


def test_kernel_source_missing_raises(tmp_path):
    with pytest.raises(OrchestratorError):
        kernel_source_from_req({"input_mode": "source"}, tmp_path)


# --------------------------------------------------------------------------- #
# _model_map
# --------------------------------------------------------------------------- #

def test_model_map_precedence(monkeypatch):
    monkeypatch.delenv("METAINFER_MODEL_STRONG", raising=False)
    monkeypatch.delenv("METAINFER_MODEL_CHEAP", raising=False)
    mm = _model_map({}, "cli-strong", "cli-cheap", "default")
    assert mm == {"strong": "cli-strong", "cheap": "cli-cheap"}


def test_model_map_env_fallback(monkeypatch):
    monkeypatch.setenv("METAINFER_MODEL_STRONG", "env-strong")
    monkeypatch.setenv("METAINFER_MODEL_CHEAP", "env-cheap")
    mm = _model_map({}, None, None, "default")
    assert mm == {"strong": "env-strong", "cheap": "env-cheap"}


def test_model_map_default_fallback(monkeypatch):
    monkeypatch.delenv("METAINFER_MODEL_STRONG", raising=False)
    monkeypatch.delenv("METAINFER_MODEL_CHEAP", raising=False)
    mm = _model_map({}, None, None, "default")
    assert mm == {"strong": "default", "cheap": "default"}


# --------------------------------------------------------------------------- #
# _extract_json
# --------------------------------------------------------------------------- #

def test_extract_json_balanced():
    text = "here's my answer: {\"language\": \"triton\", \"source\": \"x\"} trailing"
    assert _extract_json(text) == {"language": "triton", "source": "x"}


def test_extract_json_nested_string():
    text = '{"a": "brace { inside", "b": 2}'
    assert _extract_json(text)["b"] == 2


def test_extract_json_no_json():
    assert _extract_json("no braces here") == {"raw": "no braces here"}


# --------------------------------------------------------------------------- #
# hardware
# --------------------------------------------------------------------------- #

def test_hardware_default_profile():
    p = hardware_profile()
    assert p["arch"] == "gfx928"
    assert p["name"] in ("K100",)


def test_hardware_unknown_falls_back():
    p = hardware_profile("NOPE")
    assert p.get("fallback") is True
    assert p["arch"] == "gfx928"


def test_load_profiles():
    profs = load_profiles()
    assert "K100" in profs


# --------------------------------------------------------------------------- #
# guidance
# --------------------------------------------------------------------------- #

def test_guidance_store_roundtrip(tmp_path):
    gs = GuidanceStore(tmp_path / "guidance.json")
    assert gs.latest() is None
    gs.add("advice one", source="agent", cycle=1)
    gs.add("advice two", source="human")
    assert len(gs.all()) == 2
    assert gs.latest() == "advice two"


def test_guidance_store_survives_restart(tmp_path):
    p = tmp_path / "guidance.json"
    GuidanceStore(p).add("hello")
    fresh = GuidanceStore(p)
    assert fresh.latest() == "hello"
