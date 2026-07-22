"""Tests for form.yaml + plugin registration."""

from __future__ import annotations

from pathlib import Path

import yaml

from metainfer.orchestrator.tasks import all_tasks, get_task
from metainfer.server.forms import load_form_schema, validate_submission
from metainfer.server.registry import all_plugins, get


PLUGIN_TYPE = "port-model"


def test_plugin_registered():
    types = [p.task_type for p in all_tasks()]
    assert PLUGIN_TYPE in types


def test_web_plugin_registered():
    types = [p.type for p in all_plugins()]
    assert PLUGIN_TYPE in types
    plugin = get(PLUGIN_TYPE)
    assert plugin.label and "Port" in plugin.label
    assert plugin.detail_view_module == "app/pm-detail"
    assert plugin.build_router is not None


def test_task_plugin_descriptor_fields():
    p = get_task(PLUGIN_TYPE)
    assert p.cli_module.endswith(".cli")
    assert p.phases_module.endswith(".phases")
    assert isinstance(p.diagnostic_globs, tuple)


def test_form_yaml_exists_and_loads():
    schema = load_form_schema(PLUGIN_TYPE)
    assert schema is not None
    keys = {f["key"] for f in schema["fields"]}
    assert {
        "model_params_path",
        "target_framework_dir",
        "reference_sources",
        "user_notes",
        "token_budget_max_cost_usd",
    } <= keys


def test_form_required_fields_marked():
    schema = load_form_schema(PLUGIN_TYPE)
    required = {f["key"] for f in schema["fields"] if f["required"]}
    assert {"model_params_path", "target_framework_dir"} <= required


def test_form_reference_sources_uses_override():
    schema = load_form_schema(PLUGIN_TYPE)
    fields = {f["key"]: f for f in schema["fields"]}
    assert fields["reference_sources"]["override_component"] == "kv-list-path-notes"


def test_validate_submission_rejects_missing_required():
    result = validate_submission(PLUGIN_TYPE, {})
    assert result["ok"] is False
    for k in ("model_params_path", "target_framework_dir"):
        assert k in result["errors"]


def test_validate_submission_accepts_minimal():
    """reference_sources is optional; the two path fields are required."""
    result = validate_submission(PLUGIN_TYPE, {
        "model_params_path": "/tmp/m",
        "target_framework_dir": "/tmp/t",
    })
    assert result["ok"] is True, result["errors"]


def test_validate_submission_accepts_reference_list():
    result = validate_submission(PLUGIN_TYPE, {
        "model_params_path": "/tmp/m",
        "target_framework_dir": "/tmp/t",
        "reference_sources": [
            {"path": "/tmp/r1", "notes": "vllm reference"},
            {"path": "/tmp/r2", "notes": ""},
        ],
    })
    assert result["ok"] is True, result["errors"]


def test_form_yaml_is_valid_yaml():
    p = Path(__file__).resolve().parent.parent / "form.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 4
