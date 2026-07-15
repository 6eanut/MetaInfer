"""Tests for the form schema loader and the web plugin registry.

User-facing requirement (from CLAUDE.md): "新增一个任务类型 X ... 验证: -
``all_plugins()`` 应包含 'X' - 跑 ``python -m pytest tests/`` 全绿".

These tests cover the registry's public contract and the form schema
loader. We don't spin a real FastAPI app here — that's in
``test_web_app_calc.py``.
"""

from __future__ import annotations

import pytest

from metainfer.server import forms
from metainfer.server import registry
import metainfer.tasks  # noqa: F401 — side effect: auto-discover


# --------------------------------------------------------------------------- #
# Plugin registry — auto-discovery contract
# --------------------------------------------------------------------------- #

def test_calc_value_plugin_registered():
    """The CLAUDE.md doc says: ``all_plugins()`` should include
    'calc-theoretical-value'."""
    types = [p.type for p in registry.all_plugins()]
    assert "calc-theoretical-value" in types


def test_every_task_type_has_a_web_plugin():
    """Each registered task type must own a web plugin package — peer
    design, no "base" vs "extension" split. Both calc-theoretical-value
    and gen-infer-framework register a WebPlugin with at minimum a
    detail_view_module + frontend_dir + qa_config + label/description.

    sys-shell is excluded — it's the system shell, not a task type.
    """
    types = {p.type for p in registry.all_plugins()}
    assert "calc-theoretical-value" in types
    assert "gen-infer-framework" in types
    for p in registry.all_plugins():
        if p.type == "sys-shell":
            continue  # shell is not a task type
        # detail_view_module + frontend_dir are how the shell dispatches
        # the body. Every plugin needs them.
        assert p.detail_view_module, f"{p.type} missing detail_view_module"
        assert p.frontend_dir, f"{p.type} missing frontend_dir"
        assert p.frontend_dir.exists(), f"{p.type} frontend_dir missing on disk"
        # detail_view_module is an importmap key like "app/<stem>". The
        # corresponding file must exist under frontend_dir — the auto-
        # discovery in create_app registers it under that key. (Plugins
        # may also list it explicitly in importmap_entries to override.)
        stem = p.detail_view_module.split("/", 1)[-1]
        assert (p.frontend_dir / f"{stem}.js").exists(), (
            f"{p.type} detail view file {stem}.js not in frontend_dir"
        )
        # label + description come from the plugin now (no central dict).
        assert p.label, f"{p.type} missing label"
        assert p.description, f"{p.type} missing description"


def test_get_returns_none_for_unknown_type():
    assert registry.get("does-not-exist") is None


def test_register_rejects_duplicate():
    """Duplicate registration must raise — guards against plugin packages
    accidentally double-importing."""
    p = registry.WebPlugin(type="__test_dup__", label="t", description="t")
    registry.register(p)
    try:
        with pytest.raises(ValueError):
            registry.register(registry.WebPlugin(
                type="__test_dup__", label="t", description="t"))
    finally:
        # Clean up so other tests aren't polluted.
        registry._REGISTRY.pop("__test_dup__", None)


def test_all_plugins_returns_copy():
    """Mutating the returned list must not affect the registry."""
    snap1 = registry.all_plugins()
    snap1.clear()
    snap2 = registry.all_plugins()
    assert len(snap2) >= 1  # still has the production plugins


# --------------------------------------------------------------------------- #
# Form schema loading
# --------------------------------------------------------------------------- #

def test_load_form_schema_known_type():
    schema = forms.load_form_schema("calc-theoretical-value")
    assert schema is not None
    assert schema["type"] == "calc-theoretical-value"
    assert schema["label"]
    assert isinstance(schema["fields"], list) and schema["fields"]


def test_load_form_schema_unknown_type_returns_none():
    assert forms.load_form_schema("nonexistent-type") is None


def test_list_task_types_includes_registered_plugins():
    """``list_task_types`` emits one entry per registered plugin whose
    task package ships a form.yaml."""
    out = forms.list_task_types()
    ids = [t["id"] for t in out]
    assert set(ids) >= {
        "gen-infer-framework",
        "calc-theoretical-value",
    }
    # Every entry has label + description.
    for t in out:
        assert t["label"] and t["description"]


# --------------------------------------------------------------------------- #
# Field type inference
# --------------------------------------------------------------------------- #

def test_infer_field_type_explicit_form_wins():
    out = forms._infer_field_type({"form": "select", "multi": True})
    assert out == "select"


def test_infer_field_type_multi_implies_multiselect():
    assert forms._infer_field_type({"multi": True}) == "multiselect"


def test_infer_field_type_options_implies_select():
    assert forms._infer_field_type({"options": [{"label": "x"}]}) == "select"


def test_infer_field_type_text_fallback():
    assert forms._infer_field_type({}) == "text"


def test_normalize_field_requires_key():
    with pytest.raises(ValueError):
        forms._normalize_field({"header": "no key!"})


def test_normalize_field_basic_shape():
    out = forms._normalize_field({
        "key": "model_name",
        "header": "Model name",
        "question": "Which model?",
        "required": True,
    })
    assert out["key"] == "model_name"
    assert out["label"] == "Model name"
    assert out["help"] == "Which model?"
    assert out["type"] == "text"
    assert out["required"] is True
    assert out["options"] is None


# --------------------------------------------------------------------------- #
# Submission validation
# --------------------------------------------------------------------------- #

def test_validate_submission_unknown_type():
    out = forms.validate_submission("nonexistent-type", {})
    assert out["ok"] is False
    assert "_" in out["errors"]


def test_validate_submission_missing_required_field():
    """calc-theoretical-value has required fields; submit an empty form."""
    out = forms.validate_submission("calc-theoretical-value", {})
    assert out["ok"] is False
    # At least one required field should be flagged.
    assert out["errors"]
