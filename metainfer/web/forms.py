"""Task type definitions → frontend form schema.

Each task type ships a ``form.yaml`` inside its self-contained task package
at ``metainfer/tasks/<task_pkg>/form.yaml`` (see CLAUDE.md for the layout).
Stub types without a full package (e.g. ``opt-kernel``) may still ship a
YAML under ``<repo>/tasks/<type>.yaml``; both locations are supported.

Schema shape (compatible with the legacy questions.yaml format):

    - key: target_model           # unique field key
      question: "Enter the model weight path:"
      header: "Target model"       # short label, <= 12 chars
      required: true
      multi: false                 # omit for free-form text
      options:                     # omit for free-form text
        - label: "Hygon K100AI"
          description: "..."
      default: "throughput"
      # NEW: explicit form widget hint. If omitted, the type is inferred
      # from multi / options.
      form: text|textarea|select|multiselect|file|number

This module normalizes the YAML into a stable JSON schema the frontend
form renderer consumes:

    {
      "type": "gen-infer-framework",
      "label": "Build inference framework",
      "description": "...",
      "fields": [
        {
          "key": "target_model",
          "label": "Target model",
          "help": "Enter the model weight path: ...",
          "type": "file",            # canonical widget type
          "required": true,
          "default": null,
          "options": null | [{"label":..., "description":...}, ...],
          "override_component": null | "file-picker" | "shape-input" | ...
        }
      ]
    }

The renderer walks ``fields`` generically. A non-null ``override_component``
tells it to delegate to a task-specific widget instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..orchestrator import paths as _orch_paths


# Top-level task-type metadata. Augments the per-field YAML with a
# friendly label + description for the task-type picker.
TASK_TYPE_META: Dict[str, Dict[str, str]] = {
    "gen-infer-framework": {
        "label": "Build inference framework",
        "description": (
            "Build a minimal, model-specific inference framework with an "
            "OpenAI-compatible HTTP API from scratch."
        ),
    },
    "opt-kernel": {
        "label": "Optimize GPU kernel",
        "description": (
            "Optimize an existing GPU kernel (attention, GEMM, norm, RoPE) "
            "for a specific shape and platform."
        ),
    },
    "port-model": {
        "label": "Port model into framework",
        "description": (
            "Port a model architecture into an existing inference framework "
            "(vLLM / SGLang / TensorRT-LLM)."
        ),
    },
    "calc-theoretical-value": {
        "label": "Calc theoretical FLOPs / mem-traffic",
        "description": (
            "Analyze a model + inference framework and compute the "
            "theoretical FLOPs and global-memory traffic of one forward "
            "pass, with per-node breakdown and an interactive visualization."
        ),
    },
}


def _infer_field_type(entry: Dict[str, Any]) -> str:
    """Canonical widget type. Explicit ``form`` wins; otherwise infer
    from multi/options presence."""
    explicit = entry.get("form")
    if explicit:
        return explicit
    multi = bool(entry.get("multi"))
    has_options = bool(entry.get("options"))
    if multi:
        return "multiselect"
    if has_options:
        return "select"
    return "text"


def _normalize_field(entry: Dict[str, Any]) -> Dict[str, Any]:
    key = entry.get("key")
    if not key:
        raise ValueError(f"field missing 'key': {entry!r}")
    ftype = _infer_field_type(entry)
    out: Dict[str, Any] = {
        "key": key,
        "label": entry.get("header") or key,
        "help": entry.get("question") or "",
        "type": ftype,
        "required": bool(entry.get("required", False)),
        "default": entry.get("default"),
        "options": None,
        # Override hook for task-specific widgets. Read from the YAML so
        # task authors can opt a field into a custom renderer without
        # code changes to the generic form.
        "override_component": entry.get("override_component"),
    }
    opts = entry.get("options")
    if opts:
        out["options"] = [
            {"label": o.get("label", ""), "description": o.get("description", "")}
            for o in opts
        ]
    return out


def _form_yaml_for_task_type(task_type: str) -> Optional[Path]:
    """Resolve the form.yaml path for a task type.

    Priority:
      1. If a WebPlugin is registered for this task type, look for
         ``form.yaml`` at the plugin's task-package root (parent of
         ``frontend_dir``). This is the canonical self-contained layout.
      2. Fallback: legacy ``<repo>/tasks/<task_type>.yaml`` for stub
         types that don't have a full task package yet (e.g. opt-kernel,
         port-model).

    Returns ``None`` if no YAML exists.
    """
    from .registry import get as _get_plugin
    plugin = _get_plugin(task_type)
    if plugin is not None and plugin.frontend_dir is not None:
        cand = plugin.frontend_dir.parent / "form.yaml"
        if cand.exists():
            return cand
    legacy = _orch_paths.question_file(task_type)
    if legacy.exists():
        return legacy
    return None


def load_form_schema(task_type: str) -> Optional[Dict[str, Any]]:
    """Return the normalized form schema for ``task_type``, or None if
    no YAML exists for it."""
    yaml_path = _form_yaml_for_task_type(task_type)
    if yaml_path is None:
        return None
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or []
    if not isinstance(raw, list):
        raise ValueError(f"{yaml_path}: expected a list of field entries")
    fields = [_normalize_field(e) for e in raw]
    meta = TASK_TYPE_META.get(task_type, {})
    return {
        "type": task_type,
        "label": meta.get("label", task_type),
        "description": meta.get("description", ""),
        "fields": fields,
    }


def list_task_types() -> List[Dict[str, str]]:
    """Return a compact list of available task types for the task-type
    picker in the New Task form.

    Combines two sources:
      - Every registered WebPlugin (full task packages under
        ``metainfer/tasks/<pkg>/`` whose ``frontend_dir.parent/form.yaml``
        exists).
      - Every ``<repo>/tasks/<type>.yaml`` stub whose type isn't already
        covered by a plugin (so legacy stubs like opt-kernel still appear).
    """
    out: List[Dict[str, str]] = []
    seen: set = set()
    # Full task packages first — these are the primary, peer-registered types.
    from .registry import all_plugins as _all_plugins
    for plugin in _all_plugins():
        if plugin.frontend_dir is None:
            continue
        cand = plugin.frontend_dir.parent / "form.yaml"
        if not cand.exists():
            continue
        tt = plugin.type
        if tt in seen:
            continue
        seen.add(tt)
        meta = TASK_TYPE_META.get(tt, {})
        out.append({
            "id": tt,
            "label": meta.get("label", tt),
            "description": meta.get("description", ""),
        })
    # Legacy stub types: any YAML under <repo>/tasks/ whose type isn't
    # already covered by a plugin.
    legacy_dir = _orch_paths.tasks_dir()
    if legacy_dir.exists():
        for p in sorted(legacy_dir.glob("*.yaml")):
            tt = p.stem
            if tt in seen:
                continue
            seen.add(tt)
            meta = TASK_TYPE_META.get(tt, {})
            out.append({
                "id": tt,
                "label": meta.get("label", tt),
                "description": meta.get("description", ""),
            })
    return out


def validate_submission(task_type: str, answers: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a form submission against the schema. Returns a dict of
    ``{ok: bool, errors: {field_key: message}}``."""
    schema = load_form_schema(task_type)
    if schema is None:
        return {"ok": False, "errors": {"_": f"unknown task type {task_type!r}"}}
    errors: Dict[str, str] = {}
    for field in schema["fields"]:
        key = field["key"]
        val = answers.get(key)
        if field["required"] and (
            val is None
            or (isinstance(val, str) and val.strip() == "")
            or (isinstance(val, list) and len(val) == 0)
        ):
            errors[key] = "this field is required"
            continue
        if val is None:
            continue
        # Type-check against declared widget type
        t = field["type"]
        if t in ("select",) and field["options"]:
            valid_labels = {o["label"] for o in field["options"]}
            if val not in valid_labels:
                errors[key] = f"must be one of: {sorted(valid_labels)}"
        elif t == "multiselect" and field["options"]:
            valid_labels = {o["label"] for o in field["options"]}
            if not isinstance(val, list):
                errors[key] = "expected a list"
            else:
                bad = [v for v in val if v not in valid_labels]
                if bad:
                    errors[key] = f"unknown options: {bad}"
    return {"ok": not errors, "errors": errors}
