"""Task type definitions → frontend form schema.

Each task type ships a YAML file under ``<repo>/tasks/<type>.yaml`` with
this shape (compatible with the legacy questions.yaml format):

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


def load_form_schema(task_type: str) -> Optional[Dict[str, Any]]:
    """Return the normalized form schema for ``task_type``, or None if
    no YAML exists for it."""
    yaml_path = _orch_paths.question_file(task_type)
    if not yaml_path.exists():
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
    picker in the New Task form."""
    out: List[Dict[str, str]] = []
    # Walk TASK_TYPES for stable ordering; only include ones whose YAML exists.
    for tt in _orch_paths.TASK_TYPES:
        yaml_path = _orch_paths.question_file(tt)
        if not yaml_path.exists():
            continue
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
