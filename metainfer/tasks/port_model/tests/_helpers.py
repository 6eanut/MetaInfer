"""Test helpers for port-model."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


def make_requirements(
    task_id: str = "pm-1",
    *,
    model_params_path: str = "/tmp/fake/model",
    target_framework_dir: str = "/tmp/fake/target_fw",
    reference_sources: Optional[List[Dict[str, Any]]] = None,
    user_notes: str = "test notes",
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a flat (post-WebUI-flatten) requirements.json dict."""
    req: Dict[str, Any] = {
        "task_id": task_id,
        "task_type": "port-model",
        "created_at": 0.0,
        "label": "Port Model",
        "model_params_path": model_params_path,
        "target_framework_dir": target_framework_dir,
        "reference_sources": reference_sources if reference_sources is not None else [],
        "user_notes": user_notes,
    }
    if overrides:
        req.update(overrides)
    return req


def make_minimal_config() -> Dict[str, Any]:
    """A minimal HuggingFace-style config.json fixture."""
    return {
        "architectures": ["TestModelForCausalLM"],
        "model_type": "test_model",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 8192,
        "vocab_size": 32000,
        "torch_dtype": "float16",
        "tie_word_embeddings": False,
    }
