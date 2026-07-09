"""Oracle registry.

Oracles are looked up by ``task_type``. Adding a new oracle:

1. Subclass :class:`metainfer.oracles.base.Oracle` and set ``task_type``.
2. Register it in ``_REGISTRY`` below.
3. Add a branch in :mod:`metainfer.pipeline` if the artifact contract differs.
"""

from __future__ import annotations

from typing import Dict, Optional, Type

from .base import Oracle, OracleCaseResult, OracleResult


# Oracle classes are imported lazily on first lookup to avoid pulling heavy
# deps (e.g. yaml) at package import time.
_REGISTRY: Dict[str, str] = {
    "gen-infer-framework": "metainfer.oracles.infer_framework.harness:InferFrameworkOracle",
}


def get_oracle(task_type: str) -> Optional[Oracle]:
    """Return a fresh instance of the oracle for ``task_type``, or ``None``."""
    target = _REGISTRY.get(task_type)
    if target is None:
        return None
    module_path, _, cls_name = target.partition(":")
    import importlib
    mod = importlib.import_module(module_path)
    cls: Type[Oracle] = getattr(mod, cls_name)
    return cls()


def available_oracles() -> Dict[str, str]:
    """Return the registered task_type → oracle class mapping (for docs/UI)."""
    return dict(_REGISTRY)


__all__ = ["Oracle", "OracleResult", "OracleCaseResult", "get_oracle", "available_oracles"]
