"""Hardware profile lookup — build targets for the K100 DCU.

Reads ``hardware_profiles.yaml`` (shipped with the package). A profile names the
HIP arch, hipcc binary, and a nominal peak FLOPs figure used only to estimate
utilization in the perf view. Lookups never raise for an unknown device — they
fall back to the ``default`` profile so the orchestrator can always pick a build
target.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Dict, Optional

_YAML_PATH = Path(__file__).resolve().parent / "hardware_profiles.yaml"


@functools.lru_cache(maxsize=1)
def load_profiles() -> Dict[str, Any]:
    import yaml
    with open(_YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("profiles", {}) or {}


@functools.lru_cache(maxsize=1)
def _default_name() -> str:
    import yaml
    with open(_YAML_PATH, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("default", "K100")


def hardware_profile(device: Optional[str] = None) -> Dict[str, Any]:
    """Return the profile for ``device`` (default: the config's default)."""
    profiles = load_profiles()
    name = device or _default_name()
    if name in profiles:
        return dict(profiles[name], name=name)
    # unknown device -> fall back to default (never hard-fail)
    default = _default_name()
    return dict(profiles.get(default, {}), name=name, fallback=True)


__all__ = ["load_profiles", "hardware_profile"]
