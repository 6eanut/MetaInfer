"""WebPlugin descriptor for port-model."""

from __future__ import annotations

from pathlib import Path
from typing import List

from metainfer.server._helpers import workspace_dir_for
from metainfer.server.registry import WebPlugin, register

from ._qa import QA_CONFIG
from .routes import build_router

PLUGIN_TYPE = "port-model"
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"
_STATIC_PREFIX = f"/static/plugins/{PLUGIN_TYPE}"

# Inherit the shell's default state-graph / charts / iterations-table widgets.
# We only ship our own task-detail view (app/pm-detail) plus a custom
# form widget module. The "app/form-overrides/<type>" key is picked up
# by the shell's form-overrides-loader, which dynamically imports this
# module before the SPA first renders so ``registerFormWidget(...)`` has
# run by the time the NewTaskView opens.
_IMPORTMAP_ENTRIES: dict = {
    "app/form-overrides/port-model": f"{_STATIC_PREFIX}/form-overrides.js?v=CACHE_BUST",
}


def _extra_watch_paths(entry) -> List[Path]:
    """Refresh the audit panel when per-phase summary.md or memory/* change."""
    wd = workspace_dir_for(entry)
    return [
        wd / "p1" / "summary.md",
        wd / "p2" / "summary.md",
        wd / "p3" / "summary.md",
        wd / "p4" / "summary.md",
        wd / "p5" / "summary.md",
        wd / "p6" / "summary.md",
        wd / "memory" / "p1_weight_analysis.md",
        wd / "memory" / "p3_consolidated_spec.md",
    ]


plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="Port Model (6-agent)",
    description=(
        "Port a model to a target inference framework via a 6-agent "
        "pipeline: weight analysis → framework analysts → architect "
        "review → minimal framework → minimal framework verify → port."
    ),
    detail_view_module="app/pm-detail",
    qa_config=QA_CONFIG,
    build_router=build_router,
    frontend_dir=_FRONTEND_DIR,
    importmap_entries=_IMPORTMAP_ENTRIES,
    extra_stylesheets=["pm.css"],
    extra_watch_paths=_extra_watch_paths,
)

register(plugin)
