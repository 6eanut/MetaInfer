"""calc-theoretical-value web plugin definition."""

from __future__ import annotations

from pathlib import Path

from metainfer.web.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG
from .routes import register_routes

PLUGIN_TYPE = "calc-theoretical-value"
# Frontend assets live in the task package's static/ dir (sibling of this
# package). Resolve via Path(__file__) so the package is relocatable.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"
_STATIC_PREFIX = f"/static/plugins/{PLUGIN_TYPE}"

# Importmap entries for this plugin's bundled frontend. ``CACHE_BUST`` is
# a placeholder the WebUI replaces with the current static-dir mtime token
# before serving index.html (see metainfer/web/app.py::create_app).
_IMPORTMAP_ENTRIES = {
    "app/calc-detail":      f"{_STATIC_PREFIX}/calc-detail.js?v=CACHE_BUST",
    "app/calc-viz":         f"{_STATIC_PREFIX}/calc-viz.js?v=CACHE_BUST",
    "app/calc-viz-tab":     f"{_STATIC_PREFIX}/calc-viz-tab.js?v=CACHE_BUST",
    "app/calc-audit-panel": f"{_STATIC_PREFIX}/calc-audit-panel.js?v=CACHE_BUST",
    "app/calc-cell-modal":  f"{_STATIC_PREFIX}/calc-cell-modal.js?v=CACHE_BUST",
    "app/calc-rough-panel": f"{_STATIC_PREFIX}/calc-rough-panel.js?v=CACHE_BUST",
    "app/calc-iterations":  f"{_STATIC_PREFIX}/calc-iterations.js?v=CACHE_BUST",
    "app/qa-modal":         f"{_STATIC_PREFIX}/qa-modal.js?v=CACHE_BUST",
}

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    register_routes=register_routes,
    detail_view_module="app/calc-detail",
    detail_view_export="default",
    qa_config=_QA_CONFIG,
    frontend_dir=_FRONTEND_DIR,
    importmap_entries=_IMPORTMAP_ENTRIES,
)

register(plugin)
