"""gen-infer-framework web plugin definition.

Registers the detail view + QA pathsolver for the multi-iteration
ABCDEF pipeline. Peer of
:mod:`metainfer.tasks.calc_value.web_server_handler.plugin`.

Unlike calc_value, gen-infer-framework has no custom HTTP routes today
— the generic ``/api/tasks/<id>/...`` endpoints (iterations, timeline,
charts, state-graph, agents, token-budget) cover its needs because its
orchestrator writes to the shared on-disk layout. If/when gf-specific
routes are needed, add a ``routes.py`` here and wire it via
``register_routes``.
"""

from __future__ import annotations

from pathlib import Path

from metainfer.web.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG

PLUGIN_TYPE = "gen-infer-framework"
# Frontend assets live in the task package's static/ dir (sibling of this
# package). Resolve via Path(__file__) so the package is relocatable.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"
_STATIC_PREFIX = f"/static/plugins/{PLUGIN_TYPE}"

_IMPORTMAP_ENTRIES = {
    "app/gf-detail": f"{_STATIC_PREFIX}/gf-detail.js?v=CACHE_BUST",
}

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    register_routes=None,  # no custom routes today
    detail_view_module="app/gf-detail",
    detail_view_export="default",
    qa_config=_QA_CONFIG,
    frontend_dir=_FRONTEND_DIR,
    importmap_entries=_IMPORTMAP_ENTRIES,
)

register(plugin)
