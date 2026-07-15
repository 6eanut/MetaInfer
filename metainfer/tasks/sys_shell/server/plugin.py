"""sys-shell web plugin definition.

The shell is a special plugin — it owns the global WebUI chrome
(index.html, main.js, styles.css, shared components) and the type-agnostic
HTTP routes (task CRUD, lifecycle, monitoring). Unlike task-type plugins
it sets ``detail_view_module=None`` because it is not itself a task type.
"""

from __future__ import annotations

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

PLUGIN_TYPE = "sys-shell"
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"


def build_router(plugin):
    """Lazy import to avoid import-ordering issues during auto-discovery."""
    from .routes import build_router as _build
    return _build(plugin)


plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="MetaInfer Shell",
    description=(
        "System shell — task lifecycle, monitoring, and the shared "
        "WebUI chrome (layout, navigation, SSE streaming)."
    ),
    build_router=build_router,
    detail_view_module=None,  # shell is not a task type
    frontend_dir=_FRONTEND_DIR,
    importmap_entries={},
    extra_stylesheets=[],
    # No qa_config / extra_watch_paths — the shell has no task-specific data.
)

register(plugin)
