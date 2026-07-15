"""calc-theoretical-value web plugin definition."""

from __future__ import annotations

from pathlib import Path
from typing import List

from metainfer.web._helpers import workspace_dir_for
from metainfer.web.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG
from .routes import build_router

PLUGIN_TYPE = "calc-theoretical-value"
# Frontend assets live in the task package's static/ dir (sibling of this
# package). Resolve via Path(__file__) so the package is relocatable.
_FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"
_STATIC_PREFIX = f"/static/plugins/{PLUGIN_TYPE}"

# Importmap entries are auto-discovered: every ``*.js`` directly under
# ``_FRONTEND_DIR`` is registered under ``app/<stem>`` by ``create_app``.
# That covers every module this plugin ships. We only need to populate
# the dict below to OVERRIDE shell entries — e.g. ship a divergent
# ``app/state-graph`` for this task type. The shell's default
# ``app/state-graph`` / ``app/charts`` / ``app/iterations-table`` widgets
# (in ``metainfer/tasks/sys_shell/static/components/``) are inherited as-is.
_IMPORTMAP_ENTRIES: dict = {}


def _extra_watch_paths(entry) -> List[Path]:
    """Tell the SSE watcher about calc's incremental-progress files.

    calc_value streams intermediate results into ``workspace_dir`` as
    each cell completes (``step0/rough_results.json`` and
    ``step3/cells/_state.json``); the audit panel refetches on each
    change. These live under ``workspace_dir``, not ``state_dir``, so
    we have to point the watcher at them explicitly. See
    :func:`metainfer.web.sse._scan_task` for how this hook is consumed.
    """
    wd = workspace_dir_for(entry)
    return [
        wd / "step0" / "rough_results.json",
        wd / "step3" / "cells" / "_state.json",
    ]


plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="Calc theoretical FLOPs / mem-traffic",
    description=(
        "Analyze a model + inference framework and compute the "
        "theoretical FLOPs and global-memory traffic of one forward "
        "pass, with per-node breakdown and an interactive visualization."
    ),
    build_router=build_router,
    detail_view_module="app/calc-detail",
    detail_view_export="default",
    qa_config=_QA_CONFIG,
    frontend_dir=_FRONTEND_DIR,
    importmap_entries=_IMPORTMAP_ENTRIES,
    extra_stylesheets=["calc.css"],
    extra_watch_paths=_extra_watch_paths,
)

register(plugin)
