"""Web plugin registry.

Each task type that needs HTTP routes / readers / QA support / a custom
detail view registers a :class:`WebPlugin` here. The main ``create_app``
iterates :func:`all_plugins` and hands each plugin a chance to mount its
routes onto the FastAPI app.

Registration is a side-effect of importing the task package:
``metainfer/tasks/__init__.py`` auto-discovers sibling packages, and
each task package's ``__init__.py`` imports its own
``web_server_handler.plugin`` submodule so the ``register(...)`` call
fires. Adding a new task-type plugin only requires dropping a package
under ``metainfer/tasks/<name>/`` — no edits to ``app.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

from fastapi import FastAPI


class QAConfigLike(Protocol):
    """Minimal protocol a plugin's ``qa_config`` should satisfy.

    The plugin tells the generic :mod:`metainfer.web.qa` engine how to
    locate a target agent's transcript and working directory. Different
    task types have different on-disk layouts, so each plugin provides
    its own pathsolver.
    """

    def resolve_target(
        self, state_dir: Path, payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Map a request payload (e.g. ``{step, round, agent}``) to the
        analyst's targets. Returns a dict with at least::

            {
              "events_file":     Path,           # required
              "target_workdir":  Optional[Path], # None if N/A for this type
              "target_label":    str,            # human-readable, shown in UI
            }

        Raise :class:`ValueError` for malformed payloads; raise
        :class:`FileNotFoundError` (or similar) when the target can't
        be located.
        """
        ...


@dataclass
class WebDeps:
    """Shared core state handed to every plugin's ``register_routes``.

    Plugins should NOT hold references to the FastAPI ``app`` beyond
    their ``register_routes`` call — the same plugin could in principle
    be registered against multiple apps in tests.
    """
    repo_root: Path
    # An optional callable that returns a launcher, so plugins can
    # trigger orchestrator restarts / control actions. Currently set
    # by ``create_app`` post-construction.
    get_launcher: Optional[Callable[[], Any]] = None


@dataclass
class WebPlugin:
    """One registered task-type plugin.

    Attributes:
        type: The task type this plugin serves (e.g.
            ``"calc-theoretical-value"``). Must match the value the
            orchestrator writes into ``requirements.json::task_type``.
        register_routes: Optional callable that mounts routes onto the
            FastAPI app. Receives ``(app, deps)``. Plugins with no
            custom routes (only a detail-view hint, say) leave this
            ``None``.
        detail_view_module: Optional importmap key (e.g.
            ``"app/calc-viz"``) the frontend should dynamically import
            to render this task's detail view. ``None`` → default view.
        detail_view_export: Named export on ``detail_view_module``
            (default ``"default"``).
        qa_config: Optional :class:`QAConfigLike` instance. When set,
            the generic qa engine can resolve targets for this task
            type. Plugins that don't want the analyst feature leave
            this ``None``.
        frontend_dir: Optional ``Path`` to this plugin's bundled
            frontend assets (ES modules). When set, ``create_app``
            mounts it at ``/static/plugins/<type>/`` so the modules are
            web-addressable. The plugin's own ``importmap_entries``
            should reference URLs under that mount point.
        importmap_entries: Mapping of importmap keys to URL paths the
            browser should resolve. ``create_app`` injects these into
            the ``index.html`` importmap (after applying the cache-bust
            token) so plugins don't need to touch ``index.html``
            directly. Example::

                {
                    "app/calc-viz": "/static/plugins/calc-theoretical-value/calc-viz.js?v=CACHE_BUST",
                }
    """
    type: str
    register_routes: Optional[Callable[[FastAPI, WebDeps], None]] = None
    detail_view_module: Optional[str] = None
    detail_view_export: str = "default"
    qa_config: Optional[Any] = None
    frontend_dir: Optional[Path] = None
    importmap_entries: Dict[str, str] = field(default_factory=dict)


_REGISTRY: Dict[str, WebPlugin] = {}


def register(plugin: WebPlugin) -> None:
    """Register a plugin. Raises ``ValueError`` on duplicate ``type``."""
    if plugin.type in _REGISTRY:
        raise ValueError(
            f"duplicate web plugin for task type {plugin.type!r}; "
            f"already registered by {_REGISTRY[plugin.type]!r}"
        )
    _REGISTRY[plugin.type] = plugin


def get(type_name: str) -> Optional[WebPlugin]:
    return _REGISTRY.get(type_name)


def all_plugins() -> List[WebPlugin]:
    return list(_REGISTRY.values())
