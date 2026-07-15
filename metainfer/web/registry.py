"""Web plugin registry.

Each task type that needs HTTP routes / readers / QA support / a custom
detail view registers a :class:`WebPlugin` here. The main ``create_app``
iterates :func:`all_plugins` and mounts each plugin's ``build_router``
result under ``/api/{type}/{task_id}`` — the shell itself only hosts
task-agnostic endpoints (lifecycle, timeline, agents, token-budget).

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

from fastapi import APIRouter


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
    """Reserved for future shell → plugin injections.

    Historically handed to ``register_routes(app, deps, plugin)``. The
    current ``build_router(plugin)`` protocol no longer passes deps —
    plugins are expected to read everything they need from the
    :class:`WebPlugin` and from on-disk state via
    :mod:`metainfer.web._helpers`. Kept here as a stable type so older
    plugin code / tests that import it don't break during the
    transition; safe to remove once no references remain.
    """
    repo_root: Path
    get_launcher: Optional[Callable[[], Any]] = None


@dataclass
class WebPlugin:
    """One registered task-type plugin.

    Attributes:
        type: The task type this plugin serves. Must match the value the
            orchestrator writes into ``requirements.json::task_type``.
        label: Human-readable display name shown in the New Task type
            picker and as the task's default label. Required.
        description: One-line description shown under the label in the
            type picker. Required.
        register_routes (REMOVED): historical hook that mutated the
            FastAPI app directly. Replaced by :attr:`build_router`,
            which returns an :class:`fastapi.APIRouter` that the shell
            mounts under ``/api/{type}/{task_id}``. Plugins no
            longer receive the FastAPI app or :class:`WebDeps`.
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
            frontend assets (ES modules + stylesheets). When set,
            ``create_app`` mounts it at ``/static/plugins/<type>/`` so
            the files are web-addressable. The plugin's own
            ``importmap_entries`` should reference URLs under that mount
            point, and ``extra_stylesheets`` should reference CSS
            filenames relative to that mount point.
        importmap_entries: Mapping of importmap keys to URL paths the
            browser should resolve. ``create_app`` injects these into
            the ``index.html`` importmap (after applying the cache-bust
            token) so plugins don't need to touch ``index.html``
            directly. Entries here OVERRIDE any shell entry with the
            same key (so a plugin can ship its own version of a shared
            widget like ``app/state-graph``). Example::

                {
                    "app/<widget>": "/static/plugins/<type>/<widget>.js?v=CACHE_BUST",
                }

            NOTE: ``create_app`` ALSO auto-discovers every ``*.js``
            directly under ``frontend_dir`` and registers it under
            ``app/<stem>`` (plugin entries win on conflict). You only
            need to populate this dict for keys that DON'T follow that
            ``app/<filename-stem>`` convention, or to override shell
            entries.
        extra_stylesheets: List of CSS filenames (relative to
            ``frontend_dir``) that ``create_app`` should inject as
            ``<link>`` tags in ``index.html``. Use this to ship
            task-type-specific styles without editing the shell's
            ``metainfer/tasks/sys_shell/static/styles.css``. Each entry becomes
            ``/static/plugins/<type>/<filename>?v=<token>``.
        extra_watch_paths: Optional callable that returns extra files
            (under or outside ``state_dir``) the SSE watcher should
            monitor for mtime changes. Signature:
            ``f(entry: TaskEntry) -> List[Path]``. Returned paths may
            not exist yet; the watcher skips missing files. Return an
            empty list (or ``None``) if there's nothing extra to watch.
    """
    type: str
    label: str = ""
    description: str = ""
    build_router: Optional[Callable[["WebPlugin"], "APIRouter"]] = None
    """Build and return an :class:`fastapi.APIRouter` carrying all of
    this plugin's task-specific routes (relative paths only). The shell
    mounts it at ``/api/{type}/{task_id}`` — the shell itself no longer
    hosts any task-specific endpoints (no ``/iterations``, ``/charts``,
    ``/state-graph``, ``/retrospective``).

    Routes inside the router see ``task_id`` as a path param (declared
    in the shell's mount prefix) and should use the standard helpers
    from :mod:`metainfer.web._helpers` (``task_or_404``,
    ``require_task_type``, ``state_dir_for``, ``workspace_dir_for``)
    to resolve the on-disk targets.

    Plugins that want offline-QA support typically call
    :func:`metainfer.web.qa_routes.register_qa_routes` from inside
    their ``build_router`` to fold the generic QA triplet in. Plugins
    with no custom routes leave this ``None``.

    NOTE: this replaces the historical ``register_routes(app, deps,
    plugin)`` hook. The shell no longer hands plugins a chance to
    mutate the FastAPI app directly — every plugin's HTTP surface goes
    through this router so the URL boundary stays crisp."""
    detail_view_module: Optional[str] = None
    detail_view_export: str = "default"
    qa_config: Optional[Any] = None
    frontend_dir: Optional[Path] = None
    importmap_entries: Dict[str, str] = field(default_factory=dict)
    extra_stylesheets: List[str] = field(default_factory=list)
    extra_watch_paths: Optional[Callable[[Any], List[Path]]] = None


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
