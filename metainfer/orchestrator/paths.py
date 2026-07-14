"""Path resolution for the MetaInfer orchestrator.

Pluggable task layout (post task-package refactor)::

    MetaInfer/                          ← repo / install root
    ├── pyproject.toml
    ├── metainfer/
    │   ├── __init__.py
    │   ├── orchestrator/              ← THIS package (shared framework)
    │   │   └── paths.py               ← this file
    │   ├── tasks/                     ← task packages (one per task type)
    │   │   ├── calc_value/            ← orchestrator/ + web_server_handler/ + static/ + tests/ + form.yaml
    │   │   └── gen_infer_framework/   ← same layout
    │   ├── web/
    │   └── static/
    ├── tasks/                         ← LEGACY stub schemas (opt-kernel.yaml, port-model.yaml)
    └── notebooks are per-task         ← metainfer/tasks/<pkg>/notebooks/ (knowledge bases are owned by each task package)

Each task type's form schema now lives at
``metainfer/tasks/<task_pkg>/form.yaml``; stub types without a full
package still ship a ``<repo>/tasks/<type>.yaml``. The form loader
(:func:`metainfer.web.forms.load_form_schema`) checks both locations.
Knowledge bases (formerly the top-level ``notebooks/``) now live inside
their task package — e.g. ``metainfer/tasks/gen_infer_framework/notebooks/``.

All paths derive from ``__file__`` — no walk-up search, no env vars,
no hardcoded absolute paths.
"""

from __future__ import annotations

from pathlib import Path

# Legacy stub task types that ship only a <repo>/tasks/<type>.yaml form
# schema (no orchestrator package yet). Full task types live under
# ``metainfer/tasks/<pkg>/`` and are discovered via the plugin registry,
# so they don't need to be listed here.
TASK_TYPES = (
    "gen-infer-framework",
    "opt-kernel",
    "port-model",
    "calc-theoretical-value",
)


def repo_root() -> Path:
    """Absolute path to the repo / install root.

    ``paths.py`` lives at ``<repo>/metainfer/orchestrator/paths.py``. The
    repo root is therefore ``Path(__file__).resolve().parents[2]``.
    """
    return Path(__file__).resolve().parents[2]


def tasks_dir() -> Path:
    """Path to the legacy ``tasks/`` directory that holds stub form-schema
    YAMLs (e.g. ``opt-kernel.yaml``, ``port-model.yaml``). Full task
    packages live under ``metainfer/tasks/<pkg>/form.yaml``."""
    return repo_root() / "tasks"


def question_file(task_type: str) -> Path:
    """Path to the legacy question-bank YAML for one task type.

    Note: full task types (calc-theoretical-value, gen-infer-framework)
    ship their schema at ``metainfer/tasks/<pkg>/form.yaml`` instead —
    callers should use :func:`metainfer.web.forms.load_form_schema`,
    which transparently resolves both locations.
    """
    return tasks_dir() / f"{task_type}.yaml"


def notebooks_dir() -> Path:
    """Deprecated. The top-level ``notebooks/`` knowledge base has moved
    into the task package that owns it — see
    ``metainfer/tasks/gen_infer_framework/notebooks/``. Each task package
    resolves its own knowledge base via ``Path(__file__).parent.parent /
    "notebooks"``; there is no global notebooks path anymore.

    Kept only as a hard-coded pointer to gen_infer_framework's notebooks
    for any external caller that hasn't been migrated. New code should
    resolve per-task.
    """
    return repo_root() / "metainfer" / "tasks" / "gen_infer_framework" / "notebooks"


# Back-compat shims so the rest of the codebase (which still calls
# skill_root() / skill_dir()) keeps working until those call sites are
# migrated. New code should use repo_root() / tasks_dir() directly.

def skill_root() -> Path:
    """Deprecated alias for :func:`repo_root`. Kept for back-compat."""
    return repo_root()


def skill_dir(task_type: str) -> Path:
    """Deprecated. Returns the tasks-dir parent path (no per-task subdir
    in the new layout)."""
    return tasks_dir()


def skill_md(task_type: str) -> Path:
    """Deprecated. SKILL.md no longer exists in the new layout."""
    raise FileNotFoundError(
        "SKILL.md is gone in the new layout; task definitions live in "
        f"{question_file(task_type)}"
    )


def launcher() -> Path:
    """Deprecated. The orchestrator is now launched via the WebUI's
    launcher, which dispatches by task_type (see
    ``metainfer.orchestrator.registry``)."""
    raise FileNotFoundError(
        "run.py launcher is gone; orchestrators are dispatched by the "
        "WebUI via metainfer.orchestrator.registry"
    )
