"""Path resolution for the MetaInfer orchestrator.

New flat-repo layout (post skill-package refactor)::

    MetaInfer/                          ← repo / install root
    ├── pyproject.toml
    ├── metainfer/
    │   ├── __init__.py
    │   ├── orchestrator/              ← THIS package
    │   │   └── paths.py               ← this file
    │   ├── web/
    │   └── static/
    ├── tasks/                         ← questions.yaml per task type
    │   ├── gen-infer-framework.yaml
    │   ├── opt-kernel.yaml
    │   └── port-model.yaml
    └── notebooks/                     ← knowledge base

All paths derive from ``__file__`` — no walk-up search, no env vars,
no hardcoded absolute paths. The "skill root" concept is gone (we're no
longer a Claude skill).
"""

from __future__ import annotations

from pathlib import Path

# Task types whose questions.yaml sits under <repo>/tasks/.
# gen-infer-framework is the only one that uses the notebooks knowledge
# base today; the other two are still resolvable for task-type listings.
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
    """Path to the ``tasks/`` directory that holds all task-type
    questions.yaml files."""
    return repo_root() / "tasks"


def question_file(task_type: str) -> Path:
    """Path to the question bank for one task type.

    Each task type ships its own ``<task_type>.yaml`` under ``tasks/``.
    ``task_type`` is one of :data:`TASK_TYPES`.
    """
    return tasks_dir() / f"{task_type}.yaml"


def notebooks_dir() -> Path:
    """Path to the ``notebooks/`` knowledge base (top-level)."""
    return repo_root() / "notebooks"


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
