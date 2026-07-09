"""Path resolution for the gen-infer-framework skill.

Pure-skill layout — there is no plugin wrapper. The skill directory is
the unit of deployment, and it carries its own scripts and knowledge
base alongside its SKILL.md::

    MetaInfer/                          ← repo / checkout root
    └── skills/
        ├── gen-infer-framework/        ← THIS skill's root
        │   ├── SKILL.md
        │   ├── questions.yaml
        │   ├── run.py                  ← self-contained launcher
        │   ├── metainfer/              ← orchestrator package (this file)
        │   │   └── paths.py
        │   └── notebooks/              ← knowledge base
        ├── opt-kernel/
        │   ├── SKILL.md
        │   └── questions.yaml
        └── port-model/
            ├── SKILL.md
            └── questions.yaml

``paths.py`` lives at ``<skill_root>/metainfer/paths.py``, so the
gen-infer-framework skill root is its grandparent. Other skills sit as
siblings under ``<skill_root>/../``. All paths derive from ``__file__``
— no walk-up search, no env vars, no hardcoded absolute paths.
"""

from __future__ import annotations

from pathlib import Path

# Task types whose SKILL.md / questions.yaml sit alongside this skill
# under the same skills/ tree. gen-infer-framework is the only one that
# carries the orchestrator and notebooks today; the other two are still
# resolvable for cross-skill lookups (e.g. lists of available skills).
TASK_TYPES = ("gen-infer-framework", "opt-kernel", "port-model")


def skill_root() -> Path:
    """Absolute path to THIS skill's directory (gen-infer-framework).

    ``paths.py`` lives at ``<skill_root>/metainfer/paths.py``. The skill
    root is therefore ``Path(__file__).resolve().parent.parent``.
    """
    return Path(__file__).resolve().parent.parent


def skills_dir() -> Path:
    """Path to the parent ``skills/`` directory that holds this skill and
    its siblings (opt-kernel, port-model)."""
    return skill_root().parent


def skill_dir(task_type: str) -> Path:
    """Path to one task type's skill subdir under ``skills/``."""
    return skills_dir() / task_type


def skill_md(task_type: str) -> Path:
    """Path to one task type's ``SKILL.md``."""
    return skill_dir(task_type) / "SKILL.md"


def question_file(task_type: str) -> Path:
    """Path to the question bank for one task type.

    Each skill ships its own ``questions.yaml`` next to its ``SKILL.md``.
    ``task_type`` is one of :data:`TASK_TYPES`.
    """
    return skill_dir(task_type) / "questions.yaml"


def notebooks_dir() -> Path:
    """Path to the ``notebooks/`` knowledge base bundled with THIS skill."""
    return skill_root() / "notebooks"


def launcher() -> Path:
    """Path to THIS skill's self-contained Python launcher (``run.py``).

    Agents and users invoke the orchestrator via this absolute path:
        python <launcher> run requirements.json
    """
    return skill_root() / "run.py"
