#!/usr/bin/env python3
"""Self-contained launcher for the gen-infer-framework skill's orchestrator.

Why this exists
---------------
The skill ships its own ``metainfer/`` Python package (the orchestrator)
alongside ``notebooks/`` (knowledge base), ``SKILL.md``, and
``questions.yaml``. There is no plugin wrapper, no global install step,
no ``bin/metainfer`` on PATH. The skill IS the unit.

Invoke directly with the Python interpreter, from anywhere:

    python /path/to/skills/gen-infer-framework/run.py run requirements.json
    python /path/to/skills/gen-infer-framework/run.py web <state_dir>

How it works
------------
This file lives at ``<skill_root>/run.py``. It inserts ``<skill_root>``
at the front of ``sys.path`` so ``import metainfer`` resolves to the
``metainfer/`` package sitting next to it — regardless of the user's
``PYTHONPATH`` or whether the package was pip-installed. Then it
delegates to :func:`metainfer.cli.main`.

The skill's SKILL.md tells agents and users to invoke via this absolute
path. No path searching, no find /, no environment variables.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _bootstrap_path() -> Path:
    """Insert the skill root (this file's parent) at the front of sys.path
    and return it. Resolved through ``Path.resolve()`` so symlinks
    (e.g. ``~/.claude/skills/gen-infer-framework`` → real repo path)
    don't trip up relative imports inside the package.
    """
    skill_root = Path(__file__).resolve().parent
    skill_root_str = str(skill_root)
    if skill_root_str not in sys.path:
        sys.path.insert(0, skill_root_str)
    return skill_root


def main() -> int:
    _bootstrap_path()
    from metainfer.cli import main as cli_main
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
