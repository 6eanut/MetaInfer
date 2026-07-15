"""Pytest config: put the repo root on sys.path so tests can do
``from metainfer...`` and ``from metainfer.testing import ...`` even
when running from outside the repo (CI checkout layout varies).

Also redirects ``METAINFER_ROOT`` to a temp directory so pytest never
creates ``nodes/`` in the project root (``root_dir()`` defaults to cwd
when the env var is unset; without this, ``create_app()`` at module level
in ``app.py`` would create ``nodes/<hostname>/`` in the repo on first
import).
"""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# -- sys.path ----------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
s = str(_REPO_ROOT)
if s not in sys.path:
    sys.path.insert(0, s)

# -- isolate test state from repo --------------------------------------------
# Must happen BEFORE any sub-conftest imports ``metainfer.server.app``, because
# ``app = create_app()`` at module level calls ``reconcile()`` which creates
# ``nodes/<hostname>/`` under ``METAINFER_ROOT`` (default: cwd).
if "METAINFER_ROOT" not in os.environ:
    _TEST_ROOT = tempfile.mkdtemp(prefix="metainfer-test-")
    os.environ["METAINFER_ROOT"] = _TEST_ROOT

    def _cleanup_test_root() -> None:
        shutil.rmtree(_TEST_ROOT, ignore_errors=True)
    atexit.register(_cleanup_test_root)
