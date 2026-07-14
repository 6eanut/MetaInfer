"""Pytest config: put the repo root on sys.path so tests can do
``from metainfer...`` and ``from metainfer.testing import ...`` even
when running from outside the repo (CI checkout layout varies).

When the package is installed via ``pip install -e .`` this is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent
s = str(_REPO_ROOT)
if s not in sys.path:
    sys.path.insert(0, s)
