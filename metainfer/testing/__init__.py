"""Shared test infrastructure for MetaInfer.

This package is **not** business logic — it holds test doubles used across
orchestrator + web plugin tests:

* :class:`MockAgentManager` — in-process stand-in for ``SubAgentManager``
  so step code can fan out via ``launch_async`` without spawning any
  ``claude`` / ``ccb`` subprocess.
* :class:`FakeStore` — captures ``append_timeline`` events.
* :class:`FakeLauncher` + ``isolated_env`` — pytest fixture for WebUI
  route tests; points ``METAINFER_HOME`` at tmp and swaps the default
  launcher with a no-op stand-in.
* :func:`write_calc_script` — drops a minimal ``calc.py`` returning given
  prefill/decode numbers.

Tests should import from the top level:

    from metainfer.testing import (
        MockAgentManager, FakeStore, FakeLauncher,
        isolated_env, write_calc_script,
    )
"""

from .mock_agent import MockAgentManager, FakeStore
from .mock_launcher import FakeLauncher, isolated_env
from .calc_helpers import write_calc_script

__all__ = [
    "MockAgentManager",
    "FakeStore",
    "FakeLauncher",
    "isolated_env",
    "write_calc_script",
]
