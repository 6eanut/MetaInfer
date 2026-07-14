"""WebUI launcher mock + the ``isolated_env`` pytest fixture.

Route tests for WebUI plugins share these. Importing this module
registers ``isolated_env`` as a pytest fixture — but the fixture only
runs when explicitly requested by a test.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from metainfer.web import launcher as _launcher
from metainfer.web import tasks as _tasks
from metainfer.web.launcher import ProcStatus


class FakeLauncher:
    """Stand-in for ``LocalLauncher`` — no subprocesses, no IPC."""

    def __init__(self) -> None:
        self.killed: List[str] = []
        self.started: List[str] = []
        self._running: Dict[str, bool] = {}

    def status(self, task_id: str) -> ProcStatus:
        running = self._running.get(task_id, False)
        return ProcStatus(
            running=running,
            pid=None,
            started_at=None,
            finished_at=None,
            exit_hint="no-pid-file",
        )

    def kill(self, task_id: str, force: bool = False) -> bool:
        self.killed.append(task_id)
        self._running[task_id] = False
        return True

    def start(self, task_id: str, req: Dict[str, Any], state_dir: Path) -> int:
        self.started.append(task_id)
        self._running[task_id] = True
        return 99999


@pytest.fixture
def isolated_env(monkeypatch):
    """``METAINFER_HOME`` → tmp; launcher → ``FakeLauncher``.

    Yields a dict with ``home`` (Path) and ``launcher`` (FakeLauncher).
    """
    with tempfile.TemporaryDirectory() as td:
        home = Path(td) / "home"
        home.mkdir()
        monkeypatch.setenv("METAINFER_HOME", str(home))
        fake = FakeLauncher()
        monkeypatch.setattr(_launcher, "_DEFAULT", fake)
        # Force the tasks registry to re-resolve its path under the new home.
        _tasks._REGISTRY_PATH = None  # type: ignore[attr-defined]
        yield {
            "home": home,
            "launcher": fake,
        }
