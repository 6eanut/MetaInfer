"""Tests for :mod:`metainfer.server.liveness`.

Verifies that the periodic scan detects orchestrator processes that
died ungracefully mid-run and invokes the existing reaper so the UI
doesn't freeze on a stale "running" snapshot.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from metainfer.server import liveness as _liveness
from metainfer.server import tasks as _tasks
from metainfer.server.launcher import ProcStatus


def _make_entry(tid, pid, finished_at=None):
    """Minimal TaskEntry-like object for the liveness scan."""
    return MagicMock(
        id=tid, pid=pid, finished_at=finished_at,
        state_dir=f"/tmp/{tid}", workspace_dir=f"/tmp/{tid}-ws",
    )


def test_scan_skips_tasks_registry_considers_done(monkeypatch):
    """A task with pid=None OR finished_at set is already considered
    stopped by the registry — the scanner must not touch it."""
    entries = [
        _make_entry("alive", pid=123),
        _make_entry("no-pid", pid=None),
        _make_entry("finished", pid=123, finished_at=time.time()),
    ]
    monkeypatch.setattr(_tasks, "list_tasks", lambda: entries)

    launcher = MagicMock()
    launcher.status = MagicMock()  # should not be called
    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    # Only the "alive" entry should have triggered status()
    assert launcher.status.call_count == 1
    assert launcher.status.call_args[0][0] == "alive"


def test_scan_ignores_alive_orchestrator(monkeypatch):
    """A live orchestrator (status.running=True) is left alone."""
    monkeypatch.setattr(_tasks, "list_tasks",
                        lambda: [_make_entry("alive", pid=123)])
    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=True, pid=123, started_at=1.0,
        finished_at=None, exit_hint="pid-alive",
    )
    # _reap_dead_pid_file must not be called for live processes
    launcher._reap_dead_pid_file = MagicMock()
    # But LocalLauncher check uses isinstance, so make it look like one
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    launcher._reap_dead_pid_file.assert_not_called()


def test_scan_reaps_dead_orchestrator(monkeypatch):
    """The bug scenario: orchestrator pid file claims running, /proc
    says dead. Scanner must invoke the reaper so the UI flips to stopped."""
    monkeypatch.setattr(_tasks, "list_tasks",
                        lambda: [_make_entry("dead", pid=999)])
    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=False, pid=999, started_at=1.0,
        finished_at=None, exit_hint="pid-dead",
    )
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    launcher._reap_dead_pid_file.assert_called_once_with("dead", 999, 1.0)


def test_scan_ignores_already_cleared_pid_file(monkeypatch):
    """exit_hint 'no-pid-file' / 'pid-file-cleared' are bookkeeping
    states — the orchestrator already wrote finished_at itself, so we
    must not double-write a reap event."""
    monkeypatch.setattr(_tasks, "list_tasks",
                        lambda: [_make_entry("clean", pid=123)])
    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=False, pid=None, started_at=None,
        finished_at=1234.0, exit_hint="pid-file-cleared",
    )
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    launcher._reap_dead_pid_file.assert_not_called()


def test_scan_skips_when_started_at_missing(monkeypatch):
    """PID-reuse safety: without started_at we can't validate. Don't reap."""
    monkeypatch.setattr(_tasks, "list_tasks",
                        lambda: [_make_entry("risky", pid=123)])
    launcher = MagicMock()
    launcher.status.return_value = ProcStatus(
        running=False, pid=123, started_at=None,  # ← missing
        finished_at=None, exit_hint="pid-dead",
    )
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()

    launcher._reap_dead_pid_file.assert_not_called()


def test_scan_survives_per_task_exception(monkeypatch):
    """A buggy entry must not kill the watcher — other tasks still get checked."""
    entries = [_make_entry("bad", pid=1), _make_entry("good", pid=2)]
    monkeypatch.setattr(_tasks, "list_tasks", lambda: entries)

    call_log = []

    def fake_status(tid):
        call_log.append(tid)
        if tid == "bad":
            raise RuntimeError("simulated registry corruption")
        return ProcStatus(
            running=False, pid=2, started_at=2.0,
            finished_at=None, exit_hint="pid-dead",
        )

    launcher = MagicMock()
    launcher.status = fake_status
    from metainfer.server.launcher import LocalLauncher
    launcher.__class__ = LocalLauncher
    launcher._reap_dead_pid_file = MagicMock()

    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()  # must not raise

    # Both tasks were attempted
    assert call_log == ["bad", "good"]
    # "good" was reaped, "bad" was skipped (exception swallowed)
    launcher._reap_dead_pid_file.assert_called_once_with("good", 2, 2.0)


def test_scan_survives_registry_read_failure(monkeypatch):
    """If the registry itself can't be read, the scan logs and exits
    cleanly — the next interval will retry."""
    def boom():
        raise RuntimeError("disk gone")
    monkeypatch.setattr(_tasks, "list_tasks", boom)

    launcher = MagicMock()
    checker = _liveness.LivenessChecker(launcher=launcher)
    checker._scan_once()  # must not raise

    launcher.status.assert_not_called()
