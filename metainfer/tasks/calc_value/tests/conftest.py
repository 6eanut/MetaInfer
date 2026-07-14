"""Shared fixtures for calc_value plugin route tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from metainfer.testing import isolated_env  # noqa: F401 — re-export as fixture
from metainfer.web import app as app_module
from metainfer.web import tasks as _tasks
from metainfer.web.tasks import TaskEntry


@pytest.fixture
def app(isolated_env):
    return app_module.create_app()


@pytest.fixture
def client(app):
    return TestClient(app)


def _register_calc_task(state_dir, task_id: str = "ct-1") -> TaskEntry:
    state_dir.mkdir(parents=True, exist_ok=True)
    entry = TaskEntry(
        id=task_id, type="calc-theoretical-value",
        label="test calc task", state_dir=str(state_dir), created_at=0.0,
    )
    _tasks.add_task(entry)
    return entry
