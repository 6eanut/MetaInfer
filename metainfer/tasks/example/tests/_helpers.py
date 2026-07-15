"""Test helpers for this task type.

Shared mock tools come from :mod:`metainfer.testing`::

    from metainfer.testing import (
        MockAgentManager, FakeStore, FakeLauncher, isolated_env,
    )

Put task-specific helper factories here — NOT in ``metainfer/testing/``.
"""

from __future__ import annotations

from pathlib import Path


def make_example_requirements(task_id: str = "example-1") -> dict:
    """Build a minimal requirements.json payload for testing."""
    return {
        "task_id": task_id,
        "task_type": "X-type-id",
        "created_at": 0.0,
        "form": {"project_name": "test", "max_iterations": 3},
    }


def make_example_state_dir(tmp: Path) -> Path:
    """Create a state_dir and workspace_dir stub for tests."""
    sd = tmp / "state"
    sd.mkdir(parents=True)
    (sd / "requirements.json").write_text(
        '{"task_id":"ex-1","task_type":"X-type-id","created_at":0.0,"form":{}}'
    )
    wd = tmp / "workspace"
    wd.mkdir(parents=True)
    return sd
