"""Tests for the generic per-plugin QA route helper."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient

from metainfer.server import tasks as _tasks
from metainfer.server.qa_routes import register_qa_routes
from metainfer.server.registry import WebPlugin
from metainfer.server.tasks import TaskEntry


class _FakeQAConfig:
    """Minimal pathsolver: maps {agent} → a synthetic events file
    rooted INSIDE the task's state_dir (so the engine's path-traversal
    guard accepts it)."""

    def resolve_target(self, state_dir: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
        agent = payload.get("agent") or "default"
        events_file = state_dir / "transcripts" / f"{agent}.events.jsonl"
        return {
            "events_file": events_file,
            "target_workdir": None,
            "target_label": f"agent={agent}",
        }


@pytest.fixture
def fake_plugin(tmp_path, isolated_env) -> WebPlugin:
    """Register a throwaway plugin + a task of its type, mount QA routes
    on a fresh FastAPI app, and return the plugin."""
    plugin = WebPlugin(
        type="__test_qa__",
        label="t", description="t",
        qa_config=_FakeQAConfig(),
    )
    from metainfer.server.registry import register as _register, _REGISTRY
    _register(plugin)
    state_dir = isolated_env["home"] / "tasks" / "qa-1"
    state_dir.mkdir(parents=True)
    workspace_dir = isolated_env["home"] / "workspaces" / "qa-1"
    workspace_dir.mkdir(parents=True)
    _tasks.add_task(TaskEntry(
        id="qa-1", type="__test_qa__", label="t",
        state_dir=str(state_dir), workspace_dir=str(workspace_dir),
        created_at=0.0,
    ))
    app = FastAPI()
    # Mirror how the shell mounts plugin routers: build a relative
    # APIRouter and include it under /api/__test_qa__/{task_id}. This
    # exercises the post-refactor contract (register_qa_routes now
    # mounts RELATIVE paths onto whatever router/app it receives).
    router = APIRouter()
    register_qa_routes(router, plugin, prefix="/qa")
    app.include_router(router, prefix="/api/__test_qa__/{task_id}")
    plugin._test_app = app  # type: ignore[attr-defined]
    yield plugin
    # Tear down so other tests don't see this plugin / task.
    _REGISTRY.pop("__test_qa__", None)
    _tasks.remove_task("qa-1")
    try:
        _tasks.remove_task("other-1")
    except Exception:
        pass


def test_qa_routes_reject_wrong_type(fake_plugin):
    """The type guard must reject requests for tasks whose type doesn't
    match the plugin the routes were mounted for."""
    # Register a task of a different type
    _tasks.add_task(TaskEntry(
        id="other-1", type="calc-theoretical-value", label="t",
        state_dir="/tmp/x", workspace_dir="/tmp/x", created_at=0.0,
    ))
    c = TestClient(fake_plugin._test_app)
    # /api/__test_qa__/<id>/qa/start on a non-matching task
    r = c.post("/api/__test_qa__/other-1/qa/start", json={"question": "q"})
    assert r.status_code == 409


def test_qa_routes_404_unknown_session(fake_plugin):
    c = TestClient(fake_plugin._test_app)
    r = c.get("/api/__test_qa__/qa-1/qa/no-such-session")
    assert r.status_code == 404


def test_qa_routes_list_empty(fake_plugin):
    c = TestClient(fake_plugin._test_app)
    r = c.get("/api/__test_qa__/qa-1/qa")
    assert r.status_code == 200
    assert r.json() == {"sessions": []}


def test_qa_routes_resolve_target_on_missing_events(fake_plugin):
    """When the body has no events_file, the helper invokes the plugin's
    resolve_target; if the resolved file doesn't exist, the engine raises
    EventsFileNotFound → 404."""
    c = TestClient(fake_plugin._test_app)
    r = c.post("/api/__test_qa__/qa-1/qa/start",
               json={"agent": "ghost", "question": "what did you do?"})
    assert r.status_code == 404


def test_qa_routes_passthrough_events_file(fake_plugin, monkeypatch, isolated_env):
    """Frontend-driven mode: body has events_file. We mock the QA engine
    so we don't actually spawn ccb."""
    state_dir = isolated_env["home"] / "tasks" / "qa-1"
    events_file = state_dir / "agent1.events.jsonl"
    events_file.write_text('{"type":"system"}\n', encoding="utf-8")

    started = {}

    def _fake_start(state_dir, payload):
        started["payload"] = payload
        return "sid-xyz"

    import metainfer.server.qa as _qa
    monkeypatch.setattr(_qa, "start_qa_session", _fake_start)

    c = TestClient(fake_plugin._test_app)
    r = c.post("/api/__test_qa__/qa-1/qa/start", json={
        "events_file": str(events_file),
        "question": "what",
    })
    assert r.status_code == 200
    assert r.json()["session_id"] == "sid-xyz"
    # Frontend-driven: events_file is passed through unchanged.
    assert started["payload"]["events_file"] == str(events_file)
