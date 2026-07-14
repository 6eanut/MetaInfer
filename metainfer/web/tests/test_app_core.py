"""Core web endpoint tests — task-types, /api/tasks CRUD, control actions,
DELETE task.

These exercise plugin-agnostic behavior. calc_value-specific routes live
in ``metainfer/tasks/calc_value/tests/test_routes.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from metainfer.web import tasks as _tasks
from metainfer.web.registry import all_plugins as _all_plugins
from metainfer.web.tasks import TaskEntry


def _register_task(state_dir: Path, task_id: str = "ct-1",
                   task_type: str = "calc-theoretical-value") -> TaskEntry:
    state_dir.mkdir(parents=True, exist_ok=True)
    entry = TaskEntry(
        id=task_id, type=task_type,
        label="test task", state_dir=str(state_dir), created_at=0.0,
    )
    _tasks.add_task(entry)
    return entry


# --------------------------------------------------------------------------- #
# /api/task-types
# --------------------------------------------------------------------------- #

def test_task_types_endpoint(client):
    resp = client.get("/api/task-types")
    assert resp.status_code == 200
    ids = [t["id"] for t in resp.json()]
    assert "calc-theoretical-value" in ids


def test_task_type_schema_endpoint(client):
    resp = client.get("/api/task-types/calc-theoretical-value/schema")
    assert resp.status_code == 200
    schema = resp.json()
    assert schema["type"] == "calc-theoretical-value"
    assert "fields" in schema


def test_task_type_schema_404(client):
    resp = client.get("/api/task-types/no-such-type/schema")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# /api/tasks
# --------------------------------------------------------------------------- #

def test_tasks_empty_initially(client, isolated_env):
    resp = client.get("/api/tasks")
    assert resp.status_code == 200
    assert resp.json() == {"tasks": []}


def test_get_task_404(client):
    resp = client.get("/api/tasks/no-such")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# Control actions (plugin-agnostic)
# --------------------------------------------------------------------------- #

def test_control_reset_wipes_state_when_stopped(client, isolated_env):
    state_dir = isolated_env["home"] / "tasks" / "ct-1"
    _register_task(state_dir, "ct-1")
    (state_dir / "run.json").write_text(
        json.dumps({"task_id": "ct-1", "current_phase": "S3"}),
        encoding="utf-8",
    )
    (state_dir / "orchestrator.log").write_text("blah", encoding="utf-8")
    (state_dir / "iterations").mkdir()
    resp = client.post("/api/tasks/ct-1/control", json={"action": "reset"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["action"] == "reset"
    run = json.loads((state_dir / "run.json").read_text())
    assert run["current_phase"] == "idle"
    assert run["current_iteration"] == 0
    assert not (state_dir / "orchestrator.log").exists()
    assert not (state_dir / "iterations").exists()


def test_control_reset_rejects_running_task(client, isolated_env):
    state_dir = isolated_env["home"] / "tasks" / "ct-1"
    _register_task(state_dir, "ct-1")
    isolated_env["launcher"]._running["ct-1"] = True  # pretend it's running
    resp = client.post("/api/tasks/ct-1/control", json={"action": "reset"})
    assert resp.status_code == 409
    assert "running" in resp.json()["detail"].lower()


def test_control_reset_stamps_timeline(client, isolated_env):
    state_dir = isolated_env["home"] / "tasks" / "ct-1"
    _register_task(state_dir, "ct-1")
    client.post("/api/tasks/ct-1/control", json={"action": "reset"})
    timeline_path = state_dir / "timeline.jsonl"
    assert timeline_path.exists()
    lines = [ln for ln in timeline_path.read_text().splitlines() if ln.strip()]
    types = [json.loads(ln)["type"] for ln in lines]
    assert "task_reset" in types


def test_control_kill_calls_launcher(client, isolated_env):
    state_dir = isolated_env["home"] / "tasks" / "ct-1"
    _register_task(state_dir, "ct-1")
    resp = client.post("/api/tasks/ct-1/control", json={"action": "kill"})
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_control_unknown_action(client, isolated_env):
    state_dir = isolated_env["home"] / "tasks" / "ct-1"
    _register_task(state_dir, "ct-1")
    resp = client.post("/api/tasks/ct-1/control", json={"action": "wat"})
    assert resp.status_code == 400


# --------------------------------------------------------------------------- #
# DELETE task — user-visible Close (×) feature
# --------------------------------------------------------------------------- #

def test_delete_task_purge_removes_files_and_registry(client, isolated_env):
    state_dir = isolated_env["home"] / "tasks" / "ct-1"
    _register_task(state_dir, "ct-1")
    (state_dir / "run.json").write_text("{}", encoding="utf-8")
    assert state_dir.exists()
    resp = client.delete("/api/tasks/ct-1?purge=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["removed_from_registry"] is True
    assert body["purged_files"] is True
    assert not state_dir.exists()
    assert _tasks.get_task("ct-1") is None


def test_delete_task_without_purge_keeps_files(client, isolated_env):
    state_dir = isolated_env["home"] / "tasks" / "ct-1"
    _register_task(state_dir, "ct-1")
    resp = client.delete("/api/tasks/ct-1")
    assert resp.status_code == 200
    body = resp.json()
    assert body["purged_files"] is False
    assert state_dir.exists()
    assert _tasks.get_task("ct-1") is None


# --------------------------------------------------------------------------- #
# Plugin frontend mount + importmap injection
# --------------------------------------------------------------------------- #

def test_index_html_injects_plugin_importmap(client):
    """Plugins declare ``importmap_entries``; create_app must inject
    them into the served index.html and substitute the cache-bust
    token. Also verifies the placeholder is gone."""
    resp = client.get("/")
    html = resp.text
    assert "<!-- PLUGINS_IMPORTMAP -->" not in html
    # Both plugins' detail-view modules should appear with a real
    # cache-bust token (not the literal placeholder).
    assert "/static/plugins/calc-theoretical-value/calc-detail.js?v=" in html
    assert "/static/plugins/gen-infer-framework/gf-detail.js?v=" in html
    assert "CACHE_BUST" not in html  # all substitutions happened


def test_every_plugin_frontend_dir_is_served(client):
    """Every plugin's frontend_dir is mounted at /static/plugins/<type>/
    so each of its importmap URLs is actually reachable. This is the
    end-to-end invariant that makes the dynamic-import dispatch work."""
    import re
    resp = client.get("/")
    html = resp.text
    urls = set(re.findall(r"/static/plugins/[\w\-]+/[\w\-\.]+\.js\?v=\d+", html))
    assert urls, "no plugin URLs were injected"
    for url in urls:
        r = client.get(url)
        assert r.status_code == 200, f"{url} returned {r.status_code}"


def test_every_plugin_detail_view_module_is_in_importmap(client):
    """Each plugin declares a detail_view_module; the index.html must
    contain an importmap entry for it so the shell's dynamic import
    resolves. Without this, the body silently falls back to the
    'no detail view' banner."""
    resp = client.get("/")
    html = resp.text
    for plugin in _all_plugins():
        dvm = plugin.detail_view_module
        assert dvm, f"{plugin.type} has no detail_view_module"
        # The injected entry should mention the plugin's static path.
        assert f"/static/plugins/{plugin.type}/" in html, (
            f"{plugin.type} detail view not in importmap"
        )
