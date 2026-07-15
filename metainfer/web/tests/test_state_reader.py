"""Tests for metainfer.web.state_reader — read helpers + reset_state_dir.

The reset feature is the user-visible "Reset button" requirement: wipe
everything except requirements.json so the task returns to its just-
created state, with iteration/runtime counts zeroed and a single
``task_reset`` event stamped into the timeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from metainfer.web import state_reader as sr


# --------------------------------------------------------------------------- #
# read_* helpers
# --------------------------------------------------------------------------- #

def test_read_requirements_returns_none_when_missing(tmp_path):
    assert sr.read_requirements(tmp_path) is None


def test_read_requirements_reads_json(tmp_path):
    (tmp_path / "requirements.json").write_text(
        json.dumps({"task_type": "calc-theoretical-value", "fields": {"a": 1}}),
        encoding="utf-8",
    )
    out = sr.read_requirements(tmp_path)
    assert out is not None
    assert out["task_type"] == "calc-theoretical-value"


def test_read_run_returns_defaults_when_missing(tmp_path):
    out = sr.read_run(tmp_path)
    assert out["current_phase"] == "idle"
    assert out["current_iteration"] == 0
    assert out["finished"] is False
    assert out["task_id"] is None


def test_read_run_merges_defaults_for_partial_file(tmp_path):
    """A run.json missing some keys should still produce a complete dict
    so the frontend doesn't KeyError."""
    (tmp_path / "run.json").write_text(
        json.dumps({"task_id": "abc", "task_type": "calc-theoretical-value"}),
        encoding="utf-8",
    )
    out = sr.read_run(tmp_path)
    assert out["task_id"] == "abc"
    assert out["task_type"] == "calc-theoretical-value"
    # Defaults still present:
    assert "current_phase" in out and out["current_phase"] == "idle"


def test_read_timeline_handles_missing_file(tmp_path):
    assert sr.read_timeline(tmp_path) == []


def test_read_timeline_filters_by_since(tmp_path):
    path = tmp_path / "timeline.jsonl"
    path.write_text(
        "\n".join(json.dumps(e) for e in [
            {"ts": 100.0, "type": "a", "payload": {}},
            {"ts": 200.0, "type": "b", "payload": {}},
            {"ts": 300.0, "type": "c", "payload": {}},
        ]) + "\n",
        encoding="utf-8",
    )
    out = sr.read_timeline(tmp_path, since=150.0)
    assert [e["type"] for e in out] == ["b", "c"]


def test_read_timeline_skips_garbage_lines(tmp_path):
    path = tmp_path / "timeline.jsonl"
    path.write_text(
        "not json\n"
        + json.dumps({"ts": 1.0, "type": "ok", "payload": {}}) + "\n",
        encoding="utf-8",
    )
    out = sr.read_timeline(tmp_path)
    assert len(out) == 1 and out[0]["type"] == "ok"


def test_read_iterations_sorts_by_number(tmp_path):
    iter_dir = tmp_path / "iterations"
    iter_dir.mkdir()
    (iter_dir / "002.json").write_text(json.dumps({"n": 2}), encoding="utf-8")
    (iter_dir / "001.json").write_text(json.dumps({"n": 1}), encoding="utf-8")
    out = sr.read_iterations(tmp_path)
    assert [r["n"] for r in out] == [1, 2]


def test_append_timeline_event_writes_jsonl(tmp_path):
    sr.append_timeline_event(tmp_path, "evt_x", {"k": "v"})
    sr.append_timeline_event(tmp_path, "evt_y", None)
    text = (tmp_path / "timeline.jsonl").read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    assert e1["type"] == "evt_x" and e1["payload"] == {"k": "v"}
    e2 = json.loads(lines[1])
    assert e2["type"] == "evt_y" and e2["payload"] == {}


# --------------------------------------------------------------------------- #
# reset_state_dir — the user's Reset-button requirement
# --------------------------------------------------------------------------- #

def test_reset_preserves_requirements_json(tmp_path):
    """The user's requirement: reset preserves ONLY requirements.json."""
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    state_dir.mkdir()
    workspace_dir.mkdir()
    req = {"task_type": "calc-theoretical-value", "fields": {"x": 1}}
    (state_dir / "requirements.json").write_text(
        json.dumps(req), encoding="utf-8",
    )
    # Plus some stuff that should be wiped.
    (state_dir / "run.json").write_text("{}", encoding="utf-8")
    (state_dir / "orchestrator.log").write_text("blah", encoding="utf-8")
    (state_dir / "orchestrator.pid").write_text("12345", encoding="utf-8")
    (state_dir / "iterations").mkdir()
    (workspace_dir / "step3").mkdir()
    (state_dir / "timeline.jsonl").write_text("", encoding="utf-8")

    summary = sr.reset_state_dir(state_dir, workspace_dir, "tid-1", "calc-theoretical-value")

    # requirements.json survived.
    assert sr.read_requirements(state_dir) == req
    # Everything else is gone.
    assert not (state_dir / "run.json").exists() or sr.read_run(state_dir)["finished"] is False
    assert not (state_dir / "orchestrator.log").exists()
    assert not (state_dir / "orchestrator.pid").exists()
    assert not (state_dir / "iterations").exists()
    assert not (workspace_dir / "step3").exists()
    # Fresh run.json was written.
    run = sr.read_run(state_dir)
    assert run["task_id"] == "tid-1"
    assert run["task_type"] == "calc-theoretical-value"
    assert run["current_iteration"] == 0
    assert run["current_phase"] == "idle"
    assert run["finished"] is False
    # And the summary returned by reset mentions what was removed.
    names = summary["removed"]
    assert "orchestrator.log" in names
    assert "iterations/" in names
    assert summary["workspace_reset"] is True


def test_reset_stamps_task_reset_timeline_event(tmp_path):
    """The user's requirement: reset must be auditable in the timeline."""
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    state_dir.mkdir()
    workspace_dir.mkdir()
    (state_dir / "requirements.json").write_text("{}", encoding="utf-8")
    sr.reset_state_dir(state_dir, workspace_dir, "tid-2", "calc-theoretical-value")
    events = sr.read_timeline(state_dir)
    types = [e["type"] for e in events]
    assert "task_reset" in types
    evt = next(e for e in events if e["type"] == "task_reset")
    assert evt["payload"]["task_id"] == "tid-2"
    assert evt["payload"]["reset_at"] > 0
    assert evt["payload"]["removed_count"] >= 0
    assert evt["payload"]["workspace_reset"] is True


def test_reset_creates_state_dir_if_missing(tmp_path):
    """reset should be idempotent even on a never-started task dir."""
    sd = tmp_path / "doesnt_exist_yet"
    wd = tmp_path / "wd_doesnt_exist_yet"
    sr.reset_state_dir(sd, wd, "tid-3", "calc-theoretical-value")
    assert (sd / "run.json").exists()
    assert (sd / "timeline.jsonl").exists()
    assert wd.exists()


def test_reset_default_keeps_only_requirements_when_others_absent(tmp_path):
    """If only requirements.json exists, reset is essentially a no-op
    except for writing run.json + the timeline event."""
    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    state_dir.mkdir()
    workspace_dir.mkdir()
    (state_dir / "requirements.json").write_text(
        json.dumps({"task_type": "calc-theoretical-value"}),
        encoding="utf-8",
    )
    summary = sr.reset_state_dir(state_dir, workspace_dir, "tid-4", "calc-theoretical-value")
    assert summary["removed"] == []
    # The file we kept is unchanged.
    assert sr.read_requirements(state_dir) == {"task_type": "calc-theoretical-value"}
