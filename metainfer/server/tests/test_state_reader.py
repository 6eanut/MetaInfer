"""Tests for metainfer.server.state_reader — read helpers + reset_state_dir.

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

from metainfer.server import state_reader as sr


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
    so the frontend doesn't KeyError. ``task_type`` is intentionally NOT
    in run.json — it lives in requirements.json. If a stale run.json
    still has it, read_run drops it (single source of truth)."""
    (tmp_path / "run.json").write_text(
        json.dumps({"task_id": "abc", "task_type": "calc-theoretical-value"}),
        encoding="utf-8",
    )
    out = sr.read_run(tmp_path)
    assert out["task_id"] == "abc"
    # task_type is dropped — not a run.json field anymore
    assert "task_type" not in out
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


def test_read_iterations_removed_from_shell(tmp_path):
    """Iteration records are now task-private (their schema is
    calc-shaped). The shell no longer exposes a read_iterations helper
    — each task package's ``_state_readers`` owns its own copy. This
    test pins that contract: importing the symbol MUST fail so no new
    shell code accidentally grows a dependency on a task's iteration
    schema."""
    assert not hasattr(sr, "read_iterations")
    assert not hasattr(sr, "read_iteration")
    assert not hasattr(sr, "read_charts")
    assert not hasattr(sr, "read_retrospective")
    assert not hasattr(sr, "read_state_graph")


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

    summary = sr.reset_state_dir(state_dir, workspace_dir, "tid-1")

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
    # task_type is NOT in run.json anymore — read from requirements.json.
    assert "task_type" not in run
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
    sr.reset_state_dir(state_dir, workspace_dir, "tid-2")
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
    sr.reset_state_dir(sd, wd, "tid-3")
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
    summary = sr.reset_state_dir(state_dir, workspace_dir, "tid-4")
    assert summary["removed"] == []
    # The file we kept is unchanged.
    assert sr.read_requirements(state_dir) == {"task_type": "calc-theoretical-value"}


# --------------------------------------------------------------------------- #
# read_agent_tail
# --------------------------------------------------------------------------- #
def test_read_agent_tail_404_when_agent_missing(tmp_path):
    """Unknown agent name → found=False so the route can 404."""
    (tmp_path / "agents.json").write_text(json.dumps({"ts": 0, "agents": []}))
    out = sr.read_agent_tail(tmp_path, "nonexistent")
    assert out["found"] is False
    assert out["events"] == []


def test_read_agent_tail_no_log_file_returns_empty(tmp_path):
    """Agent in snapshot but missing log_file → found, no events."""
    (tmp_path / "agents.json").write_text(json.dumps({
        "ts": 0,
        "agents": [{"name": "a1", "log_file": "", "attempt": 1}],
    }))
    out = sr.read_agent_tail(tmp_path, "a1")
    assert out["found"] is True
    assert out["events"] == []


def test_read_agent_tail_parses_assistant_text_and_tool_use(tmp_path):
    """Stream-json assistant events surface as text/tool_use summaries."""
    log_dir = tmp_path / "logs" / "p6" / "iter_00"
    log_dir.mkdir(parents=True)
    events_path = log_dir / "p6-porter.attempt1.events.jsonl"
    events_path.write_text(
        "\n".join([
            json.dumps({"type": "system", "session_id": "abc"}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Thinking about the boot."},
                {"type": "tool_use", "name": "Bash",
                 "input": {"command": "nvidia-smi"}},
            ]}}),
            json.dumps({"type": "user", "message": {"content": [
                {"type": "tool_result", "content": "GPU 0: K100"}]}}),
            json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Boot succeeded."}]}}),
        ]) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "agents.json").write_text(json.dumps({
        "ts": 0,
        "agents": [{
            "name": "p6-porter",
            "log_file": str(events_path).replace(".events.jsonl", ".log"),
            "attempt": 1,
        }],
    }))
    out = sr.read_agent_tail(tmp_path, "p6-porter")
    assert out["found"] is True
    types = [e["type"] for e in out["events"]]
    # text + tool_use paired per assistant turn → expect both
    assert "tool_use" in types
    assert "text" in types
    tool_evt = next(e for e in out["events"] if e["type"] == "tool_use")
    assert tool_evt["name"] == "Bash"
    assert tool_evt["input_brief"] == "nvidia-smi"


def test_read_agent_tail_caps_at_max_events(tmp_path):
    """Only the last N events are kept (browser-friendly payload)."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    events_path = log_dir / "a.events.jsonl"
    lines = []
    for i in range(100):
        evt = {"type": "assistant", "message": {"content": [
            {"type": "text", "text": f"line {i}"}]}}
        lines.append(json.dumps(evt))
    events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (tmp_path / "agents.json").write_text(json.dumps({
        "ts": 0,
        "agents": [{"name": "a", "log_file": str(events_path).replace(".events.jsonl", ".log")}],
    }))
    out = sr.read_agent_tail(tmp_path, "a", max_events=5)
    assert len(out["events"]) == 5
    # Should be the LAST 5
    assert out["events"][-1]["text"] == "line 99"


def test_read_agent_tail_falls_back_to_raw_log_when_no_events_jsonl(tmp_path):
    """If .events.jsonl sibling is missing, tail the raw .log as text lines."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    log_path = log_dir / "a.log"
    log_path.write_text("line1\nline2\nline3\n", encoding="utf-8")
    (tmp_path / "agents.json").write_text(json.dumps({
        "ts": 0,
        "agents": [{"name": "a", "log_file": str(log_path)}],
    }))
    out = sr.read_agent_tail(tmp_path, "a", max_events=2)
    assert out["found"] is True
    assert len(out["events"]) == 2
    assert out["events"][0]["type"] == "raw"
    assert out["events"][0]["text"] == "line2"
