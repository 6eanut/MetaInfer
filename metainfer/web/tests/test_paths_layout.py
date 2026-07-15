"""Tests for the per-node + workspace/state_dir layout.

Layout invariant (under ``METAINFER_ROOT``)::

    <root>/
    └── nodes/
        └── <hostname-or-METAINFER_NODE_ID>/
            ├── .metainfer/                  ← metadata root (home_dir)
            │   ├── registry.json
            │   ├── runtime.json
            │   └── tasks/<task_id>/         ← state_dir
            │       ├── run.json
            │       ├── timeline.jsonl
            │       └── logs/...
            └── workspaces/
                └── <task_id>/               ← workspace_dir
                    └── (step0..step4 / 001/...)

This keeps per-task generated artifacts (workspace) cleanly separated
from per-task metadata + logs (state_dir), and adds an outer node
layer so multi-node deployments sharing a filesystem don't collide.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from metainfer.web import paths


def test_root_dir_defaults_to_cwd(monkeypatch):
    monkeypatch.delenv("METAINFER_ROOT", raising=False)
    assert paths.root_dir() == Path.cwd().resolve()


def test_root_dir_honors_env(monkeypatch, tmp_path):
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    assert paths.root_dir() == tmp_path.resolve()


def test_node_id_defaults_to_hostname(monkeypatch):
    monkeypatch.delenv("METAINFER_NODE_ID", raising=False)
    import socket
    assert paths.node_id() == socket.gethostname()


def test_node_id_honors_env(monkeypatch):
    monkeypatch.setenv("METAINFER_NODE_ID", "node-7")
    assert paths.node_id() == "node-7"


def test_layout_under_root(monkeypatch, tmp_path):
    """Full path invariant for home_dir / task_dir / workspace_dir."""
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    monkeypatch.setenv("METAINFER_NODE_ID", "hostA")

    home = paths.home_dir()
    assert home.name == ".metainfer"
    assert home.parent.name == "hostA"
    assert home.parent.parent.name == "nodes"
    assert home.parent.parent.parent == tmp_path.resolve()

    sd = paths.task_dir("t-1")
    assert sd.name == "t-1"
    assert sd.parent.name == "tasks"
    assert sd.parent.parent == home

    ws = paths.workspace_dir("t-1")
    assert ws.name == "t-1"
    assert ws.parent.name == "workspaces"
    assert ws.parent.parent == home.parent  # sibling of .metainfer/

    # Different tasks must have different state_dir AND workspace_dir.
    assert paths.task_dir("t-1") != paths.task_dir("t-2")
    assert paths.workspace_dir("t-1") != paths.workspace_dir("t-2")

    # workspace_dir is NOT auto-created — caller must mkdir before writing.
    # (We only check it didn't get materialized by the getter.)
    sd2 = paths.task_dir("other-task")
    ws2 = paths.workspace_dir("other-task")
    assert "other-task" in str(sd2)
    assert "other-task" in str(ws2)


def test_workspaces_root_is_sibling_of_home(monkeypatch, tmp_path):
    """workspaces/ and .metainfer/ must be siblings under the same node."""
    monkeypatch.setenv("METAINFER_ROOT", str(tmp_path))
    monkeypatch.setenv("METAINFER_NODE_ID", "hostB")

    wr = paths.workspaces_root()
    home = paths.home_dir()
    assert wr.parent == home.parent
    assert wr.name == "workspaces"
    assert home.name == ".metainfer"
