"""calc_value plugin route tests — the ``/api/tasks/<id>/calc/...`` family.

Generic web endpoints (task-types, CRUD, control) live in
``metainfer/server/tests/test_app_core.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metainfer.server import tasks as _tasks
from metainfer.server.tasks import TaskEntry


def _register_calc_task(
    state_dir: Path, task_id: str = "ct-1", *, workspace_dir: Path | None = None,
) -> TaskEntry:
    state_dir.mkdir(parents=True, exist_ok=True)
    if workspace_dir is None:
        workspace_dir = state_dir
    workspace_dir.mkdir(parents=True, exist_ok=True)
    entry = TaskEntry(
        id=task_id, type="calc-theoretical-value",
        label="test calc task", state_dir=str(state_dir),
        workspace_dir=str(workspace_dir), created_at=0.0,
    )
    _tasks.add_task(entry)
    return entry


def _seed_calc_task(client, isolated_env, *, with_rough=False, with_final=False,
                    with_cells_state=False, with_cell=False):
    """Register a calc task + materialize the requested artifacts.

    Step0..step4 outputs go under the task's ``workspace_dir``; only
    metadata lives under ``state_dir``.
    """
    state_dir = isolated_env["home"] / "tasks" / "ct-1"
    from metainfer.server import paths as _paths
    workspace_dir = _paths.workspace_dir("ct-1")
    entry = _register_calc_task(state_dir, "ct-1", workspace_dir=workspace_dir)
    if with_rough:
        per_node = workspace_dir / "step0" / "agent_rough" / "per_node"
        per_node.mkdir(parents=True)
        (per_node / "layer__attn.py").write_text(
            "def calc(batch_size, seq_len):\n"
            "    return {'prefill': {'tflops': 100.0, 'access_gb': 10.0},"
            " 'decode': {'tflops': 1.0, 'access_gb': 20.0}}\n",
            encoding="utf-8",
        )
        (workspace_dir / "step0" / "rough_results.json").write_text(json.dumps({
            "ok": True, "elapsed_s": 5.0,
            "graph": {"sections": [{"id": "layer", "kind": "layer_template",
                                     "graph": {"nodes": [
                                         {"id": "attn", "compound": "layer__attn"}]}}]},
            "results": [{
                "compound": "layer__attn", "section_id": "layer",
                "node_id": "attn", "ok": True,
                "prefill": {"tflops": 100.0, "access_gb": 10.0},
                "decode":  {"tflops": 1.0, "access_gb": 20.0},
                "tflops_picked": 100.0, "gb_picked": 10.0,
            }],
            "summary": {"total_nodes": 1, "ok_count": 1, "fail_count": 0},
        }))
    if with_final:
        final_dir = workspace_dir / "step3" / "final"
        final_dir.mkdir(parents=True)
        (final_dir / "layer__attn.py").write_text(
            "def calc(batch_size, seq_len):\n"
            "    return {'prefill': {'tflops': 50.0*seq_len, 'access_gb': 5.0*seq_len},"
            " 'decode': {'tflops': 1.0, 'access_gb': 2.0 + 0.01*seq_len}}\n",
            encoding="utf-8",
        )
        (final_dir / "layer__attn.meta.json").write_text(json.dumps({
            "node_id": "attn", "section_id": "layer",
            "section_kind": "layer_template", "section_repeat_count": 4,
            "compound_id": "layer__attn", "approximate": False,
            "source_agent": "unanimous",
        }))
    if with_cells_state:
        cells_root = workspace_dir / "step3" / "cells"
        cells_root.mkdir(parents=True)
        (cells_root / "_state.json").write_text(json.dumps({
            "round": 0,
            "nodes": {
                "layer__attn": {
                    "node_id": "attn", "section_id": "layer",
                    "cells": {
                        "a": {"tflops": 100.0, "gb": 10.0,
                              "prefill": {"tflops": 100.0, "access_gb": 10.0},
                              "decode":  {"tflops": 1.0, "access_gb": 20.0},
                              "round": 0, "status": "ok", "elapsed_s": 1.0,
                              "error": None, "script_path": None},
                        "b": {"tflops": 102.0, "gb": 10.0,
                              "prefill": {"tflops": 102.0, "access_gb": 10.0},
                              "decode":  {"tflops": 1.0, "access_gb": 20.0},
                              "round": 0, "status": "ok", "elapsed_s": 1.0,
                              "error": None, "script_path": None},
                    },
                    "converged": True, "spread_pct": 0.02, "round": 0,
                },
            },
        }))
    if with_cell:
        cell_dir = (workspace_dir / "step3" / "cells" / "layer__attn" / "a" / "round_00")
        writer = cell_dir / "writer"
        writer.mkdir(parents=True)
        (writer / "calc.py").write_text("# fake", encoding="utf-8")
        (writer / "response.txt").write_text("agent thinking", encoding="utf-8")
        (cell_dir / "result.json").write_text(json.dumps({
            "batch_size": 1, "seq_len": 512,
            "prefill": {"tflops": 100.0, "access_gb": 10.0},
            "decode":  {"tflops": 1.0, "access_gb": 20.0},
        }))
    return entry


# --------------------------------------------------------------------------- #
# /calc/rough
# --------------------------------------------------------------------------- #

def test_calc_rough_pending_when_no_s0(client, isolated_env):
    _register_calc_task(isolated_env["home"] / "tasks" / "ct-1" / "x", "ct-1")
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/rough")
    assert resp.status_code == 200
    data = resp.json()
    assert data.get("pending") is True


def test_calc_rough_returns_canonical_values(client, isolated_env):
    _seed_calc_task(client, isolated_env, with_rough=True)
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/rough")
    assert resp.status_code == 200
    data = resp.json()
    r0 = data["results"][0]
    assert r0["prefill"] == {"tflops": 100.0, "access_gb": 10.0}
    assert r0["decode"] == {"tflops": 1.0, "access_gb": 20.0}


def test_calc_rough_ondemand_recompute(client, isolated_env):
    """User can change batch_size / seq_len — server re-runs the
    per_node script at the new shape."""
    _seed_calc_task(client, isolated_env, with_rough=True)
    # Note: this script returns FIXED numbers regardless of shape, so the
    # override response should still equal the canonical values. The test
    # verifies the endpoint runs without error and returns a combo field.
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/rough?batch_size=4&seq_len=2048")
    assert resp.status_code == 200
    data = resp.json()
    assert data["combo"] == {"batch_size": 4, "seq_len": 2048}
    r0 = data["results"][0]
    assert r0["ok"] is True


def test_calc_rough_400_on_invalid_shape(client, isolated_env):
    _seed_calc_task(client, isolated_env, with_rough=True)
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/rough?batch_size=0&seq_len=512")
    assert resp.status_code == 400


def test_calc_rough_409_for_wrong_task_type(client, isolated_env):
    state_dir = isolated_env["home"] / "tasks" / "gf-1"
    _tasks.add_task(TaskEntry(
        id="gf-1", type="gen-infer-framework",
        label="x", state_dir=str(state_dir), created_at=0.0,
    ))
    resp = client.get("/api/calc-theoretical-value/gf-1/calc/rough")
    assert resp.status_code == 409


# --------------------------------------------------------------------------- #
# /calc/compute
# --------------------------------------------------------------------------- #

def test_calc_compute_aggregates_with_repeat_count(client, isolated_env):
    """``totals.prefill.tflops`` should be per-instance * repeat_count."""
    _seed_calc_task(client, isolated_env, with_final=True)
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/compute?batch_size=1&seq_len=512")
    assert resp.status_code == 200
    data = resp.json()
    assert data["totals"]["prefill"]["tflops"] == pytest.approx(50.0 * 512 * 4)
    pc = data["per_compound"]["layer__attn"]
    assert pc["prefill"]["tflops"] == pytest.approx(50.0 * 512)
    assert pc["tflops"] == pytest.approx(50.0 * 512)


def test_calc_compute_400_on_invalid_shape(client, isolated_env):
    _seed_calc_task(client, isolated_env, with_final=True)
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/compute?batch_size=-1")
    assert resp.status_code == 400


def test_calc_compute_404_when_no_final(client, isolated_env):
    _register_calc_task(isolated_env["home"] / "tasks" / "ct-1" / "x", "ct-1")
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/compute")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# /calc/cells + /calc/cell/...
# --------------------------------------------------------------------------- #

def test_calc_cells_pending_when_no_state(client, isolated_env):
    _register_calc_task(isolated_env["home"] / "tasks" / "ct-1" / "x", "ct-1")
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/cells")
    assert resp.status_code == 200
    assert resp.json().get("pending") is True


def test_calc_cells_returns_state(client, isolated_env):
    _seed_calc_task(client, isolated_env, with_cells_state=True)
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/cells")
    assert resp.status_code == 200
    data = resp.json()
    node = data["nodes"]["layer__attn"]
    assert node["converged"] is True
    a = node["cells"]["a"]
    assert a["prefill"] == {"tflops": 100.0, "access_gb": 10.0}
    assert a["decode"] == {"tflops": 1.0, "access_gb": 20.0}


def test_calc_cells_ondemand_recompute(client, isolated_env):
    """Override shape triggers server-side re-execution of each cell's
    calc.py. We materialize a calc.py at the expected cell path that
    returns shape-dependent numbers."""
    _seed_calc_task(client, isolated_env, with_cells_state=True)
    from metainfer.server import paths as _paths
    workspace_dir = _paths.workspace_dir("ct-1")
    calc_path = workspace_dir / "step3" / "cells" / "layer__attn" / "a" / "round_00" / "writer" / "calc.py"
    calc_path.parent.mkdir(parents=True, exist_ok=True)
    calc_path.write_text(
        "def calc(batch_size, seq_len):\n"
        "    return {'prefill': {'tflops': 7.0*seq_len, 'access_gb': 1.0},"
        " 'decode': {'tflops': 0.5, 'access_gb': 2.0}}\n",
        encoding="utf-8",
    )
    calc_path_b = workspace_dir / "step3" / "cells" / "layer__attn" / "b" / "round_00" / "writer" / "calc.py"
    calc_path_b.parent.mkdir(parents=True, exist_ok=True)
    calc_path_b.write_text(calc_path.read_text(encoding="utf-8"), encoding="utf-8")
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/cells?batch_size=1&seq_len=1024")
    assert resp.status_code == 200
    data = resp.json()
    a = data["nodes"]["layer__attn"]["cells"]["a"]
    assert a["prefill"]["tflops"] == pytest.approx(7.0 * 1024)
    assert a["picked_combo"] == {"batch_size": 1, "seq_len": 1024}


def test_calc_cell_detail_returns_result(client, isolated_env):
    _seed_calc_task(client, isolated_env, with_cell=True)
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/cell/layer__attn/a/0")
    assert resp.status_code == 200
    data = resp.json()
    assert data["compound"] == "layer__attn"
    assert data["angle"] == "a"
    assert data["result"]["prefill"]["tflops"] == 100.0
    assert data["calc_py"] == "# fake"


def test_calc_cell_detail_400_on_bad_angle(client, isolated_env):
    _seed_calc_task(client, isolated_env, with_cell=True)
    resp = client.get("/api/calc-theoretical-value/ct-1/calc/cell/layer__attn/c/0")
    assert resp.status_code == 400
