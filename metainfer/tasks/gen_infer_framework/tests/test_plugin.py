"""gen-infer-framework web plugin tests.

Covers plugin registration + QA pathsolver tuple-lookup behavior. The
detail-view dispatch is exercised end-to-end in
``metainfer/web/tests/test_app_core.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from metainfer.tasks.gen_infer_framework.web_server_handler._qa import (
    GenInferQAConfig,
    _resolve_gf_events_file,
)
from metainfer.web.registry import get as _get_plugin


def test_plugin_registered():
    p = _get_plugin("gen-infer-framework")
    assert p is not None
    assert p.detail_view_module == "app/gf-detail"
    assert p.frontend_dir is not None and p.frontend_dir.exists()
    assert "app/gf-detail" in p.importmap_entries
    # qa_config must be set so the analyst feature can resolve targets.
    assert p.qa_config is not None


def test_notebooks_dir_lives_inside_task_package():
    """The knowledge base moved out of the top-level ``notebooks/`` into
    this task's package. The orchestrator resolves it via
    ``Path(__file__).parent.parent / 'notebooks'`` so the package is
    relocatable; this test pins both the location and the existence of
    the canonical knowledge-base subdirs."""
    from metainfer.tasks.gen_infer_framework.orchestrator import orchestrator as orch
    nb = orch._NOTEBOOKS_DIR
    # Lives inside this task package, not at the top level.
    assert nb.name == "notebooks"
    assert "metainfer" in nb.parts
    assert "tasks" in nb.parts
    assert "gen_infer_framework" in nb.parts
    assert nb.is_dir()
    # The original top-level notebooks/ shipped these subdirs; they must
    # all have survived the git mv.
    for sub in ("00_contracts", "01_framework_design", "06_profiling",
                "07_improvementPlan"):
        assert (nb / sub).is_dir(), f"missing knowledge-base subdir {sub}"
    # And the legacy back-compat shim in orchestrator.paths still points
    # at the same path.
    from metainfer.orchestrator import paths as _orch_paths
    assert _orch_paths.notebooks_dir().resolve() == nb.resolve()


def test_qa_explicit_events_file(tmp_path: Path):
    cfg = GenInferQAConfig()
    ef = tmp_path / "agent.events.jsonl"
    ef.write_text("{}", encoding="utf-8")
    out = cfg.resolve_target(tmp_path, {"events_file": str(ef)})
    assert out["events_file"] == ef
    assert out["target_workdir"] is None
    assert "events_file=" in out["target_label"]


def test_qa_explicit_events_file_with_workdir(tmp_path: Path):
    cfg = GenInferQAConfig()
    ef = tmp_path / "x.events.jsonl"
    ef.write_text("{}", encoding="utf-8")
    wd = tmp_path / "work"
    wd.mkdir()
    out = cfg.resolve_target(tmp_path, {
        "events_file": str(ef),
        "target_workdir": str(wd),
        "target_label": "my label",
    })
    assert out["target_workdir"] == wd
    assert out["target_label"] == "my label"


def test_qa_tuple_lookup_finds_events_file(tmp_path: Path):
    # Materialize the orchestrator's standard layout for iter 1, agent foo.
    log_dir = tmp_path / "iterations" / "001" / ".metainfer-logs" / "foo"
    log_dir.mkdir(parents=True)
    ef = log_dir / "foo.attempt0.events.jsonl"
    ef.write_text(json.dumps({"hi": True}), encoding="utf-8")
    cfg = GenInferQAConfig()
    out = cfg.resolve_target(tmp_path, {"iteration": 1, "agent": "foo"})
    assert out["events_file"] == ef
    assert "iter=1" in out["target_label"]


def test_qa_tuple_lookup_falls_back_to_glob(tmp_path: Path):
    # Layout that doesn't match the candidate_dirs — should still be
    # found by the recursive glob.
    log_dir = tmp_path / "iterations" / "002" / "weird" / "nested" / "bar"
    log_dir.mkdir(parents=True)
    ef = log_dir / "bar.attempt0.events.jsonl"
    ef.write_text("{}", encoding="utf-8")
    found = _resolve_gf_events_file(tmp_path, 2, "bar")
    assert found == ef


def test_qa_tuple_lookup_raises_when_missing(tmp_path: Path):
    # Iteration dir doesn't exist at all.
    with pytest.raises(FileNotFoundError):
        _resolve_gf_events_file(tmp_path, 99, "nobody")


def test_qa_rejects_empty_payload(tmp_path: Path):
    cfg = GenInferQAConfig()
    with pytest.raises(ValueError):
        cfg.resolve_target(tmp_path, {})
