"""End-to-end pipeline test using MockAgentManager.

Drives the 6-phase state machine in-process: each phase's mock
response writes the expected artifact so the pipeline can advance.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from metainfer.orchestrator.state import StateStore
from metainfer.testing.mock_agent import MockAgentManager

from metainfer.tasks.port_model.orchestrator.pipeline import (
    Pipeline,
    PipelineConfig,
)
from metainfer.tasks.port_model.tests._helpers import (
    make_minimal_config,
    make_requirements,
)


def _make_response_fn(target_fw: Path):
    """Mock response_fn that writes the right artifact per role."""

    def response_fn(spec) -> str:
        role = getattr(spec, "role", "")
        wd: Path = Path(spec.workdir)
        wd.mkdir(parents=True, exist_ok=True)

        # Every agent writes a summary.md so the pipeline can parse outcome.
        def _summary(outcome: str = "ok", body: str = "") -> None:
            (wd / "summary.md").write_text(
                f"# stub\n\n## Outcome\n{outcome}\n\n## What I did\n{body}\n",
                encoding="utf-8",
            )

        if role == "p1_weight_analyst":
            (wd / "p1_weight_analysis.md").write_text(
                "# Weight analysis\n\nstub\n", encoding="utf-8"
            )
            _summary("ok", "read config.json")
            return "p1 done"

        if role == "p2_framework_analyst":
            # The pipeline names workdir as p2/refN and the artifact
            # as p2_ref{N+1}_analysis.md (1-based). Infer N from the
            # workdir parts and add 1.
            ref_idx_1based = 1
            for part in wd.parts:
                if part.startswith("ref"):
                    try:
                        ref_idx_1based = int(part[3:]) + 1
                    except ValueError:
                        pass
            (wd / f"p2_ref{ref_idx_1based}_analysis.md").write_text(
                f"# Reference {ref_idx_1based} analysis\n\nstub\n",
                encoding="utf-8",
            )
            _summary("ok", f"analysed ref {ref_idx_1based}")
            return "p2 done"

        if role == "p3_architect":
            (wd / "p3_consolidated_spec.md").write_text(
                "# Consolidated spec\n\nstub\n", encoding="utf-8"
            )
            _summary("ok", "merged P2 outputs")
            return "p3 done"

        if role == "p4_minimal_framework_writer":
            (wd / "run.py").write_text(
                "print('minimal framework')\n", encoding="utf-8"
            )
            (wd / "modeling_min.py").write_text("# modeling\n", encoding="utf-8")
            _summary("ok", "wrote minimal framework")
            return "p4 done"

        if role == "p5_minimal_framework_verifier":
            # P5 runs in attempt_XX subdir of cfg.p5_dir.
            (wd / "verdict.json").write_text(
                json.dumps({
                    "passed": True,
                    "batch": [
                        {"prompt": "世界上最高的山峰是",
                         "topk_text": ["珠穆朗玛峰", "珠穆朗", "珠峰"],
                         "verifier_judgment": "passed",
                         "verifier_reason": "top-1 是珠峰标准中文名"},
                        {"prompt": "中国的国旗是",
                         "topk_text": ["五星红旗", "红旗", "五星"],
                         "verifier_judgment": "passed",
                         "verifier_reason": "中国国旗正式名称"},
                        {"prompt": "人体正常体温约为",
                         "topk_text": ["三十七", "37", "三十七摄氏度"],
                         "verifier_judgment": "passed",
                         "verifier_reason": "37°C 正确"},
                    ],
                    "dump_dir": str(wd / "dumps"),
                    "log_file": str(wd / "run.log"),
                }),
                encoding="utf-8",
            )
            _summary("ok", "minimal framework verified")
            return "p5 done"

        if role == "p6_port_engineer":
            # P6 runs in iter_XX subdir.
            iter_idx = 1
            for part in wd.parts:
                if part.startswith("iter_"):
                    try:
                        iter_idx = int(part[4:]) + 1
                    except ValueError:
                        pass
            (wd / f"verdict_{iter_idx}.json").write_text(
                json.dumps({
                    "iteration": iter_idx,
                    "launched": True,
                    "batch": [
                        {"prompt": "世界上最高的山峰是",
                         "topk_text": ["珠穆朗玛峰"],
                         "verifier_judgment": "passed",
                         "verifier_reason": "top-1 一致"},
                        {"prompt": "中国的国旗是",
                         "topk_text": ["五星红旗"],
                         "verifier_judgment": "passed",
                         "verifier_reason": "top-1 一致"},
                        {"prompt": "人体正常体温约为",
                         "topk_text": ["三十七"],
                         "verifier_judgment": "passed",
                         "verifier_reason": "top-1 一致"},
                    ],
                    "similarity_min": 0.998,
                    "similarity_first_bad_layer": None,
                    "similarity_first_bad_row": None,
                    "outcome": "ok",
                    "reason": "",
                }),
                encoding="utf-8",
            )
            (wd / "summary.md").write_text(
                f"# P6 iter {iter_idx}\n\n## Outcome\nok\n",
                encoding="utf-8",
            )
            return "p6 done"

        _summary("ok", "stub")
        return "ok"

    return response_fn


def _build_layout(tmp_path: Path, ref_count: int = 1):
    """Lay out state/workspace + inputs and return (req, cfg paths)."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(make_minimal_config()), encoding="utf-8"
    )
    target_fw = tmp_path / "target_fw"
    target_fw.mkdir()
    (target_fw / "existing.py").write_text("# existing\n", encoding="utf-8")

    refs: list[dict[str, Any]] = []
    for i in range(ref_count):
        rd = tmp_path / f"ref_fw_{i}"
        rd.mkdir()
        (rd / "model.py").write_text(f"# reference {i}\n", encoding="utf-8")
        refs.append({"path": str(rd), "notes": f"ref {i} note"})

    state_dir = tmp_path / "state"
    workspace_dir = tmp_path / "workspace"
    for sub in ("p1", "p2", "p3", "p4", "p5", "p6", "memory", "dumps"):
        (workspace_dir / sub).mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    req = make_requirements(
        model_params_path=str(model_dir),
        target_framework_dir=str(target_fw),
        reference_sources=refs,
    )
    cfg = PipelineConfig(
        state_dir=state_dir,
        workspace_dir=workspace_dir,
        p1_dir=workspace_dir / "p1",
        p2_dir=workspace_dir / "p2",
        p3_dir=workspace_dir / "p3",
        p4_dir=workspace_dir / "p4",
        p5_dir=workspace_dir / "p5",
        p6_dir=workspace_dir / "p6",
        memory_dir=workspace_dir / "memory",
        dumps_dir=workspace_dir / "dumps",
        target_fw_dir=target_fw,
        model_params_path=model_dir,
        reference_sources=refs,
        user_notes="test",
    )
    return req, cfg


def test_pipeline_runs_end_to_end_single_ref(tmp_path: Path):
    req, cfg = _build_layout(tmp_path, ref_count=1)
    manager = MockAgentManager(response_fn=_make_response_fn(cfg.target_fw_dir))
    store = StateStore(cfg.state_dir)
    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)
    rc = pipeline.run()

    assert rc == 0
    run = json.loads((cfg.state_dir / "run.json").read_text(encoding="utf-8"))
    assert run["finished"] is True
    assert run["final_status"] == "success", run
    assert run["current_phase"] == "finished"

    # Canonical artifacts.
    assert (cfg.memory_dir / "p1_weight_analysis.md").is_file()
    assert (cfg.memory_dir / "p2_ref1_analysis.md").is_file()
    assert (cfg.memory_dir / "p3_consolidated_spec.md").is_file()
    # P4 run.py + P5 verdict + P6 verdict.
    assert (cfg.p4_dir / "run.py").is_file()
    assert (cfg.p5_dir / "attempt_00" / "verdict.json").is_file()
    assert (cfg.p6_dir / "iter_00" / "verdict_1.json").is_file()

    # Iteration record was written.
    iter_path = cfg.state_dir / "iterations" / "001.json"
    assert iter_path.is_file()
    rec = json.loads(iter_path.read_text(encoding="utf-8"))
    assert rec["status"] == "success"
    assert "P1_weight_analysis" in rec["phases"]
    assert "P6_port_engine" in rec["phases"]


def test_pipeline_runs_end_to_end_multi_ref(tmp_path: Path):
    """Multiple reference sources fan out to multiple P2 agents."""
    req, cfg = _build_layout(tmp_path, ref_count=3)
    manager = MockAgentManager(response_fn=_make_response_fn(cfg.target_fw_dir))
    store = StateStore(cfg.state_dir)
    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)
    rc = pipeline.run()

    assert rc == 0
    # Each reference produced a memory artifact.
    for i in (1, 2, 3):
        assert (cfg.memory_dir / f"p2_ref{i}_analysis.md").is_file()


def test_pipeline_p5_fail_then_repair(tmp_path: Path):
    """Force the first P5 attempt to crash; the second one passes."""
    req, cfg = _build_layout(tmp_path, ref_count=1)

    class _FailFirstThenPass:
        def __init__(self) -> None:
            self.p5_calls = 0

        def __call__(self, spec) -> str:
            if getattr(spec, "role", "") == "p5_minimal_framework_verifier":
                self.p5_calls += 1
                if self.p5_calls == 1:
                    # Simulate a crash.
                    wd = Path(spec.workdir)
                    wd.mkdir(parents=True, exist_ok=True)
                    (wd / "verdict.json").write_text(
                        json.dumps({
                            "passed": False,
                            "reason": "crash",
                            "error_class": "RuntimeError",
                            "error_message": "boom",
                        }),
                        encoding="utf-8",
                    )
                    (wd / "summary.md").write_text(
                        "# P5\n\n## Outcome\ntest_fail\n", encoding="utf-8",
                    )
                    return "crashed"
            # Delegate to the standard responder.
            return _make_response_fn(cfg.target_fw_dir)(spec)

    stateful = _FailFirstThenPass()
    manager = MockAgentManager(response_fn=stateful)
    store = StateStore(cfg.state_dir)
    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)
    rc = pipeline.run()

    # Pipeline should have looped P5→P4→P5 and ended successfully.
    assert rc == 0
    run = json.loads((cfg.state_dir / "run.json").read_text(encoding="utf-8"))
    assert run["final_status"] == "success", run
    assert stateful.p5_calls >= 2


def test_pipeline_p3_bounce_then_force_proceed(tmp_path: Path):
    """Architect bounces P1 twice — pipeline must force-proceed on 3rd."""
    req, cfg = _build_layout(tmp_path, ref_count=1)

    base_fn = _make_response_fn(cfg.target_fw_dir)

    def response_fn(spec) -> str:
        if getattr(spec, "role", "") == "p3_architect":
            wd = Path(spec.workdir)
            wd.mkdir(parents=True, exist_ok=True)
            (wd / "summary.md").write_text(
                "# P3\n\n## Outcome\nbounce_back\n\n"
                "## What I did\ndisagreement on layer count\n",
                encoding="utf-8",
            )
            return "bounce"
        return base_fn(spec)

    manager = MockAgentManager(response_fn=response_fn)
    store = StateStore(cfg.state_dir)
    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)
    rc = pipeline.run()

    # After 2 bounces, the pipeline force-proceeds and should complete.
    assert rc == 0
    run = json.loads((cfg.state_dir / "run.json").read_text(encoding="utf-8"))
    # Either success (if mock eventually returns ok) or stopped.
    assert run["final_status"] in ("success", "stopped"), run


def test_pipeline_p6_loops_on_needs_repair(tmp_path: Path):
    """P6 reports needs_repair on iter 1, ok on iter 2."""
    req, cfg = _build_layout(tmp_path, ref_count=1)

    state = {"p6_calls": 0}

    def response_fn(spec) -> str:
        if getattr(spec, "role", "") == "p6_port_engineer":
            state["p6_calls"] += 1
            wd = Path(spec.workdir)
            wd.mkdir(parents=True, exist_ok=True)
            n = state["p6_calls"]
            outcome = "needs_repair" if n == 1 else "ok"
            sim = 0.85 if n == 1 else 0.999
            (wd / f"verdict_{n}.json").write_text(
                json.dumps({
                    "iteration": n, "launched": True,
                    "batch": [
                        {"prompt": "世界上最高的山峰是",
                         "topk_text": ["odesk"] if n == 1 else ["珠穆朗玛峰"],
                         "verifier_judgment": "failed" if n == 1 else "passed",
                         "verifier_reason": "garbled" if n == 1 else "ok"},
                        {"prompt": "中国的国旗是",
                         "topk_text": ["odesk"] if n == 1 else ["五星红旗"],
                         "verifier_judgment": "failed" if n == 1 else "passed",
                         "verifier_reason": "garbled" if n == 1 else "ok"},
                        {"prompt": "人体正常体温约为",
                         "topk_text": ["odesk"] if n == 1 else ["三十七"],
                         "verifier_judgment": "failed" if n == 1 else "passed",
                         "verifier_reason": "garbled" if n == 1 else "ok"},
                    ],
                    "similarity_min": sim,
                    "similarity_first_bad_layer": 5 if n == 1 else None,
                    "similarity_first_bad_row": 0 if n == 1 else None,
                    "outcome": outcome, "reason": "",
                }),
                encoding="utf-8",
            )
            (wd / "summary.md").write_text(
                f"# P6\n\n## Outcome\n{outcome}\n", encoding="utf-8",
            )
            return "p6"
        return _make_response_fn(cfg.target_fw_dir)(spec)

    manager = MockAgentManager(response_fn=response_fn)
    store = StateStore(cfg.state_dir)
    pipeline = Pipeline(req=req, store=store, cfg=cfg, manager=manager)
    rc = pipeline.run()

    assert rc == 0
    assert state["p6_calls"] == 2
    run = json.loads((cfg.state_dir / "run.json").read_text(encoding="utf-8"))
    assert run["final_status"] == "success", run
