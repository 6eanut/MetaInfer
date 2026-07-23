"""H.5 integration test: port_model distributed-testing wiring.

Verifies that:
  1. ``PipelineConfig`` accepts ``worker_nodes``.
  2. ``_parse_worker_nodes`` reads both list and comma-string.
  3. ``p5_verify_minimal_prompt`` / ``p6_port_engine_prompt`` inject the
     distributed-testing block ONLY when worker_nodes is non-empty.
  4. ≥2 workers triggers the PP2 block (mentions ``submit_pp2_ranks``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from metainfer.tasks.port_model.orchestrator.pipeline import PipelineConfig
from metainfer.tasks.port_model.orchestrator.prompts import (
    p5_verify_minimal_prompt,
    p6_port_engine_prompt,
)


def _basic_req() -> Dict[str, Any]:
    return {
        "task_id": "pm-dist",
        "task_type": "port-model",
        "model_params_path": "/tmp/model",
        "target_framework_dir": "/tmp/tfw",
        "reference_sources": [],
        "user_notes": "",
    }


class TestWorkerNodesWiring:
    def test_config_accepts_worker_nodes(self, tmp_path: Path):
        cfg = PipelineConfig(
            state_dir=tmp_path, workspace_dir=tmp_path,
            p1_dir=tmp_path, p2_dir=tmp_path, p3_dir=tmp_path,
            p4_dir=tmp_path, p5_dir=tmp_path, p6_dir=tmp_path,
            memory_dir=tmp_path, dumps_dir=tmp_path,
            target_fw_dir=tmp_path, model_params_path=tmp_path,
            worker_nodes=["wA", "wB"],
        )
        assert cfg.worker_nodes == ["wA", "wB"]

    def test_config_defaults_to_empty_worker_nodes(self, tmp_path: Path):
        cfg = PipelineConfig(
            state_dir=tmp_path, workspace_dir=tmp_path,
            p1_dir=tmp_path, p2_dir=tmp_path, p3_dir=tmp_path,
            p4_dir=tmp_path, p5_dir=tmp_path, p6_dir=tmp_path,
            memory_dir=tmp_path, dumps_dir=tmp_path,
            target_fw_dir=tmp_path, model_params_path=tmp_path,
        )
        assert cfg.worker_nodes == []

    def test_parse_worker_nodes_list(self):
        from metainfer.tasks.port_model.orchestrator.orchestrator import _parse_worker_nodes
        req = {"worker_nodes": ["wA", "wB", "wC"]}
        assert _parse_worker_nodes(req) == ["wA", "wB", "wC"]

    def test_parse_worker_nodes_csv(self):
        from metainfer.tasks.port_model.orchestrator.orchestrator import _parse_worker_nodes
        req = {"worker_nodes": "wA, wB ,, wC"}
        assert _parse_worker_nodes(req) == ["wA", "wB", "wC"]

    def test_parse_worker_nodes_empty(self):
        from metainfer.tasks.port_model.orchestrator.orchestrator import _parse_worker_nodes
        assert _parse_worker_nodes({}) == []
        assert _parse_worker_nodes({"worker_nodes": ""}) == []


class TestPromptInjection:
    def test_p5_no_worker_nodes_skips_distributed_block(self, tmp_path: Path):
        prompt = p5_verify_minimal_prompt(
            req=_basic_req(), workdir=tmp_path, p4_dir=tmp_path,
        )
        assert "submit_pp2_ranks" not in prompt
        assert "Distributed workers available" not in prompt

    def test_p5_two_workers_injects_pp2_block(self, tmp_path: Path):
        prompt = p5_verify_minimal_prompt(
            req=_basic_req(), workdir=tmp_path, p4_dir=tmp_path,
            worker_nodes=["wA", "wB"],
        )
        assert "submit_pp2_ranks" in prompt
        assert "PP2-capable" in prompt
        assert "wA" in prompt and "wB" in prompt

    def test_p6_two_workers_injects_pp2_block(self, tmp_path: Path):
        prompt = p6_port_engine_prompt(
            req=_basic_req(), workdir=tmp_path,
            p3_path=tmp_path / "p3.md", p5_dumps_dir=tmp_path / "dumps",
            iteration=1,
            worker_nodes=["wA", "wB"],
        )
        assert "submit_pp2_ranks" in prompt

    def test_p5_single_worker_injects_remote_worker_block(self, tmp_path: Path):
        prompt = p5_verify_minimal_prompt(
            req=_basic_req(), workdir=tmp_path, p4_dir=tmp_path,
            worker_nodes=["wOnly"],
        )
        # Single worker: no PP2, but should mention remote worker
        assert "submit_pp2_ranks" not in prompt
        assert "Remote worker available" in prompt
        assert "wOnly" in prompt


class TestLaunchConstraintsInjection:
    def test_p5_empty_launch_constraints_omits_block(self, tmp_path: Path):
        prompt = p5_verify_minimal_prompt(
            req=_basic_req(), workdir=tmp_path, p4_dir=tmp_path,
        )
        assert "Launch constraints" not in prompt

    def test_p5_launch_constraints_injected(self, tmp_path: Path):
        req = _basic_req()
        req["launch_constraints"] = (
            "Must use --quantization compressed-tensors\n"
            "378GB model, PP2 must combine with lazy loading"
        )
        prompt = p5_verify_minimal_prompt(
            req=req, workdir=tmp_path, p4_dir=tmp_path,
        )
        assert "Launch constraints" in prompt
        assert "compressed-tensors" in prompt
        assert "lazy loading" in prompt
        assert "AUTHORITATIVE" in prompt

    def test_p6_launch_constraints_injected(self, tmp_path: Path):
        req = _basic_req()
        req["launch_constraints"] = (
            "sglang flags: --trust-remote-code --dtype bfloat16\n"
            "Timeout >= 1800s"
        )
        prompt = p6_port_engine_prompt(
            req=req, workdir=tmp_path,
            p3_path=tmp_path / "p3.md", p5_dumps_dir=tmp_path / "dumps",
            iteration=1,
        )
        assert "Launch constraints" in prompt
        assert "--trust-remote-code" in prompt
        assert "1800s" in prompt

    def test_launch_constraints_doesnt_clobber_distributed_block(self, tmp_path: Path):
        """Both blocks should coexist when both are configured."""
        req = _basic_req()
        req["launch_constraints"] = "PP2 must combine with lazy loading"
        prompt = p5_verify_minimal_prompt(
            req=req, workdir=tmp_path, p4_dir=tmp_path,
            worker_nodes=["wA", "wB"],
        )
        assert "Launch constraints" in prompt
        assert "PP2-capable" in prompt
        assert "submit_pp2_ranks" in prompt
