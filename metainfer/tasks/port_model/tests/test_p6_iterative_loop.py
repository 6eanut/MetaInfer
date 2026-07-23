"""Tests for the P6 iterative port-and-test loop rework.

Verifies that:
  1. ``p6_port_engine_prompt`` contains the structured inner loop
     (LAUNCH → DIAGNOSE → REPLACE → RELAUNCH → DUMP-CMP → BISECT).
  2. The operator replacement strategy hierarchy is present and
     ordered (flag/env → Triton → pure-torch → P4 reference).
  3. The "operator unsupported on my hardware" diagnostic guidance
     is present.
  4. Dump-driven bisection procedure is present.
  5. The verdict JSON schema lists ``inner_attempts`` and
     ``operator_replacements``.
  6. ``format_prev_p6_verdict`` correctly renders structured handover
     state for the next P6 iteration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from metainfer.tasks.port_model.orchestrator.prompts import (
    format_prev_p6_verdict,
    p6_port_engine_prompt,
)


def _basic_req() -> Dict[str, Any]:
    return {
        "task_id": "pm-p6-loop",
        "task_type": "port-model",
        "model_params_path": "/tmp/model",
        "target_framework_dir": "/tmp/tfw",
        "reference_sources": [],
        "user_notes": "",
    }


def _build_prompt(prev_failure: str = "") -> str:
    return p6_port_engine_prompt(
        req=_basic_req(),
        workdir=Path("/tmp/wd"),
        p3_path=Path("/tmp/p3.md"),
        p5_dumps_dir=Path("/tmp/dumps"),
        iteration=1,
        prev_failure=prev_failure,
    )


class TestInnerLoopPlaybook:
    def test_prompt_mentions_inner_loop_concept(self):
        p = _build_prompt()
        assert "iterative port-and-test loop" in p
        assert "NOT a one-shot launch" in p

    def test_prompt_lists_eight_step_cycle(self):
        """Each of the 8 inner-loop steps should be present."""
        p = _build_prompt()
        for step in (
            "LAUNCH",
            "DIAGNOSE",
            "REPLACE",
            "RELAUNCH",
            "INFER",
            "DUMP-CMP",
            "BISECT",
            "STOP",
        ):
            assert step in p, f"missing inner-loop step: {step}"

    def test_prompt_has_strategy_hierarchy_ordered(self):
        """Strategies must appear in the prescribed priority order."""
        p = _build_prompt()
        idx_flag = p.find("Framework-native flag")
        idx_triton = p.find("Triton re-implementation")
        idx_torch = p.find("Pure PyTorch reference")
        idx_p4 = p.find("P4 reference impl")
        # All present, in this order.
        assert -1 < idx_flag < idx_triton < idx_torch < idx_p4

    def test_prompt_has_hardware_unsupported_diagnosis(self):
        p = _build_prompt()
        assert "operator unsupported on my hardware" in p.lower() or \
               "Diagnosing \"operator unsupported" in p
        # Concrete signals the agent should recognise.
        assert "gfx928" in p or "get_device_capability" in p
        assert "METAINFER_OPS_FORCE" in p  # canonical gate name

    def test_prompt_has_dump_driven_bisection(self):
        p = _build_prompt()
        assert "Dump-driven bisection" in p
        assert "first layer" in p.lower() or "FIRST divergent" in p
        # The agent must know to record the culprit.
        assert "similarity_first_bad_layer" in p

    def test_prompt_mentions_p4_reference_as_fallback_source(self):
        p = _build_prompt()
        # P4 lives beside the workdir under ../p4/
        assert "../p4/" in p
        assert "P4 minimal framework" in p

    def test_prompt_enforces_add_only_rule(self):
        p = _build_prompt()
        assert "ADD-ONLY" in p or "ADD-only" in p
        assert "Never delete" in p or "never delete" in p


class TestVerdictSchema:
    def test_verdict_schema_includes_inner_attempts(self):
        p = _build_prompt()
        assert '"inner_attempts"' in p

    def test_verdict_schema_includes_operator_replacements(self):
        p = _build_prompt()
        assert '"operator_replacements"' in p
        # Schema documents each sub-field.
        for sub in ("env_var", "strategy", "commit_sha", "reason"):
            assert sub in p, f"operator_replacements missing sub-field docs: {sub}"

    def test_outcome_mapping_documents_needs_repair_semantics(self):
        p = _build_prompt()
        # needs_repair MUST carry forward the structured state.
        assert "needs_repair" in p
        assert "operator_replacements" in p
        assert "similarity_first_bad_layer" in p


class TestFormatPrevP6Verdict:
    def test_empty_verdict_returns_empty_string(self):
        assert format_prev_p6_verdict({}) == ""
        assert format_prev_p6_verdict(None) == ""  # type: ignore[arg-type]

    def test_renders_reason_only(self):
        out = format_prev_p6_verdict({"reason": "boot crashed on aiter import"})
        assert "reason: boot crashed on aiter import" in out

    def test_renders_similarity_summary(self):
        out = format_prev_p6_verdict({
            "similarity_min": 0.42,
            "similarity_first_bad_layer": 13,
            "similarity_first_bad_row": 1,
        })
        assert "similarity_min=0.42" in out
        assert "first_bad_layer=13" in out
        assert "first_bad_row=1" in out

    def test_renders_operator_replacements_list(self):
        out = format_prev_p6_verdict({
            "reason": "gfx928 unsupported",
            "inner_attempts": 3,
            "operator_replacements": [
                {
                    "op": "GlmMoeDSAAttention.forward",
                    "strategy": "triton",
                    "env_var": "METAINFER_OPS_FORCE_DSA_TRITON",
                    "commit_sha": "abc1234",
                    "reason": "tilelang fp8 MMAC needs gfx938+; wrote triton attention",
                },
                {
                    "op": "CompressedTensorsWNA16TritonMoE.forward",
                    "strategy": "pure-torch",
                    "env_var": "METAINFER_OPS_USE_TORCH",
                    "commit_sha": "def5678",
                    "reason": "aiter fused gemm unavailable",
                },
            ],
        })
        assert "inner_attempts_this_iter=3" in out
        assert "operator_replacements_tried:" in out
        assert "GlmMoeDSAAttention.forward" in out
        assert "strategy=triton" in out
        assert "METAINFER_OPS_FORCE_DSA_TRITON" in out
        assert "strategy=pure-torch" in out
        assert "CompressedTensorsWNA16TritonMoE.forward" in out

    def test_replacements_with_missing_fields_do_not_crash(self):
        out = format_prev_p6_verdict({
            "operator_replacements": [
                {"op": "X"},             # minimal
                "garbage",               # non-dict entry, must be skipped
                {"strategy": "triton"},  # no op
            ],
        })
        assert "op=X" in out
        assert "strategy=triton" in out
        # Non-dict entry was skipped silently.
        assert "garbage" not in out
