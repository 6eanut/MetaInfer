"""Tests for the deterministic create-time requirement-conversation engine.

The engine must never fabricate structure it can't see: it extracts categorical
slots from option labels + a small lexicon, refuses to auto-resolve conflicts
(HIP *and* Triton), keeps asking for required fields, and only settles into a
flat ``answers`` that would pass the shell form validator (required present,
select values within declared option labels).
"""

from __future__ import annotations

import pytest

from metainfer.tasks.opt_operator.orchestrator.converse import (
    interpret,
    settle,
)

RMSNORM_CONTRACT_YAML = """\
name: RMSNorm
entrypoint: rmsnorm_kernel
inputs:
  - {name: X, dtype: fp16, shape: [B, S, H]}
  - {name: W, dtype: fp16, shape: [H]}
outputs:
  - {name: Y, dtype: fp16, shape: [B, S, H]}
shapes:
  B: 1
  S: [2048, 8192]
  H: [128, 512]
"""


def _schema():
    """Normalized schema mirroring opt_operator's create form (labels as the
    frontend + validator see them)."""
    def field(key, label, ftype, options=None, required=False):
        f = {"key": key, "label": label, "type": ftype,
             "required": required, "options": options or None}
        return f
    return {
        "type": "opt-operator",
        "label": "Optimize operator (K100)",
        "fields": [
            field("input_mode", "Input mode", "select", [
                {"label": "I have a kernel source"},
                {"label": "Spec only (from scratch)"},
            ], required=True),
            field("kernel_language", "Stack", "select", [
                {"label": "HIP"}, {"label": "Triton"}], required=True),
            field("shape_mode", "Shape mode", "select", [
                {"label": "Targeted (specific shape)"},
                {"label": "General (any shape)"}], required=True),
            field("operator_contract", "Contract", "textarea", required=True),
            field("kernel_source", "Kernel src", "file"),
            field("num_gpus", "GPUs", "select", [
                {"label": "0 (auto)"}, {"label": "1"},
                {"label": "2"}, {"label": "4"}]),
            field("max_iterations", "Max iters", "select", [
                {"label": "6"}, {"label": "10"},
                {"label": "20"}, {"label": "50"}]),
        ],
    }


def test_opening_asks_and_reports_missing():
    it = interpret(_schema())
    assert it.complete is False
    # every required field is called out
    for key in ("operator_contract", "kernel_language", "shape_mode",
                "input_mode"):
        assert key in it.missing, key
    assert it.assistant  # the opening greeting


def test_conflicting_stack_is_not_auto_resolved():
    it = interpret(_schema(), request="please optimise, either HIP or Triton")
    # both aliases appear -> no single label may be chosen silently
    assert it.answers.get("kernel_language") is None
    assert "kernel_language" in it.missing or "kernel_language" in it.conflict


def test_extracts_triton_general_source():
    it = interpret(
        _schema(),
        request="optimise my RMSNorm kernel in triton, general for any shape, "
                "I have the kernel source")
    assert it.answers["kernel_language"] == "Triton"
    assert it.answers["shape_mode"] == "General (any shape)"
    assert it.answers["input_mode"] == "I have a kernel source"
    # the contract is still missing (no YAML pasted yet)
    assert "operator_contract" in it.missing


def test_targeted_specialise_and_spec_only():
    it = interpret(
        _schema(),
        request="spec only from scratch, target a single fixed shape, hip")
    assert it.answers["input_mode"] == "Spec only (from scratch)"
    assert it.answers["shape_mode"] == "Targeted (specific shape)"
    assert it.answers["kernel_language"] == "HIP"


def test_number_fields_parse_counts():
    it = interpret(_schema(), request="use 2 gpus, up to 6 iterations")
    assert it.answers["num_gpus"] == "2"
    assert it.answers["max_iterations"] == "6"


def test_contract_only_accepted_when_yaml_present():
    it = interpret(_schema(), request="RMSNorm normalisation over hidden dim")
    assert "operator_contract" in it.missing       # prose alone is not a contract
    it2 = interpret(_schema(), request=RMSNORM_CONTRACT_YAML)
    assert it2.answers.get("operator_contract", "").strip().startswith("name:")


def test_contract_is_isolated_from_surrounding_prose():
    # Chat prose may sit above a pasted contract. The value extracted for
    # operator_contract must be the YAML block alone, not the whole message —
    # the contract later feeds OperatorContract.load.
    msg = ("spec only from scratch, hip, general shapes — here is the contract:\n"
           + RMSNORM_CONTRACT_YAML + "\n(that covers it, thanks)")
    it = interpret(_schema(), request=msg)
    contract = it.answers.get("operator_contract", "")
    assert contract.startswith("name: RMSNorm")
    assert "spec only from scratch" not in contract  # prose did not leak in
    assert "thanks" not in contract
    assert "hip" not in contract.lower()


def test_contract_fenced_block_preferred():
    msg = ("please do:\n```yaml\nname: softmax\ninputs:\n  - name: x\n"
           "outputs:\n  - name: y\n```\nand target general")
    it = interpret(_schema(), request=msg)
    contract = it.answers.get("operator_contract", "")
    assert contract.startswith("name: softmax")
    assert "```" not in contract


def test_complete_when_all_required_pinned():
    # language + shape first, then input mode, then paste the contract
    a = interpret(_schema(), request="triton, general")
    assert a.answers["kernel_language"] == "Triton"
    assert a.answers["shape_mode"] == "General (any shape)"
    assert set(a.missing) == {"input_mode", "operator_contract"}
    b = interpret(_schema(), request="spec only", answers=a.answers)
    assert set(b.missing) == {"operator_contract"}
    c = interpret(_schema(), request=RMSNORM_CONTRACT_YAML, answers=b.answers)
    assert c.complete is True
    assert c.missing == []
    assert c.conflict == []


def test_settle_returns_flat_answers_and_transcript():
    # carry answers through turns + a transcript, then settle
    ans = interpret(_schema(), request="hip, targeted single shape, "
                                       "spec only").answers
    ans = interpret(_schema(), request=RMSNORM_CONTRACT_YAML,
                    answers=ans).answers
    transcript = [
        {"role": "user", "text": "triton from scratch, single shape"},
        {"role": "assistant", "text": "pasted the contract"},
    ]
    settled = settle(_schema(), ans, transcript=transcript)
    # flat answers that pass the validator: labels, not hints
    assert settled["answers"]["kernel_language"] == "HIP"
    assert settled["answers"]["shape_mode"] == "Targeted (specific shape)"
    assert settled["answers"]["input_mode"] == "Spec only (from scratch)"
    assert settled["answers"]["operator_contract"].startswith("name: RMSNorm")
    assert "[user]" in settled["raw_request"]   # dialogue retained
    assert "[assistant]" in settled["raw_request"]


def test_settle_refuses_incomplete():
    with pytest.raises(ValueError):
        settle(_schema(), interpret(_schema(),
                                    request="triton").answers)
