---
name: port-model
description: Port a model architecture into an existing inference framework (vLLM, SGLang, TensorRT-LLM). Use when the user wants to add a new model to an existing serving stack, not build a framework from scratch.
---

# port-model

Port a new model architecture into an existing inference framework
(vLLM / SGLang / TensorRT-LLM). Driven by the deterministic MetaInfer
orchestrator.

Pure Claude Code skill — no plugin, no wrapper. The skill directory
carries its own `questions.yaml` next to this SKILL.md. The orchestrator
launcher lives in the sibling `gen-infer-framework` skill.

## When to invoke

- "port DeepSeek-V3 into vLLM"
- "add Qwen3 support to SGLang"
- "get Mistral Mixtral working in TensorRT-LLM"

If the user wants to build a brand-new framework (no vLLM/SGLang/TRT-LLM
as the host), use `gen-infer-framework` instead. If they want to speed
up an already-working kernel, use `opt-kernel`.

## What it does

1. **Interview** from `questions.yaml` next to this SKILL.md (target
   framework, model, attention variant, MoE pattern if any, dtype,
   hardware).
2. **Freeze** answers into `requirements.json`.
3. **Hand off** to the orchestrator:
   `python <gen-infer-framework-skill>/run.py run requirements.json`
   (the orchestrator package lives with `gen-infer-framework`, not here).

The orchestrator runs the standard ABCD loop. Because correctness here
is framework-specific, the agent writes its own `test.sh` (see below).

## Correctness contract

The agent writes `test.sh` that actually exercises the ported model
inside the host framework (not a stub). Output contract:

```json
{"passed": true, "perf": {"tokens_per_sec": 123.4}, "notes": "..."}
```

```json
{"passed": false, "error": "<short>", "traceback": "<last 4KB>"}
```

Exit code 0 even on failure. Reviewer sub-agent checks for silent
assumptions: wrong stride, dtype mismatch, missing mask, incorrect
MoE routing, etc.

## Knowledge base

Model-specific notes live in `../gen-infer-framework/notebooks/02_model_specifics/`.
Framework integration patterns live in `../gen-infer-framework/notebooks/01_framework_design/`
and `../gen-infer-framework/notebooks/05_inference_service/`. Sub-agents
are told to consult both before writing code.
