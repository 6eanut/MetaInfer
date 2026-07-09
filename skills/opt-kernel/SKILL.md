---
name: opt-kernel
description: Optimize a custom GPU kernel (attention, quantized GEMM, norm, RoPE, sampling) for a specific target shape and platform. Use when the user wants to speed up an existing kernel without changing its math.
---

# opt-kernel

Optimize an existing GPU kernel against a perf target on a specific
GPU and tensor shape, without breaking correctness. Driven by the
deterministic MetaInfer orchestrator.

Pure Claude Code skill — no plugin, no wrapper. The skill directory
carries its own `questions.yaml` next to this SKILL.md. The orchestrator
launcher lives in the sibling `gen-infer-framework` skill.

## When to invoke

- "optimize my attention kernel for H100, batch=8, seq=2048"
- "make this GEMM faster on A100"
- "tune the RoPE kernel for shape X"

If the user wants to build a new kernel from scratch (not optimize an
existing one), this is the wrong skill — start with `gen-infer-framework`
or write the kernel inline.

## What it does

1. **Interview** from `questions.yaml` next to this SKILL.md (operator,
   target framework, GPU, dtype, shape, perf metric, baseline perf).
2. **Freeze** answers into `requirements.json`.
3. **Hand off** to the orchestrator:
   `python <gen-infer-framework-skill>/run.py run requirements.json`
   (the orchestrator package lives with `gen-infer-framework`, not here).

The orchestrator runs `A: plan → B: implement + review → C: test →
(pass) → D: optimize`, retrying from B on failure. Each D step stacks
one or two high-impact optimizations on top of the prior iteration.

## Correctness contract

Unlike `gen-infer-framework`, the agent writes its own `test.sh`. The
script must emit exactly one JSON line on stdout:

```json
{"passed": true, "perf": {"tokens_per_sec": 123.4, "ms_per_op": 0.56}, "notes": "..."}
```

On failure:

```json
{"passed": false, "error": "<short reason>", "traceback": "<last 4KB>"}
```

Exit code 0 even on failure — the orchestrator parses the JSON.

The D (optimize) phase reads `perf` from the prior C step and tries to
improve it. Correctness regressions fail the next C step and roll back
to B.

## Knowledge base

Kernel tuning notes, fused-kernel patterns, and memory-layout tips
live in `../gen-infer-framework/notebooks/03_operators/` and
`../gen-infer-framework/notebooks/06_experience/`. Sub-agents are told
to consult these before proposing optimizations.
