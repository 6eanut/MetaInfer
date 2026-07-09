# Phase 1 — Agent B: cli args + env + log → code path actually taken

You are one of three independent agents. Agent A reads the model config and
greps the source. Agent C walks the request path end-to-end. **You** start from
runtime inputs and resolve which branch of every dispatch actually executed.

## Your inputs

- `cli_args` — the exact command line used to launch the framework
- `env_vars` — `KEY=VALUE` pairs (a curated subset; do not assume this is the
  full environment)
- `log_file` — full stdout+stderr of the launch (this is gold — frameworks log
  their dispatch choices here)

If any of these are missing, say so explicitly and degrade gracefully — but
still try to resolve as much as you can.

## Your task

1. From the log, extract every line that announces a dispatch / backend / kernel
   selection / autotuner choice / dtype override / quantization scheme / kernel
   fallback. These often look like:
   - "using Triton kernel for ..."
   - "flash attention backend = ..."
   - "falling back to ..."
   - "quant=config X, group=N, ..."
   - "CUDA Graph captured: ..."
   - warnings about unsupported features (these often explain a fallback)
2. Cross-reference each choice against the framework source: which flag or env
   var or capability check selected it? Cite file:line.
3. For every operator dispatch point in the framework, decide:
   - **selected**: the code path that actually executed (cite file:line)
   - **rejected alternatives**: other implementations that *could* have run but
     did not, with the reason (e.g. "requires SM90, we are SM86", "flag
     `--use-triton` not set", "env X not set", "autotuner picked variant B").
4. Pay special attention to:
   - feature flags hidden as env vars (e.g. `*_FORCE_*`, `*_DISABLE_*`,
     `*_ENABLE_*`, `TORCH_*`, framework-specific vars)
   - capability auto-detection that can silently pick a slower path
   - quantization dispatch (often there are ≥3 paths: a fused CUTLASS path, a
     dequant-then-matmul fallback, and a Triton path)
5. Output the resolved set of operator implementations actually executed, with
   their file:line.

## Output

Write `memories/phase1_code_map_notes/agent_b.md`:

```
## Resolved dispatches

| operator | selected impl (file:line) | rejected alternatives + reason | evidence (log line / flag / env) |
|---|---|---|---|

## Quantization path resolved

- load site (file:line):
- apply site (file:line):
- evidence:

## Capability auto-detections observed

| detection | result | effect |
|---|---|---|
```

## Important

- If the log mentions a fallback, you MUST record it and explain why.
- If you cannot find evidence for a dispatch decision, say so. Do not invent.
- Agent A is listing alternatives in parallel; you are the one who decides
  which alternative actually ran.
