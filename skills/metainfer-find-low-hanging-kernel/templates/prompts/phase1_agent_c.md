# Phase 1 — Agent C: request-path trace, end to end

You are one of three independent agents. Agent A starts from `config.json` and
greps for the model registration. Agent B resolves flag-driven branches. **You**
start from the framework entry point and follow a single inference request end
to end, listing every kernel launch site in execution order.

## Your inputs

- `framework_src` — the source tree
- `cli_args` — the launch command (this tells you the entry point)
- `log_file` — startup log; the first lines usually print the entry module and
  the model loader path

## Your task

1. Identify the entry point (CLI script, `__main__.py`, server start, etc.).
2. Trace forward: entry → model load → tokenizer → request handler → prefill
   → decode loop → sampling → output.
3. At each step record every kernel/operator call site in execution order.
   For each: operator name, file:line, expected input/output dtypes + shapes
   (parametric in `B`, `S`, `Sq`, `L`, etc.).
4. Note where the path forks based on flags — but do NOT try to resolve which
   fork was taken (Agent B does that). Just mark the fork and the condition.
5. Capture both **prefill** and **decode** paths if they differ.

## Output

Write `memories/phase1_code_map_notes/agent_c.md`:

```
## Entry point

- module: <file:line>
- invoked as: <cli_args>

## Prefill path (ordered)

| order | step | operator | file:line | in shapes | out shapes | fork_cond |
|---|---|---|---|---|---|---|

## Decode path (ordered, if different)

| order | step | operator | file:line | in shapes | out shapes | fork_cond |
|---|---|---|---|---|---|---|

## Forks unresolved here (deferred to Agent B)

| location | condition | options |
|---|---|---|
```

## Important

- Walk the actual request path. Don't just enumerate every function in the
  model class — only the ones that actually run during a forward pass.
- If you find a loop body (per-layer), record it once and note the loop count
  source (`L = num_hidden_layers`).
