# Phase 2 — Agent Z: tracing-internal disambiguation

You are one of three agents analyzing a Chrome tracing profile. Your angle is
**internal to the trace itself**: use `stream`, `pid`, `tid`, `cat`, and
`args` fields (grid/block dims, shared memory size, op name suffixes, CUDA
Graph `node_seq` IDs, etc.) to disambiguate same-named kernel calls when CPU
stacks are absent.

This is the most important angle when the trace was captured **without** CPU
stacks (a common case — CPU stacks add overhead and many production captures
omit them).

## Your inputs

- `memories/phase2_tracing_map_notes/kernel_stats.json` (per-name stats and
  sample_args)
- `memories/phase2_tracing_map_notes/parsed_events.jsonl.gz` (raw per-event
  records; sample via `zcat | head` / `rg` / a small Python loop)
- `memories/phase2_tracing_map_notes/cpu_stack_summary.json`
- `memories/phase2_tracing_map_notes/cuda_graph_meta.json`

## Your task

1. Decide whether CPU stacks are usable. If `cpu_stack_summary.count` is tiny
   or zero relative to the number of GPU events, declare "no CPU stacks" and
   rely on args/pid/tid/stream instead.
2. For CUDA Graph captures: when `cuda_graph_meta.detected` is true, kernels
   launched inside a graph may share a `cat`, an `args.node_seq`, or be
   adjacent in `ts` within a graph-replay event. Use this to group.
3. For each kernel name with multiple call populations, identify
   discriminators from `args` (block_x, block_y, num_regs, shared_mem,
   stream id, correlation id, op suffix like "_fwd" / "_bf16" / "_fp8").
4. Output a per-name proposal keyed by discriminator values.

## Output

Write `memories/phase2_tracing_map_notes/agent_z.md`:

```
## CPU stack availability

- usable: yes / no / partial
- reason:

## CUDA Graph

- detected: yes / no
- implications for kernel stats: ...

## Discriminator proposals

| kernel name | discriminator field | observed value clusters | proposed subpop |
|---|---|---|---|
```

## Important

- Don't trust a single field. Cross-check at least two of {stream, tid, args}
  before proposing a split.
- If you find that two same-named kernels really do identical work (same dims,
  same stream, same call frequency), say so — that's also a valid conclusion.
