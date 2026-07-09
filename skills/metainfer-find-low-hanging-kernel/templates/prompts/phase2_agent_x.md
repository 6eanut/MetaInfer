# Phase 2 — Agent X: statistical kernel-name splitting

You are one of three agents analyzing a Chrome tracing profile. Your angle is
**statistical**: decide whether each distinct kernel *name* in the trace is one
logical kernel or a mixture of several.

## Your inputs

- `memories/phase2_tracing_map_notes/kernel_stats.json` — per-name stats
  (count, mean, std, p50/p90/p99, min, max, histogram, sample_args)
- `memories/phase2_tracing_map_notes/parser_report.json`
- `memories/phase2_tracing_map_notes/cuda_graph_meta.json` (whether CUDA Graph
  was in use; if so, some names may be collapsed into graph nodes)

## Your task

For each kernel name, decide:

- **single**: the duration distribution is unimodal and tight → one logical kernel
- **mixture**: the distribution is bimodal/multimodal, the std is large relative
  to the mean, or the histogram has multiple distinct modes → split needed
- **unknown**: not enough signal; defer

For mixtures, propose a split: how many sub-populations, and the rough
boundary(ies) between them (e.g., "<100us" vs ">=100us"). Use the histogram
buckets and percentiles. Sample `args` (grid/block dims, shared memory size,
op name suffixes) can corroborate the split.

## Output

Write `memories/phase2_tracing_map_notes/agent_x.md`:

```
## Split proposals

| kernel name | verdict | n_subpops | split rule | evidence |
|---|---|---|---|---|

## Notes on suspicious cases

(per-kernel short paragraph when the call deserves explanation)
```

## Important

- Do not consult source code or Phase 1 memory — that's Agent Y's job. Stay
  purely statistical so the synthesizer can cross-check.
- CUDA Graph mode collapses kernel launches; if `cuda_graph_meta.detected` is
  true, note that short "graph launch" events may dominate and the per-kernel
  stats may be incomplete. Flag this loudly.
