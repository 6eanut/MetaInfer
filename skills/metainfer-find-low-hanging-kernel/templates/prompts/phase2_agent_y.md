# Phase 2 — Agent Y: source-correlation disambiguation

You are one of three agents analyzing a Chrome tracing profile. Your angle is
**source correlation**: for each kernel name, map it back to its caller site(s)
in the framework, using Phase 1's code map, then group by
`(caller_site, tensor_shape_signature)` instead of by name alone.

## Your inputs

- `memories/phase2_tracing_map_notes/kernel_stats.json`
- `memories/phase1_code_map.md` — the canonical operator call-site table
- the framework source tree (read-only)
- `memories/phase2_tracing_map_notes/parsed_events.jsonl.gz` (per-event records;
  you may sample them via `scripts/parse_tracing.py`-style reads, or use
  `zcat | head` / `rg` via Bash)

## Your task

1. For each kernel name in `kernel_stats.json`, find all call sites in the
   Phase 1 table whose operator matches (or contains) that name. There may be
   several.
2. If a name has exactly one call site, it's unambiguous.
3. If a name has multiple call sites (e.g. `aten::_softmax` called from both
   attention scoring and final logits), then for each call site derive a
   **shape signature** from the Phase 1 row's input shapes. Group events by
   the (call_site, shape_signature) pair.
4. For each group, you may need to peek at `sample_args` in
   `kernel_stats.json` (grid/block dims, shared memory, etc.) to confirm which
   group corresponds to which call site.
5. Output a per-name table of logical sub-kernels keyed by call site.

## Output

Write `memories/phase2_tracing_map_notes/agent_y.md`:

```
## Name → call-site mapping

| kernel name | call_sites (file:line) | shape signature per site | sample_args discriminator |
|---|---|---|---|

## Names that need splitting because of multiple call sites

| kernel name | proposed split (site -> subpop) | confidence |
|---|---|---|
```

## Important

- Do not run your own statistical analysis — that's Agent X. Use their output
  only if it has been written; otherwise defer.
- Cite Phase 1 file:line for every call site. If a kernel name has no
  corresponding call site in Phase 1, flag it loudly — that usually means
  Phase 1 missed an operator or this is a framework-internal helper kernel.

## Tensor parallelism — required check

Before producing any shape signature, **check whether TP was active** by reading
`cli_args`, `env_vars`, and the launch log (search for `tensor_parallel`,
`tp_size`, `world_size`, `rank`, `local_rank`). If TP>1:

- Every per-call-site shape signature must be **per-rank**, not global. Derive
  per-rank shapes from the framework's TP loader code — never from the model's
  `config.json` alone.
- Classify each weight as `column_parallel` / `row_parallel` / `replicated` /
  `vocab_parallel` by reading the loader. Typical defaults for decoder-only LMs
  (must verify in source):
  - column-parallel: `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`
  - row-parallel: `down_proj`, sometimes `o_proj`
  - replicated: layernorm / RMSNorm weights
  - vocab-parallel or replicated: `embed_tokens`, `lm_head` (read the code)
- Attention head partition: `Hh_local = Hh / TP`, and `Hkv_local` follows the
  framework's GQA-on-TP policy (split, replicated, or rounded up).
- **The trace contains TP replicas.** When the same call site runs on every
  rank with the same per-rank shape, the resulting kernel events are
  *replicas*, not different populations. Use `args.rank` / `pid` / `stream` to
  recognize and collapse them — do NOT split them into different logical
  kernels.
- Conversely, two *different* call sites that share a kernel name still need
  splitting. The test is "different source location or different per-rank
  shape signature", not "different rank".
