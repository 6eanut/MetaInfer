# Phase 2 — Synthesizer

You are the synthesizer. Three agents (X, Y, Z) wrote
`memories/phase2_tracing_map_notes/agent_{x,y,z}.md`. Merge them into the
canonical `memories/phase2_tracing_map.md`.

## Required sections

### 1. Capture characteristics

- CPU stacks: usable yes/no/partial + reason
- CUDA Graph: in use yes/no + implications for the stats
- ts range, total events, total complete events (from `parser_report.json`)

### 2. Logical kernel table (the single source of truth)

| logical_kernel_id | display_name | framework_call_sites (file:line list) | call_count | mean_us | p50_us | p90_us | p99_us | std_us | histogram_summary | shape_signature | confidence |

Rules:

- `logical_kernel_id` is a stable id you mint, e.g. `k01`, `k02`, ... Phase 3
  will reference these from `source_kernel_ids`.
- One logical_kernel_id per (kernel name × resolved call site × shape
  signature). If agents X/Y/Z agree a name is one population, that's one row.
- `framework_call_sites` is a list because some logical kernels really do have
  one call site; if multiple, list all but note that they are
  indistinguishable from the trace alone.
- `histogram_summary` is a short human-readable string like "unimodal p50≈p90"
  or "bimodal: modes at 12us and 480us".

### 3. Names that were split

For every name that ended up as >1 logical kernel, write a one-paragraph
explanation citing which agent's evidence drove the split.

### 4. Confidence + open questions

Same convention as Phase 1 synthesizer: `high` / `medium` / `low`. Flag every
name where none of the three agents could disambiguate, and explain what
additional information would resolve it.

## Rules

- Do not invent statistics. Every number comes from `kernel_stats.json`. If a
  logical kernel is a subset of a name's events, recompute mean/p90/etc. from
  `parsed_events.jsonl.gz` using a short Python filter — don't guess.
- Preserve disagreements in footnotes.

## TP section (mandatory if TP>1 was detected)

Include a `## TP summary` section with:

- `tp_size`: N
- `evidence`: which cli arg / env var / log line established it
- per-weight policy table:

  | weight | policy (column_parallel / row_parallel / replicated / vocab_parallel) | framework loader file:line | per-rank dim |
  |---|---|---|---|

- `Hh_local`, `Hkv_local`, `I_local`, `V_local` definitions (formulas + numeric
  values for this run)
- confirmation that all `shape_signature` values in the logical-kernel table
  are per-rank, not global
- confirmation that TP-replica kernel events were collapsed (not split) — and
  the field used to recognize replicas (`args.rank`, `pid`, `stream`)

If TP=1, the section is just one line: `TP not active; global shapes apply.`
