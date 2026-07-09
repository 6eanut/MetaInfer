# Phase 1 — Synthesizer

You are the synthesizer. Three agents (A, B, C) have written raw notes under
`memories/phase1_code_map_notes/agent_{a,b,c}.md`. Read all three plus
re-consult the framework source as needed. Produce a single canonical memory
file: `memories/phase1_code_map.md`.

## Required sections

### 1. Model architecture

- architecture name(s) (from `config.json` → `architectures`)
- key dims (H, I, L, V, Hh, Hkv, Dh, max_pos)
- dtype (runtime, after CLI/log overrides if any)
- quantization scheme and parameters (block/group size, num bits, scheme)

### 2. Operator call-site table (the single source of truth)

| step_purpose | operator | framework_file:line | selected_by (flag/env/autodetect + value) | input dtype/shape (parametric) | output dtype/shape | rejected_alternatives + reason |

Every row must cite a file:line that exists in the framework source. If
agents disagree, prefer the one with the most concrete log evidence and say so
in a footnote.

### 3. Quantization parameter handling

- where scales/zp/group meta are loaded (file:line)
- where they are applied during inference (file:line)
- the dispatch condition that selected this path
- the rejected alternatives and why

### 4. Confidence + open questions

For every row, label confidence: `high` (multiple agents agree + log evidence),
`medium` (single agent + plausible), `low` (inferred, no direct evidence). For
each non-`high` row, write one sentence on what would be needed to raise it.

### 5. The mandatory fallback question

> "Given cli_args + env_vars + log_file, are there any other implementations of
> these operators that the runtime could have selected but did not?"

Answer this explicitly. List each alternative and the reason it was not
selected. This is the single most important part of this file — downstream
phases assume the answer is complete.

## Rules

- Do not invent file:line citations. If you cannot find one, mark `low`.
- Preserve disagreements in footnotes; do not silently collapse them.
- Keep the file machine-parseable where possible (markdown tables), since
  Phase 2 and Phase 3 will read it.
