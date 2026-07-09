---
name: metainfer-find-low-hanging-kernel
description: Use when the user wants to discover which compute operator / kernel inside an LLM inference framework run has the largest realistic optimization headroom. The skill consumes a Chrome-tracing profile, the model directory (config.json + weight files), and the inference framework source tree, then reconstructs a model-specific execution flow graph and ranks the kernels that are worth optimizing first ("low-hanging fruit"). Only run when the user explicitly asks for this analysis. The skill asks the user for inputs exactly once at the very start (Phase 0), then runs Phase 1 → 2 → 3 → 4 end-to-end without ever pausing to ask "shall I proceed?" — intermediate phases must execute autonomously and report only at the end.
---

# metainfer-find-low-hanging-kernel

This skill walks an inference framework run from a Chrome-tracing profile back to source code, builds a model-specific execution-flow graph, and highlights the kernels with the biggest realistic optimization headroom.

## Hard rules — read before doing anything

1. **Never modify user-provided data.** That means: the framework source tree, the model directory (config.json, weight files, tokenizer files), the tracing file, the run logs, the CLI args, the env vars. Everything you receive is **read-only**.
2. **Only operate inside the new skill's folder** (`skills/metainfer-find-low-hanging-kernel/`) plus the skill-scoped working directory you create at run time (default `<CWD>/.metainfer_runs/<run_id>/`). Do **not** write into the framework source, the model directory, or any other location outside the skill's purview.
3. **Do not read the tracing file directly.** Tracing files are usually `.gz`-compressed and can be hundreds of MB. Use `scripts/parse_tracing.py` for all parsing.
4. **Do not read model weight files directly.** Use `scripts/parse_safetensors.py` / `scripts/parse_ckpt.py` to extract metadata only.
5. **Each major step runs in a fresh Agent.** Do not let the orchestrator's context drift; spawn sub-agents.
6. **Cross-validate every load-bearing conclusion** with ≥2 independent agents that approach the question from different angles.
7. **Run end-to-end. No mid-execution "shall I proceed?" questions.** The only legitimate `AskUserQuestion` calls are at the very start of Phase 0: (a) the saved-config reuse prompt, (b) the input-collection prompts themselves. Once Phase 0 has the inputs (either freshly collected or reused from `.metainfer-inputs.json`), the orchestrator must execute Phase 1 → 2 → 3 → 4 to completion with **zero further user prompts**. In particular, do NOT ask "shall I proceed to Phase N?", do NOT ask the user to confirm a graph-validation patch, do NOT ask whether to retry a validation round — those decisions are made by the script's own rules (e.g. retry until clean or 5 rounds, then surface in the final report). Only the final summary is shown to the user, at the very end.

## Phase 0 — collect inputs (interactive)

### Step 0a — check for a saved config

The very first thing the orchestrator does on launch is check whether the current working directory already has a saved inputs file:

```
python3 scripts/inputs_io.py exists   # exit 0 if <CWD>/.metainfer-inputs.json exists
```

- If it does **not** exist, jump to Step 0b (fresh collection).
- If it **does** exist, immediately validate it (`python3 scripts/inputs_io.py validate`) and show its contents (`python3 scripts/inputs_io.py show`), then ask the user with `AskUserQuestion`:

  > "Found a saved inputs file at `.metainfer-inputs.json` (saved at <timestamp>). Use it?"

  Options:

  - "Use saved config (Recommended)" — load the inputs from the file, echo them back to the user as a final confirmation prompt, then proceed straight to Phase 1. Skip Step 0b entirely.
  - "Start fresh" — ignore the saved file and run Step 0b. After collection, **overwrite** the saved file (this is the explicit "重新开始" path the user is opting into, so `--overwrite` is appropriate).

  If validation failed on the saved file (e.g. a referenced path no longer exists), surface the per-field errors and default to "Start fresh" — the saved file is stale and the user needs to re-answer.

### Step 0b — fresh collection (only when starting fresh)

Greet the user briefly, then use `AskUserQuestion` to collect the following. Keep the question count low — bundle related fields.

Required:

- `tracing_file` — path to a Chrome-tracing profile. May be `.json`, `.json.gz`, or a plain `.gz`. The parser auto-detects.
- `model_dir` — directory that contains `config.json` and the model weight files (`*.safetensors`, `*.bin`, `*.pt`, etc.).
- `framework_src` — root directory of the inference framework source code (the thing the user actually launched).

Optional (but strongly recommended — without them confidence drops):

- `cli_args` — the exact command line used to launch the framework (string). e.g. `python -m myserver.serve --model ./Llama-3-8B --quant fp8 --kv-cache paged --batch 8`.
- `env_vars` — relevant environment variables. Either a path to a file with `KEY=VALUE` lines, or a single string with `KEY=VALUE;KEY2=VALUE2`. The user does **not** need to paste the full `os.environ`.
- `log_file` — full stdout+stderr of the framework launch. This is gold for resolving code-path branches.

Suggested flow:

1. One `AskUserQuestion` to gather `tracing_file`, `model_dir`, `framework_src` (use multi-field free-form via the "Other" option — actually emit **three separate single-question prompts** since these are file paths the user must type).
2. One `AskUserQuestion` (multi-select) asking whether the user also wants to supply `cli_args`, `env_vars`, `log_file`. For each one the user accepts, follow up with a single-question prompt for the value.

### Step 0c — persist inputs

After collecting (Step 0b) or confirming reuse (Step 0a "Use saved"), persist the resolved inputs to `<CWD>/.metainfer-inputs.json` using `scripts/inputs_io.py save`. When coming from Step 0b, this is a new file. When coming from Step 0a "Start fresh", pass `--overwrite`. The file is intentionally a single per-CWD file (NOT per-run) — that is what makes the next launch's fast-path possible.

```
python3 scripts/inputs_io.py save \
  --tracing-file <abs> --model-dir <abs> --framework-src <abs> \
  [--cli-args STR] [--env-vars STR] [--env-file <abs>] [--log-file <abs>] \
  [--overwrite]
```

Note: `--env-file` accepts a path to a `KEY=VALUE` file the user supplied as their env-vars answer; the orchestrator should set this instead of `--env-vars` when the user gave a file path.

### Step 0d — set up run-scoped working dir + manifest

After Step 0c, create the run-scoped working directory and write a manifest:

```
<CWD>/.metainfer_runs/<run_id>/
├── manifest.json              # every input path + resolved absolute paths
├── memories/                  # phase outputs (see below)
│   ├── phase1_code_map.md
│   ├── phase1_code_map_notes/
│   ├── phase2_tracing_map.md
│   └── phase2_tracing_map_notes/
├── graph/
│   ├── flow_graph.json
│   └── validation_report.md
└── report/
    ├── flow_graph.html
    └── ranking.md
```

`run_id` = ISO timestamp like `20260709-153012`. Use it everywhere so reruns don't clobber each other. Note that `.metainfer-inputs.json` lives at the CWD root (one per directory) while everything else lives under `.metainfer_runs/<run_id>/` (one per run) — this separation is intentional so reusing inputs does not collide with new run outputs.

**Before leaving Phase 0**, validate the inputs one more time (`python3 scripts/inputs_io.py validate`). Echo a one-line summary to the user as a status update (not a question) — for example `"Inputs OK. Starting Phase 1 (3 parallel agents mapping model → framework code)..."`. **Do not** ask the user to confirm or approve: per Hard Rule 7, after Phase 0 collects inputs the skill runs straight through to the end.

## Phase 1 — map the model → framework code

Goal: for the **specific** model the user ran, find:

1. The model class / architecture description in the framework source.
2. Every place where a concrete operator / kernel is invoked while serving this model (attention, MLP, norm, RoPE, quant/dequant, sampling, KV-cache ops, etc.).
3. Every place where quantization parameters are loaded, applied, or skipped.
4. Resolve every code-path branch that is governed by `cli_args`, `env_vars`, or auto-detected hardware/runtime features. **You must end up with the exact code that actually executed**, not a sibling implementation that the runtime silently fell back from or to.

Process:

- Spawn **three independent Agent instances** (use `subagent_type: "Explore"` or `general-purpose`). Give each a **different angle**:
  - Agent A: start from `config.json` (`architectures`, `quantization_config`, `torch_dtype`, hidden/intermediate sizes, num layers, vocab size) and grep the framework for the matching model registration. Then walk forward to all operator calls.
  - Agent B: start from `cli_args` + `env_vars` + `log_file`. Identify which feature flags, dispatchers, autotuners, backend selections are active, then walk backward to the operator calls and quant-loading sites those flags select.
  - Agent C: start from the framework's top-level entry point (found via `cli_args` / `log_file`) and forward-trace the request path end to end, listing every kernel launch site.
- Each agent writes its own raw notes into `memories/phase1_code_map_notes/agent_{a,b,c}.md`.
- A **fourth synthesizer Agent** reads all three notes plus re-reads the framework source as needed, then produces `memories/phase1_code_map.md` containing:
  - Resolved model architecture (with a citation to the framework file:line that registers it).
  - Table of `{step_name, operator_name, framework_file:line, selected_by_which_flag, expected_input_dtypes/shapes, fallbacks_that_did_NOT_run_and_why}`.
  - Table of `{quant_param, load_site_file:line, apply_site_file:line, dispatch_condition}`.
  - A short "confidence" section: anything the synthesizer could not pin down to a single code path, with a reason.

The synthesizer must explicitly answer: *"Given the cli_args + env_vars + log_file, are there any other implementations of these operators that the runtime could have selected but did not?"* — and list them with the reason they were rejected.

## Phase 2 — analyze the tracing file

Goal: turn the Chrome tracing into a kernel-level performance table, **distinguishing same-named kernels that are called from different sites with different data volumes**.

Process:

1. The orchestrator runs `scripts/parse_tracing.py` once. The script:
   - Auto-detects gzip vs plain.
   - Streams the file (do not load all events into memory at once if avoidable).
   - Emits `memories/phase2_tracing_map_notes/parsed_events.jsonl.gz` (per-event compact records: `name, cat, tid, pid, ts, dur, args`).
   - Emits `memories/phase2_tracing_map_notes/kernel_stats.json` with per-`name` aggregate stats (count, mean, p50/p90/p99, std, min, max, sum, histogram buckets).
   - Emits `memories/phase2_tracing_map_notes/cpu_stack_summary.json` if CPU stack frames are present (correlation aid).
   - Emits `memories/phase2_tracing_map_notes/cuda_graph_meta.json` if CUDA Graph capture markers are detected (so downstream knows some kernel names may be collapsed into graph nodes).
2. Spawn **three independent Agent instances**, each with a different angle:
   - Agent X: statistical — for each kernel name, look at the duration distribution and propose whether it is actually **one logical kernel** or a **mixture** (e.g., bimodal histogram, huge variance). Output a partition proposal.
   - Agent Y: source-correlation — using `memories/phase1_code_map.md`, map each kernel name back to its caller site(s) in the framework. Group by `(caller_site, tensor_shape_signature)` instead of by name alone.
   - Agent Z: tracing-internal — for kernels without CPU stacks, use `stream`, `pid`, `tid`, `cat`, and `args` (e.g., shared memory size, grid/block dims, capture tail `node_seq` IDs from CUDA Graph) to disambiguate calls.
3. Each agent writes to `memories/phase2_tracing_map_notes/agent_{x,y,z}.md`.
4. A **synthesizer Agent** merges the three into `memories/phase2_tracing_map.md`:
   - Final table of `{logical_kernel_id, display_name, framework_call_sites (file:line list), call_count, mean_dur_us, p50/p90/p99, std, histogram_summary, shape_signature_if_known, confidence}`.
   - Explicit list of names that were **split** into multiple logical kernels, with the reason.
   - Confidence notes for anything unresolved.

### TP-aware analysis — mandatory when tensor parallelism is enabled

Before doing any per-call-site shape reasoning, **all three agents + the synthesizer must check whether tensor parallelism (TP) was active**. Look at `cli_args` (e.g. `--tensor-parallel-size`, `--tp-size`, `--tp`, `-tp`, `--world-size` combined with `--tp`), `env_vars` (`TP_SIZE`, `WORLD_SIZE`, `RANK`, `LOCAL_RANK`, framework-specific vars), and the launch log (which usually prints `tp_size=N` and `rank=i/N` at startup).

When TP is active, the **per-rank loaded weight shapes differ from the global model config shapes**. The trace records the per-rank shapes, not the global ones. If you reason about shapes as if TP=1 when TP was actually >1, every shape signature will be wrong and same-name/different-site splitting will be garbage.

Concrete rules the agents must apply:

- For each weight, decide whether it is **TP-split** (column-parallel or row-parallel) or **TP-replicated**. This is governed by the framework's TP loader code (cite file:line). Typical defaults for decoder-only LMs (verify, do not assume):
  - **Column-parallel** (output dim is split across ranks): `q_proj`, `k_proj`, `v_proj`, `o_proj`* (*some frameworks column-parallel the input to o_proj instead — read the code), `gate_proj`, `up_proj`, sometimes `lm_head`.
  - **Row-parallel** (input dim is split across ranks): `down_proj`, sometimes `o_proj`.
  - **Replicated**: `input_layernorm`, `post_attention_layernorm`, `final_norm`, embedding (sometimes vocab-parallel instead — read the code).
- Attention heads are partitioned: `Hh_per_rank = Hh / TP` and `Hkv_per_rank` follows the framework's GQA-on-TP policy (often `Hkv / TP` rounded up, or replicated if `Hkv < TP`). This changes the QKV projection's output shape and the attention kernel's grid/block configuration per rank.
- The `shape_signature` in the per-call-site table must be the **per-rank** shape, with its own dim vars (e.g. `Hh_local`, `I_local`, `V_local`) defined in the synthesizer's output.
- All three agents must agree on the TP-derived per-rank shapes before any splitting proposal is trusted. If agents disagree on TP-aware shapes, mark the affected rows `low` confidence.

The synthesizer must include a `## TP summary` section listing: detected TP size, evidence, the per-weight TP policy table (split/replicated, column/row, with file:line), and the resulting per-rank dim values.

### Same-name / different-site splitting under TP

The "one kernel name, multiple call sites with different data volumes" problem is amplified under TP: the *same* call site runs with the *same* per-rank shape on every rank, so the trace will contain TP copies of the same kernel — those should **not** be split into different logical kernels (they are the same logical kernel running in parallel). Conversely, two genuinely different call sites still need splitting. Agent Y's source-correlation must distinguish "TP replicas of one call site" (collapse) from "two different call sites that happen to share a kernel name" (split). Use `pid`/`stream`/`rank` fields in `args` to tell them apart.

## Phase 3 — build and validate the execution-flow graph

Goal: produce `graph/flow_graph.json`, a model-specific execution-flow graph that an AI compiler would recognize, then validate it node-by-node.

### 3.1 Build

Spawn one Agent. Its inputs are `phase1_code_map.md` + `phase2_tracing_map.md`. It produces `graph/flow_graph.json` with this schema:

```json
{
  "run_id": "...",
  "model": { "architectures": [...], "quantization": ..., "dtype": ... },
  "dim_vars": { "B": "batch size", "S": "input seq len", "Sq": "query seq len", "L": "num layers", "H": "hidden", "I": "intermediate", "V": "vocab", "Hh": "num heads", "Hkv": "num kv heads", "Dh": "head dim" },
  "nodes": [
    {
      "id": "n01",
      "step_purpose": "compute Q/K/V projections (quantized GEMM)",
      "operator": "quantized_matmul",
      "framework_file": "framework/ops/qgemm.py:142",
      "framework_refs": [
        { "role": "call_site",     "file": "framework/ops/qgemm.py",     "line": 142, "note": "where the matmul is dispatched" },
        { "role": "kernel_def",    "file": "framework/kernels/qgemm.cu", "line": 88,  "note": "the actual CUDA kernel" },
        { "role": "dispatcher",    "file": "framework/ops/qgemm.py",     "line": 91,  "note": "shape/dtype branch that selected this path" },
        { "role": "weight_loader", "file": "framework/loader.py",        "line": 410, "note": "loads q_proj shard for this TP rank" }
      ],
      "inputs":  [ { "name": "x",  "shape": ["B", "S", "H"],  "dtype": "bf16" } ],
      "outputs": [ { "name": "qkv","shape": ["B", "S", "Hh+2*Hkv", "Dh"], "dtype": "bf16" } ],
      "perf": { "mean_us": 123.4, "p50_us": 120, "p90_us": 135, "std_us": 8.1, "count": 32, "source_kernel_ids": ["k03","k04"] },
      "preds": ["n00"],
      "succs": ["n02","n03"]
    }
  ],
  "edges": [ { "from": "n00", "to": "n01", "tensor": "x" } ]
}
```

Rules:

- Use `dim_vars` for every non-constant dimension. Be consistent across the whole graph.
- Every node must cite `framework_file:line` from Phase 1 and `source_kernel_ids` from Phase 2.
- **Every node must carry a non-empty `framework_refs` list** that pins its full provenance in the framework source. At minimum include the `call_site` (where the operator is invoked) and the `kernel_def` (the kernel/function that actually runs). Add more refs when they exist:
  - `call_site` — Python/C++ line that launches the operator (mandatory).
  - `kernel_def` — the CUDA/Triton/host function body that runs (mandatory when a distinct kernel definition exists; otherwise omit and explain in `note`).
  - `dispatcher` — the file:line of the branch that selected this implementation (the same `selected_by` field from Phase 1, but as a concrete line).
  - `weight_loader` — where the TP shard / quant params for this op are loaded (relevant for matmul / projection / quantized nodes).
  - `fallback_rejected` — a sibling implementation that *could* have run but did not (cite file:line and one-line reason). Optional but strongly recommended.
  Each ref is `{ "role": ..., "file": <path relative to framework_src>, "line": int, "note": <short> }`. Paths must be relative to `framework_src` so they stay portable.
- No orphan nodes. Root nodes (model input) have empty `preds`; sink nodes (final output) have empty `succs`.
- **Shapes are per-rank, not global.** If TP was active (see the Phase 2 TP summary), every node's input/output shapes must reflect what one TP rank actually computes, not the global model shape. Concretely:
  - Define per-rank dim vars (e.g. `Hh_local`, `Hkv_local`, `I_local`, `V_local`) in `dim_vars`, each annotated with the per-rank value or formula (e.g. `"Hh_local": "Hh / TP"`).
  - A QKV projection node on rank `r` outputs `["B", "S", "Hh_local + 2*Hkv_local", "Dh"]`, **not** `["B", "S", "Hh + 2*Hkv", "Dh"]`.
  - A `down_proj` node takes input `["B", "S", "I_local"]`, not `["B", "S", "I"]`.
  - RMS-norm / layernorm weights and (depending on the loader) the embedding take the global `H` / `V` dim because they are replicated.
  - Add a top-level `tensor_parallel` block to `flow_graph.json`: `{ "tp_size": N, "evidence": "...", "per_weight_policy": { "q_proj": "column_parallel", "down_proj": "row_parallel", "input_layernorm": "replicated", ... } }`. The per-node shapes must be consistent with this policy.
  - When the framework uses `all-reduce` / `all-gather` collectives between TP shards, model those as their own nodes (operator `collective` / `all_reduce` / `all_gather`) with their measured perf from Phase 2. Do not silently fold them into the matmul node.

### 3.2 Validate

The orchestrator runs `scripts/validate_graph.py`. The script is **deterministic** and does the following:

a. **Structural checks** (no LLM, pure Python):
   - All node `id`s unique.
   - All `preds`/`succs`/edge endpoints reference existing nodes.
   - No orphan nodes (a node is orphan if both `preds` and `succs` are empty **and** it is neither a declared root nor a declared sink).
   - Every `dim_var` used in any shape is defined in `dim_vars`.
   - Every node has `framework_file` and either `perf` or an explicit `"perf_source": "unmeasured"` reason.
   - **Source-reference existence check.** When `--framework-src` is provided, the script resolves every `framework_file` and every entry of `framework_refs[].file` against that root and verifies the file exists on disk and (when `line` is given) that the file has at least that many lines. Missing files / out-of-range lines fail the structural check with a per-node message. When `--framework-src` is **not** provided, this check is skipped (the "if the source exists" escape hatch) and a warning is printed.
   - Every node has a non-empty `framework_refs` list with at least a `call_site` entry.

b. **Per-node semantic checks** (LLM, but one node at a time):
   - The script iterates nodes. For each node it builds a prompt containing **only that node**'s JSON plus its immediate predecessor/successor nodes' JSON, and asks the LLM:
     - Does this node's stated `operator` and `step_purpose` match the code at `framework_file` and at the `call_site` entry of `framework_refs`?
     - For each entry in `framework_refs`, does the cited `file:line` actually contain what the `role` and `note` claim? (Open each cited file at the cited line ± a small window and verify. If the script's structural check already flagged the file as missing, skip the LLM step for that node and emit a `fix` verdict instead.)
     - Do the input/output shapes match the framework's expectation at that call site?
     - **Are the shapes per-rank-correct under TP?** If the Phase 2 TP summary says TP>1, verify the node uses per-rank dim vars (`Hh_local`, `I_local`, ...) consistent with the per-weight TP policy, and that collectives are modeled as their own nodes — not folded into matmuls.
     - Does the perf data line up with `phase1_code_map.md` and `phase2_tracing_map.md` for the cited `source_kernel_ids`?
   - The LLM must return `{"verdict": "ok" | "fix", "reason": "...", "proposed_patch": {...}}`.
   - The script writes per-node verdicts to `graph/validation_report.md`.
   - To go fast, the script fans out to **N parallel Claude Code Agent invocations** (the script itself just shells out to `claude -p` with a per-node prompt, OR — preferred — the orchestrator spawns parallel `Agent` tool calls, one per node, using a shared prompt template).
   - If **any** node returns `verdict: "fix"`, the script applies the proposed patch to `flow_graph.json` **automatically** (the script accepts `--apply-patches` for this; the orchestrator passes it unconditionally) and triggers **another full round**. Repeat until a round passes with zero fixes, or until 5 rounds elapse. There is **no** user confirmation step here — per Hard Rule 7, validation is fully autonomous. Patches are written to `graph/patch_log.jsonl` so a human can audit later, but the script never blocks waiting for approval. After the loop ends (clean or 5-round cap), results are folded into the final summary at the very end of the run.

The script never lets the LLM see the whole graph at once — that's the explicit anti-divergence rule.

## Phase 4 — visualize

The orchestrator runs `scripts/visualize_graph.py graph/flow_graph.json` which emits `report/flow_graph.html`. The HTML is self-contained (inline CSS+JS, no external CDN) and renders:

- A left-to-right DAG (use SVG or a small embedded `<canvas>`/`<svg>` layout; do not pull in a network library from the web).
- Each node box shows: id, step purpose, operator, framework_file:line, **the list of `framework_refs` (role + file:line)**, mean/p90 time, and a color scaled by mean time (red = hot). When `framework_refs` is empty or files were not verifiable, the box shows a small "unverified source" badge instead.
- A side panel with the full ranking table (`report/ranking.md` is also written by the same script): kernel, total time %, mean, p90, source file:line, and a one-line "why this might be low-hanging fruit" heuristic (e.g., "non-fused epilogue", "Python-side launch overhead", "redundant dequant", "no CUDA Graph capture").

Finally, the orchestrator prints the top-5 ranking to the user with absolute paths to `flow_graph.html`, `flow_graph.json`, and the two phase memory files.

## File map of this skill

```
SKILL.md                              # this file
scripts/inputs_io.py                  # Phase 0 — save/load/validate <CWD>/.metainfer-inputs.json
scripts/parse_tracing.py              # Phase 2 — Chrome tracing parser
scripts/parse_safetensors.py          # Phase 1 helper — weight metadata
scripts/parse_ckpt.py                 # Phase 1 helper — torch checkpoint metadata
scripts/validate_graph.py             # Phase 3.2 — deterministic graph validator
scripts/visualize_graph.py            # Phase 4 — HTML renderer + ranking
templates/prompts/phase1_agent_a.md   # model-config → code
templates/prompts/phase1_agent_b.md   # flags/env/log → code
templates/prompts/phase1_agent_c.md   # entry-point → code
templates/prompts/phase1_synthesizer.md
templates/prompts/phase2_agent_x.md   # statistical split
templates/prompts/phase2_agent_y.md   # source correlation
templates/prompts/phase2_agent_z.md   # tracing-internal disambiguation
templates/prompts/phase2_synthesizer.md
templates/prompts/phase3_build.md     # build flow_graph.json
templates/prompts/phase3_validate_node.md  # per-node LLM check
```

## When to stop / hand control back

- After Phase 4, print the ranking and the paths. Do not auto-modify the framework. The user asked us to *find* the low-hanging kernel, not to fix it — that is a separate skill.
- If any phase ends with unresolved low-confidence items, list them at the top of `report/ranking.md` under "Caveats" and mention them to the user in the final message.
