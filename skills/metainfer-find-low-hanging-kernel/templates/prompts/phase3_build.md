# Phase 3 — Build the execution-flow graph

You are the graph builder. You consume the two Phase-1/Phase-2 memory files
and emit `graph/flow_graph.json`.

## Inputs

- `memories/phase1_code_map.md` — operator call sites, dims, dispatches
- `memories/phase2_tracing_map.md` — logical kernel ids + perf stats
- `manifest.json` — to fill the `model` block

## Output schema

```json
{
  "run_id": "<from manifest>",
  "model": {
    "architectures": [...],
    "quantization": "...",
    "dtype": "...",
    "hidden_size": int, "intermediate_size": int, "num_hidden_layers": int,
    "num_attention_heads": int, "num_key_value_heads": int, "head_dim": int,
    "vocab_size": int
  },
  "dim_vars": {
    "B": "batch size",
    "S": "input/prompt seq len",
    "Sq": "generated query seq len so far",
    "L": "num_hidden_layers",
    "H": "hidden_size",
    "I": "intermediate_size",
    "V": "vocab_size",
    "Hh": "num_attention_heads",
    "Hkv": "num_key_value_heads",
    "Dh": "head_dim"
  },
  "declared_roots": ["n00"],
  "declared_sinks": ["n<last>"],
  "nodes": [
    {
      "id": "n00",
      "step_purpose": "embed tokens + add positional embedding",
      "operator": "embedding",
      "framework_file": "framework/models/llama.py:88",
      "framework_refs": [
        { "role": "call_site",     "file": "framework/models/llama.py",  "line": 88,  "note": "QKV proj dispatched" },
        { "role": "kernel_def",    "file": "framework/kernels/qgemm.cu", "line": 88,  "note": "the CUDA kernel that runs" },
        { "role": "dispatcher",    "file": "framework/ops/qgemm.py",     "line": 91,  "note": "shape/dtype branch taken" },
        { "role": "weight_loader", "file": "framework/loader.py",        "line": 410, "note": "loads q_proj TP shard" }
      ],
      "inputs":  [ { "name": "input_ids", "shape": ["B", "S"], "dtype": "int64" } ],
      "outputs": [ { "name": "hidden",    "shape": ["B", "S", "H"], "dtype": "bf16" } ],
      "perf": { "mean_us": 4.2, "p50_us": 4.0, "p90_us": 5.1, "std_us": 0.4, "count": 32, "sum_us": 134.4,
                "source_kernel_ids": ["k01"] },
      "preds": [],
      "succs": ["n01"]
    }
    // ...
  ],
  "edges": [ { "from": "n00", "to": "n01", "tensor": "hidden" } ]
}
```

## Rules

1. **One node per distinct step** in the model's execution flow. A "step" is
   one operator invocation; per-layer ops get one node but with `preds`/`succs`
   showing the layer loop, OR you may unroll a few layers — pick one and be
   consistent. Document your choice at the top of the file as a `_meta` field.
2. **dims are parametric.** Use `dim_vars` names. Don't hardcode `B=1`.
3. **Every node cites a Phase 1 file:line.** If you cannot find one, you don't
   get to add the node.
4. **Every node cites Phase 2 `source_kernel_ids`.** If a step has no measured
   kernel (e.g. it was too short to capture, or it ran on CPU and wasn't
   traced), set `perf_source: "unmeasured"` and skip the `perf` block.
5. **No orphans.** The graph is a single DAG. Roots are model inputs; sinks
   are the final output (logits / sampled token).
6. **Edges** carry the tensor name flowing between two nodes. If a node has
   multiple successors, give each edge a distinct tensor name.
7. **Every node carries a non-empty `framework_refs` list.** This is the
   provenance trail from "operator named X" back to "actual code that ran".
   Fill it from the Phase 1 code map:
   - `call_site` — **mandatory**. The file:line that launches the operator.
   - `kernel_def` — **mandatory when a distinct kernel/function definition
     exists** (CUDA, Triton, CUTLASS wrapper, C++ host function). If the
     operator is purely Python / torch.compile-graph inline with no separate
     kernel, omit `kernel_def` and say so in a `note` on the `call_site` ref.
   - `dispatcher` — the file:line of the branch that selected this
     implementation (from Phase 1's `selected_by` field). Useful when there
     are multiple impls and you need to remember why this one ran.
   - `weight_loader` — where the TP shard / quant params for this op are
     loaded. Add it on every matmul/projection/quantized node.
   - `fallback_rejected` — a sibling implementation that did NOT run, with a
     one-line reason in `note`. Add at least one when Phase 1 listed
     alternatives; this is what makes the graph useful for picking the next
     kernel to optimize.
   All `file` paths must be **relative to `framework_src`** (so the
   validator can resolve them on disk and so the graph stays portable).
   Never invent a path. If you cannot find the file, the node is not ready.

## Anti-patterns to avoid

- Dumping every framework function as a node. Only operators that actually ran.
- Hardcoding shapes when the framework treats them as parameters.
- Inventing perf numbers. If Phase 2 doesn't have them, mark unmeasured.
- One mega-node for "the whole transformer". The point is granularity.
- **Using global model shapes when TP>1.** Per-rank shapes are what actually
  executed; global shapes are wrong. A QKV projection under TP=4 with Hh=32
  has Hh_local=8 per rank — the node must say `Hh_local`, not `Hh`.
- **Folding TP collectives into matmul nodes.** All-reduce / all-gather
  between TP shards are real measured work — model them as separate nodes.

## TP-aware shapes (mandatory when TP>1)

Read the Phase 2 `## TP summary`. Then:

- Add a top-level `tensor_parallel` block:
  ```json
  "tensor_parallel": {
    "tp_size": 4,
    "evidence": "log line 'initialized tp_size=4'",
    "per_weight_policy": {
      "q_proj":    "column_parallel",
      "k_proj":    "column_parallel",
      "v_proj":    "column_parallel",
      "o_proj":    "row_parallel",
      "gate_proj": "column_parallel",
      "up_proj":   "column_parallel",
      "down_proj": "row_parallel",
      "input_layernorm":         "replicated",
      "post_attention_layernorm":"replicated",
      "final_norm":              "replicated",
      "embed_tokens":            "vocab_parallel",
      "lm_head":                 "vocab_parallel"
    }
  }
  ```
  (The above is an example — fill it from the Phase 2 TP table and the
  framework loader code, do not assume.)
- Extend `dim_vars` with the per-rank variables you use, e.g.
  `"Hh_local": "Hh / TP = 8"`, `"I_local": "I / TP"`, `"V_local": "V / TP"`.
- Every node whose operator touches a TP-split weight must use the `_local`
  form in its shapes.
- Add explicit `collective` nodes for inter-rank all-reduce / all-gather. Cite
  the call site in the framework code and the source kernel id(s) from Phase 2.

When done, write the file to `graph/flow_graph.json`. The orchestrator will
then hand it to `scripts/validate_graph.py`.
