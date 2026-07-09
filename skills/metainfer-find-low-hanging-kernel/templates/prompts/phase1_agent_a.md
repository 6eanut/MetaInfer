# Phase 1 — Agent A: model config → framework code

You are one of three independent agents whose job is to map a model to the
concrete operator / kernel call sites in the inference framework. You and the
other two agents approach the same question from **different starting points**;
a synthesizer will reconcile your outputs.

## Your starting point: `config.json`

Read `config.json` from the model directory. Extract at minimum:

- `architectures` (e.g. `["LlamaForCausalLM"]`)
- `model_type`
- `torch_dtype`, `quantization_config` (block size, group size, num_bits, scheme)
- `hidden_size`, `intermediate_size`, `num_hidden_layers`, `num_attention_heads`,
  `num_key_value_heads`, `head_dim`, `vocab_size`, `max_position_embeddings`
- `hidden_act`, `normalization_type`, `rope_theta`, `rope_scaling`
- any custom fields that look like they would change kernel selection

## Your task

1. Locate the model registration in the framework source. Search for the
   architecture name, the `model_type`, and the file that registers them in
   the framework's model registry / loader / dispatcher.
2. Walk forward from the model class to every place that invokes a concrete
   operator: linear/matmul/gemm, attention, RoPE, RMS-norm/LayerNorm, activation,
   softmax, sampling/topk, KV-cache append, paged-attention, quant/dequant,
   fused-add-residual, etc. For each call site record:
   - the operator name as it appears in code
   - file:line of the call
   - the input/output tensor dtypes and shapes **at that call site** (use the
     config dims, parameterize batch/seq as `B`, `S`)
   - what conditions (if any) gate this call site — but do **not** try to
     resolve flags yourself here; that's Agent B's job
3. Locate the quantization parameter loading code (where scale/zero-point
   tensors, group metadata, etc. are loaded from the checkpoint) and the code
   that applies them during inference.
4. Explicitly list any **alternative implementations** of the same operator
   that exist in the framework source (e.g. a Triton kernel, a CUDA kernel, a
   PyTorch fallback, a CUTLASS path). For each, note the dispatch condition
   field/flag (do not resolve — Agent B will).

## Output

Write your raw findings to `memories/phase1_code_map_notes/agent_a.md`. Use a
table per category:

```
## Operator call sites

| step_purpose | operator | file:line | input shapes (parametric) | output shapes | dispatch_cond_field |
|---|---|---|---|---|---|

## Quantization load sites

| tensor | load file:line | apply file:line | dispatch_cond_field |
|---|---|---|---|

## Alternative implementations seen

| operator | implementations | dispatch fields |
|---|---|---|
```

Do **not** speculate about which implementation actually ran — you do not have
the flags. Just list what the source contains.

## Important

- Be exhaustive about alternatives. Missing a fallback path is the #1 way this
  whole analysis goes wrong.
- Cite file:line for everything. The synthesizer will re-check.
- Stay within your lane. Don't try to resolve flag values.
