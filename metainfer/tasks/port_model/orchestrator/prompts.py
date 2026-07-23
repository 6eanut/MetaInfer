"""Prompt templates for the port-model orchestrator.

Each phase gets its own builder function. Every prompt begins with a
READ-ONLY banner listing paths the agent must never write to, then a
task-specific contract describing what to read, what to produce, and
the exact output filename.

Conventions:

* ``MODEL_PARAMS_PATH`` is always read-only.
* ``TARGET_FRAMEWORK_DIR`` is read-only for every phase *except* P6.
* ``REFERENCE_SOURCES`` is a list of ``{path, notes}`` dicts; each
  member path is read-only.
* Every agent MUST end by writing a per-phase ``summary.md`` to its
  ``workdir`` so the WebUI can render an iteration summary without
  parsing agent chatter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.orchestrator.requirements import req_field


# --------------------------------------------------------------------------- #
# Shared banners
# --------------------------------------------------------------------------- #

def _readonly_banner(
    *,
    model_path: str,
    reference_sources: List[Dict[str, Any]],
    target_fw: str,
    extra_readonly: str = "",
    workdir: Path,
    writable_target: bool = False,
) -> str:
    refs_lines = ""
    for i, src in enumerate(reference_sources or []):
        path = src.get("path") or "(missing path)"
        notes = (src.get("notes") or "").strip()
        refs_lines += f"\n  - REF_{i + 1}_DIR = {path}"
        if notes:
            # Indent continuation lines under the bullet.
            refs_lines += "\n    notes: " + notes.replace("\n", "\n    ")
    target_clause = (
        f"  - TARGET_FRAMEWORK_DIR = {target_fw}  ←  YOU MAY MODIFY (only this dir)\n"
        if writable_target
        else f"  - TARGET_FRAMEWORK_DIR = {target_fw}  ←  READ-ONLY\n"
    )
    return f"""\
# ⛔ READ-ONLY INPUTS

You MUST NOT create, modify, or delete any file under these directories:

  - MODEL_PARAMS_PATH = {model_path}{refs_lines}
{target_clause}{extra_readonly}
All your writes MUST land in your own workdir: {workdir}

"""


# --------------------------------------------------------------------------- #
# Summary writing contract (every agent does this at the end)
# --------------------------------------------------------------------------- #

SUMMARY_CONTRACT = """\
# Required output: summary.md

When you finish (success OR failure), write a markdown file named
``summary.md`` **in your workdir** with the following structure:

```
# <phase name> — <one-line outcome>

## Outcome
<ok | logic_fail | infra_fail | test_fail | bounce_back | needs_repair>

## What I did
- bullet list of concrete actions

## Key findings / artifacts
- path/to/artifact.md
- path/to/other.json

## Next-step notes for the next agent
- short bullet list — keep it tight, this is read by the next phase
```

If you cannot complete the task, set ``Outcome`` to ``logic_fail`` and
explain in ``What I did`` what blocked you. Do NOT silently produce
partial output without a summary.md.
"""


# --------------------------------------------------------------------------- #
# Execution discipline — anti-patterns observed in the wild that waste
# tokens and stall the pipeline. Two layers:
#
#   * CORE_DISCIPLINE — applies to every phase. Anti-polling, anti-reread,
#     sample-don't-enumerate, stop-when-done.
#   * Phase-specific discipline block — passed per phase. Sub-agent /
#     Task tool permission varies: P1/P3/P5 ban it (local-only work),
#     P2/P4/P6 allow it (legitimate code exploration / synthesis).
# --------------------------------------------------------------------------- #

CORE_DISCIPLINE = """\
# ⚠️ Execution discipline (applies to every phase)

You are one phase in a larger pipeline. The orchestrator calls the next
phase when you finish. To keep the pipeline moving:

1. **Do NOT use the ``Sleep`` tool.** If a Bash command is slow, just
   wait for it inline — Bash blocks until it returns. If you have
   nothing useful to do, write your output and finish.

2. **Do NOT poll background tasks.** Do NOT launch ``run_in_background``
   Bash commands and then ``wc -l`` / ``tail`` their output in a loop.
   Run commands synchronously; their result is returned directly.

3. **Do NOT re-read files you have already read.** Remember the result
   and refer back to it.

4. **Sample, don't enumerate.** When inspecting shards / layers /
   tensors / source files, pick a few representative ones. Do NOT
   iterate over every item in a 100GB model or a 10k-file codebase.

5. **Stop as soon as ``summary.md`` is written.** Once you have produced
   the required artifact and the summary, exit. Do NOT run extra "sanity
   checks" or re-verify — the orchestrator already has dedicated
   verifier phases (P5, P6) for that.

6. **No self-verification.** Do NOT spawn a sub-agent whose only job is
   to "check" or "verify" what you just wrote. The orchestrator's later
   phases do this. Self-verification is wasted effort.
"""

# Per-phase delegation policy. ``allow_subagents=True`` means the agent
# may use the Task/Agent tool for *productive* exploration (e.g.
# parallel reads of a large reference codebase, or a focused
# implementation sub-task). It does NOT license the self-verification
# anti-pattern (banned globally in CORE_DISCIPLINE rule 6).


def _delegation_discipline(*, allow_subagents: bool, rationale: str) -> str:
    if allow_subagents:
        return f"""\
# 🔧 Sub-agent policy for this phase

You MAY use the ``Task`` / ``Agent`` tool for **productive** sub-tasks
such as: {rationale}.

The following are NOT productive uses and remain banned:
  - Spawning a "verifier" sub-agent to re-check your own output.
  - Spawning a sub-agent to do something you could do in one Bash call.
  - Polling a sub-agent's output with Sleep + wc -l (see core rule 1-2).

When in doubt: do it yourself.
"""
    return f"""\
# 🔧 Sub-agent policy for this phase

**Sub-agents are NOT allowed in this phase.** Do NOT use the ``Task`` /
``Agent`` tool at all. {rationale}

All work must be done inline with Read / Bash / Write / Edit.
"""


# Convenience aliases for each phase's policy.
P1_DISCIPLINE = _delegation_discipline(
    allow_subagents=False,
    rationale=(
        "P1 only reads local weight files and writes one analysis doc — "
        "there is nothing to parallelise or delegate."
    ),
)
P2_DISCIPLINE = _delegation_discipline(
    allow_subagents=True,
    rationale=(
        "exploring a large reference framework codebase in parallel "
        "(e.g. one sub-agent per directory while mapping out the "
        "attention / MLP / quantization implementations)"
    ),
)
P3_DISCIPLINE = _delegation_discipline(
    allow_subagents=False,
    rationale=(
        "P3 consolidates P1/P2 outputs into one spec — pure synthesis, "
        "no exploration needed."
    ),
)
P4_DISCIPLINE = _delegation_discipline(
    allow_subagents=True,
    rationale=(
        "writing focused sub-components of the minimal framework "
        "(e.g. one sub-agent per operator implementation while you "
        "wire the top-level forward pass)"
    ),
)
P5_DISCIPLINE = _delegation_discipline(
    allow_subagents=False,
    rationale=(
        "P5 runs the framework and judges the output — sequential by "
        "nature, nothing to delegate."
    ),
)
P6_DISCIPLINE = _delegation_discipline(
    allow_subagents=True,
    rationale=(
        "porting into a large target framework codebase where parallel "
        "exploration of the existing kernel / op / layer implementations "
        "is genuinely useful"
    ),
)


# Backwards-compat alias — old code referenced EXECUTION_DISCIPLINE.
EXECUTION_DISCIPLINE = CORE_DISCIPLINE


def _user_notes_block(req: Dict[str, Any]) -> str:
    notes = (req_field(req, "user_notes") or "").strip()
    if not notes:
        return ""
    return f"""\
# 👤 User-provided notes (apply this context throughout)

```
{notes}
```

"""


def _distributed_block(worker_nodes: Optional[List[str]]) -> str:
    """Inject distributed-testing guidance when worker_nodes is configured.

    Returns "" when worker_nodes is empty (orchestrator local mode — no change
    to existing prompts). With ≥2 workers the block teaches the agent how to
    launch PP2 via the cluster SDK; the agent still owns the decision of
    whether to actually do so (single-node frameworks can ignore).
    """
    if not worker_nodes:
        return ""
    nodes_str = ", ".join(worker_nodes)
    if len(worker_nodes) >= 2:
        return f"""\
# 🌐 Distributed workers available (PP2-capable)

This task has {len(worker_nodes)} worker nodes available for cross-node
end-to-end testing: ``{nodes_str}``.

If the framework you're verifying supports tensor parallelism (TP) or
pipeline parallelism (PP), you may use the cluster SDK to launch it
across two workers simultaneously. The orchestrator pre-allocates one
GPU per worker and injects ``RANK`` / ``WORLD_SIZE`` / ``MASTER_ADDR``
/ ``MASTER_PORT`` for you; your framework's launch script only needs
to honor the standard torch.distributed env.

Minimal invocation (rank0 on first worker, rank1 on second):

```python
from metainfer.cluster.sdk import submit_pp2_ranks, PP2RankSpec
results = submit_pp2_ranks(
    rank_a=PP2RankSpec(
        worker_node_id={worker_nodes[0]!r},
        gpu_index=0,
        script_body="cd {{target_fw}} && python -m {{launcher}} --rank 0\\n",
    ),
    rank_b=PP2RankSpec(
        worker_node_id={worker_nodes[1]!r},
        gpu_index=0,
        script_body="cd {{target_fw}} && python -m {{launcher}} --rank 1\\n",
    ),
    timeout_s=1800,
)
```

See ``docs/agent-sdk-guide.md`` for the full SDK cookbook (log tailing,
error handling, status codes).

"""
    # Only one worker — still useful for GPU isolation but no PP2.
    return f"""\
# 🌐 Remote worker available

This task has one worker node configured: ``{nodes_str}``. If you want
a clean isolated GPU for the end-to-end run, you may submit the launch
command via the cluster SDK instead of running locally:

```python
from metainfer.cluster.sdk import submit_script
result = submit_script(
    worker_node_id={worker_nodes[0]!r}, gpu_slots=[({worker_nodes[0]!r}, 0)],
    script_body="python {{launcher}}\\n", timeout_s=1800,
)
```

See ``docs/agent-sdk-guide.md`` for details.

"""


# --------------------------------------------------------------------------- #
# Phase 1: Weight analysis
# --------------------------------------------------------------------------- #

def p1_weight_analysis_prompt(
    *, req: Dict[str, Any], workdir: Path,
    prev_failure: str = "",
) -> str:
    model_path = req_field(req, "model_params_path") or ""
    target_fw = req_field(req, "target_framework_dir") or ""
    refs = req_field(req, "reference_sources") or []

    banner = _readonly_banner(
        model_path=model_path, reference_sources=refs,
        target_fw=target_fw, workdir=workdir,
    )
    notes = _user_notes_block(req)
    prev_block = ""
    if prev_failure:
        prev_block = (
            "\n# ⚠️ ARCHITECT FEEDBACK — previous P1 output was bounced\n\n"
            "The senior architect rejected the previous weight analysis. "
            "Address the feedback below and rewrite:\n\n```\n"
            f"{prev_failure[:4000]}\n```\n"
        )

    return banner + CORE_DISCIPLINE + P1_DISCIPLINE + notes + prev_block + f"""\
# Task: 权重参数分析 (Weight parameter analysis)

Your job is to read the model weight files under ``MODEL_PARAMS_PATH``
and produce a structured analysis document for downstream agents.

## Steps

1. **Read config.json** under ``MODEL_PARAMS_PATH`` (or, if the path
   is a single weight file, infer config from neighbouring files in
   the same directory). Extract:
   - architecture / model_type
   - hidden_size, num_layers, num_attention_heads, num_kv_heads
   - vocab_size, max_position_embeddings
   - quantization config (group_size, bits, sym, ignored layers, …)
   - any MoE / VLM / MTP / hybrid structure indicators
   - dtype / tie_word_embeddings / etc.

2. **Inspect weight files**:
   - Identify the actual storage format (.safetensors / .bin / .npz …)
   - Read the weight-index file (e.g. ``model.safetensors.index.json``)
     to get the full tensor name list.
   - For ONE shard, sample 5-10 representative tensors (embedding,
     attention Q/K/V/O, MLP up/down, layer norm, final norm, lm_head)
     and record their actual dtype + shape.
   - If quantized, inspect the format (e.g. weight_int4 / weight_scale
     pairs, group_size alignment, double-quant tensors).

3. **Write your analysis** to ``{workdir}/p1_weight_analysis.md`` with
   the following sections:
   - Architecture overview (model_type, layer count, head count, …)
   - Quantization scheme (or "none")
   - Tensor naming convention (e.g. ``model.layers.<N>.self_attn.q_proj.weight``)
   - Per-layer weight inventory: each operator's required tensors with
     exact dtype + shape, and the dequantization rule if applicable.
   - Anything unusual that the next agent needs to know.

{SUMMARY_CONTRACT}
"""


# --------------------------------------------------------------------------- #
# Phase 2: Reference framework analysis (one agent per reference source)
# --------------------------------------------------------------------------- #

def p2_framework_analysis_prompt(
    *, req: Dict[str, Any], workdir: Path,
    ref_index: int, ref_path: str, ref_notes: str,
    p1_path: Path,
) -> str:
    model_path = req_field(req, "model_params_path") or ""
    target_fw = req_field(req, "target_framework_dir") or ""
    refs = req_field(req, "reference_sources") or []
    banner = _readonly_banner(
        model_path=model_path, reference_sources=refs,
        target_fw=target_fw, workdir=workdir,
    )
    notes = _user_notes_block(req)
    ref_note_block = ""
    if ref_notes and ref_notes.strip():
        ref_note_block = (
            f"\n# 👤 Reference-specific hint from the user (REF_{ref_index})\n\n"
            f"```\n{ref_notes.strip()}\n```\n"
        )

    return banner + CORE_DISCIPLINE + P2_DISCIPLINE + notes + ref_note_block + f"""\
# Task: 推理框架分析师 — analyse reference implementation #{ref_index}

Reference source root (READ-ONLY):
  REF_{ref_index}_DIR = {ref_path}

P1 weight analysis (READ, it's authoritative for tensor names/shapes):
  {p1_path}

## Steps

1. **Locate the model implementation** inside ``REF_{ref_index}_DIR``.
   Use the architecture name from P1 to grep for the right files
   (e.g. ``modeling_<arch>.py``). Identify:
   - File path(s) that implement the model
   - File path(s) that implement the weight loader
   - File path(s) that register the model in the framework (entry point)

2. **Describe the model structure** abstractly AND with code refs:
   - Layer list (input → embedding → transformer blocks → norm → lm_head)
   - For each operator: input dtype + shape, output dtype + shape
   - Where attention / MLP / normalization live
   - For MoE: router + per-expert MLP layout

3. **Describe the weight loading + dequantization logic**:
   - How tensor names from P1 map onto operators here
   - How quantized weights are converted to fp16/bf16/fp32 at runtime
   - Any per-layer special-casing (e.g. ignored layers in quant config)

4. **Write your analysis** to ``{workdir}/p2_ref{ref_index}_analysis.md``
   with two clearly separated sections per topic:
   - **Abstract description** (no code — usable as a spec)
   - **Implementation pointers** (file:line, function name, key snippets
     under 10 lines each)

{SUMMARY_CONTRACT}
"""


# --------------------------------------------------------------------------- #
# Phase 3: Architect review
# --------------------------------------------------------------------------- #

def p3_architect_review_prompt(
    *, req: Dict[str, Any], workdir: Path,
    p1_path: Path, p2_paths: List[Path],
    bounce_count: int,
) -> str:
    model_path = req_field(req, "model_params_path") or ""
    target_fw = req_field(req, "target_framework_dir") or ""
    refs = req_field(req, "reference_sources") or []
    banner = _readonly_banner(
        model_path=model_path, reference_sources=refs,
        target_fw=target_fw, workdir=workdir,
        extra_readonly=(
            "  - You may re-read the source dirs of the references\n"
            "    (the same REF_*_DIR paths) for spot-check verification.\n"
        ),
    )
    notes = _user_notes_block(req)
    p2_listing = "\n".join(f"  - {p}" for p in p2_paths) or "  (none)"
    bounce_clause = ""
    if bounce_count > 0:
        bounce_clause = (
            f"\nNote: you have already bounced {bounce_count} time(s). "
            f"Caps at 2; if you still see fundamental issues you may bounce "
            f"ONE more time, otherwise pick the most credible P2 and proceed.\n"
        )

    return banner + CORE_DISCIPLINE + P3_DISCIPLINE + notes + f"""\
# Task: 推理框架资深架构师 — cross-review and consolidate

P1 weight analysis (authoritative for weights):
  {p1_path}

P2 reference-framework analyses (one per reference source):
{p2_listing}
{bounce_clause}
## Steps

1. **Cross-compare** every P2 against P1 and against each other.
   Build a single consolidated picture of the model.

2. **Spot-check** at least ONE critical claim per P2 by re-reading
   the reference source. Confirm operator input/output shapes and
   dequantization rules are correct.

3. **Decide**:
   - If the P2 analyses agree (or their disagreements are minor and
     reconcilable), set Outcome=ok in your summary and write the
     consolidated spec to ``{workdir}/p3_consolidated_spec.md``.
   - If you see **fundamental divergence** on the model's basic
     structure (layer count, attention type, quantization scheme) that
     you cannot resolve by re-reading, set Outcome=bounce_back in your
     summary and write a concrete list of issues the P1/P2 agents must
     fix. The orchestrator will restart P1 and re-run P2.

4. **The consolidated spec** at ``{workdir}/p3_consolidated_spec.md``
   must contain:
   - Architecture overview (from P1, confirmed/corrected)
   - Per-operator abstract description (dtype, in-shape, out-shape)
   - For each operator: list of source-file pointers across ALL P2s
     that implement it (so the next agent can pick the cleanest one).
   - Weight loading + dequantization rules (canonical).

{SUMMARY_CONTRACT}
"""


# --------------------------------------------------------------------------- #
# Phase 4: Minimal framework builder
# --------------------------------------------------------------------------- #

def p4_minimal_framework_prompt(
    *, req: Dict[str, Any], workdir: Path,
    p3_path: Path, prev_failure: str = "",
) -> str:
    model_path = req_field(req, "model_params_path") or ""
    target_fw = req_field(req, "target_framework_dir") or ""
    refs = req_field(req, "reference_sources") or []
    banner = _readonly_banner(
        model_path=model_path, reference_sources=refs,
        target_fw=target_fw, workdir=workdir,
    )
    notes = _user_notes_block(req)
    prev_block = ""
    if prev_failure:
        prev_block = (
            "\n# ⚠️ P5 VERIFIER FEEDBACK — previous minimal framework failed\n\n"
            "Fix the issues below and rewrite:\n\n```\n"
            f"{prev_failure[:6000]}\n```\n"
        )

    return banner + CORE_DISCIPLINE + P4_DISCIPLINE + notes + prev_block + f"""\
# Task: 精简推理框架编写工程师 — minimal PyTorch forward framework

Authoritative spec (READ):
  {p3_path}

## Goal

Write a minimal PyTorch framework that can do ONE forward pass of the
model and dump intermediate hidden_states. This is a **correctness
oracle** — speed does not matter, but precision does (use fp32 or the
model's native dtype; do NOT cut corners on numerics).

## Hard requirements

1. **Per-layer lazy weight loading**. You are running on a constrained
   GPU; do NOT load the whole model into VRAM at once. The pattern is:
   ```
   for layer_idx in range(num_layers):
       weights = load_layer_weights(layer_idx)   # CPU → GPU for THIS layer
       hidden = run_layer(layer_idx, hidden, weights)
       dump_hidden(layer_idx, hidden)
       free(weights)
   ```
   For MoE, load only the experts the current token routes to.

2. **Hidden-state dump points**. At each layer, dump hidden_state at
   at least these checkpoints:
   - layer entry (input)
   - after input_layernorm
   - after attention (before residual add)
   - after attention residual add
   - after post_attention_layernorm
   - after MLP (before residual add)
   - layer exit (after second residual add)

   Dump format: numpy ``.npy`` files under ``{workdir}/dumps/``,
   one file per (layer_idx, checkpoint). Use the naming convention
   ``layer_<NNN>_<checkpoint>.npy``.

3. **Entry point**. Provide ``{workdir}/run.py`` that:
   - Takes a **list of prompt strings** (one per line on stdin, or via
     ``--prompts p1::p2::p3`` argv). Multiple prompts are run as a
     single batched forward pass.
   - Tokenises all prompts with the model's tokeniser (look under
     ``MODEL_PARAMS_PATH`` for tokeniser files), left-pads to equal
     length (left-padding so the final token of every prompt aligns
     at the same position, which is what the next-token prediction
     expects).
   - Runs **one** forward pass over the whole batch, dumps every
     checkpoint **per row** (see dump layout below), and prints the
     top-k of the final logits for each row.
   - Single-prompt invocation must still work (batch size 1).

4. **Per-row dump layout**. Hidden states are shape
   ``[batch, seq, hidden]``. Split them by batch index so downstream
   agents can read each row independently:
   ```
   {workdir}/dumps/
     row0/layer_000_input.npy
     row0/layer_000_attn.npy
     ...
     row1/layer_000_input.npy
     ...
     row2/...
   ```
   Naming: ``row<batch_idx>/layer_<NNN>_<checkpoint>.npy``. Each file
   holds the per-row slice ``[seq, hidden]`` (the batch dim is gone).

5. **No edits to TARGET_FRAMEWORK_DIR** — that's the next agent's job.

## Output files (write all of them)

- ``{workdir}/run.py``           — runnable script
- ``{workdir}/modeling_min.py``  — the minimal PyTorch model code
- ``{workdir}/loader.py``        — lazy per-layer weight loader
- ``{workdir}/README.md``        — how to run, what dumps to expect

{SUMMARY_CONTRACT}
"""


# --------------------------------------------------------------------------- #
# Phase 5: Minimal framework verifier
# --------------------------------------------------------------------------- #

def p5_verify_minimal_prompt(
    *, req: Dict[str, Any], workdir: Path,
    p4_dir: Path,
    worker_nodes: Optional[List[str]] = None,
) -> str:
    model_path = req_field(req, "model_params_path") or ""
    target_fw = req_field(req, "target_framework_dir") or ""
    refs = req_field(req, "reference_sources") or []
    banner = _readonly_banner(
        model_path=model_path, reference_sources=refs,
        target_fw=target_fw, workdir=workdir,
    )
    notes = _user_notes_block(req)

    return banner + CORE_DISCIPLINE + P5_DISCIPLINE + notes + _distributed_block(worker_nodes) + f"""\
# Task: 精简推理框架验证工程师 — verify the minimal framework

The minimal framework from P4 lives in (READ + EXECUTE):
  {p4_dir}

You will run it as a **3-prompt batch**, capture logs, and judge each
row's output for **semantic correctness** (NOT exact string match).

## Test batch (run all three in one forward pass)

```
世界上最高的山峰是
中国的国旗是
人体正常体温约为
```

These three prompts cover different knowledge domains (geography,
culture, science) so a single bug can't slip through all of them. Each
expected completion is ~3 tokens — long enough to be a meaningful
correctness signal, short enough to judge from top-k alone.

You may override the batch only if the user's notes specify different
prompts.

## Steps

1. **Feed the batch to P4**. Pipe the three prompts into the framework:
   ```
   printf '世界上最高的山峰是\\n中国的国旗是\\n人体正常体温约为\\n' \
     | python {p4_dir}/run.py
   ```
   Capture:
   - Full stdout + stderr → ``{workdir}/run.log``
   - Exit code → record in ``{workdir}/verdict.json``
   - Per-row top-k (the framework prints them).

2. **If it crashed** (nonzero exit, traceback):
   - Read the log carefully.
   - Set ``Outcome=test_fail`` in your summary.
   - In ``{workdir}/verdict.json``, write:
     ```
     {{
       "passed": false,
       "reason": "crash",
       "error_class": "<exception class>",
       "error_message": "<first 3 lines of traceback>",
       "log_file": "{workdir}/run.log"
     }}
     ```
   - The orchestrator will hand this back to P4 for repair.

3. **If it produced tokens** for every row, judge **semantic
   correctness per row**. For each prompt:
   - Does the top-k contain a token sequence that semantically
     completes the prompt? Examples of acceptable completions:
     - ``世界上最高的山峰是`` → 珠穆朗玛峰 / 珠峰 / Everest
     - ``中国的国旗是`` → 五星红旗 / 中华人民共和国国旗
     - ``人体正常体温约为`` → 三十七 / 37 / 三十七摄氏度 / 三十七度
   - Tolerate token-boundary variation (``三十七`` vs ``三十七度``
     vs ``三十七摄氏度`` all count as correct).
   - Flag failure modes: top-k full of unrelated glyphs, Chinese
     prompt producing Latin-script tokens, top-1 = ``<unk>`` or
     punctuation.
   - Verify dumped hidden_states are finite (no NaN/Inf) — sample
     one file per row.

4. **Write the batch verdict**. ``{workdir}/verdict.json``:
   ```
   {{
     "passed": <true only if every row passed>,
     "batch": [
       {{
         "prompt": "世界上最高的山峰是",
         "topk_text": ["珠穆朗玛峰", "珠穆朗", ...],
         "verifier_judgment": "passed",
         "verifier_reason": "<one sentence why this is acceptable>"
       }},
       {{
         "prompt": "中国的国旗是",
         "topk_text": [...],
         "verifier_judgment": "passed",
         "verifier_reason": "..."
       }},
       {{
         "prompt": "人体正常体温约为",
         "topk_text": [...],
         "verifier_judgment": "passed",
         "verifier_reason": "..."
       }}
     ],
     "dump_dir": "{workdir}/dumps",
     "log_file": "{workdir}/run.log"
   }}
   ```

   Overall ``passed = true`` requires ``verifier_judgment == "passed"``
   for every row. Any row failing → ``passed: false`` +
   ``Outcome=test_fail`` → P4 will be asked to repair.

5. **Make sure dumps are at** ``{workdir}/dumps/`` (symlinked or copied
   if P4 wrote them elsewhere). They MUST follow the per-row layout:
   ``dumps/row0/``, ``dumps/row1/``, ``dumps/row2/``. The next agent
   (P6) reads these as its similarity oracle.

6. **Write** ``{workdir}/summary.md`` describing what you observed
   per row, and your overall verdict.

{SUMMARY_CONTRACT}
"""


# --------------------------------------------------------------------------- #
# Phase 6: Port to target framework
# --------------------------------------------------------------------------- #

def p6_port_engine_prompt(
    *, req: Dict[str, Any], workdir: Path,
    p3_path: Path, p5_dumps_dir: Path,
    iteration: int, prev_failure: str = "",
    worker_nodes: Optional[List[str]] = None,
) -> str:
    model_path = req_field(req, "model_params_path") or ""
    target_fw = req_field(req, "target_framework_dir") or ""
    refs = req_field(req, "reference_sources") or []
    # P6 is the ONLY phase allowed to modify target_fw.
    banner = _readonly_banner(
        model_path=model_path, reference_sources=refs,
        target_fw=target_fw, workdir=workdir,
        writable_target=True,
    )
    notes = _user_notes_block(req)
    prev_block = ""
    if prev_failure:
        prev_block = (
            "\n# ⚠️ Previous P6 iteration failed — diagnose and fix\n\n"
            "```\n" + prev_failure[:6000] + "\n```\n"
        )

    return banner + CORE_DISCIPLINE + P6_DISCIPLINE + notes + prev_block + _distributed_block(worker_nodes) + f"""\
# Task: 推理引擎移植工程师 — port the model into TARGET_FRAMEWORK_DIR

P3 consolidated spec (READ):
  {p3_path}

Golden dumps from the minimal framework (READ, your similarity oracle):
  {p5_dumps_dir}

This is iteration #{iteration} of P6. Each iteration that produces a
non-empty change set MUST end with a git commit inside
``TARGET_FRAMEWORK_DIR`` (see below).

## Workflow per iteration

1. **Try to boot the model** in the target framework. Derive a sane
   launch command from the framework's conventions (vLLM/SGLang/
   TensorRT-LLM/llama.cpp/…). Capture full stdout+stderr to
   ``{workdir}/launch_attempt_{iteration}.log``.

2. **If launch fails because of an unsupported operator**:
   a. First, look for a flag / env var that falls back to a simpler
      implementation. If found, use it.
   b. Otherwise, **add a new code path** to the target framework that
      routes the failing op to your own implementation. DO NOT DELETE
      existing code — only ADD: gate the new path behind a new env var
      or a new branch in the existing function. (E.g.
      ``if os.environ.get("METAINFER_CUSTOM_OP") == "1": use_my_op()``
      inserted at the top of the function.)

3. **If launch succeeds**, send the **3-prompt batch** that P5 used:
   ```
   世界上最高的山峰是
   中国的国旗是
   人体正常体温约为
   ```
   Run them as a batched forward pass (same as P4/P5 — left-pad, one
   pass, three rows of output). Capture the output AND dump the same
   hidden_state checkpoints the minimal framework produced, using the
   **same per-row layout** (``dumps/row0/``, ``dumps/row1/``,
   ``dumps/row2/``) and the same ``layer_<NNN>_<checkpoint>.npy``
   naming inside each row subdir.

4. **Compare hidden_states** between target framework and the golden
   dumps **per row**. For each row<batch_idx>:
   - Load every ``row<batch_idx>/layer_*.npy`` from both sides.
   - Compute cosine similarity per checkpoint.
   - Require ≥ 0.99 per checkpoint.
   Do NOT use exact equality (different backends will diverge at
   1e-3 scale). Report the **minimum similarity across all rows** as
   ``similarity_min`` in the verdict.

5. **Verdict**:
   - If output is semantically reasonable AND every checkpoint has
     cosine ≥ 0.99: Outcome=ok.
   - If checkpoints diverge: identify the first divergent operator,
     fix it, Outcome=needs_repair (orchestrator will start another
     P6 iteration).
   - If you cannot even launch: Outcome=test_fail.

6. **Git commit** (only if you modified target_fw this iteration):
   ```
   cd {target_fw}
   git add -A
   git commit -m "port_model(P6 iter {iteration}): <one-line summary>"
   ```
   If ``{target_fw}`` is not yet a git repo, run ``git init`` first.
   Record the commit SHA to ``{workdir}/commit_{iteration}.txt``.

7. **Write ``{workdir}/verdict_{iteration}.json``** with:
   ```
   {{
     "iteration": {iteration},
     "launched": true|false,
     "batch": [
       {{"prompt": "世界上最高的山峰是", "topk_text": [...],
         "verifier_judgment": "passed|failed", "verifier_reason": "..."}},
       {{"prompt": "中国的国旗是", ...}},
       {{"prompt": "人体正常体温约为", ...}}
     ],
     "similarity_min": <float or null — minimum across all rows>,
     "similarity_first_bad_layer": <int or null>,
     "similarity_first_bad_row": <int or null>,
     "commit_sha": "<sha or null>",
     "outcome": "ok|needs_repair|test_fail",
     "reason": "..."
   }}
   ```
   ``output_text`` is no longer a single string — use the per-row
   ``batch[].topk_text`` instead. P5 had the same shape; keep them
   aligned so the orchestrator's parsing stays uniform.

{SUMMARY_CONTRACT}
"""
