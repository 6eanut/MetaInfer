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


def _launch_constraints_block(req: Dict[str, Any]) -> str:
    """Inject user-supplied launch constraints into P5/P6 prompts.

    The field is a free-form textarea where the user describes model-specific
    launch requirements: required CLI flags, env vars, memory budget facts,
    known OOM pitfalls, recommended strategies (e.g. "must combine PP2 with
    lazy loading"), timeout hints. The agent uses this as authoritative
    constraints when synthesising the actual launch command — port_model
    itself stays generic (no hardcoded framework names).

    Returns "" when the field is absent or empty (no change to the prompt).
    """
    constraints = (req_field(req, "launch_constraints") or "").strip()
    if not constraints:
        return ""
    return f"""\
# 📋 Launch constraints (user-supplied, AUTHORITATIVE for this model)

The user has provided the following model-specific launch constraints.
Treat these as hard requirements when synthesising the launch command /
script — they capture facts about this specific model+framework combo
that the generic port_model flow does not know:

```
{constraints}
```

"""


def _distributed_block(worker_nodes: Optional[List[str]]) -> str:
    """Inject distributed-testing guidance when worker_nodes is configured.

    Returns "" when worker_nodes is empty (orchestrator local mode — no change
    to existing prompts). With ≥2 workers the block **mandates** that the
    agent use the cluster SDK to spread the framework launch across all
    listed workers — single-node smoke tests are NOT an acceptable final
    state because the whole point of having multiple workers is end-to-end
    cross-node validation. (Single-node probes may still be used as a
    transient diagnostic step inside the inner port-test loop, but the
    iteration's final verdict must come from a multi-worker run.)

    The number of GPUs per rank is intentionally NOT hardcoded here —
    the agent reads it from the task's ``launch_constraints`` (e.g. a
    "TP=4" hint) and passes ``gpu_indices=[...]`` accordingly.
    """
    if not worker_nodes:
        return ""
    nodes_str = ", ".join(worker_nodes)
    if len(worker_nodes) >= 2:
        return f"""\
# 🌐 Distributed workers configured — multi-node launch is REQUIRED

This task lists **{len(worker_nodes)} worker nodes**: ``{nodes_str}``.
**Every end-to-end framework launch in this iteration MUST span all of
them.** Single-node / single-GPU launches are forbidden as a final
verdict — they defeat the purpose of having multi-node workers, and
the iteration's port-test verdict will be rejected if it doesn't show
evidence of a real cross-node run (scoreboard claims on every worker,
torch.distributed rendezvous env in the launch logs, results collected
from every rank).

Transient single-node probes inside the inner port-test loop are OK as
a diagnostic (e.g. "does the framework even import on this node"), but
the iteration's final launch attempt that produces the verdict MUST be
the full multi-worker launch.

## Use the cluster SDK — never invoke the framework locally

Use ``metainfer.cluster.sdk.submit_pp2_ranks`` to launch two ranks
simultaneously, one on each worker. The SDK:

- acquires GPU slots atomically across nodes,
- injects ``RANK`` / ``NODE_RANK`` / ``WORLD_SIZE`` / ``NNODES`` /
  ``NPROC_PER_NODE`` / ``TP_SIZE_PER_NODE`` / ``LOCAL_RANK`` /
  ``MASTER_ADDR`` / ``MASTER_PORT`` env vars into both ranks,
- returns each rank's ``JobResult`` (exit_code, stdout/stderr tail,
  duration) when both finish.

The number of GPUs per rank is whatever the task needs — read the
``launch_constraints`` block for the model's TP/PP requirements and
pass ``gpu_indices=[...]`` accordingly. Do NOT assume 1 GPU per worker.

## Sketch

```python
from metainfer.cluster.sdk import submit_pp2_ranks, PP2RankSpec

# Read TP-per-rank from launch_constraints; here we use 4 as an example.
# The SDK does not care about the exact number — it just acquires that
# many slots and sets NPROC_PER_NODE / TP_SIZE_PER_NODE to match.
tp_per_rank = 4  # parsed from launch_constraints, NOT hardcoded by port_model

job_id_0, job_id_1, res0, res1 = submit_pp2_ranks(
    rank0=PP2RankSpec(
        worker_node_id={worker_nodes[0]!r},
        gpu_indices=list(range(tp_per_rank)),  # ranks 0..tp_per_rank-1 on this node
        command="export PYTHONPATH=... && python -m <framework.launcher> "
                "--rank $RANK --world-size $WORLD_SIZE ...",
    ),
    rank1=PP2RankSpec(
        worker_node_id={worker_nodes[1]!r},
        gpu_indices=list(range(tp_per_rank)),
        command="<same launcher with --rank 1>",
    ),
    timeout_s=1800,  # honor launch_constraints timeout hints
)
```

Inside ``command``, the framework's launcher is responsible for:
- reading ``$RANK`` / ``$WORLD_SIZE`` / ``$MASTER_ADDR`` / ``$MASTER_PORT``
  to call ``torch.distributed.init_process_group`` (with retry — the two
  ranks start ~simultaneously and may race the rendezvous),
- spawning one local subprocess per ``$NPROC_PER_NODE`` if it needs
  intra-node tensor parallel (or letting its own internal launcher do it).

## Verification before declaring the iteration's verdict

Before writing ``verdict_*.json`` with ``outcome: ok`` or ``logic_fail``,
confirm ALL of:

1. **Scoreboard shows claims on every worker**: at some point during the
   run, ``cluster/scoreboard/<worker>/*.claim`` existed for every worker
   in the ``worker_nodes`` list. (Use ``metainfer.cluster.scoreboard.list_claims``
   or read the files directly.)
2. **Both ranks produced a result**: ``res0`` and ``res1`` from
   ``submit_pp2_ranks`` are both non-None (neither timed out nor crashed
   silently).
3. **Distributed rendezvous actually happened**: the launch logs contain
   evidence of ``init_process_group`` succeeding (or the framework's
   equivalent) — not just one rank starting and the other immediately
   erroring.

If any of these is missing, the iteration's verdict MUST reflect that
(``outcome: logic_fail`` with a clear reason), not claim success.

See ``docs/agent-sdk-guide.md`` for the full SDK cookbook (log tailing,
error handling, status codes).

"""
    # Only one worker — still useful for GPU isolation but no PP2.
    return f"""\
# 🌐 Remote worker configured — launch via cluster SDK is REQUIRED

This task has one worker node configured: ``{nodes_str}``. **Every
end-to-end framework launch MUST run on that worker via the cluster
SDK**, not locally on the orchestrator node. The whole point of having
a remote worker is GPU isolation from the orchestrator.

```python
from metainfer.cluster.sdk import submit_script

# Read GPU count from launch_constraints; do NOT assume a single GPU.
gpu_count = 4  # parsed from launch_constraints, NOT hardcoded by port_model
job_id, result = submit_script(
    worker_node_id={worker_nodes[0]!r},
    gpu_slots=[({worker_nodes[0]!r}, i) for i in range(gpu_count)],
    script_body="export PYTHONPATH=... && python -m <framework.launcher> ...",
    timeout_s=1800,
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

    return banner + CORE_DISCIPLINE + P5_DISCIPLINE + notes + _launch_constraints_block(req) + _distributed_block(worker_nodes) + f"""\
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

def format_prev_p6_verdict(prev_verdict: Dict[str, Any]) -> str:
    """Render the previous P6 iteration's verdict into a structured handover
    block for the next iteration.

    The orchestrator stores this string in its ``prev_failure`` slot and
    passes it to the next ``p6_port_engine_prompt`` call. Keeping the
    formatting in prompts.py (not in pipeline.py) ensures the rendered
    block matches what the prompt's ``# Previous iteration handover``
    section documents.

    Returns "" if ``prev_verdict`` is empty / lacks the structured fields.
    """
    if not prev_verdict or not isinstance(prev_verdict, dict):
        return ""

    parts: List[str] = []

    reason = (prev_verdict.get("reason") or "").strip()
    if reason:
        parts.append(f"reason: {reason}")

    bad_layer = prev_verdict.get("similarity_first_bad_layer")
    bad_row = prev_verdict.get("similarity_first_bad_row")
    sim_min = prev_verdict.get("similarity_min")
    if bad_layer is not None or bad_row is not None or sim_min is not None:
        parts.append(
            f"similarity_min={sim_min} first_bad_layer={bad_layer} "
            f"first_bad_row={bad_row}"
        )

    inner = prev_verdict.get("inner_attempts")
    if isinstance(inner, int):
        parts.append(f"inner_attempts_this_iter={inner}")

    replacements = prev_verdict.get("operator_replacements") or []
    if isinstance(replacements, list) and replacements:
        parts.append("operator_replacements_tried:")
        for r in replacements:
            if not isinstance(r, dict):
                continue
            op = r.get("op", "?")
            strat = r.get("strategy", "?")
            env = r.get("env_var", "-")
            sha = r.get("commit_sha", "-")
            why = (r.get("reason") or "").strip().replace("\n", " ")[:200]
            parts.append(
                f"  - op={op} strategy={strat} env={env} "
                f"commit={sha} reason={why}"
            )

    if not parts:
        return ""

    return "\n".join(parts)


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

    return banner + CORE_DISCIPLINE + P6_DISCIPLINE + notes + prev_block + _launch_constraints_block(req) + _distributed_block(worker_nodes) + f"""\
# Task: 推理引擎移植工程师 — port the model into TARGET_FRAMEWORK_DIR

P3 consolidated spec (READ):
  {p3_path}

P4 minimal framework (READ — reuse its operator implementations as
reference fallbacks; located under the workspace beside your workdir):
  ../p4/

Golden dumps from the minimal framework (READ, your similarity oracle):
  {p5_dumps_dir}

This is iteration #{iteration} of P6. Each iteration that produces a
non-empty change set MUST end with a git commit inside
``TARGET_FRAMEWORK_DIR`` (see below).

## ⚙️ This is an iterative port-and-test loop, NOT a one-shot launch

Real target frameworks (sglang / vLLM / TRT-LLM / …) will almost never
boot your model on the first try. Operator incompatibility (hardware-gen
mismatch, missing fused kernel), shape / dtype drift, unsupported
attention backend — they all show up as either launch crashes or output
divergence. **Your job is to LOOP internally until the 3-prompt batch
runs end-to-end with cosine ≥ 0.99 against the P5 golden dumps.**

Within THIS iteration, repeat the following cycle up to ~5 times (each
cycle is a launch attempt; log every attempt, never silently overwrite):

    ATTEMPT 1..N (within this single P6 invocation):
      1. LAUNCH     — boot the model with current code. Capture full
                      stdout+stderr to ``{workdir}/launch_attempt_<K>.log``
                      (K = monotonically increasing across attempts
                      WITHIN this iteration; do NOT reuse numbers).
      2. DIAGNOSE   — if it crashed, identify the SINGLE failing
                      operator from the traceback (the deepest frame
                      inside the framework's own kernel/op code, not
                      Python stdlib noise). Write one short sentence:
                      "operator X failed because Y".
      3. REPLACE    — gate a fallback code path behind a NEW env var
                      or branch. ADD-ONLY (never delete framework code).
                      Pick a strategy from the hierarchy below.
      4. RELAUNCH   — boot again with the new env var set. If it still
                      crashes on the SAME operator, your replacement is
                      wrong — revise it. If it crashes on a DIFFERENT
                      operator, progress: back to step 2 with the new
                      traceback.
      5. INFER      — when boot succeeds, run the 3-prompt batch (see below)
                      and dump hidden_states at the same checkpoints P4 used.
      6. DUMP-CMP   — load P5 golden dumps, compute cosine per checkpoint.
      7. BISECT     — if any cosine < 0.99: walk layers from 0 upward,
                      find the FIRST divergent one — the operator that
                      produces that layer's output is the next target.
                      Back to step 3 (you have a new op to replace).
      8. STOP       — cosine ≥ 0.99 everywhere AND output is semantically
                      sensible → write verdict, commit, finish.

Cap your inner attempts around 5; if you cannot converge in one P6
iteration, write the verdict with what you learned so the next P6
iteration continues from there (the orchestrator will start one).

### 🧱 Operator replacement strategy hierarchy (try in this order)

When the framework's stock operator won't run on your hardware / model,
pick the FIRST option that applies. Always gate behind a new env var so
the original code path stays untouched:

  1. **Framework-native flag / env var fallback** — many frameworks
     expose a "use simpler kernel" flag (e.g. ``--attention-backend
     triton``, ``--disable-cuda-graph``, ``SGL_ENABLE_<X>_FALLBACK=1``).
     grep the framework for env vars / config knobs referenced in the
     failing code path. Always preferred: framework maintains it.

  2. **Triton re-implementation** — write a Triton kernel with the same
     I/O dtype+shape. Good for attention, fused GEMM, RMSNorm, RoPE,
     quantized GEMM. Gate behind e.g. ``METAINFER_OPS_USE_TRITON=1``.

  3. **Pure PyTorch reference** — last-resort, slowest, always correct.
     Use the operator's mathematical definition from the P3 spec, OR
     copy the relevant routine verbatim from ``../p4/`` (the P4
     minimal framework is correctness-checked against the model — its
     operators are by-definition correct). Gate behind
     ``METAINFER_OPS_USE_TORCH=1``.

  4. **P4 reference impl as drop-in** — for MoE routers, dequantization,
     RoPE, attention scoring, the P4 code already has a working (if
     slow) reference. Read it, adapt the interface, gate it.

**Hard rules for every replacement**:
  - ADD-ONLY. Gate every replacement behind a NEW env var or branch.
    Never delete or overwrite framework code.
  - One commit per replacement inside ``{target_fw}``:
    ``git commit -m "port_model(P6 iter {iteration}): replace <op> via <strategy>"``
  - Record every replacement in your verdict's ``operator_replacements``
    field (see verdict schema below) — the next P6 iteration reads this
    to skip already-tried approaches.

### 🔬 Diagnosing "operator unsupported on my hardware"

A common failure: the framework has a kernel that only supports specific
hardware gens (e.g. aiter / tilelang fp8 MMAC requiring gfx938 / gfx92a
/ gfx946; cutlass kernels requiring sm_80+). Symptoms include:

  - Compile errors mentioning the GPU arch (``gfx928``, ``sm_80``, …) or
    ``MMAC operations are only supported on … architectures``.
  - ``RecursionError`` / ``ImportError`` from the operator's module
    (some frameworks fail at import time when the kernel can't JIT).
  - ``CUDA error: no kernel image is available for execution``.

When you see these:
  - The failing operator's module name is in the traceback — find it.
  - grep the framework for the capability check (usually an ``if`` on
    ``torch.cuda.get_device_capability()``, or a separate backend module
    selected by arch string).
  - Add a branch that forces the dispatch to a Triton or pure-torch
    fallback you write (strategy #2 or #3 above), gated by e.g.
    ``METAINFER_OPS_FORCE_<NAME>=1``.
  - This is the canonical use case for strategy #2/#3 — the framework's
    fast path simply doesn't target your hardware, so the only viable
    path is a correct (if slow) replacement.

### 🎯 Dump-driven bisection (when launch succeeds but output is wrong)

When the framework boots and produces tokens but the 3-prompt batch is
garbage (wrong language, ``<unk>``, punctuation-only, etc.), you MUST
use the P5 golden dumps to localize the bug — do NOT guess:

  - For each layer index ``L`` from 0 upward:
    * Load ``{p5_dumps_dir}/row<R>/layer_<NNN>_<checkpoint>.npy``
      (golden) and ``{workdir}/dumps/row<R>/layer_<NNN>_<checkpoint>.npy``
      (yours), for every checkpoint the P4 framework dumped.
    * Compute cosine similarity per checkpoint.
    * The **first layer** where any checkpoint has cosine < 0.99 is
      your culprit — record its index as ``similarity_first_bad_layer``.
  - Read the framework code for that layer's forward pass; identify
    which sub-operator (QK torch.mm, softmax, RoPE, MoE router, …)
    produces the divergent checkpoint.
  - Apply the replacement strategy hierarchy to that specific operator.
  - On the next inner attempt, re-run and re-compare — cosine should
    improve at that layer. If it doesn't, your replacement is wrong.

### 📥 Test batch (the 3 prompts P5 used)

When launch succeeds, run this exact batch as one left-padded forward
pass (same convention as P4 / P5):

```
世界上最高的山峰是
中国的国旗是
人体正常体温约为
```

Dump hidden_states with the SAME per-row layout as P5:
``dumps/row0/``, ``dumps/row1/``, ``dumps/row2/``, each containing
``layer_<NNN>_<checkpoint>.npy``.

### 📤 Git commit

After each successful replacement AND at iteration end, commit inside
``{target_fw}``:
```
cd {target_fw}
git add -A
git commit -m "port_model(P6 iter {iteration}): <one-line summary>"
```
If ``{target_fw}`` is not yet a git repo, run ``git init`` first.
Record the FINAL commit SHA to ``{workdir}/commit_{iteration}.txt``.

### 📋 Verdict — ``{workdir}/verdict_{iteration}.json``

```
{{
  "iteration": {iteration},
  "launched": true|false,
  "inner_attempts": <int — launches tried within this iter>,
  "operator_replacements": [
    {{"op": "<fully-qualified op name, e.g. GlmMoeDSAAttention.forward>",
      "strategy": "flag-fallback|triton|pure-torch|p4-reference",
      "env_var": "<the new env var you introduced, or null>",
      "commit_sha": "<sha of the per-replacement commit>",
      "reason": "<one short sentence — what failed, why this fixes it>"}}
  ],
  "batch": [
    {{"prompt": "世界上最高的山峰是", "topk_text": [...],
      "verifier_judgment": "passed|failed", "verifier_reason": "..."}},
    {{"prompt": "中国的国旗是", ...}},
    {{"prompt": "人体正常体温约为", ...}}
  ],
  "similarity_min": <float or null — min cosine across all rows / checkpoints>,
  "similarity_first_bad_layer": <int or null>,
  "similarity_first_bad_row": <int or null>,
  "commit_sha": "<final iter commit sha or null>",
  "outcome": "ok|needs_repair|test_fail",
  "reason": "<short — what blocked you; the next iter reads this>"
}}
```

Outcome mapping:
  - ``ok``          — cosine ≥ 0.99 everywhere AND output semantically correct.
  - ``needs_repair`` — you made progress but didn't converge (e.g. ran
                       out of inner attempts). MUST include the
                       ``operator_replacements`` you've made and the
                       ``similarity_first_bad_layer`` so the next iter
                       continues.
  - ``test_fail``   — you cannot even boot the model after exhausting
                       the strategy hierarchy. Explain in ``reason``.

{SUMMARY_CONTRACT}
"""
