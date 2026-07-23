# MetaInfer Cluster SDK — Agent Guide

This is the **agent-facing** cookbook for the cluster SDK. If you're an
agent working inside `evolve_kernel` or `port_model` and need to run
something on a remote GPU worker, this is your manual.

## TL;DR — when to use what

| Scenario | Use |
|---|---|
| Run a Python script / shell command on a worker GPU | `submit_script` |
| Spawn a claude-code sub-agent on a worker GPU | `submit_agent` |
| Launch a TP/PP2 distributed inference across 2 workers | `submit_pp2_ranks` |
| Read live stdout/stderr from a submitted job | `tail_stdout` / `tail_stderr` |

All SDK functions live in `metainfer.cluster.sdk`.

---

## 1. submit_script — run a shell command on a worker

```python
from metainfer.cluster.sdk import submit_script

job_id, result = submit_script(
    worker_node_id="gpu-node-2",
    script_body="nvidia-smi && python my_benchmark.py\n",
    gpu_slots=[("gpu-node-2", 0)],   # claim GPU 0 exclusively
    timeout_s=600.0,                  # wall-clock budget for the run
    env={"CUDA_VISIBLE_DEVICES": "0", "MY_FLAG": "1"},
)
# Blocks until worker finishes or timeout.
# result.status ∈ {"done", "failed", "timeout", "cancelled", "worker_dead"}
# result.exit_code is the subprocess exit code (0 = success).
```

**Key rules:**
- `gpu_slots` is `[(node_id, gpu_idx), ...]`. Omit it only for CPU-only work.
- Script body is multi-line shell — the worker runs it via `bash`.
- The slot is auto-released in `finally`, even on exceptions.
- Assume `script_body` runs in worker's `$METAINFER_ROOT` (shared NFS).

---

## 2. submit_agent — spawn a claude-code sub-agent

```python
from metainfer.cluster.sdk import submit_agent

job_id, result = submit_agent(
    worker_node_id="gpu-node-2",
    prompt_body="""You are a kernel-debugging agent. Read /shared/log.txt,
diagnose the failure, and write a fixed kernel to /shared/fixed.py.""",
    gpu_slots=[("gpu-node-2", 0)],
    timeout_s=3600.0,
)
```

The worker node **must** have `ccb` installed and on `$PATH`.

---

## 3. submit_pp2_ranks — distributed inference across two workers

For PP2 / TP-style 2-rank launches where rank0 runs on worker A and
rank1 runs on worker B. The SDK pre-allocates one GPU per worker and
injects the standard `torch.distributed` rendezvous env vars; your
script just needs to honor them.

```python
from metainfer.cluster.sdk import submit_pp2_ranks, PP2RankSpec

results = submit_pp2_ranks(
    rank_a=PP2RankSpec(
        worker_node_id="gpu-A",
        gpu_idx=0,
        command="cd /shared/target_fw && python -m launcher --rank 0\n",
    ),
    rank_b=PP2RankSpec(
        worker_node_id="gpu-B",
        gpu_idx=0,
        command="cd /shared/target_fw && python -m launcher --rank 1\n",
    ),
    timeout_s=1800.0,
)
# results is {rank_index: JobResult}. Check each .status / .exit_code.
```

Injected env (per rank):
- `RANK` = 0 or 1
- `WORLD_SIZE` = 2
- `LOCAL_RANK` = 0 (each worker has 1 proc)
- `NODE_RANK` = 0 or 1
- `NNODES` = 2
- `NPROC_PER_NODE` = 1
- `MASTER_ADDR` = worker A's IP
- `MASTER_PORT` = free port on worker A (chosen by SDK)

**Important:** your launcher script **must** retry the rendezvous (NCCL
init can fail if rank1 connects before rank0 listens). Wrap with a 30s
retry loop. See `docs/multi-node-architecture.md` §"PP2 启动".

---

## 4. tail_stdout / tail_stderr — read streaming logs

```python
from metainfer.cluster.sdk import tail_stdout

chunk = tail_stdout(job_id, worker_node_id="gpu-node-2", offset=0)
# Returns bytes. Pass offset = len(previous_read) for incremental tailing.
```

Logs accumulate into `inbox/<worker>/<job_id>/stdout.log` as the worker's
subprocess writes. Safe to call repeatedly.

---

## 5. Error handling patterns

### Pattern A: treat worker failure as task failure

```python
job_id, result = submit_script(...)
if result is None or result.status != "done":
    # Worker died / timed out / cancelled. Surface as failure.
    return Outcome.FAIL, f"worker status={result.status if result else 'unknown'}"
```

### Pattern B: distinguish kernel bug from worker infra bug

```python
if result.status == "worker_dead":
    # Worker crashed (hardware / OOM kill / daemon died). Not our bug.
    emit_worker_failure_event(worker_node, job_id)
    return Outcome.INFRA_FAIL
elif result.status == "timeout":
    # Kernel hung. Could be our bug or just slow.
    return Outcome.TEST_FAIL
elif result.exit_code != 0:
    # Our script crashed — almost always a kernel/framework bug.
    return Outcome.TEST_FAIL
```

### Pattern C: don't auto-requeue

The SDK does **not** auto-retry failed jobs. If you want a retry, decide
at the application layer (e.g., emit a timeline event and let the
orchestrator's pipeline decide).

---

## 6. What the agent should NOT do

- ❌ Touch `cluster/scoreboard/<...>/*.claim` directly. Always go through
  `acquire_gpus` / `release_gpus` / `submit_*`.
- ❌ Write into `cluster/inbox/<worker>/<job_id>/job.json` manually. Use
  `submit_job` / `submit_script`.
- ❌ Kill the worker's subprocess yourself. Submit a `cancel.marker`
  via the WebUI `/api/cluster/scoreboard/force-release` endpoint or just
  let the timeout fire.
- ❌ Poll `read_result` in a hot loop. Use `submit_script(block=True)` —
  it has an inline reaper that surfaces dead workers within ~5s.

---

## 7. Status codes reference

| `JobResult.status` | Meaning | What to do |
|---|---|---|
| `done` | Subprocess exited 0 | Parse stdout, proceed. |
| `failed` | Subprocess exited non-zero | Read stderr.log — likely our bug. |
| `timeout` | Wall-clock `timeout_s` exceeded | Kernel / framework hung. |
| `cancelled` | Admin force-kill via WebUI | Surface as user-interrupted. |
| `worker_dead` | Worker heartbeat went stale | Infra issue — re-queue or surface. |

---

## 8. Cookbook — evolve_kernel perf phase

The evolve_kernel pipeline already does this (see
`metainfer/tasks/evolve_kernel/orchestrator/harness.py::_run_perf_test_remote`),
but here's the pattern distilled:

```python
# Build a self-contained perf script that writes a JSON summary to stdout.
script = f"""
python3 {harness_path} {kernel_path}
"""

job_id, result = submit_script(
    worker_node_id=worker,
    script_body=script,
    gpu_slots=[(worker, 0)],
    timeout_s=600.0,
    env={"METAINFER_KERNEL_PATH": str(kernel_path)},
)

if result.status != "done":
    return False, {"passed": False, "worker_status": result.status}

# Parse stdout.log for the harness's JSON output.
from metainfer.cluster.sdk import tail_stdout
stdout_text = tail_stdout(job_id, worker_node_id=worker).decode("utf-8", "replace")
parsed = _extract_json(stdout_text)
```

---

## Reference

- Architecture deep-dive: `docs/multi-node-architecture.md`
- SDK source: `metainfer/cluster/sdk.py`
- Status constants: `metainfer/cluster/queue_schema.py`
- Example integration: `metainfer/tasks/evolve_kernel/orchestrator/harness.py`
