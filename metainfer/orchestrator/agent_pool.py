"""AgentPool: fixed-N worker pool that reuses session context across turns.

Problem this solves
-------------------
The orchestrator's per-node validators used to spawn one fresh Claude Code
subprocess per (round, node) — e.g. 43 nodes × 3 rounds = 129 cold starts,
each loading the same 80KB memory.json + graph context from scratch. Even
with prompt caching, ``cache_read_input_tokens`` dominated cost because
every fresh session pays the cache-lookup penalty on turn 1.

This module keeps ``N`` logical workers alive for the duration of a batch.
Each worker is a *session*, not a long-lived process: turn 1 starts a
fresh ``ccb`` session (capturing its id), and every subsequent turn
launches ``ccb --resume <session_id>`` so the model wakes up with the
primer + all prior Q/A still in context. Cache-hit rate on resumed turns
is ~95%, so turn 2+ cost roughly the *delta* prompt size — not the full
primer.

Distribution is round-robin: tasks ``[t0, t1, t2, t3, t4, t5, t6]`` with
``N=3`` workers dispatch as ``w0=[t0,t3,t6], w1=[t1,t4], w2=[t2,t5]``.
This balances load and keeps each worker's session queue long enough to
amortize the first-turn primer cost.

Concurrency model
-----------------
Each worker runs on its own background thread. Within a worker, turns
are strictly sequential (a session can only process one prompt at a
time). Across workers, turns run in parallel — bounded externally by
``SubAgentManager.max_concurrent`` (semaphore on ccb subprocesses).

The pool is *batch-oriented*: callers pass ``List[PoolTask]``, the pool
runs them, returns ``List[PoolTaskResult]`` (in INPUT ORDER, not
worker-grouped). There is no streaming / dynamic-dispatch API — keep it
simple until a caller actually needs it.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .subagent_manager import AgentResult, AgentSpec, SubAgentManager


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class PoolTask:
    """One unit of work submitted to the pool.

    Attributes:
        key: Stable identifier for the task (e.g. ``"mlp_0_node_3"``).
            Used to map results back to inputs. Must be unique within a
            single ``AgentPool.run`` batch.
        prompt: Full prompt text for this turn. The pool will pipe it
            through the manager's stdin mechanism (AgentSpec.prompt_file).
        workdir: Where this task's agent runs. Each task MUST have its
            own workdir so per-task output files (verdict.json,
            response.txt) don't collide — session state lives in ccb's
            own storage, not here, so workdir is just the file sandbox.
        name: Agent name used in logs / status. Must be unique within
            the batch; the pool already namespaces by worker, so the
            caller only needs to ensure cross-task uniqueness. Defaults
            to ``f"task_{key}"`` if omitted.
    """

    key: str
    prompt: str
    workdir: Path
    name: Optional[str] = None


@dataclass
class PoolTaskResult:
    """Result of one PoolTask. Returned in input order."""

    key: str
    worker_id: int
    turn: int  # 0-based turn index within this worker's queue
    success: bool
    final_text: str
    session_id: Optional[str]
    usage: Optional[Dict[str, Any]]
    error: Optional[str]
    duration_s: float


@dataclass
class _WorkerState:
    """Mutable per-worker bookkeeping. Lives on the worker thread."""
    worker_id: int
    session_id: Optional[str] = None  # captured from turn 0, resumed after
    turns_completed: int = 0


# --------------------------------------------------------------------------- #
# Pool
# --------------------------------------------------------------------------- #


class AgentPool:
    """Fixed-N worker pool that shares session context across turns.

    The pool wraps an existing :class:`SubAgentManager` — it does NOT
    spawn processes itself. Each turn becomes one ``manager.launch(spec)``
    call. The pool's only job is to (a) pin sessions to workers,
    (b) thread ``resume_session_id`` through subsequent turns, and
    (c) parallelize across workers.

    Construction is cheap; the heavy work happens in :meth:`run`. A pool
    can be reused across batches (e.g. one pool for batch N's parallel
    work, another batch for batch N+1). Each batch starts fresh
    sessions — context carryover between unrelated batches is rarely what
    callers want.
    """

    def __init__(
        self,
        manager: SubAgentManager,
        *,
        n_workers: int = 3,
        log_dir: Path,
        role: str = "pool_worker",
        name_prefix: str = "pool",
        model: Optional[str] = None,
        timeout_s: int = 600,
        stuck_timeout_s: int = 300,
        max_retries: int = 2,
        extra_args: Optional[List[str]] = None,
        env_overrides: Optional[Dict[str, str]] = None,
    ) -> None:
        if n_workers < 1:
            raise ValueError(f"n_workers must be >= 1, got {n_workers}")
        self.manager = manager
        self.n_workers = n_workers
        self.log_dir = Path(log_dir)
        self.role = role
        self.name_prefix = name_prefix
        self.model = model
        self.timeout_s = timeout_s
        self.stuck_timeout_s = stuck_timeout_s
        self.max_retries = max_retries
        self.extra_args = list(extra_args or [])
        self.env_overrides = dict(env_overrides or {})

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def run(self, tasks: List[PoolTask]) -> List[PoolTaskResult]:
        """Distribute ``tasks`` round-robin across N workers, run them,
        return results in INPUT ORDER (not worker-grouped).

        Empty batch → returns ``[]`` immediately. Caller can poll
        ``self.manager.snapshot()`` concurrently to observe progress.
        """
        if not tasks:
            return []

        # Validate keys are unique before dispatching — non-unique keys
        # would silently corrupt the result map.
        seen = set()
        for t in tasks:
            if t.key in seen:
                raise ValueError(f"duplicate PoolTask.key: {t.key!r}")
            seen.add(t.key)

        # Round-robin assignment: task[i] → worker[i % N].
        # Workers with lower indices get the extra task when N doesn't
        # divide evenly. Order WITHIN a worker is ascending task index,
        # so each worker sees a deterministic turn sequence.
        per_worker: List[List[int]] = [[] for _ in range(self.n_workers)]
        for i in range(len(tasks)):
            per_worker[i % self.n_workers].append(i)

        results: Dict[str, PoolTaskResult] = {}
        threads: List[threading.Thread] = []
        for worker_id, task_idxs in enumerate(per_worker):
            if not task_idxs:
                continue
            t = threading.Thread(
                target=self._worker_loop,
                args=(worker_id, task_idxs, tasks, results),
                name=f"{self.name_prefix}-w{worker_id}",
                daemon=True,
            )
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # Return in INPUT order so callers don't have to re-sort.
        return [results[t.key] for t in tasks if t.key in results]

    # ------------------------------------------------------------------ #
    # Worker loop
    # ------------------------------------------------------------------ #

    def _worker_loop(
        self,
        worker_id: int,
        task_idxs: List[int],
        tasks: List[PoolTask],
        results: Dict[str, PoolTaskResult],
    ) -> None:
        """Process assigned tasks sequentially, threading session ids."""
        state = _WorkerState(worker_id=worker_id)
        for turn_idx, original_i in enumerate(task_idxs):
            task = tasks[original_i]
            try:
                result = self._run_one_turn(task, state, turn_idx)
            except Exception as exc:  # noqa: BLE001
                # Worker must not die from one bad task — surface the
                # error in the result and keep going so the rest of the
                # batch completes.
                result = PoolTaskResult(
                    key=task.key,
                    worker_id=worker_id,
                    turn=turn_idx,
                    success=False,
                    final_text="",
                    session_id=state.session_id,
                    usage=None,
                    error=f"pool worker exception: {exc!r}",
                    duration_s=0.0,
                )
            results[task.key] = result

    def _run_one_turn(
        self, task: PoolTask, state: _WorkerState, turn_idx: int,
    ) -> PoolTaskResult:
        """Build an AgentSpec for this turn, launch it, capture the
        session id for the next turn."""
        agent_name = task.name or f"{self.name_prefix}_w{state.worker_id}_t{turn_idx}_{task.key}"
        agent_name = _safe_name(agent_name)

        workdir = Path(task.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        prompt_file = workdir / f"{agent_name}.prompt.txt"
        prompt_file.write_text(task.prompt, encoding="utf-8")

        # Per-worker log subdir so attempts from different workers (and
        # different turns) don't clobber each other's .log files.
        worker_log_dir = self.log_dir / f"w{state.worker_id}" / agent_name
        worker_log_dir.mkdir(parents=True, exist_ok=True)

        spec = AgentSpec(
            name=agent_name,
            role=self.role,
            prompt_file=prompt_file,
            workdir=workdir,
            log_dir=worker_log_dir,
            timeout_s=self.timeout_s,
            stuck_timeout_s=self.stuck_timeout_s,
            max_retries=self.max_retries,
            extra_args=list(self.extra_args),
            env_overrides=dict(self.env_overrides),
            model=self.model,
            # Turn 0 starts fresh; turn 1+ resumes the worker's session.
            # ccb's --resume flag pulls the full prior conversation back
            # into context — primer + earlier nodes' verdicts stay live.
            resume_session_id=(
                state.session_id if state.turns_completed > 0 else None
            ),
        )

        t0 = time.time()
        self.manager.launch(spec)
        elapsed = time.time() - t0
        ar: Optional[AgentResult] = self.manager.result(agent_name)

        if ar is not None and ar.session_id:
            # Keep the session id current so the NEXT turn resumes from
            # the latest point. For --resume runs ccb returns the same
            # id; for turn 0 it mints a fresh one we capture here.
            state.session_id = ar.session_id
        state.turns_completed += 1

        return PoolTaskResult(
            key=task.key,
            worker_id=state.worker_id,
            turn=turn_idx,
            success=bool(ar.success) if ar else False,
            final_text=(ar.final_text if ar else "") or "",
            session_id=state.session_id,
            usage=ar.usage if ar else None,
            error=ar.error if ar else "no AgentResult produced",
            duration_s=round(elapsed, 2),
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _safe_name(s: str) -> str:
    """Filesystem-safe agent name. SubAgentManager embeds this in log
    file paths, so it cannot contain ``/`` or other shell-unfriendly
    chars. Truncate to keep paths manageable on long node ids."""
    import re
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)[:120] or "pool_task"
