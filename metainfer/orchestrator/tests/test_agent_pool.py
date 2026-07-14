"""Unit tests for :mod:`metainfer.orchestrator.agent_pool`.

Run directly::

    python metainfer/orchestrator/tests/test_agent_pool.py

The tests use a FakeManager that records every AgentSpec passed to
launch() and returns a programmable AgentResult. No subprocess is
spawned — we only verify the pool's distribution / session-threading
logic.
"""

from __future__ import annotations

import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional

from metainfer.orchestrator.agent_pool import AgentPool, PoolTask
from metainfer.orchestrator.subagent_manager import AgentResult, AgentSpec


class FakeManager:
    """Records AgentSpecs the pool asks it to launch and synthesizes
    a result for each. Optionally sleeps to expose concurrency bugs."""

    def __init__(
        self,
        *,
        sleep_s: float = 0.0,
        fail_keys: Optional[set] = None,
    ) -> None:
        self.launches: List[AgentSpec] = []
        self.results = {}  # name -> AgentResult
        self._sleep_s = sleep_s
        self._fail_keys = fail_keys or set()
        self._launch_times: List[float] = []  # for concurrency check

    def launch(self, spec: AgentSpec):
        self._launch_times.append(time.time())
        self.launches.append(spec)
        if self._sleep_s:
            time.sleep(self._sleep_s)
        # Synthesize a deterministic result. Session id is derived from
        # the spec's resume_session_id (resume keeps the same id) or a
        # fresh one for turn 0.
        sid = spec.resume_session_id or f"sess-{spec.name}"
        success = spec.name not in self._fail_keys
        self.results[spec.name] = AgentResult(
            name=spec.name,
            role=spec.role,
            success=success,
            returncode=0 if success else 1,
            duration_s=0.01,
            final_text=f"answer-for-{spec.name}",
            session_id=sid,
            usage={"total_cost_usd": 0.001, "usage": {"input_tokens": 10}},
            error=None if success else "synthetic failure",
            failure_mode=None if success else "logic",
        )

    def result(self, name: str):
        return self.results.get(name)

    @property
    def launch_times(self) -> List[float]:
        return list(self._launch_times)


def _task(key: str, tmp: Path) -> PoolTask:
    return PoolTask(
        key=key,
        prompt=f"prompt for {key}",
        workdir=tmp / key,
    )


def test_empty_batch_returns_empty():
    with tempfile.TemporaryDirectory() as td:
        mgr = FakeManager()
        pool = AgentPool(mgr, n_workers=3, log_dir=Path(td))
        assert pool.run([]) == []
        assert mgr.launches == []


def test_round_robin_distribution():
    """N=3 workers, 7 tasks → w0=[0,3,6], w1=[1,4], w2=[2,5]."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mgr = FakeManager()
        pool = AgentPool(mgr, n_workers=3, log_dir=tmp / "logs",
                         name_prefix="val")
        tasks = [_task(f"t{i}", tmp) for i in range(7)]
        results = pool.run(tasks)

        # Result count and order
        assert len(results) == 7
        assert [r.key for r in results] == [f"t{i}" for i in range(7)]

        # Each task ran on the expected worker (i % 3).
        for i, r in enumerate(results):
            assert r.worker_id == i % 3, f"task {i} on wrong worker"

        # Worker 0 handled 3 turns (t0, t3, t6).
        w0_turns = sorted(r.turn for r in results if r.worker_id == 0)
        assert w0_turns == [0, 1, 2]
        w1_turns = sorted(r.turn for r in results if r.worker_id == 1)
        assert w1_turns == [0, 1]
        w2_turns = sorted(r.turn for r in results if r.worker_id == 2)
        assert w2_turns == [0, 1]


def test_session_threading_within_worker():
    """Turn 0 of each worker starts fresh; turn 1+ resumes the prior session."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mgr = FakeManager()
        pool = AgentPool(mgr, n_workers=2, log_dir=tmp / "logs")
        tasks = [_task(f"t{i}", tmp) for i in range(4)]  # 2 per worker
        results = pool.run(tasks)

        # Find specs by worker.
        w0_specs = [s for s in mgr.launches if "w0_t0" in s.name or "w0_t1" in s.name]
        w1_specs = [s for s in mgr.launches if "w1_t0" in s.name or "w1_t1" in s.name]

        # Sort by turn index embedded in name.
        def turn_of(s: AgentSpec) -> int:
            return 0 if "_t0_" in s.name else 1
        w0_specs.sort(key=turn_of)
        w1_specs.sort(key=turn_of)

        # Turn 0: no resume_session_id (fresh start).
        assert w0_specs[0].resume_session_id is None
        assert w1_specs[0].resume_session_id is None

        # Turn 1: resumes the session id captured from turn 0's result.
        # We capture sid from FakeManager's result for the turn-0 spec.
        sid_w0_t0 = mgr.results[w0_specs[0].name].session_id
        sid_w1_t0 = mgr.results[w1_specs[0].name].session_id
        assert w0_specs[1].resume_session_id == sid_w0_t0
        assert w1_specs[1].resume_session_id == sid_w1_t0

        # Results also surface the session id they ended on.
        for r in results:
            assert r.session_id is not None


def test_workers_run_in_parallel():
    """With sleep_s=0.2 and N=3 workers running 3 tasks, total time
    should be ~0.2s, NOT 0.6s (sequential)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mgr = FakeManager(sleep_s=0.2)
        pool = AgentPool(mgr, n_workers=3, log_dir=tmp / "logs")
        t0 = time.time()
        results = pool.run([_task(f"t{i}", tmp) for i in range(3)])
        elapsed = time.time() - t0
        # Parallel: ~0.2s. Sequential: ~0.6s. Generous bound at 0.5s.
        assert elapsed < 0.5, f"workers not parallel: elapsed={elapsed:.2f}s"
        assert len(results) == 3


def test_task_failure_does_not_kill_worker():
    """If one task fails, the worker continues to its remaining tasks."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        # Make t1 fail; w1 should still process t4 (turn 1).
        mgr = FakeManager(fail_keys={"pool_w1_t0_t1"})
        pool = AgentPool(mgr, n_workers=3, log_dir=tmp / "logs")
        tasks = [_task(f"t{i}", tmp) for i in range(6)]
        results = pool.run(tasks)
        by_key = {r.key: r for r in results}
        assert by_key["t1"].success is False
        # t4 is on w1 (i=4 % 3 = 1) and should still have run.
        assert by_key["t4"].success is True
        assert by_key["t4"].worker_id == 1


def test_duplicate_keys_rejected():
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mgr = FakeManager()
        pool = AgentPool(mgr, n_workers=2, log_dir=tmp / "logs")
        tasks = [_task("dup", tmp), _task("dup", tmp)]
        try:
            pool.run(tasks)
        except ValueError as e:
            assert "dup" in str(e)
        else:
            raise AssertionError("expected ValueError for duplicate keys")


def test_n_workers_validation():
    with tempfile.TemporaryDirectory() as td:
        mgr = FakeManager()
        try:
            AgentPool(mgr, n_workers=0, log_dir=Path(td))
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for n_workers=0")


def test_unique_agent_names():
    """Every spec the manager sees must have a unique name."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        mgr = FakeManager()
        pool = AgentPool(mgr, n_workers=3, log_dir=tmp / "logs")
        pool.run([_task(f"t{i}", tmp) for i in range(9)])
        names = [s.name for s in mgr.launches]
        assert len(names) == len(set(names)), "agent name collision"


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {fn.__name__}: {e!r}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {len(fns)} total")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if _run_all() else 1)
