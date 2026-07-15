"""Pipeline — the task's core iteration loop.

The pipeline controls the orchestrator's main loop:

- Reads the current phase from ``StateStore`` (or resumes from last run).
- For each phase, calls the relevant implementation (e.g. a sub-agent step).
- Transitions to the next phase based on outcomes.
- Writes iteration records and timeline events after each iteration.

Import pattern for shared infrastructure::

    from metainfer.orchestrator.state import StateStore
    from metainfer.orchestrator.iteration import IterationWorkspace
    from metainfer.orchestrator.subagent_manager import SubAgentManager
    from metainfer.orchestrator.token_budget import TokenBudget

Task-private imports (prompts, oracles) stay inside this package::

    from .prompts import SOME_TEMPLATE
    from .oracles.my_oracle import MyOracle
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class IterationRecord:
    """Task-private iteration schema.

    Each iteration record holds whatever fields this task needs.  Serialise
    with ``dataclasses.asdict(rec)`` → :meth:`StateStore.write_iteration`
    (accepts a plain dict).  The shell never interprets these fields — only
    the task's ``_state_readers.py`` and detail view know the schema.
    """
    n: int
    status: str = "running"       # "running" | "success" | "failed"
    started_at: float = 0.0
    ended_at: float = 0.0
    notes: list[str] = field(default_factory=list)
    # Add task-specific fields here, e.g. agent transcripts, scores, etc.


class Pipeline:
    """Minimal pipeline skeleton — replace with real logic."""

    def __init__(self, store, agent_manager, paths, req):
        self.store = store
        self.agent_manager = agent_manager
        self.paths = paths
        self.req = req

    def run(self, *, iter_limit: int = 10, is_resume: bool = False) -> None:
        """Run the main loop. Called by ``orchestrator.run_with_requirements``.

        Typical pattern::

            run = self.store.load_run()
            while run.current_iteration < iter_limit:
                # 1. Execute current phase
                # 2. Record iteration
                # 3. Transition to next phase via store.update_run()
                # 4. Check for terminal phase
                pass
        """
        # Dummy loop for demonstration.
        for n in range(1, iter_limit + 1):
            rec = IterationRecord(n=n, status="success")
            self.store.write_iteration(n, rec)
            self.store.update_run(current_iteration=n,
                                  current_phase="step2")
        self.store.update_run(finished=True, final_status="success",
                              current_phase="done")
