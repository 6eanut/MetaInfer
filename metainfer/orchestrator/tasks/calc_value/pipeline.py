"""Pipeline driver: S0 → S1 → S2 → S3 → S4.

Each step runs in an infinite retry loop — the system never enters a
Fail state. If a step raises, the error is logged to the timeline and
the step is retried after a short backoff. The only legitimate exit is
success (all five steps complete → ``final_status="success"``) or an
external signal (handled in ``orchestrator.py`` → ``final_status="aborted"``).

S0 (rough) is wrapped in an extra try/except: a fatal S0 failure does
NOT abort the pipeline because S1/S2/S3 don't depend on its output.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict

from . import phases as _phases


# Backoff between step retries (seconds). Constant — keeps the retry
# visible in the timeline without hammering the LLM provider.
_STEP_RETRY_BACKOFF_S = 5.0


def _set_phase(store, manager, phase: str, label: str = "") -> None:
    """Update run.json + emit a timeline event for the phase transition."""
    store.update_run(current_phase=phase, last_transition_label=label)
    store.append_timeline(
        "calc_value.phase.start",
        {"phase": phase, "label": label},
    )


def _run_step_with_retry(
    *,
    step_name: str,
    phase: str,
    store,
    manager,
    paths: Dict[str, Path],
    req: Dict[str, Any],
    runner: Callable[..., Any],
    set_phase_label: str,
    **kwargs,
) -> Any:
    """Run a calc-value step in an infinite retry loop.

    The system never gives up: on any exception, log it, back off, and
    retry. ``phase`` is re-asserted before every attempt so the WebUI's
    current-phase badge stays accurate.
    """
    attempt = 0
    last_exc: Exception | None = None
    while True:
        attempt += 1
        _set_phase(store, manager, phase,
                   label=f"{set_phase_label} (attempt {attempt})")
        try:
            return runner(req=req, store=store, manager=manager,
                          paths=paths, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            store.append_timeline(
                "calc_value.step.retry",
                {"step": step_name, "attempt": attempt, "error": str(exc)},
            )
            print(
                f"[calc-value] {step_name} attempt {attempt} failed: "
                f"{type(exc).__name__}: {exc}; retrying in "
                f"{_STEP_RETRY_BACKOFF_S}s",
                flush=True,
            )
            time.sleep(_STEP_RETRY_BACKOFF_S)


def run_pipeline(
    *,
    req: Dict[str, Any],
    store,
    manager,
    paths: Dict[str, Path],
) -> int:
    """Run the 4-step pipeline sequentially. Returns process exit code.

    Never writes ``final_status="failed"`` — the system explores forever
    and only stops on success (``final_status="success"``) or external
    signal (handled by the caller).
    """

    # Local imports so that adding stepN_*.py modules doesn't require
    # touching this file's imports if a step is mid-refactor.
    from .step0_rough import run_step0_rough
    from .step1_analyze import run_step1_analyze
    from .step2_graph import run_step2_graph
    from .step3_calculate import run_step3_calculate
    from .step4_visualize import run_step4_visualize

    # ---------------- S0: rough single-pass estimate ---------------- #
    # Runs FIRST so the WebUI has something to show within minutes.
    # Failures here are non-fatal — write empty results and move on;
    # the detailed audit (S3) will produce the authoritative numbers.
    try:
        rough_path = _run_step_with_retry(
            step_name="S0", phase=_phases.S0_ROUGH,
            store=store, manager=manager, paths=paths, req=req,
            runner=run_step0_rough,
            set_phase_label="rough single-pass estimate",
        )
        store.append_timeline("calc_value.s0.done",
                              {"rough": str(rough_path)})
    except Exception as exc:  # noqa: BLE001
        # Even the retry loop gave up. Log + keep going — S1/S2/S3 don't
        # depend on S0's output.
        store.append_timeline(
            "calc_value.s0.fatal",
            {"error": f"{type(exc).__name__}: {exc}"},
        )

    # ---------------- S1: analyze code ---------------- #
    memory_path = _run_step_with_retry(
        step_name="S1", phase=_phases.S1_ANALYZE,
        store=store, manager=manager, paths=paths, req=req,
        runner=run_step1_analyze,
        set_phase_label="analyze code from 3 angles",
    )
    store.append_timeline("calc_value.s1.done",
                          {"memory": str(memory_path)})

    # ---------------- S2: build & validate graph ---------------- #
    graph_path = _run_step_with_retry(
        step_name="S2", phase=_phases.S2_GRAPH,
        store=store, manager=manager, paths=paths, req=req,
        runner=run_step2_graph,
        set_phase_label="build & validate execution graph",
        memory_path=memory_path,
    )
    store.append_timeline("calc_value.s2.done",
                          {"graph": str(graph_path)})

    # ---------------- S3: calculate FLOPs / mem ---------------- #
    calc_dir = _run_step_with_retry(
        step_name="S3", phase=_phases.S3_CALCULATE,
        store=store, manager=manager, paths=paths, req=req,
        runner=run_step3_calculate,
        set_phase_label="3 agents × 42 combos; converge with median fallback",
        graph_path=graph_path,
    )
    store.append_timeline("calc_value.s3.done",
                          {"calc_dir": str(calc_dir)})

    # ---------------- S4: HTML viz ---------------- #
    viz_path = _run_step_with_retry(
        step_name="S4", phase=_phases.S4_VISUALIZE,
        store=store, manager=manager, paths=paths, req=req,
        runner=run_step4_visualize,
        set_phase_label="generate HTML visualization",
        graph_path=graph_path, calc_dir=calc_dir,
    )
    store.append_timeline("calc_value.s4.done",
                          {"viz": str(viz_path)})

    store.update_run(current_phase=_phases.FINISHED, finished=True,
                     final_status="success",
                     last_transition_label="all steps complete")
    store.append_timeline("calc_value.finished", {"viz": str(viz_path)})
    print(f"[calc-value] pipeline complete. viz = {viz_path}", flush=True)
    return 0
