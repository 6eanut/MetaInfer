"""QA config for the calc-theoretical-value task type.

The current calc qa flow is **frontend-driven**: the iterations reader
already exposes ``events_file`` + ``target_workdir`` per agent, so the
calc ``/qa/start`` route just forwards those to the generic
:mod:`metainfer.server.qa`. No server-side path resolution is needed.

This module is here as the home for **future** server-side resolution
if/when we want to support requests of the form
``{step, round, agent}`` instead of explicit ``events_file`` paths
(e.g. for CLI/scripted QA callers). It satisfies the
:class:`metainfer.server.registry.QAConfigLike` protocol so it can be
registered on the plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from metainfer.server._helpers import find_events_file

PLUGIN_TYPE = "calc-theoretical-value"


class CalcQAConfig:
    """Resolve calc-theoretical-value QA targets.

    Two resolution modes:

    1. **Explicit path** (default, used by current WebUI): payload
       contains ``events_file``. We just validate + return it. This is
       a no-op essentially.
    2. **Tuple lookup** (future): payload contains
       ``{step, round, agent}``. We resolve it via the step1/2/3
       directory layout. Currently raises NotImplementedError since
       no caller asks for this — kept as a stub to make the API
       shape explicit.
    """

    def resolve_target(
        self, state_dir: Path, payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        events_file_str = (payload.get("events_file") or "").strip()
        if events_file_str:
            return {
                "events_file": Path(events_file_str),
                "target_workdir": (
                    Path(payload["target_workdir"])
                    if payload.get("target_workdir")
                    else None
                ),
                "target_label": (
                    payload.get("target_label")
                    or f"events_file={Path(events_file_str).name}"
                ),
            }

        # Tuple-lookup branch — not wired up yet.
        step = payload.get("step")
        round_idx = payload.get("round")
        agent = payload.get("agent")
        if step is not None and round_idx is not None and agent is not None:
            ef = _resolve_calc_events_file(
                state_dir, int(step), round_idx, str(agent),
            )
            return {
                "events_file": ef,
                "target_workdir": None,
                "target_label": (
                    f"step={step} round={round_idx} agent={agent}"
                ),
            }

        raise ValueError(
            "payload must contain either events_file, or "
            "(step, round, agent)"
        )


def _resolve_calc_events_file(
    state_dir: Path, step: int, round_label: Any, agent: str,
) -> Path:
    """Locate ``events.jsonl`` for a calc step/round/agent tuple.

    The directory layout is::

        step1/round_NN/logs/<agent>/<agent>.attempt0.events.jsonl
        step2/rounds/<round_label>/logs/<agent>/<agent>.attempt0.events.jsonl
        step3/rounds/<node>/round_NN/logs/<agent>/<agent>.attempt0.events.jsonl
    """
    if step == 1:
        log_dir = state_dir / "step1" / f"round_{int(round_label):02d}" / "logs" / agent
    elif step == 2:
        log_dir = state_dir / "step2" / "rounds" / str(round_label) / "logs" / agent
    elif step == 3:
        # For step 3 the agent name is ``<node>_writer_<i>``; we don't
        # know the node from the tuple alone, so this branch is left
        # for callers that pass the node_id explicitly via payload.
        raise NotImplementedError(
            "step 3 QA tuple-lookup requires node_id; pass events_file "
            "explicitly instead"
        )
    else:
        raise ValueError(f"unknown step: {step}")

    ef: Optional[Path] = find_events_file(log_dir)
    if ef is None:
        raise FileNotFoundError(
            f"no events.jsonl under {log_dir} for agent {agent!r}"
        )
    return ef


# Singleton instance used by the plugin.
CONFIG = CalcQAConfig()
