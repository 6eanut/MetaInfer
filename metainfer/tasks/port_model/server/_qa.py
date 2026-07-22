"""QA config for port-model — maps QA session targets to agent event files."""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any, Dict

from metainfer.server._helpers import find_events_file


# Map a phase id to its logs/ subdir under state_dir.
_PHASE_LOG_DIR = {
    "P1_weight_analysis":   "p1",
    "P2_framework_analysis": "p2",
    "P3_architect_review":  "p3",
    "P4_minimal_framework": "p4",
    "P5_verify_minimal":    "p5",
    "P6_port_engine":       "p6",
}


class PortModelQAConfig:
    def resolve_target(
        self,
        state_dir: Path,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        events_file = payload.get("events_file")
        if events_file:
            p = Path(events_file)
            if p.is_file():
                return {
                    "events_file": str(p),
                    "target_workdir": str(p.parent),
                    "target_label": f"events: {p.parent.name}/{p.name}",
                }
            raise FileNotFoundError(f"events_file not found: {events_file}")

        phase = payload.get("phase") or payload.get("step")
        agent = payload.get("agent")
        if phase and agent:
            return self._resolve_by_phase_agent(state_dir, phase, agent)

        raise ValueError(
            "payload must have either 'events_file' or ('phase', 'agent')"
        )

    def _resolve_by_phase_agent(
        self, state_dir: Path, phase: str, agent: str,
    ) -> Dict[str, Any]:
        sub = _PHASE_LOG_DIR.get(phase)
        if sub is None:
            raise ValueError(f"unknown phase {phase!r}")
        # logs may be nested under attempt_XX / iter_XX — search recursively.
        log_root = state_dir / "logs" / sub
        # Try exact match first.
        for cand in log_root.rglob(f"{agent}.attempt*.events.jsonl"):
            return {
                "events_file": str(cand),
                "target_workdir": str(cand.parent),
                "target_label": f"{phase}/{agent}",
            }
        # Fallback: glob for agent*.events.jsonl.
        pattern = str(log_root / "**" / f"{agent}*.events.jsonl")
        matches = glob.glob(pattern, recursive=True)
        if matches:
            m = Path(matches[0])
            return {
                "events_file": str(m),
                "target_workdir": str(m.parent),
                "target_label": f"{phase}/{m.stem}",
            }
        raise FileNotFoundError(
            f"no events file found for phase={phase!r} agent={agent!r} under {log_root}"
        )


QA_CONFIG = PortModelQAConfig()
