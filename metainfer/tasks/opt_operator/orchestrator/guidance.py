"""Human guidance store — advice injected between optimization cycles.

D_review produces strong-model guidance; a human can also drop guidance into the
run's guidance file. ``latest`` merges both (human first) so the next A_plan sees
it. Guidance is advisory, never a gate.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional


class GuidanceStore:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def add(self, text: str, *, source: str = "agent", cycle: Optional[int] = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        entries = self._load()
        entries.append({"ts": time.time(), "source": source, "cycle": cycle,
                        "text": text})
        from metainfer.cluster.fs_primitives import atomic_write_text
        atomic_write_text(self.path, json.dumps(entries, indent=2, sort_keys=True))

    def _load(self) -> List[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return data if isinstance(data, list) else []

    def all(self) -> List[dict]:
        return self._load()

    def latest(self) -> Optional[str]:
        entries = self._load()
        if not entries:
            return None
        return entries[-1]["text"]


__all__ = ["GuidanceStore"]
