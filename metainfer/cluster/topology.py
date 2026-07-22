"""GPU topology detection for worker registration.

Reports per-GPU identity (uuid, name, memory, pci bus) at worker startup so the
scoreboard can address GPU slots by logical index. The index used by
``CUDA_VISIBLE_DEVICES`` matches the index reported here.

Two backends:
- NVIDIA: ``nvidia-smi --query-gpu=index,uuid,name,memory.total,pci.bus_id``
- AMD ROCm: ``rocm-smi --showproductname`` + parse (less structured; best-effort)

If neither tool is on PATH, returns an empty dict — the worker can still run
CPU-only jobs.
"""

from __future__ import annotations

import shutil
import subprocess
from typing import Dict


def detect_gpu_topology() -> Dict[int, Dict[str, object]]:
    """Return ``{gpu_index: {uuid, name, total_memory_mib, pci_id}}``.

    Indexes are integers (0-based) matching ``nvidia-smi``'s reported index.
    On ROCm, where the concept of "index" is less canonical, we use the
    rendering-card index from the ordered list returned by ``rocm-smi``.

    Empty dict means no GPU detected. The caller (worker registration) still
    records the worker — it just won't be selectable for GPU-slot acquires.
    """
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        return _detect_nvidia(nvidia)
    rocm = shutil.which("rocm-smi") or _find_rocm_smi_fallback()
    if rocm:
        return _detect_rocm(rocm)
    return {}


def _find_rocm_smi_fallback() -> str | None:
    """Mirror metainfer.orchestrator.gpu_preflight._find_rocm_smi without importing it.

    We avoid the import to keep ``metainfer.cluster`` free of dependencies on
    ``metainfer.orchestrator`` (orchestrator is a higher layer).
    """
    import os
    for cand in ("/opt/dtk/bin/rocm-smi", "/usr/bin/rocm-smi"):
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def _detect_nvidia(nvidia_smi: str) -> Dict[int, Dict[str, object]]:
    try:
        proc = subprocess.run(
            [nvidia_smi,
             "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return {}

    topo: Dict[int, Dict[str, object]] = {}
    for line in proc.stdout.splitlines():
        parts = [c.strip() for c in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        # memory.total comes back as "24576" (Mib) with nounits
        try:
            mem_mib = int(parts[3])
        except ValueError:
            mem_mib = 0
        topo[idx] = {
            "uuid": parts[1],
            "name": parts[2],
            "total_memory_mib": mem_mib,
            "pci_id": parts[4],
        }
    return topo


def _detect_rocm(rocm_smi: str) -> Dict[int, Dict[str, object]]:
    """Best-effort ROCm topology via rocm-smi.

    Format varies across versions. We try ``--showproductname --json`` first
    (newer builds), fall back to plain text parsing.
    """
    # Try JSON output (modern rocm-smi).
    try:
        proc = subprocess.run(
            [rocm_smi, "--showproductname", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
            import json
            data = json.loads(proc.stdout)
            topo: Dict[int, Dict[str, object]] = {}
            for card_key, card_data in data.items():
                # card_key like "card0"
                num = "".join(c for c in card_key if c.isdigit())
                if not num:
                    continue
                idx = int(num)
                topo[idx] = {
                    "uuid": str(card_data.get("GUID", card_key)),
                    "name": str(card_data.get("Card series", card_data.get("Card model", "unknown"))),
                    "total_memory_mib": _parse_rocm_mem(card_data.get("Memory total (RAM)")),
                    "pci_id": str(card_data.get("PCI Bus", "")),
                }
            return topo
    except (subprocess.SubprocessError, OSError, ValueError):
        pass
    return {}


def _parse_rocm_mem(s: object) -> int:
    """Parse strings like "16384MB" or "16 GiB" → MiB integer."""
    if not isinstance(s, str):
        return 0
    s = s.strip()
    # Strip common unit suffixes; result is interpreted as MiB.
    for suffix in ("MiB", "MB", "GiB", "GB"):
        if s.endswith(suffix):
            try:
                v = float(s[:-len(suffix)].strip())
            except ValueError:
                return 0
            if suffix in ("GiB", "GB"):
                v *= 1024
            return int(v)
    try:
        return int(float(s))
    except ValueError:
        return 0
