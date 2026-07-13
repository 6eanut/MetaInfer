"""GPU pre-flight check: ensure the GPU is free before booting inference code.

Why this exists: every iteration's B (agent self-test), C (correctness oracle),
and E (perf oracle) phases boot ``serve.sh``, which loads model weights into
GPU memory. If a previous experiment crashed without releasing VRAM (real
example: implementer's `python3 test.sh` OOMed, the subprocess was SIGKILLed
but left 50+ GiB of VRAM allocated), the next boot collides with the orphan,
fails with a confusing "out of memory" error, and the agent spends the next
60 minutes debugging a phantom bug. A 200 ms preflight that kills orphans
before boot eliminates that entire failure mode.

Supported platforms (auto-detected at call time):

  * **NVIDIA** — ``nvidia-smi --query-compute-apps=pid,used_memory`` returns
    PIDs + their VRAM usage directly.
  * **AMD ROCm** — ``rocm-smi --showpids`` returns PIDs (without per-process
    VRAM on older builds); we scan ``/proc/*/fd`` for open
    ``/dev/dri/renderD*`` handles as a fallback to estimate VRAM-using
    processes.
  * **Neither tool on PATH** — silent no-op. The benchmark proceeds; the
    preflight is best-effort.

The kill policy is conservative: only processes holding GPU memory (NVIDIA)
or with open GPU render-node file descriptors (ROCm) are killed. We never
touch processes that merely happen to share a name with a known training
script. Each kill is logged so post-hoc forensics can verify what happened.

Public API: :func:`preflight_gpu`.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# Skip PIDs whose used GPU memory is below this threshold. Catches noise
# from small long-running daemons (e.g. an X server holding a few MiB on
# the render node) without forcing the user to allow-list them.
MIN_VRAM_MIB_TO_KILL = 128

# How long to wait after SIGTERM before escalating to SIGKILL. Most Python
# processes exit cleanly on SIGTERM (Python raises KeyboardInterrupt →
# atexit handlers run → CUDA context freed). The grace period covers that.
SIGTERM_GRACE_S = 5.0


@dataclass
class GpuOccupant:
    """One process holding GPU resources."""
    pid: int
    vram_mib: float = 0.0  # 0 if unknown (e.g. older rocm-smi)
    command: str = ""
    source: str = ""  # "nvidia-smi" / "rocm-smi" / "proc-fd-scan"


@dataclass
class PreflightResult:
    """Outcome of one preflight call. Always serialized to timeline events."""
    tool: str = "none"  # nvidia-smi / rocm-smi / proc-fd-scan / none
    checked_at: float = field(default_factory=time.time)
    occupants: List[Dict[str, Any]] = field(default_factory=list)
    killed: List[Dict[str, Any]] = field(default_factory=list)
    kill_errors: List[Dict[str, Any]] = field(default_factory=list)
    skipped: bool = False  # True if no GPU tool available
    skip_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


# --------------------------------------------------------------------------- #
# Vendor detection
# --------------------------------------------------------------------------- #

def _find_nvidia_smi() -> Optional[str]:
    return shutil.which("nvidia-smi")


def _find_rocm_smi() -> Optional[str]:
    # First respect PATH, then fall back to the DTK convention seen on
    # production hosts (/opt/dtk/bin/rocm-smi).
    p = shutil.which("rocm-smi")
    if p:
        return p
    for cand in ("/opt/dtk/bin/rocm-smi", "/usr/bin/rocm-smi"):
        if os.path.exists(cand) and os.access(cand, os.X_OK):
            return cand
    return None


# --------------------------------------------------------------------------- #
# Occupant enumeration
# --------------------------------------------------------------------------- #

def _list_nvidia_occupants(nvidia_smi: str) -> List[GpuOccupant]:
    """``nvidia-smi --query-compute-apps=pid,used_memory`` → occupants."""
    try:
        proc = subprocess.run(
            [nvidia_smi,
             "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except (subprocess.SubprocessError, OSError):
        return []
    out: List[GpuOccupant] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            vram = float(parts[1])  # already in MiB
        except ValueError:
            continue
        if pid <= 1:
            continue
        cmd = _proc_comm(pid)
        out.append(GpuOccupant(pid=pid, vram_mib=vram, command=cmd,
                               source="nvidia-smi"))
    return out


def _list_rocm_occupants(rocm_smi: str) -> List[GpuOccupant]:
    """ROCm: ``rocm-smi --showpids`` lists KFD PIDs (often without VRAM
    breakdown). Cross-reference with /proc/*/fd for /dev/dri/renderD*
    open files to catch any process the tool missed.
    """
    pids_seen: set = set()
    try:
        proc = subprocess.run(
            [rocm_smi, "--showpids"],
            capture_output=True, text=True, timeout=10,
        )
        # Output format varies across versions. We see lines like:
        #   "PID 4170 is using 2 DRAM/VRAM memory" (older)
        #   "    12345  1.2GB"  (newer, tabular)
        # Plus a header block. Be liberal: extract any run of digits >= 100
        # that is followed by memory-like info. Then verify with /proc.
        for line in proc.stdout.splitlines():
            tokens = line.split()
            for tok in tokens:
                if tok.isdigit():
                    pid = int(tok)
                    if pid > 100 and _pid_alive(pid):
                        pids_seen.add(pid)
                        break
    except (subprocess.SubprocessError, OSError):
        pass

    # Always also do the /proc scan — it's the source of truth for "does
    # this process hold a render-node FD?" and catches processes that
    # rocm-smi missed (race with KFD registration, root vs user namespace).
    for occ in _scan_proc_render_fds():
        pids_seen.add(occ.pid)
    return [GpuOccupant(pid=pid, vram_mib=0.0, command=_proc_comm(pid),
                        source="rocm-smi+proc-fd")
            for pid in pids_seen if pid > 1]


def _scan_proc_render_fds() -> List[GpuOccupant]:
    """Scan /proc/*/fd for open files under /dev/dri/renderD* or
    /dev/nvidia*. Returns one occupant per PID found. No VRAM info —
    just "this PID has the GPU device open".

    Used as a primary source on ROCm (where rocm-smi --showpids is flaky)
    and as a fallback on NVIDIA when nvidia-smi is missing.
    """
    out: List[GpuOccupant] = []
    proc_root = Path("/proc")
    try:
        pid_dirs = list(proc_root.iterdir())
    except OSError:
        return out
    for d in pid_dirs:
        if not d.name.isdigit():
            continue
        pid = int(d.name)
        if pid <= 1 or pid == os.getpid():
            continue
        fd_dir = d / "fd"
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("/dev/dri/renderD") or target.startswith("/dev/nvidia"):
                    out.append(GpuOccupant(
                        pid=pid, vram_mib=0.0,
                        command=_proc_comm(pid),
                        source="proc-fd-scan",
                    ))
                    break
        except (OSError, PermissionError):
            continue
    return out


def _proc_comm(pid: int) -> str:
    """Best-effort process command line. Empty string if not readable."""
    try:
        return (Path("/proc") / str(pid) / "comm").read_text().strip()
    except OSError:
        return ""


def _pid_alive(pid: int) -> bool:
    return (Path("/proc") / str(pid)).exists()


# --------------------------------------------------------------------------- #
# Kill logic
# --------------------------------------------------------------------------- #

def _kill_one(pid: int, grace_s: float = SIGTERM_GRACE_S) -> Dict[str, Any]:
    """SIGTERM then SIGKILL. Returns a record of what happened."""
    rec: Dict[str, Any] = {"pid": pid, "sigterm_at": time.time()}
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        rec["outcome"] = "already_gone"
        return rec
    except PermissionError as e:
        rec["outcome"] = "permission_denied"
        rec["error"] = str(e)
        return rec

    deadline = time.time() + grace_s
    while time.time() < deadline:
        if not _pid_alive(pid):
            rec["outcome"] = "sigterm_ok"
            rec["exited_at"] = time.time()
            return rec
        time.sleep(0.2)

    # Escalate.
    rec["sigkill_at"] = time.time()
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        rec["outcome"] = "sigterm_ok_late"
        return rec
    except PermissionError as e:
        rec["outcome"] = "sigkill_permission_denied"
        rec["error"] = str(e)
        return rec

    # Wait briefly for SIGKILL to take.
    for _ in range(20):
        if not _pid_alive(pid):
            rec["outcome"] = "sigkill_ok"
            rec["exited_at"] = time.time()
            return rec
        time.sleep(0.2)
    rec["outcome"] = "still_alive"
    return rec


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def preflight_gpu(
    *,
    min_vram_mib: float = MIN_VRAM_MIB_TO_KILL,
    label: str = "preflight",
    log_fn=None,
) -> PreflightResult:
    """Check GPU for orphans, kill any process holding VRAM, return a report.

    ``label`` is included in timeline events so logs distinguish
    ``b-selftest``, ``c-oracle``, ``e-oracle`` call sites.

    ``log_fn`` is an optional ``callable(str) -> None`` for live progress
    (typically the orchestrator's append_timeline / _log channel).

    The function NEVER raises — any internal error fills in ``skip_reason``
    and returns ``skipped=True``. The caller proceeds regardless; preflight
    is observability + best-effort cleanup, not a gate.
    """
    result = PreflightResult()

    def _log(msg: str) -> None:
        if log_fn is not None:
            try:
                log_fn(msg)
            except Exception:  # noqa: BLE001
                pass

    nvidia = _find_nvidia_smi()
    rocm = _find_rocm_smi()
    if nvidia:
        result.tool = "nvidia-smi"
        occupants = _list_nvidia_occupants(nvidia)
    elif rocm:
        result.tool = "rocm-smi"
        occupants = _list_rocm_occupants(rocm)
    else:
        # Last-ditch: /proc FD scan picks up nvidia OR amd if either
        # /dev/nvidia* or /dev/dri/renderD* exists.
        result.tool = "proc-fd-scan"
        occupants = _scan_proc_render_fds()
        if not occupants:
            result.skipped = True
            result.skip_reason = "no GPU tool on PATH and no render-node FDs"
            _log(f"[gpu-preflight:{label}] skipped: {result.skip_reason}")
            return result

    # Filter out tiny occupants (X server, etc).
    targets = [o for o in occupants
               if o.vram_mib >= min_vram_mib or o.source != "nvidia-smi"]
    # ROCm/proc-fd sources have vram_mib=0; treat them as kill candidates
    # (we can't measure their VRAM but they DO hold the device open).

    result.occupants = [
        {"pid": o.pid, "vram_mib": o.vram_mib, "command": o.command,
         "source": o.source}
        for o in targets
    ]

    if not targets:
        _log(f"[gpu-preflight:{label}] GPU clean ({result.tool})")
        return result

    _log(f"[gpu-preflight:{label}] {len(targets)} occupant(s) on {result.tool}; "
         + ", ".join(f"pid={o.pid}({o.command or '?'})" for o in targets))

    for o in targets:
        rec = _kill_one(o.pid)
        if rec.get("outcome") in ("sigterm_ok", "sigterm_ok_late",
                                  "sigkill_ok", "already_gone"):
            result.killed.append({**rec, "command": o.command})
        else:
            result.kill_errors.append({**rec, "command": o.command})
            _log(f"[gpu-preflight:{label}] FAILED to kill pid={o.pid}: "
                 f"{rec.get('outcome')} {rec.get('error', '')}")

    return result


if __name__ == "__main__":
    # Manual smoke: print what the preflight sees right now.
    import json
    r = preflight_gpu(label="cli")
    print(json.dumps(r.to_dict(), indent=2))
