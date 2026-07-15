"""Process-table utilities for the WebUI.

Reads from ``/proc`` (Linux). Used for two things:

1. **PID validation**: ``orchestrator.pid`` may contain a PID that
   belongs to a recycled, unrelated process. We compare the process's
   kernel start time (field 22 of ``/proc/<pid>/stat``, in clock ticks
   since boot) to the start time we recorded when the orchestrator was
   spawned. Mismatch → the PID is stale.

2. **Startup scan**: enumerate every ``metainfer-orchestrator`` process
   on the host so the freshly-started WebUI can adopt orphaned
   orchestrators from a previous session instead of losing track of
   them.

The friendly process name (``metainfer-orchestrator``) is set by the
launcher when spawning (``args[0]``) AND by the orchestrator itself via
``prctl(PR_SET_NAME)``. We look for either / both to maximize the
chance of catching every process.
"""

from __future__ import annotations

import os
import re
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Clock tick resolution. We need this to convert the start time in
# /proc/<pid>/stat (which is in ticks since boot) into a wall-clock
# time we can compare against `started_at` recorded in our pid files.
try:
    import ctypes
    _libc = ctypes.CDLL("libc.so.6", use_errno=True)
    _sysconf_clk_tck = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
except (OSError, ValueError, AttributeError):  # pragma: no cover - non-Linux
    _libc = None
    _sysconf_clk_tck = 100  # Linux default; reasonable fallback

_BOOT_MONOTONIC_S: Optional[float] = None  # cached boot time in seconds since epoch


def _boot_time_epoch() -> float:
    """Wall-clock boot time of this machine, in seconds since the epoch.
    Cached. Used to convert /proc/<pid>/stat start_ticks (since boot)
    into a wall-clock timestamp comparable to our pid file's started_at."""
    global _BOOT_MONOTONIC_S
    if _BOOT_MONOTONIC_S is not None:
        return _BOOT_MONOTONIC_S
    # /proc/stat has a "bptime <seconds>" line; use that — avoids any
    # dependency on uptime / ps.
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("btime "):
                    _BOOT_MONOTONIC_S = float(line.split()[1])
                    return _BOOT_MONOTONIC_S
    except OSError:
        pass
    # Fallback: best-effort. If we can't read btime, derive boot time
    # from /proc/uptime (less precise, fine for our purposes).
    try:
        with open("/proc/uptime", "r", encoding="utf-8") as f:
            uptime_s = float(f.read().split()[0])
        _BOOT_MONOTONIC_S = time.time() - uptime_s
        return _BOOT_MONOTONIC_S
    except OSError:
        _BOOT_MONOTONIC_S = time.time()
        return _BOOT_MONOTONIC_S


def pid_start_time(pid: int) -> Optional[float]:
    """Wall-clock start time (seconds since epoch) of ``pid``, or None
    if the process doesn't exist or /proc isn't available.

    Field 22 of /proc/<pid>/stat is the process start time in clock
    ticks since boot. We convert ticks→seconds and add the boot time.

    Field 22 because the comm field (2) can contain spaces and parens,
    so we split on the *last* ')' rather than naively splitting on
    whitespace."""
    if pid <= 0:
        return None
    stat_path = f"/proc/{pid}/stat"
    try:
        raw = Path(stat_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # Split off the comm field first (everything between first '(' and last ')').
    rparen = raw.rfind(")")
    if rparen == -1:
        return None
    after_comm = raw[rparen + 2:].split()
    # after_comm[0] is state, [1] is ppid, ..., [19] is starttime (field 22
    # in 1-indexed /proc/<pid>/stat; we've consumed fields 1 and 2 via the
    # pre-rparen part, so after_comm is 0-indexed from field 3 → starttime
    # is at index 19).
    if len(after_comm) < 20:
        return None
    try:
        start_ticks = int(after_comm[19])
    except ValueError:
        return None
    return _boot_time_epoch() + (start_ticks / _sysconf_clk_tck)


def pid_alive(pid: int) -> bool:
    """Cheap liveness check via signal 0."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def validate_pid_started_at(
    pid: int, expected_started_at: Optional[float], tolerance: float = 2.0,
) -> bool:
    """Return True iff ``pid`` is alive AND its process start time is
    within ``tolerance`` seconds of ``expected_started_at``.

    The tolerance absorbs sub-second skew between when we write the pid
    file and when the kernel actually forks the process. 2s is generous.

    If ``expected_started_at`` is None, this reduces to ``pid_alive`` —
    use it when no start time is available."""
    if not pid_alive(pid):
        return False
    if expected_started_at is None:
        return True
    actual = pid_start_time(pid)
    if actual is None:
        return False
    return abs(actual - expected_started_at) <= tolerance


# --------------------------------------------------------------------------- #
# Orchestrator process scan
# --------------------------------------------------------------------------- #

# Match the friendly process name set by the launcher (argv[0]) and by
# the orchestrator itself (kernel comm via prctl).
ORCHESTRATOR_PROC_NAME = "metainfer-orchestrator"
ORCHESTRATOR_COMM = "metainfer-orch"  # kernel comm is truncated to 15 chars


def _read_cmdline(pid: int) -> List[str]:
    """Read /proc/<pid>/cmdline as a list of NUL-separated args."""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    if not raw:
        return []
    # Trailing NUL is optional; split handles either.
    return [a.decode("utf-8", errors="replace") for a in raw.split(b"\x00") if a]


def _parse_state_dir(cmdline: List[str]) -> Optional[str]:
    """Extract the --state-dir argument from an orchestrator cmdline."""
    for i, a in enumerate(cmdline):
        if a == "--state-dir" and i + 1 < len(cmdline):
            return cmdline[i + 1]
        if a.startswith("--state-dir="):
            return a.split("=", 1)[1]
    return None


def _parse_task_id_from_state_dir(state_dir: Optional[str]) -> Optional[str]:
    """Task id is the basename of the state_dir."""
    if not state_dir:
        return None
    return Path(state_dir).name


def is_orchestrator_process(cmdline: List[str], comm: Optional[str] = None) -> bool:
    """Heuristic: is this one of our orchestrator processes?

    Matches if ANY of:
      - argv[0] is ``metainfer-orchestrator`` (set by the launcher),
      - cmdline contains ``-m <any registered orchestrator cli module>``
        (dev-mode detection, driven by the registry so new task types
        are picked up automatically),
      - kernel comm is ``metainfer-orch`` (set by orchestrator via
        prctl — catches processes whose argv[0] was mangled by a
        wrapper shell or whose cmdline is unreadable).

    The third check is what makes detection robust: even if argv[0]
    gets changed after exec (rare but possible), the kernel comm
    persists and is unique to our orchestrator."""
    if comm and comm == ORCHESTRATOR_COMM:
        return True
    if not cmdline:
        return False
    argv0 = cmdline[0]
    if argv0 == ORCHESTRATOR_PROC_NAME:
        return True
    # Dev-mode detection: not running under the friendly argv[0], but
    # still running one of our registered orchestrator CLI modules.
    # Lazy import to avoid pulling the orchestrator package at module
    # load time (registry imports are cheap but the package __init__
    # chain isn't).
    try:
        from ..orchestrator.registry import all_cli_modules
        registered = set(all_cli_modules())
    except Exception:  # noqa: BLE001 — best-effort detection
        registered = set()
    for i, a in enumerate(cmdline):
        if a == "-m" and i + 1 < len(cmdline) \
                and cmdline[i + 1] in registered:
            return True
    return False


def list_orchestrator_processes() -> List[Dict[str, Any]]:
    """Enumerate every orchestrator process on this host.

    Returns a list of dicts, one per process::

        {
          "pid": 12346,
          "started_at": 1720000010.5,
          "state_dir": "/home/.../.metainfer/tasks/foo-abc123",
          "task_id": "foo-abc123",
          "cmdline": [...],
          "comm": "metainfer-orch",
        }

    ``started_at`` is the actual kernel start time of the process (not
    when we recorded spawning it). Two processes for the same task_id
    shouldn't happen in normal operation — but if it does (e.g.
    double-spawn race), both will be returned and reconciliation has to
    pick one.
    """
    out: List[Dict[str, Any]] = []
    try:
        pids = [int(e.name) for e in os.scandir("/proc") if e.name.isdigit()]
    except OSError:
        return out
    for pid in pids:
        cmdline = _read_cmdline(pid)
        comm = None
        try:
            comm = Path(f"/proc/{pid}/comm").read_text(
                encoding="utf-8", errors="replace"
            ).strip()
        except OSError:
            pass
        if not is_orchestrator_process(cmdline, comm):
            continue
        state_dir = _parse_state_dir(cmdline) or _state_dir_from_pidfile_fallback(pid)
        out.append({
            "pid": pid,
            "started_at": pid_start_time(pid),
            "state_dir": state_dir,
            "task_id": _parse_task_id_from_state_dir(state_dir),
            "cmdline": cmdline,
            "comm": comm,
        })
    return out


def _state_dir_from_pidfile_fallback(pid: int) -> Optional[str]:
    """Last-resort: if a process matches by comm but has no cmdline
    (rare — happens when the process is being exec'd), try to read its
    cwd to find the state dir. Not reliable; mostly defensive."""
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Signal delivery (validated)
# --------------------------------------------------------------------------- #

def kill_pid_validated(
    pid: int,
    sig: int = signal.SIGTERM,
    expected_started_at: Optional[float] = None,
) -> bool:
    """Send ``sig`` to ``pid``'s whole process group, but only if the
    process's actual start time matches ``expected_started_at``. This
    prevents killing an unrelated process that happens to have recycled
    a PID we used to own.

    Returns True iff a signal was actually delivered.
    """
    if not validate_pid_started_at(pid, expected_started_at):
        return False
    try:
        pgid = os.getpgid(pid)
    except OSError:
        # Process died between validation and getpgid — nothing to kill.
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except OSError:
        return False
