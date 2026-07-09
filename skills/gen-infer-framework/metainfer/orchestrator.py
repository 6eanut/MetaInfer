"""Top-level orchestrator wiring: reads requirements, finds repo paths,
spins up the SubAgentManager + pipeline, and (optionally) the WebUI server.
"""

from __future__ import annotations

import atexit
import json
import os
import signal
import threading
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .paths import notebooks_dir as _notebooks_dir
from .paths import skill_root
from .pipeline import Orchestrator, OrchestratorConfig
from .state import StateStore
from .subagent_manager import SubAgentManager


# Back-compat shim so legacy callers keep working. Path resolution lives
# in :mod:`metainfer.paths` (single source of truth for the skill).
def _repo_root() -> Path:
    return skill_root()


_find_repo_root = _repo_root


# --------------------------------------------------------------------------- #
# Per-port orchestrator singleton lock (user-namespace, not per-CWD)
# --------------------------------------------------------------------------- #
#
# The WebUI lives in-process. If the orchestrator exits when ``orch.run()``
# returns, the dashboard dies with it — and the user can't browse results
# after a completed run. So we keep the process alive after completion
# (see ``keepalive`` in :func:`run_with_requirements`).
#
# But that means a leftover orchestrator from a previous run (interrupted,
# or a new task — possibly started from a DIFFERENT working directory)
# would still hold the WebUI port when a new one starts. The contended
# resource is the PORT, not the CWD, so the registry file is keyed by
# port and lives in a per-user runtime directory where any new
# orchestrator can find it regardless of where it was launched from:
#
#   $XDG_RUNTIME_DIR/metainfer/orchestrator-<port>.json
#   fallback: /tmp/metainfer-<uid>/orchestrator-<port>.json
#
# Why runtime/tmp instead of ~ (home):
#   - PID files are ephemeral; they should NOT survive a reboot (PID reuse
#     would let a stale registry entry point at an unrelated process).
#   - XDG_RUNTIME_DIR is per-user-session and is cleaned up on logout;
#     /tmp/metainfer-<uid> is a safe multi-user fallback (root-owned /tmp
#     has the sticky bit, but per-uid subdirs sidestep any ambiguity).
#   - Doesn't clutter the user's home directory with lock files.
#
# Contents: {pid, port, cwd, task_id, started_at}. JSON (not bare PID) so
# we can print useful takeover messages ("taking over from task_id=X
# running in /path/to/other/cwd").


def _runtime_dir() -> Path:
    """Per-user runtime dir for ephemeral metainfer state (PID registries,
    session locks). Prefers $XDG_RUNTIME_DIR; falls back to /tmp/metainfer-<uid>.
    Created with mode 0700 so other users can't read or write our locks."""
    xdg = os.environ.get("XDG_RUNTIME_DIR")
    if xdg:
        base = Path(xdg) / "metainfer"
    else:
        base = Path("/tmp") / f"metainfer-{os.getuid()}"
    base.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(base, 0o700)
    except OSError:
        pass
    return base


# Event the keepalive loop waits on; signal handlers set it so Ctrl-C
# unblocks cleanly.
_KEEPALIVE_STOP = threading.Event()


def _registry_file_path(port: int) -> Path:
    return _runtime_dir() / f"orchestrator-{port}.json"


def _process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _process_cmdline(pid: int) -> str:
    """Best-effort read of /proc/<pid>/cmdline. Returns '' on any failure
    (non-Linux, permission denied, etc.). Used to avoid killing an
    unrelated process that happens to have reused a stale PID.
    """
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")


def _looks_like_orchestrator(pid: int) -> bool:
    """Heuristic: refuse to kill a PID unless its cmdline looks like a
    Python process running the metainfer orchestrator. Prevents PID-reuse
    footguns where a stale PID file points at someone else's process.
    """
    cmd = _process_cmdline(pid)
    if not cmd:
        # Not Linux or unreadable — be lenient and trust the registry.
        return True
    if "python" not in cmd and "metainfer" not in cmd:
        return False
    if "metainfer" in cmd or "orchestrator" in cmd:
        return True
    return False


def _read_registry(port: int) -> Optional[Dict[str, Any]]:
    """Return the registry entry for ``port`` if the recorded PID is still
    alive and looks like a metainfer orchestrator. Returns None otherwise
    (no file, corrupt JSON, dead PID, or cmdline mismatch)."""
    f = _registry_file_path(port)
    if not f.exists():
        return None
    try:
        data = json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
    pid = data.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return None
    if not _process_alive(pid):
        return None
    if not _looks_like_orchestrator(pid):
        return None
    return data


def _kill_prev_orchestrator(port: int, timeout_s: float = 5.0) -> Optional[Dict[str, Any]]:
    """SIGTERM (escalating to SIGKILL) the orchestrator currently holding
    ``port``, if any. Returns the killed registry entry (for logging) or
    None if nothing was running."""
    entry = _read_registry(port)
    if entry is None:
        return None
    pid = entry["pid"]
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        return entry
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not _process_alive(pid):
            return entry
        time.sleep(0.1)
    # Grace period expired — escalate.
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    return entry


def _write_registry(port: int, *, cwd: Path, task_id: str) -> None:
    f = _registry_file_path(port)
    f.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": os.getpid(),
        "port": port,
        "cwd": str(cwd),
        "task_id": task_id,
        "started_at": time.time(),
    }
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(f)


def _clear_registry_if_ours(port: int) -> None:
    f = _registry_file_path(port)
    try:
        if not f.exists():
            return
        data = json.loads(f.read_text(encoding="utf-8"))
        if data.get("pid") == os.getpid():
            f.unlink()
    except (ValueError, OSError):
        pass


def _install_keepalive_signal_handlers() -> None:
    """Translate SIGINT/SIGTERM into 'stop the keepalive loop' instead of
    the default 'kill the process immediately'. The first signal wins;
    a second one (or any during a stuck state) reverts to default behavior.
    """
    def handler(signum, frame):
        _KEEPALIVE_STOP.set()
    for s in (signal.SIGINT, signal.SIGTERM):
        # signal.signal only works on the main thread; orchestrator is
        # the main thread so this is fine.
        try:
            signal.signal(s, handler)
        except (ValueError, OSError):
            pass


def run_with_requirements(
    requirements_path: Path,
    *,
    web_port: int = 8765,
    no_web: bool = False,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "acceptEdits",
    max_iterations: Optional[int] = None,
    extra_claude_args: Optional[list] = None,
    keepalive: bool = True,
    effort: str = "max",
) -> int:
    """Entry point used by ``metainfer run <requirements.json>``."""
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = __import__("json").loads(
        requirements_path.read_text(encoding="utf-8")
    )
    task_id = req.get("task_id", "task")

    # Working directory is the user's CWD.
    #
    # Layout (since the layout refactor):
    #   <cwd>/<task_id>/001/                ← iteration CODE (visible, top-level)
    #   <cwd>/<task_id>/002/                ← next iteration's code (copied forward)
    #   <cwd>/.metainfer/state/<task_id>/   ← run state + iteration records (hidden)
    #   <cwd>/.metainfer/logs/<task_id>/001/← per-iteration agent/oracle logs (hidden)
    #
    # Code lives OUTSIDE .metainfer/ so the user can browse iterations directly;
    # tracking metadata + debug logs stay INSIDE .metainfer/.
    cwd = Path.cwd()
    metainfer_root = cwd / ".metainfer"
    state_dir = metainfer_root / "state" / task_id
    iterations_root = cwd / task_id
    logs_root = metainfer_root / "logs" / task_id
    metainfer_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    iterations_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    # Take over from any prior orchestrator holding this port — regardless
    # of which CWD it was launched from. The registry lives in the user's
    # home directory keyed by port, so a new run from a different working
    # directory can still find and SIGTERM the old one before binding.
    prev = _read_registry(web_port)
    if prev is not None:
        prev_cwd = prev.get("cwd", "?")
        prev_task = prev.get("task_id", "?")
        prev_pid = prev.get("pid", "?")
        print(f"[metainfer] taking over WebUI port {web_port} from prior "
              f"orchestrator (pid {prev_pid}, task_id={prev_task}, "
              f"cwd={prev_cwd})")
        _kill_prev_orchestrator(web_port)
    _write_registry(web_port, cwd=cwd, task_id=task_id)
    atexit.register(_clear_registry_if_ours, web_port)

    # Copy requirements into the state dir if invoked from elsewhere
    target_req = state_dir / "requirements.json"
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(requirements_path.read_text(encoding="utf-8"), encoding="utf-8")

    repo_root = _repo_root()
    notebooks_dir = _notebooks_dir()

    store = StateStore(state_dir)
    # Ensure run state is initialized fresh (orchestrator's run() calls init_run again)
    cfg = OrchestratorConfig(
        workdir=cwd,
        repo_root=repo_root,
        notebooks_dir=notebooks_dir,
        iterations_root=iterations_root,
        logs_root=logs_root,
        state_dir=state_dir,
        max_iterations=max_iterations or _extract_max_iter(req, default=20),
        claude_bin=claude_bin,
        model=model,
        permission_mode=permission_mode,
        extra_claude_args=list(extra_claude_args or []),
    )

    manager = SubAgentManager(
        claude_bin=claude_bin,
        default_model=model,
        permission_mode=permission_mode,
        effort=effort,
        # Sub-agent prompts point at these paths outside the iteration dir:
        #   - notebooks/: read-only knowledge base every prompt tells the
        #     agent to consult
        #   - skill_root: holds SKILL.md and questions.yaml that prompts
        #     also reference
        #   - logs_root: where the reviewer writes review.md and where the
        #     prev-iter diagnostic snapshot lives; without --add-dir the
        #     sandbox blocks both the write and the read
        extra_add_dirs=[notebooks_dir, repo_root, logs_root],
    )
    orch = Orchestrator(req=req, store=store, cfg=cfg, manager=manager)

    # WebUI: start in a daemon thread (or subprocess if you prefer isolation)
    web_thread = None
    if not no_web:
        from .web.server import start_server_in_thread
        web_thread = start_server_in_thread(
            store=store, manager=manager, iterations_root=iterations_root,
            port=web_port,
        )

    print(f"[metainfer] task_id        = {task_id}")
    print(f"[metainfer] code dir       = {iterations_root}")
    print(f"[metainfer] state dir      = {state_dir}")
    print(f"[metainfer] logs dir       = {logs_root}")
    print(f"[metainfer] notebooks      = {notebooks_dir}")
    if web_thread is not None:
        print(f"[metainfer] WebUI          = http://127.0.0.1:{web_port}/")
    print(f"[metainfer] hand-off complete; orchestrator is now driving.")

    orch.run()

    # Keep the WebUI alive after completion so the user can browse the
    # finished run's state, iteration history, perf charts, agent logs,
    # etc. The orchestrator process blocks here until Ctrl-C / SIGTERM.
    # The next `metainfer run` in this CWD will SIGTERM us via the PID
    # file logic at the top of this function and take over.
    #
    # Disable keepalive under --no-web (no point holding the process if
    # there's no dashboard), when the caller explicitly opts out, OR
    # when the orchestrator no-op'd because the task was already finished
    # (holding the process up in that case just looks like a freeze —
    # there's no new state to browse that the user hasn't already seen).
    keep_this_alive = (not no_web) and keepalive and not orch.nooped
    if keep_this_alive:
        print(f"[metainfer] run finished; WebUI stays alive at "
              f"http://127.0.0.1:{web_port}/ for browsing.")
        print(f"[metainfer] press Ctrl-C (or send SIGTERM) to exit and "
              f"release the port.")
        _install_keepalive_signal_handlers()
        try:
            _KEEPALIVE_STOP.wait()
        finally:
            _clear_registry_if_ours(web_port)
        print("[metainfer] exiting; WebUI shut down.")
    else:
        if orch.nooped:
            print(f"[metainfer] task was already finished; exiting without "
                  f"keepalive (no new state to browse).")
        _clear_registry_if_ours(web_port)
    return 0


def _extract_max_iter(req: Dict[str, Any], default: int = 20) -> int:
    """Read max_iterations from requirements, preferring top-level.

    The interview writes ``max_iterations`` as a TOP-LEVEL field on
    requirements.json (alongside ``target_model``, ``target_hardware``,
    etc. — see ``_render_req`` which dumps top-level keys directly). The
    old code looked under ``answers.`` only, so the user's value was
    silently dropped and runs always stopped at the hardcoded default
    (20). Top-level takes precedence; ``answers.`` is checked as a
    back-compat fallback for older requirements files.
    """
    v = req.get("max_iterations")
    if v is None:
        v = req.get("answers", {}).get("max_iterations")
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
