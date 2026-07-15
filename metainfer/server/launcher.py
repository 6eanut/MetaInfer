"""Orchestrator launcher: spawns and supervises per-task subprocesses.

The WebUI server is the long-lived parent. When the user submits a new
task via the web form, it calls :meth:`LocalLauncher.start`, which
``subprocess.Popen``s ``python -m <orchestrator_cli> run ...`` where
``<orchestrator_cli>`` is resolved from the task_type via
:mod:`metainfer.orchestrator.registry`. The launcher itself contains no
task-type-specific logic — adding a new orchestrator is a registry edit.

The :class:`Launcher` :class:`Protocol` is the seam for future
multi-machine support — ``RemoteLauncher`` will implement the same
interface but dispatch spawn/kill over HTTPS to a remote metainfer-web
node. The WebUI never special-cases local vs remote; it goes through
the protocol.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

from . import paths as _paths
from . import proc as _proc
from . import runtime as _runtime
from . import tasks as _tasks


class ProcStatus:
    """Snapshot of an orchestrator subprocess's state, derived from the
    PID file under the task's state_dir."""
    def __init__(
        self,
        *,
        running: bool,
        pid: Optional[int],
        started_at: Optional[float],
        finished_at: Optional[float],
        exit_hint: Optional[str],
    ) -> None:
        self.running = running
        self.pid = pid
        self.started_at = started_at
        self.finished_at = finished_at
        # Human-readable hint about why we think it's running/finished:
        #   "pid-alive" | "pid-file-cleared" | "no-pid-file" | "pid-dead"
        self.exit_hint = exit_hint

    def to_dict(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "pid": self.pid,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "exit_hint": self.exit_hint,
        }


class Launcher(Protocol):
    """Abstract spawn/kill/status interface. LocalLauncher implements it
    via subprocess; RemoteLauncher (future) will implement it via HTTP."""

    def start(self, task_id: str, requirements: Dict[str, Any],
              state_dir: Path, workspace_dir: Path,
              extra_args: Optional[list] = None) -> int: ...
    def status(self, task_id: str) -> ProcStatus: ...
    def kill(self, task_id: str, force: bool = False) -> bool: ...


# --------------------------------------------------------------------------- #
# Local subprocess launcher
# --------------------------------------------------------------------------- #


def _python_executable() -> str:
    """Python to use for the orchestrator subprocess. Defaults to the
    same interpreter running the WebUI; override via
    ``METAINFER_PYTHON`` for venv-style isolation."""
    return os.environ.get("METAINFER_PYTHON", sys.executable)


def _orchestrator_cmd(
    requirements_path: Path, state_dir: Path, workspace_dir: Path,
    extra_args: Optional[list] = None,
    task_type: Optional[str] = None,
) -> list:
    """Build the orchestrator command line.

    Dispatches the orchestrator module by ``task_type`` via the registry
    in :mod:`metainfer.orchestrator.registry`. Unknown task types raise
    ``KeyError`` (caught by the caller) — there is NO silent default
    fallback. An unknown task_type is a bug (typo in requirements.json
    or a missing registry entry), and we'd rather fail at dispatch than
    silently run the wrong pipeline.

    The orchestrator receives BOTH ``--state-dir`` (metadata + logs) and
    ``--workspace-dir`` (generated artifacts) — they're parallel trees
    under the per-node root; see :mod:`metainfer.server.paths`.

    Note: ``args[0]`` is set to ``metainfer-orchestrator`` rather than
    the python binary path. This makes the process trivially findable
    via ``ps aux`` / pgrep. The actual binary executed is still python
    (passed via the ``executable=`` arg to Popen).
    """
    from ..orchestrator.registry import get_orchestrator
    if not task_type:
        raise ValueError(
            "requirements.json is missing 'task_type'; cannot dispatch "
            "orchestrator. Registered types: see metainfer.orchestrator.registry"
        )
    entry = get_orchestrator(task_type)  # raises KeyError for unknown types
    cmd = [
        _proc.ORCHESTRATOR_PROC_NAME,  # friendly argv[0], shows in ps -f / aux
        "-m", entry.cli_module,
        "run", str(requirements_path),
        "--state-dir", str(state_dir),
        "--workspace-dir", str(workspace_dir),
    ]
    if extra_args:
        cmd += list(extra_args)
    return cmd


def _pid_file_path(task_id: str) -> Path:
    return _paths.task_dir(task_id) / "orchestrator.pid"


def _read_pid_file(task_id: str) -> Dict[str, Any]:
    p = _pid_file_path(task_id)
    if not p.exists():
        return {}
    try:
        return json_loads_safe(p.read_text(encoding="utf-8"))
    except OSError:
        return {}


def json_loads_safe(s: str) -> Dict[str, Any]:
    import json
    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else {}
    except ValueError:
        return {}


class LocalLauncher:
    """Spawns orchestrator subprocesses on THIS machine. The default
    launcher; what 99% of users will use.

    Lifecycle:
      - ``start()``: writes requirements.json under state_dir, then
        ``Popen``s the orchestrator. Returns the PID immediately — the
        orchestrator runs in the background.
      - ``status()``: reads the PID file under state_dir + checks liveness.
      - ``kill()``: SIGTERM (escalating to SIGKILL with ``force=True``).
    """

    def __init__(self, boot_id: Optional[str] = None) -> None:
        """``boot_id`` is the WebUI session id (see
        :mod:`metainfer.server.runtime`). Recorded with each spawn so we
        can tell which session owns which orchestrator. May be None —
        e.g. for ad-hoc CLI use of the launcher."""
        self._boot_id = boot_id

    def start(
        self,
        task_id: str,
        requirements: Dict[str, Any],
        state_dir: Path,
        workspace_dir: Path,
        extra_args: Optional[list] = None,
    ) -> int:
        state_dir.mkdir(parents=True, exist_ok=True)
        workspace_dir.mkdir(parents=True, exist_ok=True)
        req_path = state_dir / "requirements.json"
        import json
        req_path.write_text(json.dumps(requirements, indent=2), encoding="utf-8")

        log_path = state_dir / "orchestrator.log"
        log_fp = open(log_path, "ab", buffering=0)

        try:
            cmd = _orchestrator_cmd(req_path, state_dir, workspace_dir, extra_args,
                                    task_type=requirements.get("task_type"))
        except (KeyError, ValueError) as exc:
            # Unknown / missing task_type — surface a clear error rather
            # than spawning a doomed process. The requirements file has
            # already been written (above) so the user can inspect it.
            log_fp.close()
            raise
        env = dict(os.environ)
        # Make sure the orchestrator subprocess can import the metainfer
        # package even when launched from a dev checkout (where the
        # package isn't pip-installed). PYTHONPATH is the parent's
        # sys.path joined with os.pathsep, deduped.
        python_path = os.pathsep.join(
            p for p in sys.path
            if p and p not in env.get("PYTHONPATH", "").split(os.pathsep)
        )
        if python_path:
            existing = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{python_path}{os.pathsep}{existing}".rstrip(os.pathsep)
                if existing else python_path
            )
        # Detach the child from any controlling TTY so Ctrl-C on the
        # WebUI doesn't propagate to running orchestrators.
        env.setdefault("METAINFER_TASK_ID", task_id)
        # ``executable=`` is the actual binary (python); ``args[0]`` is
        # the friendly name that shows up in ps. Without explicit
        # executable=, Popen would try to exec "metainfer-orchestrator"
        # as a program path.
        proc = subprocess.Popen(
            cmd,
            executable=_python_executable(),
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(state_dir),
            env=env,
            start_new_session=True,
        )
        log_fp.close()  # child holds the fd now; parent doesn't need it
        spawn_time = time.time()
        # Cache pid in the registry so the task list view doesn't have to
        # stat every state_dir to render status.
        _tasks.update_task(
            task_id, pid=proc.pid, started_at=spawn_time, finished_at=None,
        )
        # Record in runtime.json so reconciliation on next WebUI start
        # can recognize the process as ours.
        if self._boot_id:
            _runtime.record_task_spawn(
                task_id, proc.pid, state_dir, self._boot_id,
                started_at=spawn_time,
            )
        return proc.pid

    def status(self, task_id: str) -> ProcStatus:
        data = _read_pid_file(task_id)
        pid = data.get("pid")
        finished_at = data.get("finished_at")
        started_at = data.get("started_at")
        if pid is None:
            # Pid file is "cleared" (orchestrator exited cleanly) iff
            # finished_at is set; otherwise the orchestrator hasn't
            # started or its pid file was wiped.
            if finished_at is not None:
                return ProcStatus(
                    running=False, pid=None, started_at=started_at,
                    finished_at=finished_at, exit_hint="pid-file-cleared",
                )
            return ProcStatus(
                running=False, pid=None, started_at=None,
                finished_at=None, exit_hint="no-pid-file",
            )
        # VALIDATED liveness: check that the kernel start time of `pid`
        # matches what we recorded. A bare os.kill(pid, 0) check would
        # pass even if the original process died and the PID was
        # recycled to an unrelated process.
        alive = _proc.validate_pid_started_at(pid, started_at)
        return ProcStatus(
            running=alive, pid=pid, started_at=started_at,
            finished_at=finished_at if not alive else None,
            exit_hint="pid-alive" if alive else "pid-dead",
        )

    def kill(self, task_id: str, force: bool = False) -> bool:
        data = _read_pid_file(task_id)
        pid = data.get("pid")
        started_at = data.get("started_at")
        if not pid:
            return False
        sig = signal.SIGKILL if force else signal.SIGTERM
        # First, reap any sub-agent children. They live in their own
        # process groups (ccb is started with start_new_session=True in
        # subagent_manager.py), so killing the orchestrator's process
        # group won't reach them. Read agents.json for their PIDs.
        self._kill_subagents(task_id, sig, force=force)
        # Then kill the orchestrator itself, with PID-reuse validation.
        ok = _proc.kill_pid_validated(pid, sig=sig, expected_started_at=started_at)
        if not ok:
            # Orchestrator is already dead (crashed / SIGKILLed / OOM-killed)
            # but its pid file + run.json still show "running". The user
            # clicking Kill wants this task to STOP, so clean up the zombie
            # state: stamp finished_at on the pid file and mark the task
            # as not running in the registry. Without this the UI keeps
            # showing the task as alive but the kill button does nothing.
            self._reap_dead_pid_file(task_id, pid, started_at)
        return ok

    def _reap_dead_pid_file(
        self, task_id: str, pid: int, started_at: Optional[float],
    ) -> None:
        """Best-effort cleanup when an orchestrator pid is gone but its
        bookkeeping files still claim it's running.

        - Stamp ``finished_at`` in orchestrator.pid so ``status()`` reports
          not-running.
        - Clear the live pid in the task registry so the list view's
          status pill flips.
        - Append a timeline marker so the audit trail explains why the
          task transitioned to stopped without a clean shutdown.
        """
        import json
        import time
        try:
            sd = _paths.task_dir(task_id)
            pf = sd / "orchestrator.pid"
            if pf.exists():
                d = _read_pid_file(task_id)
                d.setdefault("pid", pid)
                d.setdefault("started_at", started_at)
                d["finished_at"] = time.time()
                d["exit_hint"] = "reaped-by-kill-on-dead-pid"
                pf.write_text(
                    json.dumps(d, indent=2), encoding="utf-8",
                )
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        try:
            _tasks.update_task(
                task_id, pid=None, started_at=started_at,
                finished_at=time.time(),
            )
        except Exception:  # noqa: BLE001
            pass
        try:
            from . import state_reader as _sr
            _sr.append_timeline_event(
                _paths.task_dir(task_id), "kill_reaped_dead_pid",
                {"task_id": task_id, "dead_pid": pid,
                 "started_at": started_at},
            )
        except Exception:  # noqa: BLE001
            pass

    def _kill_subagents(
        self, task_id: str, sig: int, force: bool = False,
    ) -> None:
        """Read agents.json for this task and signal each live sub-agent
        process group. Best-effort; failures are swallowed because the
        orchestrator's own signal handler will normally shut them down
        cleanly on SIGTERM. This is the safety net for SIGKILL / hard
        crashes where the handler doesn't get to run."""
        import json
        sd = _paths.task_dir(task_id)
        snap_path = sd / "agents.json"
        if not snap_path.exists():
            return
        try:
            snap = json.loads(snap_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return
        agents = snap.get("agents") if isinstance(snap, dict) else None
        if not isinstance(agents, list):
            return
        for a in agents:
            if not isinstance(a, dict):
                continue
            pid = a.get("pid")
            started = a.get("started_at")
            if not pid or not isinstance(pid, int):
                continue
            # Only kill agents marked as still running in the snapshot.
            # Finished agents should already be reaped.
            if a.get("status") and a["status"] not in ("running", "starting"):
                continue
            _proc.kill_pid_validated(pid, sig=sig, expected_started_at=started)


# --------------------------------------------------------------------------- #
# Registry-backed default instance
# --------------------------------------------------------------------------- #

_DEFAULT: Optional[LocalLauncher] = None


def get_default_launcher() -> LocalLauncher:
    """Return the process-wide default launcher. Always LocalLauncher
    for now; once RemoteLauncher exists this will dispatch by task.

    The launcher is created lazily on first call. If the WebUI has
    already stamped its boot_id into runtime.json (see
    :func:`metainfer.server.runtime.record_webui_start`), the launcher
    picks that up automatically — spawn records get tagged with it."""
    global _DEFAULT
    if _DEFAULT is None:
        boot_id = None
        try:
            state = _runtime.read_state()
            webui = state.get("webui")
            if webui and webui.get("pid") == os.getpid():
                boot_id = webui.get("boot_id")
        except Exception:  # noqa: BLE001 — best-effort
            pass
        _DEFAULT = LocalLauncher(boot_id=boot_id)
    return _DEFAULT
