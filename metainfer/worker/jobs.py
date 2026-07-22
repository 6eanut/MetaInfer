"""Per-job subprocess execution on the worker side.

Each job runs in its own thread (the daemon's supervisor). The subprocess is
spawned with ``start_new_session=True`` so it has its own process group —
critical for clean SIGTERM/SIGKILL on timeout or cancel.

Streaming logs: stdout/stderr file handles are opened in append-binary mode
and passed to ``Popen`` as ``stdout`` / ``stderr``. The OS handles the streaming
for us; we just close the fds when the child exits.

Timeout & cancel: a watchdog thread polls ``cancel.marker`` and the deadline.
On either trigger: SIGTERM → 5s grace → SIGKILL (mirrors
``metainfer.orchestrator.gpu_preflight._kill_one`` semantics).

The worker does NOT release GPU slots — the orchestrator owns the LeaseToken
and releases in its ``finally``. The worker just reports exit status.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from metainfer.cluster import mqueue, paths
from metainfer.cluster.queue_schema import (
    JobHandle,
    JobResult,
    STATUS_CANCELLED,
    STATUS_DONE,
    STATUS_TIMEOUT,
)


# Grace period between SIGTERM and SIGKILL.
KILL_GRACE_S = 5.0
# Polling interval for cancel.marker / deadline check.
WATCHDOG_POLL_S = 0.5


def _resolve_cuda_visible_devices(gpu_slots: List[Tuple[str, int]], own_node_id: str) -> str:
    """Filter ``gpu_slots`` to those on ``own_node_id``, return as comma-string for CUDA_VISIBLE_DEVICES."""
    indices = sorted(idx for node, idx in gpu_slots if node == own_node_id)
    return ",".join(str(i) for i in indices)


def _build_script_command(script_path: Path) -> List[str]:
    return ["bash", str(script_path)]


def _build_agent_command(prompt_path: Path, workdir: Path) -> List[str]:
    """Build a ccb (claude-code-binary) invocation that reads the prompt from a file.

    Mirrors ``SubAgentManager._build_command`` from
    ``metainfer.orchestrator.subagent_manager`` but trimmed to the worker context:
    no session resume, no extra dirs beyond the workdir.
    """
    ccb = _find_ccb()
    return [
        ccb,
        "-p",
        "--output-format", "stream-json",
        "--input-format", "text",
        "--verbose",
        "--permission-mode", "bypassPermissions",
        "--add-dir", str(workdir),
    ]


def _find_ccb() -> str:
    """Locate the claude-code-binary. Honors METAINFER_CCB / CLAUDE_BIN env overrides."""
    p = os.environ.get("METAINFER_CCB") or os.environ.get("CLAUDE_BIN")
    if p:
        return p
    for name in ("ccb", "claude"):
        path = shutil.which(name)
        if path:
            return path
    return "ccb"


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """SIGTERM the child's process group, wait KILL_GRACE_S, then SIGKILL."""
    if proc.poll() is not None:
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, OSError):
        return
    deadline = time.time() + KILL_GRACE_S
    while time.time() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        pass


def _watch_and_kill(proc: subprocess.Popen, job_dir: Path,
                    deadline: float, stop_event: threading.Event) -> None:
    """Watchdog thread body. Polls cancel.marker + deadline; kills child if either trips."""
    while not stop_event.is_set():
        if proc.poll() is not None:
            return
        if paths.job_cancel_marker(job_dir).exists():
            _terminate_process_group(proc)
            return
        if time.time() >= deadline:
            _terminate_process_group(proc)
            return
        time.sleep(WATCHDOG_POLL_S)


def run_job(handle: JobHandle, own_node_id: str) -> JobResult:
    """Execute one job end-to-end. Returns a :class:`JobResult`.

    Called by the daemon in a supervisor thread. The daemon is responsible for
    writing the result via :func:`metainfer.cluster.mqueue.write_result`.
    """
    spec = handle.spec
    job_dir = Path(handle.job_dir)
    started_at = time.time()
    deadline = started_at + spec.timeout_s

    cwd = spec.cwd or str(job_dir)
    if spec.cwd:
        Path(cwd).mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.update({str(k): str(v) for k, v in spec.env.items()})
    cuda_vis = _resolve_cuda_visible_devices(spec.gpu_slots, own_node_id)
    if cuda_vis:
        env["CUDA_VISIBLE_DEVICES"] = cuda_vis

    stdout_path = job_dir / "stdout.log"
    stderr_path = job_dir / "stderr.log"
    stdout_fd = open(stdout_path, "ab")
    stderr_fd = open(stderr_path, "ab")
    stdin_fd = None
    proc: Optional[subprocess.Popen] = None
    stop_watchdog = threading.Event()

    try:
        if spec.type == "script":
            cmd = _build_script_command(job_dir / "script.sh")
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout_fd, stderr=stderr_fd,
                start_new_session=True, close_fds=True,
            )
        else:
            # agent: ccb reads prompt on stdin
            prompt_path = job_dir / "prompt.txt"
            if not prompt_path.exists():
                prompt_path.write_text(spec.prompt_body)
            cmd = _build_agent_command(prompt_path, Path(cwd))
            stdin_fd = open(prompt_path, "rb")
            proc = subprocess.Popen(
                cmd, cwd=cwd, env=env,
                stdin=stdin_fd,
                stdout=stdout_fd, stderr=stderr_fd,
                start_new_session=True, close_fds=True,
            )

        wd = threading.Thread(target=_watch_and_kill,
                              args=(proc, job_dir, deadline, stop_watchdog),
                              daemon=True)
        wd.start()
        rc = proc.wait()
        stop_watchdog.set()
        wd.join(timeout=2.0)

        duration = time.time() - started_at
        if paths.job_cancel_marker(job_dir).exists():
            return JobResult(job_id=spec.job_id, status=STATUS_CANCELLED,
                             signal=signal.SIGTERM, duration_s=duration)
        if time.time() >= deadline and rc != 0:
            return JobResult(job_id=spec.job_id, status=STATUS_TIMEOUT,
                             signal=signal.SIGKILL, duration_s=duration)
        return JobResult(job_id=spec.job_id, status=STATUS_DONE,
                         exit_code=rc, duration_s=duration)

    except FileNotFoundError as e:
        return JobResult(job_id=spec.job_id, status="failed",
                         error=f"executable not found: {e!s}",
                         duration_s=time.time() - started_at)
    except OSError as e:
        return JobResult(job_id=spec.job_id, status="failed",
                         error=f"os error: {e!s}",
                         duration_s=time.time() - started_at)
    finally:
        stop_watchdog.set()
        for fd in (stdout_fd, stderr_fd, stdin_fd):
            if fd is not None:
                try:
                    fd.close()
                except OSError:
                    pass
