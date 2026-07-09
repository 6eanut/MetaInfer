"""SubAgentManager: deterministic lifecycle management for Claude Code subprocesses.

Each "sub-agent" is a `claude -p` invocation run as a child process. The
manager:

* spawns the process with a prompt file piped via stdin
* streams `--output-format stream-json` events to a per-agent log file
* records lifecycle events (start / events / done / killed) into a JSON
  sidecar file so the WebUI can render progress without scraping logs
* detects deadlocks (no new output for ``stuck_timeout_s``)
* kills stuck / timed-out processes
* retries failures up to ``max_retries``
* exposes :meth:`snapshot` for the WebUI to poll

This module is deliberately free of any LLM-driven control flow. The
orchestrator decides *what* to run; the manager owns *how* to keep the
process alive and observable.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Exit codes / signals that indicate infrastructure rather than logic failure.
# 124 = timeout (coreutils convention), 137 = SIGKILL (128+9), 143 = SIGTERM (128+15).
_INFRA_EXIT_CODES = {124, 137, 143}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #


@dataclass
class AgentSpec:
    """Declarative description of one sub-agent invocation."""
    name: str
    role: str
    prompt_file: Path
    workdir: Path
    log_dir: Path
    timeout_s: int = 1800
    stuck_timeout_s: int = 600
    max_retries: int = 2
    extra_args: List[str] = field(default_factory=list)
    env_overrides: Dict[str, str] = field(default_factory=dict)
    # Session continuation. When ``resume_session_id`` is set the agent is
    # launched with ``ccb --resume <id>`` so it inherits the prior turn's
    # full conversation (loaded files, prior tool results, in-flight
    # diagnoses). ``cache_read_input_tokens`` typically takes ~95% of the
    # context, so subsequent turns are ~10x cheaper than re-seeding from
    # scratch. ``session_id`` (when set on the FIRST turn only) pins the
    # session UUID so the caller can later resume by the same id. If
    # neither is set, ccb mints a fresh session and exposes its id via
    # ``AgentResult.session_id`` so the caller can resume from it.
    session_id: Optional[str] = None
    resume_session_id: Optional[str] = None

    def log_file(self, attempt: int) -> Path:
        return self.log_dir / f"{self.name}.attempt{attempt}.log"

    def events_file(self, attempt: int) -> Path:
        return self.log_dir / f"{self.name}.attempt{attempt}.events.jsonl"

    def status_file(self) -> Path:
        return self.log_dir / f"{self.name}.status.json"


@dataclass
class AgentHandle:
    """Runtime handle for a launched (or being-retried) agent."""

    spec: AgentSpec
    attempt: int
    process: Optional[subprocess.Popen] = None
    started_at: float = 0.0
    last_output_at: float = 0.0
    killed: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def elapsed(self) -> float:
        if not self.started_at:
            return 0.0
        return time.time() - self.started_at


@dataclass
class AgentResult:
    name: str
    role: str
    success: bool
    returncode: int
    duration_s: float
    events: List[Dict[str, Any]] = field(default_factory=list)
    final_text: str = ""
    error: Optional[str] = None
    attempts: int = 0
    # ccb session UUID this agent ran under. Set on every successful launch
    # (extracted from the first ``system`` event in the stream). The
    # caller can pass this back via ``AgentSpec.resume_session_id`` to
    # continue the same conversation in a later invocation — re-reading
    # files / re-doing analysis then hits the cache instead of paying
    # full input-token cost again.
    session_id: Optional[str] = None
    # Why the agent failed, if it did:
    #   "infra"  — killed (timeout / stuck / signal) → orchestrator retries
    #              the same phase in place without consuming an iteration
    #   "logic"  — nonzero exit with a "real" error → orchestrator's
    #              transition table decides what to do
    failure_mode: Optional[Literal["infra", "logic"]] = None


# --------------------------------------------------------------------------- #
# Manager
# --------------------------------------------------------------------------- #


class SubAgentManager:
    """Owns the lifecycle of every Claude Code subprocess in a task.

    The manager is single-threaded *for control* (launch / wait / kill calls
    are serialized by an internal lock) but each agent's stdout is drained on
    its own background thread so a chatty agent never blocks the orchestrator.
    """

    def __init__(
        self,
        claude_bin: str = "ccb",
        default_model: Optional[str] = None,
        max_concurrent: int = 4,
        permission_mode: str = "acceptEdits",
        extra_add_dirs: Optional[List[Path]] = None,
        effort: str = "max",
    ) -> None:
        self.claude_bin = claude_bin
        self.default_model = default_model
        self.max_concurrent = max_concurrent
        # Claude Code "effort" level — controls how much extended thinking
        # the model is allowed to do per turn. Choices: low / medium / high
        # / max. Iteration logs show reviewers writing only ~550 tokens of
        # text after 13k tokens of thinking, which is fine in principle but
        # was being silently throttled by the default effort setting, so
        # the thinking got cut off mid-analysis. Default "max" lets the
        # model finish its reasoning. Override per-invocation via the
        # METAINFER_EFFORT env var or the CLI --effort flag.
        self.effort = effort
        # Directories every sub-agent is allowed to read from, in addition to
        # the per-invocation workdir. Prompts tell sub-agents to consult the
        # knowledge base under notebooks/ and the SKILL.md at the skill root;
        # without --add-dir the Claude Code sandbox blocks those reads and the
        # agent loops forever against the sandbox. Resolve to absolute, real
        # paths so the flag stays valid even when invoked via a symlink.
        self.extra_add_dirs: List[Path] = [
            Path(p).resolve() for p in (extra_add_dirs or []) if p
        ]
        # Claude Code permission mode for sub-agents. Sub-agents run non-
        # interactively (`-p` with stdin), so `default` mode is unusable:
        # every Edit/Write hangs on a permission prompt nobody can answer.
        # `acceptEdits` = auto-accept file edits (still prompts for shell).
        # `bypassPermissions` = skip ALL prompts (fully autonomous; also
        # unlocks shell commands inside the iteration dir).
        self.permission_mode = permission_mode
        self._handles: Dict[str, AgentHandle] = {}
        self._results: Dict[str, AgentResult] = {}
        self._ctrl_lock = threading.Lock()
        self._semaphore = threading.Semaphore(max_concurrent)
        self._stop = threading.Event()

    # ------------------------------------------------------------------ #
    # Launch / wait
    # ------------------------------------------------------------------ #

    def launch(self, spec: AgentSpec) -> AgentHandle:
        """Launch ``spec`` (with retries handled internally). Blocking call.

        Returns once the agent has either succeeded or exhausted its retries.
        The returned handle carries the final process; see :meth:`result`.
        """
        spec.log_dir.mkdir(parents=True, exist_ok=True)
        spec.prompt_file.parent.mkdir(parents=True, exist_ok=True)

        attempt = 0
        last_result: Optional[AgentResult] = None
        while attempt <= spec.max_retries:
            attempt += 1
            handle = self._run_once(spec, attempt)
            result = self._materialize_result(handle, spec, attempt)
            last_result = result
            self._write_status(spec, handle, result, attempt, phase="completed")
            if result.success:
                with self._ctrl_lock:
                    self._results[spec.name] = result
                return handle
            # failed -> retry
            self._write_status(spec, handle, result, attempt, phase="retrying")
            time.sleep(2.0)  # brief backoff

        # exhausted retries
        assert last_result is not None
        with self._ctrl_lock:
            self._results[spec.name] = last_result
        # Return a synthetic handle so caller has something to inspect
        return AgentHandle(spec=spec, attempt=attempt - 1)

    def launch_async(self, spec: AgentSpec, on_done: Optional[callable] = None) -> threading.Thread:
        """Run :meth:`launch` on a background thread."""
        t = threading.Thread(
            target=self._async_wrapper, args=(spec, on_done), name=f"agent-{spec.name}", daemon=True
        )
        t.start()
        return t

    def _async_wrapper(self, spec: AgentSpec, on_done: Optional[callable]) -> None:
        try:
            self.launch(spec)
        except Exception as exc:  # noqa: BLE001
            self._results[spec.name] = AgentResult(
                name=spec.name, role=spec.role, success=False, returncode=-1,
                duration_s=0.0, error=f"manager exception: {exc!r}",
            )
        finally:
            self._semaphore.release()
            if on_done:
                try:
                    on_done(spec.name)
                except Exception:  # noqa: BLE001
                    pass

    def _run_once(self, spec: AgentSpec, attempt: int) -> AgentHandle:
        with self._semaphore:
            handle = AgentHandle(spec=spec, attempt=attempt)
            with self._ctrl_lock:
                self._handles[spec.name] = handle
            self._write_status(spec, handle, None, attempt, phase="starting")

            cmd = self._build_command(spec)
            env = self._build_env(spec)
            log_fp = open(spec.log_file(attempt), "wb")
            events_fp = open(spec.events_file(attempt), "w", encoding="utf-8")

            try:
                handle.started_at = time.time()
                handle.last_output_at = handle.started_at
                with open(spec.prompt_file, "rb") as prompt_fp:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=prompt_fp,
                        stdout=subprocess.PIPE,
                        stderr=log_fp,
                        cwd=str(spec.workdir),
                        env=env,
                        text=False,
                        start_new_session=True,
                    )
                handle.process = proc
                self._write_status(spec, handle, None, attempt, phase="running")

                # drain stdout on a thread (stream-json is line-delimited JSON)
                stop_evt = threading.Event()

                def drain() -> None:
                    assert proc.stdout is not None
                    for raw in proc.stdout:
                        line = raw.decode("utf-8", errors="replace")
                        events_fp.write(line)
                        events_fp.flush()
                        handle.last_output_at = time.time()
                        log_fp.write(raw)
                        log_fp.flush()
                    stop_evt.set()

                reader = threading.Thread(target=drain, name=f"drain-{spec.name}", daemon=True)
                reader.start()

                # watchdog: timeout / stuck detection
                while proc.poll() is None:
                    if self._stop.is_set():
                        self._terminate(handle)
                        break
                    now = time.time()
                    if now - handle.started_at > spec.timeout_s:
                        self._terminate(handle, reason="timeout")
                        break
                    if now - handle.last_output_at > spec.stuck_timeout_s:
                        self._terminate(handle, reason="stuck")
                        break
                    time.sleep(2.0)

                proc.wait(timeout=30)
                reader.join(timeout=10)
            finally:
                log_fp.close()
                events_fp.close()
            return handle

    # ------------------------------------------------------------------ #
    # Process control
    # ------------------------------------------------------------------ #

    def _terminate(self, handle: AgentHandle, reason: str = "manual") -> None:
        with handle.lock:
            if handle.killed or handle.process is None:
                return
            handle.killed = True
            proc = handle.process
        try:
            # try graceful on the process group first
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                return
            except Exception:  # noqa: BLE001
                proc.terminate()
            # hard kill after grace period
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except Exception:  # noqa: BLE001
                    proc.kill()
        finally:
            # write a marker into the events file for forensics
            try:
                ef = handle.spec.events_file(handle.attempt)
                with open(ef, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "__meta__", "killed": reason}) + "\n")
            except Exception:  # noqa: BLE001
                pass

    def kill(self, name: str) -> bool:
        with self._ctrl_lock:
            handle = self._handles.get(name)
        if handle is None:
            return False
        self._terminate(handle, reason="kill-requested")
        return True

    def kill_all(self) -> None:
        with self._ctrl_lock:
            handles = list(self._handles.values())
        for h in handles:
            self._terminate(h, reason="shutdown")

    def shutdown(self) -> None:
        self._stop.set()
        self.kill_all()

    # ------------------------------------------------------------------ #
    # Health / snapshot
    # ------------------------------------------------------------------ #

    def health(self, name: str) -> Literal["running", "stuck", "done", "failed", "unknown"]:
        with self._ctrl_lock:
            handle = self._handles.get(name)
            result = self._results.get(name)
        if result is not None:
            return "done" if result.success else "failed"
        if handle is None or handle.process is None:
            return "unknown"
        if handle.process.poll() is not None:
            return "done" if handle.process.returncode == 0 and not handle.killed else "failed"
        if time.time() - handle.last_output_at > handle.spec.stuck_timeout_s:
            return "stuck"
        return "running"

    def snapshot(self) -> List[Dict[str, Any]]:
        """Return a list of agent status dicts for the WebUI."""
        out: List[Dict[str, Any]] = []
        with self._ctrl_lock:
            names = list(self._handles.keys())
        for name in names:
            with self._ctrl_lock:
                handle = self._handles.get(name)
                result = self._results.get(name)
            if handle is None:
                continue
            out.append(
                {
                    "name": name,
                    "role": handle.spec.role,
                    "attempt": handle.attempt,
                    "phase": self.health(name),
                    "elapsed_s": round(handle.elapsed, 1),
                    "last_output_age_s": round(time.time() - handle.last_output_at, 1)
                    if handle.last_output_at
                    else None,
                    "killed": handle.killed,
                    "success": result.success if result else None,
                    "error": result.error if result else None,
                    "log_file": str(handle.spec.log_file(handle.attempt)),
                }
            )
        return out

    def result(self, name: str) -> Optional[AgentResult]:
        with self._ctrl_lock:
            return self._results.get(name)

    def results(self) -> Dict[str, AgentResult]:
        with self._ctrl_lock:
            return dict(self._results)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_command(self, spec: AgentSpec) -> List[str]:
        cmd = [
            self.claude_bin,
            "-p",  # print (non-interactive) mode; prompt read from stdin
            "--output-format", "stream-json",
            "--input-format", "text",
            "--verbose",
            # Sub-agents can't answer permission prompts (non-interactive).
            # Default to a mode that lets them write code without hanging.
            # The iteration dir is the agent's CWD, so file ops stay scoped
            # to that folder.
            "--permission-mode", self.permission_mode,
            # Explicitly add the iteration dir as an allowed working dir.
            # Belt-and-suspenders: cwd is already set to spec.workdir below,
            # but --add-dir guarantees the agent treats it as writable.
            "--add-dir", str(spec.workdir),
        ]
        # Read-only knowledge sources (notebooks/, skill root with SKILL.md).
        # The manager-level list applies to every sub-agent so individual
        # phase code doesn't have to remember to opt in.
        for d in self.extra_add_dirs:
            cmd += ["--add-dir", str(d)]
        if self.default_model:
            cmd += ["--model", self.default_model]
        # Effort level controls extended-thinking budget. "max" lets the
        # model finish long reasoning chains instead of getting cut off
        # mid-analysis (which iteration logs showed happening to reviewers).
        if self.effort:
            cmd += ["--effort", self.effort]
        # Session continuation. ``--resume`` takes precedence (explicitly
        # continuing an existing conversation); otherwise ``--session-id``
        # pins the UUID for the first turn so the caller knows what to
        # resume later. If neither is set, ccb mints a fresh session and
        # the manager captures its id from the stream.
        if spec.resume_session_id:
            cmd += ["--resume", spec.resume_session_id]
        elif spec.session_id:
            cmd += ["--session-id", spec.session_id]
        cmd += list(spec.extra_args)
        return cmd

    def _build_env(self, spec: AgentSpec) -> Dict[str, str]:
        env = dict(os.environ)
        env.update(spec.env_overrides)
        # Keep the agent from going interactive
        env.setdefault("DISABLE_INTERACTIVITY", "1")
        return env

    def _materialize_result(
        self, handle: AgentHandle, spec: AgentSpec, attempt: int
    ) -> AgentResult:
        proc = handle.process
        rc = proc.returncode if proc is not None else -1
        events: List[Dict[str, Any]] = []
        ef = spec.events_file(attempt)
        if ef.exists():
            for ln in ef.read_text(encoding="utf-8", errors="replace").splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    events.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        final_text = ""
        for ev in reversed(events):
            if ev.get("type") == "assistant" and isinstance(ev.get("message"), dict):
                content = ev["message"].get("content")
                if isinstance(content, list):
                    for blk in content:
                        if isinstance(blk, dict) and blk.get("type") == "text":
                            final_text = blk.get("text", "")
                            break
                if final_text:
                    break
            if ev.get("type") == "result":
                final_text = ev.get("result", "") or final_text
                break
        # Session id: emitted on the very first ``system`` event of the
        # stream and again on every ``result`` event. Prefer the result's
        # value (it's the final, post-turn session id; for ``--resume``
        # invocations this matches the resumed-from id and confirms the
        # continuation actually happened).
        session_id = None
        for ev in events:
            sid = ev.get("session_id")
            if sid:
                session_id = sid
                if ev.get("type") == "result":
                    break
        error = None
        failure_mode: Optional[Literal["infra", "logic"]] = None
        success = (rc == 0) and not handle.killed
        if handle.killed:
            error = "killed (see events log tail)"
            failure_mode = "infra"
        elif rc != 0:
            error = f"nonzero exit {rc}"
            # negative rc = killed by signal; specific positive codes = timeout/etc.
            if rc < 0 or rc in _INFRA_EXIT_CODES:
                failure_mode = "infra"
            else:
                failure_mode = "logic"
        return AgentResult(
            name=spec.name,
            role=spec.role,
            success=success,
            returncode=rc,
            duration_s=handle.elapsed,
            events=events,
            final_text=final_text,
            error=error,
            attempts=attempt,
            session_id=session_id,
            failure_mode=failure_mode,
        )

    def _write_status(
        self,
        spec: AgentSpec,
        handle: AgentHandle,
        result: Optional[AgentResult],
        attempt: int,
        phase: str,
    ) -> None:
        status = {
            "name": spec.name,
            "role": spec.role,
            "attempt": attempt,
            "phase": phase,
            "elapsed_s": round(handle.elapsed, 1),
            "success": result.success if result else None,
            "error": result.error if result else None,
            "updated_at": time.time(),
            "log_file": str(spec.log_file(attempt)),
            "events_file": str(spec.events_file(attempt)),
        }
        tmp = spec.status_file().with_suffix(".tmp")
        tmp.write_text(json.dumps(status, indent=2), encoding="utf-8")
        tmp.replace(spec.status_file())
