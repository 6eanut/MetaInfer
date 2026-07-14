"""Offline QA over an agent's conversation history.

When a calc-value pipeline agent finishes, its full stream-json
conversation is on disk at ``<log_dir>/<name>.attempt<N>.events.jsonl``.
This module lets the user (via the WebUI) ask follow-up questions about
what that agent did: a fresh ``ccb`` subprocess is spawned with read
access to the transcript, and answers the user's question by inspecting
the events log with Read/Grep.

Lifecycle::

    start_qa_session(state_dir, payload) -> session_id
        # writes request.json + prompt.txt + status.json (status=running)
        # spawns threading.Thread(target=_run_analyst)
        # updates index.json
        # returns session_id immediately

    _run_analyst(session_dir, ...)
        # Popen ccb, stream stdout to analyst_events.jsonl
        # on exit: extract final_text, write answer.txt + status.json
        # update index.json

    get_qa_session(state_dir, sid) -> dict
    list_qa_sessions(state_dir, ...) -> list[dict]

The WebUI process owns these background threads. They are daemon, so
they die with the process; on WebUI restart, in-flight sessions are
abandoned (status stays "running" — MVP acceptable, see plan).
"""

from __future__ import annotations

import fcntl
import json
import os
import shlex
import signal
import subprocess
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional


# Caps to keep the WebUI process honest. Tunable via env at module import
# so ops can override without code edits.
MAX_CONCURRENT_ANALYSTS = int(os.environ.get("METAINFER_QA_MAX_CONCURRENT", "4"))
ANALYST_TIMEOUT_S = int(os.environ.get("METAINFER_QA_TIMEOUT_S", "600"))  # 10 min
ANALYST_STUCK_S = int(os.environ.get("METAINFER_QA_STUCK_S", "180"))  # 3 min no output
ANALYST_MAX_RETRIES = 1

# Process-wide semaphore — protects against a user spamming Ask buttons.
_ANALYST_SEMAPHORE = threading.Semaphore(MAX_CONCURRENT_ANALYSTS)


class EventsFileNotFound(Exception):
    """Raised by start_qa_session when the target events.jsonl does not
    exist. Distinct type so the FastAPI route can map ONLY this to 404
    without catching unrelated FileNotFoundErrors from index writes."""


class BudgetExhausted(Exception):
    """Raised by start_qa_session when the task's token-cost budget has
    been exhausted and no new analyst may launch. Route maps this to
    HTTP 429."""


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #

ANALYST_PROMPT_TEMPLATE = """\
You are a READ-ONLY analyst. Another LLM agent previously ran as part of
the MetaInfer calc-value pipeline. The user wants to understand what that
agent did and why, by asking you questions about that agent's conversation
history.

The target agent's full stream-json conversation log is at:
  {events_file}

The target agent's working directory (where it wrote intermediate artifacts):
  {target_workdir}

Target identity: {target_label}

The events file is line-delimited JSON. Each line is one event. Types:
  - {{"type": "system", ...}}           session init, contains session_id
  - {{"type": "user", "message": {{"content": [...]}}}}   the original prompt fed to the agent
  - {{"type": "assistant", "message": {{"content": [{{"type": "text"|"tool_use", ...}}]}}}}
  - {{"type": "tool_result", ...}}      results of tool calls (Read/Bash/Grep/etc.)
  - {{"type": "result", "result": "..."}}  final result text emitted at end of turn

HARD CONSTRAINTS:
- You are READ-ONLY. Do NOT call Edit, Write, or any mutating tool.
- Use only Read, Grep, Glob to inspect the events file and the target workdir.
- Answer based on what the target agent ACTUALLY said or did in its
  transcript. Quote specific events (with their line number in the file)
  when the user wants evidence.
- If the transcript does not contain enough information to answer, say
  so plainly. Do NOT fabricate.
- Keep your answer focused and reasonably concise — the user is reading
  it in a chat window.

User question:
{question}
"""


def _build_analyst_prompt(
    *,
    question: str,
    events_file: Path,
    target_workdir: Optional[Path],
    target_label: str,
) -> str:
    return ANALYST_PROMPT_TEMPLATE.format(
        events_file=str(events_file),
        target_workdir=str(target_workdir) if target_workdir else "(unknown)",
        target_label=target_label,
        question=question,
    )


# --------------------------------------------------------------------------- #
# ccb invocation
# --------------------------------------------------------------------------- #

def _claude_bin() -> str:
    return os.environ.get("METAINFER_CLAUDE_BIN", "ccb")


def _build_command(
    *,
    events_file: Path,
    target_workdir: Optional[Path],
    session_workdir: Path,
) -> List[str]:
    """Build the ccb command line for one analyst invocation.

    Mirrors subagent_manager._build_command but simpler — no retries,
    no session resume, no oracles. Just one fresh read-only ccb call.
    """
    cmd = [
        _claude_bin(),
        "-p",                                   # non-interactive, stdin prompt
        "--output-format", "stream-json",
        "--input-format", "text",
        "--verbose",
        "--permission-mode", "bypassPermissions",
        # The analyst's CWD = its session workdir (under qa_sessions/).
        "--add-dir", str(session_workdir),
        # Read-only data sources.
        "--add-dir", str(events_file.resolve().parent),
    ]
    if target_workdir is not None:
        cmd += ["--add-dir", str(target_workdir.resolve())]
    # Higher effort = better reasoning when digging through long transcripts.
    effort = os.environ.get("METAINFER_EFFORT", "max")
    if effort:
        cmd += ["--effort", effort]
    return cmd


def _build_env() -> Dict[str, str]:
    """Copy subagent_manager._build_env's IS_SANDBOX=1 trick so ccb
    doesn't refuse to run as root under bypassPermissions."""
    env = dict(os.environ)
    env.setdefault("DISABLE_INTERACTIVITY", "1")
    env["IS_SANDBOX"] = "1"
    return env


# --------------------------------------------------------------------------- #
# Event parsing — extract final assistant text from analyst_events.jsonl
# --------------------------------------------------------------------------- #

def _extract_final_text(events_jsonl_path: Path) -> str:
    """Same logic as subagent_manager._materialize_result's final_text
    extraction: walk events in reverse, return the last assistant text
    block, falling back to a result event's `result` field."""
    if not events_jsonl_path.exists():
        return ""
    events: List[Dict[str, Any]] = []
    for ln in events_jsonl_path.read_text(encoding="utf-8", errors="replace").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            events.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    for ev in reversed(events):
        if ev.get("type") == "assistant" and isinstance(ev.get("message"), dict):
            content = ev["message"].get("content")
            if isinstance(content, list):
                for blk in content:
                    if isinstance(blk, dict) and blk.get("type") == "text":
                        return blk.get("text", "") or ""
        if ev.get("type") == "result":
            r = ev.get("result", "")
            if r:
                return r
    return ""


# --------------------------------------------------------------------------- #
# Index file (qa_sessions/index.json)
# --------------------------------------------------------------------------- #

def _index_path(state_dir: Path) -> Path:
    return state_dir / "qa_sessions" / "index.json"


def _read_index(state_dir: Path) -> Dict[str, Any]:
    p = _index_path(state_dir)
    if not p.exists():
        return {"sessions": []}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("sessions"), list):
            return {"sessions": []}
        return data
    except (ValueError, OSError):
        return {"sessions": []}


def _write_index(state_dir: Path, data: Dict[str, Any]) -> None:
    """Write the index. Caller MUST hold _index_lock(state_dir) to avoid
    interleaved read-modify-writes from concurrent threads / processes.
    Uses a unique per-writer tmp filename so two writers can't steal
    each other's tmp file even if the lock is misused.
    """
    p = _index_path(state_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(
        f".index.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex[:8]}",
    )
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


@contextmanager
def _index_lock(state_dir: Path):
    """Exclusive flock on qa_sessions/index.lock. Cross-process safe —
    matches the registry-lock pattern in metainfer/web/tasks.py:_lock().
    Held across the read-modify-write so two concurrent starts can't
    lose each other's entries or fight over a shared tmp file.
    """
    lock_path = state_dir / "qa_sessions" / "index.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(lock_path, "a+", encoding="utf-8")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    finally:
        f.close()


def _index_entry_from_session_dir(session_dir: Path, request: Dict[str, Any],
                                  status: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": session_dir.name,
        "step": request.get("step"),
        "round": request.get("round"),
        "round_label": request.get("round_label"),
        "agent": request.get("agent"),
        "target_label": request.get("target_label"),
        "question": request.get("question"),
        "status": status.get("status"),
        "created_at": status.get("started_at"),
        "finished_at": status.get("finished_at"),
    }


def _upsert_index(state_dir: Path, entry: Dict[str, Any]) -> None:
    """Insert or update one session in the index by id.

    The whole read-modify-write is serialized via _index_lock so that
    concurrent starts (and concurrent finalize_session calls from
    finishing analysts) can't drop each other's entries or corrupt
    the shared index file.
    """
    with _index_lock(state_dir):
        data = _read_index(state_dir)
        sessions = data.get("sessions") or []
        found = False
        for i, s in enumerate(sessions):
            if s.get("id") == entry.get("id"):
                sessions[i] = entry
                found = True
                break
        if not found:
            sessions.append(entry)
        # Keep newest-first for nicer UI rendering.
        sessions.sort(key=lambda s: s.get("created_at") or 0, reverse=True)
        data["sessions"] = sessions
        _write_index(state_dir, data)


# --------------------------------------------------------------------------- #
# Path validation
# --------------------------------------------------------------------------- #

def _validate_path_inside(path: Path, root: Path) -> Path:
    """Resolve ``path`` and confirm it lives under ``root``. Returns the
    resolved path. Raises ValueError on traversal."""
    try:
        resolved = path.resolve(strict=False)
        root_resolved = root.resolve(strict=False)
    except OSError as exc:
        raise ValueError(f"cannot resolve path: {exc}") from exc
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"path {resolved} is outside task state dir {root_resolved}"
        ) from exc
    return resolved


# --------------------------------------------------------------------------- #
# Status file
# --------------------------------------------------------------------------- #

def _status_path(session_dir: Path) -> Path:
    return session_dir / "status.json"


def _write_status(session_dir: Path, status: Dict[str, Any]) -> None:
    p = _status_path(session_dir)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(p)


# --------------------------------------------------------------------------- #
# Session launch / run
# --------------------------------------------------------------------------- #

def start_qa_session(
    state_dir: Path,
    payload: Dict[str, Any],
) -> str:
    """Validate payload, persist request + prompt + initial status, spawn
    the analyst background thread, return the session id.

    ``payload`` keys:
        - events_file: absolute or state_dir-relative path to events.jsonl
        - target_workdir (optional): path to the agent's workdir
        - target_label: human-readable identifier (e.g. "step=3 round=0
          agent=embedding_writer_0")
        - question: user question
        - step / round / round_label / agent: optional metadata for
          filtering in /qa/list
    """
    state_dir = state_dir.resolve()
    # Required fields.
    events_file_str = (payload.get("events_file") or "").strip()
    question = (payload.get("question") or "").strip()
    if not events_file_str:
        raise ValueError("events_file is required")
    if not question:
        raise ValueError("question is required")

    events_file = _validate_path_inside(Path(events_file_str), state_dir)
    if not events_file.exists():
        # Use a typed sentinel so the route handler can map ONLY this
        # failure to 404 — other FileNotFoundErrors (e.g. index write
        # races) must NOT silently become 404s.
        raise EventsFileNotFound(f"events file not found: {events_file}")

    target_workdir = None
    if payload.get("target_workdir"):
        try:
            target_workdir = _validate_path_inside(
                Path(payload["target_workdir"]), state_dir,
            )
        except ValueError:
            # Don't fail the whole request — just drop the workdir.
            target_workdir = None

    target_label = payload.get("target_label") or (
        f"events_file={events_file.name}")

    # Token-budget pre-check: refuse to spawn a new analyst if the
    # task's cost budget is exhausted. The orchestrator process and
    # the WebUI process share state via the persisted JSON, so we
    # re-load the budget from disk here to see the latest totals.
    # (Concurrent record between the two processes is rare and at
    # worst causes one record to be lost on collision — acceptable
    # for the MVP soft-abort use case.)
    budget = _load_budget_for_check(state_dir)
    if budget is not None:
        refusal = budget.check_launch_allowed(f"qa:{target_label}")
        if refusal is not None:
            raise BudgetExhausted(refusal)

    prompt = _build_analyst_prompt(
        question=question,
        events_file=events_file,
        target_workdir=target_workdir,
        target_label=target_label,
    )

    session_id = uuid.uuid4().hex[:16]
    sessions_root = state_dir / "qa_sessions"
    session_dir = sessions_root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    request_rec = {
        "id": session_id,
        "step": payload.get("step"),
        "round": payload.get("round"),
        "round_label": payload.get("round_label"),
        "agent": payload.get("agent"),
        "target_label": target_label,
        "question": question,
        "events_file": str(events_file),
        "target_workdir": str(target_workdir) if target_workdir else None,
        "created_at": time.time(),
    }
    (session_dir / "request.json").write_text(
        json.dumps(request_rec, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    prompt_path = session_dir / "prompt.txt"
    prompt_path.write_text(prompt, encoding="utf-8")

    started_at = time.time()
    status = {
        "id": session_id,
        "status": "running",
        "started_at": started_at,
        "finished_at": None,
        "error": None,
    }
    _write_status(session_dir, status)
    _upsert_index(state_dir, _index_entry_from_session_dir(
        session_dir, request_rec, status,
    ))

    # Spawn the analyst. Daemon so it dies with the WebUI process.
    t = threading.Thread(
        target=_run_analyst,
        args=(state_dir, session_dir, events_file, target_workdir, prompt_path,
              started_at),
        name=f"qa-analyst-{session_id}",
        daemon=True,
    )
    t.start()

    return session_id


def _run_analyst(
    state_dir: Path,
    session_dir: Path,
    events_file: Path,
    target_workdir: Optional[Path],
    prompt_path: Path,
    started_at: float,
) -> None:
    """Background-thread worker: spawn ccb, drain stdout, finalize."""
    cmd = _build_command(
        events_file=events_file,
        target_workdir=target_workdir,
        session_workdir=session_dir,
    )
    env = _build_env()
    events_out = session_dir / "analyst_events.jsonl"
    log_out = session_dir / "analyst.log"

    proc: Optional[subprocess.Popen] = None
    err_msg: Optional[str] = None
    try:
        with semaphore_slot():
            log_fp = open(log_out, "wb")
            events_fp = open(events_out, "w", encoding="utf-8")
            try:
                with open(prompt_path, "rb") as stdin_fp:
                    proc = subprocess.Popen(
                        cmd,
                        stdin=stdin_fp,
                        stdout=subprocess.PIPE,
                        stderr=log_fp,
                        cwd=str(session_dir),
                        env=env,
                        text=False,
                        start_new_session=True,
                    )
                last_output_at = time.time()
                # Drain stdout on a thread so we can watchdog it.
                stop_evt = threading.Event()

                def drain() -> None:
                    nonlocal last_output_at
                    assert proc is not None and proc.stdout is not None
                    for raw in proc.stdout:
                        events_fp.write(raw.decode("utf-8", errors="replace"))
                        events_fp.flush()
                        log_fp.write(raw)
                        log_fp.flush()
                        last_output_at = time.time()

                reader = threading.Thread(target=drain, name=f"qa-drain-{session_dir.name}", daemon=True)
                reader.start()

                # Watchdog.
                while proc.poll() is None:
                    now = time.time()
                    if now - started_at > ANALYST_TIMEOUT_S:
                        _terminate_proc(proc, reason="timeout")
                        err_msg = f"analyst timed out after {ANALYST_TIMEOUT_S}s"
                        break
                    if now - last_output_at > ANALYST_STUCK_S:
                        _terminate_proc(proc, reason="stuck")
                        err_msg = (f"analyst stuck (no output for "
                                   f"{ANALYST_STUCK_S}s)")
                        break
                    time.sleep(2.0)
                proc.wait(timeout=30)
                reader.join(timeout=10)
            finally:
                log_fp.close()
                events_fp.close()
    except Exception as exc:  # noqa: BLE001
        err_msg = f"analyst crashed: {exc!r}"
    finally:
        # Always finalize status, even on failure.
        finalize_session(state_dir, session_dir, started_at, err_msg)


def finalize_session(
    state_dir: Path, session_dir: Path, started_at: float,
    err_msg: Optional[str],
) -> None:
    """Write the terminal status.json + answer.txt + update index."""
    finished_at = time.time()
    events_out = session_dir / "analyst_events.jsonl"
    answer = _extract_final_text(events_out) if events_out.exists() else ""

    if err_msg is not None:
        status = "failed"
    elif not answer.strip():
        # Process produced no parseable final text — call it failed.
        status = "failed"
        err_msg = "analyst produced no parseable final text"
    else:
        status = "done"

    new_status = {
        "id": session_dir.name,
        "status": status,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": err_msg,
    }
    _write_status(session_dir, new_status)
    (session_dir / "answer.txt").write_text(answer, encoding="utf-8")

    # Fold the analyst's cost into the task's token budget. Best-effort;
    # helper swallows its own errors.
    request_p = session_dir / "request.json"
    request_rec: Dict[str, Any] = {}
    if request_p.exists():
        try:
            request_rec = json.loads(request_p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    _record_analyst_usage(state_dir, session_dir,
                          request_rec.get("target_label"))

    # Update the index entry.
    _upsert_index(state_dir, _index_entry_from_session_dir(
        session_dir, request_rec, new_status,
    ))


class semaphore_slot:
    """Context manager wrapping _ANALYST_SEMAPHORE so the worker thread
    can give up its slot in a finally block."""
    def __enter__(self):
        _ANALYST_SEMAPHORE.acquire()
        return self
    def __exit__(self, *exc):
        _ANALYST_SEMAPHORE.release()


def _terminate_proc(proc: subprocess.Popen, reason: str = "manual") -> None:
    """Mirror subagent_manager._terminate: graceful SIGTERM, hard SIGKILL
    after grace period. Mark events.jsonl with a kill meta-record."""
    try:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            return
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                proc.kill()
    finally:
        pass  # events file is closed by the caller


# --------------------------------------------------------------------------- #
# Token budget integration
# --------------------------------------------------------------------------- #
#
# The qa module runs in the WebUI process, which is SEPARATE from the
# orchestrator process that owns the canonical SubAgentManager. Both
# processes share state via the persisted ``<state_dir>/token_budget.json``
# file. We construct a fresh TokenBudget for each operation (pre-check
# and post-run record) so we always see the latest totals from disk.
#
# Concurrency caveat: if the orchestrator and an analyst both finish
# within the same ~10ms window, one record may be lost on file replace.
# Acceptable for the soft-abort use case (running total is best-effort
# at the boundary); a file-lock-based scheme would be needed for
# strict accounting.


def _load_budget_for_check(state_dir: Path) -> Optional[Any]:
    """Construct a TokenBudget that picks up limits + history from disk.

    Returns None if no budget file exists (task has no limit configured
    — the common case for legacy tasks created before this feature).
    """
    # Local import — keeps the qa module importable even if some
    # downstream user pulls it out of the orchestrator package.
    from ..orchestrator.token_budget import TokenBudget  # noqa: PLC0415
    budget_path = state_dir / "token_budget.json"
    if not budget_path.exists():
        return None
    # Passing no limits forces TokenBudget to honor whatever is on disk
    # (which is exactly what we want — the WebUI shouldn't second-guess
    # the limits the orchestrator was launched with).
    return TokenBudget(state_dir)


def _record_analyst_usage(
    state_dir: Path, session_dir: Path, target_label: Optional[str],
) -> None:
    """Scan analyst_events.jsonl for the ``result`` event and record its
    usage into the task's budget. Best-effort: any error is swallowed
    so a recording failure can't make a successful analyst look failed.
    """
    try:
        from ..orchestrator.token_budget import (  # noqa: PLC0415
            TokenBudget, usage_from_result_event,
        )
        budget = _load_budget_for_check(state_dir)
        if budget is None:
            return  # task has no budget configured
        events_path = session_dir / "analyst_events.jsonl"
        if not events_path.exists():
            return
        result_ev: Optional[Dict[str, Any]] = None
        with open(events_path, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    ev = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if isinstance(ev, dict) and ev.get("type") == "result":
                    result_ev = ev
        if result_ev is None:
            return  # no result event → killed/crashed → nothing to record
        rec = usage_from_result_event(
            result_ev,
            agent=f"qa:{target_label or session_dir.name}",
            source="web_qa",
            phase=None,
        )
        budget.record(rec)
    except Exception:  # noqa: BLE001
        # Budget recording must never break the QA flow. The analyst
        # already produced a valid answer; lose the record rather than
        # the answer.
        return


# --------------------------------------------------------------------------- #
# Read API
# --------------------------------------------------------------------------- #

def get_qa_session(state_dir: Path, session_id: str) -> Optional[Dict[str, Any]]:
    """Return one session's full state for the GET endpoint."""
    session_dir = state_dir / "qa_sessions" / session_id
    if not session_dir.is_dir():
        return None
    out: Dict[str, Any] = {"id": session_id}
    request_p = session_dir / "request.json"
    if request_p.exists():
        try:
            out["request"] = json.loads(request_p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            out["request"] = None
    status_p = _status_path(session_dir)
    if status_p.exists():
        try:
            out["status"] = json.loads(status_p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            out["status"] = {"status": "unknown"}
    else:
        out["status"] = {"status": "unknown"}
    answer_p = session_dir / "answer.txt"
    if answer_p.exists():
        out["answer"] = answer_p.read_text(encoding="utf-8")
    else:
        out["answer"] = None
    return out


def list_qa_sessions(
    state_dir: Path,
    *,
    step: Optional[Any] = None,
    round_: Optional[Any] = None,
    agent: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return sessions filtered by target. Each step/round/agent may be
    a string or int; comparison is by stringified equality.

    Takes the shared lock too — the index file is replaced atomically
    by writers, but reading under the lock gives a consistent snapshot
    and avoids any chance of catching a half-written file.
    """
    with _index_lock(state_dir):
        data = _read_index(state_dir)
    out: List[Dict[str, Any]] = []
    for s in data.get("sessions") or []:
        if step is not None and str(s.get("step")) != str(step):
            continue
        if round_ is not None and str(s.get("round")) != str(round_):
            continue
        if agent is not None and s.get("agent") != agent:
            continue
        out.append(s)
    return out


def reconcile_on_startup(state_dir: Path) -> int:
    """On WebUI startup, mark any 'running' sessions as failed since
    their background thread died with the previous process. Returns
    the count of sessions reconciled."""
    sessions_root = state_dir / "qa_sessions"
    if not sessions_root.exists():
        return 0
    count = 0
    for sd in sessions_root.iterdir():
        if not sd.is_dir():
            continue
        status_p = _status_path(sd)
        if not status_p.exists():
            continue
        try:
            status = json.loads(status_p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if status.get("status") != "running":
            continue
        status["status"] = "failed"
        status["error"] = "webui restarted (analyst thread lost)"
        status["finished_at"] = time.time()
        _write_status(sd, status)
        # Reflect in index.
        request_p = sd / "request.json"
        request_rec: Dict[str, Any] = {}
        if request_p.exists():
            try:
                request_rec = json.loads(request_p.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                pass
        _upsert_index(state_dir, _index_entry_from_session_dir(
            sd, request_rec, status,
        ))
        count += 1
    return count
