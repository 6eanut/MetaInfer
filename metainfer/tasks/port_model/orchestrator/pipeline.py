"""port-model pipeline — 6-agent state machine driver.

State graph (see ``phases.py``)::

    P1 → P2 → P3 → P4 → P5 → P6 → finished
                ↘ ↑       ↘ ↑    ↘ ↑
                 bounce   repair  self-repair

* P1 reads model weights + config.json → ``p1_weight_analysis.md``.
* P2 fans out: one analyst agent per reference source. Outputs are
  merged sequentially (they don't depend on each other).
* P3 cross-validates; may bounce back to P1 (capped at
  ``MAX_P3_BOUNCE = 2``).
* P4 builds the minimal PyTorch framework; P5 verifies it. Failure
  routes back to P4 (capped at ``MAX_P5_REPAIR = 3``).
* P6 ports to ``target_framework_dir``, runs its own internal
  similarity-debug loop (capped at ``MAX_P6_ITER = 5``). Each non-empty
  P6 iteration commits inside target_fw_dir (auto-init git if needed).
* Each phase writes ``summary.md`` to its workdir; the orchestrator
  also creates a git commit in ``workspace_dir`` per phase for
  auditability (the WebUI exposes these via /memory endpoints).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from metainfer.orchestrator.requirements import req_field
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import (
    AgentResult,
    AgentSpec,
    SubAgentManager,
)

from . import phases as P
from . import prompts as PP
from .iteration_record import (
    IterationRecord,
    PhaseRecord,
    next_iteration_number,
    read_summary_excerpt,
    write_iteration,
)


# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #

MAX_P3_BOUNCE = 2
MAX_P5_REPAIR = 3
MAX_P6_ITER = 5

PER_AGENT_TIMEOUT_S = {
    "P1_weight_analysis":   1200,
    "P2_framework_analysis": 1800,
    "P3_architect_review":  1800,
    "P4_minimal_framework": 2400,
    "P5_verify_minimal":    1800,
    "P6_port_engine":       3600,
}

# Default test prompt when the user did not supply one.
DEFAULT_TEST_PROMPT = "中国的首都是"


# --------------------------------------------------------------------------- #
# Config dataclass
# --------------------------------------------------------------------------- #


@dataclass
class PipelineConfig:
    state_dir: Path
    workspace_dir: Path
    # Per-phase workdirs under workspace:
    p1_dir: Path
    p2_dir: Path
    p3_dir: Path
    p4_dir: Path
    p5_dir: Path
    p6_dir: Path
    # Canonical artifact locations the WebUI reads:
    memory_dir: Path
    dumps_dir: Path
    target_fw_dir: Path
    model_params_path: Path
    reference_sources: List[Dict[str, Any]] = field(default_factory=list)
    user_notes: str = ""
    # Optional worker node IDs for distributed end-to-end testing (P5/P6).
    # Empty = run locally on the orchestrator node.
    worker_nodes: List[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _failure_outcome(mode: Optional[str]) -> P.Outcome:
    return P.INFRA_FAIL if mode == "infra" else P.LOGIC_FAIL


def _write_prompt_file(workdir: Path, name: str, text: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / f"{name}.prompt.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _launch_blocking(
    *,
    manager: SubAgentManager,
    name: str,
    role: str,
    workdir: Path,
    logs_dir: Path,
    prompt_text: str,
    timeout: int,
    extra_args: Optional[List[str]] = None,
    resume_session_id: Optional[str] = None,
) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
    """Launch one agent, block until done, return (ok, err, mode, session)."""
    prompt_file = _write_prompt_file(logs_dir, name, prompt_text)
    spec = AgentSpec(
        name=name,
        role=role,
        prompt_file=prompt_file,
        workdir=workdir,
        log_dir=logs_dir,
        timeout_s=timeout,
        stuck_timeout_s=max(120, timeout // 3),
        extra_args=list(extra_args or []),
        resume_session_id=resume_session_id,
    )
    manager.launch(spec)
    r = manager.result(name)
    if r is None:
        return False, "no result recorded", "infra", None
    return r.success, r.error, r.failure_mode, r.session_id


def _git_commit_in(dir_path: Path, message: str, allow_init: bool = True) -> Optional[str]:
    """Create a git commit in ``dir_path``. Returns the SHA or None.

    Auto-inits a repo if allow_init and the dir isn't yet a git repo.
    Best-effort — git failures return None and the pipeline continues.
    """
    if not dir_path.is_dir():
        return None
    git_dir = dir_path / ".git"
    if not git_dir.exists():
        if not allow_init:
            return None
        try:
            subprocess.run(["git", "init", "-q"], cwd=str(dir_path), check=True,
                           timeout=30)
            # Set a local identity if none configured (best-effort).
            try:
                subprocess.run(
                    ["git", "config", "user.email"],
                    cwd=str(dir_path), check=True,
                    capture_output=True, timeout=10,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    ["git", "config", "user.email", "metainfer@example.com"],
                    cwd=str(dir_path), check=False, timeout=10,
                )
                subprocess.run(
                    ["git", "config", "user.name", "MetaInfer port-model"],
                    cwd=str(dir_path), check=False, timeout=10,
                )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
            return None
    try:
        subprocess.run(["git", "add", "-A"], cwd=str(dir_path), check=False,
                       timeout=120)
        # Are there staged changes?
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=str(dir_path),
            capture_output=True, timeout=30,
        )
        if diff.returncode == 0:
            # Nothing to commit.
            return None
        subprocess.run(
            ["git", "commit", "-q", "-m", message],
            cwd=str(dir_path), check=True, timeout=120,
        )
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(dir_path), capture_output=True, text=True, check=True,
            timeout=30,
        ).stdout.strip()
        return sha or None
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return None


def _read_summary(workdir: Path) -> Optional[str]:
    p = workdir / "summary.md"
    if not p.is_file():
        return None
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _parse_summary_outcome(summary_text: Optional[str], default: str) -> str:
    """Extract the ``## Outcome`` value from summary.md, fall back to default."""
    if not summary_text:
        return default
    for line in summary_text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("## outcome"):
            continue
        # First non-empty line after the header.
        if stripped and not stripped.startswith("#"):
            token = stripped.split()[0].lower()
            if token in P.ALL_OUTCOMES:
                return token
            return default
    return default


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #


class Pipeline:
    def __init__(
        self,
        req: Dict[str, Any],
        store: StateStore,
        cfg: PipelineConfig,
        manager: SubAgentManager,
        budget: Any = None,
        extra_claude_args: Optional[List[str]] = None,
    ) -> None:
        self.req = req
        self.store = store
        self.cfg = cfg
        self.manager = manager
        self.budget = budget
        self.extra_claude_args = list(extra_claude_args or [])
        self.task_id = req.get("task_id", "task")
        self._stop = False

        # Counters for repair / bounce loops.
        self._p3_bounce_count = 0
        self._p5_repair_count = 0
        self._p6_iter_count = 0

        # Current iteration record (one per top-level pass).
        self._iter_rec = IterationRecord(
            iteration=next_iteration_number(cfg.state_dir),
            goal="port_model: full 6-phase pass",
        )
        write_iteration(cfg.state_dir, self._iter_rec)

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(self) -> int:
        # Ensure run.json exists before any update_run call.
        self.store.init_or_resume(self.task_id)
        self.store.append_timeline("port_model.start", {
            "task_id": self.task_id,
            "iteration": self._iter_rec.iteration,
        })

        phase: P.Phase = self._resume_phase()
        last_outcome: Optional[P.Outcome] = None
        last_failure_msg: Optional[str] = None

        try:
            while not self._stop and not P.is_terminal(phase):
                if self.budget is not None and self.budget.snapshot().exhausted:
                    self._abort_budget()
                    return 0

                self._set_phase(phase)
                outcome, failure_msg = self._dispatch(
                    phase, prev_failure=last_failure_msg,
                )
                last_outcome = outcome
                last_failure_msg = failure_msg

                self.store.append_timeline("transition", {
                    "from": phase, "outcome": outcome, "failure": failure_msg,
                })

                # Update loop counters BEFORE computing next phase so
                # caps can fire.
                if phase == "P3_architect_review" and outcome == P.BOUNCE_BACK:
                    self._p3_bounce_count += 1
                    if self._p3_bounce_count > MAX_P3_BOUNCE:
                        self.store.append_timeline("p3_bounce_capped", {
                            "count": self._p3_bounce_count,
                        })
                        outcome = P.OK  # force-proceed
                elif phase == "P5_verify_minimal" and outcome in (P.TEST_FAIL, P.INFRA_FAIL):
                    self._p5_repair_count += 1
                    if self._p5_repair_count > MAX_P5_REPAIR:
                        self.store.append_timeline("p5_repair_capped", {
                            "count": self._p5_repair_count,
                        })
                        self._fail_run(
                            f"P5 minimal-framework verification failed after "
                            f"{MAX_P5_REPAIR} repair attempts"
                        )
                        return 1
                elif phase == "P6_port_engine" and outcome in (
                    P.NEEDS_REPAIR, P.TEST_FAIL, P.INFRA_FAIL,
                ):
                    self._p6_iter_count += 1
                    if self._p6_iter_count > MAX_P6_ITER:
                        self.store.append_timeline("p6_iter_capped", {
                            "count": self._p6_iter_count,
                        })
                        self._fail_run(
                            f"P6 target-framework porting did not converge "
                            f"after {MAX_P6_ITER} iterations"
                        )
                        return 1

                t = P.next_transition(phase, outcome)
                if t is None:
                    self._fail_run(f"no transition for ({phase}, {outcome})")
                    return 1

                self.store.update_run(
                    current_phase=t.to_phase,
                    last_outcome=outcome,
                    last_transition_label=t.label,
                )
                phase = t.to_phase

            # Reached terminal phase.
            final_status = (
                "success" if last_outcome == P.OK else "stopped"
            )
            self._iter_rec.ended_at = time.time()
            self._iter_rec.duration_s = (
                self._iter_rec.ended_at - self._iter_rec.started_at
            )
            self._iter_rec.status = (
                "success" if final_status == "success" else "failed"
            )
            self._iter_rec.final_status = final_status
            write_iteration(self.cfg.state_dir, self._iter_rec)

            self.store.update_run(
                finished=True,
                final_status=final_status,
                current_phase="finished",
                last_outcome=last_outcome,
            )
            self.store.append_timeline("port_model.end", {
                "task_id": self.task_id, "final_status": final_status,
            })
            return 0
        except KeyboardInterrupt:
            self.store.append_timeline(
                "port_model.abort", {"reason": "keyboard-interrupt"}
            )
            self.store.update_run(
                finished=True, final_status="aborted", current_phase="finished",
            )
            return 130
        finally:
            try:
                self.manager.shutdown()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ #
    # Resume detection
    # ------------------------------------------------------------------ #

    def _resume_phase(self) -> P.Phase:
        """Pick up where we left off, based on artifact presence.

        Used after a restart-from-crash. For WebUI-triggered rerun_step
        the route handler wipes the step dirs first, so the artifacts
        we look for here won't exist and we re-run from scratch.
        """
        if (self.cfg.p3_dir / "p3_consolidated_spec.md").is_file():
            if (self.cfg.p4_dir / "run.py").is_file():
                # Check if P5 dumps exist.
                if any(self.cfg.dumps_dir.glob("layer_*.npy")):
                    return "P6_port_engine"
                return "P5_verify_minimal"
            return "P4_minimal_framework"
        if (self.cfg.p1_dir / "p1_weight_analysis.md").is_file():
            return "P3_architect_review"
        return "P1_weight_analysis"

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #

    def _dispatch(
        self, phase: P.Phase, *, prev_failure: Optional[str],
    ) -> Tuple[P.Outcome, Optional[str]]:
        if phase == "P1_weight_analysis":
            return self._do_p1(prev_failure=prev_failure or "")
        if phase == "P2_framework_analysis":
            return self._do_p2()
        if phase == "P3_architect_review":
            return self._do_p3()
        if phase == "P4_minimal_framework":
            return self._do_p4(prev_failure=prev_failure or "")
        if phase == "P5_verify_minimal":
            return self._do_p5()
        if phase == "P6_port_engine":
            return self._do_p6(prev_failure=prev_failure or "")
        return P.LOGIC_FAIL, f"no handler for phase {phase!r}"

    # ------------------------------------------------------------------ #
    # P1: weight analysis
    # ------------------------------------------------------------------ #

    def _do_p1(self, *, prev_failure: str) -> Tuple[P.Outcome, Optional[str]]:
        cfg = self.cfg
        # Clear stale P1 outputs when re-running (bounce-back or rerun).
        if not prev_failure:
            self._wipe_phase("P1_weight_analysis")

        logs_dir = cfg.state_dir / "logs" / "p1"
        workdir = cfg.p1_dir
        prompt = PP.p1_weight_analysis_prompt(
            req=self.req, workdir=workdir, prev_failure=prev_failure,
        )
        started = time.time()
        ok, err, mode, _ = _launch_blocking(
            manager=self.manager, name="p1-weight-analyst",
            role="p1_weight_analyst", workdir=workdir, logs_dir=logs_dir,
            prompt_text=prompt, timeout=PER_AGENT_TIMEOUT_S["P1_weight_analysis"],
            extra_args=self.extra_claude_args,
        )

        summary = _read_summary(workdir)
        outcome_str = _parse_summary_outcome(summary, P.OK if ok else "logic_fail")
        phase_rec = PhaseRecord(
            phase="P1_weight_analysis",
            outcome=outcome_str if ok else _failure_outcome(mode),
            started_at=started, ended_at=time.time(),
            duration_s=time.time() - started,
            agent_name="p1-weight-analyst",
            summary_path=str(workdir / "summary.md") if summary else None,
            summary_excerpt=read_summary_excerpt(str(workdir / "summary.md")),
            error=err,
            artifacts=[str(workdir / "p1_weight_analysis.md")] if
                (workdir / "p1_weight_analysis.md").is_file() else [],
        )
        self._iter_rec.upsert_phase(phase_rec)
        write_iteration(cfg.state_dir, self._iter_rec)

        if not ok:
            return _failure_outcome(mode), f"P1 failed: {err}"

        # Persist canonical copy under memory/.
        canonical = cfg.memory_dir / "p1_weight_analysis.md"
        src = workdir / "p1_weight_analysis.md"
        if src.is_file():
            cfg.memory_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, canonical)
                self._iter_rec.p1_artifact = str(canonical)
            except OSError as exc:
                return P.INFRA_FAIL, f"failed to persist P1 artifact: {exc}"
        if not canonical.is_file():
            return P.LOGIC_FAIL, "P1 produced no p1_weight_analysis.md"

        self._commit_workspace("port_model(P1): weight analysis complete")
        # Clear P3 bounce counter if P1 succeeds.
        return P.OK, None

    # ------------------------------------------------------------------ #
    # P2: framework analysis (fan-out, one per reference source)
    # ------------------------------------------------------------------ #

    def _do_p2(self) -> Tuple[P.Outcome, Optional[str]]:
        cfg = self.cfg
        self._wipe_phase("P2_framework_analysis")
        refs = cfg.reference_sources
        if not refs:
            # Nothing to analyse — skip straight to P3.
            self.store.append_timeline("p2_skipped", {"reason": "no references"})
            self._iter_rec.upsert_phase(PhaseRecord(
                phase="P2_framework_analysis", outcome=P.OK,
                started_at=time.time(), ended_at=time.time(),
                error="no reference sources supplied",
            ))
            write_iteration(cfg.state_dir, self._iter_rec)
            return P.OK, None

        p1_path = cfg.memory_dir / "p1_weight_analysis.md"
        logs_dir = cfg.state_dir / "logs" / "p2"

        # Fan out: one thread per reference source. SubAgentManager
        # caps concurrency via max_concurrent.
        threads: List[threading.Thread] = []
        results: Dict[int, Tuple[bool, Optional[str], Optional[str]]] = {}
        results_lock = threading.Lock()

        def _one(idx: int, ref: Dict[str, Any]) -> None:
            ref_path = ref.get("path") or ""
            ref_notes = ref.get("notes") or ""
            sub_workdir = cfg.p2_dir / f"ref{idx}"
            sub_logs = logs_dir / f"ref{idx}"
            # The subprocess Popen() uses sub_workdir as cwd — must exist
            # before launch. P5/P6 mkdir their attempt dirs; P2 has to
            # do the same per-reference, otherwise the agent crashes on
            # spawn with FileNotFoundError (and the thread dies silently
            # because the exception is bound to a worker thread).
            sub_workdir.mkdir(parents=True, exist_ok=True)
            prompt = PP.p2_framework_analysis_prompt(
                req=self.req, workdir=sub_workdir,
                ref_index=idx + 1, ref_path=ref_path, ref_notes=ref_notes,
                p1_path=p1_path,
            )
            ok, err, mode, _ = _launch_blocking(
                manager=self.manager,
                name=f"p2-analyst-ref{idx}",
                role="p2_framework_analyst",
                workdir=sub_workdir, logs_dir=sub_logs,
                prompt_text=prompt,
                timeout=PER_AGENT_TIMEOUT_S["P2_framework_analysis"],
                extra_args=self.extra_claude_args,
            )
            with results_lock:
                results[idx] = (ok, err, mode)

        started = time.time()
        for i, ref in enumerate(refs):
            t = threading.Thread(target=_one, args=(i, ref), daemon=True,
                                 name=f"p2-ref{i}")
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # Collect per-ref artifacts + phase record.
        artifacts: List[str] = []
        any_ok = False
        first_err: Optional[str] = None
        for i in range(len(refs)):
            sub_workdir = cfg.p2_dir / f"ref{i}"
            artifact = sub_workdir / f"p2_ref{i + 1}_analysis.md"
            ok, err, mode = results.get(i, (False, "no result", "infra"))
            if ok and artifact.is_file():
                canon = cfg.memory_dir / f"p2_ref{i + 1}_analysis.md"
                cfg.memory_dir.mkdir(parents=True, exist_ok=True)
                try:
                    shutil.copy2(artifact, canon)
                    artifacts.append(str(canon))
                    any_ok = True
                except OSError as exc:
                    if first_err is None:
                        first_err = f"P2 ref{i}: copy failed: {exc}"
            else:
                if first_err is None:
                    first_err = f"P2 ref{i}: {err}"

        self._iter_rec.upsert_phase(PhaseRecord(
            phase="P2_framework_analysis",
            outcome=P.OK if any_ok else P.LOGIC_FAIL,
            started_at=started, ended_at=time.time(),
            duration_s=time.time() - started,
            agent_name="p2-analyst-pool",
            artifacts=artifacts,
            error=first_err,
            extra={"ref_count": len(refs)},
        ))
        self._iter_rec.p2_artifacts = artifacts
        write_iteration(cfg.state_dir, self._iter_rec)

        if not any_ok:
            return P.LOGIC_FAIL, first_err or "P2 produced no analyses"
        self._commit_workspace("port_model(P2): framework analyses complete")
        return P.OK, None

    # ------------------------------------------------------------------ #
    # P3: architect review
    # ------------------------------------------------------------------ #

    def _do_p3(self) -> Tuple[P.Outcome, Optional[str]]:
        cfg = self.cfg
        self._wipe_phase("P3_architect_review")
        p1_path = cfg.memory_dir / "p1_weight_analysis.md"
        p2_paths = sorted(cfg.memory_dir.glob("p2_ref*_analysis.md"))
        if not p1_path.is_file():
            return P.LOGIC_FAIL, "P1 analysis missing"
        if not p2_paths:
            # If user provided zero refs, treat the consolidated spec
            # as just P1's output (P4 will work from there).
            self.store.append_timeline("p3_no_p2", {"reason": "no P2 artifacts"})

        logs_dir = cfg.state_dir / "logs" / "p3"
        workdir = cfg.p3_dir
        prompt = PP.p3_architect_review_prompt(
            req=self.req, workdir=workdir,
            p1_path=p1_path, p2_paths=p2_paths,
            bounce_count=self._p3_bounce_count,
        )
        started = time.time()
        ok, err, mode, _ = _launch_blocking(
            manager=self.manager, name="p3-architect",
            role="p3_architect", workdir=workdir, logs_dir=logs_dir,
            prompt_text=prompt,
            timeout=PER_AGENT_TIMEOUT_S["P3_architect_review"],
            extra_args=self.extra_claude_args,
        )
        summary = _read_summary(workdir)
        outcome_str = _parse_summary_outcome(
            summary, P.OK if ok else "logic_fail",
        )

        # If the agent says bounce_back, surface that even on "ok" exit.
        final_outcome = P.OK
        if outcome_str == P.BOUNCE_BACK and self._p3_bounce_count < MAX_P3_BOUNCE:
            final_outcome = P.BOUNCE_BACK
        elif not ok:
            final_outcome = _failure_outcome(mode)

        self._iter_rec.upsert_phase(PhaseRecord(
            phase="P3_architect_review",
            outcome=final_outcome,
            started_at=started, ended_at=time.time(),
            duration_s=time.time() - started,
            agent_name="p3-architect",
            summary_path=str(workdir / "summary.md") if summary else None,
            summary_excerpt=read_summary_excerpt(str(workdir / "summary.md")),
            error=err,
            artifacts=[str(workdir / "p3_consolidated_spec.md")] if
                (workdir / "p3_consolidated_spec.md").is_file() else [],
        ))
        write_iteration(cfg.state_dir, self._iter_rec)

        if final_outcome == P.BOUNCE_BACK:
            return P.BOUNCE_BACK, summary or "architect bounced P1"
        if not ok:
            return _failure_outcome(mode), f"P3 failed: {err}"

        # Persist consolidated spec.
        spec = workdir / "p3_consolidated_spec.md"
        if spec.is_file():
            cfg.memory_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(spec, cfg.memory_dir / "p3_consolidated_spec.md")
                self._iter_rec.p3_artifact = str(cfg.memory_dir / "p3_consolidated_spec.md")
            except OSError as exc:
                return P.INFRA_FAIL, f"failed to persist P3 spec: {exc}"
        else:
            return P.LOGIC_FAIL, "P3 produced no p3_consolidated_spec.md"

        self._commit_workspace("port_model(P3): architect review complete")
        return P.OK, None

    # ------------------------------------------------------------------ #
    # P4: minimal framework
    # ------------------------------------------------------------------ #

    def _do_p4(self, *, prev_failure: str) -> Tuple[P.Outcome, Optional[str]]:
        cfg = self.cfg
        # Clear stale P4 outputs (so dump filenames don't pile up), but
        # keep the previous verdict if any so we don't lose context.
        if not prev_failure:
            self._wipe_phase("P4_minimal_framework")

        p3_path = cfg.memory_dir / "p3_consolidated_spec.md"
        if not p3_path.is_file():
            return P.LOGIC_FAIL, "P3 consolidated spec missing"

        logs_dir = cfg.state_dir / "logs" / "p4"
        workdir = cfg.p4_dir
        prompt = PP.p4_minimal_framework_prompt(
            req=self.req, workdir=workdir,
            p3_path=p3_path, prev_failure=prev_failure,
        )
        started = time.time()
        ok, err, mode, _ = _launch_blocking(
            manager=self.manager, name="p4-min-builder",
            role="p4_minimal_framework_writer",
            workdir=workdir, logs_dir=logs_dir,
            prompt_text=prompt,
            timeout=PER_AGENT_TIMEOUT_S["P4_minimal_framework"],
            extra_args=self.extra_claude_args,
        )
        self._iter_rec.upsert_phase(PhaseRecord(
            phase="P4_minimal_framework",
            outcome=P.OK if ok else _failure_outcome(mode),
            started_at=started, ended_at=time.time(),
            duration_s=time.time() - started,
            agent_name="p4-min-builder",
            summary_path=str(workdir / "summary.md"),
            summary_excerpt=read_summary_excerpt(str(workdir / "summary.md")),
            error=err,
            artifacts=[str(workdir / "run.py")] if
                (workdir / "run.py").is_file() else [],
        ))
        self._iter_rec.p4_artifact = str(workdir) if ok else None
        write_iteration(cfg.state_dir, self._iter_rec)

        if not ok:
            return _failure_outcome(mode), f"P4 failed: {err}"
        if not (workdir / "run.py").is_file():
            return P.LOGIC_FAIL, "P4 produced no run.py"
        self._commit_workspace("port_model(P4): minimal framework built")
        return P.OK, None

    # ------------------------------------------------------------------ #
    # P5: verify minimal framework
    # ------------------------------------------------------------------ #

    def _do_p5(self) -> Tuple[P.Outcome, Optional[str]]:
        cfg = self.cfg
        # Note: P5 verdict dir is per-attempt to keep logs distinguishable.
        attempt = self._p5_repair_count  # 0,1,2,...
        attempt_dir = cfg.p5_dir / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        cfg.dumps_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = cfg.state_dir / "logs" / "p5" / f"attempt_{attempt:02d}"
        prompt = PP.p5_verify_minimal_prompt(
            req=self.req, workdir=attempt_dir, p4_dir=cfg.p4_dir,
            worker_nodes=cfg.worker_nodes,
        )
        started = time.time()
        ok, err, mode, _ = _launch_blocking(
            manager=self.manager, name=f"p5-verifier-a{attempt}",
            role="p5_minimal_framework_verifier",
            workdir=attempt_dir, logs_dir=logs_dir,
            prompt_text=prompt,
            timeout=PER_AGENT_TIMEOUT_S["P5_verify_minimal"],
            extra_args=self.extra_claude_args,
        )

        verdict_path = attempt_dir / "verdict.json"
        verdict: Dict[str, Any] = {}
        if verdict_path.is_file():
            try:
                verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                verdict = {}

        passed = bool(verdict.get("passed"))
        outcome = P.OK if (ok and passed) else (
            _failure_outcome(mode) if not ok else P.TEST_FAIL
        )

        # Copy any dumps the agent produced into the canonical dumps dir.
        produced_dumps_dir = attempt_dir / "dumps"
        if produced_dumps_dir.is_dir():
            for f in produced_dumps_dir.glob("layer_*.npy"):
                try:
                    shutil.copy2(f, cfg.dumps_dir / f.name)
                except OSError:
                    pass

        self._iter_rec.upsert_phase(PhaseRecord(
            phase="P5_verify_minimal",
            outcome=outcome,
            started_at=started, ended_at=time.time(),
            duration_s=time.time() - started,
            agent_name=f"p5-verifier-a{attempt}",
            summary_path=str(attempt_dir / "summary.md"),
            summary_excerpt=read_summary_excerpt(str(attempt_dir / "summary.md")),
            error=err,
            artifacts=[str(verdict_path)] if verdict_path.is_file() else [],
            extra={"attempt": attempt, "verdict": verdict},
        ))
        self._iter_rec.p5_verdict = verdict
        self._iter_rec.p5_dumps_dir = str(cfg.dumps_dir)
        write_iteration(cfg.state_dir, self._iter_rec)

        if outcome == P.OK:
            self._commit_workspace(
                f"port_model(P5): minimal framework verified (attempt {attempt})"
            )
            return P.OK, None
        # Surface the failure message for P4 to consume.
        msg = verdict.get("error_message") or verdict.get("reason") or err or "verify failed"
        log_path = attempt_dir / "run.log"
        if log_path.is_file():
            try:
                log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-3000:]
                msg = f"{msg}\n--- run.log tail ---\n{log_tail}"
            except OSError:
                pass
        return outcome, msg

    # ------------------------------------------------------------------ #
    # P6: port to target framework
    # ------------------------------------------------------------------ #

    def _do_p6(self, *, prev_failure: str) -> Tuple[P.Outcome, Optional[str]]:
        cfg = self.cfg
        p3_path = cfg.memory_dir / "p3_consolidated_spec.md"
        if not p3_path.is_file():
            return P.LOGIC_FAIL, "P3 consolidated spec missing"

        iter_idx = self._p6_iter_count  # 0-based; display as 1-based
        attempt_dir = cfg.p6_dir / f"iter_{iter_idx:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = cfg.state_dir / "logs" / "p6" / f"iter_{iter_idx:02d}"

        prompt = PP.p6_port_engine_prompt(
            req=self.req, workdir=attempt_dir,
            p3_path=p3_path, p5_dumps_dir=cfg.dumps_dir,
            iteration=iter_idx + 1, prev_failure=prev_failure,
            worker_nodes=cfg.worker_nodes,
        )
        started = time.time()
        ok, err, mode, _ = _launch_blocking(
            manager=self.manager, name=f"p6-porter-i{iter_idx}",
            role="p6_port_engineer",
            workdir=attempt_dir, logs_dir=logs_dir,
            prompt_text=prompt,
            timeout=PER_AGENT_TIMEOUT_S["P6_port_engine"],
            extra_args=self.extra_claude_args,
        )

        verdict_path = attempt_dir / f"verdict_{iter_idx + 1}.json"
        verdict: Dict[str, Any] = {}
        if verdict_path.is_file():
            try:
                verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                verdict = {}

        commit_sha: Optional[str] = None
        commit_sha_file = attempt_dir / f"commit_{iter_idx + 1}.txt"
        if commit_sha_file.is_file():
            try:
                commit_sha = commit_sha_file.read_text(encoding="utf-8").strip() or None
            except OSError:
                commit_sha = None
        if commit_sha is None:
            # Best-effort: query the target_fw repo for HEAD if it's a repo.
            commit_sha = _git_commit_in(
                cfg.target_fw_dir,
                f"port_model(P6 iter {iter_idx + 1}): porter auto-commit",
                allow_init=True,
            )

        verdict_outcome = verdict.get("outcome") or ("ok" if ok else "")
        if verdict_outcome == "ok":
            outcome = P.OK
        elif verdict_outcome in ("needs_repair", "test_fail"):
            outcome = P.NEEDS_REPAIR if verdict_outcome == "needs_repair" else P.TEST_FAIL
        elif not ok:
            outcome = _failure_outcome(mode)
        else:
            outcome = P.NEEDS_REPAIR  # default: another iteration

        rec = PhaseRecord(
            phase="P6_port_engine",
            outcome=outcome,
            started_at=started, ended_at=time.time(),
            duration_s=time.time() - started,
            agent_name=f"p6-porter-i{iter_idx}",
            summary_path=str(attempt_dir / "summary.md"),
            summary_excerpt=read_summary_excerpt(str(attempt_dir / "summary.md")),
            error=err,
            artifacts=[str(verdict_path)] if verdict_path.is_file() else [],
            extra={
                "iteration": iter_idx + 1,
                "verdict": verdict,
                "commit_sha": commit_sha,
            },
        )
        self._iter_rec.upsert_phase(rec)
        if iter_idx + 1 not in [v.get("iteration") for v in self._iter_rec.p6_iterations]:
            self._iter_rec.p6_iterations.append({
                "iteration": iter_idx + 1,
                "verdict": verdict,
                "commit_sha": commit_sha,
                "outcome": verdict_outcome,
            })
        if commit_sha and commit_sha not in self._iter_rec.p6_commit_shas:
            self._iter_rec.p6_commit_shas.append(commit_sha)
        write_iteration(cfg.state_dir, self._iter_rec)

        self.store.append_timeline("p6_iteration", {
            "iteration": iter_idx + 1,
            "outcome": outcome,
            "commit_sha": commit_sha,
            "verdict": verdict,
        })

        if outcome == P.OK:
            self._commit_workspace(
                f"port_model(P6): target framework ported (iter {iter_idx + 1})"
            )
            return P.OK, None
        # Hand the next P6 iteration a STRUCTURED view of this iter's
        # verdict (operator_replacements, first_bad_layer, reason) so it
        # resumes from known progress instead of starting blind. The
        # format_prev_p6_verdict helper lives in prompts.py so what the
        # next agent reads matches the prompt's contract.
        handover = PP.format_prev_p6_verdict(verdict)
        if not handover:
            handover = verdict.get("reason") or err or "P6 iteration incomplete"
        return outcome, handover

    # ------------------------------------------------------------------ #
    # Helpers: phase wipe, commit, abort
    # ------------------------------------------------------------------ #

    def _wipe_phase(self, phase: str) -> None:
        """Erase a phase's workdir before re-running it (rerun_step / repair)."""
        cfg = self.cfg
        target = {
            "P1_weight_analysis": cfg.p1_dir,
            "P2_framework_analysis": cfg.p2_dir,
            "P3_architect_review": cfg.p3_dir,
            "P4_minimal_framework": cfg.p4_dir,
        }.get(phase)
        if target is None:
            return
        # Don't remove the dir itself — agents may have prompts cached
        # alongside. Just clear contents.
        if target.is_dir():
            for entry in target.iterdir():
                try:
                    if entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
                    else:
                        entry.unlink()
                except OSError:
                    pass
        # Also drop the canonical memory copy so resume logic re-detects.
        canon = {
            "P1_weight_analysis": "p1_weight_analysis.md",
            "P3_architect_review": "p3_consolidated_spec.md",
        }.get(phase)
        if canon:
            mp = cfg.memory_dir / canon
            if mp.is_file():
                try:
                    mp.unlink()
                except OSError:
                    pass

    def _commit_workspace(self, message: str) -> None:
        """One git commit per phase in workspace_dir for auditability."""
        sha = _git_commit_in(self.cfg.workspace_dir, message, allow_init=True)
        if sha is not None:
            self.store.append_timeline("port_model.workspace_commit", {
                "sha": sha, "message": message,
            })

    def _set_phase(self, phase: P.Phase) -> None:
        self.store.update_run(current_phase=phase)
        self.store.append_timeline("phase_start", {"phase": phase})

    def _abort_budget(self) -> None:
        snap = self.budget.snapshot() if self.budget is not None else None
        self.store.update_run(
            finished=True, final_status="aborted", current_phase="finished",
            last_outcome=P.ABORTED,
            last_transition_label=(
                f"budget exhausted: ${snap.total_cost_usd:.4f} "
                f">= ${snap.limit_cost_usd:.4f}" if snap else "budget exhausted"
            ),
        )
        self.store.append_timeline("port_model.budget_exhausted", {
            "used_cost_usd": snap.total_cost_usd if snap else 0,
            "limit_cost_usd": snap.limit_cost_usd if snap else 0,
            "agent_count": snap.agent_count if snap else 0,
        })
        self._iter_rec.status = "aborted"
        self._iter_rec.final_status = "aborted"
        self._iter_rec.failure_reason = "budget exhausted"
        self._iter_rec.ended_at = time.time()
        write_iteration(self.cfg.state_dir, self._iter_rec)

    def _fail_run(self, reason: str) -> None:
        self.store.update_run(
            finished=True, final_status="stopped",
            current_phase="finished", last_outcome=P.LOGIC_FAIL,
            last_transition_label=reason,
        )
        self.store.append_timeline("port_model.fail", {"reason": reason})
        self._iter_rec.status = "failed"
        self._iter_rec.final_status = "stopped"
        self._iter_rec.failure_reason = reason
        self._iter_rec.ended_at = time.time()
        write_iteration(self.cfg.state_dir, self._iter_rec)
