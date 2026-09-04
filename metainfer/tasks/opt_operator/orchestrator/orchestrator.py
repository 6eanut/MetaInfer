"""Orchestrator bootstrap — wire contract → oracle → pool → backend → pipeline.

Flow::

    requirements.json
      → contract (OperatorContract)                 [contract_from_req]
      → frozen oracle (ReferenceLibrary.resolve)     [build_oracle]
      → GPU pool + backend + ledger + agent runner
      → Pipeline.run()

Model tiering: the CLI supplies ``model_strong`` / ``model_cheap`` (falling back
to env ``METAINFER_MODEL_STRONG`` / ``METAINFER_MODEL_CHEAP``). These map onto the
pipeline's phase tiers: strong models plan/review/analyze, cheap models
implement/repair/write. When neither is given the orchestrator's ``--model``
default is used for every tier.

The GPU pool uses idle-dispatch: each conformance/perf task acquires whichever
slot is free and waits (with a deadline) when none is. No pre-sharding of the
case matrix.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from metainfer.orchestrator._bootstrap import (
    clear_pid_file,
    install_subagent_shutdown_handlers,
    make_subagent_manager,
    set_process_name,
    write_pid_file,
)
from metainfer.orchestrator.iteration import IterationWorkspace
from metainfer.orchestrator.requirements import req_field, req_field_int
from metainfer.orchestrator.state import StateStore
from metainfer.orchestrator.subagent_manager import AgentSpec

from .backend import RealBackend
from .contract import ContractError, OperatorContract
from .gpu_pool import GpuPool
from .guidance import GuidanceStore
from .ledger import ChampionLedger
from .oracle import FrozenOracle
from .pipeline import Pipeline, PipelineConfig
from .reference_lib import ReferenceLibrary


class OrchestratorError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# Contract resolution
# --------------------------------------------------------------------------- #

def _read_value(raw: Any, base_dir: Path) -> Any:
    """Field that may be an inline dict / YAML string / file path → its value."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return None
    # A multi-line value is almost certainly inline YAML, never a path.
    if "\n" not in text and len(text) < 512:
        candidates = [Path(text)]
        if not candidates[0].is_absolute():
            candidates.append(base_dir / text)
        for p in candidates:
            try:
                if p.is_file():
                    return p.read_text(encoding="utf-8")
            except OSError:
                break  # not a usable path (e.g. too long)
    return text  # inline yaml


def contract_from_req(req: Dict[str, Any], base_dir: Path) -> OperatorContract:
    """Build an OperatorContract from the flat requirements fields."""
    import yaml
    raw = _read_value(req_field(req, "operator_contract"), base_dir)
    if raw is None:
        raise OrchestratorError(
            "requirements.json is missing 'operator_contract' (inline YAML, file "
            "path, or a form-synthesized contract)")
    if isinstance(raw, dict):
        data = raw
    else:
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise OrchestratorError(f"operator_contract is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise OrchestratorError("operator_contract must resolve to a mapping")
    try:
        return OperatorContract.load(data)
    except ContractError as exc:
        raise OrchestratorError(f"bad operator_contract: {exc}") from exc


def kernel_source_from_req(req: Dict[str, Any], base_dir: Path):
    """Return ``(source, language)`` for the user-provided kernel (mode A), or
    ``(None, None)`` in spec mode (mode B)."""
    mode = req_field(req, "input_mode") or "spec"
    if mode != "source":
        return None, None
    src = _read_value(req_field(req, "kernel_source"), base_dir)
    language = req_field(req, "kernel_language")
    if not src:
        raise OrchestratorError("input_mode=source requires a kernel_source")
    if language not in ("hip", "triton"):
        raise OrchestratorError(f"kernel_language must be hip|triton, got {language!r}")
    return str(src), str(language)


# --------------------------------------------------------------------------- #
# Oracle resolution
# --------------------------------------------------------------------------- #

def build_oracle(
    contract: OperatorContract,
    executor,
    ref_lib: ReferenceLibrary,
    *,
    system_oracle_dir: Path,
    run_id: str,
    user_reference: Optional[str] = None,
    generated_reference: Optional[str] = None,
    baseline_language: Optional[str] = None,
    baseline_source: Optional[str] = None,
) -> FrozenOracle:
    return ref_lib.resolve(
        contract, executor,
        system_oracle_dir=system_oracle_dir,
        run_id=run_id,
        user_reference=user_reference,
        generated_reference=generated_reference,
        baseline_language=baseline_language,
        baseline_source=baseline_source,
    )


def _resolve_reference_dir(req: Dict[str, Any]) -> Path:
    env = os.environ.get("METAINFER_ROOT")
    base = Path(env) if env else Path.cwd()
    return base / "opt_operator" / "reference_library"


# --------------------------------------------------------------------------- #
# Task subdirs
# --------------------------------------------------------------------------- #

def _task_subdirs(state_dir: Path, workspace_dir: Path) -> Dict[str, Path]:
    state_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    for p in (state_dir / "logs", state_dir / "iterations",
              workspace_dir / "candidates"):
        p.mkdir(parents=True, exist_ok=True)
    return {
        "state_dir": state_dir,
        "workspace_dir": workspace_dir,
        "logs_root": state_dir / "logs",
        "iterations_state": state_dir / "iterations",
        "system_oracle_dir": state_dir / "system_oracle",
        # Authoritative append-only kernel pool (see OPT_KERNEL_SPEC §4). The
        # ChampionLedger is a derived lineage view over this file.
        "pool_path": state_dir / "kernel_pool.jsonl",
        "guidance_path": state_dir / "guidance.json",
        "pid_file": state_dir / "orchestrator.pid",
    }


def _model_map(req: Dict[str, Any], model_strong: Optional[str],
               model_cheap: Optional[str], default_model: Optional[str]) -> Dict[str, Optional[str]]:
    strong = model_strong or os.environ.get("METAINFER_MODEL_STRONG") or default_model
    cheap = model_cheap or os.environ.get("METAINFER_MODEL_CHEAP") or default_model
    return {"strong": strong, "cheap": cheap}


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def run_with_requirements(
    requirements_path: Path,
    *,
    state_dir: Optional[Path] = None,
    workspace_dir: Optional[Path] = None,
    claude_bin: str = "ccb",
    model: Optional[str] = None,
    permission_mode: str = "bypassPermissions",
    max_iterations: Optional[int] = None,
    extra_claude_args: Optional[list] = None,
    effort: str = "max",
    model_strong: Optional[str] = None,
    model_cheap: Optional[str] = None,
) -> int:
    if not requirements_path.exists():
        raise FileNotFoundError(f"requirements file not found: {requirements_path}")

    req: Dict[str, Any] = json.loads(requirements_path.read_text(encoding="utf-8"))
    task_id = req.get("task_id", "task")
    set_process_name("metainfer-orch")

    if state_dir is None:
        state_dir = Path.cwd() / "nodes" / "localhost" / ".metainfer" / "tasks" / task_id
    if workspace_dir is None:
        workspace_dir = Path.cwd() / "nodes" / "localhost" / "workspaces" / task_id
    paths = _task_subdirs(state_dir, workspace_dir)

    target_req = paths["state_dir"] / "requirements.json"
    if requirements_path.resolve() != target_req.resolve():
        target_req.write_text(requirements_path.read_text(encoding="utf-8"),
                              encoding="utf-8")

    write_pid_file(paths["pid_file"], task_id)

    # -- resolve the pieces (all injectable upstream; production env-bound) -- #
    contract = contract_from_req(req, workspace_dir)
    initial_source, initial_language = kernel_source_from_req(req, workspace_dir)

    executor = _make_executor(req)
    ref_lib = ReferenceLibrary(_resolve_reference_dir(req))
    oracle = build_oracle(
        contract, executor, ref_lib,
        system_oracle_dir=paths["system_oracle_dir"],
        run_id=task_id,
        user_reference=_read_value(req_field(req, "reference"), workspace_dir),
        generated_reference=_read_value(req_field(req, "generated_reference"), workspace_dir),
        baseline_language=initial_language or contract.language,
        baseline_source=initial_source,
    )

    store = StateStore(state_dir)
    run, is_resume = store.init_or_resume(task_id)
    workspace = IterationWorkspace(workspace_dir, paths["logs_root"])
    ledger = ChampionLedger(paths["pool_path"])
    pool = _make_pool(req, workspace_dir)
    backend = RealBackend(pool, executor, contract.language)
    model_map = _model_map(req, model_strong, model_cheap, model)

    manager = make_subagent_manager(
        claude_bin=claude_bin, model=model, permission_mode=permission_mode,
        effort=effort,
        extra_add_dirs=[workspace_dir, paths["system_oracle_dir"], paths["logs_root"]],
        snapshot_file=paths["state_dir"] / "agents.json",
    )
    runner = _make_production_runner(manager, model_map, req)
    guidance = GuidanceStore(paths["guidance_path"])

    cfg = PipelineConfig(
        max_iterations=max_iterations or req_field_int(req, "max_iterations", default=20),
        job_id=task_id,
    )
    pipe = Pipeline(
        store=store, workspace=workspace, backend=backend, agent_runner=runner,
        ledger=ledger, contract=contract, oracle=oracle,
        initial_source=initial_source, initial_language=initial_language,
        cfg=cfg,
    )

    print(f"[metainfer] task_id={task_id} contract={contract.name} "
          f"mode={'source' if initial_source else 'spec'} "
          f"oracle_origin={oracle.origin}")
    restore_signals = install_subagent_shutdown_handlers(manager, pid_file=paths["pid_file"])
    try:
        pipe.run(is_resume=is_resume)
    finally:
        restore_signals()
        clear_pid_file(paths["pid_file"])
    return 0


# --------------------------------------------------------------------------- #
# Production seam factories (env-bound; kept small so the loop stays testable)
# --------------------------------------------------------------------------- #

def _make_executor(req: Dict[str, Any]):
    # numpy-backed oracle executor (imported lazily by NumpyReferenceExecutor)
    from .oracle import NumpyReferenceExecutor
    return NumpyReferenceExecutor()


def _make_pool(req: Dict[str, Any], workspace_dir: Path) -> GpuPool:
    from .gpu_pool import default_discover, default_acquire, default_release
    return GpuPool(
        node_id=os.environ.get("METAINFER_NODE_ID", os.uname().nodename),
        holder="opt_operator",
        discover=default_discover,
        acquire=default_acquire,
        release=default_release,
        slot_deadline_s=req_field_int(req, "slot_deadline_s", default=0.5),
    )


def _make_production_runner(manager, model_map: Dict[str, Optional[str]], req):
    """AgentRunner wrapping SubAgentManager; maps tier → AgentSpec.model.

    Agent time budgets are read from the requirements (``agent_timeout_s`` /
    ``agent_stuck_timeout_s``) so a run can give an implementer enough wall-clock
    and no-output grace to author + GPU-probe a kernel without being killed
    mid-flight. Defaults stay conservative for other uses.
    """
    from .pipeline import AgentRunner
    agent_timeout_s = req_field_int(req, "agent_timeout_s", default=600)
    stuck_timeout_s = req_field_int(req, "agent_stuck_timeout_s", default=120)

    def runner(phase: str, tier: str, prompt: str, iter_dir: Path,
               n: int) -> Dict[str, Any]:
        model = model_map.get(tier)
        name = f"{phase}-it{n}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        prompt_file = iter_dir / f"{name}.prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        spec = AgentSpec(name=name, role=phase, prompt_file=prompt_file,
                         workdir=iter_dir, log_dir=iter_dir,
                         timeout_s=agent_timeout_s,
                         stuck_timeout_s=stuck_timeout_s, model=model)
        manager.launch(spec)
        result = manager.result(name)
        if result is None or not result.final_text:
            return {"error": "agent produced no output"}
        return _extract_json(result.final_text)

    return runner


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort: pull the first balanced JSON object out of agent output."""
    import json as _json
    start = text.find("{")
    if start == -1:
        return {"raw": text.strip()}
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return _json.loads(text[start:i + 1])
                except _json.JSONDecodeError:
                    return {"raw": text[start:i + 1]}
    return {"raw": text.strip()}


__all__ = ["OrchestratorError", "contract_from_req", "kernel_source_from_req",
           "build_oracle", "run_with_requirements", "_extract_json", "_model_map"]
