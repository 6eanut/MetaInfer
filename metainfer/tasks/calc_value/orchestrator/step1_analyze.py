"""Step 1: analyze model + framework code from 2 angles.

Spawns 2 parallel agents using SubAgentManager.launch_async, each
getting the same model_dir + framework_source_dir but a different
analysis strategy:

* agent_a: top-down from config.json
* agent_b: bottom-up from cmdline flags + env vars

Each agent Writes its structured findings to ``output.json`` in its
workdir; the loader (:func:`deterministic.load_agent_json`) reads that
file and only falls back to scraping the natural-language response if
the file is missing. The 2 outputs are merged into a consensus
memory.json (see :func:`deterministic.merge_memories`).

If the merge surfaces disputes on CRITICAL_FIELDS, the 2 agents are
re-prompted with the disputes and produce a new round. Max 3 rounds;
after that the merge (with disputes noted) is accepted.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from metainfer.orchestrator.subagent_manager import AgentSpec
from . import deterministic as det
from . import prompts as P


MAX_ROUNDS = 3
PER_AGENT_TIMEOUT_S = 1800  # 30 min per agent


def _format_env_block(env_vars: str) -> str:
    """Pretty-print the user's env-vars textarea input."""
    if not env_vars:
        return "(none)"
    return "\n".join(f"  {ln}" for ln in env_vars.splitlines() if ln.strip())


def _write_prompt(workdir: Path, name: str, text: str) -> Path:
    workdir.mkdir(parents=True, exist_ok=True)
    p = workdir / f"{name}.prompt.txt"
    p.write_text(text, encoding="utf-8")
    return p


def _build_agent_spec(
    *,
    name: str,
    workdir: Path,
    log_dir: Path,
    prompt_text: str,
) -> AgentSpec:
    prompt_file = _write_prompt(workdir, name, prompt_text)
    return AgentSpec(
        name=name,
        role="code_analyzer",
        prompt_file=prompt_file,
        workdir=workdir,
        log_dir=log_dir,
        timeout_s=PER_AGENT_TIMEOUT_S,
        stuck_timeout_s=900,
        max_retries=2,
    )


def _launch_round(
    *,
    manager,
    step1_dir: Path,
    round_idx: int,
    req: Dict[str, Any],
    prev_outputs: Optional[List[Dict[str, Any]]] = None,
    disputes: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Launch 2 parallel agents, return their parsed JSON outputs.

    On round 0 the 2 angle-prompts are used. On rounds >=1 the
    disagreement prompt is used (each agent gets the same disputes +
    its own previous output).
    """
    model_dir = req["model_dir"]
    framework_dir = req["framework_source_dir"]
    cmdline = req.get("cmdline_args") or "(none)"
    env_block = _format_env_block(req.get("env_vars") or "")

    common = {
        "readonly": P.READONLY_WARNING.format(
            model_dir=model_dir, framework_dir=framework_dir,
        ),
        "cmdline": cmdline,
        "env_block": env_block,
        "output_schema": P.STEP1_OUTPUT_SCHEMA,
    }

    round_dir = step1_dir / f"round_{round_idx:02d}"
    round_dir.mkdir(parents=True, exist_ok=True)

    if round_idx == 0:
        spec_builders = [
            ("agent_a", P.STEP1_AGENT_A_PROMPT),
            ("agent_b", P.STEP1_AGENT_B_PROMPT),
        ]
    else:
        # Disagreement round: both agents use the disagreement prompt,
        # but each gets ITS OWN previous output.
        if prev_outputs is None or disputes is None:
            raise ValueError("disagreement round requires prev_outputs + disputes")
        spec_builders = None  # handled below

    specs: List[AgentSpec] = []
    if round_idx == 0:
        for name, tmpl in spec_builders:
            text = tmpl.format(**common)
            workdir = round_dir / name
            log_dir = round_dir / "logs" / name
            specs.append(_build_agent_spec(
                name=name, workdir=workdir, log_dir=log_dir, prompt_text=text,
            ))
    else:
        for i, prev in enumerate(prev_outputs):
            name = f"agent_{chr(ord('a') + i)}"
            text = P.STEP1_DISAGREEMENT_PROMPT.format(
                **common,
                disputes=json.dumps(disputes, indent=2, ensure_ascii=False),
                your_prev=json.dumps(prev, indent=2, ensure_ascii=False),
            )
            workdir = round_dir / name
            log_dir = round_dir / "logs" / name
            specs.append(_build_agent_spec(
                name=name, workdir=workdir, log_dir=log_dir, prompt_text=text,
            ))

    # Launch both in parallel (max_concurrent on the manager gates).
    threads = []
    for spec in specs:
        t = manager.launch_async(spec)
        threads.append(t)
    for t in threads:
        t.join()

    # Parse each agent's response.
    outputs: List[Dict[str, Any]] = []
    for spec in specs:
        result = manager.result(spec.name)
        if result is None or not result.success:
            err = (result.error if result else "no result")
            # Record the failure but treat as empty memory — merge handles
            # missing agents gracefully (majority becomes 2/3 or 1/3).
            print(f"[calc-value.S1] agent {spec.name} failed: {err}", flush=True)
            outputs.append({})
            continue
        text = result.final_text or ""
        # response.txt = agent's natural-language narrative (human log).
        # The structured data lives in output.json which the agent was
        # told to Write; load_agent_json reads the file first and only
        # falls back to scraping response text if the file is missing.
        (spec.workdir / "response.txt").write_text(text, encoding="utf-8")
        parsed, source = det.load_agent_json(spec.workdir, "output.json", text)
        if not isinstance(parsed, dict):
            print(f"[calc-value.S1] agent {spec.name} returned no JSON "
                  f"(checked output.json + response.txt); treating as empty",
                  flush=True)
            (spec.workdir / "parse_error.txt").write_text(
                "No parseable JSON found in output.json or response.txt.\n"
                f"Response first 500 chars:\n{text[:500]}",
                encoding="utf-8",
            )
            outputs.append({})
            continue
        # Save the parsed memory (canonical structured artifact).
        (spec.workdir / "memory.json").write_text(
            json.dumps(parsed, indent=2, ensure_ascii=False), encoding="utf-8",
        )
        if source == "response":
            # Agent ignored the Write-file instruction but emitted JSON
            # inline — record so the UI can flag it.
            (spec.workdir / "source_note.txt").write_text(
                "JSON recovered from response.txt fallback "
                "(agent did not Write output.json as instructed).",
                encoding="utf-8",
            )
        outputs.append(parsed)

    return outputs


def run_step1_analyze(
    *,
    req: Dict[str, Any],
    store,
    manager,
    paths: Dict[str, Path],
) -> Path:
    """Run Step 1: 2-angle analysis with up to 3 rounds of reconciliation.

    Returns the path to ``step1/memory.json``.
    """
    step1_dir = paths["step1_dir"]
    store.append_timeline("calc_value.s1.agents.launched",
                          {"angles": ["a_topdown", "b_bottomup"]})

    prev_outputs: Optional[List[Dict[str, Any]]] = None
    prev_disputes: Optional[List[Dict[str, Any]]] = None
    last_merged: Optional[Dict[str, Any]] = None

    for round_idx in range(MAX_ROUNDS):
        t0 = time.time()
        outputs = _launch_round(
            manager=manager, step1_dir=step1_dir, round_idx=round_idx,
            req=req, prev_outputs=prev_outputs, disputes=prev_disputes,
        )
        elapsed = time.time() - t0
        store.append_timeline(
            "calc_value.s1.round.done",
            {"round": round_idx, "elapsed_s": round(elapsed, 1),
             "non_empty": sum(1 for o in outputs if o)},
        )

        merged, disputes = det.merge_memories(outputs)
        last_merged = merged
        # Save intermediate memory so the WebUI / debugging can see it.
        memory_path = step1_dir / f"memory.round_{round_idx:02d}.json"
        memory_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8",
        )

        # Reconcile only on CRITICAL disputes. Disputes on non-critical
        # fields (intermediate_size, etc.) are accepted as-is.
        critical_disputes = [
            d for d in disputes
            if d.get("field") in det.CRITICAL_FIELDS
        ]
        if not critical_disputes:
            store.append_timeline(
                "calc_value.s1.converged",
                {"round": round_idx, "critical_disputes": 0,
                 "non_critical_disputes": len(disputes)},
            )
            break
        if round_idx == MAX_ROUNDS - 1:
            store.append_timeline(
                "calc_value.s1.did_not_converge",
                {"round": round_idx,
                 "remaining_critical_disputes": len(critical_disputes),
                 "note": "accepting merge with disputes after max rounds"},
            )
            print(f"[calc-value.S1] failed to converge on "
                  f"{len(critical_disputes)} critical field(s) after "
                  f"{MAX_ROUNDS} rounds; accepting merge with disputes.",
                  flush=True)
            break
        print(f"[calc-value.S1] round {round_idx} produced "
              f"{len(critical_disputes)} critical dispute(s); retrying.",
              flush=True)
        prev_outputs = outputs
        prev_disputes = critical_disputes

    # Final memory.json (the last merged result).
    assert last_merged is not None
    final_path = step1_dir / "memory.json"
    final_path.write_text(
        json.dumps(last_merged, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    return final_path
