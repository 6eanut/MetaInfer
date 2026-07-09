#!/usr/bin/env python3
"""Deterministic validator for the execution-flow graph (Phase 3.2).

Two layers:

(a) Structural checks (pure Python, no LLM):
    - all node ids unique
    - all preds/succs/edge endpoints reference existing nodes
    - no orphan nodes (a node with empty preds AND empty succs that is not
      declared as a root or sink)
    - every dim var used in any shape is defined in dim_vars
    - every node has framework_file and either perf or an explicit
      perf_source="unmeasured"
    - every node has a unique id and a non-empty step_purpose

(b) Per-node semantic checks (LLM, but one node at a time):
    For each node, we build a prompt containing ONLY that node + its immediate
    neighbors. The prompt is written to validation_prompts/node_<id>.txt.
    Verdicts are read back from validation_verdicts/node_<id>.json.

Execution modes:

  --mode structural   : only run (a). Always safe, always fast.
  --mode emit-prompts : run (a) then emit per-node prompts (no LLM call). The
                        orchestrator then dispatches them to parallel Agents.
  --mode collect      : read previously-written verdict files, summarize,
                        write validation_report.md. Apply proposed patches
                        when --apply-patches is also set.
  --mode claude-cli   : run (a), emit prompts, then shell out to `claude -p`
                        per node in parallel (requires the Claude Code CLI in
                        PATH), then collect. Convenience for end-to-end runs.
  --mode full         : structural + claude-cli + collect (default if --claude-cli).

This script NEVER sends the whole graph to an LLM. Each prompt contains at
most one node plus its immediate predecessor/successor neighbors.

Usage:
  python validate_graph.py <graph.json> --root <run_dir> [--mode ...] [--round N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from typing import Any


# ---------- structural checks ----------------------------------------------

def check_structure(graph: dict, framework_src: str | None = None) -> tuple[bool, list[str]]:
    errors: list[str] = []
    warns: list[str] = []

    if not isinstance(graph, dict):
        return False, ["graph is not a JSON object"]
    nodes = graph.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        return False, ["graph.nodes missing or empty"]

    ids = [n.get("id") for n in nodes]
    seen = set()
    for i in ids:
        if i is None:
            errors.append("a node is missing 'id'")
            continue
        if i in seen:
            errors.append(f"duplicate node id: {i}")
        seen.add(i)

    node_map = {n.get("id"): n for n in nodes if isinstance(n, dict)}

    # edges
    edges = graph.get("edges", [])
    for e in edges:
        if not isinstance(e, dict):
            errors.append(f"edge is not an object: {e!r}")
            continue
        fr, to = e.get("from"), e.get("to")
        if fr not in node_map:
            errors.append(f"edge from unknown node: {fr}")
        if to not in node_map:
            errors.append(f"edge to unknown node: {to}")

    # preds/succs consistency
    declared_succs = {nid: set() for nid in node_map}
    declared_preds = {nid: set() for nid in node_map}
    for n in nodes:
        nid = n.get("id")
        for s in n.get("succs", []) or []:
            declared_succs[nid].add(s)
            if s not in node_map:
                errors.append(f"node {nid}.succs references unknown {s}")
        for p in n.get("preds", []) or []:
            declared_preds[nid].add(p)
            if p not in node_map:
                errors.append(f"node {nid}.preds references unknown {p}")

    # orphan check
    declared_roots = set(graph.get("declared_roots", []))
    declared_sinks = set(graph.get("declared_sinks", []))
    for nid, n in node_map.items():
        has_pred = bool(declared_preds[nid] or n.get("preds"))
        has_succ = bool(declared_succs[nid] or n.get("succs"))
        if not has_pred and not has_succ:
            if nid not in declared_roots and nid not in declared_sinks:
                errors.append(f"orphan node {nid} (no preds/succs, not declared root/sink)")

    # dim_vars consistency
    dim_vars = set((graph.get("dim_vars") or {}).keys())
    used_vars: set[str] = set()
    for n in nodes:
        for grp in ("inputs", "outputs"):
            for t in n.get(grp, []) or []:
                shape = (t or {}).get("shape", []) if isinstance(t, dict) else []
                for d in shape:
                    if isinstance(d, str):
                        used_vars.add(d)
    undef = used_vars - dim_vars
    for v in undef:
        errors.append(f"undefined dim var used: {v}")

    # required per-node fields
    for nid, n in node_map.items():
        if not n.get("step_purpose"):
            errors.append(f"node {nid} missing step_purpose")
        if not n.get("operator"):
            errors.append(f"node {nid} missing operator")
        if not n.get("framework_file"):
            errors.append(f"node {nid} missing framework_file")
        if not n.get("perf") and n.get("perf_source") != "unmeasured":
            warns.append(f"node {nid} has no perf data and no perf_source=unmeasured")

    # framework_refs requirement + on-disk existence verification
    refs_warned_global = (framework_src is None)
    if refs_warned_global:
        warns.append("--framework-src not provided; skipping source-reference existence check")

    def _check_path(nid: str, role: str, file_str: str, line: Any) -> None:
        # resolve relative to framework_src
        candidates = []
        if os.path.isabs(file_str):
            candidates.append(file_str)
        if framework_src:
            candidates.append(os.path.join(framework_src, file_str))
        else:
            candidates.append(file_str)
        found = next((c for c in candidates if os.path.isfile(c)), None)
        if found is None:
            errors.append(f"node {nid} ref[{role}] file does not exist: {file_str}")
            return
        if line is not None:
            try:
                ln = int(line)
            except (TypeError, ValueError):
                errors.append(f"node {nid} ref[{role}] bad line: {line!r}")
                return
            with open(found, "rb") as f:
                nl = sum(1 for _ in f)
            if nl < ln:
                errors.append(f"node {nid} ref[{role}] line {ln} out of range "
                              f"(file has {nl} lines): {file_str}")

    for nid, n in node_map.items():
        refs = n.get("framework_refs")
        if not isinstance(refs, list) or not refs:
            errors.append(f"node {nid} missing non-empty framework_refs (need at least call_site)")
            continue
        roles = [r.get("role") for r in refs if isinstance(r, dict)]
        if "call_site" not in roles:
            errors.append(f"node {nid} framework_refs missing required 'call_site' role")
        seen_roles = set()
        for r in refs:
            if not isinstance(r, dict):
                errors.append(f"node {nid} framework_refs has non-object entry: {r!r}")
                continue
            role = r.get("role") or "<unnamed>"
            if role in seen_roles:
                # duplicates of role are tolerated but warned
                warns.append(f"node {nid} framework_refs duplicate role: {role}")
            seen_roles.add(role)
            fpath = r.get("file")
            if not fpath:
                errors.append(f"node {nid} ref[{role}] missing 'file'")
                continue
            if framework_src:
                _check_path(nid, role, fpath, r.get("line"))

    ok = not errors
    return ok, errors + ["[warn] " + w for w in warns]


# ---------- prompt emission ------------------------------------------------

PROMPT_HEADER = """You are auditing ONE node of an LLM inference execution-flow graph.
You will receive:
  1. This node's JSON (id, step_purpose, operator, framework_file, inputs, outputs, perf, preds, succs).
  2. The JSON of its immediate predecessor and successor nodes (only those, NOT the whole graph).
  3. Pointers to read-only memory files and the framework source tree.

Your job: decide whether this node's description is faithful to the framework source
and consistent with the recorded performance data. DO NOT propose speculative
refactors. Only flag concrete factual inconsistencies.

Return STRICT JSON only, on a single line, no prose:
  {{"verdict": "ok" | "fix", "reason": "<short>", "proposed_patch": {{"<field>": <new_value>}}}}

`proposed_patch` may be {{}} when verdict is "ok". When verdict is "fix", it MUST
contain at least one field path that exists on this node (top-level field, or
dotted for nested, e.g. "perf.mean_us").
"""


def _neighbor_json(graph: dict, node: dict) -> str:
    node_map = {n["id"]: n for n in graph.get("nodes", []) if isinstance(n, dict)}
    preds = [node_map[p] for p in (node.get("preds") or []) if p in node_map]
    succs = [node_map[s] for s in (node.get("succs") or []) if s in node_map]
    return json.dumps({"preds": preds, "succs": succs}, indent=2)


def emit_prompts(graph: dict, run_dir: str, phase1_md: str, phase2_md: str,
                 framework_src: str, log_file: str | None) -> str:
    pdir = os.path.join(run_dir, "validation_prompts")
    os.makedirs(pdir, exist_ok=True)
    for n in graph.get("nodes", []):
        nid = n.get("id")
        body = {
            "node": n,
            "neighbors": _neighbor_json(graph, n),
        }
        prompt = []
        prompt.append(PROMPT_HEADER)
        prompt.append("")
        prompt.append(f"# Framework source tree (read-only): {framework_src}")
        if log_file:
            prompt.append(f"# Run log (read-only): {log_file}")
        prompt.append(f"# Phase 1 code map (read-only): {phase1_md}")
        prompt.append(f"# Phase 2 tracing map (read-only): {phase2_md}")
        prompt.append("")
        prompt.append("## This node:")
        prompt.append(json.dumps(n, indent=2))
        prompt.append("")
        prompt.append("## Immediate neighbors only:")
        prompt.append(_neighbor_json(graph, n))
        prompt.append("")
        prompt.append("## Instructions")
        prompt.append("- Read ONLY the file:line cited in `framework_file` (and surrounding ~40 lines).")
        prompt.append("- Cross-check operator + shapes against that code.")
        prompt.append("- Cross-check perf.source_kernel_ids against Phase 2 memory.")
        prompt.append("- Cross-check step_purpose against the Phase 1 call-site table.")
        prompt.append("- Output STRICT JSON, one line. Nothing else.")
        with open(os.path.join(pdir, f"node_{nid}.txt"), "w") as f:
            f.write("\n".join(prompt))
    return pdir


# ---------- claude-cli runner ----------------------------------------------

def _run_one_claude(prompt_path: str, model: str | None) -> str:
    cmd = ["claude", "-p", "--output-format", "text"]
    if model:
        cmd += ["--model", model]
    with open(prompt_path) as f:
        proc = subprocess.run(cmd, stdin=f, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        return json.dumps({"verdict": "fix",
                           "reason": f"claude CLI failed: rc={proc.returncode} err={proc.stderr[:200]}",
                           "proposed_patch": {}})
    out = proc.stdout.strip()
    # try to pull the last {...} object
    start = out.rfind("{")
    end = out.rfind("}")
    if start >= 0 and end > start:
        candidate = out[start:end + 1]
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass
    return json.dumps({"verdict": "fix",
                       "reason": f"unparseable LLM output: {out[:200]}",
                       "proposed_patch": {}})


def run_claude_cli(prompts_dir: str, verdicts_dir: str, workers: int,
                   model: str | None) -> None:
    os.makedirs(verdicts_dir, exist_ok=True)
    files = sorted(os.listdir(prompts_dir))
    prompt_paths = [os.path.join(prompts_dir, f) for f in files if f.endswith(".txt")]
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(lambda p: (os.path.basename(p), _run_one_claude(p, model)), prompt_paths))
    for name, body in results:
        with open(os.path.join(verdicts_dir, name.replace(".txt", ".json")), "w") as f:
            f.write(body)


# ---------- collect --------------------------------------------------------

def _apply_patch(node: dict, patch: dict) -> list[str]:
    changes = []
    for k, v in patch.items():
        parts = k.split(".")
        cur = node
        for p in parts[:-1]:
            if not isinstance(cur, dict) or p not in cur:
                cur = None
                break
            cur = cur[p]
        if isinstance(cur, dict):
            cur[parts[-1]] = v
            changes.append(f"{k}={v!r}")
    return changes


def collect_verdicts(graph_path: str, run_dir: str, apply_patches: bool) -> tuple[bool, str]:
    vdir = os.path.join(run_dir, "validation_verdicts")
    if not os.path.isdir(vdir):
        return True, "no verdicts directory; nothing to collect"
    with open(graph_path) as f:
        graph = json.load(f)
    node_map = {n["id"]: n for n in graph.get("nodes", [])}
    report_lines = []
    any_fix = False
    fixes_applied = 0
    for fn in sorted(os.listdir(vdir)):
        if not fn.endswith(".json"):
            continue
        nid = fn[len("node_"):-len(".json")]
        with open(os.path.join(vdir, fn)) as f:
            try:
                v = json.load(f)
            except Exception as e:
                report_lines.append(f"- {nid}: UNPARSEABLE ({e})")
                any_fix = True
                continue
        verdict = v.get("verdict", "?")
        reason = v.get("reason", "")
        patch = v.get("proposed_patch", {}) or {}
        if verdict == "ok":
            report_lines.append(f"- {nid}: ok")
        else:
            any_fix = True
            applied = []
            if apply_patches and nid in node_map and patch:
                applied = _apply_patch(node_map[nid], patch)
                fixes_applied += len(applied)
            report_lines.append(f"- {nid}: FIX ({reason}); applied={applied}")
    if apply_patches and fixes_applied:
        with open(graph_path, "w") as f:
            json.dump(graph, f, indent=2)
    report = "# Validation report\n\n"
    report += f"any_fix_needed={any_fix}; patches_applied={fixes_applied}\n\n## Per-node\n\n"
    report += "\n".join(report_lines)
    with open(os.path.join(run_dir, "validation_report.md"), "w") as f:
        f.write(report)
    return (not any_fix), report


# ---------- main ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_json")
    ap.add_argument("--root", required=True, help="run directory (contains memories/, graph/)")
    ap.add_argument("--mode", default="structural",
                    choices=["structural", "emit-prompts", "collect", "claude-cli", "full"])
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--framework-src", default=None)
    ap.add_argument("--phase1-md", default=None)
    ap.add_argument("--phase2-md", default=None)
    ap.add_argument("--log-file", default=None)
    ap.add_argument("--apply-patches", action="store_true")
    ap.add_argument("--claude-model", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    with open(args.graph_json) as f:
        graph = json.load(f)

    ok, msgs = check_structure(graph, framework_src=args.framework_src)
    print(f"[validate] structural: ok={ok}", file=sys.stderr)
    for m in msgs:
        print(f"  {m}", file=sys.stderr)
    if not ok:
        with open(os.path.join(args.root, "validation_report.md"), "w") as f:
            f.write("# Validation report\n\nstructural FAIL\n\n" + "\n".join(msgs))
        return 1

    if args.mode == "structural":
        return 0

    # emit prompts
    phase1 = args.phase1_md or os.path.join(args.root, "memories", "phase1_code_map.md")
    phase2 = args.phase2_md or os.path.join(args.root, "memories", "phase2_tracing_map.md")
    fw = args.framework_src or "<not provided>"
    emit_prompts(graph, args.root, phase1, phase2, fw, args.log_file)
    print(f"[validate] emitted per-node prompts to {args.root}/validation_prompts/", file=sys.stderr)

    if args.mode == "emit-prompts":
        return 0

    if args.mode in ("claude-cli", "full"):
        prompts_dir = os.path.join(args.root, "validation_prompts")
        verdicts_dir = os.path.join(args.root, "validation_verdicts")
        # clear old verdicts
        os.makedirs(verdicts_dir, exist_ok=True)
        for f in os.listdir(verdicts_dir):
            os.remove(os.path.join(verdicts_dir, f))
        run_claude_cli(prompts_dir, verdicts_dir, args.workers, args.claude_model)
        # then collect
        ok2, _ = collect_verdicts(args.graph_json, args.root, args.apply_patches)
        return 0 if ok2 else 2

    if args.mode == "collect":
        ok2, _ = collect_verdicts(args.graph_json, args.root, args.apply_patches)
        return 0 if ok2 else 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
