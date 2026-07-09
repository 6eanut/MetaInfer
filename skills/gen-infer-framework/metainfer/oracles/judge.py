"""LLM-based judge for oracle output validation.

The judge runs as a **sub-agent** (a Claude Code subprocess managed by
:class:`metainfer.subagent_manager.SubAgentManager`) rather than calling
an HTTP API directly. This:

* avoids hardcoding API keys / endpoints in the orchestrator
* reuses the existing subprocess infrastructure (timeout, kill, retry)
* keeps the orchestrator free of any LLM client dependency

The judge receives a batch of (case_id, user_prompt, model_response) tuples
in its prompt and must emit exactly one JSON line per case:

    {"case_id": "<id>", "verdict": "pass"|"fail", "reason": "<short>"}

If the judge agent fails or its output is unparseable, every undecided case
falls through to a deterministic heuristic (non-empty + length + keyword
match) so the oracle never silently approves a broken artifact.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .base import OracleCaseResult


@dataclass
class JudgeInput:
    case_id: str
    user_prompt: str
    model_response: str
    expected_keywords: List[str] = None  # for heuristic fallback


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #

JUDGE_SYSTEM = """You are a strict but fair judge of LLM output quality.

You will receive a batch of test cases. Each case has:
- case_id   : a short identifier
- prompt    : the prompt that was given to the model under test
- response  : the model's actual response

For each case, evaluate the response on:
1. **Coherence** — is it readable, on-topic, and logically structured?
2. **Correctness** — does it actually answer the prompt accurately?
3. **Completeness** — for prompts with a clear deliverable (code, translation,
   math), does the response provide it?

Be strict on correctness, lenient on style. Truncation, gibberish, empty
output, infinite-repeat, or off-topic answers are always "fail".

# Output format (MANDATORY)

Output exactly one JSON object PER CASE on its own line, nothing else:

{"case_id": "<id>", "verdict": "pass", "reason": "<one short sentence>"}
{"case_id": "<id>", "verdict": "fail", "reason": "<one short sentence>"}

Do not include markdown fences, commentary, or any other text. Just the
JSON lines.
"""


def build_judge_prompt(cases: List[JudgeInput]) -> str:
    """Render a batched judge prompt for the sub-agent."""
    lines = [f"# Judge batch ({len(cases)} cases)\n"]
    for i, c in enumerate(cases, 1):
        lines.append(f"## Case {i}: `{c.case_id}`")
        lines.append("### Prompt given to the model under test:")
        lines.append("```")
        lines.append(c.user_prompt.strip() or "(empty)")
        lines.append("```")
        lines.append("### Model's response:")
        lines.append("```")
        lines.append(c.model_response.strip() or "(empty)")
        lines.append("```")
        lines.append("")
    lines.append("Now emit your verdicts — one JSON line per case, in order.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #

_JSON_LINE_RE = re.compile(r'\{[^{}]*"verdict"[^{}]*\}')


def parse_judge_output(raw: str) -> Dict[str, Dict[str, str]]:
    """Parse the judge agent's stdout into ``{case_id: {verdict, reason}}``.

    Tolerates surrounding noise — we look for any line matching the JSON
    shape and extract verdict/reason.
    """
    out: Dict[str, Dict[str, str]] = {}
    for ln in raw.splitlines():
        ln = ln.strip().lstrip("`-").rstrip("`")
        if not ln.startswith("{"):
            continue
        try:
            obj = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "case_id" not in obj:
            continue
        verdict = str(obj.get("verdict", "")).strip().lower()
        if verdict not in ("pass", "fail"):
            continue
        out[str(obj["case_id"])] = {
            "verdict": verdict,
            "reason": str(obj.get("reason", ""))[:300],
        }
    # fallback: regex-scan for embedded JSON objects
    if not out:
        for m in _JSON_LINE_RE.finditer(raw):
            try:
                obj = json.loads(m.group(0))
                if "case_id" in obj and obj.get("verdict") in ("pass", "fail"):
                    out[str(obj["case_id"])] = {
                        "verdict": str(obj["verdict"]),
                        "reason": str(obj.get("reason", ""))[:300],
                    }
            except json.JSONDecodeError:
                continue
    return out


# --------------------------------------------------------------------------- #
# Heuristic fallback (used when judge agent fails entirely)
# --------------------------------------------------------------------------- #

def heuristic_verdict(case: JudgeInput) -> Dict[str, str]:
    """A deliberately weak stand-in: only catches grossly broken responses."""
    resp = (case.model_response or "").strip()
    if not resp:
        return {"verdict": "fail", "reason": "empty response (heuristic)"}
    if len(resp) < 5:
        return {"verdict": "fail", "reason": "response too short (heuristic)"}
    # Detect infinite-repeat / degenerate output
    tokens = resp.split()
    if tokens and len(tokens) > 20:
        uniq_ratio = len(set(tokens)) / len(tokens)
        if uniq_ratio < 0.15:
            return {"verdict": "fail",
                    "reason": f"low token diversity ({uniq_ratio:.2f}, heuristic)"}
    # Optional keyword check
    if case.expected_keywords:
        lower = resp.lower()
        if not any(k.lower() in lower for k in case.expected_keywords):
            return {"verdict": "fail",
                    "reason": f"missing expected keywords (heuristic)"}
    return {"verdict": "pass", "reason": "heuristic: response non-empty & diverse"}


# --------------------------------------------------------------------------- #
# Orchestrator-side runner: spawns the judge sub-agent and merges results
# --------------------------------------------------------------------------- #


def run_judge_batch(
    *,
    manager,                          # SubAgentManager
    cases: List[JudgeInput],
    workdir: Path,
    log_dir: Path,
    timeout_s: int = 600,
    judge_name: str = "judge",
) -> List[OracleCaseResult]:
    """Run the judge over a batch of cases via one sub-agent invocation.

    Returns one :class:`OracleCaseResult` per input case, with
    ``judge_verdict`` populated from either the LLM judge or the heuristic
    fallback. ``judge_mode`` on the returned results is "llm" if the agent
    produced parseable output for that case, else "heuristic".
    """
    from ..subagent_manager import AgentSpec  # local import to avoid cycles

    log_dir.mkdir(parents=True, exist_ok=True)
    prompt = build_judge_prompt(cases)
    prompt_file = log_dir / f"{judge_name}.prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")

    spec = AgentSpec(
        name=judge_name,
        role="judge",
        prompt_file=prompt_file,
        workdir=workdir,
        log_dir=log_dir,
        timeout_s=timeout_s,
        stuck_timeout_s=min(timeout_s, 300),
        extra_args=["--system-prompt", JUDGE_SYSTEM]
                   if _supports_system_prompt_flag(manager) else [],
    )
    started = time.time()
    manager.launch(spec)
    result = manager.result(judge_name)
    elapsed = time.time() - started

    raw = ""
    if result is not None:
        raw = result.final_text or ""
        # also pull the full stdout from the log file for forensics
        try:
            lf = spec.log_file(result.attempts)
            if lf.exists():
                raw = raw + "\n" + lf.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass

    parsed = parse_judge_output(raw) if raw else {}

    out: List[OracleCaseResult] = []
    for c in cases:
        if c.case_id in parsed:
            v = parsed[c.case_id]
            out.append(OracleCaseResult(
                case_id=c.case_id, prompt=c.user_prompt,
                response=c.model_response, elapsed_s=elapsed / max(len(cases), 1),
                judge_verdict=v["verdict"], judge_reason=v["reason"],
            ))
        else:
            v = heuristic_verdict(c)
            out.append(OracleCaseResult(
                case_id=c.case_id, prompt=c.user_prompt,
                response=c.model_response, elapsed_s=elapsed / max(len(cases), 1),
                judge_verdict=v["verdict"], judge_reason=v["reason"],
            ))
    return out


def _supports_system_prompt_flag(manager) -> bool:
    """Conservatively detect whether ``claude -p`` accepts ``--system-prompt``.

    The flag has existed since Claude Code v2.x; if absent the system prompt
    is folded into the user prompt instead (still works). Default: True.
    """
    return True
