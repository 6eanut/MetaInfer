"""Inference-framework oracle.

Artifact contract (what the agent must produce in ``iter_dir/``):

    serve.sh PORT
        Bash script, executable. Starts the inference framework's HTTP
        server on the given port. Must:
        * expose an OpenAI-compatible API:
            POST /v1/chat/completions
            POST /v1/completions
            GET  /v1/models    (optional, used for health-check)
        * BLOCK in the foreground (no daemonize) — the oracle owns the
          process lifecycle.
        * honor deterministic decoding (temperature=0, fixed seed) where
          possible.

What this oracle does:

1. Picks a free localhost port.
2. Launches ``serve.sh PORT`` as a subprocess (in a new process group).
3. Polls ``GET /v1/models`` (falls back to ``POST /v1/chat/completions``
   with a trivial prompt) until the server is up or startup_timeout_s.
4. For each canned case in ``prompts.yaml``:
   * Sends an OpenAI-style request via :mod:`urllib` (stdlib, zero deps).
   * Captures status code, latency, response body.
5. Batches all (prompt, response) pairs into one judge sub-agent and parses
   the verdicts.
6. Aggregates: pass iff every case passes (with at least one judge_mode=llm).
7. Kills the server (SIGTERM → SIGKILL).
8. Writes ``oracle-report.json`` to ``report_dir``.

The oracle is **immutable** from the agent's perspective: it lives inside
the MetaInfer package, never inside ``iter_dir/``. Agents cannot edit it.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from ..base import Oracle, OracleCaseResult, OracleResult
from ..judge import JudgeInput, run_judge_batch


PROMPTS_FILE = Path(__file__).parent / "prompts.yaml"


class InferFrameworkOracle(Oracle):
    task_type = "gen-infer-framework"

    # ------------------------------------------------------------------ #
    # Public entry
    # ------------------------------------------------------------------ #

    def run(
        self,
        *,
        iter_dir: Path,
        req: Dict[str, Any],
        report_dir: Path,
        timeout_s: int = 600,
        manager=None,
    ) -> OracleResult:
        report_dir.mkdir(parents=True, exist_ok=True)
        serve_sh = iter_dir / "serve.sh"
        if not serve_sh.exists():
            return self._fail(report_dir, f"no serve.sh at {serve_sh}")

        port = _pick_free_port()
        cases_cfg = _load_cases(req)
        if not cases_cfg:
            return self._fail(report_dir, "no test cases configured")

        proc = None
        try:
            # Pass the real model path (if captured in requirements) to
            # serve.sh via env var so a weakly-written serve.sh still finds
            # the weights instead of falling into mock mode.
            model_dir = req.get("target_model") or (req.get("answers") or {}).get("target_model")
            proc = _start_server(serve_sh, port, report_dir, model_dir=model_dir)
            ok, err = _wait_healthy(port, startup_timeout_s=min(120, timeout_s))
            if not ok:
                return self._fail(report_dir, f"server did not become healthy: {err}")

            # Hit each case
            case_results_raw: List[Tuple[Dict[str, Any], str, Optional[int], float, Optional[str]]] = []
            t_start = time.time()
            for cfg in cases_cfg:
                resp, status, elapsed, err = _send_request(port, cfg, timeout_s=60)
                case_results_raw.append((cfg, resp, status, elapsed, err))
                if time.time() - t_start > timeout_s:
                    break

            # Build judge inputs (only judge cases that produced a response)
            judge_inputs: List[JudgeInput] = []
            preliminary: List[OracleCaseResult] = []
            for cfg, resp, status, elapsed, err in case_results_raw:
                if err is not None or status is None or status >= 500:
                    # infra-level failure — no point judging
                    preliminary.append(OracleCaseResult(
                        case_id=cfg["id"], prompt=cfg["prompt"], response=resp,
                        elapsed_s=elapsed, http_status=status, error=err,
                        judge_verdict="error",
                        judge_reason=f"http error: status={status} err={err}",
                    ))
                else:
                    preliminary.append(None)  # placeholder, filled below
                    judge_inputs.append(JudgeInput(
                        case_id=cfg["id"],
                        user_prompt=cfg["prompt"],
                        model_response=resp,
                        expected_keywords=cfg.get("expected_keywords") or [],
                    ))

            # Run judge sub-agent (if manager available)
            judged: List[OracleCaseResult] = []
            if manager is not None and judge_inputs:
                judged = run_judge_batch(
                    manager=manager, cases=judge_inputs,
                    workdir=report_dir, log_dir=report_dir,
                    timeout_s=max(120, min(300, timeout_s)),
                    judge_name="infer-framework-judge",
                )
                judge_mode = "llm"
            else:
                # Heuristic-only path (no manager passed — e.g. dry-run)
                from ..judge import heuristic_verdict
                judged = []
                for ji in judge_inputs:
                    v = heuristic_verdict(ji)
                    judged.append(OracleCaseResult(
                        case_id=ji.case_id, prompt=ji.user_prompt,
                        response=ji.model_response, elapsed_s=0.0,
                        judge_verdict=v["verdict"], judge_reason=v["reason"],
                    ))
                judge_mode = "heuristic"

            # Merge: replace placeholders with judged results
            final_cases: List[OracleCaseResult] = []
            ji_idx = 0
            for entry in preliminary:
                if entry is None:
                    # augment judged entry with http status + timing
                    j = judged[ji_idx]; ji_idx += 1
                    # find raw timing
                    raw = next((c for c in case_results_raw if c[0]["id"] == j.case_id), None)
                    if raw:
                        j.http_status = raw[2]
                        j.elapsed_s = raw[3]
                    final_cases.append(j)
                else:
                    final_cases.append(entry)

            # Aggregate
            total = len(final_cases)
            passed = sum(1 for c in final_cases if c.judge_verdict == "pass")
            failed_cases = [c for c in final_cases if c.judge_verdict != "pass"]
            all_passed = total > 0 and passed == total

            # Perf: average first-token-ish latency (rough proxy)
            avg_latency = (sum(c.elapsed_s for c in final_cases) / total) if total else 0.0
            perf = {
                "oracle_avg_http_latency_ms": round(avg_latency * 1000, 2),
                "oracle_cases_total": float(total),
                "oracle_cases_passed": float(passed),
            }

            reason = None
            if not all_passed:
                bits = [f"{c.case_id}={c.judge_verdict}" for c in failed_cases[:5]]
                reason = f"{len(failed_cases)}/{total} cases failed: {', '.join(bits)}"

            result = OracleResult(
                passed=all_passed,
                failure_reason=reason,
                perf=perf,
                cases=final_cases,
                notes=f"server on port {port}; judge_mode={judge_mode}",
                judge_mode=judge_mode,
                report_path=str(report_dir / "oracle-report.json"),
            )
            (report_dir / "oracle-report.json").write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result

        finally:
            if proc is not None:
                _kill_server(proc)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _fail(self, report_dir: Path, reason: str) -> OracleResult:
        result = OracleResult(
            passed=False, failure_reason=reason, judge_mode="disabled",
            report_path=str(report_dir / "oracle-report.json"),
        )
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "oracle-report.json").write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        return result


# --------------------------------------------------------------------------- #
# Server lifecycle
# --------------------------------------------------------------------------- #


def _pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_server(
    serve_sh: Path, port: int, report_dir: Path,
    model_dir: Optional[str] = None,
) -> subprocess.Popen:
    log_fp = open(report_dir / "server.stdout.log", "wb")
    err_fp = open(report_dir / "server.stderr.log", "wb")
    env = dict(os.environ)
    env["METAINFER_ORACLE_PORT"] = str(port)
    # Surface the real model path to the agent's serve.sh. Without this,
    # serve.sh has no way to know where the weights live (the oracle only
    # passes the port as $1), and any sane defensive fallback inside
    # serve.sh ends up serving mock responses — every C step then fails.
    if model_dir:
        env["MODEL_DIR"] = str(model_dir)
    # Best-effort determinism hints
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "8")
    return subprocess.Popen(
        ["bash", str(serve_sh), str(port)],
        stdout=log_fp, stderr=err_fp,
        cwd=str(serve_sh.parent),
        env=env, start_new_session=True,
    )


def _wait_healthy(port: int, startup_timeout_s: int) -> Tuple[bool, Optional[str]]:
    deadline = time.time() + startup_timeout_s
    url_models = f"http://127.0.0.1:{port}/v1/models"
    url_chat = f"http://127.0.0.1:{port}/v1/chat/completions"
    last_err: Optional[str] = None
    while time.time() < deadline:
        # Try /v1/models first (cheap)
        try:
            req = urllib.request.Request(url_models, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                if 200 <= resp.status < 500:
                    return True, None
        except urllib.error.HTTPError as e:
            # 404 / 405 means the server is up but doesn't implement /v1/models;
            # fall through to the chat-completions probe.
            if e.code in (404, 405, 401):
                if _probe_chat(url_chat):
                    return True, None
            last_err = f"HTTP {e.code}"
        except urllib.error.URLError as e:
            last_err = f"URLError: {e.reason!r}"
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0)
    return False, last_err


def _probe_chat(url: str) -> bool:
    body = json.dumps({
        "model": "probe", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1, "temperature": 0,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return 200 <= resp.status < 500
    except Exception:  # noqa: BLE001
        return False


def _kill_server(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        try:
            proc.terminate()
        except Exception:  # noqa: BLE001
            return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:  # noqa: BLE001
            try:
                proc.kill()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# HTTP request to the framework under test (stdlib only — no third-party deps)
# --------------------------------------------------------------------------- #


def _send_request(
    port: int, cfg: Dict[str, Any], timeout_s: int
) -> Tuple[str, Optional[int], float, Optional[str]]:
    """Send an OpenAI-style chat completion request. Returns
    (extracted_text, http_status, elapsed_s, error_or_None)."""
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": cfg.get("model", "default"),
        "messages": [{"role": "user", "content": cfg["prompt"]}],
        "max_tokens": int(cfg.get("max_tokens", 256)),
        "temperature": float(cfg.get("temperature", 0.0)),
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            elapsed = time.time() - t0
            text = _extract_text(raw)
            return text, resp.status, elapsed, None
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            err_body = ""
        return err_body, e.code, elapsed, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return "", None, time.time() - t0, f"URLError: {e.reason!r}"
    except Exception as e:  # noqa: BLE001
        return "", None, time.time() - t0, f"{type(e).__name__}: {e}"


def _extract_text(raw_body: str) -> str:
    """Extract the assistant message text from an OpenAI-format response.

    Tolerates minor shape variations (e.g. choices[0].message.content vs
    choices[0].text for /v1/completions-style).
    """
    try:
        obj = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body[:2000]
    choices = obj.get("choices") if isinstance(obj, dict) else None
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    msg = first.get("message")
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        return msg["content"]
    if isinstance(first.get("text"), str):
        return first["text"]
    return ""


# --------------------------------------------------------------------------- #
# Case loading
# --------------------------------------------------------------------------- #


def _load_cases(req: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Load canned cases from prompts.yaml, optionally overridden by user.

    A user can drop a custom ``prompts.yaml`` into
    ``<state_dir>/oracle/prompts.yaml`` to replace or extend the defaults.
    """
    cases: List[Dict[str, Any]] = []
    # 1. defaults
    if PROMPTS_FILE.exists():
        data = yaml.safe_load(PROMPTS_FILE.read_text(encoding="utf-8")) or []
        cases.extend([c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c])
    # 2. user overrides — look in req answers for a path, else skip
    custom = (req.get("answers") or {}).get("oracle_prompts_path")
    if custom:
        cp = Path(custom)
        if cp.exists():
            data = yaml.safe_load(cp.read_text(encoding="utf-8")) or []
            cases = [c for c in data if isinstance(c, dict) and "id" in c and "prompt" in c]
    return cases
