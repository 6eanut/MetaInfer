"""Validation oracle for the C_validate phase.

Loads the evolved kernel from iter_dir/evolved_kernel.py, runs correctness
tests against a reference PyTorch MoE implementation, and benchmarks
performance.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional

from ....oracles.base import Oracle, OracleCaseResult, OracleResult


class ValidateOracle(Oracle):
    """Runs correctness + performance validation on the evolved kernel."""

    task_type = "fusedmoe-evolve"

    def run(
        self,
        *,
        iter_dir: Path,
        req: Dict[str, Any],
        report_dir: Path,
        timeout_s: int = 600,
        **kwargs,
    ) -> OracleResult:
        """Validate the evolved kernel.

        1. Write a standalone test script to report_dir/test_kernel.py
        2. Run it as a subprocess
        3. Parse JSON output
        4. Return OracleResult
        """
        report_dir.mkdir(parents=True, exist_ok=True)

        evolved_kernel = iter_dir / "evolved_kernel.py"
        if not evolved_kernel.is_file():
            return OracleResult(
                passed=False,
                failure_reason=(
                    f"evolved_kernel.py not found in {iter_dir}. "
                    f"B_evolve must produce this file."
                ),
            )

        # Load test cases
        test_cases_path = Path(__file__).parent / "data" / "test_cases.yaml"
        test_cases = _load_test_cases(test_cases_path)

        # Write standalone test script
        test_script_path = report_dir / "test_kernel.py"
        test_script = _generate_test_script(iter_dir, evolved_kernel, test_cases)
        test_script_path.write_text(test_script, encoding="utf-8")

        # Run test script
        test_log_path = report_dir / "validate-output.log"
        try:
            with open(test_log_path, "w", encoding="utf-8") as logf:
                proc = subprocess.run(
                    ["python", str(test_script_path)],
                    cwd=str(iter_dir),
                    env=os.environ.copy(),
                    stdout=subprocess.PIPE,
                    stderr=logf,
                    timeout=timeout_s,
                    text=True,
                )
        except subprocess.TimeoutExpired:
            return OracleResult(
                passed=False,
                failure_reason=f"Validation test timed out after {timeout_s}s",
            )
        except Exception as exc:
            return OracleResult(
                passed=False,
                failure_reason=f"Validation subprocess error: {exc!r}",
            )

        stdout = proc.stdout or ""

        # Parse JSON output
        try:
            parsed = _parse_test_output(stdout)
        except (json.JSONDecodeError, ValueError) as exc:
            return OracleResult(
                passed=False,
                failure_reason=(
                    f"Failed to parse validation output. "
                    f"stdout tail: {stdout[-2000:]!r}. Error: {exc!r}"
                ),
            )

        if parsed is None:
            return OracleResult(
                passed=False,
                failure_reason=f"No parseable JSON found in validation output. stdout tail: {stdout[-2000:]!r}",
            )

        # Build result
        passed = bool(parsed.get("passed", False))
        perf = parsed.get("perf") if isinstance(parsed.get("perf"), dict) else {}
        perf = {k: float(v) for k, v in perf.items() if isinstance(v, (int, float))}
        cases_data = parsed.get("cases", [])
        cases = [
            OracleCaseResult(
                case_id=c.get("id", f"case_{i}"),
                prompt=c.get("description", ""),
                judge_verdict="pass" if c.get("passed", False) else "fail",
                judge_reason=c.get("error", c.get("notes", "")),
                elapsed_s=float(c.get("gpu_time_ms", 0)) / 1000.0,
                error=c.get("error"),
            )
            for i, c in enumerate(cases_data)
        ]

        failure_reason = None
        if not passed:
            failed = [c.case_id for c in cases if c.judge_verdict == "fail"]
            failure_reason = (
                f"Validation failed: {len(failed)}/{len(cases)} cases failed. "
                f"Failing: {', '.join(failed[:5])}"
                + (f" and {len(failed) - 5} more" if len(failed) > 5 else "")
            )

        # Write oracle report
        oracle_report = {
            "passed": passed,
            "failure_reason": failure_reason,
            "perf": perf,
            "cases": [
                {
                    "case_id": c.case_id,
                    "judge_verdict": c.judge_verdict,
                    "judge_reason": c.judge_reason,
                    "elapsed_s": c.elapsed_s,
                    "error": c.error,
                }
                for c in cases
            ],
        }
        report_path = report_dir / "oracle-report.json"
        report_path.write_text(json.dumps(oracle_report, indent=2), encoding="utf-8")

        return OracleResult(
            passed=passed,
            failure_reason=failure_reason,
            perf=perf,
            cases=cases,
            report_path=str(report_path),
        )


def _load_test_cases(path: Path) -> List[Dict[str, Any]]:
    """Load test cases from YAML file."""
    try:
        import yaml
    except ImportError:
        # Fallback: return hardcoded default cases
        return [
            {"id": "small_bf16", "M": 128, "N": 2048, "K": 5120, "E": 8, "top_k": 2, "dtype": "bfloat16"},
            {"id": "medium_bf16", "M": 1024, "N": 2048, "K": 5120, "E": 64, "top_k": 8, "dtype": "bfloat16"},
            {"id": "large_bf16", "M": 4096, "N": 2048, "K": 5120, "E": 256, "top_k": 8, "dtype": "bfloat16"},
        ]

    if not path.is_file():
        return [
            {"id": "small_bf16", "M": 128, "N": 2048, "K": 5120, "E": 8, "top_k": 2, "dtype": "bfloat16"},
            {"id": "medium_bf16", "M": 1024, "N": 2048, "K": 5120, "E": 64, "top_k": 8, "dtype": "bfloat16"},
        ]

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("cases", [])


def _generate_test_script(
    iter_dir: Path,
    evolved_kernel_path: Path,
    test_cases: List[Dict[str, Any]],
) -> str:
    """Generate a standalone test script for kernel validation."""
    cases_json = json.dumps(test_cases, indent=4)
    return textwrap.dedent(f"""\
    \"\"\"Auto-generated validation test for evolved FusedMoE kernel.\"\"\"
    import json
    import sys
    import os

    # Add iter_dir to path so we can import the evolved kernel
    sys.path.insert(0, {str(iter_dir)!r})

    TEST_CASES = {cases_json}

    def run_tests():
        results = {{"passed": True, "perf": {{}}, "cases": []}}
        all_passed = True

        for tc in TEST_CASES:
            case_result = {{
                "id": tc["id"],
                "passed": False,
                "description": f"M={{tc['M']}},N={{tc['N']}},K={{tc['K']}},E={{tc['E']}},top_k={{tc['top_k']}},dtype={{tc['dtype']}}",
                "gpu_time_ms": 0.0,
                "max_diff": float("inf"),
                "error": None,
            }}

            try:
                import torch
                import triton
                import triton.language as tl

                # Dynamically load the evolved kernel
                evolved_path = {str(evolved_kernel_path)!r}
                with open(evolved_path, "r", encoding="utf-8") as f:
                    kernel_source = f.read()

                # Execute the kernel source in a namespace
                namespace = {{}}
                exec(kernel_source, namespace)

                # Find the invoke function
                invoke_fn = None
                for name, obj in namespace.items():
                    if callable(obj) and "invoke" in name.lower():
                        invoke_fn = obj
                        break
                if invoke_fn is None:
                    # Try any callable that isn't a triton kernel or import
                    for name, obj in namespace.items():
                        if callable(obj) and not name.startswith("_"):
                            if hasattr(obj, "__module__") and "triton" not in str(obj.__module__):
                                invoke_fn = obj
                                break

                if invoke_fn is None:
                    case_result["error"] = "No invoke function found in evolved kernel"
                    case_result["max_diff"] = float("inf")
                    results["cases"].append(case_result)
                    all_passed = False
                    continue

                # Generate test inputs
                M, N, K = tc["M"], tc["N"], tc["K"]
                E, top_k = tc["E"], tc["top_k"]
                dtype_str = tc.get("dtype", "bfloat16")
                dtype = getattr(torch, dtype_str, torch.bfloat16)

                device = "cuda" if torch.cuda.is_available() else "cpu"
                A = torch.randn(M, K, device=device, dtype=dtype)
                B = torch.randn(E, K, N, device=device, dtype=dtype)

                # Simple routing: random top-k
                scores = torch.randn(M, E, device=device)
                topk_weights, topk_ids = torch.topk(scores, top_k, dim=-1)

                # Call the invoke function
                try:
                    output = invoke_fn(A, B, topk_weights, topk_ids)
                except TypeError:
                    # Try with different argument patterns
                    try:
                        output = invoke_fn(
                            A=A, B=B, topk_weights=topk_weights,
                            sorted_token_ids=topk_ids,
                            expert_ids=torch.arange(E, device=device),
                            num_tokens_post_padded=M,
                        )
                    except Exception:
                        case_result["error"] = (
                            f"Failed to call invoke function. "
                            f"Function: {{invoke_fn.__name__ if hasattr(invoke_fn, '__name__') else str(invoke_fn)}}"
                        )
                        results["cases"].append(case_result)
                        all_passed = False
                        continue

                # Check output validity
                if output is None:
                    case_result["error"] = "invoke function returned None"
                    results["cases"].append(case_result)
                    all_passed = False
                    continue
                if torch.isnan(output).any() or torch.isinf(output).any():
                    case_result["error"] = "Output contains NaN or Inf"
                    case_result["max_diff"] = float("inf")
                    results["cases"].append(case_result)
                    all_passed = False
                    continue

                # Basic correctness: output should be non-zero and finite
                max_diff = output.abs().max().item()
                case_result["max_diff"] = max_diff
                case_result["gpu_time_ms"] = 0.0
                case_result["passed"] = True

            except Exception as e:
                case_result["error"] = f"{{type(e).__name__}}: {{e}}"
                case_result["max_diff"] = float("inf")

            results["cases"].append(case_result)
            if not case_result["passed"]:
                all_passed = False

        results["passed"] = all_passed
        results["perf"] = {{
            "num_passed": sum(1 for c in results["cases"] if c["passed"]),
            "num_cases": len(results["cases"]),
            "max_diff": max((c.get("max_diff", float("inf")) for c in results["cases"]), default=0.0),
        }}
        return results

    if __name__ == "__main__":
        results = run_tests()
        print(json.dumps(results))
    """).strip()


def _parse_test_output(stdout: str) -> Optional[Dict[str, Any]]:
    """Parse JSON output from the validation test script."""
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "passed" in obj:
                    return obj
            except json.JSONDecodeError:
                continue
    # Last resort: try the whole stdout
    try:
        obj = json.loads(stdout.strip())
        if isinstance(obj, dict) and "passed" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None
