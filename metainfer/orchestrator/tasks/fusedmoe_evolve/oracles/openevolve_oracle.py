"""OpenEvolve oracle for the B_evolve phase.

Shells out to openevolve-run.py with the iteration's prepared files.
Parses the output directory to extract the best evolved kernel.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from ....oracles.base import Oracle, OracleResult


class OpenEvolveOracle(Oracle):
    """Runs OpenEvolve as a subprocess on the prepared iteration files."""

    task_type = "fusedmoe-evolve"

    def run(
        self,
        *,
        iter_dir: Path,
        req: Dict[str, Any],
        report_dir: Path,
        timeout_s: int = 7200,
        openevolve_path: Optional[Path] = None,
        openevolve_iterations: Optional[int] = None,
        **kwargs,
    ) -> OracleResult:
        """Run OpenEvolve on the prepared files.

        1. Validate prerequisites (initial_program.py, evaluator.py, config.yaml)
        2. Shell out to openevolve-run.py
        3. Parse output to extract best evolved kernel
        4. Write oracle-report.json
        """
        report_dir.mkdir(parents=True, exist_ok=True)

        # Resolve paths
        initial_program = iter_dir / "initial_program.py"
        evaluator = iter_dir / "evaluator.py"
        config_yaml = iter_dir / "config.yaml"

        # Validate prerequisites
        missing = []
        for name, path in [
            ("initial_program.py", initial_program),
            ("evaluator.py", evaluator),
            ("config.yaml", config_yaml),
        ]:
            if not path.is_file():
                missing.append(name)

        if missing:
            return OracleResult(
                passed=False,
                failure_reason=(
                    f"Missing required files in {iter_dir}: {', '.join(missing)}. "
                    f"A_prepare must generate all three files."
                ),
            )

        # Resolve openevolve path
        if openevolve_path is None:
            openevolve_path = Path(
                os.environ.get(
                    "METAINFER_OPENEVOLVE_PATH",
                    "/home/jiakai/0716-fusedmoe-sglang/openevolve",
                )
            )
        openevolve_path = Path(openevolve_path)
        oe_runner = openevolve_path / "openevolve-run.py"
        if not oe_runner.is_file():
            return OracleResult(
                passed=False,
                failure_reason=(
                    f"openevolve-run.py not found at {oe_runner}. "
                    f"Check openevolve_path={openevolve_path}"
                ),
            )

        # Resolve iterations
        if openevolve_iterations is None:
            openevolve_iterations = _extract_oe_iterations(req, default=50)

        output_dir = iter_dir / "openevolve_output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build command
        cmd = [
            "python",
            str(oe_runner),
            str(initial_program),
            str(evaluator),
            "--config", str(config_yaml),
            "--iterations", str(openevolve_iterations),
            "--output", str(output_dir),
        ]

        # Log file for openevolve stdout/stderr
        oe_log_path = report_dir / "openevolve-output.log"

        # Run openevolve as subprocess
        try:
            with open(oe_log_path, "w", encoding="utf-8") as logf:
                logf.write(f"Command: {' '.join(cmd)}\n")
                logf.write(f"Working dir: {iter_dir}\n")
                logf.write(f"Timeout: {timeout_s}s\n")
                logf.write("=" * 60 + "\n")
                logf.flush()

                proc = subprocess.run(
                    cmd,
                    cwd=str(iter_dir),
                    env=os.environ.copy(),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                    text=True,
                )
        except subprocess.TimeoutExpired:
            return OracleResult(
                passed=False,
                failure_reason=(
                    f"OpenEvolve timed out after {timeout_s}s. "
                    f"Log at {oe_log_path}"
                ),
            )
        except Exception as exc:
            return OracleResult(
                passed=False,
                failure_reason=f"OpenEvolve subprocess error: {exc!r}",
            )

        if proc.returncode != 0:
            return OracleResult(
                passed=False,
                failure_reason=(
                    f"OpenEvolve exited with code {proc.returncode}. "
                    f"See log: {oe_log_path}"
                ),
            )

        # Parse output — find best program
        best_program_path = output_dir / "best_program.json"
        perf: Dict[str, float] = {
            "best_score": 0.0,
            "median_score": 0.0,
            "num_valid": 0,
            "total_generations": openevolve_iterations,
        }

        if best_program_path.is_file():
            try:
                best_data = json.loads(best_program_path.read_text(encoding="utf-8"))
                if isinstance(best_data, dict):
                    perf["best_score"] = float(best_data.get("score", 0))
                    source = best_data.get("source") or best_data.get("program_text") or ""
                    if source:
                        evolved_kernel_path = iter_dir / "evolved_kernel.py"
                        evolved_kernel_path.write_text(str(source), encoding="utf-8")
            except (json.JSONDecodeError, OSError, ValueError) as exc:
                return OracleResult(
                    passed=False,
                    failure_reason=f"Failed to parse best_program.json: {exc!r}",
                )
        else:
            # Try to find from checkpoints
            checkpoints_dir = output_dir / "checkpoints"
            if checkpoints_dir.is_dir():
                checkpoints = sorted(
                    [d for d in checkpoints_dir.iterdir() if d.is_dir()],
                    key=lambda p: p.name,
                )
                if checkpoints:
                    latest = checkpoints[-1]
                    snapshot = latest / "database_snapshot.json"
                    if snapshot.is_file():
                        try:
                            db = json.loads(snapshot.read_text(encoding="utf-8"))
                            programs = db.get("programs") or db.get("entries") or []
                            if programs:
                                best = max(programs, key=lambda p: float(p.get("score", -9999)))
                                perf["best_score"] = float(best.get("score", 0))
                                scores = [float(p.get("score", -9999)) for p in programs]
                                scores.sort()
                                mid = len(scores) // 2
                                perf["median_score"] = scores[mid] if scores else 0.0
                                perf["num_valid"] = sum(1 for s in scores if s > -999)
                                source = best.get("source") or best.get("program_text") or ""
                                if source:
                                    evolved_kernel_path = iter_dir / "evolved_kernel.py"
                                    evolved_kernel_path.write_text(str(source), encoding="utf-8")
                        except (json.JSONDecodeError, OSError, ValueError):
                            pass

        evolved_kernel = iter_dir / "evolved_kernel.py"
        if not evolved_kernel.is_file():
            return OracleResult(
                passed=False,
                failure_reason=(
                    f"OpenEvolve completed but no valid program was evolved. "
                    f"Output in {output_dir}"
                ),
                perf=perf,
            )

        # Write oracle report
        oracle_report = {
            "passed": True,
            "perf": perf,
            "openevolve_output_dir": str(output_dir),
            "evolved_kernel_path": str(evolved_kernel),
            "log_path": str(oe_log_path),
        }
        report_path = report_dir / "oracle-report.json"
        report_path.write_text(json.dumps(oracle_report, indent=2), encoding="utf-8")

        return OracleResult(
            passed=True,
            perf=perf,
            report_path=str(report_path),
        )


def _extract_oe_iterations(req: Dict[str, Any], default: int = 50) -> int:
    """Extract openevolve_iterations from requirements."""
    v = req.get("openevolve_iterations")
    if v is None:
        v = req.get("answers", {}).get("openevolve_iterations")
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
