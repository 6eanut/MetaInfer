"""Oracle framework for MetaInfer correctness verification.

An **oracle** is an immutable, orchestrator-owned test harness that lives
OUTSIDE the agent's writable ``iter_dir/``. It defines "what counts as
correct" for a particular task type, independent of any code or test the
agent writes.

Rationale
---------
Letting the same agent that writes the code also write its own test makes
the agent both contestant and referee. The oracle pattern restores an
external reference: the agent produces a runnable artifact (e.g.
``serve.sh`` for an HTTP-served inference framework), the orchestrator
boots the oracle, the oracle probes the artifact against pre-written
inputs and an LLM judge, and returns a structured verdict.

Oracles are dispatched by ``task_type`` via :func:`metainfer.oracles.get_oracle`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class OracleCaseResult:
    case_id: str
    prompt: str
    response: str = ""
    judge_verdict: str = "skip"        # pass | fail | skip | error
    judge_reason: str = ""
    elapsed_s: float = 0.0
    http_status: Optional[int] = None
    error: Optional[str] = None
    # Whether this case gates the overall pass verdict. ``"soft"`` cases
    # are recorded + surfaced in failure_reason for visibility but do
    # NOT flip OracleResult.passed to False on their own. Use for cases
    # whose failure reflects a model-quality limitation (e.g. 8B model
    # can't reliably do arithmetic) rather than a code defect in the
    # framework under test.
    gating: str = "hard"              # hard | soft


@dataclass
class OracleResult:
    """Structured verdict returned by an oracle run."""

    passed: bool
    failure_reason: Optional[str] = None
    perf: Dict[str, float] = field(default_factory=dict)
    cases: List[OracleCaseResult] = field(default_factory=list)
    notes: str = ""
    # Whether the LLM judge was actually used (vs heuristic fallback)
    judge_mode: str = "llm"            # llm | heuristic | disabled
    report_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        hard_cases = [c for c in self.cases if c.gating != "soft"]
        soft_cases = [c for c in self.cases if c.gating == "soft"]
        return {
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "perf": self.perf,
            "cases": [c.__dict__ for c in self.cases],
            "notes": self.notes,
            "judge_mode": self.judge_mode,
            "report_path": self.report_path,
            "cases_total": len(self.cases),
            "cases_passed": sum(1 for c in self.cases if c.judge_verdict == "pass"),
            "cases_failed": sum(1 for c in self.cases if c.judge_verdict == "fail"),
            # Soft-gate visibility: hard failures gate the pass verdict;
            # soft failures are surfaced in the report but don't flip
            # passed=False. Useful for model-quality probes (math on an
            # 8B model, etc.) that aren't the framework's fault.
            "hard_total": len(hard_cases),
            "hard_passed": sum(1 for c in hard_cases if c.judge_verdict == "pass"),
            "hard_failed": sum(1 for c in hard_cases if c.judge_verdict != "pass"),
            "soft_total": len(soft_cases),
            "soft_passed": sum(1 for c in soft_cases if c.judge_verdict == "pass"),
            "soft_failed": sum(1 for c in soft_cases if c.judge_verdict != "pass"),
        }


class Oracle(ABC):
    """Abstract base class for task-type-specific oracles."""

    #: The ``task_type`` this oracle handles (must match the value in
    #: ``requirements.json``).
    task_type: str = ""

    @abstractmethod
    def run(
        self,
        *,
        iter_dir: Path,
        req: Dict[str, Any],
        report_dir: Path,
        timeout_s: int = 600,
    ) -> OracleResult:
        """Run the oracle against the artifact in ``iter_dir``.

        Implementations must:
        * be deterministic given the same inputs + seeds
        * write a full JSON report to ``report_dir / "oracle-report.json"``
        * never execute anything inside ``iter_dir`` other than the
          artifact the agent was told to produce
        """
        ...
