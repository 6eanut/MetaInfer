"""End-to-end: run the pipeline against a real file store, then read the web overview.

Closes the loop between the orchestrator (writes run.json + ledger + iterations +
system_oracle) and the WebUI readers (read_overview / read_lineage), using fake
backend + agent runner so no GPU/LLM/numpy is required.
"""

from __future__ import annotations

from pathlib import Path

from metainfer.orchestrator.iteration import IterationWorkspace
from metainfer.orchestrator.state import StateStore

from metainfer.tasks.opt_operator.orchestrator.build import BuildResult, kernel_digest
from metainfer.tasks.opt_operator.orchestrator.conformance import (
    CaseResult,
    ConformanceReport,
)
from metainfer.tasks.opt_operator.orchestrator.contract import OperatorContract
from metainfer.tasks.opt_operator.orchestrator.ledger import ChampionLedger
from metainfer.tasks.opt_operator.orchestrator.oracle import (
    freeze_reference,
    write_oracle_artifacts,
)
from metainfer.tasks.opt_operator.orchestrator.pipeline import Pipeline, PipelineConfig
from metainfer.tasks.opt_operator.orchestrator.profiler import PerfResult
from metainfer.tasks.opt_operator.server import _state_readers
from metainfer.tasks.opt_operator.tests._helpers import contract_dict


class _Backend:
    def __init__(self):
        self._i = 0
        self.curve = [100.0, 80.0]

    def build(self, source, language, contract, kernel_dir):
        return BuildResult(language=language, artifact=str(kernel_dir / "k"),
                           workspace_dir=kernel_dir, digest=kernel_digest(source, language))

    def conformance(self, contract, oracle, build, job_id):
        results = [CaseResult(c.id, True, 0.0, 0.0) for c in contract.generate_cases()]
        return ConformanceReport(True, results, contract.name)

    def profile(self, contract, build, job_id, reps):
        lat = self.curve[min(self._i, len(self.curve) - 1)]
        self._i += 1
        return {c.id: PerfResult(c.id, lat) for c in contract.generate_cases()}


def _runner(phase, tier, prompt, iter_dir, n):
    if phase == "A_plan":
        return {"approach": "tile", "done": False}
    if phase == "F_perf_plan":
        return {"next_plan": "ok", "done": n >= 1}
    return {"language": "triton", "source": f"// src {phase} {n}"}


def test_pipeline_to_overview(tmp_path):
    contract = OperatorContract.load(contract_dict(shapes={"B": 1, "S": 8, "H": 4}))
    state_dir = tmp_path / "state"
    store = StateStore(state_dir)
    store.init_or_resume("e2e")
    workspace = IterationWorkspace(tmp_path / "ws", tmp_path / "logs")
    oracle = freeze_reference("RMSNorm", contract, "def f(t): return t", "library")
    write_oracle_artifacts(state_dir / "system_oracle" / "run1", oracle)
    ledger = ChampionLedger(state_dir / "champion_ledger.jsonl")

    pipe = Pipeline(
        store=store, workspace=workspace, backend=_Backend(), agent_runner=_runner,
        ledger=ledger, contract=contract, oracle=oracle,
        initial_source="// src", initial_language="triton",
        cfg=PipelineConfig(max_iterations=3),
    )
    pipe.run()

    # Orchestrator artifacts are on disk and the web layer reads them.
    ov = _state_readers.read_overview(state_dir)
    assert ov["run"]["finished"] is True
    assert len(ov["lineage"]) == 2          # genesis + one promotion
    assert ov["summary"]["promotions"] == 1
    assert ov["reference"]["origin"] == "library"
    # Genesis best 100, champion 80 -> 1.25x
    assert ov["summary"]["speedup_vs_genesis"] == 1.25

    iters = _state_readers.read_iterations(state_dir)
    assert any(r["promoted"] for r in iters)
