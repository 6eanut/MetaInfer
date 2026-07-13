"""The TaskPlugin descriptor for gen-infer-framework."""

from ..base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="gen-infer-framework",
    name="Inference Framework Optimizer",
    description=(
        "6-phase iteration loop (plan→implement→test→review→perf→"
        "perf_plan). Optimizes inference serving code for a target "
        "model on target hardware."
    ),
    cli_module="metainfer.orchestrator.tasks.gen_infer_framework.cli",
    phases_module="metainfer.orchestrator.tasks.gen_infer_framework.phases",
)
