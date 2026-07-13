"""The TaskPlugin descriptor for calc-theoretical-value."""

from ..base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="calc-theoretical-value",
    name="Theoretical Value Calculator",
    description=(
        "Linear 4-step pipeline (analyze→graph→calculate→visualize). "
        "Computes theoretical FLOPs / memory traffic for a model on "
        "target hardware."
    ),
    cli_module="metainfer.orchestrator.tasks.calc_value.cli",
    phases_module="metainfer.orchestrator.tasks.calc_value.phases",
)
