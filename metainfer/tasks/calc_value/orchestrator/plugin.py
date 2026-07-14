"""The TaskPlugin descriptor for calc-theoretical-value."""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="calc-theoretical-value",
    name="Theoretical Value Calculator",
    description=(
        "Linear 5-step pipeline (rough→analyze→graph→calculate→visualize). "
        "Computes theoretical FLOPs / memory traffic for a model on "
        "target hardware."
    ),
    cli_module="metainfer.tasks.calc_value.orchestrator.cli",
    phases_module="metainfer.tasks.calc_value.orchestrator.phases",
)
