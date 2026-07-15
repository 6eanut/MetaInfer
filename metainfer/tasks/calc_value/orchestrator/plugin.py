"""The TaskPlugin descriptor for calc-theoretical-value."""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="calc-theoretical-value",
    cli_module="metainfer.tasks.calc_value.orchestrator.cli",
    phases_module="metainfer.tasks.calc_value.orchestrator.phases",
)
