"""The TaskPlugin descriptor for gen-infer-framework."""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="gen-infer-framework",
    cli_module="metainfer.tasks.gen_infer_framework.orchestrator.cli",
    phases_module="metainfer.tasks.gen_infer_framework.orchestrator.phases",
)
