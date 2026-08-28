"""TaskPlugin — static metadata for the opt_operator task type.

The launcher reads ``cli_module`` and runs::

    python -m <cli_module> run <requirements.json> --state-dir … --workspace-dir …
"""

from metainfer.orchestrator.tasks.base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="opt-operator",
    cli_module="metainfer.tasks.opt_operator.orchestrator.cli",
    phases_module="metainfer.tasks.opt_operator.orchestrator.phases",
    diagnostic_globs=("*.prompt.txt", "*.log", "*.jsonl"),
)
