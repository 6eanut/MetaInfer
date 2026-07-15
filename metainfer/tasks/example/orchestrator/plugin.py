"""TaskPlugin — static metadata that tells the framework about this task type.

The launcher reads ``cli_module`` and runs::

    python -m <cli_module> run <requirements.json> --state-dir … --workspace-dir …

``phases_module`` points to the state-machine definition — used by the WebUI's
state-graph endpoint. Set to ``""`` if this task has no state graph.

``diagnostic_globs`` tells ``IterationWorkspace`` which diagnostic files to
copy forward from the previous iteration into the next one's ``prev-iter/``
directory. Empty tuple = no copy-forward.
"""

# --- Uncomment and customise for a real task ---

# from metainfer.orchestrator.tasks.base import TaskPlugin
#
# PLUGIN = TaskPlugin(
#     task_type="X-type-id",                        # unique; matches WebPlugin.type
#     cli_module="metainfer.tasks.X.orchestrator.cli",
#     phases_module="metainfer.tasks.X.orchestrator.phases",
#     diagnostic_globs=("*.jsonl", "*.log"),        # optional; see IterationWorkspace
# )
