"""The TaskPlugin descriptor for gen-infer-framework."""

from metainfer.orchestrator.tasks.base import TaskPlugin


# Diagnostic files copied forward from iteration N-1's logs dir into
# iteration N's logs/prev-iter/ subdir at open time. Without this, the
# next agent has no visibility into WHY the previous C step failed —
# the failure_reason text alone is rarely enough to debug a server
# crash or a judge verdict. These names are gf-specific (oracle /
# judge / test logs); tasks with different diagnostic vocabularies
# declare their own globs.
DIAGNOSTIC_GLOBS = (
    "oracle-report.json",
    "server.stdout.log",
    "server.stderr.log",
    "*-test.log",
    "test.log",
    "judge.*",
    "*.prompt.txt",
)


PLUGIN = TaskPlugin(
    task_type="gen-infer-framework",
    cli_module="metainfer.tasks.gen_infer_framework.orchestrator.cli",
    phases_module="metainfer.tasks.gen_infer_framework.orchestrator.phases",
    diagnostic_globs=DIAGNOSTIC_GLOBS,
)
