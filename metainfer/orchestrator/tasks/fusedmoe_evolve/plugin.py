"""The TaskPlugin descriptor for fusedmoe-evolve."""

from ..base import TaskPlugin


PLUGIN = TaskPlugin(
    task_type="fusedmoe-evolve",
    name="FusedMoE Kernel Evolver",
    description=(
        "4-phase iteration loop (prepare→evolve→validate→review). "
        "Uses OpenEvolve to iteratively optimize sglang's Triton fused "
        "MoE kernel for target hardware."
    ),
    cli_module="metainfer.orchestrator.tasks.fusedmoe_evolve.cli",
    phases_module="metainfer.orchestrator.tasks.fusedmoe_evolve.phases",
)
