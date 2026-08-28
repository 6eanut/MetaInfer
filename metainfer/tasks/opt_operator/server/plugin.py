"""WebPlugin registration for opt_operator."""

from pathlib import Path

from metainfer.server.registry import WebPlugin, register

from ._qa import CONFIG as _QA_CONFIG
from .routes import build_router

PLUGIN_TYPE = "opt-operator"

plugin = WebPlugin(
    type=PLUGIN_TYPE,
    label="Optimize operator (K100)",
    description=(
        "Contract-first, self-certifying operator optimization (HIP/Triton → K100) "
        "with append-only champion lineage and idle-dispatch multi-GPU."
    ),
    detail_view_module="app/opt-operator-detail",
    qa_config=_QA_CONFIG,
    build_router=build_router,
    frontend_dir=Path(__file__).resolve().parent.parent / "static",
    importmap_entries={},
    extra_stylesheets=["opt-operator.css"],
)

register(plugin)
