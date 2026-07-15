"""WebPlugin registration for this task type.

Every task type must have a WebPlugin that tells the framework:

- How to label this task in the WebUI picker.
- Which importmap key to use for the detail view.
- Where its frontend static files live.
- How to wire QA (even if it's minimal).
- (Optionally) any task-specific HTTP routes, extra stylesheets, SSE watch paths.
"""

from pathlib import Path

# --- Uncomment and customise for a real task ---

# from metainfer.server.registry import WebPlugin, register
#
# _FRONTEND_DIR = Path(__file__).resolve().parent.parent / "static"
#
# plugin = WebPlugin(
#     type="X-type-id",                   # MUST match TaskPlugin.task_type
#     label="Human-Readable Name",        # shown in task-type picker
#     description="What this task does.",
#     detail_view_module="app/X-detail",  # importmap key → static/X-detail.js
#     frontend_dir=_FRONTEND_DIR,
#     qa_config=XQAConfig(),              # see _qa.py; even minimal impl required
#     build_router=build_router,          # optional; X-specific HTTP endpoints
#     importmap_entries={},               # optional; auto-discovered normally
#     extra_stylesheets=["X.css"],        # optional
# )
# register(plugin)
