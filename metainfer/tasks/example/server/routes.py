"""Task-specific HTTP routes.

Each route is a relative-path endpoint (no ``/api/{type}/{task_id}`` prefix).
The shell mounts this router at ``/api/{type}/{task_id}``, so the ``task_id``
path parameter is already carried by the mount prefix.

Use the helpers from :mod:`metainfer.server._helpers` to resolve on-disk state::

    from metainfer.server._helpers import (
        task_or_404, state_dir_for, workspace_dir_for, require_task_type,
    )

For QA support, call ``register_qa_routes`` — do NOT copy-paste QA routes::

    from metainfer.server.qa_routes import register_qa_routes
    register_qa_routes(router, plugin, prefix="/qa")
"""

from __future__ import annotations

from fastapi import APIRouter, Path as PathParam

from metainfer.server._helpers import task_or_404, state_dir_for


def build_router(plugin):
    """Return a router of X-specific endpoints.

    Called by ``create_app()`` which mounts the result at
    ``/api/{plugin.type}/{task_id}``.
    """
    router = APIRouter()

    @router.get("/iterations")
    def list_iterations(
        task_id: str = PathParam(..., description="Task ID"),
    ):
        entry = task_or_404(task_id)
        state_dir = state_dir_for(entry)
        # In a real task, call your _state_readers here:
        # from ._state_readers import read_iterations
        # return read_iterations(state_dir)
        return {"task_id": task_id, "state_dir": str(state_dir), "iterations": []}

    @router.get("/state-graph")
    def state_graph(
        task_id: str = PathParam(..., description="Task ID"),
    ):
        entry = task_or_404(task_id)
        state_dir = state_dir_for(entry)
        # from ._state_readers import read_state_graph
        # return read_state_graph(state_dir)
        return {"current": "idle", "nodes": [], "edges": []}

    # --- Uncomment to register QA routes ---
    # from metainfer.server.qa_routes import register_qa_routes
    # register_qa_routes(router, plugin, prefix="/qa")

    return router
