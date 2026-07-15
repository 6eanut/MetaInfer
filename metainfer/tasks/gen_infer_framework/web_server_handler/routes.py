"""FastAPI route handlers for the gen-infer-framework task type.

Currently the only routes gf needs are the offline-QA triplet, and
those are mounted via the generic :func:`metainfer.web.qa_routes.register_qa_routes`
helper — backed by the task-agnostic QA engine in
:mod:`metainfer.web.qa` and gf's own qa_config pathsolver. If/when
gf-specific routes are needed, add them here and wire from
``plugin.py::register_routes``.
"""

from __future__ import annotations

from fastapi import FastAPI

from metainfer.web.qa_routes import register_qa_routes


def register_routes(app: FastAPI, deps, plugin) -> None:
    """Mount gf's routes onto ``app``.

    Today: just the generic QA routes at the standard ``/qa`` prefix.
    """
    register_qa_routes(app, plugin, prefix="/qa")
