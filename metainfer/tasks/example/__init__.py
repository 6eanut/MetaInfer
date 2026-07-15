"""Canonical example task package for MetaInfer.

This package is **documentation in code form**. It demonstrates the full
skeleton of a MetaInfer task type. To create a new task type:

1. Copy this entire directory::

       cp -r metainfer/tasks/example metainfer/tasks/<your_task_name>

2. Rename ``X`` / ``X-type-id`` / ``example`` to your task's name in every file.
3. Uncomment the ``register(...)`` calls in ``orchestrator/__init__.py`` and
   ``server/plugin.py``.
4. Implement the actual pipeline logic in ``orchestrator/pipeline.py``.
5. Write tests in ``tests/``.

Auto-discovery **will** import this package (``pkgutil.iter_modules`` loads
every non-underscore sibling), but since no ``register()`` calls are
active, it does nothing — no fake task type appears in the WebUI.
"""

# In a real task package, uncomment these to trigger registration:
# from .orchestrator import plugin as _task_plugin  # noqa: F401
# from .server import plugin as _web_plugin          # noqa: F401
