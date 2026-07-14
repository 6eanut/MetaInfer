"""Task package auto-discovery.

Each sibling directory under this package is one task type's self-contained
package, with the standard layout::

    metainfer/tasks/<task_pkg>/
    ├── __init__.py
    ├── orchestrator/          # task pipeline (cli, pipeline, plugin, ...)
    ├── web_server_handler/    # routes / readers / qa / WebPlugin registration
    ├── static/                # task-specific ES modules
    ├── tests/                 # all tests for this task
    └── form.yaml              # task-creation form schema (read by web/forms.py)

Each task package's ``__init__.py`` is responsible for importing its own
``orchestrator.plugin`` and ``web_server_handler.plugin`` submodules so
their ``register(...)`` calls fire. This module just walks ``__path__``
and imports every non-underscore-prefixed sibling package — that's the
single discovery point for both the orchestrator TaskPlugin registry
and the web WebPlugin registry.

Adding a new task type: drop a package under ``metainfer/tasks/<name>/``
with the layout above. No edits to any other file.
"""

from __future__ import annotations

import importlib
import pkgutil

# Iterate siblings and import each. Import errors propagate (fail-fast)
# rather than being silently swallowed — a broken task package should
# not be a silent no-op.
for _finder, _name, _is_pkg in pkgutil.iter_modules(__path__):
    if _name.startswith("_"):
        continue
    importlib.import_module(f"{__name__}.{_name}")
