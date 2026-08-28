"""opt_operator task package.

Importing this package registers the orchestrator TaskPlugin (and, in Stage 4,
the web WebPlugin) via the auto-discovery in ``metainfer.tasks.__init__``.
"""

from __future__ import annotations

from .orchestrator import plugin as _task_plugin  # noqa: F401 — registers TaskPlugin
from .server import plugin as _web_plugin  # noqa: F401 — registers WebPlugin
