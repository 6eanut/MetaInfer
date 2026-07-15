"""sys-shell task package.

The system shell is treated as a special task package — not a real task
type, but a plugin that owns the global WebUI chrome (shell router +
static frontend). Importing this package registers its ``WebPlugin``.
"""

from .web_server_handler import plugin  # noqa: F401 — registers WebPlugin
