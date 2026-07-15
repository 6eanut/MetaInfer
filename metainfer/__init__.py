"""MetaInfer: deterministic orchestrator + lightweight WebUI for LLM
inference engineering.

Top-level subpackages:

- :mod:`metainfer.orchestrator` — the ABCDEF orchestrator that spawns
  Claude Code sub-agents. Runs as a per-task subprocess.
- :mod:`metainfer.server` — FastAPI WebUI backend (the main process).
- :mod:`metainfer.static` — frontend SPA (Preact, served by the WebUI).
"""

__version__ = "0.2.0"
