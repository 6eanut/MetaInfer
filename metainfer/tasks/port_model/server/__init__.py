"""port-model web server handler.

FastAPI routes + JSON readers + QA pathsolver + the WebPlugin
descriptor that mounts them into the WebUI. ``plugin`` is imported
lazily from ``metainfer.tasks.port_model.__init__`` which triggers
``server.plugin`` (and its ``register(plugin)`` side-effect) exactly
once.
"""
