"""Worker daemon package.

Entry point: ``python -m metainfer.worker`` (see ``__main__.py``).

Responsibilities:
- Register with the worker registry on startup (writes ``cluster/workers/<id>.json``)
- Touch heartbeat every 15s while alive
- Poll ``cluster/inbox/<self>/<job_id>/`` for new jobs
- For each job: spawn subprocess (script or ccb agent) with ``CUDA_VISIBLE_DEVICES``
  set per the job's GPU slots on this node; stream stdout/stderr; enforce timeout;
  write result back to ``replies/<orchestrator>/<job_id>.result.json``
"""
