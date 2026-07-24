"""CLI entry: ``python -m metainfer.worker``.

Runs the worker daemon. Honors ``METAINFER_NODE_ID`` and ``METAINFER_ROOT``
env vars (same as orchestrator).

Usage:
    python -m metainfer.worker --node-id workerA --metainfer-root /shared/metainfer
    METAINFER_NODE_ID=workerA python -m metainfer.worker
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

from metainfer.cluster.worker_registry import WorkerIdentityConflict
from metainfer.worker.daemon import WorkerConfig, WorkerDaemon


def main(argv: list[str] | None = None) -> int:
    """``python -m metainfer.worker`` entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(prog="metainfer.worker", description=__doc__)
    parser.add_argument("--node-id", default=os.environ.get("METAINFER_NODE_ID") or socket.gethostname(),
                        help="Worker node identifier (defaults to $METAINFER_NODE_ID or hostname)")
    parser.add_argument("--metainfer-root", default=os.environ.get("METAINFER_ROOT"),
                        help="Shared filesystem root (defaults to $METAINFER_ROOT)")
    parser.add_argument("--ip", default=None, help="Override auto-detected IP")
    parser.add_argument("--hostname", default=None, help="Override auto-detected hostname")
    parser.add_argument("--mac", default=None, help="Override auto-detected MAC")
    parser.add_argument("--max-concurrent-jobs", type=int, default=1,
                        help="Max in-flight jobs (default: 1)")
    args = parser.parse_args(argv)

    cfg = WorkerConfig(
        node_id=args.node_id,
        metainfer_root=args.metainfer_root,
        ip=args.ip,
        hostname=args.hostname,
        mac=args.mac,
        max_concurrent_jobs=args.max_concurrent_jobs,
    )
    daemon = WorkerDaemon(cfg)
    try:
        daemon.run_forever()
    except KeyboardInterrupt:
        daemon.stop()
        return 0
    except WorkerIdentityConflict as e:
        # Fatal: this daemon's METAINFER_NODE_ID conflicts with an existing
        # worker record on a different physical box. Refuse to enter the main
        # loop — silently continuing would either (a) clobber the legit record
        # (now blocked at the registry layer) or (b) consume jobs intended for
        # the real worker. Surface loudly and exit non-zero so systemd / the
        # operator sees the failure.
        print(f"[metainfer.worker] FATAL: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
