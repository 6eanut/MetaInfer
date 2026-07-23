"""``metainfer-cluster`` CLI — wraps worker_registry, scoreboard, and mqueue operations.

Subcommands:
  workers ls                              — list registered workers
  workers show NODE_ID                    — one worker's detail
  scoreboard show                         — all GPU claims
  scoreboard force-release NODE GPU_IDX   — admin kill
  queue submit --worker N --script PATH   — submit a script job
  queue ls [--worker N]                   — list jobs in inbox(es)
  queue reset --worker N                  — wipe inbox
  tail STDOUT|STDERR WORKER JOB_ID        — read log
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="metainfer-cluster")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # workers
    p_workers = sub.add_parser("workers", help="worker registry ops")
    wsub = p_workers.add_subparsers(dest="wcmd", required=True)
    wsub.add_parser("ls")
    p_show = wsub.add_parser("show")
    p_show.add_argument("node_id")

    # scoreboard
    p_sb = sub.add_parser("scoreboard", help="GPU scoreboard ops")
    ssub = p_sb.add_subparsers(dest="scmd", required=True)
    ssub.add_parser("show")
    p_force = ssub.add_parser("force-release")
    p_force.add_argument("node_id")
    p_force.add_argument("gpu_idx", type=int)
    p_force.add_argument("--reason", default="cli-force-release")

    # queue
    p_q = sub.add_parser("queue", help="job queue ops")
    qsub = p_q.add_subparsers(dest="qcmd", required=True)
    p_submit = qsub.add_parser("submit")
    p_submit.add_argument("--worker", required=True)
    p_submit.add_argument("--script", required=True, help="path to script file")
    p_submit.add_argument("--timeout", type=float, default=1800.0)
    p_submit.add_argument("--gpu", type=int, action="append", default=[])
    p_submit.add_argument("--no-block", action="store_true")
    p_ls = qsub.add_parser("ls")
    p_ls.add_argument("--worker", default=None)
    p_reset = qsub.add_parser("reset")
    p_reset.add_argument("--worker", required=True)

    # tail
    p_tail = sub.add_parser("tail", help="tail a job's log")
    p_tail.add_argument("stream", choices=["stdout", "stderr"])
    p_tail.add_argument("worker")
    p_tail.add_argument("job_id")
    p_tail.add_argument("--offset", type=int, default=0)

    args = parser.parse_args(argv)
    # Import after parsing to keep --help fast.
    from metainfer.cluster import mqueue, scoreboard, worker_registry, sdk

    if args.cmd == "workers":
        if args.wcmd == "ls":
            for w in worker_registry.list_workers():
                alive = worker_registry.is_worker_alive(w.node_id)
                print(f"{w.node_id:20s}  {'alive' if alive else 'DEAD':6s}  "
                      f"ip={w.ip}  hostname={w.hostname}  "
                      f"gpus={len(w.gpu_topology)}")
        elif args.wcmd == "show":
            rec = worker_registry.read_worker(args.node_id)
            if rec is None:
                print(f"worker {args.node_id} not registered", file=sys.stderr)
                return 1
            print(json.dumps(rec.to_dict(), indent=2))
        return 0

    if args.cmd == "scoreboard":
        if args.scmd == "show":
            claims = scoreboard.list_claims()
            if not claims:
                print("(no GPU claims)")
                return 0
            for c in claims:
                print(f"{c['node_id']:20s}  gpu-{c['gpu_idx']}  "
                      f"holder={c['holder']:20s}  job={c['job_id'][:8]}  "
                      f"age={c['acquired_ago_s']:.1f}s  "
                      f"lease_left={c['lease_remaining_s']:.1f}s")
        elif args.scmd == "force-release":
            existed = scoreboard.force_release((args.node_id, args.gpu_idx),
                                                reason=args.reason)
            print(f"{args.node_id}/gpu-{args.gpu_idx}: "
                  f"{'released' if existed else 'was-not-held'}")
        return 0

    if args.cmd == "queue":
        if args.qcmd == "submit":
            script_body = Path(args.script).read_text()
            slots = [(args.worker, g) for g in args.gpu] if args.gpu else None
            job_id, result = sdk.submit_script(
                worker_node_id=args.worker,
                script_body=script_body,
                gpu_slots=slots,
                timeout_s=args.timeout,
                block=not args.no_block,
            )
            print(f"job_id={job_id}")
            if result is not None:
                print(f"status={result.status} exit_code={result.exit_code} "
                      f"duration={result.duration_s:.2f}s")
        elif args.qcmd == "ls":
            for j in mqueue.list_jobs(worker_node_id=args.worker):
                print(f"{j['job_id'][:12]}  worker={j['worker_node_id']:20s}  "
                      f"submitter={j['submitter']:20s}  status={j['status']}  "
                      f"age={j['submitted_ago_s']:.1f}s")
        elif args.qcmd == "reset":
            mqueue.reset_queue(args.worker)
            print(f"reset inbox for {args.worker}")
        return 0

    if args.cmd == "tail":
        if args.stream == "stdout":
            data = sdk.tail_stdout(args.job_id, args.worker, offset=args.offset)
        else:
            data = sdk.tail_stderr(args.job_id, args.worker, offset=args.offset)
        sys.stdout.buffer.write(data)
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
