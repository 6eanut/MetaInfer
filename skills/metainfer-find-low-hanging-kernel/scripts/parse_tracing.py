#!/usr/bin/env python3
"""Chrome tracing parser for metainfer-find-low-hanging-kernel.

Reads a Chrome tracing file (plain JSON, .json.gz, or arbitrary .gz containing
JSON) and emits compact, downstream-friendly artifacts. Designed to be safe to
run on files that are hundreds of MB — it never holds the full event list in
memory twice and it streams gunzip.

Outputs (all under --out-dir, default cwd):
  parsed_events.jsonl.gz   one compact JSON record per event
  kernel_stats.json        per-name aggregate stats + histogram
  cpu_stack_summary.json   CPU stack frames if present (correlation aid)
  cuda_graph_meta.json     CUDA Graph markers if detected
  parser_report.json       high-level summary (counts, ranges, flags detected)

Usage:
  python parse_tracing.py <trace_file> [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from typing import Any


# ---------- input openers ---------------------------------------------------

def _open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "rt", encoding="utf-8", errors="replace")


def _load_trace_object(path: str) -> dict | list:
    """Load the tracing file as a single JSON value. Chrome traces are either
    {"traceEvents": [...]} or a bare [...] (sometimes one JSON object per line,
    "JSON-with-exceptions"). We handle all three."""
    fp = _open_maybe_gzip(path)
    try:
        # skip leading whitespace (some traces start with \n)
        head = fp.read(1)
        while head and head.isspace():
            head = fp.read(1)
        fp.seek(0)
        if head == "[":
            # could be a single huge array
            return json.load(fp)
        if head == "{":
            # could be {"traceEvents": [...]} (single object) OR a stream of
            # objects separated by whitespace (the "JSON-with-exceptions" form
            # that chrome://tracing can emit). Try object first.
            try:
                return json.load(fp)
            except json.JSONDecodeError:
                fp.seek(0)
                events = []
                decoder = json.JSONDecoder()
                buf = fp.read()
                i = 0
                n = len(buf)
                while i < n:
                    while i < n and buf[i].isspace():
                        i += 1
                    if i >= n:
                        break
                    obj, end = decoder.raw_decode(buf, i)
                    if isinstance(obj, list):
                        events.extend(obj)
                    elif isinstance(obj, dict):
                        if "traceEvents" in obj and isinstance(obj["traceEvents"], list):
                            events.extend(obj["traceEvents"])
                        else:
                            events.append(obj)
                    i = end
                return {"traceEvents": events}
        # fallback: line-delimited JSON
        events = []
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return {"traceEvents": events}
    finally:
        fp.close()


def _iter_events(trace: dict | list):
    if isinstance(trace, list):
        yield from trace
        return
    evs = trace.get("traceEvents") if isinstance(trace, dict) else None
    if evs:
        yield from evs
        return
    # some traces put metadata at top level alongside traceEvents
    if isinstance(trace, dict):
        for k, v in trace.items():
            if k == "traceEvents":
                continue
            if isinstance(v, list):
                # unknown top-level list — skip, we only want events
                continue


# ---------- stats helpers ---------------------------------------------------

def _percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_vals[int(k)])
    return float(sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f))


def _histogram(vals: list[float], bins: int = 16) -> list[dict]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return [{"lo": lo, "hi": hi, "count": len(vals)}]
    width = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        idx = int((v - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return [
        {"lo": round(lo + i * width, 3), "hi": round(lo + (i + 1) * width, 3), "count": c}
        for i, c in enumerate(counts)
    ]


# ---------- main ------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_file")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument(
        "--min-dur-us",
        type=float,
        default=0.0,
        help="drop events shorter than this (complete events only)",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[parse_tracing] loading {args.trace_file} ...", file=sys.stderr)
    trace = _load_trace_object(args.trace_file)
    print(f"[parse_tracing] loaded, iterating events ...", file=sys.stderr)

    # per-event output (compressed)
    events_path = os.path.join(args.out_dir, "parsed_events.jsonl.gz")
    # accumulators
    by_name_durs: dict[str, list[float]] = defaultdict(list)
    by_name_cats: dict[str, set] = defaultdict(set)
    by_name_args: dict[str, list[dict]] = defaultdict(list)
    by_name_tids: dict[str, set] = defaultdict(set)
    by_name_pids: dict[str, set] = defaultdict(set)

    cpu_stacks: list[dict] = []
    cuda_graph_markers: list[dict] = []
    total_events = 0
    complete_count = 0
    first_ts = None
    last_ts = None

    with gzip.open(events_path, "wt", encoding="utf-8") as evout:
        for ev in _iter_events(trace):
            if not isinstance(ev, dict):
                continue
            total_events += 1
            ph = ev.get("ph")
            name = ev.get("name", "<unnamed>")
            cat = ev.get("cat", "")
            ts = ev.get("ts")
            dur = ev.get("dur")
            pid = ev.get("pid")
            tid = ev.get("tid")
            eargs = ev.get("args") or {}

            if first_ts is None and isinstance(ts, (int, float)):
                first_ts = ts
            if isinstance(ts, (int, float)) and (last_ts is None or ts > last_ts):
                last_ts = ts

            # compact event record
            rec = {"name": name, "cat": cat, "ph": ph, "pid": pid, "tid": tid, "ts": ts}
            if dur is not None:
                rec["dur"] = dur
            if eargs:
                # keep args small — drop huge nested blobs
                small_args = {}
                for k, v in eargs.items():
                    if isinstance(v, (int, float, str, bool)) or v is None:
                        small_args[k] = v
                    elif isinstance(v, list) and len(v) <= 16:
                        small_args[k] = v
                    elif isinstance(v, dict) and len(v) <= 16:
                        small_args[k] = v
                    else:
                        small_args[k] = f"<{type(v).__name__} len={len(v) if hasattr(v, '__len__') else '?'}>"
                rec["args"] = small_args
            evout.write(json.dumps(rec, separators=(",", ":")) + "\n")

            # CPU stack frames
            if ph == "M" and name in ("thread_name", "thread_sort_index", "process_name"):
                cpu_stacks.append({"kind": name, "pid": pid, "tid": tid, "args": eargs})
                continue
            if ph in ("B", "E") and cat and "cpu" in str(cat).lower():
                cpu_stacks.append({"name": name, "cat": cat, "ph": ph, "pid": pid, "tid": tid, "ts": ts})

            # CUDA Graph markers
            lname = name.lower()
            if ("cuda" in lname and "graph" in lname) or "graph_capture" in lname or "node_seq" in str(eargs):
                cuda_graph_markers.append({"name": name, "cat": cat, "ph": ph, "pid": pid, "tid": tid, "ts": ts, "dur": dur, "args": eargs})

            # complete events with duration -> kernel/perf stats
            if ph == "X" and isinstance(dur, (int, float)) and dur >= args.min_dur_us:
                complete_count += 1
                by_name_durs[name].append(float(dur))
                by_name_cats[name].add(cat)
                by_name_tids[name].add(tid)
                by_name_pids[name].add(pid)
                if eargs and len(by_name_args[name]) < 20:
                    by_name_args[name].append(eargs)

    # build kernel_stats.json
    stats = {}
    for name, durs in by_name_durs.items():
        sd = sorted(durs)
        n = len(sd)
        mean = sum(sd) / n
        var = sum((x - mean) ** 2 for x in sd) / n if n > 1 else 0.0
        stats[name] = {
            "count": n,
            "mean_us": round(mean, 4),
            "std_us": round(math.sqrt(var), 4),
            "min_us": sd[0],
            "max_us": sd[-1],
            "p50_us": round(_percentile(sd, 0.50), 4),
            "p90_us": round(_percentile(sd, 0.90), 4),
            "p99_us": round(_percentile(sd, 0.99), 4),
            "sum_us": round(sum(sd), 4),
            "histogram": _histogram(sd, bins=16),
            "cats": sorted([str(c) for c in by_name_cats[name]]),
            "tids": sorted([str(t) for t in by_name_tids[name]]),
            "pids": sorted([str(p) for p in by_name_pids[name]]),
            "sample_args": by_name_args[name][:10],
        }

    stats_path = os.path.join(args.out_dir, "kernel_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, separators=(",", ":"))

    with open(os.path.join(args.out_dir, "cpu_stack_summary.json"), "w") as f:
        json.dump({"count": len(cpu_stacks), "items": cpu_stacks[:2000]}, f, separators=(",", ":"))

    with open(os.path.join(args.out_dir, "cuda_graph_meta.json"), "w") as f:
        json.dump({"detected": len(cuda_graph_markers) > 0, "count": len(cuda_graph_markers),
                    "items": cuda_graph_markers[:1000]}, f, separators=(",", ":"))

    report = {
        "trace_file": os.path.abspath(args.trace_file),
        "total_events": total_events,
        "complete_events": complete_count,
        "distinct_names": len(stats),
        "ts_range_us": [first_ts, last_ts],
        "cpu_stack_events": len(cpu_stacks),
        "cuda_graph_detected": len(cuda_graph_markers) > 0,
    }
    with open(os.path.join(args.out_dir, "parser_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # top-20 to stderr for human eyeballs
    top = sorted(stats.items(), key=lambda kv: kv[1]["sum_us"], reverse=True)[:20]
    print("[parse_tracing] top-20 by total time:", file=sys.stderr)
    for name, s in top:
        print(f"  {s['sum_us']:>14.1f} us  {s['count']:>7d}x  mean={s['mean_us']:>10.2f}  p90={s['p90_us']:>10.2f}  {name}", file=sys.stderr)

    print(f"[parse_tracing] done. outputs in {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
