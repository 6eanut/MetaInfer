#!/usr/bin/env python3
"""Render graph/flow_graph.json into a self-contained HTML file plus a markdown
ranking. No external CDNs; everything is inline.

Usage:
  python visualize_graph.py <flow_graph.json> [--out-html PATH] [--out-rank PATH]

The HTML is a left-to-right DAG drawn with absolute-positioned divs; layout is
a simple BFS-by-layers (longest-path layering). Good enough for ~hundreds of
nodes which is the realistic ceiling for an LLM-generated graph.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import defaultdict, deque


def _layer_nodes(nodes: list[dict]) -> dict[str, int]:
    """Longest-path layering: each node's layer = max(pred layer) + 1, roots=0."""
    layer: dict[str, int] = {}
    # Kahn-like topological pass with multiple rounds for safety
    idset = {n["id"] for n in nodes}
    preds = {n["id"]: [p for p in (n.get("preds") or []) if p in idset] for n in nodes}
    indeg = {nid: len(ps) for nid, ps in preds.items()}
    q = deque([nid for nid, d in indeg.items() if d == 0])
    for nid in q:
        layer[nid] = 0
    succs = defaultdict(list)
    for n in nodes:
        for p in (n.get("preds") or []):
            if p in idset:
                succs[p].append(n["id"])
    while q:
        u = q.popleft()
        for v in succs[u]:
            indeg[v] -= 1
            layer[v] = max(layer.get(v, 0), layer[u] + 1)
            if indeg[v] == 0:
                q.append(v)
    # any leftover (cycle) -> put on max+1
    if len(layer) < len(idset):
        m = max(layer.values()) if layer else 0
        for nid in idset:
            if nid not in layer:
                m += 1
                layer[nid] = m
    return layer


HTML_TMPL = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Execution Flow Graph — {title}</title>
<style>
* {{ box-sizing: border-box; }}
body {{ font: 13px/1.4 -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0; }}
#header {{ padding: 12px 16px; background: #1e1e2e; color: #cdd6f4; border-bottom: 2px solid #45475a; }}
#header h1 {{ margin: 0 0 6px 0; font-size: 16px; }}
#header .meta {{ color: #a6adc8; font-size: 12px; }}
#main {{ display: flex; }}
#stage-wrap {{ flex: 1; overflow: auto; padding: 16px; }}
#side {{ width: 360px; border-left: 1px solid #cdd6f4; background: #f5f5f7; padding: 12px; }}
#side h2 {{ margin: 0 0 8px 0; font-size: 14px; }}
#side table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
#side th, #side td {{ border-bottom: 1px solid #d8d8df; padding: 4px 6px; text-align: left; vertical-align: top; }}
#side td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
#side tr.hot {{ background: #ffe2e2; }}
.node {{ position: absolute; width: 230px; padding: 8px 10px; border: 1px solid #888;
         border-radius: 6px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.08); cursor: pointer; }}
.node .id {{ font-weight: 700; font-size: 11px; color: #555; }}
.node .op {{ font-weight: 700; }}
.node .perf {{ color: #444; font-size: 11px; }}
.node .file {{ color: #666; font-size: 10px; word-break: break-all; }}
.node .refs {{ margin-top: 4px; border-top: 1px dashed #ccc; padding-top: 3px; }}
.node .ref {{ font-size: 10px; color: #444; line-height: 1.25; word-break: break-all; }}
.node .ref .role {{ display: inline-block; min-width: 70px; color: #1a73e8; font-weight: 600; }}
.node .ref .refpath {{ color: #555; font-family: ui-monospace, Menlo, monospace; }}
.node .ref .refnote {{ color: #888; }}
.node .unverified {{ display: inline-block; padding: 1px 6px; background: #fbb; color: #800;
                     border-radius: 3px; font-size: 10px; font-weight: 600; }}
#stage {{ position: relative; min-width: {stage_w}px; height: {stage_h}px; }}
.edge {{ position: absolute; background: #999; height: 1px; transform-origin: 0 0; pointer-events: none; }}
.legend {{ font-size: 11px; color: #333; margin-top: 8px; }}
.legend span {{ display: inline-block; padding: 2px 6px; margin-right: 4px; border-radius: 3px; color: #fff; }}
</style></head>
<body>
<div id="header">
  <h1>Execution Flow Graph</h1>
  <div class="meta">{meta_html}</div>
</div>
<div id="main">
  <div id="stage-wrap"><div id="stage">{nodes_html}{edges_html}</div>
    <div class="legend">
      <span style="background:#6aa84f">cold</span>
      <span style="background:#f1c232">warm</span>
      <span style="background:#e06666">hot</span>
      color = mean_us relative to graph max
    </div>
  </div>
  <div id="side">
    <h2>Kernel ranking (by total time)</h2>
    <table>
      <thead><tr><th>op</th><th class="num">total%</th><th class="num">mean&mu;s</th><th class="num">p90&mu;s</th><th>note</th></tr></thead>
      <tbody>{rank_rows}</tbody>
    </table>
  </div>
</div>
</body></html>
"""


def _color_for(mean_us: float, max_us: float) -> str:
    if max_us <= 0:
        return "#6aa84f"
    r = mean_us / max_us
    if r < 0.15:
        return "#6aa84f"
    if r < 0.45:
        return "#f1c232"
    return "#e06666"


def _heur_note(node: dict) -> str:
    op = (node.get("operator") or "").lower()
    perf = node.get("perf") or {}
    mean = perf.get("mean_us", 0)
    if any(k in op for k in ("python", "host", "launch", "trace")):
        return "Python-side launch overhead candidate"
    if "quant" in op or "dequant" in op:
        return "Possible redundant (de)quant/fusion opportunity"
    if "norm" in op or "rms" in op or "rope" in op:
        return "Small kernel — check for fusion / kernel-launch overhead"
    if "matmul" in op or "gemm" in op or "linear" in op:
        return "Big GEMM — check workspace/epilogue/precision choice"
    if "attn" in op or "attention" in op:
        return "Attention — check KV layout, paged/multistream, Flash backend"
    if "sampler" in op or "sample" in op or "softmax" in op:
        return "Tail epilogue — check if fused with logits"
    if mean and mean < 5:
        return "Very short kernel — likely launch-bound; CUDA Graph candidate"
    return ""


def render(graph_path: str, out_html: str, out_rank: str) -> None:
    with open(graph_path) as f:
        graph = json.load(f)
    nodes = graph.get("nodes", [])
    if not nodes:
        raise SystemExit("graph has no nodes")

    layer = _layer_nodes(nodes)
    max_layer = max(layer.values()) if layer else 0

    # positions
    by_layer: dict[int, list[str]] = defaultdict(list)
    for nid, l in layer.items():
        by_layer[l].append(nid)
    node_pos: dict[str, tuple[int, int]] = {}

    # node height grows with number of framework_refs (each ref ~14px line)
    def _node_h(n: dict) -> int:
        refs = n.get("framework_refs") or []
        # base ~120px for header/op/perf/purpose, plus 16px per ref (min 1 line)
        return 120 + max(1, len(refs)) * 16

    LAYER_W = 300
    V_GAP = 24
    # assign per-layer y positions based on actual node heights
    node_h_map: dict[str, int] = {n["id"]: _node_h(n) for n in nodes}
    y_cursor: dict[int, int] = {l: 20 for l in range(max_layer + 1)}
    for l in range(max_layer + 1):
        for nid in by_layer.get(l, []):
            x = 20 + l * LAYER_W
            y = y_cursor[l]
            node_pos[nid] = (x, y)
            y_cursor[l] = y + node_h_map[nid] + V_GAP

    max_us = 0.0
    for n in nodes:
        p = n.get("perf") or {}
        if p.get("mean_us"):
            max_us = max(max_us, float(p["mean_us"]))
    total_sum = sum(float((n.get("perf") or {}).get("sum_us", 0) or
                          ((n.get("perf") or {}).get("mean_us", 0) *
                           (n.get("perf") or {}).get("count", 0))) for n in nodes)

    node_map = {n["id"]: n for n in nodes}

    # nodes html
    nodes_html = []
    for n in nodes:
        nid = n["id"]
        x, y = node_pos[nid]
        h = node_h_map[nid]
        p = n.get("perf") or {}
        mean = p.get("mean_us", 0)
        p90 = p.get("p90_us", 0)
        cnt = p.get("count", 0)
        color = _color_for(float(mean or 0), max_us)
        refs = n.get("framework_refs") or []
        if refs:
            refs_rows = []
            for r in refs:
                if not isinstance(r, dict):
                    continue
                role = html.escape(str(r.get("role", "?")))
                fpath = html.escape(str(r.get("file", "?")))
                line = r.get("line")
                line_s = f":{line}" if line is not None else ""
                note = html.escape(str(r.get("note", "")))[:80]
                refs_rows.append(
                    f'<div class="ref"><span class="role">{role}</span> '
                    f'<span class="refpath">{fpath}{line_s}</span>'
                    + (f' <span class="refnote">— {note}</span>' if note else "") + '</div>'
                )
            refs_html = '<div class="refs">' + "".join(refs_rows) + '</div>'
        else:
            refs_html = '<div class="refs"><span class="unverified">unverified source</span></div>'
        nodes_html.append(
            f'<div class="node" style="left:{x}px; top:{y}px; height:{h}px; border-left:6px solid {color}">'
            f'<div class="id">{html.escape(str(nid))}</div>'
            f'<div class="op">{html.escape(str(n.get("operator","?")))}</div>'
            f'<div class="perf">mean={mean:.2f}&mu;s · p90={p90:.2f}&mu;s · n={cnt}</div>'
            f'<div class="file">{html.escape(str(n.get("framework_file","?")))}</div>'
            f'<div style="font-size:11px;color:#333">{html.escape(str(n.get("step_purpose",""))[:120])}</div>'
            f'{refs_html}'
            f'</div>'
        )

    # edges html — draw simple right-angle polylines via inline SVG overlay
    svg_w = 20 + (max_layer + 1) * LAYER_W
    # svg_h based on the tallest layer using actual node heights
    per_layer_heights = []
    for l, members in by_layer.items():
        h = sum(node_h_map[nid] + V_GAP for nid in members) + 40
        per_layer_heights.append(h)
    svg_h = max(per_layer_heights) if per_layer_heights else 200
    edges = graph.get("edges", [])
    if not edges:
        # derive from preds/succs
        for n in nodes:
            for s in (n.get("succs") or []):
                edges.append({"from": n["id"], "to": s})
    svg_paths = []
    for e in edges:
        fr, to = e.get("from"), e.get("to")
        if fr not in node_pos or to not in node_pos:
            continue
        x1, y1 = node_pos[fr]
        x2, y2 = node_pos[to]
        x1 += 230  # node width (matches CSS .node width)
        y1 += node_h_map[fr] // 2
        y2 += node_h_map[to] // 2
        mx = (x1 + x2) / 2
        d = f"M {x1},{y1} C {mx},{y1} {mx},{y2} {x2},{y2}"
        svg_paths.append(f'<path d="{d}" stroke="#999" fill="none" stroke-width="1"/>')

    svg = (f'<svg style="position:absolute;left:0;top:0;" width="{svg_w}" height="{svg_h}" '
           f'aria-hidden="true">{"".join(svg_paths)}</svg>')

    # ranking
    ranked = sorted(nodes, key=lambda n: -float((n.get("perf") or {}).get("mean_us", 0) *
                                                 (n.get("perf") or {}).get("count", 0)))
    rank_rows = []
    rank_lines = ["# Kernel ranking", ""]
    rank_lines.append("| op | total% | mean_us | p90_us | count | framework | heuristic |")
    rank_lines.append("|----|--------|---------|--------|-------|-----------|-----------|")
    for n in ranked:
        p = n.get("perf") or {}
        s = float(p.get("mean_us", 0)) * float(p.get("count", 0))
        pct = (100.0 * s / total_sum) if total_sum > 0 else 0.0
        note = _heur_note(n)
        cls = ' class="hot"' if pct > 10 else ""
        rank_rows.append(
            f'<tr{cls}><td>{html.escape(str(n.get("operator","?")))}</td>'
            f'<td class="num">{pct:.1f}</td>'
            f'<td class="num">{p.get("mean_us",0):.2f}</td>'
            f'<td class="num">{p.get("p90_us",0):.2f}</td>'
            f'<td>{p.get("count",0)}</td>'
            f'<td>{html.escape(str(n.get("framework_file","?")))}</td>'
            f'<td>{html.escape(note)}</td></tr>'
        )
        rank_lines.append(f"| {n.get('operator','?')} | {pct:.1f}% | {p.get('mean_us',0):.2f} | "
                          f"{p.get('p90_us',0):.2f} | {p.get('count',0)} | "
                          f"{n.get('framework_file','?')} | {note} |")

    model = graph.get("model", {})
    meta_html = (f"architectures: {html.escape(str(model.get('architectures','?')))} · "
                 f"quant: {html.escape(str(model.get('quantization','?')))} · "
                 f"dtype: {html.escape(str(model.get('dtype','?')))} · "
                 f"nodes: {len(nodes)} · run_id: {html.escape(str(graph.get('run_id','?')))}")

    page = HTML_TMPL.format(
        title=html.escape(str(graph.get("run_id", "flow_graph"))),
        meta_html=meta_html,
        stage_w=svg_w + 40,
        stage_h=svg_h + 40,
        nodes_html="".join(nodes_html),
        edges_html=svg,
        rank_rows="".join(rank_rows),
    )

    with open(out_html, "w") as f:
        f.write(page)
    with open(out_rank, "w") as f:
        f.write("\n".join(rank_lines) + "\n")
    print(f"wrote {out_html} and {out_rank}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("flow_graph_json")
    ap.add_argument("--out-html", default=None)
    ap.add_argument("--out-rank", default=None)
    args = ap.parse_args()
    out_html = args.out_html or os.path.splitext(args.flow_graph_json)[0] + ".html"
    out_rank = args.out_rank or os.path.join(os.path.dirname(out_html) or ".", "ranking.md")
    render(args.flow_graph_json, out_html, out_rank)
    return 0


if __name__ == "__main__":
    sys.exit(main())
