// opt-operator — dark modern dashboard detail body.
//
// Tabs:
//   - overview (概览): KPI stat row + phase stepper + champion lineage curve +
//     summary / reference / GPU pool
//   - iterations (迭代): per-iteration cards with expandable conformance + perf
//   - runtime (运行状态): shell-shared agents panel + timeline
//
// Live updates: polls /overview + /iterations every 5s and also streams the shell
// SSE watcher via /events so panels refresh as soon as run.json / ledger change.

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";

const PLUGIN_TYPE = "opt-operator";

const TABS = [
  { id: "overview", label: "概览" },
  { id: "iterations", label: "迭代" },
  { id: "runtime", label: "运行状态" },
];

const ORIGIN_LABEL = {
  user: "用户提供",
  library: "参考库",
  generated: "生成+审查",
  unknown: "未知",
};

// --------------------------------------------------------------- data hook

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

async function getJson(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

function useData(taskId) {
  const [data, setData] = useState({ overview: null, iterations: [] });
  const refresh = useCallback(async () => {
    if (!taskId) return;
    const base = `/api/${PLUGIN_TYPE}/${encodeURIComponent(taskId)}`;
    const [ov, it] = await Promise.all([
      withTimeout(getJson(`${base}/overview`).catch((e) => { console.warn("overview:", e); return null; })),
      withTimeout(getJson(`${base}/iterations`).catch((e) => { console.warn("iterations:", e); return []; })),
    ]);
    setData({ overview: ov, iterations: it || [] });
  }, [taskId]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);
  return { ...data, refresh };
}

// ----------------------------------------------------------------- helpers

function fmtLatency(ns) {
  if (ns == null) return "—";
  if (ns >= 1e6) return { v: (ns / 1e6).toFixed(2), u: "ms" };
  if (ns >= 1e3) return { v: (ns / 1e3).toFixed(1), u: "µs" };
  return { v: ns.toFixed(0), u: "ns" };
}

function runState(run) {
  if (run.finished) {
    if (run.final_status === "success") return { cls: "finished", label: "已完成" };
    if (run.final_status === "failed") return { cls: "failed", label: "已失败" };
    return { cls: "finished", label: "已结束" };
  }
  return { cls: "running", label: "运行中" };
}

function phaseName(graph) {
  if (!graph || !graph.current) return null;
  const n = (graph.nodes || []).find((x) => x.id === graph.current);
  return n ? n.label : graph.current;
}

// ------------------------------------------------------- phase stepper view

function orderedNodes(graph) {
  const nodes = (graph.nodes || []).slice();
  const byId = {};
  nodes.forEach((n) => { byId[n.id] = n; });
  const edges = graph.edges || [];
  const incoming = new Set(edges.map((e) => e.to));
  let cur = nodes.find((n) => !incoming.has(n.id));
  const order = [];
  while (cur) {
    order.push(cur);
    const e = edges.find((x) => x.from === cur.id);
    cur = e ? byId[e.to] : undefined;
  }
  return order;
}

function PhaseStepper({ graph }) {
  if (!graph || !graph.nodes || !graph.nodes.length) {
    return html`<div class="muted">尚无阶段数据。</div>`;
  }
  const nodes = orderedNodes(graph);
  const parts = [];
  nodes.forEach((n, i) => {
    if (i > 0) parts.push(html`<div class=${"oo-step conn" + (n.state === "done" || n.state === "current" ? " done" : "")} key=${"c" + i}></div>`);
    parts.push(html`
      <div class=${"oo-step " + (n.state === "current" ? "current" : n.state === "done" ? "done" : "pending")} key=${n.id}
           title=${(n.label || n.id) + (n.tier ? " (" + n.tier + " tier)" : "")}>
        <div class="dot"></div>
        <div class="lbl">${n.label || n.id}</div>
        ${n.tier ? html`<div class=${"oo-tier " + n.tier}>${n.tier === "strong" ? "强" : "便宜"}</div>` : ""}
      </div>`);
  });
  // terminal node (finished) shown as a trailing step when active
  const term = (graph.terminal_nodes || [])[0];
  if (term) {
    const isCurrent = graph.current === "finished";
    parts.push(html`<div class=${"oo-step conn" + (isCurrent ? " current" : "")} key="t-conn"></div>`);
    parts.push(html`
      <div class=${"oo-step " + (isCurrent ? "current" : "pending")} key="finished">
        <div class="dot"></div><div class="lbl">${term.label}</div>
      </div>`);
  }
  return html`<div class="oo-steps">${parts}</div>`;
}

// --------------------------------------------------------- lineage SVG chart

function LineageChart({ lineage }) {
  if (!lineage || !lineage.length) return null;
  const W = 620, H = 190, PL = 46, PR = 14, PT = 12, PB = 26;
  const iw = W - PL - PR, ih = H - PT - PB;
  const xs = lineage.map((e) => e.iteration);
  const bests = lineage.map((e) => e.best_latency_ns).filter((v) => v != null);
  const avgs = lineage.map((e) => e.avg_latency_ns).filter((v) => v != null);
  const lo = Math.min(...bests, ...avgs, 1);
  const hi = Math.max(...bests, ...avgs, 1);
  const span = Math.max(hi - lo, 1);
  const pad = span * 0.06;
  const ymax = hi + pad, ymin = Math.max(0, lo - pad);
  const yrange = Math.max(ymax - ymin, 1);
  const x = (it) => {
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    const denom = maxX - minX || 1;
    return PL + ((it - minX) / denom) * iw;
  };
  const y = (v) => PT + (1 - (v - ymin) / yrange) * ih;

  const gid = "ooLatGrad";
  const ptsBest = lineage.filter((e) => e.best_latency_ns != null);
  const bestPath = ptsBest.map((e, i) => (i ? "L" : "M") + x(e.iteration).toFixed(1) + " " + y(e.best_latency_ns).toFixed(1)).join(" ");
  const areaPath = bestPath
    + " L" + (ptsBest.length ? x(ptsBest[ptsBest.length - 1].iteration).toFixed(1) : x(lineage[0].iteration)) + " " + (PT + ih)
    + " L" + (ptsBest.length ? x(ptsBest[0].iteration) : x(lineage[0].iteration)) + " " + (PT + ih) + " Z";

  const ptAvgs = lineage.filter((e) => e.avg_latency_ns != null);
  const avgPath = ptAvgs.map((e, i) => (i ? "L" : "M") + x(e.iteration).toFixed(1) + " " + y(e.avg_latency_ns).toFixed(1)).join(" ");

  const gridLines = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const gy = PT + (1 - f) * ih;
    const val = ymin + (1 - f) * yrange;
    return html`<g key=${f}><line class="grid" x1=${PL} y1=${gy.toFixed(1)} x2=${W - PR} y2=${gy.toFixed(1)} />
      <text class="axis" x=${PL - 6} y=${(gy + 3).toFixed(1)} text-anchor="end">${val >= 1e6 ? (val / 1e6).toFixed(1) : (val / 1e3).toFixed(0)}</text></g>`;
  });

  return html`
    <div class="oo-chart">
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Champion latency across iterations (lower is better)">
        <defs>
          <linearGradient id=${gid} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stop-color="rgba(88,166,255,0.28)"/>
            <stop offset="100%" stop-color="rgba(88,166,255,0.02)"/>
          </linearGradient>
        </defs>
        ${gridLines}
        <path class="area" d=${areaPath} />
        ${ptAvgs.length > 1 ? html`<path class="line-avg" d=${avgPath} />` : ""}
        ${ptsBest.length > 1 ? html`<path class="line-best" d=${bestPath} />` : ""}
        ${lineage.map((e) => {
          if (e.best_latency_ns == null) return "";
          const cls = e.parent_iteration == null ? "pt genesis" : (e.iteration === lineage[lineage.length - 1].iteration ? "pt champ" : "pt");
          const t = `iter ${e.iteration} · ${(e.best_latency_ns / 1e6).toFixed(3)} ms` + (e.speedup_vs_genesis ? ` · ${e.speedup_vs_genesis.toFixed(2)}×` : " · genesis");
          return html`<circle class=${cls} cx=${x(e.iteration).toFixed(1)} cy=${y(e.best_latency_ns).toFixed(1)} r="3"><title>${t}</title></circle>`;
        })}
      </svg>
      <div class="oo-chart-legend">
        <span><i class="solid"></i>最佳延迟</span>
        <span><i class="dash"></i>平均延迟</span>
        <span class="muted">（越低越好，仅晋升点）</span>
      </div>
    </div>`;
}

// -------------------------------------------------------------- overview tab

function StatTile({ cls = "", k, v, unit, s }) {
  return html`
    <div class=${"oo-stat " + cls}>
      <div class="k">${k}</div>
      <div class="v">${v}${unit ? html`<span class="unit">${unit}</span>` : ""}</div>
      ${s ? html`<div class="s">${s}</div>` : ""}
    </div>`;
}

function OverviewStats({ ov }) {
  const run = (ov && ov.run) || {};
  const summary = (ov && ov.summary) || {};
  const champ = summary.champion;
  const st = runState(run);
  const best = champ && champ.best_latency_ns != null ? fmtLatency(champ.best_latency_ns) : null;
  const speedup = summary.speedup_vs_genesis;
  const graph = ov && ov.state_graph;
  const phase = phaseName(graph) || (run.finished ? "finished" : run.current_phase || "idle");
  const pool = ov && ov.gpu_pool;
  const gpuUsed = Array.isArray(pool) ? pool.length : 0;
  return html`
    <div class="oo-stats">
      <${StatTile} cls="accent" k="当前阶段" v=${phase}
        s=${"iteration " + (run.current_iteration ?? 0)} />
      <${StatTile} k="状态" v=${st.label} cls=${st.cls === "failed" ? "bad" : st.cls === "running" ? "good" : "accent"}
        s=${run.finished ? "run finished" : "auto-updating…"} />
      ${best ? html`<${StatTile} k="Champion 最佳延迟" v=${best.v} unit=${best.u} s=${"avg " + (champ && champ.avg_latency_ns != null ? fmtLatency(champ.avg_latency_ns).v + " " + fmtLatency(champ.avg_latency_ns).u : "—")} />` : ""}
      <${StatTile} cls=${speedup != null && speedup > 1 ? "good" : ""} k="相对初始加速" v=${speedup != null ? speedup.toFixed(2) + "×" : "—"}
        s=${"champion iter " + ((champ && champ.iteration) ?? "—")} />
      <${StatTile} k="晋升次数" v=${summary.promotions ?? 0}
        s=${"lineage " + (champ && champ.case_count ? champ.case_count + " cases" : "—")} />
      <${StatTile} k="GPU 占用" v=${gpuUsed}
        s=${(ov && ov.gpu_pool && ov.gpu_pool[0] && ov.gpu_pool[0].node_id) || "—"} />
    </div>`;
}

function GpuPool({ pool }) {
  if (!pool || !pool.length) return html`<div class="muted">当前无 GPU 租约（空闲）。</div>`;
  return html`
    <div class="oo-chips">
      ${pool.map((g) => {
        const held = g.status !== "free";
        const cls = held ? "accent" : "ok";
        return html`<span class=${"oo-chip " + cls} key=${g.node_id + "-" + g.slot}>
          <span class="sw"></span>${g.node_id}:${g.slot}
          ${held ? html`<span class="muted">· ${g.job_id || "occupied"}</span>` : ""}
        </span>`;
      })}
    </div>`;
}

function SummaryCard({ ov }) {
  const run = (ov && ov.run) || {};
  const summary = (ov && ov.summary) || {};
  const champ = summary.champion;
  return html`
    <div class="oo-panel">
      <div class="oo-panel-h"><h3><span class="tick">▸</span>汇总</h3></div>
      ${!champ ? html`<div class="muted">尚无 champion。</div>` : html`
        <ul class="oo-kv">
          <li><span class="k">当前 champion</span><span class="v mono">${(champ.kernel_digest || "").slice(0, 12)}</span></li>
          <li><span class="k">语言</span><span class="v">${champ.language || "—"}</span></li>
          <li><span class="k">所在迭代</span><span class="v num">#${champ.iteration}</span></li>
          <li><span class="k">最佳延迟</span><span class="v num">${champ.best_latency_ns != null ? (champ.best_latency_ns / 1e6).toFixed(3) + " ms" : "—"}</span></li>
          <li><span class="k">平均延迟</span><span class="v num">${champ.avg_latency_ns != null ? (champ.avg_latency_ns / 1e6).toFixed(3) + " ms" : "—"}</span></li>
          <li><span class="k">加速 vs 初始</span><span class="v num">${summary.speedup_vs_genesis != null ? summary.speedup_vs_genesis.toFixed(2) + "×" : "—"}</span></li>
          <li><span class="k">case 数</span><span class="v num">${champ.case_count || "—"}</span></li>
          <li><span class="k">run</span><span class="v">${(run.finished ? "finished" : run.current_phase) || "idle"} · iter ${run.current_iteration ?? 0}</span></li>
        </ul>`}
    </div>`;
}

function ReferenceCard({ ov }) {
  const ref = (ov && ov.reference) || {};
  if (!ref.op_id && !ref.origin) {
    return html`<div class="oo-panel"><div class="oo-panel-h"><h3><span class="tick">▸</span>参考 Oracle</h3></div><div class="muted">暂无参考信息。</div></div>`;
  }
  const origin = ref.origin || "unknown";
  const label = ORIGIN_LABEL[origin] || origin;
  const ocls = origin === "library" ? "accent" : origin === "user" ? "ok" : origin === "generated" ? "warn" : "idle";
  return html`
    <div class="oo-panel">
      <div class="oo-panel-h"><h3><span class="tick">▸</span>参考 Oracle</h3><span class="oo-hint"></span></div>
      <ul class="oo-kv">
        <li><span class="k">算子</span><span class="v mono">${ref.op_id || "—"}</span></li>
        <li><span class="k">来源</span><span class="v"><span class=${"oo-chip " + ocls}><span class="sw"></span>${label}</span></span></li>
        <li><span class="k">冻结摘要</span><span class="v oo-digest">${(ref.digest || "").slice(0, 16) || "—"}</span></li>
      </ul>
    </div>`;
}

function OverviewTab({ ov }) {
  return html`
    <div class="oo-panel">
      <div class="oo-panel-h"><h3><span class="tick">▸</span>阶段流转</h3><span class="oo-hint oo-chip accent"><span class="sw"></span>强=规划/审查 · 便宜=实现</span></div>
      <${PhaseStepper} graph=${ov && ov.state_graph} />
    </div>
    <div class="oo-layout-main">
      <div class="oo-col">
        <div class="oo-panel">
          <div class="oo-panel-h"><h3><span class="tick">▸</span>Champion 溯源</h3><span class="oo-hint muted">每代最佳延迟</span></div>
          <${LineageChart} lineage=${ov && ov.lineage} />
        </div>
      </div>
      <div class="oo-col">
        <${SummaryCard} ov=${ov} />
        <${ReferenceCard} ov=${ov} />
        <div class="oo-panel">
          <div class="oo-panel-h"><h3><span class="tick">▸</span>GPU 池</h3></div>
          <${GpuPool} pool=${ov && ov.gpu_pool} />
        </div>
      </div>
    </div>`;
}

// ------------------------------------------------------------ conformance UI

function ConformanceBody({ conformance }) {
  if (!conformance) return null;
  const results = conformance.results || [];
  if (!results.length) {
    return html`<div class="muted">无 conformance case 记录。</div>`;
  }
  return html`
    <div class="oo-conf-grid">
      ${results.map((r) => {
        const bad = !r.passed;
        const detail = bad
          ? ("abs " + (r.max_abs_err != null ? r.max_abs_err.toExponential(2) : "?")
             + " / rel " + (r.max_rel_err != null ? r.max_rel_err.toExponential(2) : "?"))
          : "";
        return html`<div class=${"oo-conf-row " + (bad ? "err" : "ok")} key=${r.case_id}>
          <span class="cid">${r.case_id}</span>
          ${bad
            ? html`<span class="err-detail">${detail}${r.detail ? " · " + r.detail : ""}</span>`
            : html`<span class="mark">通过</span>`}
        </div>`;
      })}
    </div>`;
}

function PerfTable({ perf }) {
  if (!perf) return null;
  const rows = Object.entries(perf);
  if (!rows.length) return html`<div class="muted">无性能数据。</div>`;
  return html`
    <table class="oo-perf-table">
      <thead><tr><th>Case</th><th class="oo-perf-num">延迟</th></tr></thead>
      <tbody>
        ${rows.map(([cid, p]) => {
          const l = fmtLatency(p && p.latency_ns);
          return html`<tr key=${cid}><td class="mono">${cid}</td><td class="oo-perf-num">${l.v} ${l.u}</td></tr>`;
        })}
      </tbody>
    </table>`;
}

function IterationCard({ it }) {
  const [open, setOpen] = useState(false);
  const ok = it.status === "success";
  const failed = it.status === "failed";
  const confPassed = it.conformance && it.conformance.passed;
  const badge = html`
    <span class=${"oo-chip " + (ok && confPassed ? "ok" : failed ? "err" : ok ? "warn" : "idle")}>
      <span class="sw"></span>${it.status || "running"}
    </span>`;
  return html`
    <div class=${"oo-iter-card " + (confPassed ? "good" : failed ? "fail" : "")}>
      <div class="oo-iter-head" onClick=${() => setOpen(!open)}>
        <span class="oo-iter-num">#${it.iteration}</span>
        <span class="muted" style="flex:1">${it.phase || ""}</span>
        ${badge}
        ${it.promoted ? html`<span class="oo-chip ok"><span class="sw"></span>已晋升</span>` : ""}
        ${it.candidate_language ? html`<span class="oo-chip idle">${it.candidate_language}</span>` : ""}
        <span class="muted">${open ? "▾" : "▸"}</span>
      </div>
      ${open ? html`
        <div class="oo-iter-body">
          ${it.plan && it.plan.approach ? html`<ul class="oo-kv" style="margin:0 0 8px"><li><span class="k">方案</span><span class="v">${it.plan.approach}</span></li></ul>` : ""}
          ${confPassed != null || it.conformance ? html`
            <div style="margin-bottom:8px"><div class="muted" style="font-size:11px;margin-bottom:4px">正确性（conformance）</div>
              <${ConformanceBody} conformance=${it.conformance} /></div>` : ""}
          ${it.perf ? html`
            <div><div class="muted" style="font-size:11px;margin-bottom:4px">每 case 延迟</div>
              <${PerfTable} perf=${it.perf} /></div>` : ""}
          ${it.guidance ? html`<div class="muted" style="font-size:11px;margin-top:8px">审查意见：${it.guidance}</div>` : ""}
        </div>` : ""}
    </div>`;
}

function IterationsTab({ iterations }) {
  if (!iterations || !iterations.length) {
    return html`<div class="oo-empty">尚无迭代记录——等待优化循环推进。</div>`;
  }
  const total = iterations.length;
  const okCount = iterations.filter((i) => i.status === "success").length;
  const failedCount = iterations.filter((i) => i.status === "failed").length;
  const promotedCount = iterations.filter((i) => i.promoted).length;
  return html`
    <div class="oo-stats" style="margin-bottom:0">
      <${StatTile} k="总迭代" v=${total} />
      <${StatTile} cls="good" k="成功" v=${okCount} />
      ${failedCount ? html`<${StatTile} cls="bad" k="失败/跳过" v=${failedCount} />` : ""}
      <${StatTile} cls="accent" k="其中晋升" v=${promotedCount} />
    </div>
    <div class="oo-iter-rows">
      ${iterations.slice().reverse().map((it) => html`<${IterationCard} it=${it} key=${it.iteration} />`)}
    </div>`;
}

// --------------------------------------------------------------- main view

export default function OptOperatorDetailView({ taskId, run, timeline, agents, loadState, lastErr }) {
  const { overview, iterations, refresh } = useData(taskId);

  useEffect(() => {
    if (!taskId) return;
    const es = new EventSource(`/api/${PLUGIN_TYPE}/${encodeURIComponent(taskId)}/events`);
    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg && msg.type === "task_changed") refresh();
      } catch (e) { /* ignore malformed keepalive */ }
    };
    return () => es.close();
  }, [taskId, refresh]);

  const [tab, setTab] = useState("overview");
  const st = runState((overview && overview.run) || {});
  return html`
    <div class="opt-operator">
      <div class="oo-toolbar">
        <h2 class="oo-title">算子优化</h2>
        <span class="muted">opt-operator · ${taskId}</span>
        <span class=${"oo-live " + st.cls}>${st.label}</span>
      </div>
      <${OverviewStats} ov=${overview} />
      <div class="oo-tabs" role="tablist">
        ${TABS.map((t) => {
          const cnt = t.id === "iterations" ? iterations.length : null;
          return html`<button class=${"oo-tab" + (tab === t.id ? " active" : "")}
            role="tab" aria-selected=${tab === t.id}
            onClick=${() => setTab(t.id)}>${t.label}
            ${cnt != null ? html`<span class="count">${cnt}</span>` : ""}
          </button>`;
        })}
      </div>
      ${tab === "overview" ? html`<div style="display:flex;flex-direction:column;gap:12px"><${OverviewTab} ov=${overview} /></div>` : ""}
      ${tab === "iterations" ? html`<${IterationsTab} iterations=${iterations} />` : ""}
      ${tab === "runtime" ? html`
        <div class="oo-runtime-grid">
          <div class="oo-panel"><div class="oo-panel-h"><h3><span class="tick">▸</span>Agents</h3></div><${AgentsPanel} taskId=${taskId} /></div>
          <div class="oo-panel"><div class="oo-panel-h"><h3><span class="tick">▸</span>Timeline</h3></div><${Timeline} taskId=${taskId} /></div>
        </div>` : ""}
    </div>`;
}
