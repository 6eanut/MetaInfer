// opt-operator task detail body.
//
// Two-level view:
//   - overview (宏观): phase state graph + champion lineage curve +
//     reference-origin + GPU pool + summary
//   - iterations (当前工作 drill-in): per-iteration conformance + latency
//   - runtime (运行状态): shell-shared agents panel + timeline
//
// Live updates: polls /overview on an interval and also streams the shell SSE
// watcher via /events so panels refresh as soon as run.json / ledger / iterations
// change. Shell passes {run, timeline, agents, loadState, lastErr} via `data`.

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
  const [data, setData] = useState({
    overview: null, iterations: [],
  });
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

function PhaseGraph({ graph }) {
  if (!graph) return html`<div class="muted">等待 state-graph 数据…</div>`;
  const nodes = graph.nodes || [];
  const current = graph.current || "idle";
  return html`
    <div class="phase-nodes">
      ${nodes.map((n) => html`
        <span class=${"phase-pill" + (n.id === current ? " active" : " " + (n.state || ""))}
              title=${(n.tier ? "model tier: " + n.tier : "")}>
          ${n.label}
        </span>
      `)}
    </div>
  `;
}

function LineageCurve({ lineage }) {
  if (!lineage || !lineage.length) {
    return html`<div class="muted">尚无 champion 溯源记录。</div>`;
  }
  const max = Math.max(...lineage.map((e) => e.best_latency_ns || 0), 1);
  return html`
    <table class="iter-table">
      <thead>
        <tr><th>#</th><th>Kernel</th><th>Lang</th><th>best ns</th><th>speedup vs genesis</th><th>cases</th></tr>
      </thead>
      <tbody>
        ${lineage.map((e) => html`
          <tr key=${e.iteration}>
            <td>${e.iteration}</td>
            <td><code class="digest">${(e.kernel_digest || "").slice(0, 10)}</code></td>
            <td>${e.language}</td>
            <td>${e.best_latency_ns != null ? e.best_latency_ns.toFixed(1) : "—"}</td>
            <td>${e.speedup_vs_genesis != null ? e.speedup_vs_genesis.toFixed(2) + "×" : "—"}</td>
            <td>${e.case_count}</td>
          </tr>
        `)}
      </tbody>
    </table>
    <div class="lineage-bars">
      ${lineage.map((e) => {
        const h = e.best_latency_ns != null ? Math.round((e.best_latency_ns / max) * 100) : 0;
        return html`<div class="bar" style=${`height:${h}%`} title=${`iter ${e.iteration}: ${e.best_latency_ns}`}></div>`;
      })}
    </div>
  `;
}

function GpuPool({ pool }) {
  if (!pool) return html`<div class="muted">GPU 池不可用。</div>`;
  return html`
    <div class="gpu-grid">
      ${pool.map((g) => html`
        <div class=${"gpu-slot " + (g.status === "held" ? "held" : "free")} key=${g.node_id + "-" + g.slot}>
          <div>${g.node_id}:${g.slot} <span class="muted">${g.status}</span></div>
          ${g.status === "held" ? html`<div class="muted">${g.job_id || ""}</div>` : ""}
        </div>
      `)}
    </div>
  `;
}

function ConformanceCell({ conformance }) {
  if (!conformance) return html`<span class="muted">—</span>`;
  const results = (conformance.results || []);
  const passed = conformance.passed;
  return html`
    <div class=${"conf " + (passed ? "pass" : "fail")}>
      ${passed ? "PASS" : "FAIL"} (${results.length} cases)
    </div>
  `;
}

function IterationsTable({ iterations }) {
  if (!iterations || !iterations.length) {
    return html`<div class="muted">尚无迭代记录。</div>`;
  }
  return html`
    <table class="iter-table">
      <thead>
        <tr><th>Iter</th><th>Phase</th><th>Status</th><th>Conformance</th><th>Promoted</th><th>Lang</th></tr>
      </thead>
      <tbody>
        ${iterations.map((it) => html`
          <tr key=${it.iteration}>
            <td>${it.iteration}</td>
            <td>${it.phase || ""}</td>
            <td>${it.status || ""}</td>
            <td>${html`<${ConformanceCell} conformance=${it.conformance} />`}</td>
            <td>${it.promoted ? "✓" : ""}</td>
            <td>${it.candidate_language || ""}</td>
          </tr>
        `)}
      </tbody>
    </table>
  `;
}

function OverviewTab({ ov }) {
  const run = (ov && ov.run) || {};
  const summary = (ov && ov.summary) || {};
  const ref = (ov && ov.reference) || {};
  return html`
    <div class="overview-grid">
      <section class="card">
        <h3>阶段</h3>
        <${PhaseGraph} graph=${ov && ov.state_graph} />
        <p class="muted">
          phase=${run.current_phase} iter=${run.current_iteration}
          ${run.finished ? " · finished" + (run.final_status ? " (" + run.final_status + ")" : "") : ""}
        </p>
      </section>
      <section class="card">
        <h3>Champion 溯源</h3>
        <${LineageCurve} lineage=${ov && ov.lineage} />
      </section>
      <section class="card">
        <h3>汇总</h3>
        <ul class="kv">
          <li><span>promotions</span><span>${summary.promotions ?? 0}</span></li>
          <li><span>speedup vs genesis</span><span>${summary.speedup_vs_genesis != null ? summary.speedup_vs_genesis.toFixed(2) + "×" : "—"}</span></li>
          <li><span>reference origin</span><span>${ref.origin || "—"}</span></li>
          <li><span>op_id</span><span>${ref.op_id || "—"}</span></li>
        </ul>
      </section>
      <section class="card">
        <h3>GPU 池</h3>
        <${GpuPool} pool=${ov && ov.gpu_pool} />
      </section>
    </div>
  `;
}

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
  return html`
    <div class="opt-operator">
      <div class="tabs">
        ${TABS.map((t) => html`
          <button class=${"tab" + (tab === t.id ? " active" : "")} onClick=${() => setTab(t.id)}>${t.label}</button>
        `)}
      </div>
      ${tab === "overview" ? html`<${OverviewTab} ov=${overview} />` : ""}
      ${tab === "iterations" ? html`<${IterationsTable} iterations=${iterations} />` : ""}
      ${tab === "runtime" ? html`
        <div class="runtime-grid">
          <section class="card"><h3>Agents</h3><${AgentsPanel} taskId=${taskId} /></section>
          <section class="card"><h3>Timeline</h3><${Timeline} taskId=${taskId} /></section>
        </div>
      ` : ""}
    </div>
  `;
}
