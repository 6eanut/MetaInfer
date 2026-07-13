// Rough estimate panel — renders S0's quick back-of-envelope numbers.
// Polls /api/tasks/<id>/calc/rough every few seconds until S0 produces
// rough_results.json; then renders a per-node table.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { getCalcRough } from "app/api";

function fmtNum(n, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  if (!isFinite(n)) return "∞";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  return n.toFixed(digits);
}

export function CalcRoughPanel({ taskId }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const d = await getCalcRough(taskId);
        if (alive) { setData(d); setErr(null); }
      } catch (e) {
        if (alive) setErr(String(e));
      }
    };
    tick();
    const id = setInterval(tick, 5000);
    return () => { alive = false; clearInterval(id); };
  }, [taskId]);

  if (err) {
    return html`
      <section class="panel">
        <h2>粗略评估</h2>
        <div class="task-banner task-banner-err">加载失败：${err}</div>
      </section>
    `;
  }
  if (!data) {
    return html`
      <section class="panel">
        <h2>粗略评估</h2>
        <p class="muted">加载中…</p>
      </section>
    `;
  }
  if (data.pending) {
    return html`
      <section class="panel">
        <h2>粗略评估</h2>
        <p class="muted">S0 尚未启动 — 单 Agent 会先跑一遍粗略评估，几分钟内出结果。</p>
      </section>
    `;
  }
  if (!data.ok) {
    return html`
      <section class="panel">
        <h2>粗略评估</h2>
        <div class="task-banner task-banner-err">
          S0 失败：${data.error || "unknown error"}
        </div>
      </section>
    `;
  }

  const rows = data.results || [];
  const elapsed = data.elapsed_s;
  const summary = data.summary || {};

  return html`
    <section class="panel calc-rough-panel">
      <h2>
        粗略评估
        <span class="muted small">
          （S0 单 Agent · ${elapsed != null ? elapsed.toFixed(1) + "s" : ""}
          · ${summary.ok_count || 0}/${summary.total_nodes || 0} 节点成功）
        </span>
      </h2>
      <p class="muted small">
        这是粗略值，会被详细审计（S3）覆盖。MoE 类节点只算 6KHI，不算 scaling/combine。
        每 cell 同时显示 prefill（处理整段 prompt）和 decode（生成 1 token，含 KV cache 读取）。
      </p>
      ${rows.length === 0 ? html`<p class="muted">S0 未产出节点。</p>` : html`
        <table class="calc-table">
          <thead>
            <tr>
              <th class="left" rowspan="2">section</th>
              <th class="left" rowspan="2">node</th>
              <th colspan="2" class="center">prefill (B=1,S=512)</th>
              <th colspan="2" class="center">decode (B=1,S=512)</th>
              <th class="left" rowspan="2">status</th>
            </tr>
            <tr>
              <th class="right">tflops</th>
              <th class="right">access_gb</th>
              <th class="right">tflops</th>
              <th class="right">access_gb</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map((r) => {
              const pre = r.prefill || {};
              const dec = r.decode || {};
              return html`
                <tr key=${r.compound} class=${r.ok ? "" : "row-err"}>
                  <td>${r.section_id}</td>
                  <td><code>${r.node_id}</code></td>
                  <td class="right">${r.ok ? fmtNum(pre.tflops) : "—"}</td>
                  <td class="right">${r.ok ? fmtNum(pre.access_gb) : "—"}</td>
                  <td class="right">${r.ok ? fmtNum(dec.tflops) : "—"}</td>
                  <td class="right">${r.ok ? fmtNum(dec.access_gb) : "—"}</td>
                  <td class="muted">
                    ${r.ok ? "ok" : (r.error || "failed")}
                  </td>
                </tr>
              `;
            })}
          </tbody>
        </table>
      `}
    </section>
  `;
}
