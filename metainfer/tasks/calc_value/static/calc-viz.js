// Calc-value task detail view.
//
// Replaces the default TaskDetailView when run.task_type == "calc-theoretical-value".
// Layout:
//   ┌── step progress bar (S1 → S2 → S3 → S4) ──┐
//   ├── live sub-agents panel (shared with default view)
//   ├── event timeline (shared)
//   └── visualization iframe (loads /calc/viz) with external batch_size / seq_len
//       controls that drive /calc/compute directly (no postMessage needed
//       because the iframe is same-origin with the WebUI).

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import {
  getTimeline, getAgents, controlTask,
} from "app/api";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";
import { CalcIterations } from "app/calc-iterations";
import { labelFor } from "app/utils";

const STEPS = [
  { id: "s1_analyze", label: "S1: Analyze code (2 agents)" },
  { id: "s2_graph", label: "S2: Build & validate graph" },
  { id: "s3_calculate", label: "S3: Calculate FLOPs / mem" },
  { id: "s4_visualize", label: "S4: Generate visualization" },
];

export function CalcVizView({ taskId, run, status }) {
  const [timeline, setTimeline] = useState({ events: [], since: 0 });
  const [agents, setAgents] = useState({ ts: 0, agents: [] });
  const [summary, setSummary] = useState(null);
  const [batchSize, setBatchSize] = useState(1);
  const [seqLen, setSeqLen] = useState(512);
  const [compute, setCompute] = useState(null);
  const [computeErr, setComputeErr] = useState(null);
  const [iterations, setIterations] = useState(null);
  const [iterErr, setIterErr] = useState(null);

  const withTimeout = (p, ms = 8000) =>
    Promise.race([
      p,
      new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
    ]);

  const refresh = useCallback(async () => {
    if (!taskId) return;
    try {
      const [tl, ag, sm] = await Promise.all([
        withTimeout(getTimeline(taskId, timeline.since || 0))
          .catch(() => ({ events: [] })),
        withTimeout(getAgents(taskId)).catch(() => ({ ts: 0, agents: [] })),
        withTimeout(
          fetch(`/api/calc-theoretical-value/${taskId}/calc/summary`, { cache: "no-store" })
            .then((r) => r.ok ? r.json() : null)
        ).catch(() => null),
      ]);
      setTimeline((prev) => ({
        events: prev.events.concat((tl && tl.events) || []),
        since: Date.now() / 1000,
      }));
      setAgents(ag || { ts: 0, agents: [] });
      setSummary(sm);
    } catch (e) {
      console.error("calc refresh:", e);
    }
  }, [taskId]);

  const refreshIterations = useCallback(async () => {
    if (!taskId) return;
    setIterErr(null);
    try {
      const r = await fetch(`/api/calc-theoretical-value/${taskId}/calc/iterations`,
        { cache: "no-store" });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setIterations(await r.json());
    } catch (e) {
      setIterErr(String(e));
    }
  }, [taskId]);

  useEffect(() => {
    setTimeline({ events: [], since: 0 });
    setAgents({ ts: 0, agents: [] });
    setSummary(null);
    setIterations(null);
    setIterErr(null);
    refresh();
    refreshIterations();
  }, [taskId, refresh, refreshIterations]);

  // Poll iterations while the task is running — rounds accumulate on disk.
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refreshIterations, 7000);
    return () => clearInterval(id);
  }, [taskId, refreshIterations]);

  // Poll every 5s while the task is running.
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);

  // Re-compute when batch_size/seq_len change AND step 3 is done.
  const recompute = useCallback(async () => {
    if (!summary?.steps?.s3_calculate?.done) return;
    if (!batchSize || !seqLen) return;
    setComputeErr(null);
    try {
      const r = await fetch(
        `/api/calc-theoretical-value/${taskId}/calc/compute?batch_size=${batchSize}&seq_len=${seqLen}`,
        { cache: "no-store" },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setCompute(await r.json());
    } catch (e) {
      setComputeErr(String(e));
    }
  }, [taskId, summary, batchSize, seqLen]);

  useEffect(() => { recompute(); }, [recompute]);

  const onControl = async (action, extra = {}) => {
    try {
      await controlTask(taskId, action, extra);
      setTimeout(refresh, 400);
    } catch (e) { console.error(e); }
  };

  const phase = run?.current_phase || "idle";
  const finished = !!run?.finished;
  const finalStatus = run?.final_status;
  const running = !!status?.running;

  const vizReady = !!summary?.steps?.s4_visualize?.done;
  const calcReady = !!summary?.steps?.s3_calculate?.done;

  return html`
    <div class="task-detail calc-value-detail">
      <header class="task-header">
        <div class="task-id">
          <span class="label">task</span>
          <code>${run?.task_id || taskId}</code>
          <span class="muted">· ${run?.task_type || "?"}</span>
        </div>
        <div class="task-stats">
          <span class="stat">
            <span class="stat-label">phase</span>
            <span class="pill ${phase}">${labelFor(phase)}</span>
          </span>
          <span class="stat">
            <span class="stat-label">status</span>
            ${finished
              ? html`<span class="pill ${finalStatus}">${finalStatus}</span>`
              : (running
                ? html`<span class="pill running">running</span>`
                : html`<span class="pill idle">idle</span>`)}
          </span>
        </div>
        <div class="task-controls">
          ${running
            ? html`<button class="btn danger"
                onClick=${() => onControl("kill", { force: true })}>Kill</button>`
            : html`<button class="btn ghost"
                onClick=${() => onControl("restart")}>Restart</button>`}
        </div>
      </header>

      <section class="panel calc-steps">
        <h2>Pipeline steps</h2>
        <div class="step-row">
          ${STEPS.map((s) => {
            const info = summary?.steps?.[s.id];
            const done = !!info?.done;
            return html`
              <div class=${`step ${done ? "done" : ""}`}>
                <span class="step-dot">${done ? "✓" : "·"}</span>
                <span class="step-label">${s.label}</span>
                ${s.id === "s2_graph" && info?.node_count != null
                  ? html`<span class="muted">(${info.node_count} nodes)</span>` : null}
                ${s.id === "s3_calculate" && info?.node_count != null
                  ? html`<span class="muted">(${info.node_count} scripts)</span>` : null}
              </div>
            `;
          })}
        </div>
      </section>

      ${vizReady ? html`
        <section class="panel calc-viz-panel">
          <h2>Visualization
            <span class="muted">(iframe loads /calc/viz)</span>
          </h2>
          <div class="calc-controls">
            <label>batch_size:
              <input type="number" min="1" value=${batchSize}
                onInput=${(e) => setBatchSize(parseInt(e.target.value, 10) || 1)} />
            </label>
            <label>seq_len:
              <input type="number" min="1" value=${seqLen}
                onInput=${(e) => setSeqLen(parseInt(e.target.value, 10) || 1)} />
            </label>
            <button onClick=${recompute}>Recompute</button>
            <span class="muted">(controls below are also available inside the iframe)</span>
          </div>
          ${computeErr
            ? html`<div class="task-banner task-banner-err">compute failed: ${computeErr}</div>`
            : null}
          ${compute ? html`
            <div class="calc-totals">
              <span>Total TFLOPs: <strong>${compute.totals.tflops.toFixed(4)}</strong></span>
              <span>Total GB: <strong>${compute.totals.access_gb.toFixed(4)}</strong></span>
              <span>Arithmetic Intensity:
                <strong>${compute.totals.arithmetic_intensity.toFixed(3)}</strong>
                TFLOPs/GB
              </span>
              ${compute.approximate_nodes.length > 0
                ? html`<span class="warn">${compute.approximate_nodes.length} node(s)
                    marked approximate</span>`
                : null}
            </div>
          ` : null}
          <iframe class="calc-iframe"
            src=${`/api/calc-theoretical-value/${taskId}/calc/viz`}
            sandbox="allow-scripts allow-same-origin" />
        </section>
      ` : html`
        <section class="panel calc-viz-panel placeholder">
          <p class="muted">Visualization will appear here once Step 4 completes.</p>
        </section>
      `}

      <${CalcIterations}
        taskId=${taskId}
        iterations=${iterations}
        loading=${!iterations}
        error=${iterErr} />

      <div class="task-grid">
        <section class="panel">
          <h2>Live sub-agents</h2>
          <${AgentsPanel} agents=${agents} />
        </section>

        <section class="panel timeline-panel">
          <h2>Event timeline</h2>
          <${Timeline} events=${timeline.events} />
        </section>
      </div>
    </div>
  `;
}
