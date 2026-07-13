// Task detail view. The dashboard for a single task. Aggregates:
//   - Header strip: task id / type / phase pill / final status / control btns
//   - For calc-theoretical-value tasks: 3 sub-tabs (Rough / Detailed Audit / Status)
//   - For other task types: panel grid (no tabs)
//
// Tab content for "Status" is the legacy 5-panel grid. "Rough" and
// "Detailed Audit" are calc-value-specific components that read the
// S0 / S3 streaming state files.
//
// Pulls everything from files via the per-task API endpoints. SSE drives
// refreshes — when a task_changed event for this id arrives, we refetch
// only the affected panels (hinted by the `changed` array).

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import {
  getRun, getIterations, getTimeline, getCharts,
  getStateGraph, getAgents, controlTask,
} from "app/api";
import { StateGraph } from "app/state-graph";
import { IterationsTable } from "app/iterations-table";
import { Charts } from "app/charts";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";
import { RetrospectiveModal } from "app/retrospective-modal";
import { CalcRoughPanel } from "app/calc-rough-panel";
import { CalcAuditPanel } from "app/calc-audit-panel";
import { CalcVizTab } from "app/calc-viz-tab";
import { labelFor } from "app/utils";

const CALC_TYPE = "calc-theoretical-value";

export function TaskDetailView({ taskId, run, status, onChange, onOpenRetro }) {
  const [iterations, setIterations] = useState([]);
  const [timeline, setTimeline] = useState({ events: [], since: 0 });
  const [charts, setCharts] = useState(null);
  const [graph, setGraph] = useState(null);
  const [agents, setAgents] = useState({ ts: 0, agents: [] });
  const [selectedIter, setSelectedIter] = useState(null);
  const [loadState, setLoadState] = useState("loading"); // loading | ok | error
  const [lastErr, setLastErr] = useState(null);
  const [activeTab, setActiveTab] = useState("rough");

  const withTimeout = (p, ms = 8000) =>
    Promise.race([
      p,
      new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
    ]);

  const refreshAll = useCallback(async () => {
    if (!taskId) return;
    setLastErr(null);
    try {
      const [it, tl, ch, g, ag] = await Promise.all([
        withTimeout(getIterations(taskId)).catch((e) => { console.warn("iterations:", e); return []; }),
        withTimeout(getTimeline(taskId, timeline.since || 0))
          .catch((e) => { console.warn("timeline:", e); return { events: [] }; }),
        withTimeout(getCharts(taskId)).catch((e) => { console.warn("charts:", e); return null; }),
        withTimeout(getStateGraph(taskId)).catch((e) => { console.warn("state-graph:", e); return null; }),
        withTimeout(getAgents(taskId)).catch((e) => { console.warn("agents:", e); return { ts: 0, agents: [] }; }),
      ]);
      setIterations(it || []);
      setTimeline((prev) => ({
        events: prev.events.concat((tl && tl.events) || []),
        since: Date.now() / 1000,
      }));
      setCharts(ch);
      setGraph(g);
      setAgents(ag || { ts: 0, agents: [] });
      setLoadState("ok");
    } catch (e) {
      console.error("refreshAll failed:", e);
      setLastErr(String(e));
      setLoadState("error");
    }
  }, [taskId]);

  useEffect(() => {
    setIterations([]);
    setTimeline({ events: [], since: 0 });
    setCharts(null);
    setGraph(null);
    setAgents({ ts: 0, agents: [] });
    setSelectedIter(null);
    setLoadState("loading");
    setLastErr(null);
    refreshAll();
  }, [taskId, refreshAll]);

  useEffect(() => {
    if (onChange == null) return;
    refreshAll();
  }, [onChange, refreshAll]);

  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refreshAll, 5000);
    return () => clearInterval(id);
  }, [taskId, refreshAll]);

  const onControl = async (action, extra = {}) => {
    try {
      await controlTask(taskId, action, extra);
      setTimeout(refreshAll, 400);
    } catch (e) {
      console.error(e);
    }
  };

  const phase = run?.current_phase || "idle";
  const finished = !!run?.finished;
  const finalStatus = run?.final_status;
  const running = !!status?.running;
  const isCalc = (run?.task_type || "") === CALC_TYPE;

  const retroIter = selectedIter != null
    ? (typeof selectedIter === "number" ? selectedIter : null)
    : null;

  const renderTabs = () => {
    if (!isCalc) return null;
    const tabs = [
      { id: "rough",   label: "粗略评估" },
      { id: "audit",   label: "详细审计" },
      { id: "viz",     label: "可视化" },
      { id: "runtime", label: "运行状态" },
    ];
    return html`
      <nav class="task-tabs">
        ${tabs.map((t) => html`
          <button
            key=${t.id}
            class=${`task-tab ${activeTab === t.id ? "active" : ""}`}
            onClick=${() => setActiveTab(t.id)}>
            ${t.label}
          </button>
        `)}
      </nav>
    `;
  };

  const renderRuntimePanels = () => html`
    <div class="task-grid">
      <section class="panel">
        <h2>State machine</h2>
        <${StateGraph} graph=${graph} />
      </section>

      <section class="panel">
        <h2>Iterations <span class="muted">(click for retrospective)</span></h2>
        <${IterationsTable}
          iterations=${iterations}
          selectedN=${retroIter}
          onSelect=${(n) => setSelectedIter(n)} />
      </section>

      <section class="panel">
        <h2>Live sub-agents</h2>
        <${AgentsPanel} agents=${agents} />
      </section>

      <section class="panel">
        <h2>Performance &amp; duration</h2>
        <${Charts} payload=${charts} />
      </section>

      <section class="panel timeline-panel">
        <h2>Event timeline</h2>
        <${Timeline} events=${timeline.events} />
      </section>
    </div>
  `;

  const renderBody = () => {
    if (!isCalc) {
      return renderRuntimePanels();
    }
    if (activeTab === "rough") {
      return html`<${CalcRoughPanel} taskId=${taskId} phase=${phase} />`;
    }
    if (activeTab === "audit") {
      return html`<${CalcAuditPanel} taskId=${taskId} phase=${phase} timelineEvents=${timeline.events} />`;
    }
    if (activeTab === "viz") {
      return html`<${CalcVizTab} taskId=${taskId} />`;
    }
    return renderRuntimePanels();
  };

  return html`
    <div class="task-detail">
      ${lastErr ? html`
        <div class="task-banner task-banner-err">
          <strong>刷新失败：</strong> ${lastErr}
          <span class="muted">（轮询会自动重试）</span>
        </div>` : null}
      <header class="task-header">
        <div class="task-id">
          <span class="label">task</span>
          <code>${run?.task_id || taskId}</code>
          <span class="muted">· ${run?.task_type || "?"}</span>
        </div>
        <div class="task-stats">
          <span class="stat">
            <span class="stat-label">iter</span>
            <strong>${run?.current_iteration ?? 0}</strong>
          </span>
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

      ${renderTabs()}
      ${renderBody()}

      ${selectedIter != null ? html`
        <${RetrospectiveModal}
          taskId=${taskId}
          iteration=${selectedIter}
          onClose=${() => setSelectedIter(null)} />
      ` : null}
    </div>
  `;
}
