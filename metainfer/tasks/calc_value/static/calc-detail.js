// calc-theoretical-value task detail body.
//
// Rendered by the task-detail shell when detail_view_module === "app/calc-detail".
// Owns the tab UI (rough / audit / viz / runtime). The first three tabs are
// calc-specific components; "runtime" fetches its own iterations/charts/graph
// data from /api/calc-theoretical-value/<id>/* and renders the panel grid using
// calc-shipped widgets (calc-state-graph / calc-iterations-table / calc-charts
// plus the shell-shared agents-panel / timeline).
//
// Shell passes {run, timeline, agents, loadState, lastErr} via the `data`
// prop — agents and timeline come from shell-owned endpoints; everything
// else is fetched here.

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { StateGraph } from "app/calc-state-graph";
import { IterationsTable } from "app/calc-iterations-table";
import { Charts } from "app/calc-charts";
import { RetrospectiveModal } from "app/calc-retrospective-modal";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";
import { CalcRoughPanel } from "app/calc-rough-panel";
import { CalcAuditPanel } from "app/calc-audit-panel";
import { CalcVizTab } from "app/calc-viz-tab";
import {
  getIterations, getCharts, getStateGraph,
} from "app/calc-runtime-api";

const TABS = [
  { id: "rough",   label: "粗略评估" },
  { id: "audit",   label: "详细审计" },
  { id: "viz",     label: "可视化" },
  { id: "runtime", label: "运行状态" },
];

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

function useRuntimeData(taskId) {
  // Fetches iterations / charts / graph — calc-specific endpoints
  // mounted by calc_value's web_server_handler router.
  const [data, setData] = useState({
    iterations: [], charts: null, graph: null,
  });
  const refresh = useCallback(async () => {
    if (!taskId) return;
    const [it, ch, g] = await Promise.all([
      withTimeout(getIterations(taskId)).catch((e) => { console.warn("iterations:", e); return []; }),
      withTimeout(getCharts(taskId)).catch((e) => { console.warn("charts:", e); return null; }),
      withTimeout(getStateGraph(taskId)).catch((e) => { console.warn("state-graph:", e); return null; }),
    ]);
    setData({ iterations: it || [], charts: ch, graph: g });
  }, [taskId]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);
  return { ...data, refresh };
}

export default function CalcDetailView({
  taskId,
  run,
  status,
  data,
}) {
  const [activeTab, setActiveTab] = useState("rough");
  const [selectedIter, setSelectedIter] = useState(null);
  const phase = run?.current_phase || "idle";
  const { timeline, agents } = data;
  const rt = useRuntimeData(taskId);

  const renderTabs = () => html`
    <nav class="task-tabs">
      ${TABS.map((t) => html`
        <button
          key=${t.id}
          class=${`task-tab ${activeTab === t.id ? "active" : ""}`}
          onClick=${() => setActiveTab(t.id)}>
          ${t.label}
        </button>
      `)}
    </nav>
  `;

  const renderRuntimePanels = () => html`
    <div class="task-grid">
      <section class="panel">
        <h2>State machine</h2>
        <${StateGraph} graph=${rt.graph} />
      </section>
      <section class="panel">
        <h2>Iterations <span class="muted">(click for retrospective)</span></h2>
        <${IterationsTable}
          iterations=${rt.iterations}
          selectedN=${selectedIter}
          onSelect=${(n) => setSelectedIter(n)} />
      </section>
      <section class="panel">
        <h2>Live sub-agents</h2>
        <${AgentsPanel} agents=${agents} />
      </section>
      <section class="panel">
        <h2>Performance &amp; duration</h2>
        <${Charts} payload=${rt.charts} />
      </section>
      <section class="panel timeline-panel">
        <h2>Event timeline</h2>
        <${Timeline} events=${timeline.events} />
      </section>
    </div>
  `;

  const renderBody = () => {
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
    ${renderTabs()}
    ${renderBody()}
    ${selectedIter != null ? html`
      <${RetrospectiveModal}
        taskId=${taskId}
        iteration=${selectedIter}
        onClose=${() => setSelectedIter(null)} />
    ` : null}
  `;
}
