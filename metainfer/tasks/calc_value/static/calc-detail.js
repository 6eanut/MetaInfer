// calc-theoretical-value task detail body.
//
// Rendered by the task-detail shell when detail_view_module === "app/calc-detail".
// Owns the tab UI (rough / audit / viz / runtime). The first three tabs are
// calc-specific components; "runtime" reuses the shared panel grid by
// delegating to GenInferDetailView-style layout (state-graph / iterations /
// agents / charts / timeline).
//
// Receives shared data as props from the shell — same contract as
// gf-detail.js.

import { html } from "htm/preact";
import { useState } from "preact/hooks";
import { StateGraph } from "app/state-graph";
import { IterationsTable } from "app/iterations-table";
import { Charts } from "app/charts";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";
import { CalcRoughPanel } from "app/calc-rough-panel";
import { CalcAuditPanel } from "app/calc-audit-panel";
import { CalcVizTab } from "app/calc-viz-tab";

const TABS = [
  { id: "rough",   label: "粗略评估" },
  { id: "audit",   label: "详细审计" },
  { id: "viz",     label: "可视化" },
  { id: "runtime", label: "运行状态" },
];

export default function CalcDetailView({
  taskId,
  run,
  status,
  data,
  onOpenRetro,
}) {
  const [activeTab, setActiveTab] = useState("rough");
  const phase = run?.current_phase || "idle";
  const { iterations, timeline, charts, graph, agents } = data;
  const retroIter = data.selectedIter != null
    ? (typeof data.selectedIter === "number" ? data.selectedIter : null)
    : null;

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
        <${StateGraph} graph=${graph} />
      </section>
      <section class="panel">
        <h2>Iterations <span class="muted">(click for retrospective)</span></h2>
        <${IterationsTable}
          iterations=${iterations}
          selectedN=${retroIter}
          onSelect=${(n) => onOpenRetro && onOpenRetro(n)} />
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
  `;
}
