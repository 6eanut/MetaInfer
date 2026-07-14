// gen-infer-framework task detail body.
//
// Rendered by the task-detail shell when detail_view_module === "app/gf-detail".
// Receives shared data (iterations, timeline, charts, graph, agents) as props
// from the shell — the shell owns data fetching so multiple plugins don't
// duplicate the work. This component just composes the panels.
//
// Layout: 5-panel grid (state-graph / iterations / live agents / charts /
// timeline) + a "no iterations yet" empty state when the orchestrator hasn't
// started the first iteration.

import { html } from "htm/preact";
import { StateGraph } from "app/state-graph";
import { IterationsTable } from "app/iterations-table";
import { Charts } from "app/charts";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";

export default function GenInferDetailView({
  taskId,
  run,
  status,
  data,
  onOpenRetro,
}) {
  const { iterations, timeline, charts, graph, agents, loadState, lastErr } = data;
  const retroIter = data.selectedIter != null
    ? (typeof data.selectedIter === "number" ? data.selectedIter : null)
    : null;

  if (loadState === "error" && lastErr) {
    return html`
      <div class="task-banner task-banner-err">
        <strong>刷新失败：</strong> ${lastErr}
        <span class="muted">（轮询会自动重试）</span>
      </div>
    `;
  }

  return html`
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
}
