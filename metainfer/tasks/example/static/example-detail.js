/** Example detail-view body component.

  Rendered by the task-detail shell when ``detail_view_module === "app/X-detail"``.
  Replace "X-detail" / "X-state-graph" / "X-iterations-table" / "X-charts" with
  your task's widget names.

  Shell passes these props:
    - taskId:  string
    - run:     RunStatus object (current_phase, current_iteration, …)
    - status:  "running" | "done" | "idle" | …
    - data:    { run, timeline, agents, loadState, lastErr, refreshShell }

  Fetch your own data (iterations / charts / state-graph) from
  ``/api/<type>/<taskId>/...`` using your task's X-runtime-api.js.

  importmap auto-discovery: every ``*.js`` file under your ``static/`` dir
  becomes ``app/<stem>``.  So ``X-detail.js`` → ``app/X-detail``,
  ``X-charts.js`` → ``app/X-charts``, etc.
*/

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";

// Replace "X" with your task name in the import below.
// import { StateGraph } from "app/X-state-graph";
// import { IterationsTable } from "app/X-iterations-table";
// import { Charts } from "app/X-charts";
// import { getIterations, getCharts, getStateGraph } from "app/X-runtime-api";

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

function useRuntimeData(taskId) {
  const [data, setData] = useState({ iterations: [], charts: null, graph: null });
  const refresh = useCallback(async () => {
    if (!taskId) return;
    // In a real task, uncomment these:
    // const [it, ch, g] = await Promise.all([
    //   withTimeout(getIterations(taskId)).catch((e) => { console.warn(e); return []; }),
    //   withTimeout(getCharts(taskId)).catch((e) => { console.warn(e); return null; }),
    //   withTimeout(getStateGraph(taskId)).catch((e) => { console.warn(e); return null; }),
    // ]);
    // setData({ iterations: it || [], charts: ch, graph: g });
  }, [taskId]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);
  return { ...data, refresh };
}

export default function XDetailView({ taskId, run, status, data }) {
  const { timeline, agents, loadState, lastErr } = data;
  const rt = useRuntimeData(taskId);

  if (loadState === "error" && lastErr) {
    return html`<div class="task-banner task-banner-err">
      <strong>Refresh failed:</strong> ${lastErr}
    </div>`;
  }

  return html`
    <div class="task-grid">
      <section class="panel">
        <h2>State machine</h2>
        <!-- <${StateGraph} graph=${rt.graph} /> -->
      </section>
      <section class="panel">
        <h2>Iterations</h2>
        <!-- <${IterationsTable} iterations=${rt.iterations} /> -->
      </section>
      <section class="panel">
        <h2>Live sub-agents</h2>
        <!-- Use <${AgentsPanel} agents=${agents} /> from app/agents-panel -->
      </section>
      <section class="panel">
        <h2>Performance & duration</h2>
        <!-- <${Charts} payload=${rt.charts} /> -->
      </section>
      <section class="panel timeline-panel">
        <h2>Event timeline</h2>
        <!-- <${Timeline} events=${timeline.events} /> from app/timeline -->
      </section>
    </div>
  `;
}
