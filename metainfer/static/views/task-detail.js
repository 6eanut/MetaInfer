// Task detail view. The dashboard for a single task. Aggregates:
//   - Header strip: task id / type / phase pill / final status / control btns
//   - Left rail: state graph + live agents
//   - Right rail: iterations table + perf charts + timeline
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
import { labelFor } from "app/utils";

export function TaskDetailView({ taskId, run, status, onChange, onOpenRetro }) {
  const [iterations, setIterations] = useState([]);
  const [timeline, setTimeline] = useState({ events: [], since: 0 });
  const [charts, setCharts] = useState(null);
  const [graph, setGraph] = useState(null);
  const [agents, setAgents] = useState({ ts: 0, agents: [] });
  const [selectedIter, setSelectedIter] = useState(null);
  const [loadState, setLoadState] = useState("loading"); // loading | ok | error
  const [lastErr, setLastErr] = useState(null);

  // Wrap each fetch with a timeout so one stuck endpoint can't freeze the
  // whole panel. 8s is generous; file reads return in <50ms normally.
  const withTimeout = (p, ms = 8000) =>
    Promise.race([
      p,
      new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
    ]);

  // Initial fetch + polling fallback (in case SSE drops). Task detail
  // polls itself every 5s — SSE is an optimization, not the only path.
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

  // Caller signals refresh via `onChange` (from SSE dispatch).
  useEffect(() => {
    if (onChange == null) return;
    refreshAll();
  }, [onChange, refreshAll]);

  // Self-poll fallback: refresh every 5s whether or not SSE fires. SSE
  // is best-effort; polling guarantees the panel updates for tasks that
  // are still running even if the event stream is wedged.
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

  const retroIter = selectedIter != null
    ? (typeof selectedIter === "number" ? selectedIter : null)
    : null;

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

      ${selectedIter != null ? html`
        <${RetrospectiveModal}
          taskId=${taskId}
          iteration=${selectedIter}
          onClose=${() => setSelectedIter(null)} />
      ` : null}
    </div>
  `;
}
