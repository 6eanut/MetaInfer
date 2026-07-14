// Task detail shell. Owns the chrome shared across every task type:
//   - Header strip: task id / type / phase pill / final status / control btns
//   - Budget bar
//   - Reset / Retrospective modals
//   - Shared data fetching (iterations / timeline / charts / state-graph / agents)
//
// The BODY (tab layout, panel composition) is rendered by the active task
// type's plugin via dynamic import. The backend hands us a
// `detail_view_module` importmap key (e.g. "app/calc-detail", "app/gf-detail");
// we resolve it at render time so this file doesn't need to know which plugins
// exist. Adding a new task type with a custom detail view just requires
// registering an importmap entry — no edits here.
//
// If the import fails (plugin not registered, module missing), we fall back
// to a minimal "no detail view" body so the shell stays usable for debugging.

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import {
  getRun, getIterations, getTimeline, getCharts,
  getStateGraph, getAgents, controlTask,
} from "app/api";
import { RetrospectiveModal } from "app/retrospective-modal";
import { ConfirmActionModal } from "app/confirm-action-modal";
import { BudgetBar } from "app/budget-bar";
import { labelFor } from "app/utils";

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

function usePluginBody(detailViewModule) {
  // Resolve the detail-view component for the active task type. Returns
  // null while loading, or a component reference (or null on miss, which
  // renders the fallback body).
  const [Body, setBody] = useState(null);
  useEffect(() => {
    let cancelled = false;
    if (!detailViewModule) {
      setBody(null);
      return;
    }
    import(/* importmap key */ detailViewModule)
      .then((m) => {
        if (cancelled) return;
        setBody(() => (m && m.default) || null);
      })
      .catch((e) => {
        console.error(`detail view ${detailViewModule} failed to load:`, e);
        if (!cancelled) setBody(null);
      });
    return () => { cancelled = true; };
  }, [detailViewModule]);
  return Body;
}

export function TaskDetailView({ taskId, run, status, onChange, onOpenRetro, label, detailViewModule = null }) {
  const [iterations, setIterations] = useState([]);
  const [timeline, setTimeline] = useState({ events: [], since: 0 });
  const [charts, setCharts] = useState(null);
  const [graph, setGraph] = useState(null);
  const [agents, setAgents] = useState({ ts: 0, agents: [] });
  const [selectedIter, setSelectedIter] = useState(null);
  const [loadState, setLoadState] = useState("loading"); // loading | ok | error
  const [lastErr, setLastErr] = useState(null);
  const [showReset, setShowReset] = useState(false);

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
      throw e;
    }
  };

  const phase = run?.current_phase || "idle";
  const finished = !!run?.finished;
  const finalStatus = run?.final_status;
  const running = !!status?.running;
  const taskName = label || run?.task_id || taskId;

  const Body = usePluginBody(detailViewModule);
  const sharedData = {
    iterations, timeline, charts, graph, agents,
    loadState, lastErr, selectedIter,
  };

  return html`
    <div class="task-detail">
      <${BudgetBar} taskId=${taskId} refreshKey=${onChange} />
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
          ${!running
            ? html`<button class="btn danger"
                disabled=${running}
                title=${running ? "任务运行中，无法重置" : "清除所有迭代/日志，保留原始任务输入"}
                onClick=${() => setShowReset(true)}>Reset</button>`
            : null}
        </div>
      </header>

      ${Body
        ? html`<${Body}
            taskId=${taskId}
            run=${run}
            status=${status}
            data=${sharedData}
            onOpenRetro=${(n) => setSelectedIter(n)} />`
        : html`<div class="task-banner">
            ${detailViewModule
              ? html`<span class="muted">详情视图 ${detailViewModule} 加载失败 — 检查插件是否注册</span>`
              : html`<span class="muted">该任务类型未注册详情视图插件</span>`}
          </div>`}

      ${selectedIter != null ? html`
        <${RetrospectiveModal}
          taskId=${taskId}
          iteration=${selectedIter}
          onClose=${() => setSelectedIter(null)} />
      ` : null}

      ${showReset ? html`
        <${ConfirmActionModal}
          title="重置任务到初始状态"
          promptText="将删除所有日志、迭代产物、调试信息、运行记录，仅保留创建任务时的原始输入（requirements.json）。任务运行次数清零。重置后点击 Restart 即可重新开始。"
          confirmText=${taskName}
          confirmLabel="重置"
          onConfirm=${() => onControl("reset")}
          onClose=${() => setShowReset(false)} />
      ` : null}
    </div>
  `;
}
