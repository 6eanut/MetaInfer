// SPA entry. Renders the shell (topbar + tabstrip + workspace), drives
// task list polling, and dispatches SSE task_changed events to the
// currently-active task view.
//
// No router — view state is just: { activeTabId | null, showNewTask | false }.
// A null activeTabId shows an empty-state.

import { html, render } from "htm/preact";
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import {
  listTasks, getTask,
} from "app/api";
import { TaskDetailView } from "app/task-detail";
import { NewTaskView } from "app/new-task";
import { labelFor } from "app/utils";

function App() {
  // List of tasks (registry view, lightweight).
  const [tasks, setTasks] = useState([]);
  // Per-task cache: { [id]: { run, status, lastFetched } } — drives the
  // active task view + the tabstrip pills.
  const [taskCache, setTaskCache] = useState({});
  // Active task id (which tab is in focus). null = empty state.
  const [activeId, setActiveId] = useState(null);
  // New task overlay visibility.
  const [showNewTask, setShowNewTask] = useState(false);
  // Refresh counter — bumping drives the detail view to refetch.
  const [refreshTick, setRefreshTick] = useState(0);
  // Last list fetch error.
  const [listErr, setListErr] = useState(null);

  // SSE subscription. Single connection, lives for the app's lifetime.
  const sseRef = useRef(null);

  const refreshList = useCallback(async () => {
    try {
      const data = await listTasks();
      setTasks(data.tasks || []);
      setListErr(null);
    } catch (e) {
      setListErr(String(e));
    }
  }, []);

  // Initial list fetch.
  useEffect(() => { refreshList(); }, [refreshList]);

  // Polling fallback for the list (SSE only fires on file changes; new
  // task spawns + process deaths happen between SSE pings too).
  useEffect(() => {
    const id = setInterval(refreshList, 5000);
    return () => clearInterval(id);
  }, [refreshList]);

  // SSE setup.
  useEffect(() => {
    let es;
    try {
      es = new EventSource("/api/events");
      es.addEventListener("hello", () => { /* stream alive */ });
      es.addEventListener("task_changed", (e) => {
        try {
          const ev = JSON.parse(e.data);
          // Trigger refresh of the list view (status pill updates).
          refreshList();
          // If the changed task is active, bump the refresh tick so the
          // detail view refetches its panels.
          if (ev.task_id === activeIdRef.current) {
            setRefreshTick((n) => n + 1);
          }
        } catch (_) { /* ignore */ }
      });
      es.onerror = () => { /* browser auto-reconnects */ };
      sseRef.current = es;
    } catch (_) { /* SSE unsupported; polling covers us */ }
    return () => { if (es) es.close(); };
  }, [refreshList]);

  // Whenever the active task changes, fetch its run + status for the
  // header. We DON'T depend on `tasks` here (the list polling mutates
  // that array every 5s, which would cancel in-flight getTask requests
  // before they resolve). Polling fallback below keeps the cache fresh.
  const activeIdRef = useRef(activeId);
  activeIdRef.current = activeId;
  const fetchActiveTask = useCallback(async () => {
    if (!activeId) return;
    try {
      const t = await getTask(activeId);
      // Guard against tab switches during fetch.
      if (activeIdRef.current !== activeId) return;
      setTaskCache((prev) => ({
        ...prev,
        [activeId]: {
          run: t.run, status: t.status,
          type: t.type, label: t.label,
        },
      }));
    } catch (e) {
      // Don't silently swallow — surface in console for debugging.
      console.warn("getTask failed:", e);
    }
  }, [activeId]);

  useEffect(() => {
    if (!activeId) return;
    fetchActiveTask();
    // Poll every 5s as a fallback (SSE only fires on file mtime changes;
    // orchestrator exit / pid file mutation isn't always caught).
    const id = setInterval(fetchActiveTask, 5000);
    return () => clearInterval(id);
  }, [activeId, fetchActiveTask, refreshTick]);

  // Tabs derived from registry order (newest first).
  const tabs = [...tasks].sort((a, b) => b.created_at - a.created_at);
  const active = activeId
    ? tabs.find((t) => t.id === activeId)
    : null;
  const cached = activeId ? taskCache[activeId] : null;

  const onNewTask = (newId) => {
    setShowNewTask(false);
    setActiveId(newId);
    refreshList();
  };

  const counts = (() => {
    let active_ = 0, done = 0;
    for (const t of tasks) {
      if (t.status?.running) active_++;
      else done++;
    }
    return { active_, done };
  })();

  return html`
    <${Shell}
      tabs=${tabs}
      activeId=${activeId}
      onSelectTab=${setActiveId}
      onCloseTab=${(id) => {
        // Closing a tab just deselects it; the registry stays intact.
        if (activeId === id) setActiveId(null);
      }}
      counts=${counts}
      listErr=${listErr}
      onNewTask=${() => setShowNewTask(true)}
      onRefresh=${refreshList}>
      ${active
        ? html`<${TaskDetailView}
            taskId=${active.id}
            run=${cached?.run}
            status=${active.status}
            onChange=${refreshTick}
            onOpenRetro=${() => {}} />`
        : html`<${EmptyState} onNewTask=${() => setShowNewTask(true)} />`}
    </${Shell}>

    ${showNewTask ? html`
      <${NewTaskView}
        onClose=${() => setShowNewTask(false)}
        onCreated=${onNewTask} />
    ` : null}
  `;
}

function Shell({
  tabs, activeId, onSelectTab, onCloseTab,
  counts, listErr, onNewTask, onRefresh, children,
}) {
  return html`
    <div class="topbar">
      <div class="topbar-left">
        <span class="brand">MetaInfer</span>
        <span class="topbar-stats">
          <span class="dot active"></span>${counts?.active_ ?? 0} active
          <span class="dot done"></span>${counts?.done ?? 0} done
        </span>
        ${listErr ? html`<span class="topbar-err">${listErr}</span>` : null}
      </div>
      <div class="topbar-right">
        <button class="btn ghost" onClick=${onRefresh}>↻</button>
        <button class="btn primary" onClick=${onNewTask}>+ New Task</button>
      </div>
    </div>

    <div class="tabstrip">
      ${tabs.length === 0
        ? html`<span class="tab-empty muted">no tasks yet</span>`
        : tabs.map((t) => html`
          <button key=${t.id}
              class=${`tab ${activeId === t.id ? "active" : ""}`}
              onClick=${() => onSelectTab(t.id)}>
            <span class=${`dot ${t.status?.running ? "active" : "done"}`}></span>
            <span class="tab-label">${t.label || t.id}</span>
            <span class="tab-id">${t.id}</span>
            <span class="tab-close"
              onClick=${(e) => { e.stopPropagation(); onCloseTab(t.id); }}>×</span>
          </button>
        `)}
    </div>

    <main class="workspace">${children}</main>
  `;
}

function EmptyState({ onNewTask }) {
  return html`
    <div class="empty-state">
      <h2>No task selected</h2>
      <p class="muted">Pick a tab above to inspect a running or finished task,
        or start a new one.</p>
      <button class="btn primary" onClick=${onNewTask}>+ New Task</button>
    </div>
  `;
}

render(html`<${App} />`, document.getElementById("app"));
