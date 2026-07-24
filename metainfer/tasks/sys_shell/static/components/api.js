// Thin fetch wrappers. Every call returns parsed JSON or throws.
//
// All endpoints are task-scoped: pass `taskId` everywhere. The API base
// is `/api/sys-shell/<id>/...`. Task-type endpoints live at `/api/<type>/<id>/...`.

export async function listTaskTypes() {
  const r = await fetch("/api/sys-shell/task-types");
  if (!r.ok) throw new Error(`task-types: ${r.status}`);
  return r.json();
}

export async function loadFormSchema(taskType) {
  const r = await fetch(`/api/sys-shell/task-types/${encodeURIComponent(taskType)}/schema`);
  if (!r.ok) throw new Error(`schema ${taskType}: ${r.status}`);
  return r.json();
}

export async function listTasks() {
  const r = await fetch("/api/sys-shell/tasks");
  if (!r.ok) throw new Error(`tasks: ${r.status}`);
  return r.json();
}

export async function getTask(taskId) {
  const r = await fetch(`/api/sys-shell/${encodeURIComponent(taskId)}`);
  if (!r.ok) throw new Error(`task ${taskId}: ${r.status}`);
  return r.json();
}

export async function createTask(payload) {
  const r = await fetch("/api/sys-shell/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(`create: ${r.status}`);
    // FastAPI's HTTPException wraps the payload in {detail: ...}; unwrap
    // it so callers can read err.detail.errors / err.detail.error directly.
    err.detail = data.detail || data;
    throw err;
  }
  return data;
}

export async function deleteTask(taskId, purge = false) {
  const r = await fetch(
    `/api/sys-shell/${encodeURIComponent(taskId)}?purge=${purge ? "1" : "0"}`,
    { method: "DELETE" },
  );
  if (!r.ok) throw new Error(`delete ${taskId}: ${r.status}`);
  return r.json();
}

export async function controlTask(taskId, action, extra = {}) {
  const r = await fetch(`/api/sys-shell/${encodeURIComponent(taskId)}/control`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ action, ...extra }),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(`control ${action}: ${r.status}`);
    err.detail = data.detail || data;
    throw err;
  }
  return data;
}

// ---- Shell-owned observable endpoints ------------------------------------ //
//
// The shell only fetches what its OWN chrome renders:
//   - run.json           → header (phase pill / iter counter / controls)
//   - timeline.jsonl     → activity feed
//   - agents.json        → sub-agent panel
//   - token_budget.json  → budget bar
//   - orchestrator.log   → log tail
//
// Task-specific data (iterations / charts / state-graph / retrospective)
// is fetched by each plugin's own runtime-api module against
// /api/<type>/<id>/... . The shell stays ignorant of those shapes.

const TASK_SCOPE = (taskId) => `/api/sys-shell/${encodeURIComponent(taskId)}`;

export async function getRun(taskId) {
  const r = await fetch(`${TASK_SCOPE(taskId)}/run`);
  if (!r.ok) throw new Error(`run: ${r.status}`);
  return r.json();
}

export async function getTimeline(taskId, since = 0) {
  const r = await fetch(
    `${TASK_SCOPE(taskId)}/timeline?since=${since}`,
  );
  if (!r.ok) throw new Error(`timeline: ${r.status}`);
  return r.json();
}

export async function getAgents(taskId) {
  // cache:"no-store" bypasses the browser's heuristic cache AND any
  // intermediate proxy cache. The Live sub-agents panel polls this
  // every few seconds; without no-store, GETs frequently return stale
  // 304s or the previously-cached body even though the orchestrator
  // is actively rewriting agents.json. The server also sends
  // Cache-Control: no-store now, but we belt-and-suspenders it here
  // in case a proxy strips that header.
  const r = await fetch(`${TASK_SCOPE(taskId)}/agents`, { cache: "no-store" });
  if (!r.ok) throw new Error(`agents: ${r.status}`);
  return r.json();
}

export async function getAgentTail(taskId, agentName, maxEvents = 50) {
  // Tail of one agent's stream-json activity (text responses + tool uses).
  // Used by the expandable row in AgentsPanel so the operator can see what
  // an agent is currently doing without leaving the WebUI.
  const url = `${TASK_SCOPE(taskId)}/agents/${encodeURIComponent(agentName)}/tail?max_events=${maxEvents}`;
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) {
    // 404 when agent isn't in the snapshot anymore (finished + rotated out)
    if (r.status === 404) return { agent_name: agentName, found: false, events: [] };
    throw new Error(`agent tail: ${r.status}`);
  }
  return r.json();
}

export async function getTokenBudget(taskId) {
  // Returns the task's cost-budget snapshot. 200 with {configured:false}
  // when no budget file exists yet — caller should render nothing.
  const r = await fetch(`${TASK_SCOPE(taskId)}/token-budget`,
                        { cache: "no-store" });
  if (!r.ok) throw new Error(`token-budget: ${r.status}`);
  return r.json();
}

export async function updateTokenBudget(taskId, payload) {
  // payload: { max_cost_usd?: number|null, max_cost_usd_hard?: number|null }
  // Returns the post-update snapshot (same shape as getTokenBudget).
  const r = await fetch(`${TASK_SCOPE(taskId)}/token-budget`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload || {}),
  });
  if (!r.ok) {
    let detail = `${r.status}`;
    try { detail += `: ${(await r.json()).detail}`; } catch (_) { /* ignore */ }
    throw new Error(`token-budget update: ${detail}`);
  }
  return r.json();
}

export async function getLog(taskId, tailBytes = 65536) {
  const r = await fetch(
    `${TASK_SCOPE(taskId)}/log?tail_bytes=${tailBytes}`,
  );
  if (!r.ok) throw new Error(`log: ${r.status}`);
  return r.json();
}
