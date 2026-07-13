// Thin fetch wrappers. Every call returns parsed JSON or throws.
//
// All endpoints are task-scoped: pass `taskId` everywhere. The API base
// is `/api/tasks/<id>/...`. Task-list / type metadata sit at the top level.

export async function listTaskTypes() {
  const r = await fetch("/api/task-types");
  if (!r.ok) throw new Error(`task-types: ${r.status}`);
  return r.json();
}

export async function loadFormSchema(taskType) {
  const r = await fetch(`/api/task-types/${encodeURIComponent(taskType)}/schema`);
  if (!r.ok) throw new Error(`schema ${taskType}: ${r.status}`);
  return r.json();
}

export async function listTasks() {
  const r = await fetch("/api/tasks");
  if (!r.ok) throw new Error(`tasks: ${r.status}`);
  return r.json();
}

export async function getTask(taskId) {
  const r = await fetch(`/api/tasks/${encodeURIComponent(taskId)}`);
  if (!r.ok) throw new Error(`task ${taskId}: ${r.status}`);
  return r.json();
}

export async function createTask(payload) {
  const r = await fetch("/api/tasks", {
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
    `/api/tasks/${encodeURIComponent(taskId)}?purge=${purge ? "1" : "0"}`,
    { method: "DELETE" },
  );
  if (!r.ok) throw new Error(`delete ${taskId}: ${r.status}`);
  return r.json();
}

export async function controlTask(taskId, action, extra = {}) {
  const r = await fetch(`/api/tasks/${encodeURIComponent(taskId)}/control`, {
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

// ---- Per-task observable panels (all read files under state_dir) -------- //

const TASK_SCOPE = (taskId) => `/api/tasks/${encodeURIComponent(taskId)}`;

export async function getRun(taskId) {
  const r = await fetch(`${TASK_SCOPE(taskId)}/run`);
  if (!r.ok) throw new Error(`run: ${r.status}`);
  return r.json();
}

export async function getIterations(taskId) {
  const r = await fetch(`${TASK_SCOPE(taskId)}/iterations`);
  if (!r.ok) throw new Error(`iterations: ${r.status}`);
  return r.json();
}

export async function getRetrospective(taskId, n) {
  const r = await fetch(
    `${TASK_SCOPE(taskId)}/iterations/${n}/retrospective`,
  );
  if (!r.ok) throw new Error(`retro ${n}: ${r.status}`);
  return r.json();
}

export async function getTimeline(taskId, since = 0) {
  const r = await fetch(
    `${TASK_SCOPE(taskId)}/timeline?since=${since}`,
  );
  if (!r.ok) throw new Error(`timeline: ${r.status}`);
  return r.json();
}

export async function getCharts(taskId) {
  const r = await fetch(`${TASK_SCOPE(taskId)}/charts`);
  if (!r.ok) throw new Error(`charts: ${r.status}`);
  return r.json();
}

export async function getStateGraph(taskId) {
  const r = await fetch(`${TASK_SCOPE(taskId)}/state-graph`);
  if (!r.ok) throw new Error(`state-graph: ${r.status}`);
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

export async function getLog(taskId, tailBytes = 65536) {
  const r = await fetch(
    `${TASK_SCOPE(taskId)}/log?tail_bytes=${tailBytes}`,
  );
  if (!r.ok) throw new Error(`log: ${r.status}`);
  return r.json();
}

// ---- calc-theoretical-value: rough + streaming cells --------------------- //

export async function getCalcRough(taskId) {
  const r = await fetch(`${TASK_SCOPE(taskId)}/calc/rough`,
    { cache: "no-store" });
  if (!r.ok) throw new Error(`calc/rough: ${r.status}`);
  return r.json();
}

export async function getCalcCells(taskId, batchSize, seqLen) {
  let url = `${TASK_SCOPE(taskId)}/calc/cells`;
  // Only append query params when an explicit combo is requested; without
  // them the server returns the default picked values from _state.json.
  if (batchSize != null && seqLen != null) {
    url += `?batch_size=${encodeURIComponent(batchSize)}&seq_len=${encodeURIComponent(seqLen)}`;
  }
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`calc/cells: ${r.status}`);
  return r.json();
}

export async function getCalcCellDetail(taskId, compound, angle, roundIdx) {
  const r = await fetch(
    `${TASK_SCOPE(taskId)}/calc/cell/${encodeURIComponent(compound)}` +
    `/${encodeURIComponent(angle)}/${encodeURIComponent(roundIdx)}`,
    { cache: "no-store" },
  );
  if (!r.ok) throw new Error(`calc/cell: ${r.status}`);
  return r.json();
}

export async function startCalcCellQa(taskId, payload) {
  const r = await fetch(`${TASK_SCOPE(taskId)}/calc/qa/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await r.json().catch(() => ({}));
  if (!r.ok) {
    const err = new Error(`qa/start: ${r.status}`);
    err.detail = data.detail || data;
    throw err;
  }
  return data;
}

export async function getCalcQa(taskId, sessionId) {
  const r = await fetch(
    `${TASK_SCOPE(taskId)}/calc/qa/${encodeURIComponent(sessionId)}`,
    { cache: "no-store" },
  );
  if (!r.ok) throw new Error(`qa/get: ${r.status}`);
  return r.json();
}
