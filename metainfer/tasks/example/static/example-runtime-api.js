// X-runtime-api — fetch helpers for task-specific endpoints.
//
// All routes live under /api/<type>/<taskId>/* (mounted by the shell).
// The shell's components/api.js does NOT know about these — every task
// owns its own runtime fetcher.  This file is auto-discovered by
// importmap globbing: X-runtime-api.js → app/X-runtime-api.

const TASK_BASE = (taskId) =>
  `/api/X-type-id/${encodeURIComponent(taskId)}`;

export async function getIterations(taskId) {
  const r = await fetch(`${TASK_BASE(taskId)}/iterations`);
  if (!r.ok) throw new Error(`iterations: ${r.status}`);
  return r.json();
}

export async function getCharts(taskId) {
  const r = await fetch(`${TASK_BASE(taskId)}/charts`);
  if (!r.ok) throw new Error(`charts: ${r.status}`);
  return r.json();
}

export async function getStateGraph(taskId) {
  const r = await fetch(`${TASK_BASE(taskId)}/state-graph`,
                        { cache: "no-store" });
  if (!r.ok) throw new Error(`state-graph: ${r.status}`);
  return r.json();
}
