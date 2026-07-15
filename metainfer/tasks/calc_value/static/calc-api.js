// calc-theoretical-value API helpers. Task-package-local — the shell's
// app/api.js only carries task-agnostic endpoints; anything prefixed with
// /calc/... lives here so other task packages don't have to import (or
// edit) calc_value's URLs.

const TASK_SCOPE = (taskId) => `/api/calc-theoretical-value/${encodeURIComponent(taskId)}`;

export async function getCalcRough(taskId, batchSize, seqLen) {
  let url = `${TASK_SCOPE(taskId)}/calc/rough`;
  if (batchSize != null && seqLen != null) {
    url += `?batch_size=${encodeURIComponent(batchSize)}&seq_len=${encodeURIComponent(seqLen)}`;
  }
  const r = await fetch(url, { cache: "no-store" });
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
