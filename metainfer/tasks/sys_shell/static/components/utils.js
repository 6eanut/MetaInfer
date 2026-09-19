// Small format helpers shared across panels. Pure functions, no deps.

export function fmtDur(s) {
  if (s == null || Number.isNaN(s)) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const r = Math.round(s % 60);
  if (m < 60) return `${m}m${r}s`;
  const h = Math.floor(m / 60);
  return `${h}h${m % 60}m`;
}

export function fmtAgo(s) {
  if (s == null || Number.isNaN(s)) return "—";
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

export function fmtTs(ts) {
  if (!ts) return "—";
  const d = new Date(ts * 1000);
  return d.toLocaleTimeString();
}

export function fmtRelative(ts) {
  if (!ts) return "—";
  const diff = Date.now() / 1000 - ts;
  if (diff < 0) return "just now";
  return fmtAgo(diff);
}

export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;",
    '"': "&quot;", "'": "&#39;",
  })[c]);
}

// Timeline batching. The events feed is polled incrementally: each request
// asks for events `since` the last cursor, and the result is appended to what
// the view already holds. The endpoint filters with `ts >= since`, so the
// newest event of one batch comes back again on the next; this drops that
// boundary event and advances the cursor to the newest timestamp actually
// seen. Returning an empty `fresh` lets the caller skip a render. Pure, so it
// is unit-tested without a DOM (see components/utils.test.mjs).
export function takeTimelineBatch(batch, cursor = 0) {
  const events = Array.isArray(batch) ? batch : [];
  const ts = (ev) => (ev && Number(ev.ts)) || 0;
  const since = events.reduce((m, ev) => Math.max(m, ts(ev)), cursor || 0);
  const fresh = cursor > 0 ? events.filter((ev) => ts(ev) > cursor) : events;
  return { fresh, since };
}

// Phase label cache — populated by state-graph fetches. The backend is the
// source of truth for labels (reads phases.py), so the frontend just caches.
const _phaseLabels = new Map();
export function setPhaseLabels(nodes = []) {
  for (const n of nodes) {
    if (n?.id && n?.label) _phaseLabels.set(n.id, n.label);
  }
}
export function labelFor(id) {
  return _phaseLabels.get(id) || id;
}
