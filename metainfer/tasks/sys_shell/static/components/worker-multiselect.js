// Worker-multi-select widget: dynamically loads the worker list from
// /api/cluster/workers and renders checkboxes. Selected values are stored as
// a comma-joined string (form.yaml convention for text fields).
//
// Registered under name "worker-multiselect".

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";

export function WorkerMultiSelect({ field, value, onChange }) {
  const [workers, setWorkers] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/cluster/workers")
      .then((r) => r.json())
      .then((data) => {
        if (!cancelled) {
          setWorkers(data || []);
          setError(null);
        }
      })
      .catch((e) => {
        if (!cancelled) setError(String(e));
      });
    return () => { cancelled = true; };
  }, []);

  // Parse value (comma-string or array) into a Set.
  const selected = new Set(
    Array.isArray(value)
      ? value
      : (typeof value === "string" && value.trim())
        ? value.split(",").map((s) => s.trim()).filter(Boolean)
        : []
  );

  const toggle = (nodeId) => {
    const next = new Set(selected);
    if (next.has(nodeId)) next.delete(nodeId);
    else next.add(nodeId);
    onChange([...next].join(","));
  };

  if (error) {
    return html`<div class="form-field-error">Failed to load workers: ${error}</div>`;
  }
  if (workers === null) {
    return html`<div class="muted">Loading workers…</div>`;
  }
  if (workers.length === 0) {
    return html`<div class="muted">
      No workers registered. Start one with
      <code>python -m metainfer.worker --node-id NAME</code>.
    </div>`;
  }

  return html`
    <div class="checkbox-group">
      ${workers.map((w) => html`
        <label class=${`checkbox-pill ${w.alive ? "" : "disabled"}`}>
          <input type="checkbox"
            checked=${selected.has(w.node_id)}
            disabled=${!w.alive}
            onClick=${() => toggle(w.node_id)} />
          <span>${w.node_id}</span>
          <span class="muted small">
            (${w.alive ? "alive" : "DEAD"}, ${Object.keys(w.gpu_topology || {}).length} GPU)
          </span>
        </label>
      `)}
    </div>
  `;
}
