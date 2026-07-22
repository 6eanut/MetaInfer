// port-model form widget: a dynamic list of {path, notes} entries.
//
// Used for the ``reference_sources`` field — the user can paste any
// number of reference implementation directories, each with an optional
// free-form note. Value is a JSON-serialisable array of
// ``{path: string, notes: string}`` objects (or null/undefined for
// "no entries").
//
// Registered with the form-registry under the name
// ``kv-list-path-notes`` from ``form-overrides.js``.

import { html } from "htm/preact";

export function KvListPathNotes({ field, value, onChange, error }) {
  const items = Array.isArray(value) ? value : [];
  const update = (next) => onChange(next.length ? next : null);

  const setItem = (i, patch) => {
    const next = items.map((it, idx) => (idx === i ? { ...it, ...patch } : it));
    update(next);
  };
  const removeItem = (i) => update(items.filter((_, idx) => idx !== i));
  const addItem = () => update([...items, { path: "", notes: "" }]);

  return html`
    <div class="kv-list">
      ${items.length === 0
        ? html`<p class="form-help" style=${{ marginBottom: "0.5rem" }}>
            点击「新增一条」添加参考实现源码目录（可选）。
          </p>`
        : null}
      ${items.map((it, i) => html`
        <div class="kv-list-row" key=${i}>
          <input type="text" class="input kv-list-path"
            placeholder="/path/to/reference/source"
            value=${it.path || ""}
            onInput=${(e) => setItem(i, { path: e.target.value })} />
          <input type="text" class="input kv-list-notes"
            placeholder="备注（可选）"
            value=${it.notes || ""}
            onInput=${(e) => setItem(i, { notes: e.target.value })} />
          <button type="button" class="btn kv-list-remove"
            onClick=${() => removeItem(i)}>✕</button>
        </div>
      `)}
      <button type="button" class="btn kv-list-add" onClick=${addItem}>
        + 新增一条
      </button>
      ${error ? html`<p class="form-err">${error}</p>` : null}
    </div>
  `;
}
