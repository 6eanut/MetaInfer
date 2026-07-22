// Built-in form widget: filesystem path input (free-text — the
// orchestrator resolves and validates the path server-side).
// Registered under name "file".

import { html } from "htm/preact";

export function FileField({ field, value, onInput }) {
  return html`
    <input type="text" class="input" placeholder="/path/to/file or dir"
      value=${value || ""}
      onInput=${(e) => onInput(e.target.value)} />
  `;
}
