// Built-in form widget: single-line text input.
// Registered under name "text".

import { html } from "htm/preact";

export function TextField({ field, value, onInput }) {
  return html`
    <input type="text" class="input"
      value=${value || ""}
      placeholder=${field.help || ""}
      onInput=${(e) => onInput(e.target.value)} />
  `;
}
