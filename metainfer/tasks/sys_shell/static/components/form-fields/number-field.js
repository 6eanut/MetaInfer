// Built-in form widget: numeric input. Stored as JS number (null when empty).
// Registered under name "number".

import { html } from "htm/preact";

export function NumberField({ field, value, onInput }) {
  const num = value == null || value === "" ? "" : Number(value);
  return html`
    <input type="number" class="input"
      value=${num}
      placeholder=${field.help || ""}
      onInput=${(e) => {
        const v = e.target.value;
        onInput(v === "" ? null : Number(v));
      }} />
  `;
}
