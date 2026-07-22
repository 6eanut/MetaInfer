// Built-in form widget: multi-line textarea.
// Registered under name "textarea".

import { html } from "htm/preact";

export function TextAreaField({ field, value, onInput }) {
  return html`
    <textarea class="input" rows="3"
      placeholder=${field.help || ""}
      onInput=${(e) => onInput(e.target.value)}>${value || ""}</textarea>
  `;
}
