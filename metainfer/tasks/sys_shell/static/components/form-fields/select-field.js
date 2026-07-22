// Built-in form widget: dropdown. Options come from ``field.options``.
// Registered under name "select".

import { html } from "htm/preact";

export function SelectField({ field, value, onChange }) {
  const opts = (field.options || []).map((o) => html`
    <option value=${o.label} selected=${o.label === value}>${o.label}</option>
  `);
  return html`
    <select class="input" value=${value || ""}
        onChange=${(e) => onChange(e.target.value)}>
      <option value="" disabled=${!field.required}>— pick —</option>
      ${opts}
    </select>
  `;
}
