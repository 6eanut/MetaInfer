// Built-in form widget: multi-select checkbox pill group.
// Value is an array of selected option labels.
// Registered under name "multiselect".

import { html } from "htm/preact";

export function MultiSelectField({ field, value, onChange }) {
  const selected = new Set(Array.isArray(value) ? value : []);
  const toggle = (label) => {
    const next = new Set(selected);
    if (next.has(label)) next.delete(label);
    else next.add(label);
    onChange([...next]);
  };
  return html`
    <div class="checkbox-group">
      ${(field.options || []).map((o) => html`
        <label class="checkbox-pill">
          <input type="checkbox"
            checked=${selected.has(o.label)}
            onClick=${() => toggle(o.label)} />
          <span>${o.label}</span>
        </label>
      `)}
    </div>
  `;
}
