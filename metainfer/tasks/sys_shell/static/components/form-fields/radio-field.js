// Built-in form widget: radio button pill group.
// Registered under name "radio".

import { html } from "htm/preact";

export function RadioField({ field, value, onChange }) {
  return html`
    <div class="radio-group">
      ${(field.options || []).map((o) => html`
        <label class="radio-pill">
          <input type="radio" name=${field.key}
            checked=${o.label === value}
            onClick=${() => onChange(o.label)} />
          <span>${o.label}</span>
        </label>
      `)}
    </div>
  `;
}
