// Generic form renderer. Walks the schema's ``fields`` array and emits
// one control per field.
//
// The widget for each field is resolved via the form-registry:
//
//   - If ``field.override_component`` is set and a widget with that
//     name is registered, use it. This lets a plugin swap the widget
//     for one specific field without changing its ``type``.
//   - Otherwise, look up a widget whose name matches ``field.type``
//     (the built-ins: text / textarea / number / select / multiselect
//     / radio / file).
//   - If neither matches, render a friendly "unknown widget" label
//     instead of crashing — the form still works for the other fields.
//
// All built-in widgets and any plugin-supplied widgets live in the
// registry. The renderer itself owns no widget code.

import { html } from "htm/preact";
import { getWidgetForField } from "app/form-registry";

function UnknownWidget({ field }) {
  return html`
    <div class="form-field">
      <label class="form-label">
        ⚠ unknown widget: ${field.override_component || field.type}
      </label>
    </div>
  `;
}

export function FormRenderer({ schema, values, errors, setField }) {
  if (!schema) return null;
  const fields = schema.fields || [];
  return html`
    <div class="form-grid">
      ${fields.map((field) => {
        const val = values[field.key];
        const err = errors && errors[field.key];
        const setValue = (v) => setField(field.key, v);
        const Widget = getWidgetForField(field) || UnknownWidget;
        const control = html`<${Widget} field=${field} value=${val}
          onChange=${setValue} onInput=${setValue} error=${err} />`;
        return html`
          <div class="form-field" key=${field.key}>
            <label class="form-label">
              ${field.label}${field.required ? html`<span class="req">*</span>` : null}
            </label>
            ${control}
            ${field.help && field.type !== "select"
              ? html`<p class="form-help">${field.help}</p>` : null}
            ${err ? html`<p class="form-err">${err}</p>` : null}
          </div>
        `;
      })}
    </div>
  `;
}
