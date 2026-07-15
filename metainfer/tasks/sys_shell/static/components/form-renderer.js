// Generic form renderer. Walks the schema's `fields` array and emits
// one control per field. Fields with `override_component` are delegated
// to a task-specific widget — handled here so the new-task view stays
// agnostic.

import { html } from "htm/preact";

function TextField({ field, value, onInput }) {
  return html`
    <input type="text" class="input"
      value=${value || ""}
      placeholder=${field.help || ""}
      onInput=${(e) => onInput(e.target.value)} />
  `;
}

function TextAreaField({ field, value, onInput }) {
  return html`
    <textarea class="input" rows="3"
      placeholder=${field.help || ""}
      onInput=${(e) => onInput(e.target.value)}>${value || ""}</textarea>
  `;
}

function NumberField({ field, value, onInput }) {
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

function SelectField({ field, value, onChange }) {
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

function MultiSelectField({ field, value, onChange }) {
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

function RadioField({ field, value, onChange }) {
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

function FileField({ field, value, onInput }) {
  return html`
    <input type="text" class="input" placeholder="/path/to/file or dir"
      value=${value || ""}
      onInput=${(e) => onInput(e.target.value)} />
  `;
}

// Override dispatch. Each named override is a hand-written control; the
// form renderer just looks it up here.
const OVERRIDES = {
  // Future: "shape-input", "kv-list", etc.
};

function OverrideField({ field, value, onChange, error }) {
  const Cmp = OVERRIDES[field.override_component];
  if (!Cmp) {
    return html`
      <div class="form-field">
        <label class="form-label">⚠ unknown override: ${field.override_component}</label>
      </div>
    `;
  }
  return html`<${Cmp} field=${field} value=${value} onChange=${onChange} error=${error} />`;
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
        let control;
        if (field.override_component) {
          control = html`<${OverrideField} field=${field} value=${val}
            onChange=${setValue} error=${err} />`;
        } else {
          switch (field.type) {
            case "textarea":
              control = html`<${TextAreaField} field=${field} value=${val} onInput=${setValue} />`;
              break;
            case "number":
              control = html`<${NumberField} field=${field} value=${val} onInput=${setValue} />`;
              break;
            case "select":
              control = html`<${SelectField} field=${field} value=${val} onChange=${setValue} />`;
              break;
            case "multiselect":
              control = html`<${MultiSelectField} field=${field} value=${val} onChange=${setValue} />`;
              break;
            case "file":
              control = html`<${FileField} field=${field} value=${val} onInput=${setValue} />`;
              break;
            default:
              control = html`<${TextField} field=${field} value=${val} onInput=${setValue} />`;
          }
        }
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
