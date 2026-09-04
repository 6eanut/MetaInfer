// New Task overlay. Two-step flow:
//   1. Pick a task type (cards).
//   2. Fill the form (renderer + optional raw_request / extra_args).
//
// On submit, calls createTask() and bubbles the new id up so the parent
// can switch the active tab.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { listTaskTypes, loadFormSchema, createTask } from "app/api";
import { FormRenderer } from "app/form-renderer";

// A conversational task type (form.yaml `__meta__.conversational: true`)
// names its own wizard module via `conversation_module` (an importmap key,
// conventionally `app/<type>-conversation`). This host dynamically imports
// that module and renders it in place of the flat form. The wizard receives
// `{schema, onSubmit}` and calls `onSubmit({answers, raw_request, label})`
// when the user confirms; the shell stays type-agnostic.
function ConversationHost({ schema, onSubmit }) {
  const [Comp, setComp] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    let alive = true;
    setComp(null);
    setErr(null);
    const mod = schema && schema.conversation_module;
    if (!mod) { setErr("schema did not declare a conversation_module"); return; }
    import(mod).then((m) => {
      if (alive) setComp(m.default || m);
    }).catch((e) => { if (alive) setErr(String(e)); });
    return () => { alive = false; };
  }, [schema && schema.conversation_module]);
  if (err) return html`<p class="form-err">conversation module failed to load: ${err}</p>`;
  if (!Comp || !schema) return html`<p class="muted">loading conversation…</p>`;
  return html`<${Comp} schema=${schema} onSubmit=${onSubmit} />`;
}

export function NewTaskView({ onClose, onCreated }) {
  const [types, setTypes] = useState([]);
  const [selectedType, setSelectedType] = useState(null);
  const [schema, setSchema] = useState(null);
  const [values, setValues] = useState({});
  const [errors, setErrors] = useState({});
  const [submitErr, setSubmitErr] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [label, setLabel] = useState("");
  const [rawRequest, setRawRequest] = useState("");
  const [extraArgs, setExtraArgs] = useState("");

  useEffect(() => {
    listTaskTypes().then(setTypes).catch((e) => setSubmitErr(String(e)));
  }, []);

  useEffect(() => {
    if (!selectedType) return;
    setSchema(null);
    setValues({});
    setErrors({});
    loadFormSchema(selectedType).then(setSchema).catch((e) => setSubmitErr(String(e)));
  }, [selectedType]);

  const setField = (key, v) => {
    setValues((prev) => ({ ...prev, [key]: v }));
    setErrors((prev) => ({ ...prev, [key]: null }));
  };

  // Launch a task from either the flat form (payload empty → use the form's
  // own values/label/rawRequest) or a conversational wizard (payload carries
  // the settled flat answers + transcript + optional label).
  const doLaunch = async (payload) => {
    if (submitting) return;
    setSubmitting(true);
    setSubmitErr(null);
    const p = payload || {};
    // Build extra args (space-separated passthrough to orchestrator CLI).
    const extra = extraArgs.trim() ? extraArgs.trim().split(/\s+/) : [];
    try {
      const res = await createTask({
        type: selectedType,
        label: p.label || label || undefined,
        answers: p.answers || values,
        raw_request: p.raw_request || rawRequest || undefined,
        extra_args: extra,
      });
      onCreated && onCreated(res.task_id);
    } catch (e2) {
      const det = e2.detail;
      if (det && det.errors) {
        // Validation errors — show inline.
        if (typeof det.errors === "object") setErrors(det.errors);
        else setSubmitErr(String(det.errors));
      } else {
        setSubmitErr(String(e2.message || e2));
      }
    } finally {
      setSubmitting(false);
    }
  };

  const onSubmitFlat = async (e) => {
    e.preventDefault();
    await doLaunch({});
  };

  const onPickType = (t) => {
    setSelectedType(t);
    setLabel("");
    setRawRequest("");
    setExtraArgs("");
  };

  const onBack = () => {
    setSelectedType(null);
    setSchema(null);
    setErrors({});
    setSubmitErr(null);
  };

  return html`
    <div class="modal-overlay open">
      <div class="modal new-task-modal">
        <div class="modal-header">
          <h3>${selectedType ? `New task: ${selectedType}` : "New task"}</h3>
          <button class="close" onClick=${onClose}>×</button>
        </div>
        <div class="modal-body">
          ${!selectedType ? html`
            <div class="type-picker">
              ${(types || []).map((t) => html`
                <button class="type-card" key=${t.id}
                    onClick=${() => onPickType(t.id)}>
                  <div class="type-card-title">${t.label}</div>
                  <div class="type-card-id">${t.id}</div>
                  <div class="type-card-desc">${t.description}</div>
                </button>
              `)}
              ${types.length === 0
                ? html`<p class="muted">No task types available. Check that tasks/*.yaml exist.</p>`
                : null}
            </div>
          ` : schema && schema.conversational ? html`
            <${ConversationHost} schema=${schema}
              onSubmit=${(payload) => doLaunch(payload)} />
            ${submitErr ? html`<p class="form-err">${submitErr}</p>` : null}
          ` : html`
            <form onSubmit=${onSubmitFlat}>
              <${FormRenderer} schema=${schema} values=${values}
                errors=${errors} setField=${setField} />

              <div class="form-field">
                <label class="form-label">Label (optional)</label>
                <input type="text" class="input" value=${label}
                  placeholder="short name for the task tab"
                  onInput=${(e) => setLabel(e.target.value)} />
              </div>

              <div class="form-field">
                <label class="form-label">Raw request (optional)</label>
                <textarea class="input" rows="3" value=${rawRequest}
                  placeholder="free-form instruction passed to the orchestrator"
                  onInput=${(e) => setRawRequest(e.target.value)}></textarea>
              </div>

              <div class="form-field">
                <label class="form-label">Extra orchestrator args (optional)</label>
                <input type="text" class="input" value=${extraArgs}
                  placeholder="--max-iterations 5 --model sonnet"
                  onInput=${(e) => setExtraArgs(e.target.value)} />
              </div>

              ${submitErr
                ? html`<p class="form-err">${submitErr}</p>`
                : null}

              <div class="form-actions">
                <button type="button" class="btn ghost" onClick=${onBack}>← Back</button>
                <button type="submit" class="btn primary"
                  disabled=${submitting || !schema}>
                  ${submitting ? "Spawning…" : "Start task"}
                </button>
              </div>
            </form>
          `}
        </div>
      </div>
    </div>
  `;
}
