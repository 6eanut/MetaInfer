// Generic destructive-action confirmation modal.
//
// Requires the user to type a specific confirmation string (the task
// name) before the Confirm button enables. Used by:
//   - Reset button in task-detail (resets task to initial state)
//   - Tab close × in main.js (deletes task entirely)
//
// The confirmation string is shown prominently with a hint that the
// user can copy it. The Confirm button is disabled until the typed
// value matches exactly (case-sensitive).

import { html } from "htm/preact";
import { useState } from "preact/hooks";

export function ConfirmActionModal({
  title,
  promptText,
  confirmText,         // exact string user must type
  confirmLabel = "确认",
  cancelLabel = "取消",
  danger = true,
  onConfirm,
  onClose,
}) {
  const [draft, setDraft] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [err, setErr] = useState(null);

  const trimmed = draft.trim();
  const matches = trimmed !== "" && trimmed === confirmText;

  const onSubmit = async (ev) => {
    ev.preventDefault();
    if (!matches || submitting) return;
    setSubmitting(true);
    setErr(null);
    try {
      await onConfirm();
      onClose();
    } catch (e) {
      setSubmitting(false);
      setErr(String(e?.message || e));
    }
  };

  return html`
    <div class="modal-overlay open" onClick=${(e) => {
      if (e.target === e.currentTarget && !submitting) onClose();
    }}>
      <div class="modal confirm-action-modal">
        <div class="modal-header">
          <h3>${title}</h3>
          <button class="close"
            onClick=${() => { if (!submitting) onClose(); }}
            disabled=${submitting}>×</button>
        </div>
        <div class="modal-body">
          <p class="confirm-prompt">${promptText}</p>
          <div class="confirm-hint">
            请输入任务名称以确认：
            <code class="confirm-target" title="点击选中后复制">${confirmText}</code>
            <button class="btn ghost small"
              type="button"
              onClick=${() => {
                navigator.clipboard?.writeText(confirmText).catch(() => {});
              }}>复制</button>
          </div>
          <form onSubmit=${onSubmit} class="confirm-form">
            <input
              type="text"
              class="confirm-input"
              autoFocus
              autoComplete="off"
              spellcheck=${false}
              value=${draft}
              onInput=${(e) => setDraft(e.target.value)}
              placeholder=${confirmText} />
            <div class="confirm-actions">
              <button type="button" class="btn ghost"
                onClick=${() => { if (!submitting) onClose(); }}
                disabled=${submitting}>${cancelLabel}</button>
              <button type="submit"
                class=${`btn ${danger ? "danger" : "primary"}`}
                disabled=${!matches || submitting}>
                ${submitting ? "处理中…" : confirmLabel}
              </button>
            </div>
          </form>
          ${err ? html`<div class="task-banner task-banner-err">${err}</div>` : null}
        </div>
      </div>
    </div>
  `;
}
