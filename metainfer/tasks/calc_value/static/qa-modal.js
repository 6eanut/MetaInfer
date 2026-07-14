// QA modal — offline chat with an agent whose events.jsonl transcript
// lives on disk. The user picks an agent in the iterations panel (e.g.
// step1/round_0/agent_a), types a question, and a NEW analyst ccb
// subprocess is spawned that reads the target agent's transcript and
// answers.
//
// The analyst runs async on the server. We POST to /qa/start to get a
// session_id, then poll /qa/<sid> every 2s until status=done|failed.

import { html } from "htm/preact";
import { useEffect, useState, useCallback, useRef } from "preact/hooks";
import { escapeHtml } from "app/utils";

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_MS = 11 * 60 * 1000; // 11 min (server times out at 10)

export function QAModal({ taskId, target, onClose }) {
  // target = { step, round, round_label, agent, events_file, target_workdir, label }
  const [history, setHistory] = useState([]);     // prior sessions for this target
  const [histErr, setHistErr] = useState(null);
  const [draft, setDraft] = useState("");
  const [active, setActive] = useState(null);     // {sid, question, status, answer?, error?}
  const [submitErr, setSubmitErr] = useState(null);

  const targetKey = `${target?.step}|${target?.round}|${target?.agent}`;

  // Load prior sessions for this target.
  const refreshHistory = useCallback(async () => {
    if (!taskId || !target) return;
    try {
      const qs = new URLSearchParams({
        step: String(target.step ?? ""),
        round: String(target.round ?? ""),
        agent: target.agent ?? "",
      });
      const r = await fetch(
        `/api/tasks/${taskId}/calc/qa?${qs.toString()}`,
        { cache: "no-store" },
      );
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const data = await r.json();
      setHistory(data.sessions || []);
      setHistErr(null);
    } catch (e) {
      setHistErr(String(e));
    }
  }, [taskId, target]);

  useEffect(() => {
    setHistory([]); setHistErr(null);
    setActive(null); setSubmitErr(null); setDraft("");
    refreshHistory();
  }, [taskId, targetKey]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Poll the active session until it finishes.
  useEffect(() => {
    if (!active || active.status === "done" || active.status === "failed") return;
    const start = Date.now();
    const id = setInterval(async () => {
      if (Date.now() - start > MAX_POLL_MS) {
        setActive((a) => ({ ...a, status: "failed", error: "client poll timeout" }));
        clearInterval(id);
        return;
      }
      try {
        const r = await fetch(
          `/api/tasks/${taskId}/calc/qa/${active.sid}`,
          { cache: "no-store" },
        );
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const s = await r.json();
        const status = s.status?.status || "unknown";
        setActive({
          sid: active.sid,
          question: s.request?.question || active.question,
          status,
          answer: s.answer,
          error: s.status?.error,
        });
        if (status === "done" || status === "failed") {
          clearInterval(id);
          refreshHistory();
        }
      } catch (e) {
        // Soft-fail: keep polling a bit longer, server may be transient.
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [active?.sid, active?.status, taskId]);  // eslint-disable-line react-hooks/exhaustive-deps

  const onSubmit = useCallback(async (ev) => {
    ev.preventDefault();
    const q = draft.trim();
    if (!q || !target?.events_file) return;
    setSubmitErr(null);
    try {
      const r = await fetch(`/api/tasks/${taskId}/calc/qa/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          events_file: target.events_file,
          target_workdir: target.target_workdir || null,
          target_label: target.label || `step=${target.step} round=${target.round} agent=${target.agent}`,
          question: q,
          step: target.step,
          round: target.round,
          round_label: target.round_label || null,
          agent: target.agent,
        }),
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(d.detail || `HTTP ${r.status}`);
      }
      const { session_id } = await r.json();
      setActive({ sid: session_id, question: q, status: "running", answer: null, error: null });
      setDraft("");
    } catch (e) {
      setSubmitErr(String(e));
    }
  }, [draft, taskId, target]);

  if (!target) return null;

  return html`
    <div class="modal-overlay open" onClick=${(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <div class="modal qa-modal">
        <div class="modal-header">
          <h3>
            <span class="qa-target-icon">💬</span>
            Ask an analyst
          </h3>
          <button class="close" onClick=${onClose}>×</button>
        </div>
        <div class="modal-body">
          <div class="qa-target-info">
            <div><strong>Target agent:</strong> ${target.label || target.agent}</div>
            ${target.events_file
              ? html`<div class="muted small">
                  transcript: <code>${target.events_file}</code>
                </div>`
              : html`<div class="warn">no transcript available for this agent</div>`}
          </div>

          ${histErr ? html`<div class="task-banner task-banner-err">
            history load failed: ${histErr}</div>` : null}

          ${history.length > 0 ? html`
            <details class="qa-history" ${active ? null : ""}>
              <summary>Prior Q&A for this agent (${history.length})</summary>
              <div class="qa-history-list">
                ${history.map((s) => html`
                  <div key=${s.id} class="qa-turn qa-turn-past">
                    <div class="qa-q">
                      <span class="qa-role">you</span>
                      <span>${s.question}</span>
                    </div>
                    <div class="qa-meta">
                      <span class=${`pill ${s.status === "done" ? "done" : "failed"}`}>${s.status}</span>
                      <span class="muted">${fmtTime(s.finished_at || s.created_at)}</span>
                    </div>
                  </div>
                `)}
              </div>
            </details>
          ` : null}

          ${active ? html`
            <div class="qa-active">
              <div class="qa-turn">
                <div class="qa-q">
                  <span class="qa-role">you</span>
                  <span>${active.question}</span>
                </div>
              </div>
              ${active.status === "running"
                ? html`<div class="qa-thinking">
                    <span class="spinner"></span>
                    analyst is reading the transcript…
                  </div>`
                : null}
              ${active.status === "done" && active.answer
                ? html`<div class="qa-a">
                    <span class="qa-role">analyst</span>
                    <div class="qa-answer">${html`${formatAnswer(active.answer)}`}</div>
                  </div>`
                : null}
              ${active.status === "failed"
                ? html`<div class="task-banner task-banner-err">
                    analyst failed: ${active.error || "(unknown error)"}
                  </div>`
                : null}
            </div>
          ` : null}

          ${submitErr ? html`<div class="task-banner task-banner-err">
            submit failed: ${submitErr}</div>` : null}

          <form class="qa-input-row" onSubmit=${onSubmit}>
            <input
              type="text"
              placeholder=${target.events_file
                ? "ask anything about what this agent did…"
                : "(no transcript for this agent)"}
              value=${draft}
              disabled=${!target.events_file || (active && active.status === "running")}
              onInput=${(e) => setDraft(e.target.value)} />
            <button class="btn primary"
              disabled=${!draft.trim() || !target.events_file
                || (active && active.status === "running")}>
              ${active && active.status === "running" ? "…" : "Ask"}
            </button>
          </form>
          <p class="muted small qa-foot">
            Starts a new read-only analyst agent that reads the transcript
            and answers. Each question is independent.
          </p>
        </div>
      </div>
    </div>
  `;
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  return d.toLocaleString();
}

// Render markdown-ish answer text into safe HTML. The analyst's answer is
// LLM-generated, so escape everything, then restore a tiny whitelist
// (code spans, line breaks).
function formatAnswer(text) {
  if (!text) return html`<span class="muted">(empty)</span>`;
  // htm/preact will HTML-escape when interpolating a string into ${}.
  // Just return the raw string; Preact handles the escaping.
  return html`<pre class="qa-answer-pre">${text}</pre>`;
}
