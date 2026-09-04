// opt-operator — create-time conversational requirement confirmation wizard.
//
// Rendered in place of the flat New-run form when opt-operator's form.yaml
// declares `__meta__.conversational: true` (see sys_shell new-task.js
// ConversationHost). This module is the opt-operator-owned side of that chat:
// the shell stays type-agnostic and only forwards {schema, onSubmit}.
//
// Flow (driven by the pure, deterministic converse engine — no LLM):
//   1. On mount we seed one turn (empty request) → the engine replies with an
//      opening guide and an editable interpretation card.
//   2. The user describes the operator in the chat box. Each turn POSTs to the
//      shell's /converse endpoint, which extracts as much structure as it can
//      and asks follow-up questions until every required field is pinned down.
//   3. When the card is complete (all required fields satisfied, no conflict),
//      the engine signals `complete` and we surface the interpretation card +
//      a Confirm button. The user may still edit card fields.
//   4. Confirm → /converse (settle) validates + returns { answers, raw_request }
//      where raw_request is the formatted dialogue transcript. onSubmit carries
//      that to the normal POST /tasks create path.

import { html } from "htm/preact";
import { useEffect, useRef, useState } from "preact/hooks";
import { converse, converseSettle } from "app/api";

// Inject a small, scoped stylesheet once per page load. Scoping under the
// `.optop-conv` prefix keeps opt-operator's look out of the shared shell CSS.
if (typeof document !== "undefined" && !document.getElementById("optop-conv-css")) {
  const st = document.createElement("style");
  st.id = "optop-conv-css";
  st.textContent = `
.optop-conv { display:flex; flex-direction:column; gap:14px; min-height: 320px; }
.optop-conv .oc-chat { display:flex; flex-direction:column; gap:8px;
  max-height: 260px; overflow:auto; padding:2px; }
.optop-conv .oc-row { display:flex; }
.optop-conv .oc-row.user { justify-content:flex-end; }
.optop-conv .oc-bubble { max-width:82%; padding:8px 11px; border-radius:10px;
  white-space:pre-wrap; word-break:break-word; font-size:13px; line-height:1.45; }
.optop-conv .oc-row.assistant .oc-bubble { background:#1d2333; color:#cfd6e6;
  border:1px solid #2a3247; border-bottom-left-radius:2px; }
.optop-conv .oc-row.user .oc-bubble { background:#22406b; color:#dfe8f7;
  border-bottom-right-radius:2px; }
.optop-conv .oc-inputrow { display:flex; gap:8px; align-items:center; }
.optop-conv .oc-inputrow .input { flex:1; }
.optop-conv .oc-card { border:1px solid #2a3247; border-radius:10px;
  background:#12161f; padding:12px; display:flex; flex-direction:column; gap:10px; }
.optop-conv .oc-card h4 { margin:0 0 4px; font-size:12px; letter-spacing:.08em;
  text-transform:uppercase; color:#8a94ad; }
.optop-conv .oc-field { display:grid; grid-template-columns:150px 1fr; gap:8px;
  align-items:start; font-size:13px; }
.optop-conv .oc-field .oc-lab { color:#b7c0d4; padding-top:6px; }
.optop-conv .oc-field .oc-req { color:#e07a7a; margin-left:3px; }
.optop-conv .oc-note { color:#6c7590; font-size:12px; }
.optop-conv .oc-field select, .optop-conv .oc-field input,
.optop-conv .oc-field textarea { width:100%; }
.optop-conv .oc-status { color:#8a94ad; font-size:12px; font-style:italic; }
.optop-conv .oc-err { color:#e07a7a; font-size:12px; }
`;
  document.head.appendChild(st);
}

const KIND_LABEL = { file: "file (attach in a later step)" };

export default function OptOperatorConversation({ schema, onSubmit }) {
  const type = schema && schema.type;
  const [answers, setAnswers] = useState({});
  const [card, setCard] = useState([]);
  const [complete, setComplete] = useState(false);
  const [transcript, setTranscript] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [runLabel, setRunLabel] = useState("");
  const chatRef = useRef(null);
  const started = useRef(false);

  // Seed the opening turn once.
  useEffect(() => {
    if (!type || started.current) return;
    started.current = true;
    setBusy(true);
    converse(type, "", {}, [])
      .then((i) => {
        setCard(i.card || []);
        setAnswers(i.answers || {});
        setComplete(!!i.complete);
        setTranscript([{ role: "assistant", text: i.assistant || "" }].filter((t) => t.text));
      })
      .catch((e) => setErr(String((e && e.detail) || (e && e.message) || e)))
      .finally(() => setBusy(false));
  }, [type]);

  // Keep the chat scrolled to the latest line.
  useEffect(() => {
    if (chatRef.current) chatRef.current.scrollTop = chatRef.current.scrollHeight;
  }, [transcript]);

  const byKey = {};
  for (const c of card) byKey[c.key] = c;

  const send = async () => {
    const text = (input || "").trim();
    if (!text || busy) return;
    setBusy(true);
    setErr(null);
    const withUser = [...transcript, { role: "user", text }];
    try {
      const i = await converse(type, text, answers, withUser);
      setCard(i.card || []);
      setAnswers(i.answers || {});
      setComplete(!!i.complete);
      const t2 = [...withUser];
      if (i.assistant) t2.push({ role: "assistant", text: i.assistant });
      setTranscript(t2);
      setInput("");
    } catch (e2) {
      setErr(String((e2 && e2.detail) || (e2 && e2.message) || e2));
    } finally {
      setBusy(false);
    }
  };

  const edit = (key, v) => setAnswers((a) => ({ ...a, [key]: v }));

  const confirm = async () => {
    if (busy) return;
    setBusy(true);
    setErr(null);
    try {
      const s = await converseSettle(type, answers, transcript);
      onSubmit({ answers: s.answers, raw_request: s.raw_request, label: runLabel || undefined });
    } catch (e3) {
      setErr(String((e3 && e3.detail) || (e3 && e3.message) || e3));
    } finally {
      setBusy(false);
    }
  };

  const enter = (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
  };

  return html`
    <div class="optop-conv">
      <div class="oc-chat" ref=${chatRef}>
        ${(transcript || []).map((t, i) => html`
          <div class="oc-row ${t.role === "user" ? "user" : "assistant"}" key=${i}>
            <div class="oc-bubble">${t.text}</div>
          </div>
        `)}
        ${busy ? html`<div class="oc-row assistant"><div class="oc-bubble oc-status">…</div></div>` : null}
      </div>

      <div class="oc-inputrow">
        <textarea class="input" rows="1" value=${input}
          placeholder=${complete ? "Anything to add, or review the card below and confirm." : "Describe the operator: paste a contract (name/inputs/outputs/shapes), or say the operator + shapes + stack."}
          onInput=${(e) => setInput(e.target.value)} onKeyDown=${enter} />
        <button class="btn primary" onClick=${send} disabled=${busy || !input.trim()}>
          Send
        </button>
      </div>

      ${complete ? html`
        <div class="oc-card">
          <h4>Interpretation — edit anything wrong, then confirm</h4>
          ${card.filter((c) => c.kind !== "file").map((c) => html`
            <div class="oc-field" key=${c.key}>
              <label class="oc-lab">${c.label}${c.required ? html`<span class="oc-req">*</span>` : null}</label>
              ${c.kind === "select"
                ? html`<select value=${answers[c.key] || ""}
                    onChange=${(e) => edit(c.key, e.target.value)}>
                    <option value="" disabled>choose…</option>
                    ${(c.options || []).map((o) => html`<option value=${o} key=${o}>${o}</option>`)}
                  </select>`
                : c.kind === "textarea"
                ? html`<textarea rows=${(answers[c.key] || "").split("\n").length > 2 ? 4 : 2}
                    value=${answers[c.key] || ""}
                    onInput=${(e) => edit(c.key, e.target.value)} />`
                : html`<input class="input" type="text"
                    value=${answers[c.key] == null ? "" : answers[c.key]}
                    onInput=${(e) => edit(c.key, e.target.value)} />`}
            </div>
          `)}
          ${card.some((c) => c.kind === "file")
            ? html`<div class="oc-note">
                File inputs (${card.filter((c) => c.kind === "file").map((c) => c.label).join(", ")})
                aren't uploadable from the chat; they stay optional here.
              </div>` : null}
          <div class="oc-field">
            <label class="oc-lab">Run label (optional)</label>
            <input class="input" type="text" value=${runLabel}
              placeholder="short name for the task tab"
              onInput=${(e) => setRunLabel(e.target.value)} />
          </div>
          ${err ? html`<div class="oc-err">${err}</div>` : null}
          <div class="form-actions" style=${{ marginTop: "2px" }}>
            <button class="btn primary" onClick=${confirm} disabled=${busy}>
              Confirm & start optimization
            </button>
          </div>
        </div>
      ` : html`
        ${err ? html`<div class="oc-err">${err}</div>` : null}
      `}
    </div>
  `;
}
