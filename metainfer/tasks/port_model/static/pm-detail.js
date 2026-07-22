/**
 * port-model detail view — Preact component mounted inside the sys-shell
 * task-detail chrome. Renders:
 *   - Control bar: rerun-step picker / kill / restart + budget gauge
 *   - 6-agent state-graph (simple chain with bounce/repair edges)
 *   - Per-phase summary cards
 *   - Iterations table
 *   - P6 commit log + verdicts
 *   - Memory-doc viewer
 *
 * Importmap key: app/pm-detail
 * Shell passes: { run, timeline, agents, loadState, lastErr, refreshShell }
 */

import { html } from "htm/preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { marked } from "marked";
import { AgentsPanel } from "app/agents-panel";
import { Timeline } from "app/timeline";

const API = (taskId) => `/api/port-model/${taskId}`;

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function fetchText(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  return res.text();
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!res.ok) {
    let detail = "";
    try { detail = JSON.stringify(await res.json()); } catch (_) { /* ignore */ }
    throw new Error(`${res.status}: ${detail || res.statusText}`);
  }
  return res.json();
}

const withTimeout = (p, ms = 8000) =>
  Promise.race([
    p,
    new Promise((_, rej) => setTimeout(() => rej(new Error("timeout")), ms)),
  ]);

const PHASE_ORDER = [
  "P1_weight_analysis",
  "P2_framework_analysis",
  "P3_architect_review",
  "P4_minimal_framework",
  "P5_verify_minimal",
  "P6_port_engine",
];

const PHASE_LABELS = {
  P1_weight_analysis:   "1 · 权重参数分析",
  P2_framework_analysis: "2 · 推理框架分析师 (fan-out)",
  P3_architect_review:  "3 · 资深架构师",
  P4_minimal_framework: "4 · 精简推理框架",
  P5_verify_minimal:    "5 · 精简框架验证",
  P6_port_engine:       "6 · 推理引擎移植",
  finished:             "✓ 完成",
  idle:                 "idle",
};

function phaseStatus(currentPhase, phase) {
  if (currentPhase === "finished") return "ok";
  const idx = PHASE_ORDER.indexOf(phase);
  const curIdx = PHASE_ORDER.indexOf(currentPhase);
  if (curIdx === -1) return "idle";
  if (idx < curIdx) return "ok";
  if (idx === curIdx) return "running";
  return "idle";
}

function phaseOutcomeLabel(o) {
  if (!o) return "";
  const m = {
    ok: "ok", logic_fail: "logic ✗", infra_fail: "infra ↻",
    test_fail: "test ✗", bounce_back: "↩ redo",
    needs_repair: "↻ repair", aborted: "aborted",
  };
  return m[o] || o;
}

function excerptToParagraphs(text, n = 3) {
  if (!text) return [];
  // Strip markdown headers for the excerpt preview.
  const lines = text.split("\n")
    .filter((l) => !l.startsWith("#") || l.startsWith("##"))
    .filter((l) => l.trim().length > 0);
  return lines.slice(0, n);
}

/* ---------------- Runtime data hook ---------------- */

function usePortRuntime(taskId) {
  const [data, setData] = useState({
    iterations: [], graph: null, p6: [], dumps: null,
  });
  const refresh = useCallback(async () => {
    if (!taskId) return;
    const [it, g, p6, dumps] = await Promise.all([
      withTimeout(fetchJSON(`${API(taskId)}/iterations`)).catch(() => []),
      withTimeout(fetchJSON(`${API(taskId)}/state-graph`)).catch(() => null),
      withTimeout(fetchJSON(`${API(taskId)}/p6-iterations`)).catch(() => []),
      withTimeout(fetchJSON(`${API(taskId)}/dumps`)).catch(() => null),
    ]);
    setData({ iterations: it || [], graph: g, p6: p6 || [], dumps });
  }, [taskId]);
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => {
    if (!taskId) return;
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, [taskId, refresh]);
  return { ...data, refresh };
}

/* ---------------- Control bar ---------------- */

function ControlBar({ taskId, onStarted }) {
  const [step, setStep] = useState(PHASE_ORDER[0]);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const call = async (payload) => {
    setBusy(true); setErr(null);
    try {
      await postJSON(`${API(taskId)}/control`, payload);
      if (onStarted) onStarted();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };

  return html`
    <div class="pm-controls">
      <select value=${step} onChange=${(e) => setStep(e.target.value)}>
        ${PHASE_ORDER.map((p) => html`<option value=${p}>${PHASE_LABELS[p]}</option>`)}
      </select>
      <button class="pm-btn primary" disabled=${busy}
        onClick=${() => call({ action: "rerun_step", step })}>
        ${busy ? "…" : "↻ Rerun from step"}
      </button>
      <button class="pm-btn" disabled=${busy}
        onClick=${() => call({ action: "restart" })}>
        ${busy ? "…" : "▶ Restart"}
      </button>
      <button class="pm-btn danger" disabled=${busy}
        onClick=${() => call({ action: "kill", force: true })}>
        ${busy ? "…" : "■ Kill"}
      </button>
    </div>
    ${err ? html`<div class="form-err" style=${{ marginBottom: "0.5rem" }}>${err}</div>` : null}
  `;
}

/* ---------------- State-graph (simple SVG-less chain) ---------------- */

function StateGraph({ graph }) {
  if (!graph) return html`<p class="form-help">loading state graph…</p>`;
  const nodes = graph.nodes || [];
  const edges = graph.edges || [];
  const cur = graph.current;
  return html`
    <div class="pm-phase-grid">
      ${nodes.map((n) => {
        const status = phaseStatus(cur, n.id);
        const incoming = edges.filter((e) => e.to === n.id);
        return html`
          <div class=${`pm-phase-card ${status}`}>
            <div class="pm-phase-head">
              <span>${n.label}</span>
              <span class="pm-phase-status">${status}</span>
            </div>
            <div class="pm-phase-meta">
              <span>${incoming.length ? `← ${incoming.map((e) => e.label).join(", ")}` : "entry"}</span>
            </div>
            ${n.description
              ? html`<div class="pm-phase-excerpt"><p>${n.description}</p></div>`
              : null}
          </div>
        `;
      })}
    </div>
  `;
}

/* ---------------- Per-phase summary cards ---------------- */

function PhaseSummaryModal({ phase, label, summary, rec, onClose }) {
  // Render full summary.md as HTML via marked. Body scroll locked
  // while open. Click on backdrop closes; Esc closes.
  useEffect(() => {
    const onKey = (e) => { if (e.key === "Escape") onClose(); };
    document.addEventListener("keydown", onKey);
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.removeEventListener("keydown", onKey);
      document.body.style.overflow = prev;
    };
  }, [onClose]);

  const htmlBody = summary
    ? marked.parse(summary, { breaks: true, gfm: true })
    : "<p class='muted'>no summary.md for this phase yet</p>";

  return html`
    <div class="pm-modal-backdrop" onClick=${(e) => {
      if (e.target.classList.contains("pm-modal-backdrop")) onClose();
    }}>
      <div class="pm-modal-card">
        <header class="pm-modal-head">
          <div>
            <strong>${label}</strong>
            ${rec && rec.outcome
              ? html`<span class=${`pm-phase-status ${rec.outcome}`}
                  style=${{ marginLeft: "0.5rem" }}>
                  ${phaseOutcomeLabel(rec.outcome)}
                </span>`
              : null}
          </div>
          <div class="pm-modal-meta">
            ${rec && rec.duration_s ? html`<span>⏱ ${rec.duration_s.toFixed(1)}s</span>` : null}
            ${rec && rec.agent_name ? html`<span>👤 ${rec.agent_name}</span>` : null}
          </div>
          <button class="btn ghost" onClick=${onClose} title="关闭 (Esc)">×</button>
        </header>
        ${rec && rec.error
          ? html`<div class="form-err" style=${{ marginBottom: "0.5rem" }}>${rec.error}</div>`
          : null}
        <div class="pm-modal-body markdown-body" dangerouslySetInnerHTML=${{ __html: htmlBody }} />
      </div>
    </div>
  `;
}

function PhaseCards({ taskId, currentPhase, iterations }) {
  const [summaries, setSummaries] = useState({});
  const [openPhase, setOpenPhase] = useState(null);

  // Pull per-phase summaries from the LATEST iteration record, if any.
  const latest = iterations.length ? iterations[iterations.length - 1] : null;
  const phaseRecs = (latest && latest.phases) || {};

  useEffect(() => {
    let cancelled = false;
    PHASE_ORDER.forEach(async (ph) => {
      const txt = await fetchText(`${API(taskId)}/phase-summary/${ph}`);
      if (!cancelled) setSummaries((s) => ({ ...s, [ph]: txt }));
    });
    return () => { cancelled = true; };
  }, [taskId, iterations.length]);

  return html`
    <div class="pm-phase-grid">
      ${PHASE_ORDER.map((ph) => {
        const status = phaseStatus(currentPhase, ph);
        const rec = phaseRecs[ph] || {};
        const summary = summaries[ph];
        const preview = excerptToParagraphs(summary, 1).join(" · ");
        return html`
          <div class=${`pm-phase-card ${status}`}>
            <div class="pm-phase-head">
              <span>${PHASE_LABELS[ph]}</span>
              <span class="pm-phase-status">
                ${rec.outcome ? phaseOutcomeLabel(rec.outcome) : status}
              </span>
            </div>
            <div class="pm-phase-meta">
              ${rec.duration_s ? html`<span>⏱ ${rec.duration_s.toFixed(1)}s</span>` : null}
              ${rec.agent_name ? html`<span>👤 ${rec.agent_name}</span>` : null}
              ${rec.error ? html`<span style=${{ color: "var(--err)" }}>${rec.error}</span>` : null}
            </div>
            ${preview
              ? html`<p class="pm-phase-preview">${preview}</p>`
              : html`<p class="form-help" style=${{ fontSize: "0.75rem" }}>
                  no summary.md yet
                </p>`}
            ${summary
              ? html`<button class="pm-btn" onClick=${() => setOpenPhase(ph)}
                  style=${{ alignSelf: "flex-start", marginTop: "0.25rem" }}>
                  📖 查看完整 summary
                </button>`
              : null}
          </div>
        `;
      })}
    </div>
    ${openPhase ? html`
      <${PhaseSummaryModal}
        phase=${openPhase}
        label=${PHASE_LABELS[openPhase]}
        summary=${summaries[openPhase]}
        rec=${phaseRecs[openPhase] || null}
        onClose=${() => setOpenPhase(null)} />
    ` : null}
  `;
}

/* ---------------- Iterations table ---------------- */

function IterationsTable({ iterations }) {
  if (!iterations.length) {
    return html`<p class="form-help">no iterations yet</p>`;
  }
  return html`
    <div class="pm-iterations">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>status</th>
            <th>duration</th>
            <th>P3 bounce</th>
            <th>P5 repair</th>
            <th>P6 iters</th>
            <th>final</th>
            <th>started</th>
          </tr>
        </thead>
        <tbody>
          ${iterations.map((it) => html`
            <tr key=${it.iteration}>
              <td>${it.iteration}</td>
              <td class=${it.status}>${it.status}</td>
              <td>${it.duration_s ? `${it.duration_s.toFixed(1)}s` : "—"}</td>
              <td>${(it.phases || {}).P3_architect_review ? "✓" : "—"}</td>
              <td>${(it.phases || {}).P5_verify_minimal ? "✓" : "—"}</td>
              <td>${(it.p6_iterations || []).length || 0}</td>
              <td>${it.final_status || "—"}</td>
              <td>${it.started_at ? new Date(it.started_at * 1000).toLocaleString() : "—"}</td>
            </tr>
          `)}
        </tbody>
      </table>
    </div>
  `;
}

/* ---------------- P6 commits / verdicts ---------------- */

function P6List({ p6 }) {
  if (!p6 || !p6.length) {
    return html`<p class="form-help">no P6 iterations yet</p>`;
  }
  return html`
    <div class="pm-p6-list">
      ${p6.map((it) => html`
        <div class="pm-p6-iter" key=${it.name}>
          <div class="pm-p6-head">
            <strong>${it.name}</strong>
            <span>
              ${it.verdict && it.verdict.outcome
                ? html`<span class=${it.verdict.outcome === "ok" ? "ok" : "fail"}>
                    ${it.verdict.outcome}
                  </span>`
                : "(no verdict)"}
              ${it.commit_sha
                ? html` · <code>${it.commit_sha.slice(0, 8)}</code>`
                : " · (no commit)"}
            </span>
          </div>
          ${it.verdict && Array.isArray(it.verdict.batch)
            ? html`<div class="pm-p6-batch">
                ${it.verdict.batch.map((row, i) => html`
                  <div class="pm-p6-row" key=${i}>
                    <code class="pm-p6-prompt">${row.prompt}</code>
                    <span class=${`pm-p6-judgment ${row.verifier_judgment || ""}`}>
                      ${row.verifier_judgment || "?"}
                    </span>
                    ${row.topk_text && row.topk_text.length
                      ? html`<span class="muted">
                          top: ${row.topk_text.slice(0, 5).join(" / ")}
                        </span>`
                      : null}
                    ${row.verifier_reason
                      ? html`<span class="muted pm-p6-reason">${row.verifier_reason}</span>`
                      : null}
                  </div>
                `)}
              </div>`
            : (it.verdict && it.verdict.output_text
              ? html`<div>output: <em>${it.verdict.output_text.slice(0, 200)}</em></div>`
              : null)}
          ${it.verdict && typeof it.verdict.similarity_min === "number"
            ? html`<div>min cosine: ${it.verdict.similarity_min.toFixed(4)}${
                it.verdict.similarity_first_bad_layer !== null && it.verdict.similarity_first_bad_layer !== undefined
                  ? ` (first bad layer: ${it.verdict.similarity_first_bad_layer})`
                  : ""
              }</div>`
            : null}
          ${it.summary
            ? html`<pre>${it.summary.slice(0, 800)}</pre>`
            : null}
        </div>
      `)}
    </div>
  `;
}

/* ---------------- Memory doc viewer ---------------- */

function MemoryViewer({ taskId }) {
  const MEMORY_DOCS = [
    { name: "p1_weight_analysis", label: "P1: weight analysis" },
    { name: "p3_consolidated_spec", label: "P3: consolidated spec" },
  ];
  const P2_DOCS = [];
  // P2 docs are dynamic — we don't know the count up front; pull 1..4 best-effort.
  for (let i = 1; i <= 4; i++) {
    P2_DOCS.push({ name: `p2_ref${i}_analysis`, label: `P2: ref ${i} analysis` });
  }
  const [selected, setSelected] = useState("p1_weight_analysis");
  const [content, setContent] = useState(null);
  const [err, setErr] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setErr(null);
    fetchText(`${API(taskId)}/memory/${selected}`).then((txt) => {
      if (!cancelled) {
        if (txt == null) setErr(`memory/${selected}.md not yet written`);
        setContent(txt);
      }
    });
    return () => { cancelled = true; };
  }, [taskId, selected]);

  return html`
    <div>
      <select value=${selected}
        onChange=${(e) => setSelected(e.target.value)}>
        ${[...MEMORY_DOCS, ...P2_DOCS].map((d) => html`
          <option key=${d.name} value=${d.name}>${d.label}</option>
        `)}
      </select>
      ${err ? html`<p class="form-err">${err}</p>` : null}
      ${content
        ? html`<div class="pm-memory">
            <pre>${content}</pre>
          </div>`
        : null}
    </div>
  `;
}

/* ---------------- Tabs ---------------- */

const TABS = [
  { id: "overview",   label: "概览" },
  { id: "phases",     label: "Agent 进度" },
  { id: "iterations", label: "迭代" },
  { id: "p6",         label: "P6 移植" },
  { id: "memory",     label: "Memory" },
  { id: "live",       label: "实时" },
];

export default function PortModelDetailView({
  taskId,
  run,
  status,
  data,
}) {
  const [activeTab, setActiveTab] = useState("overview");
  const phase = run?.current_phase || "idle";
  const { timeline, agents } = data || {};
  const rt = usePortRuntime(taskId);

  return html`
    <div class="pm-root">
      <${ControlBar} taskId=${taskId} onStarted=${rt.refresh} />

      <nav class="pm-tabs">
        ${TABS.map((t) => html`
          <button class=${`pm-tab${activeTab === t.id ? " active" : ""}`}
                  onClick=${() => setActiveTab(t.id)}>${t.label}</button>
        `)}
      </nav>

      ${activeTab === "overview" ? html`
        <section>
          <h3 style=${{ fontSize: "1rem", margin: "0.5rem 0" }}>Pipeline state</h3>
          <${StateGraph} graph=${rt.graph} />
          ${rt.dumps && rt.dumps.configured
            ? html`<p class="form-help" style=${{ marginTop: "0.75rem" }}>
                💾 ${rt.dumps.dumps.length} hidden_state dumps under
                <code>${rt.dumps.dumps_dir}</code>
              </p>`
            : null}
        </section>
        <section style=${{ marginTop: "1rem" }}>
          <h3 style=${{ fontSize: "1rem", margin: "0.5rem 0" }}>
            Live agents
            <span class="muted" style=${{ fontSize: "0.8rem", fontWeight: "normal" }}>
              （Elapsed / Last output 用于判断是否卡死）
            </span>
          </h3>
          <${AgentsPanel} agents=${agents || []} />
        </section>
      ` : null}

      ${activeTab === "phases" ? html`
        <${PhaseCards} taskId=${taskId} currentPhase=${phase}
          iterations=${rt.iterations} />
      ` : null}

      ${activeTab === "iterations" ? html`
        <${IterationsTable} iterations=${rt.iterations} />
      ` : null}

      ${activeTab === "p6" ? html`
        <${P6List} p6=${rt.p6} />
      ` : null}

      ${activeTab === "memory" ? html`
        <${MemoryViewer} taskId=${taskId} />
      ` : null}

      ${activeTab === "live" ? html`
        <div style=${{ display: "grid", gap: "1rem", gridTemplateColumns: "1fr 1fr" }}>
          <div>
            <h3 style=${{ fontSize: "0.9rem" }}>Agents</h3>
            <${AgentsPanel} agents=${agents || []} />
          </div>
          <div>
            <h3 style=${{ fontSize: "0.9rem" }}>Timeline</h3>
            <${Timeline} events=${timeline || []} />
          </div>
        </div>
      ` : null}
    </div>
  `;
}
