// Per-round, per-agent analysis browser for calc-value tasks.
//
// Shows each agent's individual output (including disagreements) at every
// iteration round of every step. Surfaces what each agent said before the
// deterministic merge / convergence logic collapsed them into a consensus.

import { html } from "htm/preact";
import { useState, useCallback } from "preact/hooks";
import { escapeHtml } from "app/utils";
import { QAModal } from "app/qa-modal";

const STEP1_LABELS = {
  agent_a: "A · top-down (config.json)",
  agent_b: "B · bottom-up (cmdline/env)",
  agent_c: "C · weight-driven (load paths)",
};

function MemoryRow({ label, value }) {
  if (value == null || value === "" || (Array.isArray(value) && value.length === 0)) {
    return null;
  }
  let display = value;
  if (typeof value === "object") {
    display = JSON.stringify(value);
  } else {
    display = String(value);
  }
  if (display.length > 240) display = display.slice(0, 240) + "…";
  return html`
    <div class="mem-row">
      <span class="mem-key">${label}</span>
      <span class="mem-val">${display}</span>
    </div>
  `;
}

function AgentCard({ agent, angleLabel, onAsk }) {
  const [showFull, setShow] = useState(false);
  const mem = agent.memory || {};
  const memKeys = Object.keys(mem);
  const isEmpty = !agent.has_memory && !agent.response_excerpt;

  return html`
    <div class=${`agent-card ${isEmpty ? "empty" : ""}`}>
      <div class="agent-card-head">
        <span class="agent-name">${agent.name}</span>
        ${angleLabel ? html`<span class="muted">${angleLabel}</span>` : null}
        ${agent.parse_error
          ? html`<span class="pill failed">parse error</span>`
          : agent.has_memory
            ? html`<span class="pill done">parsed</span>`
            : html`<span class="pill idle">empty</span>`}
        ${onAsk && agent.events_file
          ? html`<button class="link-btn qa-ask-btn"
              onClick=${() => onAsk(agent)}>💬 Ask</button>`
          : null}
      </div>
      ${agent.parse_error
        ? html`<pre class="agent-error">${agent.parse_error}</pre>`
        : null}
      ${memKeys.length > 0
        ? html`<div class="agent-memory">
            ${memKeys.slice(0, showFull ? memKeys.length : 6).map(
              (k) => html`<${MemoryRow} key=${k} label=${k} value=${mem[k]} />`,
            )}
          </div>`
        : null}
      ${memKeys.length > 6
        ? html`<button class="link-btn"
            onClick=${() => setShow((v) => !v)}>
            ${showFull ? "show less" : `show all ${memKeys.length} keys`}
          </button>`
        : null}
      ${agent.response_excerpt
        ? html`<details class="agent-raw">
            <summary>raw response (${agent.response_excerpt.length * 1}+ chars)</summary>
            <pre>${escapeHtml(agent.response_excerpt)}</pre>
          </details>`
        : null}
    </div>
  `;
}

function DisputeList({ disputes }) {
  if (!disputes || disputes.length === 0) {
    return html`<div class="disputes none">
      <span class="pill done">no disputes</span>
    </div>`;
  }
  return html`
    <div class="disputes">
      <div class="disputes-head">
        <span class="pill failed">${disputes.length} dispute(s)</span>
      </div>
      <ul>
        ${disputes.map((d, i) => html`
          <li key=${i}>
            <code>${d.field || "?"}</code>:
            ${d.values
              ? d.values.map((v, j) => html`
                  <span key=${j} class="disp-val">${JSON.stringify(v)}</span>
                `)
              : (d.note || "")}
          </li>
        `)}
      </ul>
    </div>
  `;
}

function Step1Round({ round, onAsk }) {
  return html`
    <details class="iter-round" open>
      <summary>
        <span class="round-label">Round ${round.round}</span>
        ${round.converged
          ? html`<span class="pill done">converged</span>`
          : html`<span class="pill failed">disputes → reconcile</span>`}
        <span class="muted">${round.agents.filter((a) => a.has_memory).length}/3
          agents produced memory</span>
      </summary>
      <${DisputeList} disputes=${round.disputes} />
      <div class="agent-grid">
        ${round.agents.map((a) => html`
          <${AgentCard} key=${a.name} agent=${a}
            angleLabel=${STEP1_LABELS[a.name] || null}
            onAsk=${onAsk ? (agent) => onAsk(round, agent) : null} />
        `)}
      </div>
    </details>
  `;
}

function Step2Round({ round }) {
  const kindLabel = {
    build: "build", validate: "validate", fix: "fix", other: "other",
  }[round.kind] || round.kind;
  const sections = round.sections || [];
  return html`
    <details class="iter-round">
      <summary>
        <span class="round-label">${round.dir}</span>
        <span class="pill idle">${kindLabel}</span>
        ${round.node_count != null
          ? html`<span class="muted">
              ${round.section_count || "?"} section(s) ·
              ${round.node_count} template nodes ·
              ${round.aggregated_node_count ?? round.node_count} aggregated ·
              ${round.edge_count || 0} edges
            </span>`
          : null}
        ${round.pass != null
          ? html`<span class="muted">${round.pass} pass · ${round.reject} reject</span>`
          : null}
      </summary>
      ${sections.length > 0
        ? html`<div class="section-badges">
            ${sections.map((s, i) => html`
              <span key=${i} class="section-badge ${s.kind || ""}">
                <code>${s.id || "?"}</code>
                <span class="kind">${s.kind || "?"}</span>
                ${s.repeat_count > 1
                  ? html`<span class="repeat">×${s.repeat_count}</span>`
                  : null}
                <span class="muted">${s.node_count}n</span>
              </span>
            `)}
          </div>`
        : null}
      ${round.verdicts && round.verdicts.length > 0
        ? html`<table class="verdicts-table">
            <thead><tr><th>Section</th><th>Node</th><th>Verdict</th><th>Reason</th></tr></thead>
            <tbody>
              ${round.verdicts.map((v, i) => html`
                <tr key=${i} class=${v.verdict === "reject" ? "rejected" : "passed"}>
                  <td><code>${v.section_id || "?"}</code></td>
                  <td><code>${v.node_id || "?"}</code></td>
                  <td><span class=${`pill ${v.verdict === "pass" ? "done" : "failed"}`}>
                    ${v.verdict}</span></td>
                  <td>${(v.reason || "").slice(0, 200)}</td>
                </tr>
              `)}
            </tbody>
          </table>`
        : null}
      ${round.validators && round.validators.length > 0
        ? html`<details class="agent-raw">
            <summary>per-validator raw responses (${round.validators.length})</summary>
            ${round.validators.map((v) => html`
              <div key=${v.name} class="validator-raw">
                <div class="agent-card-head">
                  <span class="agent-name">${v.name}</span>
                </div>
                ${v.response_excerpt
                  ? html`<pre>${escapeHtml(v.response_excerpt)}</pre>`
                  : html`<span class="muted">(no response)</span>`}
              </div>
            `)}
          </details>`
        : null}
    </details>
  `;
}

function WriterCard({ writer, onAsk }) {
  return html`
    <div class=${`agent-card ${writer.has_script ? "" : "empty"}`}>
      <div class="agent-card-head">
        <span class="agent-name">${writer.name}</span>
        ${writer.error
          ? html`<span class="pill failed">error</span>`
          : writer.has_script
            ? html`<span class="pill done">calc.py</span>`
            : html`<span class="pill idle">no script</span>`}
        ${onAsk && writer.events_file
          ? html`<button class="link-btn qa-ask-btn"
              onClick=${() => onAsk(writer)}>💬 Ask</button>`
          : null}
      </div>
      ${writer.error
        ? html`<pre class="agent-error">${writer.error}</pre>`
        : null}
      ${writer.script_excerpt
        ? html`<details class="agent-raw" open>
            <summary>calc.py</summary>
            <pre>${escapeHtml(writer.script_excerpt)}</pre>
          </details>`
        : null}
      ${writer.response_excerpt
        ? html`<details class="agent-raw">
            <summary>raw response</summary>
            <pre>${escapeHtml(writer.response_excerpt)}</pre>
          </details>`
        : null}
    </div>
  `;
}

function Step3NodeRound({ round, onAsk }) {
  return html`
    <details class="iter-round">
      <summary>
        <span class="round-label">Round ${round.round}</span>
        ${round.ok
          ? html`<span class="pill done">agreement</span>`
          : round.median_fallback
            ? html`<span class="pill failed">median fallback</span>`
            : html`<span class="pill failed">${round.mismatch_count || "?"} mismatches</span>`}
        <span class="muted">${round.writers.filter((w) => w.has_script).length}/3
          writers produced calc.py</span>
      </summary>
      ${round.mismatches_excerpt && round.mismatches_excerpt.length > 0
        ? html`<div class="mismatches">
            <h4>First mismatches (showing ${round.mismatches_excerpt.length})</h4>
            <ul>${round.mismatches_excerpt.map((m, i) => html`
              <li key=${i}>
                batch_size=${m.batch_size}, seq_len=${m.seq_len}:
                ${(m.values || []).map((v, j) => html`
                  <span key=${j} class="disp-val">
                    tflops=${v?.tflops?.toFixed?.(6) ?? JSON.stringify(v?.tflops)},
                    gb=${v?.access_gb?.toFixed?.(6) ?? JSON.stringify(v?.access_gb)}
                  </span>
                `)}
              </li>
            `)}</ul>
          </div>`
        : null}
      <div class="agent-grid">
        ${round.writers.map((w) => html`
          <${WriterCard} key=${w.name} writer=${w}
            onAsk=${onAsk ? (writer) => onAsk(round, writer) : null} />
        `)}
      </div>
    </details>
  `;
}

function Step3Node({ node, onAsk }) {
  const title = node.section_id
    ? `${node.section_id}__${node.node_id}`
    : node.node_id;
  return html`
    <details class="iter-node">
      <summary>
        <code class="node-id">${title}</code>
        ${node.section_kind
          ? html`<span class="pill idle">${node.section_kind}</span>`
          : null}
        ${node.section_repeat_count > 1
          ? html`<span class="muted">×${node.section_repeat_count}</span>`
          : null}
        <span class="muted">${node.rounds.length} round(s)</span>
      </summary>
      ${node.rounds.map((r) => html`
        <${Step3NodeRound} key=${r.round} round=${r}
          onAsk=${onAsk ? (round, writer) => onAsk(node, round, writer) : null} />
      `)}
    </details>
  `;
}

export function CalcIterations({ iterations, loading, error, taskId }) {
  const [qaTarget, setQaTarget] = useState(null);

  // Each step's onAsk builds a uniform "target" shape for QAModal.
  const onAskS1 = useCallback((round, agent) => {
    setQaTarget({
      step: 1,
      round: round.round,
      round_label: `S1 round ${round.round}`,
      agent: agent.name,
      events_file: agent.events_file,
      target_workdir: agent.target_workdir,
      label: `S1 · round ${round.round} · ${agent.name}`,
    });
  }, []);

  const onAskS3 = useCallback((node, round, writer) => {
    setQaTarget({
      step: 3,
      round: round.round,
      round_label: `S3 ${node.node_id} round ${round.round}`,
      agent: writer.name,
      // S3 writer spec name is `{safe_nid}_writer_{i}` — but the events
      // file path returned by the iterations endpoint already points
      // to the right log_dir, so we just pass it through.
      node_id: node.node_id,
      events_file: writer.events_file,
      target_workdir: writer.target_workdir,
      label: `S3 · ${node.node_id} · round ${round.round} · ${writer.name}`,
    });
  }, []);

  if (loading && !iterations) {
    return html`<div class="panel calc-iter"><p class="muted">loading iterations…</p></div>`;
  }
  if (error) {
    return html`<div class="panel calc-iter">
      <p class="err">failed to load: ${error}</p>
    </div>`;
  }
  if (!iterations) return null;

  const s1 = iterations.s1_analyze || [];
  const s2 = iterations.s2_graph || [];
  const s3 = iterations.s3_calculate || [];

  return html`
    <div class="panel calc-iter">
      <h2>Per-agent iterations
        <span class="muted">(each agent's individual analysis, including disagreements)</span>
      </h2>

      ${s1.length === 0 && s2.length === 0 && s3.length === 0
        ? html`<p class="muted">No iteration rounds recorded yet.</p>` : null}

      ${s1.length > 0 ? html`
        <section class="iter-step">
          <h3>Step 1 · code analysis (2 agents × ${s1.length} round${s1.length === 1 ? "" : "s"})</h3>
          ${s1.map((r) => html`
            <${Step1Round} key=${r.round} round=${r}
              onAsk=${taskId ? onAskS1 : null} />
          `)}
        </section>` : null}

      ${s2.length > 0 ? html`
        <section class="iter-step">
          <h3>Step 2 · graph build + validate (${s2.length} round${s2.length === 1 ? "" : "s"})</h3>
          ${s2.map((r, i) => html`<${Step2Round} key=${r.dir || i} round=${r} />`)}
        </section>` : null}

      ${s3.length > 0 ? html`
        <section class="iter-step">
          <h3>Step 3 · calc writers (${s3.length} node(s))</h3>
          ${s3.map((n) => html`
            <${Step3Node} key=${n.node_id} node=${n}
              onAsk=${taskId ? onAskS3 : null} />
          `)}
        </section>` : null}
    </div>

    ${qaTarget ? html`
      <${QAModal}
        taskId=${taskId}
        target=${qaTarget}
        onClose=${() => setQaTarget(null)} />
    ` : null}
  `;
}
