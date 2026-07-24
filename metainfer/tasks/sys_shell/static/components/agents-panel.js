// Live sub-agent panel. Shows every running agent + a tail of the most
// recently-finished ones. The snapshot file grows unbounded across a
// long run, so we cap finished rows to keep the panel useful.
//
// Each row is expandable: click the row to fetch the agent's latest
// stream-json activity (text responses + tool calls) so the operator
// can see at a glance whether the agent is heading in the right direction.

import { html } from "htm/preact";
import { useEffect, useState, useCallback } from "preact/hooks";
import { labelFor, fmtDur, fmtAgo } from "app/utils";
import { getAgentTail } from "app/api";

const MAX_SHOWN_FINISHED = 5;

export function AgentsPanel({ agents, taskId }) {
  const all = agents?.agents || [];
  const nowRunning = all.filter((a) => a.success === null);
  const finished = all
    .filter((a) => a.success !== null)
    .slice()
    .sort(
      (a, b) => (a.last_output_age_s ?? Infinity) - (b.last_output_age_s ?? Infinity),
    );
  const shownFinished = finished.slice(0, MAX_SHOWN_FINISHED);
  const hiddenFinished = finished.length - shownFinished.length;
  const visible = nowRunning.concat(shownFinished);

  const rows = visible.map((a) => {
    return html`<${AgentRow} key=${a.name} agent=${a} taskId=${taskId} />`;
  });

  const parts = [`${nowRunning.length} running`];
  if (shownFinished.length) parts.push(`${shownFinished.length} recent finished`);
  if (hiddenFinished > 0) parts.push(`${hiddenFinished} older (hidden)`);

  return html`
    <div class="agents-panel">
      <div class="agents-summary">
        ${parts.join(" · ")}
        <span class="muted hint">(click row to expand latest output)</span>
      </div>
      ${rows.length === 0
        ? html`<p class="muted">no active agents</p>`
        : html`<table class="agents-table">
            <thead>
              <tr>
                <th></th>
                <th>Name</th><th>Role</th><th>Phase</th><th>Attempt</th>
                <th>Elapsed</th><th>Last output</th><th>Success</th><th>Log</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>`}
    </div>
  `;
}

function AgentRow({ agent, taskId }) {
  const [expanded, setExpanded] = useState(false);
  const [tail, setTail] = useState(null);
  const [tailErr, setTailErr] = useState(null);
  const [loadingTail, setLoadingTail] = useState(false);

  const loadTail = useCallback(async () => {
    if (!taskId || !agent.name) return;
    setLoadingTail(true);
    try {
      const r = await getAgentTail(taskId, agent.name, 10);
      setTail(r);
      setTailErr(null);
    } catch (e) {
      setTailErr(e.message || String(e));
    } finally {
      setLoadingTail(false);
    }
  }, [taskId, agent.name]);

  // When expanded, poll every 5s. When collapsed, stop.
  useEffect(() => {
    if (!expanded) return;
    loadTail();
    const id = setInterval(loadTail, 5000);
    return () => clearInterval(id);
  }, [expanded, loadTail]);

  const toggle = useCallback(() => setExpanded((v) => !v), []);
  const logShort = (agent.log_file || "").split("/").slice(-2).join("/");

  return html`
    <${Frag}>
      <tr class=${"agents-row" + (expanded ? " expanded" : "")} onClick=${toggle}>
        <td class="expand-cell">${expanded ? "▼" : "▶"}</td>
        <td class="mono">${agent.name}</td>
        <td>${agent.role}</td>
        <td><span class="pill ${agent.phase}">${labelFor(agent.phase)}</span></td>
        <td>${agent.attempt}</td>
        <td>${fmtDur(agent.elapsed_s)}</td>
        <td>${fmtAgo(agent.last_output_age_s)}</td>
        <td>${agent.success === null
          ? "—"
          : (agent.success
            ? html`<span class="pill success">ok</span>`
            : html`<span class="pill failed">fail</span>`)}</td>
        <td class="mono" title=${agent.log_file || ""}>${logShort}</td>
      </tr>
      ${expanded ? html`<${AgentTailRow}
          agent=${agent} tail=${tail} err=${tailErr} loading=${loadingTail} />` : null}
    <//>
  `;
}

function AgentTailRow({ agent, tail, err, loading }) {
  // Render as a sibling row that spans all columns.
  const events = tail?.events || [];
  return html`
    <tr class="agents-tail-row">
      <td colSpan=${9}>
        <div class="agents-tail">
          <div class="agents-tail-header">
            <span class="mono muted">${tail?.log_file || agent.log_file || ""}</span>
            ${loading ? html`<span class="muted"> (loading…)</span>` : null}
            ${err ? html`<span class="failed"> · ${err}</span>` : null}
          </div>
          ${events.length === 0
            ? html`<p class="muted">(no events yet)</p>`
            : html`<ul class="agents-tail-events">
                ${events.map((e, i) => html`<${TailEvent} key=${i} evt=${e} />`)}
              </ul>`}
        </div>
      </td>
    </tr>
  `;
}

function TailEvent({ evt }) {
  const t = evt.type;
  if (t === "text") {
    return html`<li class="evt-text">
      <span class="evt-tag text">text</span>
      <span class="evt-body">${evt.text}</span>
    </li>`;
  }
  if (t === "tool_use") {
    return html`<li class="evt-tool-use">
      <span class="evt-tag tool">tool</span>
      <span class="evt-tool-name">${evt.name}</span>
      <span class="evt-tool-input mono">${evt.input_brief || ""}</span>
    </li>`;
  }
  if (t === "tool_result") {
    return html`<li class="evt-tool-result">
      <span class="evt-tag result">→</span>
      <span class="evt-body muted">${(evt.brief || "").slice(0, 200)}</span>
    </li>`;
  }
  if (t === "raw") {
    return html`<li class="evt-raw">
      <span class="evt-tag raw">log</span>
      <span class="evt-body mono">${evt.text}</span>
    </li>`;
  }
  return html`<li><span class="muted">${JSON.stringify(evt).slice(0, 200)}</span></li>`;
}

// Minimal fragment wrapper so we can return sibling <tr> rows from AgentRow.
function Frag({ children }) { return children; }
