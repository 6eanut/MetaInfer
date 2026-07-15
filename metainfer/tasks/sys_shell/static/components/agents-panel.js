// Live sub-agent panel. Shows every running agent + a tail of the most
// recently-finished ones. The snapshot file grows unbounded across a
// long run, so we cap finished rows to keep the panel useful.

import { html } from "htm/preact";
import { labelFor, fmtDur, fmtAgo } from "app/utils";

const MAX_SHOWN_FINISHED = 5;

export function AgentsPanel({ agents }) {
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
    const logShort = (a.log_file || "").split("/").slice(-2).join("/");
    return html`
      <tr key=${a.name}>
        <td class="mono">${a.name}</td>
        <td>${a.role}</td>
        <td><span class="pill ${a.phase}">${labelFor(a.phase)}</span></td>
        <td>${a.attempt}</td>
        <td>${fmtDur(a.elapsed_s)}</td>
        <td>${fmtAgo(a.last_output_age_s)}</td>
        <td>${a.success === null
          ? "—"
          : (a.success
            ? html`<span class="pill success">ok</span>`
            : html`<span class="pill failed">fail</span>`)}</td>
        <td class="mono" title=${a.log_file || ""}>${logShort}</td>
      </tr>
    `;
  });

  const parts = [`${nowRunning.length} running`];
  if (shownFinished.length) parts.push(`${shownFinished.length} recent finished`);
  if (hiddenFinished > 0) parts.push(`${hiddenFinished} older (hidden)`);

  return html`
    <div class="agents-panel">
      <div class="agents-summary">${parts.join(" · ")}</div>
      ${rows.length === 0
        ? html`<p class="muted">no active agents</p>`
        : html`<table>
            <thead>
              <tr>
                <th>Name</th><th>Role</th><th>Phase</th><th>Attempt</th>
                <th>Elapsed</th><th>Last output</th><th>Success</th><th>Log</th>
              </tr>
            </thead>
            <tbody>${rows}</tbody>
          </table>`}
    </div>
  `;
}
