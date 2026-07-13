// Event timeline stream. Renders the last N events, newest first.

import { html } from "htm/preact";
import { fmtTs, escapeHtml } from "app/utils";

export function Timeline({ events }) {
  const list = (events || []).slice(-80).reverse();
  if (list.length === 0) {
    return html`<div class="timeline"><p class="muted">No events yet.</p></div>`;
  }
  const items = list.map((ev, i) => {
    const type = ev.type || "";
    let cls = "";
    if (type.includes("fail") || type.includes("err")) cls = "err-line";
    else if (type.includes("success") || type.includes("end")) cls = "ok-line";
    const payload = ev.payload ? JSON.stringify(ev.payload) : "";
    return html`
      <li key=${`${ev.ts}-${i}`} class=${cls}>
        <span class="t">${fmtTs(ev.ts)}</span>
        <strong>${type}</strong>
        <span class="muted">${payload}</span>
      </li>
    `;
  });
  return html`<ul class="timeline">${items}</ul>`;
}
