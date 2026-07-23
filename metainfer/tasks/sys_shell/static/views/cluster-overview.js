/**
 * Cluster admin overview — workers list + scoreboard grid.
 *
 * Polls /api/cluster/workers and /api/cluster/scoreboard every few seconds.
 * Click a held GPU to open force-release confirmation.
 */

import { html } from "htm/preact";
import { useEffect, useState, useCallback } from "preact/hooks";

const POLL_MS = 3000;

export function ClusterOverview({ onClose }) {
  const [workers, setWorkers] = useState(null);
  const [claims, setClaims] = useState(null);
  const [error, setError] = useState(null);
  const [pendingKill, setPendingKill] = useState(null); // {node_id, gpu_idx, holder}

  const refresh = useCallback(async () => {
    try {
      const [w, s] = await Promise.all([
        fetch("/api/cluster/workers").then((r) => r.json()),
        fetch("/api/cluster/scoreboard").then((r) => r.json()),
      ]);
      setWorkers(w);
      setClaims(s);
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, []);

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, POLL_MS);
    return () => clearInterval(id);
  }, [refresh]);

  const confirmForceRelease = useCallback(async () => {
    if (!pendingKill) return;
    try {
      await fetch("/api/cluster/scoreboard/force-release", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          node_id: pendingKill.node_id,
          gpu_idx: pendingKill.gpu_idx,
          reason: "webui-admin",
        }),
      });
    } catch (e) {
      setError(String(e));
    }
    setPendingKill(null);
    refresh();
  }, [pendingKill, refresh]);

  return html`
    <div class="cluster-overview">
      <div class="cluster-header">
        <h2>Cluster</h2>
        <button class="btn ghost" onClick=${onClose}>← Back to tasks</button>
      </div>

      ${error ? html`<div class="cluster-err">${error}</div>` : null}

      <section>
        <h3>Workers (${workers ? workers.length : 0})</h3>
        <table class="cluster-table">
          <thead>
            <tr>
              <th>Node ID</th><th>Status</th><th>IP</th><th>Hostname</th>
              <th>GPUs</th><th>Last heartbeat</th>
            </tr>
          </thead>
          <tbody>
            ${(workers || []).map((w) => html`
              <tr key=${w.node_id}>
                <td>${w.node_id}</td>
                <td><span class=${`pill ${w.alive ? "ok" : "dead"}`}>
                  ${w.alive ? "alive" : "DEAD"}
                </span></td>
                <td>${w.ip}</td>
                <td>${w.hostname}</td>
                <td>${Object.keys(w.gpu_topology || {}).length}</td>
                <td>${w.last_heartbeat_ago_s != null
                  ? `${w.last_heartbeat_ago_s.toFixed(0)}s ago`
                  : "—"}</td>
              </tr>
            `)}
          </tbody>
        </table>
      </section>

      <section>
        <h3>Scoreboard (${claims ? claims.length : 0} GPU${claims && claims.length === 1 ? "" : "s"} held)</h3>
        ${(claims && claims.length > 0) ? html`
          <table class="cluster-table">
            <thead>
              <tr>
                <th>Node</th><th>GPU</th><th>Holder</th><th>Job</th>
                <th>Held for</th><th>Lease remaining</th><th></th>
              </tr>
            </thead>
            <tbody>
              ${claims.map((c) => html`
                <tr key=${`${c.node_id}-${c.gpu_idx}`}>
                  <td>${c.node_id}</td>
                  <td>gpu-${c.gpu_idx}</td>
                  <td>${c.holder}</td>
                  <td><code>${(c.job_id || "").slice(0, 8)}</code></td>
                  <td>${c.acquired_ago_s.toFixed(0)}s</td>
                  <td>${c.lease_remaining_s.toFixed(0)}s</td>
                  <td>
                    <button class="btn danger small"
                      onClick=${() => setPendingKill({
                        node_id: c.node_id, gpu_idx: c.gpu_idx, holder: c.holder,
                      })}>Force release</button>
                  </td>
                </tr>
              `)}
            </tbody>
          </table>
        ` : html`<p class="muted">No GPUs currently held.</p>`}
      </section>

      ${pendingKill ? html`
        <div class="modal-backdrop" onClick=${() => setPendingKill(null)}>
          <div class="modal" onClick=${(e) => e.stopPropagation()}>
            <h3>Force release ${pendingKill.node_id}/gpu-${pendingKill.gpu_idx}?</h3>
            <p>Held by <code>${pendingKill.holder}</code>. The worker will receive a
              cancel marker and SIGTERM its child process.</p>
            <div class="modal-actions">
              <button class="btn ghost" onClick=${() => setPendingKill(null)}>Cancel</button>
              <button class="btn danger" onClick=${confirmForceRelease}>Force release</button>
            </div>
          </div>
        </div>
      ` : null}
    </div>
  `;
}
