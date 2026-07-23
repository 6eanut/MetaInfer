/**
 * Per-job detail panel — live stdout/stderr tail + force-kill button.
 *
 * Polls /api/cluster/jobs/{worker}/{job_id}/stdout?offset=N for incremental
 * content. Each poll sends the new offset so we only fetch new bytes.
 */

import { html } from "htm/preact";
import { useEffect, useState, useRef, useCallback } from "preact/hooks";

const POLL_MS = 1000;

export function ClusterJobDetail({ workerNodeId, jobId, onClose }) {
  const [stdoutOffset, setStdoutOffset] = useState(0);
  const [stderrOffset, setStderrOffset] = useState(0);
  const [stdout, setStdout] = useState("");
  const [stderr, setStderr] = useState("");
  const [error, setError] = useState(null);
  const stdoutRef = useRef(null);

  const pollOnce = useCallback(async () => {
    try {
      const u = await fetch(
        `/api/cluster/jobs/${workerNodeId}/${jobId}/stdout?offset=${stdoutOffset}`
      );
      if (u.ok) {
        const text = await u.text();
        if (text.length > 0) {
          setStdout((prev) => prev + text);
          setStdoutOffset((prev) => prev + text.length);
        }
      }
      const r = await fetch(
        `/api/cluster/jobs/${workerNodeId}/${jobId}/stderr?offset=${stderrOffset}`
      );
      if (r.ok) {
        const text = await r.text();
        if (text.length > 0) {
          setStderr((prev) => prev + text);
          setStderrOffset((prev) => prev + text.length);
        }
      }
      setError(null);
    } catch (e) {
      setError(String(e));
    }
  }, [workerNodeId, jobId, stdoutOffset, stderrOffset]);

  useEffect(() => {
    pollOnce();
    const id = setInterval(pollOnce, POLL_MS);
    return () => clearInterval(id);
  }, [pollOnce]);

  // Auto-scroll stdout to bottom on update.
  useEffect(() => {
    if (stdoutRef.current) {
      stdoutRef.current.scrollTop = stdoutRef.current.scrollHeight;
    }
  }, [stdout]);

  const forceKill = useCallback(async () => {
    if (!confirm(`Force-release GPU slot for ${workerNodeId}/${jobId}?`)) return;
    // The job's GPU slot is held by worker_node_id; we don't know the exact
    // gpu_idx here. For now, fetch scoreboard and find one whose job_id matches.
    try {
      const claims = await fetch("/api/cluster/scoreboard").then((r) => r.json());
      const match = claims.find((c) => c.job_id === jobId);
      if (!match) {
        alert("No active claim found for this job (it may already be released).");
        return;
      }
      await fetch("/api/cluster/scoreboard/force-release", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          node_id: match.node_id,
          gpu_idx: match.gpu_idx,
          reason: "webui-job-detail",
        }),
      });
    } catch (e) {
      setError(String(e));
    }
  }, [workerNodeId, jobId]);

  return html`
    <div class="cluster-job-detail">
      <div class="cluster-header">
        <h2>Job <code>${jobId.slice(0, 12)}</code></h2>
        <div>
          <button class="btn danger small" onClick=${forceKill}>Force kill</button>
          <button class="btn ghost" onClick=${onClose}>← Back</button>
        </div>
      </div>
      <p class="muted">on worker <code>${workerNodeId}</code></p>

      ${error ? html`<div class="cluster-err">${error}</div>` : null}

      <h3>stdout</h3>
      <pre class="log-box" ref=${stdoutRef}>${stdout || "(empty)"}</pre>

      <h3>stderr</h3>
      <pre class="log-box">${stderr || "(empty)"}</pre>
    </div>
  `;
}
