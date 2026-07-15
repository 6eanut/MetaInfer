// Cell detail modal — shows ONE (compound × angle × round) cell.
//
// Three sections:
//   1. Agent thinking — response.txt rendered as markdown (the agent's
//      reasoning + commentary about its calc.py).
//   2. calc.py source — the actual Python script.
//   3. Canonical-shape result + mismatches vs the other angle.
//   4. Q&A — user can ask an analyst questions about why this agent
//      produced this output. Reuses the calc/qa/start endpoint with
//      the writer's events.jsonl as the target transcript.

import { html } from "htm/preact";
import { useEffect, useState, useCallback } from "preact/hooks";
import { marked } from "marked";
import {
  getCalcCellDetail, startCalcCellQa, getCalcQa,
} from "app/calc-api";
import { escapeHtml } from "app/utils";

const POLL_INTERVAL_MS = 2000;
const MAX_POLL_MS = 11 * 60 * 1000;

function fmtNum(n, digits = 3) {
  if (n == null || Number.isNaN(n)) return "—";
  if (!isFinite(n)) return "∞";
  return n.toFixed(digits);
}

export function CalcCellModal({ taskId, compound, angle, roundIdx, onClose }) {
  const [detail, setDetail] = useState(null);
  const [err, setErr] = useState(null);
  const [draft, setDraft] = useState("");
  const [active, setActive] = useState(null);
  const [submitErr, setSubmitErr] = useState(null);

  // Fetch cell detail.
  useEffect(() => {
    let alive = true;
    setDetail(null); setErr(null);
    getCalcCellDetail(taskId, compound, angle, roundIdx)
      .then((d) => { if (alive) setDetail(d); })
      .catch((e) => { if (alive) setErr(String(e)); });
    return () => { alive = false; };
  }, [taskId, compound, angle, roundIdx]);

  // Poll active QA session.
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
        const s = await getCalcQa(taskId, active.sid);
        const status = s.status?.status || "unknown";
        setActive({
          sid: active.sid,
          question: s.request?.question || active.question,
          status,
          answer: s.answer,
          error: s.status?.error,
        });
        if (status === "done" || status === "failed") clearInterval(id);
      } catch (e) { /* keep polling */ }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, [active?.sid, active?.status, taskId]);

  const onSubmit = useCallback(async (ev) => {
    ev.preventDefault();
    const q = draft.trim();
    if (!q || !detail?.events_file) return;
    setSubmitErr(null);
    try {
      const resp = await startCalcCellQa(taskId, {
        events_file: detail.events_file,
        target_workdir: detail.workdir,
        target_label: `S3 cell compound=${compound} angle=${angle} round=${roundIdx}`,
        question: q,
        step: "S3_calculate",
        round: String(roundIdx),
        agent: `s3_${compound}_${angle}_r${roundIdx}`,
      });
      setActive({
        sid: resp.session_id, question: q,
        status: "running", answer: null, error: null,
      });
      setDraft("");
    } catch (e) {
      setSubmitErr(String(e));
    }
  }, [draft, taskId, detail, compound, angle, roundIdx]);

  const thinkingHtml = detail?.response
    ? marked.parse(detail.response)
    : "";

  return html`
    <div class="modal-overlay open" onClick=${(e) => {
      if (e.target === e.currentTarget) onClose();
    }}>
      <div class="modal calc-cell-modal">
        <div class="modal-header">
          <h3>
            Cell detail
            <span class="muted small">
              · <code>${compound}</code> · angle=${angle} · round=${roundIdx}
            </span>
          </h3>
          <button class="close" onClick=${onClose}>×</button>
        </div>
        <div class="modal-body">
          ${err ? html`<div class="task-banner task-banner-err">${err}</div>` : null}
          ${!detail && !err ? html`<p class="muted">加载中…</p>` : null}

          ${detail ? html`
            ${detail.calc_py ? html`
              <section class="cell-section">
                <h4>calc.py 源码</h4>
                <pre class="code-block"><code>${detail.calc_py}</code></pre>
              </section>
            ` : html`<div class="warn">无 calc.py（agent 未产出源码）</div>`}

            ${detail.response ? html`
              <section class="cell-section">
                <h4>Agent 思考过程 (response.txt)</h4>
                <div class="markdown-body" dangerouslySetInnerHTML=${{ __html: thinkingHtml }}></div>
              </section>
            ` : null}

            ${detail.result ? html`
              <section class="cell-section">
                <h4>计算结果 <span class="muted small">
                  （canonical shape B=${detail.result.batch_size}, S=${detail.result.seq_len}
                  · decode = 1 token + seq_len KV cache 读取）
                </span></h4>
                <table class="calc-table compact">
                  <thead>
                    <tr>
                      <th colspan="2" class="center">prefill</th>
                      <th colspan="2" class="center">decode</th>
                    </tr>
                    <tr>
                      <th class="right">tflops</th>
                      <th class="right">access_gb</th>
                      <th class="right">tflops</th>
                      <th class="right">access_gb</th>
                    </tr>
                  </thead>
                  <tbody>
                    ${(() => {
                      const pre = detail.result.prefill || {};
                      const dec = detail.result.decode || {};
                      return html`
                        <tr>
                          <td class="right">${fmtNum(pre.tflops)}</td>
                          <td class="right">${fmtNum(pre.access_gb)}</td>
                          <td class="right">${fmtNum(dec.tflops)}</td>
                          <td class="right">${fmtNum(dec.access_gb)}</td>
                        </tr>
                      `;
                    })()}
                  </tbody>
                </table>
              </section>
            ` : null}

            ${detail.mismatches && detail.mismatches.length > 0 ? html`
              <section class="cell-section">
                <h4>2 视角分歧（${detail.mismatches.length} 条）</h4>
                <p class="muted small">
                  canonical shape 下 2 视角的 prefill/decode tflops/gb 差异。
                  spread &gt; 5% 容差时本轮该节点会被标记为 disputed 进入下一轮。
                </p>
                <details>
                  <summary>展开 mismatch 详情</summary>
                  <pre class="code-block small">${JSON.stringify(detail.mismatches.slice(0, 5), null, 2)}</pre>
                </details>
              </section>
            ` : null}

            <section class="cell-section cell-qa">
              <h4>提问 / 讨论</h4>
              ${detail.events_file ? html`
                <p class="muted small">
                  transcript: <code>${detail.events_file}</code>
                </p>
              ` : html`<div class="warn">无 transcript（无法提问）</div>`}
              ${active ? html`
                <div class="qa-active">
                  <div class="qa-turn">
                    <div class="qa-q">
                      <span class="qa-role">you</span>
                      <span>${active.question}</span>
                    </div>
                  </div>
                  ${active.status === "running"
                    ? html`<div class="qa-thinking">分析师正在读 transcript…</div>`
                    : (active.status === "done"
                      ? html`<div class="qa-a markdown-body">${
                        active.answer ? marked.parse(active.answer) : "(empty)"
                      }</div>`
                      : html`<div class="task-banner task-banner-err">${
                        active.error || "QA failed"
                      }</div>`)}
                </div>
              ` : null}
              ${detail.events_file ? html`
                <form onSubmit=${onSubmit} class="qa-form">
                  <input
                    type="text"
                    value=${draft}
                    onInput=${(e) => setDraft(e.target.value)}
                    placeholder="针对这个 cell 提问，例如：为什么 MoE 没算 scaling factor？"
                    class="qa-input" />
                  <button type="submit" class="btn primary" disabled=${!draft.trim()}>
                    提问
                  </button>
                </form>
              ` : null}
              ${submitErr ? html`<div class="task-banner task-banner-err">${submitErr}</div>` : null}
            </section>
          ` : null}
        </div>
      </div>
    </div>
  `;
}
