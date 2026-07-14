// Detailed audit panel — renders the streaming [node × angle] grid.
//
// Reads /api/tasks/<id>/calc/cells every 3s. Each row is one compound
// node; columns are the 2 angles (a/b) plus a spread% and status.
// Cells are clickable — opens CalcCellModal with the agent's thinking
// + calc.py source.
//
// User-controlled batch_size / seq_len inputs at the top re-run each
// cell's calc.py on the server (via ?batch_size=&seq_len= query params)
// and override the displayed values. Defaults to B=1,S=512 which matches
// the canonical shape baked into _state.json.

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { getCalcCells } from "app/api";
import { CalcCellModal } from "app/calc-cell-modal";

const ANGLES = ["a", "b"];
const DEFAULT_BATCH = 1;
const DEFAULT_SEQ = 512;

function fmtNum(n, digits = 2) {
  if (n == null || Number.isNaN(n)) return "—";
  if (!isFinite(n)) return "∞";
  if (Math.abs(n) >= 1000) return n.toFixed(0);
  return n.toFixed(digits);
}

function fmtPct(p) {
  if (p == null) return "—";
  return (p * 100).toFixed(2) + "%";
}

function cellStatusIcon(status) {
  if (status === "ok") return "✓";
  if (status === "pending") return "·";
  if (status === "failed" || status === "no_source" || status === "runtime_error") return "✗";
  return status;
}

export function CalcAuditPanel({ taskId }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [modal, setModal] = useState(null); // {compound, angle, round}
  const [batchSize, setBatchSize] = useState(DEFAULT_BATCH);
  const [seqLen, setSeqLen] = useState(DEFAULT_SEQ);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const d = await getCalcCells(taskId, batchSize, seqLen);
        if (alive) { setData(d); setErr(null); }
      } catch (e) {
        if (alive) setErr(String(e));
      }
    };
    tick();
    const id = setInterval(tick, 3000);
    return () => { alive = false; clearInterval(id); };
  }, [taskId, batchSize, seqLen]);

  if (err) {
    return html`
      <section class="panel">
        <h2>详细审计</h2>
        <div class="task-banner task-banner-err">加载失败：${err}</div>
      </section>
    `;
  }
  if (!data) {
    return html`
      <section class="panel">
        <h2>详细审计</h2>
        <p class="muted">加载中…</p>
      </section>
    `;
  }
  if (data.pending || !data.nodes) {
    return html`
      <section class="panel">
        <h2>详细审计</h2>
        <p class="muted">
          S3 尚未启动 — 详细审计在 S1+S2 完成后开始。2 个视角串行运行，
          每视角内部 5 节点并发；每完成一格立即显示。
        </p>
      </section>
    `;
  }

  const compounds = Object.keys(data.nodes).sort();
  const round = data.round || 0;

  const onCell = (compound, angle, cell) => {
    if (!cell || cell.status === "pending") return;
    setModal({ compound, angle, round: cell.round ?? round });
  };

  return html`
    <section class="panel calc-audit-panel">
      <h2>
        详细审计
        <span class="muted small">
          （round ${round} · ${compounds.length} 节点）
        </span>
      </h2>
      <div class="calc-combo-controls">
        <label>
          batch_size
          <input
            type="number"
            min="1"
            step="1"
            value=${batchSize}
            onInput=${(e) => {
              const v = parseInt(e.target.value, 10);
              if (v && v > 0) setBatchSize(v);
            }} />
        </label>
        <label>
          seq_len
          <input
            type="number"
            min="1"
            step="1"
            value=${seqLen}
            onInput=${(e) => {
              const v = parseInt(e.target.value, 10);
              if (v && v > 0) setSeqLen(v);
            }} />
        </label>
        <span class="muted small">
          表格内每个 cell 的 tflops/gb 是该 (batch, seq) 组合下按需重跑 calc.py 的结果。
          改了之后下一次轮询（3s）生效。
        </span>
      </div>
      <p class="muted small">
        点击任意已完成的单元格查看 Agent 思考过程 + calc.py 源码 + 计算结果 + 提问。
      </p>
      <div class="calc-audit-table-wrapper">
        <table class="calc-table calc-audit-table">
          <thead>
            <tr>
              <th class="left" rowspan="2">section</th>
              <th class="left" rowspan="2">node</th>
              ${ANGLES.map((a) => html`
                <th colspan="4" class="center">angle ${a}</th>
              `)}
              <th class="right" rowspan="2">spread%</th>
              <th class="left" rowspan="2">status</th>
            </tr>
            <tr>
              ${ANGLES.map(() => html`
                <th class="right">pre.tf</th>
                <th class="right">pre.gb</th>
                <th class="right">dec.tf</th>
                <th class="right">dec.gb</th>
              `)}
            </tr>
          </thead>
          <tbody>
            ${compounds.map((compound) => {
              const node = data.nodes[compound];
              const cells = node.cells || {};
              const pendingCount = ANGLES.filter((a) => (cells[a]?.status || "pending") === "pending").length;
              const doneCount = ANGLES.length - pendingCount;
              return html`
                <tr key=${compound}>
                  <td>${node.section_id}</td>
                  <td><code>${node.node_id}</code></td>
                  ${ANGLES.map((a) => {
                    const c = cells[a] || {};
                    const clickable = c.status && c.status !== "pending";
                    const pre = c.prefill || {};
                    const dec = c.decode || {};
                    // Pick the value to show in each column. Fall back to
                    // legacy top-level tflops/gb for cells written before
                    // the prefill/decode split landed.
                    const preT = pre.tflops != null ? pre.tflops : c.tflops;
                    const preG = pre.access_gb != null ? pre.access_gb : c.gb;
                    const decT = dec.tflops;
                    const decG = dec.access_gb;
                    return html`
                      <td class=${`right cell-tflops ${clickable ? "clickable" : ""}`}
                          onClick=${() => onCell(compound, a, c)}
                          title=${clickable ? "点击查看 Agent 思考过程 + 源码" : ""}>
                        ${preT != null ? fmtNum(preT) : "—"}
                      </td>
                      <td class=${`right cell-gb ${clickable ? "clickable" : ""}`}
                          onClick=${() => onCell(compound, a, c)}>
                        ${preG != null ? fmtNum(preG) : "—"}
                      </td>
                      <td class=${`right cell-tflops-decode ${clickable ? "clickable" : ""}`}
                          onClick=${() => onCell(compound, a, c)}>
                        ${decT != null ? fmtNum(decT) : "—"}
                      </td>
                      <td class=${`right cell-gb-decode ${clickable ? "clickable" : ""}`}
                          onClick=${() => onCell(compound, a, c)}>
                        ${decG != null ? fmtNum(decG) : "—"}
                      </td>
                    `;
                  })}
                  <td class="right">${fmtPct(node.spread_pct)}</td>
                  <td>
                    ${doneCount === 0
                      ? html`<span class="muted">pending</span>`
                      : doneCount < ANGLES.length
                        ? html`<span class="muted">${doneCount}/${ANGLES.length} …</span>`
                        : (node.converged
                          ? html`<span class="ok">✓ converged</span>`
                          : html`<span class="warn">⚠ disputed</span>`)}
                  </td>
                </tr>
              `;
            })}
          </tbody>
        </table>
      </div>

      ${modal ? html`
        <${CalcCellModal}
          taskId=${taskId}
          compound=${modal.compound}
          angle=${modal.angle}
          roundIdx=${modal.round}
          onClose=${() => setModal(null)} />
      ` : null}
    </section>
  `;
}
