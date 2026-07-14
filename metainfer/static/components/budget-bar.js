// Token / cost budget progress bar. Rendered at the top of the task
// detail view. Polls GET /api/tasks/<id>/token-budget and renders:
//
//   ┌─────────────────────────────────────────────────────────┐
//   │ Token budget   $1.23 / $50.00  (2.46%)  ━━━━━━━━━━━━━━━ │
//   │                 12 agents · 1.2M in · 240k out tokens  │
//   └─────────────────────────────────────────────────────────┘
//
// When exhausted, bar goes red and shows a "BUDGET EXHAUSTED" tag.
// When no budget is configured for the task, renders nothing (so the
// header stays uncluttered for legacy / unlimited tasks).

import { html } from "htm/preact";
import { useEffect, useState } from "preact/hooks";
import { getTokenBudget, updateTokenBudget } from "app/api";

export function BudgetBar({ taskId, refreshKey = 0 }) {
  const [budget, setBudget] = useState(null);
  const [err, setErr] = useState(null);
  const [editing, setEditing] = useState(false);

  useEffect(() => {
    if (!taskId) return;
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const b = await getTokenBudget(taskId);
        if (!cancelled) {
          setBudget(b);
          setErr(null);
        }
      } catch (e) {
        if (!cancelled) setErr(String(e));
      }
    };
    fetchOnce();
    // Poll every 5s — matches the task-detail refresh cadence. Cheap
    // endpoint (one file read server-side).
    const id = setInterval(fetchOnce, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, [taskId, refreshKey]);

  if (!budget) {
    // Initial load or fetch error before any successful fetch.
    if (err) return null;
    return null;
  }
  if (!budget.configured) {
    // Task has no cost limit set. Offer a slim "set budget" affordance
    // instead of nothing — lets the user retroactively cap a running
    // task without editing requirements.json.
    return html`
      <div class="budget-bar-wrap budget-bar-unconfigured">
        <div class="budget-bar-row">
          <span class="budget-label">Token budget</span>
          <span class="muted">未设置上限（任务无 cost 限制）</span>
          <button class="btn ghost budget-edit-btn"
                  onClick=${() => setEditing(true)}>设置预算</button>
        </div>
        ${editing ? html`
          <${BudgetEditor}
            taskId=${taskId}
            currentSoft=${null}
            currentHard=${null}
            used=${0}
            onClose=${() => setEditing(false)}
            onUpdated=${(newSnap) => {
              setBudget({...newSnap, configured: true});
              setEditing(false);
            }} />
        ` : null}
      </div>
    `;
  }

  const used = budget.used_cost_usd || 0;
  const limit = budget.limit_cost_usd;
  const pct = budget.used_pct != null ? budget.used_pct : 0;
  const exhausted = !!budget.exhausted;
  const hardExhausted = !!budget.hard_exhausted;

  const fmtUSD = (v) => {
    if (v == null) return "—";
    const n = Number(v);
    if (!isFinite(n)) return "—";
    if (n >= 1000) return `$${n.toFixed(0)}`;
    if (n >= 1) return `$${n.toFixed(2)}`;
    return `$${n.toFixed(4)}`;
  };
  const fmtTok = (n) => {
    if (!n) return "0";
    if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
    if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
    return String(n);
  };

  // Color thresholds: green < 70%, yellow < 90%, orange < 100%, red at/past.
  let barCls = "budget-bar-fill";
  if (hardExhausted) barCls += " budget-hard";
  else if (exhausted || pct >= 100) barCls += " budget-exhausted";
  else if (pct >= 90) barCls += " budget-hot";
  else if (pct >= 70) barCls += " budget-warm";

  return html`
    <div class="budget-bar-wrap${exhausted ? " exhausted" : ""}">
      <div class="budget-bar-row">
        <span class="budget-label">Token budget</span>
        <span class="budget-numbers">
          <strong>${fmtUSD(used)}</strong>
          <span class="muted"> / ${fmtUSD(limit)}</span>
        </span>
        <span class="budget-pct">${pct.toFixed(2)}%</span>
        ${hardExhausted
          ? html`<span class="budget-tag budget-tag-hard">HARD EXHAUSTED</span>`
          : (exhausted
            ? html`<span class="budget-tag budget-tag-soft">BUDGET EXHAUSTED</span>`
            : null)}
        ${budget.hard_limit_cost_usd != null
          ? html`<span class="budget-hardlimit muted">hard cap ${fmtUSD(budget.hard_limit_cost_usd)}</span>`
          : null}
        <button class="btn ghost budget-edit-btn"
                onClick=${() => setEditing(true)}
                title="调整 token 预算上限">调整阈值</button>
      </div>
      <div class="budget-bar-track">
        <div class=${barCls} style=${{ width: `${Math.min(100, pct)}%` }}></div>
      </div>
      <div class="budget-bar-meta muted">
        ${budget.agent_count || 0} agents
        · ${fmtTok(budget.total_input_tokens || 0)} in
        · ${fmtTok(budget.total_output_tokens || 0)} out
        ${budget.total_cache_read_input_tokens
          ? html`· ${fmtTok(budget.total_cache_read_input_tokens)} cache-read`
          : null}
      </div>
      ${editing ? html`
        <${BudgetEditor}
          taskId=${taskId}
          currentSoft=${limit}
          currentHard=${budget.hard_limit_cost_usd}
          used=${used}
          onClose=${() => setEditing(false)}
          onUpdated=${(newSnap) => {
            setBudget(newSnap);
            setEditing(false);
          }} />
      ` : null}
    </div>
  `;
}

function BudgetEditor({ taskId, currentSoft, currentHard, used, onClose, onUpdated }) {
  // Local form state. Default to "bump to 2x current total" as a
  // sensible midpoint between "give a tiny bit more room" and "let it
  // run forever" — easy for the user to dial up/down from there.
  const suggestedSoft = (currentSoft != null)
    ? Math.max(currentSoft * 2, (used || 0) * 1.5)
    : Math.max((used || 0) * 1.5, 10.0);
  const [softVal, setSoftVal] = useState(
    String(Math.round(suggestedSoft * 100) / 100));
  const [hardVal, setHardVal] = useState(
    currentHard != null ? String(currentHard) : "");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState(null);

  const submit = async (e) => {
    e.preventDefault();
    setSaving(true);
    setError(null);
    const payload = {};
    const s = parseFloat(softVal);
    const h = hardVal.trim() === "" ? null : parseFloat(hardVal);
    if (!isFinite(s) || s <= 0) {
      setError("soft 上限必须是正数");
      setSaving(false);
      return;
    }
    payload.max_cost_usd = s;
    if (h !== null) {
      if (!isFinite(h) || h <= 0) {
        setError("hard 上限必须是正数（或留空清除）");
        setSaving(false);
        return;
      }
      payload.max_cost_usd_hard = h;
    } else {
      // Allow explicit clear by sending null.
      payload.max_cost_usd_hard = null;
    }
    try {
      const snap = await updateTokenBudget(taskId, payload);
      onUpdated(snap);
    } catch (e2) {
      setError(String(e2.message || e2));
    } finally {
      setSaving(false);
    }
  };

  return html`
    <div class="budget-editor-modal" onClick=${(e) => {
      if (e.target.classList.contains("budget-editor-modal")) onClose();
    }}>
      <div class="budget-editor-card">
        <h3>调整 token 预算</h3>
        <p class="muted">
          已使用 <strong>$${(used || 0).toFixed(4)}</strong>。
          上调上限后，运行中的编排器会在下一次轮询时自动 pick up；
          若任务已因预算超限被中止（aborted），需点击 Restart 恢复运行。
        </p>
        <form onSubmit=${submit}>
          <label>
            <span>Soft 上限（USD）</span>
            <input type="number" step="0.01" min="0" value=${softVal}
                   onInput=${(e) => setSoftVal(e.target.value)}
                   disabled=${saving} required />
            <small class="muted">超过此值 → 不再启动新 agent，等当前 agent 完成</small>
          </label>
          <label>
            <span>Hard 上限（USD，可空）</span>
            <input type="number" step="0.01" min="0" value=${hardVal}
                   placeholder="留空清除"
                   onInput=${(e) => setHardVal(e.target.value)}
                   disabled=${saving} />
            <small class="muted">超过此值 → 立即 SIGTERM 所有运行中的 agent</small>
          </label>
          ${error ? html`<div class="form-err">${error}</div>` : null}
          <div class="budget-editor-actions">
            <button type="button" class="btn ghost" onClick=${onClose}
                    disabled=${saving}>取消</button>
            <button type="submit" class="btn primary" disabled=${saving}>
              ${saving ? "保存中…" : "保存"}
            </button>
          </div>
        </form>
      </div>
    </div>
  `;
}
