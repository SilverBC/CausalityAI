/* Queue literature investigations of several variables, or of every variable on the graph.
   The server runs 3 at a time and queues the rest (40 running + queued at most per analysis). */
import { useMemo, useState } from "react";
import { ROLE_LABEL } from "./graph/theme.js";

export const NOT_INVESTIGABLE = new Set(["latent", "fixed_effect", "group"]);
const PARALLEL = 3, LIMIT = 40, WARN_AT = 8;
const ROLE_ORDER = { treatment: 0, outcome: 1, confounder: 2 };

/** rounds = ceil(n / 3) -> roughly max(1, round(0.5·rounds)) to max(2, ceil(2·rounds)) minutes */
export function estimateMinutes(n) {
  const rounds = Math.ceil(n / PARALLEL);
  return [Math.max(1, Math.round(rounds * 0.5)), Math.max(2, Math.ceil(rounds * 2))];
}

function heavyWarning(n, room, all) {
  const k = Math.min(n, room);
  const [lo, hi] = estimateMinutes(k);
  // Temporary native confirm; stage 04 replaces it with the app's own dialog.
  return window.confirm(
    `${all ? `Investigate all ${n} variables?` : `Investigate ${n} variables?`}\n\n` +
    `This queues ${k} literature searches. Each is a Claude web search that takes 30 seconds to 2 minutes, and ${PARALLEL} run at a ` +
    `time, so expect roughly ${lo}–${hi} minutes in total.\n\nIt's heavy and uses a lot of your Claude usage. Every search adds ` +
    "hypothetical variables to the graph.\n" + (n > room ? `Only ${room} fit in the queue right now; the rest are skipped.\n` : "") +
    "\nYou can cancel queued searches at any time.");
}

export default function MultiInvestigate({ nodes, items, startMany, cancel, cancelQueued, onSelect, nameOf, error }) {
  const [sel, setSel] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const byNode = useMemo(() => Object.fromEntries(items.map((i) => [i.node, i])), [items]);
  const candidates = useMemo(() => (nodes || [])
    .filter((n) => !NOT_INVESTIGABLE.has(n.role))
    .sort((a, b) => (!!a.hypothetical - !!b.hypothetical) || ((ROLE_ORDER[a.role] ?? 9) - (ROLE_ORDER[b.role] ?? 9))
      || ((b.influence || 0) - (a.influence || 0))), [nodes]);
  const open = (n) => { const st = byNode[n.id]?.status; return !st || st === "error"; };
  const available = candidates.filter(open);
  const running = items.filter((i) => i.status === "running").length;
  const queued = items.filter((i) => i.status === "queued").length;
  const room = Math.max(0, LIMIT - running - queued);
  const chosen = available.filter((n) => sel.has(n.id));
  const toggle = (id) => setSel((s) => { const x = new Set(s); x.has(id) ? x.delete(id) : x.add(id); return x; });

  async function queue(list, all) {
    if (!list.length) return;
    if ((all || list.length >= WARN_AT) && !heavyWarning(list.length, room, all)) return;
    setBusy(true);
    await startMany(list.slice(0, room));
    setSel(new Set());
    setBusy(false);
  }

  return (
    <div className="multi">
      <h2>Investigate several variables</h2>
      <p className="small muted">
        Tick the variables to look up in published research. {PARALLEL} searches run at a time; the rest wait in a queue and
        start on their own, even if you leave this page.
      </p>
      {(running > 0 || queued > 0) && (
        <div className="multi-status small" role="status">
          <span><span className="spinner" aria-hidden="true" />{running} running{queued ? ` · ${queued} queued` : ""}</span>
          {queued > 0 && <button className="link-btn danger" onClick={cancelQueued}>Cancel queued</button>}
        </div>
      )}
      <div className="multi-tools small">
        <span className="muted">{available.length} of {candidates.length} not investigated yet</span>
        <span>
          <button className="link-btn" disabled={!available.length} onClick={() => setSel(new Set(available.map((n) => n.id)))}>Select all</button>
          {sel.size > 0 && <> · <button className="link-btn" onClick={() => setSel(new Set())}>Clear</button></>}
        </span>
      </div>
      <ul className="multi-list">
        {candidates.map((n) => {
          const inv = byNode[n.id], st = inv?.status, can = open(n);
          return (
            <li key={n.id} className={sel.has(n.id) && can ? "checked" : ""}>
              <label className={`multi-row${can ? "" : " off"}`}>
                <input type="checkbox" disabled={!can} checked={can && sel.has(n.id)} onChange={() => toggle(n.id)} />
                <span className="multi-name">
                  {nameOf(n.id, n.label)}
                  <span className="small muted"> · {n.hypothetical ? "hypothetical" : ROLE_LABEL[n.role] || n.role}</span>
                </span>
              </label>
              <span className="multi-state small">
                {st === "done" && <button className="link-btn ok" onClick={() => onSelect(n.id)} title="Show the findings">✓ {inv.result.factors.length} found</button>}
                {st === "running" && <span className="muted"><span className="spinner" aria-hidden="true" />{Math.round(inv.elapsed)}s</span>}
                {st === "queued" && (
                  <>
                    <span className="muted">#{inv.position} in queue</span>
                    <button className="row-del" onClick={() => cancel(n.id)} aria-label={`Cancel the queued investigation of ${nameOf(n.id, n.label)}`} title="Cancel">×</button>
                  </>
                )}
                {st === "error" && <span className="err" title={inv.error}>failed</span>}
              </span>
            </li>
          );
        })}
      </ul>
      {error && <p className="err small" role="alert">{error}</p>}
      <div className="multi-actions">
        <button className="btn primary" disabled={!chosen.length || busy || !room} onClick={() => queue(chosen, false)}>
          Investigate selected{chosen.length ? ` (${chosen.length})` : ""}
        </button>
        <button className="btn ghost" disabled={!available.length || busy || !room} onClick={() => queue(available, true)}
          title="Queues a search for every variable not investigated yet">
          Investigate all ({available.length})
        </button>
      </div>
      {!room && <p className="small muted">The queue is full ({LIMIT}). Wait for some to finish or cancel queued ones.</p>}
    </div>
  );
}
