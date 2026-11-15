/* Investigation page: the graph plus literature "sleuthing" on chosen variables. Findings are hypothetical. */
import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import { fetchJob } from "../api.js";
import { fmt } from "../format.js";
import GraphCanvas, { sourceLine } from "../components/graph/GraphCanvas.jsx";
import { buildGraph } from "../components/graph/model.js";
import { depthStatus, useDepthGraph, useInvestigations } from "../components/graph/hooks.js";
import { useNodeNames } from "../components/graph/names.js";
import { ROLE_LABEL, useThemeColors } from "../components/graph/theme.js";
import { GraphLegend } from "../components/InfluenceGraph.jsx";
import MultiInvestigate, { NOT_INVESTIGABLE } from "../components/MultiInvestigate.jsx";
import ReliabilityBreakdown from "../components/ReliabilityBreakdown.jsx";

// Display-only names. Everything sent to the backend keeps the original id and label.
export const Names = createContext({ names: {}, rename: () => {}, nameOf: (id, original) => original });

export function Disclaimer() {
  return (
    <div className="disclaimer" role="note">
      <b>Beyond your data — hypothetical.</b> Investigations search published research with an AI web search. The variables
      and links they add come from that research, <b>not from the dataset you provided</b>. They are leads to check, not
      measured effects, so their links carry no strengths, and summaries or sources may be incomplete or wrong. Hypothetical
      variables are sized by a <b>Reliability Score</b>: it says how established the research is, not whether the factor
      actually confounds your data.
    </div>
  );
}

function Sources({ sources }) {
  return (
    <ul className="sources">
      {sources.map((s, i) => (
        <li key={i}>
          <a href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a>
          {s.year ? <span className="muted"> · {s.year}</span> : null}
          {s.venue_kind && <div className={`muted src-meta${s.verified ? "" : " unverified"}`}>{sourceLine(s)}</div>}
        </li>
      ))}
    </ul>
  );
}

function Reliability({ score, factor }) {
  const [open, setOpen] = useState(false);
  const { nameOf } = useContext(Names);
  const label = factor ? nameOf(factor.id, factor.label) : "";
  const band = score >= 70 ? "high" : score >= 40 ? "mid" : "low";
  return (
    <div className="reliability" title="From the sources' citations, venue type and how many independent sources agree">
      <span className="small">Reliability Score</span>
      <span className="rel-bar" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={score} aria-label="Reliability Score">
        <span className={band} style={{ width: `${score}%` }} />
      </span>
      <b className="small">{score}/100</b>
      {factor && <button className="rel-info" onClick={() => setOpen(true)} aria-label={`How the Reliability Score of ${label} is calculated`} title="How is this calculated?">?</button>}
      {factor && <ReliabilityBreakdown factor={factor} label={label} open={open} onClose={() => setOpen(false)} />}
    </div>
  );
}

function Factor({ f, onSelect }) {
  const { nameOf } = useContext(Names);
  return (
    <div className="factor">
      <div className="factor-head">
        <button className="link-btn" onClick={() => onSelect(f.id)}>{nameOf(f.id, f.label)}</button>
        <span className={`pill ${f.in_data ? "in-data" : "hypo"}`}>{f.in_data ? "in your data" : "hypothetical"}</span>
      </div>
      {f.reliability != null && <Reliability score={f.reliability} factor={f} />}
      <p className="small">{f.explanation}</p>
      <div className="small muted">affects: {f.affects.join(", ")}</div>
      <Sources sources={f.sources} />
    </div>
  );
}

function NodeName({ node }) {
  const { names, rename } = useContext(Names);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const custom = names[node.id];
  const display = custom || node.label;
  useEffect(() => setEditing(false), [node.id]);
  const save = () => { rename(node.id, draft.trim() === node.label ? "" : draft); setEditing(false); };

  if (editing)
    return (
      <form className="name-edit" onSubmit={(e) => { e.preventDefault(); save(); }}>
        <input autoFocus value={draft} maxLength={60} aria-label={`Display name for ${node.label}`}
          onChange={(e) => setDraft(e.target.value)} onKeyDown={(e) => e.key === "Escape" && setEditing(false)} />
        <div className="row">
          <button type="submit" className="btn sm primary">Save</button>
          <button type="button" className="btn sm ghost" onClick={() => setEditing(false)}>Cancel</button>
        </div>
        <span className="small muted">Display only: kept in this browser, never sent to the server or to Claude.</span>
      </form>
    );
  return (
    <div className="name-view">
      <div className="row">
        <h2>{display}</h2>
        <button className="icon-btn sm" onClick={() => { setDraft(display); setEditing(true); }} aria-label={`Rename ${display}`} title="Rename (display only)">✎</button>
      </div>
      {custom && <div className="small muted">Original name: {node.label} · <button className="link-btn reset" onClick={() => rename(node.id, "")}>Reset</button></div>}
    </div>
  );
}

export function NodePanel({ node, inv, onInvestigate, onSelect, startError, onDelete, onCancel, slotsFull }) {
  const { nameOf } = useContext(Names);
  const display = nameOf(node.id, node.label);
  const can = !NOT_INVESTIGABLE.has(node.role);
  return (
    <div className="inv-node">
      <div className="card-title-row">
        <NodeName node={node} />
        <span className={`pill ${node.hypothetical ? "hypo" : node.observed === false ? "na" : "in-data"}`}>
          {node.hypothetical ? "hypothetical · not in your data" : node.observed === false ? "not measured" : "in your data"}
        </span>
      </div>
      <div className="small muted">{ROLE_LABEL[node.role] || node.role}</div>

      {node.hypothetical && (node.reliability != null
        ? <Reliability score={node.reliability} factor={node.relFactor} />
        : <div className="small muted">Reliability Score: not scored</div>)}

      {!node.literature && node.observed !== false && (
        <dl className="dl small">
          <dt>Influence</dt><dd>{fmt(node.influence)}</dd>
          {node.total_effect_on_outcome != null && <><dt>Total effect on the outcome</dt><dd>{fmt(node.total_effect_on_outcome)} SD</dd></>}
          {node.depth != null && <><dt>Steps from the outcome</dt><dd>{node.depth}</dd></>}
        </dl>
      )}

      {node.lit?.length > 0 && (
        <div className="lit-notes">
          {node.lit.map((n, i) => (
            <div key={i} className="lit-note">
              <div className="small muted">Suggested while investigating <b>{nameOf(n.fromId, n.from)}</b></div>
              <p className="small">{n.explanation}</p>
              <Sources sources={n.sources} />
            </div>
          ))}
        </div>
      )}

      <div className="inv-action">
        {!can ? (
          <p className="small muted">This node stands for something unmeasured or grouped, so there is nothing specific to look up.</p>
        ) : !inv || inv.status === "error" ? (
          <>
            <button className="btn primary" onClick={() => onInvestigate(node)}>
              {inv ? "Retry the investigation" : slotsFull ? `Queue an investigation of “${display}”` : `Investigate causes of “${display}”`}
            </button>
            <p className="small muted">
              Searches published research with Claude, usually 30 seconds to 2 minutes. Results are hypothetical and added to the graph.
              {slotsFull && " Three searches are already running, so this one waits in the queue."}
            </p>
            {inv?.status === "error" && <p className="err small" role="alert">{inv.error}</p>}
            {startError && <p className="err small" role="alert">{startError}</p>}
          </>
        ) : inv.status === "queued" ? (
          <div className="inv-done-row">
            <p className="small"><span className="queue-dot" aria-hidden="true" />Queued · #{inv.position} in line. It starts when a running search finishes.</p>
            <button className="link-btn danger small" onClick={() => onCancel(inv.node)}>Cancel</button>
          </div>
        ) : inv.status === "running" ? (
          <p className="small" role="status"><span className="spinner" aria-hidden="true" />Searching the literature… {Math.round(inv.elapsed)}s</p>
        ) : (
          <>
            <div className="inv-done-row">
              <p className="small muted">Investigated in {Math.round(inv.elapsed)}s.</p>
              <button className="link-btn danger small" onClick={() => onDelete(inv.node, display)}>Delete investigation</button>
            </div>
            {inv.result.summary && <p className="small">{inv.result.summary}</p>}
            {!inv.result.factors.length && <p className="small muted">No causes with a real source were found.</p>}
            {inv.result.factors.map((f) => <Factor key={f.id} f={f} onSelect={onSelect} />)}
          </>
        )}
      </div>
    </div>
  );
}

function InvestigationLog({ items, onSelect, onDelete, onCancel }) {
  const { nameOf } = useContext(Names);
  if (!items.length) return null;
  return (
    <div className="inv-log">
      <h3 className="small muted">Investigations</h3>
      {items.map((i) => {
        const name = nameOf(i.node, i.label);
        return (
          <div key={i.id} className="inv-log-item">
            <button className="inv-log-row" onClick={() => onSelect(i.node)}>
              <span className={`dot ${i.status}`} aria-hidden="true" />
              <span className="inv-log-name">{name}</span>
              <span className="small muted">
                {i.status === "running" ? `${Math.round(i.elapsed)}s…` : i.status === "queued" ? `#${i.position} in queue` : i.status === "error" ? "failed" : `${i.result.factors.length} found`}
              </span>
            </button>
            {i.status === "queued" ? (
              <button className="row-del" onClick={() => onCancel(i.node)} aria-label={`Cancel the queued investigation of ${name}`} title="Cancel">×</button>
            ) : i.status !== "running" && (
              <button className="row-del" onClick={() => onDelete(i.node, name)} aria-label={`Delete the investigation of ${name}`} title="Delete investigation">×</button>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function InvestigatePage({ jobId, focus = null }) {
  const [job, setJob] = useState(null);
  const [loadError, setLoadError] = useState("");
  useEffect(() => {
    setJob(null);
    setLoadError("");
    if (!jobId) return;
    fetchJob(jobId)
      .then((j) => (j.status === "done" ? setJob(j) : setLoadError("That analysis has not finished yet.")))
      .catch((e) => setLoadError(e.status === 404 ? "This analysis is not on the server (it may have been deleted)." : e.message));
  }, [jobId]);

  const result = job?.result;
  const base = result?.influence && !result.influence.error && result.influence.nodes?.length ? result.influence : null;
  const colors = useThemeColors();
  const { inf, depth, setDepth, defaultDepth, dataMax, loading, error } = useDepthGraph(base, jobId);
  const { items, start, remove, startMany, cancel, cancelQueued, error: startError } = useInvestigations(jobId);
  const slotsFull = items.filter((i) => i.status === "running").length >= 3;
  const [selectedId, setSelectedId] = useState(null);
  const [showHypo, setShowHypo] = useState(true);
  const nodeNames = useNodeNames(jobId);
  useEffect(() => setSelectedId(null), [jobId]);

  // rebuild only when the data graph, the finished investigations or the toggles change (not on every poll)
  const doneKey = items.filter((i) => i.status === "done").map((i) => i.id).join(",");
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const positions = useRef(new Map()); // id -> last drawn node, so hidden nodes come back where they were
  const data = useMemo(() => {
    if (!inf?.nodes?.length) return null;
    const g = buildGraph(inf, itemsRef.current, [...positions.current.values()], { hideHypothetical: !showHypo, maxDepth: depth });
    g.nodes.forEach((n) => positions.current.set(n.id, n));
    return g;
  }, [inf, doneKey, showHypo, depth]); // eslint-disable-line react-hooks/exhaustive-deps

  // steps from the outcome for every literature variable, on the analysis's own graph (independent of the slider)
  const litDepth = useMemo(() => (base ? buildGraph(base, itemsRef.current).litDepth : new Map()), [base, doneKey]); // eslint-disable-line react-hooks/exhaustive-deps
  const maxLitDepth = Math.max(0, ...[...litDepth.values()].filter((d) => d != null));
  const sliderMax = Math.max(dataMax, maxLitDepth);
  const litBeyond = [...litDepth].filter(([id, d]) => d != null && d > depth && (showHypo || !id.startsWith("hyp:"))).length;

  // when an investigation finishes, raise the slider (never lower it) so none of its findings are hidden
  const seenDone = useRef(new Set());
  useEffect(() => {
    let need = 0;
    for (const i of items) {
      if (i.status !== "done" || seenDone.current.has(i.id)) continue;
      seenDone.current.add(i.id);
      for (const f of i.result.factors) need = Math.max(need, litDepth.get(f.id) ?? 0);
    }
    if (need > depth) setDepth(need);
  }, [doneKey, litDepth]); // eslint-disable-line react-hooks/exhaustive-deps

  const hypoTotal = useMemo(() => {
    const ids = new Set();
    items.forEach((i) => i.status === "done" && i.result.factors.forEach((f) => !f.in_data && ids.add(f.id)));
    return ids.size;
  }, [items]);

  function selectNode(id) {
    // a hidden variable picked from a list: show hypotheticals again and/or deepen the slider
    if (id && !showHypo && id.startsWith("hyp:")) setShowHypo(true);
    const d = id ? litDepth.get(id) : null;
    if (d != null && d > depth) setDepth(d);
    setSelectedId(id);
    // keep the address bar in step (shareable, and a link to this node elsewhere still navigates); no history entry
    const hash = `#/investigate/${jobId}${id ? `?node=${encodeURIComponent(id)}` : ""}`;
    if (window.location.hash !== hash) window.history.replaceState(null, "", hash);
  }

  // #/investigate/<id>?node=<node>: select that node once it is on the graph. Following the same link again (e.g.
  // from the sidebar after clicking elsewhere) selects it again, so the handled link is forgotten on every navigation.
  const focused = useRef(null);
  const [navTick, setNavTick] = useState(0);
  useEffect(() => {
    const on = () => { focused.current = null; setNavTick((t) => t + 1); };
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  useEffect(() => {
    if (!focus || focused.current === focus || !data) return;
    if (!data.nodes.some((n) => n.id === focus) && !litDepth.has(focus)) return;
    focused.current = focus;
    selectNode(focus);
  }, [focus, data, litDepth, navTick]); // eslint-disable-line react-hooks/exhaustive-deps

  const pendingIds = useMemo(() => new Set(items.filter((i) => i.status === "running").map((i) => i.node)), [items]);
  const queuedIds = useMemo(() => new Set(items.filter((i) => i.status === "queued").map((i) => i.node)), [items]);
  const selected = data?.nodes.find((n) => n.id === selectedId) || null;
  const selectedInv = selected ? items.find((i) => i.node === selected.id) : null;

  if (!jobId || loadError || !job) {
    return (
      <section className="card placeholder">
        <h1>Investigation</h1>
        <p className={loadError ? "err" : "muted"}>{loadError || (jobId ? "Loading the analysis…" : "Run an analysis first, then open the investigation from its graph.")}</p>
        <a className="btn" href="#/">← Back to analysis</a>
      </section>
    );
  }
  if (!base) {
    return (
      <section className="card placeholder">
        <h1>Investigation</h1>
        <p className="muted">This analysis has no graph to investigate{result?.mode === "claim-only" ? " (it has no dataset)" : ""}.</p>
        <a className="btn" href={`#/?job=${jobId}`}>← Back to the analysis</a>
      </section>
    );
  }

  return (
    <Names.Provider value={nodeNames}>
      <section className="card inv-head">
        <div className="card-title-row">
          <div>
            <h1>Investigation <span className="muted inv-title">· {job.title}</span></h1>
            <p className="muted">
              Pick a variable and search what published research says causes it. Confounders your data never measured show up
              as hypothetical variables.
            </p>
          </div>
          <a className="btn ghost" href={`#/?job=${jobId}`}>← Back to analysis</a>
        </div>
        <Disclaimer />
      </section>

      <div className="inv-layout">
        <section className="card graph-card" aria-label="Investigation graph">
          <div className="graph-toolbar">
            <label className="switch">
              <input type="checkbox" role="switch" checked={showHypo} disabled={hypoTotal === 0}
                onChange={(e) => setShowHypo(e.target.checked)} />
              <span className="switch-track" aria-hidden="true"><span className="switch-thumb" /></span>
              <span>Show hypothetical variables</span>
            </label>
            <span className="small muted">
              {hypoTotal === 0 ? "None yet. Investigate a variable to find some."
                : showHypo ? `${hypoTotal} from the literature, not in your data` : `${hypoTotal} hidden`}
            </span>
            <button className="btn sm ghost several-btn" onClick={() => selectNode(null)} aria-controls="inv-panel">Investigate several…</button>
          </div>

          {sliderMax > 1 && (
            <div className="depth-ctl">
              <label htmlFor="depth" className="depth-label">Depth <span className="depth-val">{depth}</span></label>
              <div className="depth-track">
                <input id="depth" type="range" min={1} max={sliderMax} step={1} value={depth}
                  onChange={(e) => setDepth(Number(e.target.value))} aria-describedby="depth-status" />
                <div className="depth-ticks" aria-hidden="true">
                  {Array.from({ length: sliderMax }, (_, i) => i + 1).map((d) => (
                    <span key={d} className={d === defaultDepth ? "is-default" : ""}>{d}{d === defaultDepth && <em>default</em>}</span>
                  ))}
                </div>
              </div>
              <p id="depth-status" className="small muted depth-status" aria-live="polite">
                {loading ? "Searching the data…" : error ? <span className="err">{error}</span> : (
                  <>
                    {depthStatus(inf, depth, defaultDepth, dataMax)}
                    {litBeyond > 0 && ` ${litBeyond} literature variable${litBeyond === 1 ? " is" : "s are"} beyond this depth.`}
                  </>
                )}
              </p>
            </div>
          )}

          <GraphCanvas data={data} colors={colors} height={620} selectedId={selectedId} onSelect={(n) => selectNode(n ? n.id : null)}
            pendingIds={pendingIds} queuedIds={queuedIds} names={nodeNames.names} empty="Loading…" />
          {data && <GraphLegend data={data} colors={colors} />}
        </section>

        <aside className="card inv-panel" id="inv-panel" aria-label={selected ? "Variable details" : "Investigate several variables"}>
          {selected ? (
            <>
              <button className="link-btn small back-several" onClick={() => selectNode(null)}>← Investigate several</button>
              <NodePanel node={selected} inv={selectedInv} onInvestigate={start} onSelect={selectNode} startError={startError}
                onDelete={remove} onCancel={cancel} slotsFull={slotsFull} />
              <InvestigationLog items={items} onSelect={selectNode} onDelete={remove} onCancel={cancel} />
            </>
          ) : (
            <>
              <p className="small muted">
                Click a variable on the graph to see what is known about it. A good first pick is the treatment: whatever drives
                who gets treated, and also moves the outcome, is a confounder.
              </p>
              <MultiInvestigate nodes={data?.nodes} items={items} startMany={startMany} cancel={cancel} cancelQueued={cancelQueued}
                onSelect={selectNode} nameOf={nodeNames.nameOf} error={startError} />
            </>
          )}
        </aside>
      </div>
    </Names.Provider>
  );
}
