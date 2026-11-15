/* Phone: the causal graph full screen. Tapping a variable opens a bottom sheet with the same details and Investigate
   button as the desktop Investigation page; the canvas shrinks above the sheet and refits. */
import { useEffect, useMemo, useRef, useState } from "react";
import { fetchJob } from "../api.js";
import GraphCanvas from "../components/graph/GraphCanvas.jsx";
import { buildGraph } from "../components/graph/model.js";
import { useInvestigations } from "../components/graph/hooks.js";
import { useNodeNames } from "../components/graph/names.js";
import { useThemeColors } from "../components/graph/theme.js";
import MultiInvestigate from "../components/MultiInvestigate.jsx";
import { Disclaimer, Names, NodePanel } from "../pages/InvestigatePage.jsx";
import MobileBar from "./MobileBar.jsx";

function useViewportHeight() {
  const [h, setH] = useState(window.innerHeight);
  useEffect(() => {
    const on = () => setH(window.innerHeight);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return h;
}

const BAR = 56, TOOLS = 48; // header + toolbar heights

export default function GraphView({ datasetId, focus }) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [selectedId, setSelectedId] = useState(focus || null);
  const [showHypo, setShowHypo] = useState(true);
  const [info, setInfo] = useState(false);
  const [multi, setMulti] = useState(false);
  const colors = useThemeColors();
  const nodeNames = useNodeNames(datasetId);
  const { items, start, remove, startMany, cancel, cancelQueued, error: startError } = useInvestigations(datasetId);
  const slotsFull = items.filter((i) => i.status === "running").length >= 3;
  const vh = useViewportHeight();

  useEffect(() => {
    fetchJob(datasetId)
      .then((j) => (j.status === "done" ? setJob(j) : setError("The analysis has not finished yet.")))
      .catch((e) => setError(e.status === 404 ? "This dataset is not on the server (it may have been deleted)." : e.message));
  }, [datasetId]);
  useEffect(() => { if (focus) setSelectedId(focus); }, [focus]);

  const base = job?.result?.influence && !job.result.influence.error ? job.result.influence : null;
  const doneKey = items.filter((i) => i.status === "done").map((i) => i.id).join(",");
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const positions = useRef(new Map());
  const data = useMemo(() => {
    if (!base?.nodes?.length) return null;
    // no depth slider on the phone: every literature variable is shown
    const g = buildGraph(base, itemsRef.current, [...positions.current.values()], { hideHypothetical: !showHypo, maxDepth: null });
    g.nodes.forEach((n) => positions.current.set(n.id, n));
    return g;
  }, [base, doneKey, showHypo]); // eslint-disable-line react-hooks/exhaustive-deps

  const pendingIds = useMemo(() => new Set(items.filter((i) => i.status === "running").map((i) => i.node)), [items]);
  const queuedIds = useMemo(() => new Set(items.filter((i) => i.status === "queued").map((i) => i.node)), [items]);
  const selected = data?.nodes.find((n) => n.id === selectedId) || null;
  const selectedInv = selected ? items.find((i) => i.node === selected.id) : null;
  const hypoTotal = data?.nodes.filter((n) => n.hypothetical).length || 0;
  const sheet = !!selected || multi;
  const canvasH = sheet ? Math.max(200, Math.round(vh * 0.4)) : Math.max(320, vh - BAR - TOOLS);
  const select = (id) => { setSelectedId(id); if (id) setMulti(false); };
  const back = () => (window.history.length > 1 ? window.history.back() : (window.location.hash = "#/m"));

  return (
    <Names.Provider value={nodeNames}>
      <div className="m-screen m-graph">
        <MobileBar>
          <button className="icon-btn m-icon" onClick={back} aria-label="Back">‹</button>
          <div className="m-bar-title">
            <b>Investigate</b>
            <span className="small muted">{job?.title || ""}</span>
          </div>
          <button className="icon-btn m-icon" onClick={() => setInfo((v) => !v)} aria-label="About hypothetical findings" aria-expanded={info}>ⓘ</button>
        </MobileBar>
        {info && <div className="m-info" role="note"><Disclaimer /></div>}
        <div className="m-graph-tools">
          <label className="switch">
            <input type="checkbox" role="switch" checked={showHypo} disabled={!hypoTotal && showHypo} onChange={(e) => setShowHypo(e.target.checked)} />
            <span className="switch-track" aria-hidden="true"><span className="switch-thumb" /></span>
            <span className="small">Hypothetical{hypoTotal ? ` (${hypoTotal})` : ""}</span>
          </label>
          <button className="btn sm ghost" onClick={() => { setSelectedId(null); setMulti(true); }}>Several…</button>
        </div>
        <div className="m-graph-canvas" style={{ height: canvasH }}>
          {error ? <p className="err small m-pad" role="alert">{error}</p> : (
            <GraphCanvas data={data} colors={colors} height={canvasH} selectedId={selectedId} onSelect={(n) => select(n ? n.id : null)}
              pendingIds={pendingIds} queuedIds={queuedIds} names={nodeNames.names}
              empty={job ? "No graph for this analysis." : "Loading…"} />
          )}
        </div>
        {sheet && (
          <div className="m-sheet" role="region" aria-label={selected ? `Details for ${nodeNames.nameOf(selected.id, selected.label)}` : "Investigate several variables"}>
            <div className="m-sheet-grip">
              <span aria-hidden="true" />
              <button className="icon-btn m-icon" onClick={() => { setSelectedId(null); setMulti(false); }} aria-label="Close">✕</button>
            </div>
            <div className="m-sheet-body">
              {selected ? (
                <NodePanel node={selected} inv={selectedInv} onInvestigate={start} onSelect={select} startError={startError}
                  onDelete={remove} onCancel={cancel} slotsFull={slotsFull} />
              ) : (
                <MultiInvestigate nodes={data?.nodes} items={items} startMany={startMany} cancel={cancel} cancelQueued={cancelQueued}
                  onSelect={select} nameOf={nodeNames.nameOf} error={startError} />
              )}
            </div>
          </div>
        )}
      </div>
    </Names.Provider>
  );
}
