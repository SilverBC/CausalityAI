import { Component, useEffect, useRef, useState } from "react";
import * as P from "./panes.jsx";
import { downloadHTML, downloadJSON } from "../report/export.js";
import { DESIGNS } from "./Intake.jsx";

/** Which tabs show, in a fixed order (06-results-tabs-ux.md §3). Pipeline is added last by <Results>. */
export function tabsFor(r) {
  if (r.mode === "claim-only")
    return [["overview", "Overview", P.ClaimOverviewPane], ["lit", "Literature", P.ClaimLiteraturePane],
      ["design", "Future Design", P.ClaimDesignPane]];
  const rep = r.report || {};
  const audit = rep.mode !== "explore";
  const tabs = [["overview", "Overview", P.OverviewPane]];
  // 2: the usability report always comes second
  if (rep.design?.grade || (rep.sections || []).some((s) => s.key !== "literature"))
    tabs.push(["report", "Data Usability Report", P.UsabilityPane]);
  // 3: the assumed DAG in audit mode; in explore mode Discovery takes its slot and draws report.dag itself if needed
  if (audit && rep.dag) tabs.push(["dag", "Causal DAG", P.DagPane]);
  if (!audit && rep.explore) tabs.push(["discovery", "Causal Discovery", P.DiscoveryPane]);
  // 4: in audit mode the tab never disappears; with no number it explains why
  if (audit || rep.estimate) tabs.push(["estimate", "Treatment Effect", P.EstimatePane]);
  if ((rep.sections || []).some((s) => s.key === "literature")) tabs.push(["lit", "Literature", P.LiteraturePane]);
  if (audit || rep.suggestion) tabs.push(["design", "Future Design", P.FutureDesignPane]);
  tabs.push(["code", "Python Code", P.CodePane]);
  if (rep.data && Object.keys(rep.data).length) tabs.push(["data", "Data", P.DataPane]);
  return tabs;
}

/** One broken tab shows an error card; the other tabs keep working. */
class PaneBoundary extends Component {
  state = { error: null };
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error) { console.error(error); }
  render() {
    if (this.state.error)
      return <div className="card err" role="alert">This tab could not be shown ({this.state.error.message}). The other tabs still work.</div>;
    return this.props.children;
  }
}

/** `autoScroll`: the run finished in this browser tab, so bring the results into view (not when opening a saved one).
    `runKey` changes only when the result itself is new (another job, or a re-run), not on every poll of the same job. */
export default function Results({ result, job, autoScroll = false, runKey }) {
  const tabs = [...tabsFor(result), ["pipeline", "Pipeline", P.PipelinePane]];
  const [active, setActive] = useState("overview");
  const listRef = useRef(null);
  const sectionRef = useRef(null);
  // back to Overview whenever a new result arrives: a new job, a saved analysis, or a re-run after answering questions
  useEffect(() => {
    setActive("overview");
    if (!autoScroll) return;
    // wait for the graph above to take its full height (it grows when the result arrives); scrolling in the same
    // frame gets cancelled by the browser's scroll anchoring
    const t = setTimeout(() => {
      const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      sectionRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
    }, 150);
    return () => clearTimeout(t);
  }, [runKey]); // eslint-disable-line react-hooks/exhaustive-deps
  // The selected tab is never half hidden in the scrolling tab strip. Scroll only the strip, sideways: scrollIntoView
  // would also scroll the page to bring the tab bar on screen.
  useEffect(() => {
    const list = listRef.current;
    const tab = list?.querySelector(`[data-tab="${active}"]`);
    if (!tab) return;
    const left = tab.offsetLeft - list.offsetLeft, right = left + tab.offsetWidth;
    if (left < list.scrollLeft) list.scrollLeft = Math.max(0, left - 8);
    else if (right > list.scrollLeft + list.clientWidth) list.scrollLeft = right - list.clientWidth + 8;
  }, [active, runKey]);

  const current = tabs.find(([k]) => k === active) || tabs[0];
  const Pane = current[2];

  function onKey(e) { // arrow keys move between tabs (WAI-ARIA tabs pattern)
    const i = tabs.findIndex(([k]) => k === current[0]);
    const to = { ArrowRight: i + 1, ArrowLeft: i - 1, Home: 0, End: tabs.length - 1 }[e.key];
    if (to == null) return;
    e.preventDefault();
    const k = tabs[(to + tabs.length) % tabs.length][0];
    setActive(k);
    listRef.current?.querySelector(`[data-tab="${k}"]`)?.focus();
  }

  return (
    <section className="results" aria-label="Results" ref={sectionRef}>
      <div className="results-bar">
        <div className="tabs" role="tablist" aria-label="Result sections" ref={listRef} onKeyDown={onKey}>
          {tabs.map(([k, label]) => (
            <button key={k} type="button" data-tab={k} id={`tab-${k}`} className="tab" role="tab" aria-selected={k === current[0]}
              aria-controls="result-pane" tabIndex={k === current[0] ? 0 : -1} onClick={() => setActive(k)}>
              {label}
            </button>
          ))}
        </div>
        <div className="export-btns">
          <button className="btn sm ghost" onClick={() => downloadJSON(result, job.title)} title="The full result as JSON">Download JSON</button>
          {result.report && (
            <button className="btn sm ghost" onClick={() => downloadHTML(result.report, job.title)} title="One HTML file that shows this report offline">
              Download report
            </button>
          )}
        </div>
      </div>
      <div className="pane" id="result-pane" role="tabpanel" aria-labelledby={`tab-${current[0]}`}>
        <PaneBoundary key={current[0]}>
          <Pane r={result} job={job} />
        </PaneBoundary>
      </div>
    </section>
  );
}

/** The analysis stopped to ask (which column the claim means, which design). The answers re-run it in place. */
export function NeedsInput({ job, onAnswer, busy }) {
  const qs = job.result?.report?.needs_input || [];
  const [answers, setAnswers] = useState(() => Object.fromEntries(qs.map((q, i) => [i, q.role === "population" ? "" : q.options?.[0] ?? ""])));
  const designName = Object.fromEntries(DESIGNS);
  function submit(e) {
    e.preventDefault();
    const confirm = {};
    let design = null;
    qs.forEach((q, i) => {
      if (q.kind === "choose_design") design = answers[i];
      else if (q.role && answers[i]) confirm[q.role] = answers[i];
    });
    onAnswer({ confirm, design });
  }
  return (
    <form className="card needs-input" onSubmit={submit}>
      <h2>We need your answer before we can go on</h2>
      <p className="small muted">The analysis stopped instead of guessing. Answer, and it runs again with your choices.</p>
      {qs.map((q, i) => (
        <label key={i} className="ni-q">
          <span className="ni-text">{q.question}</span>
          <select value={answers[i]} onChange={(e) => setAnswers({ ...answers, [i]: e.target.value })}>
            {q.role === "population" && <option value="">Analyse everyone</option>}
            {(q.options || []).map((o) => <option key={o} value={o}>{q.kind === "choose_design" ? designName[o] || o : o}</option>)}
          </select>
        </label>
      ))}
      <button className="btn primary" disabled={busy}>{busy ? "Running…" : "Use these answers and analyze again"}</button>
    </form>
  );
}
