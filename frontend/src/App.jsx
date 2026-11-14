import { useCallback, useEffect, useRef, useState } from "react";
import { answerQuestions, datasetsChanged, fetchHealth, fetchJob, startAnalysis } from "./api.js";
import Header from "./components/Header.jsx";
import Sidebar from "./components/Sidebar.jsx";
import Intake, { EMPTY_FORM } from "./components/Intake.jsx";
import InfluenceGraph from "./components/InfluenceGraph.jsx";
import Progress from "./components/Progress.jsx";
import Results, { NeedsInput } from "./components/Results.jsx";
import { go, useRoute } from "./router.js";

const POLL_MS = 800;
const SIDE_KEY = "causal-audit:sidebar";
const wide = () => window.matchMedia("(min-width: 1100px)").matches;

/** The analysis named in the URL (#/?job=<id>): loaded once, then polled while it runs. */
function useJob(jobId) {
  const [job, setJob] = useState(null);
  const [error, setError] = useState("");
  const [tick, setTick] = useState(0);
  useEffect(() => {
    setJob(null);
    setError("");
    if (!jobId) return;
    let stop = false, timer = null;
    const load = async () => {
      try {
        const j = await fetchJob(jobId);
        if (stop) return;
        setJob(j);
        if (j.status === "running") timer = setTimeout(load, POLL_MS);
        else datasetsChanged();
      } catch (e) {
        if (stop) return;
        if (e.status === 404) setError("This analysis is not on the server (it may have been deleted).");
        else { setError(e.message); timer = setTimeout(load, 3000); } // transient: keep trying
      }
    };
    load();
    return () => { stop = true; clearTimeout(timer); };
  }, [jobId, tick]);
  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { job, error, reload };
}

function useSidebar() {
  // open by default on wide screens (the choice is remembered there); closed on narrow ones, where it overlays
  const [open, setOpen] = useState(() => {
    if (!wide()) return false;
    try { const v = localStorage.getItem(SIDE_KEY); if (v) return v === "1"; } catch { /* storage blocked */ }
    return true;
  });
  const set = (v) => {
    setOpen(v);
    if (wide()) try { localStorage.setItem(SIDE_KEY, v ? "1" : "0"); } catch { /* ignore */ }
  };
  return [open, set];
}

export default function App() {
  const route = useRoute();
  const [health, setHealth] = useState(null);
  const [form, setForm] = useState(EMPTY_FORM);
  const [submitting, setSubmitting] = useState(false);
  const [runError, setRunError] = useState("");
  const [sideOpen, setSideOpen] = useSidebar();
  const jobId = route.page === "analysis" ? route.jobId : null;
  const { job, error: jobError, reload } = useJob(jobId);
  // after Run or an answer, bring the progress card into view (on a phone it is a screen below the form)
  const outputRef = useRef(null);
  const scrollToOutput = useRef(false);
  useEffect(() => {
    if (!scrollToOutput.current || !job || !outputRef.current) return;
    scrollToOutput.current = false;
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    outputRef.current.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "start" });
  }, [job]);

  useEffect(() => { fetchHealth().then(setHealth).catch(() => setHealth({ llm: false })); }, []);
  // refresh the Claude badge once an analysis finishes (a summary may have failed)
  useEffect(() => { if (job && job.status !== "running") fetchHealth().then(setHealth).catch(() => {}); }, [job?.status]); // eslint-disable-line react-hooks/exhaustive-deps
  // on narrow screens the sidebar overlays the page: close it after navigating
  useEffect(() => { if (!wide()) setSideOpen(false); }, [route.page, jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  async function run() {
    setRunError("");
    setSubmitting(true);
    try {
      const { job_id } = await startAnalysis(form);
      scrollToOutput.current = true;
      go(`#/?job=${job_id}`); // reload-safe, and the sidebar highlights it
      datasetsChanged();
    } catch (e) {
      setRunError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  async function answer(a) {
    setSubmitting(true);
    try {
      await answerQuestions(job.id, a);
      scrollToOutput.current = true;
      reload();
      datasetsChanged();
    } catch (e) {
      setRunError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  function reset() {
    setForm(EMPTY_FORM);
    setRunError("");
    go("#/");
  }

  const running = submitting || job?.status === "running";
  const result = job?.status === "done" ? job.result : null;

  let body;
  if (route.page !== "analysis") {
    body = (
      <section className="card placeholder">
        <h1>{route.page === "investigate" ? "Investigation" : "Phone app"}</h1>
        <p className="muted">This page arrives in a later stage of the rebuild.</p>
        <a className="btn" href="#/">← Back to analysis</a>
      </section>
    );
  } else {
    body = (
      <>
        <Intake form={form} setForm={setForm} running={running} error={runError} onRun={run} onReset={reset}
          llm={health ? !!health.llm : undefined} />
        <div className="output-anchor" ref={outputRef} aria-hidden="true" />
        {jobError && <div className="card err" role="alert">{jobError} <a href="#/">Start a new analysis</a></div>}
        {job?.status === "error" && (
          <div className="card banner-error" role="alert"><b>The analysis failed.</b> {job.error}</div>
        )}
        {job?.status === "needs_input" && <NeedsInput key={job.updated} job={job} onAnswer={answer} busy={submitting} />}
        {job && job.mode !== "claim-only" && <InfluenceGraph result={result} running={job.status === "running"} />}
        <div>
          {result ? <Results result={result} job={job} /> : job?.status === "running" ? <Progress job={job} title="Running the analysis" /> : null}
        </div>
      </>
    );
  }

  return (
    <>
      <Header health={health} route={route} sideOpen={sideOpen} onToggleSidebar={() => setSideOpen(!sideOpen)} />
      <div className={`shell${sideOpen ? " side-open" : ""}`}>
        {sideOpen && (
          <>
            <div className="side-backdrop" onClick={() => setSideOpen(false)} aria-hidden="true" />
            <Sidebar currentJobId={jobId} onClose={() => setSideOpen(false)} onNew={() => { reset(); if (!wide()) setSideOpen(false); }} />
          </>
        )}
        <div className="shell-content">
          <main id="main">{body}</main>
          <footer className="foot muted small">
            Analysis in Python (pandas · statsmodels · scikit-learn · DoWhy · EconML · causal-learn · LiNGAM). Claude writes the
            summary when it's connected; the analysis never depends on it.
          </footer>
        </div>
      </div>
    </>
  );
}
