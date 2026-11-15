/* Phone: upload a dataset (and an optional claim). Runs the same analysis as the desktop, then opens a new chat. */
import { useEffect, useState } from "react";
import { createChat, datasetsChanged, fetchJob, fetchSampleFile, fetchSamples, startAnalysis } from "../api.js";
import { useAuth } from "../auth.jsx";
import { EMPTY_FORM } from "../components/Intake.jsx";
import { go } from "../router.js";
import MobileBar from "./MobileBar.jsx";

export default function NewDataset() {
  const [form, setForm] = useState(EMPTY_FORM);
  const [samples, setSamples] = useState([]);
  const [job, setJob] = useState(null);
  const [jobId, setJobId] = useState(null);
  const [error, setError] = useState("");
  const { user, emailNotifications } = useAuth();
  const [notify, setNotify] = useState(true);

  useEffect(() => { fetchSamples().then(setSamples).catch(() => {}); }, []);

  useEffect(() => { // follow the analysis; when it is done, open a chat about it
    if (!jobId) return;
    let stop = false, t = null;
    const tick = async () => {
      try {
        const j = await fetchJob(jobId);
        if (stop) return;
        setJob(j);
        if (j.status === "done") {
          datasetsChanged();
          const c = await createChat(jobId);
          go(`#/m/c/${c.id}`);
          return;
        }
        if (j.status === "error" || j.status === "needs_input") {
          setError(j.status === "error" ? `The analysis failed: ${j.error}`
            : "The analysis needs an answer (which column the claim means, or which design). Open it in the desktop version to answer.");
          setJobId(null);
          return;
        }
      } catch { /* transient: try again */ }
      t = setTimeout(tick, 1000);
    };
    tick();
    return () => { stop = true; clearTimeout(t); };
  }, [jobId]);

  async function pickSample(s) {
    try {
      const file = await fetchSampleFile(s);
      const roles = Object.fromEntries(Object.entries(s.roles || {}).map(([k, v]) => [k, typeof v === "number" ? String(v) : v]));
      setForm({ ...EMPTY_FORM, file, claim: s.claim || "", design: s.design || "", roles: { ...EMPTY_FORM.roles, ...roles } });
    } catch { /* ignore */ }
  }

  async function run() {
    setError("");
    try {
      const { job_id } = await startAnalysis({ ...form, notify: !!user && emailNotifications && notify });
      setJobId(job_id);
    } catch (e) {
      setError(e.message);
    }
  }

  const steps = job?.steps || [];
  const done = steps.filter((s) => ["done", "error", "skipped"].includes(s.status)).length;
  const current = steps.find((s) => s.status === "running");
  const claim = form.claim.trim();

  return (
    <div className="m-screen">
      <MobileBar>
        <button className="icon-btn m-icon" onClick={() => go("#/m")} aria-label="Back to your datasets">‹</button>
        <div className="m-bar-title"><b>New dataset</b></div>
        <span className="m-icon" aria-hidden="true" />
      </MobileBar>
      <main className="m-scroll m-form" id="main">
        {jobId ? (
          <div className="m-progress" role="status">
            <p><b>Analyzing {form.file?.name || "your claim"}…</b></p>
            <div className="m-bar-track"><span style={{ width: `${steps.length ? Math.max(5, (100 * done) / steps.length) : 5}%` }} /></div>
            <p className="small muted">{current ? current.label : "Starting…"}</p>
            <p className="small muted">You can go back: the dataset appears in your list, and the chat opens here when it's ready.</p>
          </div>
        ) : (
          <>
            <label className="m-file">
              <input type="file" accept=".csv,.tsv,.txt,.xlsx,.xls,.json,.parquet"
                onChange={(e) => setForm((f) => ({ ...EMPTY_FORM, claim: f.claim, file: e.target.files[0] || null }))} />
              <span className="m-plus" aria-hidden="true">{form.file ? "✓" : "+"}</span>
              <span>
                <b>{form.file ? form.file.name : "Choose a data file"}</b>
                <span className="small muted">{form.file ? `${(form.file.size / 1024).toFixed(0)} KB · tap to change` : "CSV, TSV, Excel, JSON or Parquet"}</span>
              </span>
            </label>
            <label className="lbl" htmlFor="m-claim">Causal claim to audit <span className="muted small">(optional)</span></label>
            <textarea id="m-claim" rows={3} value={form.claim} onChange={(e) => setForm((f) => ({ ...f, claim: e.target.value }))}
              placeholder="e.g. Job training increases 1978 earnings by $1,800" />
            {samples.length > 0 && (
              <>
                <div className="small muted m-gap">Or try a sample</div>
                <div className="chips m-samples">
                  {samples.map((s) => <button key={s.id} type="button" className={`chip${s.kind === "real" ? " real" : ""}`} onClick={() => pickSample(s)}>{s.title}</button>)}
                </div>
              </>
            )}
            {user && emailNotifications && (
              <label className="check-opt small m-gap">
                <input type="checkbox" checked={notify} onChange={(e) => setNotify(e.target.checked)} />
                <span>Email me at {user.email} when it's done</span>
              </label>
            )}
            {error && <p className="err small" role="alert">{error}</p>}
            <button className="btn primary m-wide" disabled={!form.file && !claim} onClick={run}>
              {form.file ? (claim ? "Audit the claim" : "Explore the data") : "Check the claim"}
            </button>
          </>
        )}
      </main>
    </div>
  );
}
