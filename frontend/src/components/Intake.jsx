import { useEffect, useRef, useState } from "react";
import { fetchColumns, fetchSampleFile, fetchSamples } from "../api.js";
import { UploadIcon } from "./icons.jsx";

export const DESIGNS = [
  ["", "Infer it from the data"],
  ["experiment", "Randomized experiment"],
  ["did", "Difference-in-differences (panel)"],
  ["iv", "Instrumental variable"],
  ["rd", "Regression discontinuity (cutoff)"],
  ["observational", "Observational comparison"],
];

const ROLE_FIELDS = [
  ["treatment", "Treatment (the cause)"],
  ["outcome", "Outcome (the effect)"],
  ["unit", "Unit id"],
  ["time", "Time"],
  ["instrument", "Instrument"],
  ["running", "Running variable"],
  ["cluster", "Cluster"],
  ["strata", "Strata"],
  ["saturation", "Share treated per cluster"],
];

export const EMPTY_FORM = {
  file: null, claim: "", design: "", online: false,
  roles: { ...Object.fromEntries(ROLE_FIELDS.map(([k]) => [k, ""])), cutoff: "", randomized: false, post_treatment: [] },
};

export function modeOf(form) {
  const c = form.claim.trim();
  if (!form.file && !c) return null;
  if (!form.file) return "claim-only";
  return c ? "audit" : "explore";
}

const MODE_PILL = { audit: ["audit", "Audit"], explore: ["explore", "Explore"], "claim-only": ["audit", "Claim only"] };
const MODE_HINT = {
  audit: "We check the claim against the data: design, assumptions, estimate and verdict.",
  explore: "No claim, so we look for what may cause what. These are leads to study, not conclusions.",
  "claim-only": "No data, so we search published research on the claim and say what data would test it. No verdict.",
};
const ACCEPT = ".csv,.tsv,.txt,.xlsx,.xls,.json,.parquet";

export default function Intake({ form, setForm, running, error, onRun, onReset, llm, notify = null }) {
  const [samples, setSamples] = useState([]);
  const [dragOver, setDragOver] = useState(false);
  const [cols, setCols] = useState({ state: "idle", list: [], rows: null, error: "" });
  const inputRef = useRef(null);
  const mode = modeOf(form);
  const set = (patch) => setForm((f) => ({ ...f, ...patch }));
  const setRole = (k, v) => setForm((f) => ({ ...f, roles: { ...f.roles, [k]: v } }));

  useEffect(() => { fetchSamples().then(setSamples).catch(() => {}); }, []);
  useEffect(() => { if (!form.file && inputRef.current) inputRef.current.value = ""; }, [form.file]);

  // list the file's columns for the role pickers
  useEffect(() => {
    if (!form.file) { setCols({ state: "idle", list: [], rows: null, error: "" }); return; }
    let live = true;
    setCols((c) => ({ ...c, state: "loading", error: "" }));
    fetchColumns(form.file)
      .then((r) => live && setCols({ state: "ok", list: r.columns, rows: r.n_rows, error: "" }))
      .catch((e) => live && setCols({ state: "error", list: [], rows: null, error: e.message }));
    return () => { live = false; };
  }, [form.file]);

  async function pickSample(s, e) {
    const dataOnly = e.shiftKey; // shift-click: the data only (Explore mode)
    try {
      const file = await fetchSampleFile(s);
      const roles = dataOnly ? {} : Object.fromEntries(Object.entries(s.roles || {}).map(([k, v]) => [k, typeof v === "number" ? String(v) : v]));
      setForm({ ...EMPTY_FORM, file, claim: dataOnly ? "" : s.claim || "", design: dataOnly ? "" : s.design || "",
        roles: { ...EMPTY_FORM.roles, ...roles } });
    } catch { /* the chip simply does nothing */ }
  }

  const onDrag = (over) => (e) => { e.preventDefault(); setDragOver(over); };
  const [pillCls, pillText] = MODE_PILL[mode] || ["", "—"];
  const colNames = cols.list.map((c) => c.name);
  const optionsFor = (value) => (value && !colNames.includes(value) ? [value, ...colNames] : colNames);
  const nSet = Object.entries(form.roles).filter(([k, v]) => (k === "post_treatment" ? v.length : v)).length + (form.design ? 1 : 0);
  const claimOnlyBlocked = mode === "claim-only" && llm === false;

  return (
    <section className="intake card" aria-labelledby="intake-title">
      <div className="intake-head">
        <div>
          <h1 id="intake-title">Verify a causal claim, or explore what causes what</h1>
          <p className="muted">
            Upload a dataset and/or state a claim. We never claim more than the data can show: anything that can't be
            tested is marked <i>assumed</i>, never a pass.
          </p>
        </div>
        <div className={`mode-pill ${pillCls}`} aria-live="polite">Mode: {pillText}</div>
      </div>

      <div className="intake-grid">
        <label className={`drop${dragOver ? " over" : ""}${form.file ? " has" : ""}`} tabIndex={0}
          onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); inputRef.current?.click(); } }}
          onDragEnter={onDrag(true)} onDragOver={onDrag(true)} onDragLeave={onDrag(false)}
          onDrop={(e) => { onDrag(false)(e); if (e.dataTransfer.files[0]) set({ file: e.dataTransfer.files[0] }); }}>
          <input ref={inputRef} type="file" accept={ACCEPT} hidden aria-label="Data file"
            onChange={(e) => e.target.files[0] && set({ file: e.target.files[0] })} />
          <UploadIcon />
          {form.file ? (
            <div className="drop-file">
              <b>{form.file.name}</b>
              <span className="muted small">
                {(form.file.size / 1024).toFixed(1)} KB
                {cols.state === "ok" && ` · ${cols.list.length} columns · ${cols.rows?.toLocaleString()} rows`}
                {cols.state === "loading" && " · reading columns…"}
              </span>
              <span className="muted small">Click to replace</span>
              {cols.state === "error" && <span className="err small">{cols.error}</span>}
            </div>
          ) : (
            <div className="drop-file">
              <span><b>Drop a data file</b> or click to browse</span>
              <span className="muted small">CSV, TSV, Excel, JSON, Parquet · up to 50 MB</span>
            </div>
          )}
        </label>

        <div className="claim-box">
          <label htmlFor="claim" className="lbl">Causal claim <span className="muted">(optional)</span></label>
          <textarea id="claim" rows={3} placeholder="e.g. Job training increases 1978 earnings by $1,800"
            value={form.claim} onChange={(e) => set({ claim: e.target.value })} />
          {mode && <p className="small muted mode-hint">{MODE_HINT[mode]}</p>}
          {claimOnlyBlocked && <p className="small err">A claim without data needs Claude, which isn't connected. Add a dataset.</p>}
        </div>
      </div>

      <details className="adv">
        <summary>Study design and roles <span className="muted">(optional{nSet ? `, ${nSet} set` : ", otherwise inferred from the data"})</span></summary>
        <div className="adv-grid">
          <label className="span2">
            Study design
            <select value={form.design} onChange={(e) => set({ design: e.target.value })}>
              {DESIGNS.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </label>
          {ROLE_FIELDS.map(([key, label]) => (
            <label key={key}>
              {label}
              <select value={form.roles[key]} onChange={(e) => setRole(key, e.target.value)} disabled={!form.file}>
                <option value="">{form.file ? "Work it out" : "Add a file first"}</option>
                {optionsFor(form.roles[key]).map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          ))}
          <label>
            Cutoff
            <input type="number" step="any" inputMode="decimal" placeholder="e.g. 50" value={form.roles.cutoff}
              onChange={(e) => setRole("cutoff", e.target.value)} />
          </label>
        </div>
        <label className="check-opt small">
          <input type="checkbox" checked={form.roles.randomized} onChange={(e) => setRole("randomized", e.target.checked)} />
          <span>Treatment was randomized <span className="muted">(we never call a study an experiment unless you say so)</span></span>
        </label>
        {colNames.length > 0 && (
          <fieldset className="post-set">
            <legend className="small"><b>Measured after treatment</b> <span className="muted">(never adjusted for)</span></legend>
            <div className="post-list">
              {colNames.map((c) => (
                <label key={c} className="check-opt small">
                  <input type="checkbox" checked={form.roles.post_treatment.includes(c)}
                    onChange={(e) => setRole("post_treatment", e.target.checked
                      ? [...form.roles.post_treatment, c] : form.roles.post_treatment.filter((x) => x !== c))} />
                  <span>{c}</span>
                </label>
              ))}
            </div>
          </fieldset>
        )}
      </details>

      <div className="samples">
        <span className="muted small">Try a sample:</span>
        <div className="chips">
          {samples.map((s) => (
            <button key={s.id} type="button" className={`chip${s.kind === "real" ? " real" : ""}`}
              title={`${s.description}${s.claim ? `\nClaim: ${s.claim}` : ""}\nShift-click for the data only`} onClick={(e) => pickSample(s, e)}>
              {s.title}
            </button>
          ))}
        </div>
      </div>

      <div className="actions">
        <button className="btn primary" disabled={!mode || running || claimOnlyBlocked} onClick={onRun}>
          {running ? <><span className="spinner on-accent" aria-hidden="true" />Running…</> : "Run analysis"}
        </button>
        <button className="btn ghost" onClick={onReset}>Clear</button>
        <label className="check-opt small" title="Sends only the treatment and outcome words to a search service">
          <input type="checkbox" checked={form.online} onChange={(e) => set({ online: e.target.checked })} />
          <span>Search for related reading online</span>
        </label>
        {notify && (
          <label className="check-opt small">
            <input type="checkbox" checked={notify.on} onChange={(e) => notify.set(e.target.checked)} />
            <span>Email me at {notify.email} when it's done</span>
          </label>
        )}
        {error && <span className="err" role="alert">{error}</span>}
      </div>
    </section>
  );
}
