/* Your saved analyses: title, grade, mode and age; a spinner while running. Refreshes while anything runs and
   whenever an analysis starts or finishes. (Delete buttons and investigation links come in later stages.) */
import { useEffect, useState } from "react";
import { DATASETS_EVENT, fetchDatasets } from "../api.js";
import { MODE_LABEL, ago } from "../format.js";

const GRADE_CLASS = { A: "pass", B: "pass", C: "warn", D: "fail", E: "fail", F: "fail" };

function meta(d) {
  if (d.status === "running") return <><span className="spinner" aria-hidden="true" />Analyzing…</>;
  if (d.status === "error") return <span className="err-text">Failed</span>;
  if (d.status === "needs_input") return <span className="warn-text">Needs your answer</span>;
  return `${MODE_LABEL[d.mode] || d.mode} · ${ago(d.created)}`;
}

export default function Sidebar({ currentJobId, onNew, onClose }) {
  const [items, setItems] = useState(null);
  const [error, setError] = useState("");

  const load = () => fetchDatasets().then((x) => { setItems(x); setError(""); }).catch((e) => setError(e.message));
  useEffect(() => {
    load();
    window.addEventListener(DATASETS_EVENT, load);
    return () => window.removeEventListener(DATASETS_EVENT, load);
  }, []);
  const busy = items?.some((d) => d.status === "running");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, [busy]);

  return (
    <nav className="sidebar" id="sidebar" aria-label="Your analyses">
      <div className="side-head">
        <button className="btn primary sm side-new" onClick={onNew}>+ New analysis</button>
        <button className="icon-btn side-close" onClick={onClose} aria-label="Hide the sidebar" title="Hide sidebar">«</button>
      </div>
      <div className="side-label">Your analyses</div>
      {!items && !error && <p className="small muted side-pad">Loading…</p>}
      {error && <p className="err small side-pad" role="alert">{error}</p>}
      {items?.length === 0 && <p className="small muted side-pad">Nothing yet. Analyses you run are saved here.</p>}
      <ul className="side-list">
        {items?.map((d) => {
          const active = d.id === currentJobId;
          return (
            <li key={d.id} className={active ? "active" : ""}>
              <a className="side-item" href={`#/?job=${d.id}`} aria-current={active ? "page" : undefined} title={d.claim || d.title}>
                <span className="side-title">{d.title}</span>
                {d.grade && <span className={`side-grade ${GRADE_CLASS[d.grade] || ""}`} title={`Design grade ${d.grade}`}>{d.grade}</span>}
              </a>
              <div className="side-meta small muted">{meta(d)}</div>
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
