/* Your saved analyses: title, grade, mode and age; a spinner while running; a link to each graph's Investigation page
   with counts, and for the open analysis its investigated variables. Refreshes while anything runs and whenever an
   analysis or investigation starts or finishes, and when the user signs in or out. Hovering an analysis swaps its
   grade for a delete button; every delete asks first. */
import { useEffect, useState } from "react";
import { DATASETS_EVENT, INVESTIGATIONS_EVENT, deleteAnalysis, deleteInvestigation, fetchDatasets, investigationsChanged } from "../api.js";
import { useOnAuthChange } from "../auth.jsx";
import { confirmAction, whatGoesWith } from "./ConfirmDialog.jsx";
import { TrashIcon } from "./icons.jsx";
import { MODE_LABEL, ago } from "../format.js";

const GRADE_CLASS = { A: "pass", B: "pass", C: "warn", D: "fail", E: "fail", F: "fail" };

function meta(d) {
  if (d.status === "running") return <><span className="spinner" aria-hidden="true" />Analyzing…</>;
  if (d.status === "error") return <span className="err-text">Failed</span>;
  if (d.status === "needs_input") return <span className="warn-text">Needs your answer</span>;
  return `${MODE_LABEL[d.mode] || d.mode} · ${ago(d.created)}`;
}

function InvestigationLinks({ d, active, route, onError }) {
  const invs = d.investigations || [];
  const done = invs.filter((i) => i.status === "done");
  const running = invs.filter((i) => i.status === "running").length;
  const queued = invs.filter((i) => i.status === "queued").length;
  const onPage = active && route?.page === "investigate";
  const focus = onPage ? route.query.get("node") : null;
  return (
    <>
      <a className="side-sub" href={`#/investigate/${d.id}`} aria-current={onPage && !focus ? "page" : undefined}>
        <span aria-hidden="true">⌖</span>
        <span className="side-sub-text">
          Investigation
          <span className="muted">
            {running ? ` · ${running} running` : ""}{queued ? ` · ${queued} queued` : ""}
            {!running && !queued && done.length ? ` · ${done.length} investigated` : ""}
          </span>
        </span>
        {running > 0 && <span className="spinner" aria-hidden="true" />}
      </a>
      {active && done.length > 0 && (
        <ul className="side-invs">
          {done.map((i) => (
            <li key={i.node}>
              <a href={`#/investigate/${d.id}?node=${encodeURIComponent(i.node)}`} title={`Causes of ${i.label}: ${i.found} found`}
                aria-current={focus === i.node ? "page" : undefined}>
                {i.label} <span className="muted">· {i.found}</span>
              </a>
              <button className="row-del" onClick={() => removeInvestigation(d, i, onError)} aria-label={`Delete the investigation of ${i.label}`}
                title="Delete investigation"><TrashIcon size={14} /></button>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

async function removeInvestigation(d, i, onError) {
  if (!(await confirmAction({ title: `Delete the investigation of “${i.label}”?`, confirmLabel: "Delete investigation",
    message: "Its hypothetical findings are removed from the graph. You can investigate this variable again later." }))) return;
  try {
    await deleteInvestigation(d.id, i.node);
    investigationsChanged(d.id);
  } catch (e) { onError(e.message); }
}

export default function Sidebar({ currentJobId, route, onNew, onClose, onDeleted }) {
  const [items, setItems] = useState(null);
  const [error, setError] = useState("");

  async function removeAnalysis(d) {
    const extra = whatGoesWith({ investigations: d.investigations?.length || 0, chats: d.chats?.length || 0 });
    if (!(await confirmAction({ title: `Delete “${d.title}”?`, confirmLabel: "Delete analysis",
      message: `${extra}This can't be undone.` }))) return;
    setError("");
    try {
      await deleteAnalysis(d.id);
      setItems((xs) => xs.filter((x) => x.id !== d.id));
      onDeleted?.(d.id);
    } catch (e) { setError(e.message); }
  }

  const load = () => fetchDatasets().then((x) => { setItems(x); setError(""); }).catch((e) => setError(e.message));
  useOnAuthChange(() => load());
  useEffect(() => {
    load();
    window.addEventListener(DATASETS_EVENT, load);
    window.addEventListener(INVESTIGATIONS_EVENT, load);
    return () => { window.removeEventListener(DATASETS_EVENT, load); window.removeEventListener(INVESTIGATIONS_EVENT, load); };
  }, []);
  const busy = items?.some((d) => d.status === "running" || d.investigations?.some((i) => i.status === "running" || i.status === "queued"));
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
              <a className="side-item" href={`#/?job=${d.id}`} aria-current={active && route?.page !== "investigate" ? "page" : undefined} title={d.claim || d.title}>
                <span className="side-title">{d.title}</span>
                {d.grade && <span className={`side-grade ${GRADE_CLASS[d.grade] || ""}`} title={`Design grade ${d.grade}`}>{d.grade}</span>}
              </a>
              {d.status !== "running" && (
                <button className="row-del side-del" onClick={() => removeAnalysis(d)} aria-label={`Delete ${d.title}`} title="Delete analysis">
                  <TrashIcon />
                </button>
              )}
              <div className="side-meta small muted">{meta(d)}</div>
              {d.status === "done" && d.has_graph && <InvestigationLinks d={d} active={active} route={route} onError={setError} />}
            </li>
          );
        })}
      </ul>
    </nav>
  );
}
