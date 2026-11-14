/* The phone app (#/m…): chat-first. Home lists your analyses ("datasets") with their chats; each chat talks to the
   Investigator about one analysis; the graph screen is the Investigation page as a full-screen canvas + bottom sheet. */
import { useEffect, useState } from "react";
import { createChat, deleteAnalysis, deleteChat, fetchDatasets } from "../api.js";
import { AccountButton, GuestNudge, useOnAuthChange } from "../auth.jsx";
import { confirmAction, whatGoesWith } from "../components/ConfirmDialog.jsx";
import { Logo } from "../components/icons.jsx";
import { MODE_LABEL, ago } from "../format.js";
import { go } from "../router.js";
import { ThemeToggle } from "../theme.jsx";
import ChatView from "./ChatView.jsx";
import GraphView from "./GraphView.jsx";
import MobileBar from "./MobileBar.jsx";
import NewDataset from "./NewDataset.jsx";

const GRADE_CLASS = { A: "pass", B: "pass", C: "warn", D: "fail", E: "fail", F: "fail" };

function Home() {
  const [datasets, setDatasets] = useState(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(null);

  const load = () => fetchDatasets().then((d) => { setDatasets(d); setError(""); }).catch((e) => setError(e.message));
  useEffect(() => { load(); }, []);
  useOnAuthChange(() => load());
  const running = datasets?.some((d) => d.status === "running");
  useEffect(() => { // poll while anything is still being analyzed
    if (!running) return;
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, [running]);

  async function newChat(d) {
    setBusy(d.id);
    try {
      const c = await createChat(d.id);
      go(`#/m/c/${c.id}`);
    } catch (e) {
      setError(e.message);
      setBusy(null);
    }
  }

  async function removeDataset(d) {
    const extra = whatGoesWith({ investigations: d.investigations?.length || 0, chats: d.chats.length });
    if (!(await confirmAction({ title: `Delete “${d.title}”?`, confirmLabel: "Delete analysis", message: `${extra}This can't be undone.` }))) return;
    await deleteAnalysis(d.id).catch((e) => setError(e.message));
    load();
  }

  async function removeChat(c) {
    if (!(await confirmAction({ title: `Delete the chat “${c.title}”?`, confirmLabel: "Delete chat",
      message: "Its messages will be deleted. This can't be undone." }))) return;
    await deleteChat(c.id).catch((e) => setError(e.message));
    load();
  }

  return (
    <div className="m-screen">
      <MobileBar>
        <div className="m-brand"><Logo size={22} /><span>Causal Audit</span></div>
        <ThemeToggle />
        <AccountButton />
        <a className="icon-btn m-icon" href="#/" title="Desktop version" aria-label="Open the desktop version">⧉</a>
      </MobileBar>
      <main className="m-scroll" id="main">
        <button className="m-upload" onClick={() => go("#/m/new")}>
          <span className="m-plus" aria-hidden="true">+</span>
          <span>
            <b>Upload a dataset</b>
            <span className="small muted">A data file plus an optional causal claim to audit</span>
          </span>
        </button>
        <GuestNudge className="m-nudge" />
        {error && <p className="err small" role="alert">{error}</p>}
        {!datasets && !error && <p className="muted small m-pad">Loading…</p>}
        {datasets?.length === 0 && <p className="muted small m-pad">No datasets yet. Upload one, or try a sample, and start chatting about it.</p>}

        {datasets?.map((d) => (
          <section key={d.id} className="m-ds" aria-label={d.title}>
            <div className="m-ds-head">
              <div className="m-ds-title">
                <b>{d.title}</b>
                <span className="small muted">
                  {d.status === "running" ? <><span className="spinner" aria-hidden="true" />Analyzing…</>
                    : d.status === "error" ? <span className="err-text">Analysis failed</span>
                    : d.status === "needs_input" ? <span className="warn-text">Needs your answer (open the desktop version)</span>
                    : `${MODE_LABEL[d.mode] || d.mode} · ${ago(d.created)}`}
                </span>
              </div>
              {d.grade && <span className={`m-grade ${GRADE_CLASS[d.grade] || ""}`} title={`Design grade ${d.grade}`}>{d.grade}</span>}
            </div>
            {d.claim && <p className="small m-claim">“{d.claim}”</p>}
            {d.chats.length > 0 && (
              <ul className="m-chats">
                {d.chats.map((c) => (
                  <li key={c.id}>
                    <a href={`#/m/c/${c.id}`} className="m-chat-row">
                      <span className="m-chat-title">{c.title}</span>
                      <span className="small muted">{ago(c.updated)}</span>
                    </a>
                    <button className="row-del" onClick={() => removeChat(c)} aria-label={`Delete the chat ${c.title}`} title="Delete chat">✕</button>
                  </li>
                ))}
              </ul>
            )}
            <div className="m-ds-actions">
              <button className="btn sm primary" disabled={d.status !== "done" || busy === d.id} onClick={() => newChat(d)}>
                {busy === d.id ? "Opening…" : "+ New chat"}
              </button>
              {d.status === "done" && d.has_graph && (
                <a className="btn sm ghost" href={`#/m/g/${d.id}`}>
                  Investigate graph{d.investigations?.length ? ` · ${d.investigations.length}` : ""}
                </a>
              )}
              {d.status !== "running" && (
                <button className="btn sm ghost m-ds-del" onClick={() => removeDataset(d)} aria-label={`Delete ${d.title}`}>Delete</button>
              )}
            </div>
          </section>
        ))}
        <p className="small muted m-pad m-note">Guest work is kept in this browser only: clearing its site data loses it. Create an account to keep it.</p>
      </main>
    </div>
  );
}

export default function MobileApp({ route }) {
  useEffect(() => {
    document.documentElement.classList.add("m-root");
    return () => document.documentElement.classList.remove("m-root");
  }, []);
  if (route.view === "c" && route.id) return <ChatView chatId={route.id} />;
  if (route.view === "g" && route.id) return <GraphView datasetId={route.id} focus={route.query.get("node")} />;
  if (route.view === "new") return <NewDataset />;
  return <Home />;
}
