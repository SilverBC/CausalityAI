/* A chat with the Investigator about one analysis. Shared by the desktop bubble and the phone chat screen.
   useChat(chatId, {datasetId, onCreated}) holds the state; <ChatThread> draws the messages, cards and composer. */
import { useEffect, useRef, useState } from "react";
import { createChat, fetchChat, fetchJob, investigationsChanged, sendChatMessage } from "../api.js";
import { fmt } from "../format.js";
import { useInvestigations } from "../components/graph/hooks.js";
import { useNodeNames } from "../components/graph/names.js";

export const SUGGESTIONS = [
  "Does the data support the claim?",
  "What's the biggest problem with this data?",
  "Explain the estimate in plain words",
  "Investigate what else could drive the treatment",
];

// this repo's statuses: "assumed" is never shown as a pass
const STATUS = { tested_ok: ["pass", "tested: OK"], tested_concern: ["warn", "concern"], failed: ["fail", "failed"], assumed: ["na", "assumed"] };
const VERDICT = (label = "") => (label === "Supported" ? "pass" : label.startsWith("Partly") || label === "Inconclusive" ? "warn" : label === "Contradicted" ? "fail" : "na");

function EstimateCard({ rep }) {
  const e = rep.estimate;
  if (e.refused || e.estimate == null)
    return (
      <div className="m-card">
        <div className="m-card-kind">Estimate</div>
        <p className="small">No number was given: {e.refusal_reason || "the effect could not be estimated."}</p>
      </div>
    );
  const rel = e.baseline_mean ? e.estimate / Math.abs(e.baseline_mean) : null;
  return (
    <div className="m-card">
      <div className="m-card-kind">Estimated effect</div>
      <div className="m-est">{e.estimate > 0 ? "+" : ""}{fmt(e.estimate)}</div>
      <div className="small muted">{e.method}</div>
      {e.ci95 && <div className="small muted">95% confidence interval {fmt(e.ci95[0])} to {fmt(e.ci95[1])}</div>}
      {rel != null && isFinite(rel) && <div className="small muted">{rel > 0 ? "+" : ""}{(rel * 100).toFixed(1)}% of the comparison group's mean</div>}
      {rep.verdict && <div className={`status ${VERDICT(rep.verdict.label)} m-verdict`}>Claim: {rep.verdict.label}</div>}
    </div>
  );
}

function CheckCard({ s }) {
  const [cls, text] = STATUS[s.status] || ["na", "information"];
  return (
    <div className="m-card">
      <div className="m-card-head">
        <span className="m-card-kind">{s.title}</span>
        <span className={`status ${cls}`}>{text}</span>
      </div>
      <p className="small">{s.summary}</p>
    </div>
  );
}

function InvestigationCard({ inv, graphHref, nameOf }) {
  if (!inv) return null;
  const name = nameOf(inv.node, inv.label);
  return (
    <a className="m-card m-card-link" href={graphHref(inv.node)}>
      <div className="m-card-head">
        <span className="m-card-kind">Investigation · {name}</span>
        <span className="pill hypo">hypothetical</span>
      </div>
      {inv.status === "running" ? <p className="small"><span className="spinner" aria-hidden="true" />Searching published research… {Math.round(inv.elapsed)}s</p>
        : inv.status === "queued" ? <p className="small"><span className="queue-dot" aria-hidden="true" />Queued · #{inv.position} in line</p>
        : inv.status === "error" ? <p className="small err">{inv.error}</p>
        : <p className="small">Found {inv.result.factors.length} possible cause{inv.result.factors.length === 1 ? "" : "s"}
            {inv.result.factors.length ? `: ${inv.result.factors.map((f) => nameOf(f.id, f.label)).join(", ")}` : ""}.</p>}
      <span className="m-card-cta">Open in graph →</span>
    </a>
  );
}

function Cards({ cards, result, invs, graphHref, nameOf }) {
  if (!cards?.length || !result) return null;
  const rep = result.report || {};
  return (
    <div className="m-cards">
      {cards.map((c, i) => {
        if (c.kind === "estimate" && rep.estimate) return <EstimateCard key={i} rep={rep} />;
        const s = c.kind === "check" && (rep.sections || []).find((x) => x.key === c.ref);
        if (s) return <CheckCard key={i} s={s} />;
        if (c.kind === "graph")
          return (
            <a key={i} className="m-card m-card-link" href={graphHref(null)}>
              <div className="m-card-kind">Causal graph</div>
              <p className="small">See how the variables connect, and investigate any of them.</p>
              <span className="m-card-cta">Open the graph →</span>
            </a>
          );
        if (c.kind === "investigation")
          return <InvestigationCard key={i} inv={invs.find((x) => x.node === c.ref)} graphHref={graphHref} nameOf={nameOf} />;
        return null;
      })}
    </div>
  );
}

/** One chat. With chatId null and a datasetId, the first message creates the chat (then onCreated(id)), so opening
    the Investigator without writing anything leaves no empty chats behind. */
export function useChat(chatId, { datasetId: dsId = null, onCreated = null } = {}) {
  const [chat, setChat] = useState(null);
  const [result, setResult] = useState(null);
  const [title, setTitle] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(null); // the user's message while waiting for the reply
  const datasetId = chat?.dataset_id || dsId;
  const { items: invs, refresh: refreshInvs } = useInvestigations(datasetId);
  const { nameOf } = useNodeNames(datasetId);
  const chatRef = useRef(chat);
  chatRef.current = chat;

  useEffect(() => {
    setError("");
    if (!chatId) { setChat(dsId ? { id: null, dataset_id: dsId, title: "New chat", messages: [] } : null); return; }
    if (chatRef.current?.id === chatId) return; // just created here: already current
    setChat(null);
    fetchChat(chatId).then(setChat).catch((e) => setError(e.status === 404 ? "This chat no longer exists." : e.message));
  }, [chatId, dsId]);
  useEffect(() => {
    setResult(null);
    if (!datasetId) return;
    fetchJob(datasetId).then((j) => { setTitle(j.title); if (j.status === "done") setResult(j.result); }).catch(() => {});
  }, [datasetId]);

  async function send(msg) {
    msg = (msg || "").trim();
    if (!msg || pending || !chat) return false;
    setError("");
    setPending(msg);
    try {
      let id = chat.id;
      if (!id) {
        const c = await createChat(datasetId);
        id = c.id;
        chatRef.current = { ...chatRef.current, ...c };
        setChat((x) => ({ ...x, ...c }));
        onCreated?.(id);
      }
      const r = await sendChatMessage(id, msg);
      setChat((c) => ({ ...c, ...r.chat, messages: [...(c.messages || []), r.user, r.assistant] }));
      if (r.assistant.cards?.some((c) => c.kind === "investigation")) {
        refreshInvs();
        investigationsChanged(datasetId); // any open graph picks it up
      }
      return true;
    } catch (e) {
      setError(e.message); // e.g. "Claude is unavailable: <reason>"; nothing was stored, so it can be sent again
      return false;
    } finally {
      setPending(null);
    }
  }

  return { chat, result, title, error, pending, send, invs, nameOf, datasetId };
}

/** Messages + composer. `graphHref(nodeId | null)` builds links into the graph (desktop or phone). */
export function ChatThread({ c, graphHref, autoFocus = false }) {
  const { chat, result, error, pending, send, invs, nameOf } = c;
  const [text, setText] = useState("");
  const endRef = useRef(null);
  const messages = chat?.messages || [];
  useEffect(() => {
    const smooth = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    endRef.current?.scrollIntoView({ behavior: smooth ? "smooth" : "auto", block: "end" });
  }, [messages.length, pending]);

  async function submit(t) {
    const msg = (t ?? text).trim();
    if (!msg) return;
    setText("");
    if (!(await send(msg))) setText(msg); // failed: give the text back
  }

  return (
    <>
      <div className="m-scroll m-thread" aria-live="polite">
        {!chat && !error && <p className="muted small m-pad">Loading…</p>}
        {chat && messages.length === 0 && !pending && (
          <div className="m-empty">
            <p className="muted">Ask the Investigator anything about this analysis. Answers use the real numbers from your data.</p>
            <div className="m-suggest">
              {SUGGESTIONS.map((s) => <button key={s} type="button" className="chip" onClick={() => submit(s)}>{s}</button>)}
            </div>
          </div>
        )}
        {messages.map((m) => (
          <div key={m.id} className={`m-msg ${m.role}`}>
            <div className="m-bubble">{m.content}</div>
            {m.role === "assistant" && <Cards cards={m.cards} result={result} invs={invs} graphHref={graphHref} nameOf={nameOf} />}
          </div>
        ))}
        {pending && (
          <>
            <div className="m-msg user"><div className="m-bubble">{pending}</div></div>
            <div className="m-msg assistant"><div className="m-bubble m-typing" role="status" aria-label="The Investigator is typing"><i /><i /><i /></div></div>
          </>
        )}
        {error && <p className="err small m-pad" role="alert">{error}</p>}
        <div ref={endRef} />
      </div>
      <form className="m-composer" onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <textarea rows={1} value={text} placeholder="Ask about this analysis…" aria-label="Message" autoFocus={autoFocus}
          maxLength={4000} onChange={(e) => setText(e.target.value)} disabled={!chat}
          onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); } }} />
        <button className="m-send" disabled={!text.trim() || !!pending || !chat} aria-label="Send">↑</button>
      </form>
    </>
  );
}
