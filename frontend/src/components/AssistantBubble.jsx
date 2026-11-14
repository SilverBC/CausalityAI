/* The Investigator on desktop: a round mascot button (bottom right) that opens a chat about the current analysis.
   Same chats as the phone app; card links go to the desktop Investigation page. */
import { useEffect, useState } from "react";
import { fetchDataset } from "../api.js";
import { ChatThread, useChat } from "../chat/ChatThread.jsx";
import ChatPicker from "./ChatPicker.jsx";
import { GatorIcon } from "./icons.jsx";

const LAST = (id) => `causal-audit:last-chat:${id}`;
const readLast = (id) => { try { return localStorage.getItem(LAST(id)); } catch { return null; } };
const writeLast = (id, chatId) => { try { chatId ? localStorage.setItem(LAST(id), chatId) : localStorage.removeItem(LAST(id)); } catch { /* ignore */ } };

function Panel({ jobId, onClose }) {
  const [chats, setChats] = useState([]);
  const [chatId, setChatId] = useState(() => readLast(jobId)); // the last chat of this analysis is remembered
  const [title, setTitle] = useState("");

  const loadChats = () => fetchDataset(jobId).then((d) => {
    setTitle(d.title);
    setChats(d.chats);
    setChatId((cur) => { if (cur && !d.chats.some((c) => c.id === cur)) { writeLast(jobId, null); return null; } return cur; });
  }).catch(() => {});
  useEffect(() => { loadChats(); }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  const pick = (id) => { setChatId(id || null); writeLast(jobId, id || null); };
  const c = useChat(chatId, { datasetId: jobId, onCreated: (id) => { pick(id); loadChats(); } });
  const graphHref = (node) => `#/investigate/${jobId}${node ? `?node=${encodeURIComponent(node)}` : ""}`;

  // the first message names the chat: refresh the picker's titles
  const n = c.chat?.messages?.length || 0;
  useEffect(() => { if (n === 2) loadChats(); }, [n]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="ab-panel" role="dialog" aria-label="Chat with the Investigator">
      <header className="ab-head">
        <span className="ab-avatar" aria-hidden="true"><GatorIcon size={30} /></span>
        <div className="ab-title">
          <b>Ask the Investigator</b>
          <span className="small muted">{title}</span>
        </div>
        <button className="icon-btn" onClick={onClose} aria-label="Close the Investigator">✕</button>
      </header>
      <div className="ab-chats">
        <ChatPicker chats={chats} value={chatId} onChange={pick} />
        <button className="btn sm ghost" onClick={() => pick(null)} disabled={!chatId} title="Start a new chat">+ New</button>
      </div>
      <ChatThread c={c} graphHref={graphHref} autoFocus />
    </div>
  );
}

// ---- the teaser speech bubble
const TEASERS = [
  "Ready to sleuth?", "Got a hunch? Let's test it.", "Ask me what might be confounding this.", "Something fishy in the data?",
  "Let's follow the evidence.", "Want me to dig into the literature?", "Elementary. Ask away.", "Who's really behind this effect?",
];
const RETURN_LINES = ["Back on the case?", "Picking up where we left off?", "Fresh eyes on this one?", "Shall we reopen the case?"];
export const TEASER = { FIRST_MS: 2500, SHOW_MS: 5500, GAP_MS: 11000, MAX_SHOWS: 6 };
const OFF_KEY = "causal-audit:teaser-off"; // set for the browser session once the chat is opened or the teaser dismissed
const LAST_JOB_KEY = "causal-audit:teaser-last-job"; // the analysis the bubble last belonged to, this session
const isOff = () => { try { return sessionStorage.getItem(OFF_KEY) === "1"; } catch { return false; } };
const setOffKey = () => { try { sessionStorage.setItem(OFF_KEY, "1"); } catch { /* ignore */ } };
const pickOne = (xs) => xs[Math.floor(Math.random() * xs.length)];

function useTeaser(active, jobId) {
  const [line, setLine] = useState(null);
  const [off, setOff] = useState(isOff);

  // A different saved analysis opened (from the sidebar, a link, an email): one "welcome back" line, even when the
  // teaser is off. The last analysis is remembered for the browser session, so reloading the page or moving between
  // the same analysis's pages (Analysis / Investigation) does not count as "different".
  useEffect(() => {
    if (!jobId || !active) return;
    let last = null;
    try { last = sessionStorage.getItem(LAST_JOB_KEY); sessionStorage.setItem(LAST_JOB_KEY, jobId); } catch { /* ignore */ }
    if (!last || last === jobId) return;
    const t1 = setTimeout(() => setLine(pickOne(RETURN_LINES)), TEASER.FIRST_MS);
    const t2 = setTimeout(() => setLine(null), TEASER.FIRST_MS + TEASER.SHOW_MS);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [jobId]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { // the regular cycle: show a line, hide it, wait, show the next; at most MAX_SHOWS per page load
    if (!active) { setLine(null); return; }
    if (off) return;
    let shows = 0, idx = Math.floor(Math.random() * TEASERS.length), t;
    const show = () => {
      if (shows++ >= TEASER.MAX_SHOWS) return;
      setLine(TEASERS[idx++ % TEASERS.length]);
      t = setTimeout(hide, TEASER.SHOW_MS);
    };
    const hide = () => { setLine(null); t = setTimeout(show, TEASER.GAP_MS); };
    t = setTimeout(show, TEASER.FIRST_MS);
    return () => clearTimeout(t);
  }, [active, off]);

  const stop = () => { setOffKey(); setOff(true); setLine(null); };
  return { line, stop };
}

export default function AssistantBubble({ jobId }) {
  const [open, setOpen] = useState(false);
  const teaser = useTeaser(!!jobId && !open, jobId);
  const openChat = () => { teaser.stop(); setOpen(true); };
  useEffect(() => { if (!jobId) setOpen(false); }, [jobId]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e) => { if (e.key === "Escape" && !e.defaultPrevented) setOpen(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);
  if (!jobId) return null;
  return (
    <>
      {open && <Panel key={jobId} jobId={jobId} onClose={() => setOpen(false)} />}
      {teaser.line && (
        <div className="ab-teaser" key={teaser.line}>
          <button className="ab-teaser-text" onClick={openChat}>{teaser.line}</button>
          <button className="ab-teaser-x" onClick={teaser.stop} aria-label="Dismiss" title="Don't show again this session">✕</button>
        </div>
      )}
      <button className={`ab-bubble${open ? " open" : ""}${teaser.line ? " nudge" : ""}`} onClick={() => (open ? setOpen(false) : openChat())}
        aria-label={open ? "Close the Investigator" : "Ask the Investigator about this analysis"} aria-expanded={open}
        title={open ? "Close" : "Ask the Investigator"}>
        {open ? <span aria-hidden="true" className="ab-x">✕</span> : <GatorIcon size={54} />}
      </button>
    </>
  );
}
