/* Phone: one chat with the Investigator. */
import { ChatThread, useChat } from "../chat/ChatThread.jsx";
import { GraphIcon } from "../components/icons.jsx";
import { go } from "../router.js";
import MobileBar from "./MobileBar.jsx";

export default function ChatView({ chatId }) {
  const c = useChat(chatId);
  const { chat, result, title, datasetId } = c;
  const graphHref = (node) => `#/m/g/${datasetId}${node ? `?node=${encodeURIComponent(node)}` : ""}`;
  return (
    <div className="m-screen">
      <MobileBar>
        <button className="icon-btn m-icon" onClick={() => go("#/m")} aria-label="Back to your datasets">‹</button>
        <div className="m-bar-title">
          <b>{chat?.title || "Chat"}</b>
          {title && <span className="small muted">{title}</span>}
        </div>
        {datasetId && result?.influence?.nodes?.length ? (
          <a className="icon-btn m-icon m-graph-btn" href={graphHref(null)} title="Investigate the causal graph" aria-label="Investigate the causal graph">
            <GraphIcon />
          </a>
        ) : <span className="m-icon" aria-hidden="true" />}
      </MobileBar>
      <ChatThread c={c} graphHref={graphHref} />
    </div>
  );
}
