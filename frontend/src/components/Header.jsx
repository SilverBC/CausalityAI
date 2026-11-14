import { AccountButton } from "../auth.jsx";
import { ThemeToggle } from "../theme.jsx";
import { Logo, SidebarIcon } from "./icons.jsx";

/** Claude status from /api/health: the analysis itself never needs it, summaries and literature searches do. */
function ClaudeBadge({ health }) {
  let b = { cls: "muted", text: "Checking Claude…", title: "Checking which language model is available" };
  if (health) {
    if (!health.llm) b = { cls: "off", text: "Claude not connected", title: "No language model: summaries use a template. The analysis itself does not need one." };
    else if (health.llm_last_error) b = { cls: "warn", text: "Claude unavailable", title: `Claude is unavailable: ${health.llm_last_error}` };
    else b = { cls: "on", text: health.llm_backend === "cli" ? "Claude Code connected" : health.llm_backend === "fake" ? "Test mode (fake Claude)" : "Claude API connected", title: health.llm_label };
  }
  return (
    <span className={`badge ${b.cls}`} title={b.title} role="status">
      <i className="badge-dot" aria-hidden="true" />
      <span className="badge-text">{b.text}</span>
    </span>
  );
}

export default function Header({ health, route, graphJobId, sideOpen, onToggleSidebar }) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="icon-btn side-toggle" onClick={onToggleSidebar} aria-expanded={sideOpen} aria-controls="sidebar"
          aria-label={sideOpen ? "Hide your analyses" : "Show your analyses"} title={sideOpen ? "Hide sidebar" : "Your analyses"}>
          <SidebarIcon />
        </button>
        <a className="brand" href="#/" aria-label="Causal Audit, new analysis"><Logo /><span>Causal Audit</span></a>
        <nav className="nav-links" aria-label="Pages">
          <a href={route.page === "investigate" && route.jobId ? `#/?job=${route.jobId}` : "#/"}
            aria-current={route.page === "analysis" ? "page" : undefined}>Analysis</a>
          {graphJobId ? (
            <a href={`#/investigate/${graphJobId}`} aria-current={route.page === "investigate" ? "page" : undefined}>Investigation</a>
          ) : (
            <span className="disabled" aria-disabled="true" title="Open a finished analysis with a graph first">Investigation</span>
          )}
          <span className="disabled" aria-disabled="true" title="The phone app arrives in a later stage">Phone app</span>
        </nav>
      </div>
      <div className="topbar-right">
        <ClaudeBadge health={health} />
        <ThemeToggle />
        <AccountButton />
      </div>
    </header>
  );
}
