import { AccountButton } from "../auth.jsx";
import { ThemeToggle } from "../theme.jsx";
import { Logo, SidebarIcon } from "./icons.jsx";

export default function Header({ route, graphJobId, sideOpen, onToggleSidebar }) {
  return (
    <header className="topbar">
      <div className="topbar-left">
        <button className="icon-btn side-toggle" onClick={onToggleSidebar} aria-expanded={sideOpen} aria-controls="sidebar"
          aria-label={sideOpen ? "Hide your analyses" : "Show your analyses"} title={sideOpen ? "Hide sidebar" : "Your analyses"}>
          <SidebarIcon />
        </button>
        <a className="brand" href="#/" aria-label="Bluff-or-Buff, new analysis"><Logo /><span>Bluff-or-Buff</span></a>
        <nav className="nav-links" aria-label="Pages">
          <a href={route.page === "investigate" && route.jobId ? `#/?job=${route.jobId}` : "#/"}
            aria-current={route.page === "analysis" ? "page" : undefined}>Analysis</a>
          {graphJobId ? (
            <a href={`#/investigate/${graphJobId}`} aria-current={route.page === "investigate" ? "page" : undefined}>Investigation</a>
          ) : (
            <span className="disabled" aria-disabled="true" title="Open a finished analysis with a graph first">Investigation</span>
          )}
          <a href="#/m" title="The chat-first phone version">Phone app</a>
        </nav>
      </div>
      <div className="topbar-right">
        <ThemeToggle />
        <AccountButton />
      </div>
    </header>
  );
}
