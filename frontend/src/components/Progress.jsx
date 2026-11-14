/* Live steps while an analysis runs; also the last results tab ("Pipeline"). */
import { useEffect, useState } from "react";
import { MODE_LABEL } from "../format.js";

const LABEL = { pending: "waiting", running: "running", done: "done", error: "failed", skipped: "skipped" };

/** Seconds since `since`, ticking every second while `on`. */
function useElapsed(since, on, fixed) {
  const [now, setNow] = useState(Date.now() / 1000);
  useEffect(() => {
    if (!on) return;
    const t = setInterval(() => setNow(Date.now() / 1000), 1000);
    return () => clearInterval(t);
  }, [on]);
  return on ? Math.max(0, now - since) : fixed;
}

export default function Progress({ job, title = "Pipeline" }) {
  const running = job.status === "running";
  const elapsed = useElapsed(job.created, running, job.elapsed);
  const steps = job.steps || [];
  return (
    <section className="card progress" aria-live="polite" aria-busy={running}>
      <div className="prog-head">
        <h2>{title}</h2>
        <span className="muted small">{MODE_LABEL[job.mode] || job.mode} · {Math.round(elapsed)}s{running ? " so far" : ""}</span>
      </div>
      {!steps.length && running && <p className="small muted"><span className="spinner" aria-hidden="true" />Starting…</p>}
      <ol className="steps">
        {steps.map((s) => (
          <li key={s.key} className={`step ${s.status}`}>
            <span className="dot" aria-hidden="true" />
            <span className="step-label">{s.label}</span>
            <span className="note">
              <span className="sr-only">{LABEL[s.status]}</span>
              {s.status === "running" ? <span className="spinner" aria-hidden="true" /> : s.seconds != null ? `${s.seconds}s` : s.status === "skipped" ? "skipped" : ""}
            </span>
            {s.note && <span className="step-note err small">{s.note}</span>}
          </li>
        ))}
      </ol>
    </section>
  );
}
