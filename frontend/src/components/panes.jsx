/* Result tabs. The report content is built by the original CausalityAI renderer (src/report/render.js): pure HTML
   builders that escape every value and never show "assumed" as a pass. These panes decide what goes on which tab. */
import { useMemo } from "react";
import R from "../report/render.js";
import { download } from "../format.js";
import Progress from "./Progress.jsx";

/** Report HTML from the renderer (already escaped there). */
export const Html = ({ html }) => (html ? <div className="rh" dangerouslySetInnerHTML={{ __html: html }} /> : null);

const checks = (rep) => (rep?.sections || []).filter((s) => s && s.key !== "literature");
const literature = (rep) => (rep?.sections || []).find((s) => s?.key === "literature");

export function SummaryCard({ r }) {
  const s = r.summary;
  if (!s?.text) return null;
  return (
    <section className="card">
      <div className="card-title-row">
        <h2>Executive summary</h2>
        <span className="tag" title={s.source === "llm" ? "Claude wrote it from the report's numbers" : "Built from the report's verdict and numbers"}>
          {s.source === "llm" ? "written by Claude from the report" : "from the report"}
        </span>
      </div>
      <p className="summary-text">{s.text}</p>
      {s.llm_error && <p className="small muted">Claude was unavailable for the summary: {s.llm_error}</p>}
    </section>
  );
}

// ------------------------------------------------------------------------------------------- data modes
export function OverviewPane({ r }) {
  const rep = r.report;
  const html = useMemo(() => (rep.mode === "explore"
    ? R.errorsBanner(rep) + R.exploreIntro(rep)
    : R.errorsBanner(rep) + R.verdictBand(rep) + R.subVerdicts(rep) + R.assumedPanel(rep) +
      '<div class="cards two">' + R.claimCard(rep) + R.designCard(rep) + "</div>"), [rep]);
  return <><SummaryCard r={r} /><Html html={html} /></>;
}

export function UsabilityPane({ r }) {
  const rep = r.report;
  const html = useMemo(() => {
    const d = rep.design;
    const grade = d?.grade
      ? `<section class="card"><header class="card-head"><h3>Design grade: ${R.esc(d.grade)}</h3>` +
        `<span class="tag">set by the worst problem found${d.kind === "observational" ? "; observational studies top out at C" : ""}</span>` +
        `</header>${R.list(d.grade_reasons)}</section>`
      : "";
    return grade + R.assumedPanel(rep) + checks(rep).map((s) => R.sectionCard(s)).join("");
  }, [rep]);
  return <Html html={html} />;
}

export function DagPane({ r }) {
  const html = useMemo(() => R.dagCard(r.report), [r.report]);
  return (
    <>
      <Html html={html} />
      <p className="small muted pane-note">
        This is the diagram the analysis <i>assumed</i> from the design and roles. The influence graph above weights the
        same arrows with standardized effects estimated from your data.
      </p>
    </>
  );
}

export function DiscoveryPane({ r }) {
  const html = useMemo(() => R.sectionCard(r.report.explore, "explore-card prominent"), [r.report]);
  return <Html html={html} />;
}

export function EstimatePane({ r }) {
  const html = useMemo(() => R.estimateCard(r.report), [r.report]);
  return <Html html={html} />;
}

export function LiteraturePane({ r, job }) {
  const s = literature(r.report);
  const html = useMemo(() => (s ? R.sectionCard(s) : ""), [s]);
  return (
    <>
      <Html html={html} />
      <section className="card">
        <h2>Go beyond the data</h2>
        <p className="small">
          The reading list never changes the grade. To find causes your dataset could not contain, search the literature for
          what drives any variable on the Investigation page. Findings appear as <b>hypothetical</b> variables with a
          Reliability Score.
        </p>
        {r.influence?.nodes?.length > 0 && <a className="btn sm primary" href={`#/investigate/${job.id}`}>Open the Investigation page →</a>}
      </section>
    </>
  );
}

export function FutureDesignPane({ r }) {
  const html = useMemo(() => R.sectionCard(r.report.suggestion, "suggestion-card"), [r.report]);
  return <Html html={html} />;
}

export function DataPane({ r }) {
  const html = useMemo(() => R.dataCard(r.report), [r.report]);
  return <Html html={html} />;
}

// ------------------------------------------------------------------------------------------- reproducible code
const py = (v) => (v == null ? "None" : JSON.stringify(v, null, 4)
  .replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None"));

export function pythonScript(job) {
  const p = job.params || {};
  const name = job.filename || "data.csv";
  const ext = name.split(".").pop().toLowerCase();
  const reader = { xlsx: "read_excel", xls: "read_excel", json: "read_json", parquet: "read_parquet" }[ext] || "read_csv";
  const sep = ext === "tsv" ? ', sep="\\t"' : "";
  return `"""Reproduce the Causal Audit analysis "${job.title}".

Runs the same core analysis (backend/causality) with the same inputs.
Put ${name} next to this script and run it from the project's backend/ folder:
    python reproduce_${job.id}.py
"""
import json

import pandas as pd

from causality import pipeline
from causality.schema import to_jsonable

df = pd.${reader}(${JSON.stringify(name)}${sep})

report = pipeline.run(
    df,
    claim_text=${py(job.claim || "")},
    design_decl=${py(p.design || null)},
    roles=${py(p.roles || {})},
    confirm=${py(p.confirm || {})},
    online_literature=${py(!!p.online_literature)},
)
r = to_jsonable(report)

if r["needs_input"]:
    for q in r["needs_input"]:
        print("Question:", q["question"])
else:
    if r.get("verdict"):
        print("Verdict:", r["verdict"]["label"], "-", r["verdict"]["sentence"])
    if r.get("design"):
        print("Design:", r["design"]["kind"], "| grade", r["design"]["grade"])
    e = r.get("estimate") or {}
    if e and not e.get("refused") and e.get("estimate") is not None:
        print(f"Estimate ({e['method']}): {e['estimate']:.4g}, 95% interval {e['ci95']}")
    for a in r["assumed_panel"]:
        print("Assumed, not tested:", a["name"])

with open("report_${job.id}.json", "w") as f:
    json.dump(r, f, indent=2)
`;
}

export function CodePane({ job }) {
  const code = useMemo(() => pythonScript(job), [job]);
  return (
    <section className="card">
      <div className="card-title-row">
        <div>
          <h2>Python code</h2>
          <p className="small muted" style={{ margin: 0 }}>Re-runs this exact analysis, with the same inputs, through the open-source core.</p>
        </div>
        <button className="btn sm" onClick={() => download(`reproduce_${job.id}.py`, code, "text/x-python")}>Download script</button>
      </div>
      <pre className="code"><code>{code}</code></pre>
    </section>
  );
}

export const PipelinePane = ({ job }) => <Progress job={job} />;

// ------------------------------------------------------------------------------------------- claim only
const DIRECTION = { supports: "pass", contradicts: "fail", mixed: "warn" };

export function ClaimOverviewPane({ r, job }) {
  const c = r.claim_only || {};
  return (
    <>
      <SummaryCard r={r} />
      <section className="card">
        <div className="disclaimer" role="note">
          <b>No data, so no verdict.</b> This is what published research reports about the claim, found by an AI web
          search. It may be incomplete or wrong. Upload a dataset to test the claim from data.
        </div>
        <dl className="dl">
          <dt>Claim</dt><dd>{job.claim}</dd>
          <dt>Cause</dt><dd>{c.treatment || "not stated"}</dd>
          <dt>Effect</dt><dd>{c.outcome || "not stated"}</dd>
          <dt>Population</dt><dd>{c.population || "not stated"}</dd>
        </dl>
      </section>
    </>
  );
}

export function ClaimLiteraturePane({ r }) {
  const ev = r.claim_only?.evidence || [];
  return (
    <section className="card">
      <h2>What the research says</h2>
      {!ev.length && <p className="muted small">No findings with verifiable sources came back.</p>}
      {ev.map((e, i) => (
        <div key={i} className="finding">
          <div className="card-title-row">
            <span>{e.finding}</span>
            <span className={`status ${DIRECTION[e.direction] || "na"}`}>{e.direction}</span>
          </div>
          <ul className="sources">
            {e.sources.map((s, j) => (
              <li key={j}>
                <a href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a>{s.year ? <span className="muted"> · {s.year}</span> : null}
              </li>
            ))}
          </ul>
        </div>
      ))}
    </section>
  );
}

export function ClaimDesignPane({ r }) {
  const c = r.claim_only || {};
  return (
    <div className="grid2">
      <section className="card">
        <h2>What would bias a naive comparison</h2>
        <ul className="list">{(c.confounders || []).map((x, i) => <li key={i}><b>{x.name}</b>: {x.why}</li>)}</ul>
      </section>
      <section className="card">
        <h2>Designs that could test it</h2>
        <ul className="list">{(c.designs || []).map((x, i) => <li key={i}><b>{x.design}</b>: {x.why}</li>)}</ul>
        <h2 className="h-gap">Data you would need</h2>
        <ul className="list">{(c.data_needed || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
      </section>
    </div>
  );
}
