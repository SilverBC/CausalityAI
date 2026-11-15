/* Result tabs. Layout per rebuild-plan/06-results-tabs-ux.md; look matched to the previous app's results tabs
   (rebuild-plan/screenshots-v1-results-tabs, Causality v1 panes/*.jsx and charts.jsx), on this repo's report.
   A few blocks still come from the original CausalityAI renderer (src/report/render.js: escaped HTML builders);
   the rest is React. Every card title inside a pane is an h3. "Assumed" is never shown as a pass. */
import { useMemo, useState } from "react";
import R from "../report/render.js";
import { download, fmt } from "../format.js";
import { Chart, DAG_LEGEND, Dag, Discovery, Forest, Legend, LovePlot, MirrorHist } from "./vcharts.jsx";

// ------------------------------------------------------------------------------------------- shared blocks
/** Report HTML from the renderer (already escaped there). */
export const Html = ({ html }) => (html ? <div className="rh" dangerouslySetInnerHTML={{ __html: html }} /> : null);

function Card({ title, sub = null, right = null, className = "", children }) {
  return (
    <section className={`card pane-card ${className}`}>
      {(title || right) && <div className="card-title-row"><h3>{title}{sub && <span className="h3-sub"> {sub}</span>}</h3>{right}</div>}
      {children}
    </section>
  );
}

// v1's pills: PASS / WARN / FAIL, plus ASSUMED (dashed, neutral: never green)
const STATUS = {
  tested_ok: ["pass", "Pass", "Tested: OK"], tested_concern: ["warn", "Warn", "Tested: concern"],
  failed: ["fail", "Fail", "Failed"], assumed: ["na assumed", "Assumed", "Assumed (not tested)"],
};
export function StatusPill({ status, text = null }) {
  const s = STATUS[status];
  if (!s) return null;
  return <span className={`status ${s[0]}`} title={s[2]}>{text || s[1]}</span>;
}

const fmtP = (p) => (p == null ? "–" : p < 0.001 ? "< 0.001" : fmt(p, 3));
const signed = (v) => (v == null ? "–" : `${v > 0 ? "+" : ""}${fmt(v)}`);
const ci = (c) => (Array.isArray(c) && c.length >= 2 ? `[${fmt(c[0])}, ${fmt(c[1])}]` : "–");
const pct = (v) => (v == null || !isFinite(v) ? "–" : `${Math.abs(v) >= 1000 ? fmt(v) : v.toFixed(1)}%`);
const checksOf = (rep) => (rep?.sections || []).filter((s) => s && s.key !== "literature");
const literatureOf = (rep) => (rep?.sections || []).find((s) => s?.key === "literature");
const label = (k) => { const t = String(k).replace(/_/g, " "); return t.charAt(0).toUpperCase() + t.slice(1); };
const DESIGN_NAME = { experiment: "Randomized experiment", did: "Difference-in-differences", iv: "Instrumental variable",
  rd: "Regression discontinuity", observational: "Observational comparison" };
const SCOPE = { full: "everyone in the data", treated: "the treated units", compliers: "units whose treatment followed the instrument",
  local_cutoff: "units close to the cutoff", overlap_trimmed: "units with comparable counterparts (extremes trimmed)" };
const TARGET = { ATE: "average effect over everyone analysed", ATT: "average effect on the treated",
  LATE: "average effect on compliers", local_at_cutoff: "effect at the cutoff", dose_response: "change per unit of treatment" };
const GRADE_CLASS = { A: "pass", B: "pass", C: "warn", D: "fail", E: "fail", F: "fail" };
const SHORT = { balance: "Balance", overlap: "Overlap", hidden_confounding: "Confounding", missing_outcomes: "Missing outcomes",
  interference: "Spillovers" };
const shortTitle = (s) => SHORT[s.key] || s.title.split(/\s+/).slice(0, 2).join(" ");
const rolesOf = (d) => Object.entries(d?.roles || {}).filter(([, v]) => v !== null && v !== false && !(Array.isArray(v) && !v.length));
const roleValue = (v) => (Array.isArray(v) ? v.join(", ") : v === true ? "yes" : String(v));

/** `backtick` spans in engine notes render as <code>. */
const WithCode = ({ text }) => String(text).split(/`([^`]+)`/).map((part, i) => (i % 2 ? <code key={i}>{part}</code> : part));

/** First sentence bold (v1 opens every check with a bold one-line finding), the rest as plain text. */
function Lead({ text }) {
  if (!text) return null;
  const m = String(text).match(/^(.+?[.!?])(\s+[A-Z(].*)?$/s);
  return <p className="lead">{m ? <><b>{m[1]}</b>{m[2] || ""}</> : <b>{text}</b>}</p>;
}

function KV({ rows }) {
  const shown = rows.filter((r) => r && r[1] !== undefined && r[1] !== null && r[1] !== "" && r[1] !== false);
  if (!shown.length) return null;
  return <dl className="dl">{shown.map(([k, v]) => [<dt key={`${k}-k`}>{k}</dt>, <dd key={`${k}-v`}>{v}</dd>])}</dl>;
}

/** The 76 px grade square; "why?" opens the reasons. */
export function GradeTile({ grade, reasons = [] }) {
  return (
    <details className={`grade-tile ${GRADE_CLASS[grade] || "na"}`}>
      <summary aria-label={`Design grade ${grade}. Why?`}>
        <span className="grade-letter" aria-hidden="true">{grade}</span>
        <span className="grade-why small">why?</span>
      </summary>
      {reasons.length > 0 ? <ul className="list">{reasons.map((r, i) => <li key={i}>{r}</li>)}</ul> : <p className="small muted">No reasons given.</p>}
    </details>
  );
}

export function SummaryCard({ r }) {
  const s = r.summary;
  if (!s?.text) return null;
  return (
    <Card title="Executive summary" right={<span className="small muted">{s.source === "llm" ? "written by Claude" : "auto-generated"}</span>}>
      <p className="summary-text">{s.text}</p>
      {s.llm_error && <p className="small muted">Claude was unavailable for the summary: {s.llm_error}</p>}
    </Card>
  );
}

// ------------------------------------------------------------------------------------------- Overview
const VERDICT_CLASS = (l) => {
  const s = String(l || "").toLowerCase();
  return s === "supported" ? "pass" : s.startsWith("partly") ? "warn" : s === "contradicted" ? "fail" : "na";
};

function VerdictChip({ v }) {
  if (!v?.label) return null;
  return (
    <div className="verdict-block">
      <span className={`verdict-chip ${VERDICT_CLASS(v.label)}`}>Claim {String(v.label).toLowerCase()}</span>
      {v.sentence && <p>{v.sentence}</p>}
      {v.caveat && <p className="small"><b>Only if</b> {v.caveat}.</p>}
      {(v.reasons || []).length > 0 && (
        <details className="why">
          <summary>Why this verdict</summary>
          <ul className="list">{v.reasons.map((x, i) => <li key={i}>{x}</li>)}</ul>
        </details>
      )}
    </div>
  );
}

function HeadlineEstimate({ rep }) {
  const e = rep.estimate;
  const roles = rep.data?.roles || {};
  const effectOf = (roles.treatment || roles.outcome) && (
    <p className="small muted effect-of">Effect of <b>{roles.treatment}</b> on <b>{roles.outcome}</b>
      {e?.target && <> · {e.target} ({TARGET[e.target] || e.target})</>}</p>
  );
  if (!e || e.refused || e.estimate == null)
    return (
      <Card title="No credible estimate" className="hero-estimate">
        {effectOf}
        <p>{e?.refusal_reason || "The effect could not be estimated from this data."}</p>
        <VerdictChip v={rep.verdict} />
      </Card>
    );
  const base = e.baseline_mean;
  const rel = base != null && Math.abs(base) > 1e-9;
  return (
    <section className="card pane-card hero-estimate">
      <div className="small muted">{e.method}</div>
      <div className="hero-number">
        <span className="hero-big">{signed(e.estimate)}</span>
        <span className="muted">95% CI {ci(e.ci95)} · p {fmtP(e.p_value)}</span>
      </div>
      {effectOf}
      <KV rows={[
        rel && ["Relative to baseline", <>{pct((100 * e.estimate) / base)}{e.ci95 && <span className="muted"> [{pct((100 * e.ci95[0]) / base)}, {pct((100 * e.ci95[1]) / base)}]</span>}</>],
        base != null && ["Baseline (comparison) mean", fmt(base)],
        e.outcome_sd && ["Standardized (÷ SD)", fmt(e.estimate / e.outcome_sd)],
        e.n && ["N", e.n.toLocaleString()],
      ]} />
      <VerdictChip v={rep.verdict} />
    </section>
  );
}

function UsabilityCard({ rep }) {
  const d = rep.design;
  if (!d?.grade) return null;
  return (
    <section className="card pane-card hero-usability">
      <div className="usability-row">
        <GradeTile grade={d.grade} reasons={d.grade_reasons || []} />
        <div>
          <div className="small muted">Data usability</div>
          <div className="grade-line">Grade {d.grade}</div>
          <div>{DESIGN_NAME[d.kind] || d.kind} <span className="muted">({d.declared ? "you said so" : "inferred from the data"})</span></div>
        </div>
      </div>
      <p className="usability-why">
        {GRADE_CLASS[d.grade] === "pass" ? `Usable for causal inference as a ${(DESIGN_NAME[d.kind] || d.kind).toLowerCase()}.`
          : GRADE_CLASS[d.grade] === "warn" ? `Usable with caution as a ${(DESIGN_NAME[d.kind] || d.kind).toLowerCase()}.`
            : "Not reliable for causal inference as it stands."}
        {(d.grade_reasons || [])[0] && <span className="small muted"> Worst problem: {d.grade_reasons[0]}</span>}
      </p>
      <div className="mini-pills">
        {checksOf(rep).filter((s) => STATUS[s.status]).map((s) => (
          <span key={s.key} className="mini-pill"><StatusPill status={s.status} />{shortTitle(s)}</span>
        ))}
      </div>
    </section>
  );
}

/** Assumptions nothing tested: kept on Overview (the result holds only if each is true), one line each. */
function AssumedCard({ rep }) {
  const items = (rep.assumed_panel || []).filter(Boolean);
  if (!items.length) return null;
  return (
    <section className="card pane-card assumed-card" aria-labelledby="assumed-h">
      <div className="card-title-row"><h3 id="assumed-h">Assumed, not tested</h3><StatusPill status="assumed" /></div>
      <p className="small muted">These cannot be checked from this data. The result holds only if each is true; none of them passed a test.</p>
      <ul className="assumed-lines">
        {items.map((a) => (
          <li key={a.key}>
            <details>
              <summary><b>{a.name}</b></summary>
              {a.detail && <p className="small">{a.detail}</p>}
              {a.source && <p className="small muted">from: {a.source}</p>}
            </details>
          </li>
        ))}
      </ul>
    </section>
  );
}

const notScored = (s) => s && (/^not[_ ]scored/i.test(String(s.label || "")) || (s.aspect === "size" && s.label !== "not_claimed"));

const SUB_CLASS = (l) => {
  const s = String(l || "").toLowerCase();
  return s === "supported" || s === "full" ? "pass" : s === "rejected" || s === "contradicted" ? "fail"
    : /concern|partly|partial|narrow|unclear|not_tested|inconclusive/.test(s) ? "warn" : "na";
};

/** v1's "Parsed claim" table, with this engine's per-part verdicts in a third column. */
function ParsedClaim({ rep }) {
  const c = rep.claim;
  if (!c) return null;
  const sub = Object.fromEntries((rep.verdict?.sub || []).map((s) => [s.aspect, s]));
  const col = (phrase, colName) => (phrase ? <>{phrase}{colName && <span className="muted"> → <code>{colName}</code></span>}</> : colName ? <code>{colName}</code> : "not stated");
  const rows = [
    ["Treatment", col(c.treatment_phrase, rep.data?.roles?.treatment || c.treatment_col)],
    ["Outcome", col(c.outcome_phrase, rep.data?.roles?.outcome || c.outcome_col)],
    (c.instrument_phrase || c.instrument_col) && ["Instrument", col(c.instrument_phrase, c.instrument_col)],
    (c.design_hint || c.design_hint_phrase) && ["Design hint (your wording)",
      <>{c.design_hint_phrase && `“${c.design_hint_phrase}”`}{c.design_hint_phrase && c.design_hint && " → "}{c.design_hint && (DESIGN_NAME[c.design_hint] || c.design_hint)}</>],
    ["Direction", c.negated ? `${c.direction || "no effect"} (negated)` : c.direction || "not stated", sub.direction],
    ["Magnitude", c.size != null ? `${c.bound ? `${c.bound.replace(/_/g, " ")} ` : ""}${fmt(c.size)} ${c.size_unit || ""}`.trim() : "not stated", sub.size],
    ["Functional form", c.shape || "not stated", sub.shape],
    ["Target population", c.population_phrase || "unspecified", sub.population],
    ["Who the estimate covers", SCOPE[rep.estimate?.scope] || "–", sub.scope],
    ["Data quality", "–", sub.data_quality],
  ];
  return (
    <Card title="Parsed claim" sub="(rule-based)">
      <p className="claim-quote">“{c.raw}”</p>
      <div className="tbl-wrap">
        <table className="tbl claim-tbl">
          <thead><tr><th>Part</th><th>The claim says</th><th>The data say</th></tr></thead>
          <tbody>
            {rows.filter(Boolean).map(([k, v, s]) => (
              <tr key={k}>
                <th scope="row">{k}</th>
                <td>{v}</td>
                <td>{s ? (notScored(s)
                  ? <><span className="sub-label na">{label(String(s.label).replace(/^not[_ ]scored\s*:?\s*/i, "") || "note")} <span className="muted">(not scored)</span></span>{s.detail && <div className="small muted">{s.detail}</div>}</>
                  : <><span className={`sub-label ${SUB_CLASS(s.label)}`}>{label(s.label)}</span>{s.detail && <div className="small muted">{s.detail}</div>}</>) : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {(c.notes || []).length > 0 && <ul className="list small">{c.notes.map((n, i) => <li key={i}>{n}</li>)}</ul>}
    </Card>
  );
}

export function OverviewPane({ r }) {
  const rep = r.report;
  const explore = rep.mode === "explore";
  const top = useMemo(() => R.errorsBanner(rep), [rep]);
  const intro = useMemo(() => (explore ? R.exploreIntro(rep) : ""), [rep, explore]);
  return (
    <>
      <Html html={top} />
      <SummaryCard r={r} />
      {explore ? (
        <>
          <Html html={intro} />
          {rep.design?.grade && <UsabilityCard rep={rep} />}
        </>
      ) : (
        <>
          <div className="hero"><HeadlineEstimate rep={rep} /><UsabilityCard rep={rep} /></div>
          <AssumedCard rep={rep} />
          <ParsedClaim rep={rep} />
        </>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------- Data Usability Report
const CHECK_ORDER = ["balance", "overlap", "hidden_confounding", "missing_outcomes", "interference"];

/** The charts of one check: the balance bars become v1's Love plot, the propensity histogram its mirrored version. */
function CheckCharts({ s }) {
  const m = s.metrics || {};
  const out = [];
  let rest = s.charts || [];
  if (s.key === "balance" && m.per_column && Object.keys(m.per_column).length) {
    const band = m.chance_band || 0.1;
    out.push(
      <Chart key="love">
        <LovePlot band={band} rows={Object.entries(m.per_column).map(([variable, smd]) => ({ variable, smd, p: m.per_column_p?.[variable] }))} />
        <Legend items={[["var(--pass)", `|difference| ≤ ${fmt(band, 2)} (within chance)`], ["var(--fail)", `|difference| > ${fmt(band, 2)}`]]} />
      </Chart>,
    );
    rest = rest.filter((c) => c.kind !== "bar" || !/difference/i.test(c.title || ""));
  }
  const hist = rest.find((c) => c.kind === "hist" && (c.data?.series || []).length === 2);
  if (s.key === "overlap" && hist) {
    const [t, c] = hist.data.series;
    out.push(
      <Chart key="mirror">
        <MirrorHist bins={hist.data.bins} treated={t.counts} control={c.counts} />
        <Legend items={[["var(--treat)", t.name.toLowerCase()], ["var(--ctrl)", c.name.toLowerCase()], [null, "x-axis: propensity score"]]} />
      </Chart>,
    );
    rest = rest.filter((x) => x !== hist);
  }
  const html = useMemo(() => R.chartsHtml(rest), [s]); // eslint-disable-line react-hooks/exhaustive-deps
  return <>{out}<Html html={html} /></>;
}

/** One check, v1 style: title + pill, a bold one-line finding, bullets, the chart; every number one click away. */
function CheckCard({ s }) {
  const metrics = s.metrics || {};
  const small = {}, tables = [];
  Object.entries(metrics).forEach(([k, v]) => {
    if (R.isRecordArray(v) && v.length > 1) tables.push([k, v]);
    else small[k] = v;
  });
  const numbers = useMemo(() => (Object.keys(small).length ? R.kvTable(small, 0) : "") + tables.map(([k, v]) =>
    `<h4 class="sub-h">${R.esc(R.human(k))}</h4>${R.recordTable(v, 60)}`).join(""), [s]); // eslint-disable-line react-hooks/exhaustive-deps
  const bullets = [
    ...(s.assumptions || []).map((a, i) => (
      <li key={`a${i}`}><StatusPill status={a.status} /> <b>{a.name}</b>{a.detail ? <>: {a.detail}</> : null}</li>
    )),
    ...(s.warnings || []).map((w, i) => <li key={`w${i}`} className="warn-li">{w}</li>),
  ];
  return (
    <section className="card pane-card check-card" id={`sec-${s.key}`}>
      <div className="card-title-row check-head">
        <h3>{s.title}</h3>
        <span className="check-tags">
          {s.caps_verdict && <span className="tag tag-cap" title="Because of this check, the verdict cannot be better than Inconclusive">Caps the verdict</span>}
          <StatusPill status={s.status} />
        </span>
      </div>
      <Lead text={s.summary} />
      {s.error && <div className="banner-error card-banner">{s.error}</div>}
      {bullets.length > 0 && <ul className="list bullets">{bullets}</ul>}
      <CheckCharts s={s} />
      {numbers && (
        <details className="metrics rh">
          <summary>Numbers</summary>
          <div dangerouslySetInnerHTML={{ __html: numbers }} />
        </details>
      )}
    </section>
  );
}

function DesignCard({ rep }) {
  const d = rep.design;
  if (!d) return null;
  const relies = [...(d.conflicts || []), ...(d.assumptions || [])];
  return (
    <Card title={`Study design: ${DESIGN_NAME[d.kind] || d.kind}`}>
      <p className="small muted">{d.declared ? "You declared this design." : "Inferred from the data."}
        {d.kind !== "experiment" && " We never call a study an experiment unless you say treatment was randomized."}</p>
      {(d.reasons || []).length > 0 && <ul className="list">{d.reasons.filter((x) => x !== "You declared this design.").map((x, i) => <li key={i}>{x}</li>)}</ul>}
      {relies.length > 0 && (
        <>
          <h4 className="sub-h">What this design relies on</h4>
          <ul className="list bullets">{relies.map((a, i) => <li key={i}><StatusPill status={a.status} /> <b>{a.name}</b>{a.detail ? <>: {a.detail}</> : null}</li>)}</ul>
        </>
      )}
    </Card>
  );
}

export function UsabilityPane({ r }) {
  const rep = r.report;
  const d = rep.design;
  const roles = rolesOf(rep.data);
  const checks = useMemo(() => [...checksOf(rep)].sort((a, b) => {
    const ia = CHECK_ORDER.indexOf(a.key), ib = CHECK_ORDER.indexOf(b.key);
    return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib); // fixed order, then any others in engine order (stable sort)
  }), [rep]);
  return (
    <>
      <div className="grid2">
        <DesignCard rep={rep} />
        <Card title="Variable roles" right={d?.grade ? <GradeTile grade={d.grade} reasons={d.grade_reasons || []} /> : null}>
          {roles.length ? <KV rows={roles.map(([k, v]) => [label(k).toLowerCase(), roleValue(v)])} />
            : <p className="small muted">No roles were set.</p>}
          {(rep.data?.adjustment_set || []).length > 0 && <KV rows={[["covariates", rep.data.adjustment_set.join(", ")]]} />}
          <h4 className="card-h">Preprocessing</h4>
          {rep.data?.notes?.length ? <ul className="list">{rep.data.notes.map((n, i) => <li key={i}><WithCode text={n} /></li>)}</ul>
            : <p className="small muted">Nothing to note.</p>}
        </Card>
      </div>
      <div className="grid2 checks-grid">
        {checks.map((s) => <CheckCard key={s.key} s={s} />)}
      </div>
    </>
  );
}

// ------------------------------------------------------------------------------------------- Causal DAG / Discovery
export function DagPane({ r }) {
  const rep = r.report;
  const dag = rep.dag?.data || {};
  const roles = rep.data?.roles || {};
  const adj = rep.data?.adjustment_set || [];
  const latent = (dag.nodes || []).some((n) => /hidden|latent|unobserved|unmeasured/.test(n.role || "")) || (dag.edges || []).some((e) => e.type === "bidirected");
  const est = rep.estimand || {};
  const kinds = new Set((dag.nodes || []).map((n) => n.role));
  return (
    <>
      <Card title={rep.dag?.title || "Assumed causal diagram"}>
        <p className="small muted">Built from the declared design and roles: measured columns on the left, the treatment in the middle, the
          outcome on the right. Every arrow is a hypothesis, not something the data proved; dashed purple nodes are not in the data.</p>
        <Chart><Dag nodes={dag.nodes} edges={dag.edges} /></Chart>
        <Legend items={DAG_LEGEND.filter(([, t], i) => i < 2 || (i === 2 && [...kinds].some((k) => /^(instrument|running)/.test(k)))
          || (i === 3 && [...kinds].some((k) => /post|mediator|collider/.test(k))) || (i === 4 && latent))} />
        {(dag.groups || []).filter((g) => g?.name && !["treatment", "outcome"].includes(g.name)).length > 0 && (
          <>
            <h4 className="sub-h">Groups in the diagram</h4>
            <ul className="list">
              {dag.groups.filter((g) => g?.name && !["treatment", "outcome"].includes(g.name)).map((g) => (
                <li key={g.name}><b>{g.name}</b>{(g.members || []).length > 0 && <span className="muted"> ({g.members.join(", ")})</span>}{g.note && <>: {g.note}</>}</li>
              ))}
            </ul>
          </>
        )}
      </Card>
      <div className="grid2">
        <Card title="Identification">
          <KV rows={[
            ["Back-door set", adj.length ? adj.join(", ") : "∅ (none)"],
            ["Instruments", roles.instrument || "none"],
            roles.running && ["Running variable", `${roles.running}${roles.cutoff != null ? ` (cutoff ${roles.cutoff})` : ""}`],
            ["Latent confounding", latent ? "assumed possible (dashed U)" : "not assumed"],
          ]} />
          {(dag.missing_edges || []).length > 0 && (
            <>
              <h4 className="sub-h">Arrows assumed absent</h4>
              <ul className="list">{dag.missing_edges.map((m, i) => <li key={i}>{m.from} → {m.to}: {m.label}</li>)}</ul>
            </>
          )}
        </Card>
        <Card title="Estimand">
          <pre className="code estimand">{[est.text, `target: ${est.target || "–"}`, `contrast: ${est.contrast || "–"}`,
            `population: ${est.population || "–"}`].filter(Boolean).join("\n")}</pre>
        </Card>
      </div>
    </>
  );
}

const SUPPORT = (p) => [p.fci, p.pc, p.lingam].filter((x) => x && !/no edge|none/i.test(String(x))).length;

export function DiscoveryPane({ r }) {
  const rep = r.report;
  const ex = rep.explore;
  const m = ex.metrics || {};
  const dagChart = (ex.charts || []).find((c) => c.kind === "dag");
  const otherCharts = (ex.charts || []).filter((c) => c.kind !== "dag");
  const others = useMemo(() => R.chartsHtml(otherCharts), [ex]); // eslint-disable-line react-hooks/exhaustive-deps
  const pairs = m.pairs || [];
  const support = Object.fromEntries(pairs.map((p) => [[p.a, p.b].sort().join("\u0000"), SUPPORT(p)]));
  const edges = (dagChart?.data?.edges || []).map((e) => ({ ...e, support: support[[e.from, e.to].sort().join("\u0000")] }));
  const hasLatent = edges.some((e) => e.type === "bidirected");
  return (
    <>
      <Card title="Discovered causal graph" right={ex.status ? <StatusPill status={ex.status} /> : null}>
        <p className="small muted">
          FCI (main graph) · PC · LiNGAM on {m.n_rows != null ? `${Number(m.n_rows).toLocaleString()} rows` : "all rows"}
          {m.alpha != null && `, α = ${m.alpha}`}{m.test && ` (${m.test} test)`}.
          {m.stability_rule && <> {m.stability_rule}.</>} Line weight = number of algorithms that found the edge; dashed purple = a
          hidden common cause (↔).
        </p>
        {ex.summary && <p>{ex.summary}</p>}
        {dagChart && <Chart><Discovery nodes={dagChart.data.nodes} edges={edges} /></Chart>}
        {dagChart && <Legend items={[["var(--ink2)", "arrow: direction found (o→ one end uncertain)"], ["var(--muted)", "line: direction unknown"],
          hasLatent && ["var(--latent)", "↔ hidden common cause"]]} />}
        {(ex.warnings || []).length > 0 && <ul className="list warn-list">{ex.warnings.map((w, i) => <li key={i}>{w}</li>)}</ul>}
        <Html html={others} />
      </Card>
      {pairs.length > 0 && (
        <Card title="Edges">
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>Edge</th><th>FCI</th><th>PC</th><th>LiNGAM</th><th className="num">Support</th><th className="num">Stability</th></tr></thead>
              <tbody>
                {pairs.map((p, i) => (
                  <tr key={i}>
                    <td><b>
                      {p.drawn ? `${p.drawn.from} ${p.drawn.type === "directed" ? "→" : p.drawn.type === "partial" ? "o→" : p.drawn.type === "bidirected" ? "↔" : "—"} ${p.drawn.to}` : `${p.a} · ${p.b}`}
                    </b>{p.drawn?.type === "bidirected" && <span className="tag">latent</span>}</td>
                    <td>{p.fci || "–"}</td><td>{p.pc || "–"}</td><td>{p.lingam || "–"}</td>
                    <td className="num">{SUPPORT(p)}/3</td>
                    <td className="num">{p.stability?.confidence != null ? `${Math.round(100 * p.stability.confidence)}%` : "–"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
      {pairs.length > 0 && (
        <Card title="Sanity checks on discovered pairs">
          <p className="small muted">Each pair is checked for a plain correlation, a correlation that survives holding every other variable fixed,
            and a direction hint from residual independence.</p>
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>Pair</th><th className="num">Correlation (p)</th><th className="num">Partial correlation (p)</th><th>Direction hint</th><th>Agreement</th><th>Result</th></tr></thead>
              <tbody>
                {pairs.map((p, i) => {
                  const c = Object.fromEntries((p.checks || []).map((x) => [x.name, x]));
                  const marg = c["marginal correlation"], part = c["partial correlation given all other variables"];
                  const hint = (p.checks || []).find((x) => x.prefers);
                  const tested = (p.checks || []).filter((x) => typeof x.ok === "boolean");
                  const ok = tested.length > 0 && tested.every((x) => x.ok);
                  return (
                    <tr key={i}>
                      <td><b>{p.a} · {p.b}</b></td>
                      <td className="num">{marg ? `${fmt(marg.value, 2)} (${fmtP(marg.p_value)})` : "–"}</td>
                      <td className="num">{part ? `${fmt(part.value, 2)} (${fmtP(part.p_value)})` : "–"}</td>
                      <td>{hint ? hint.prefers.replace("a", p.a).replace("b", p.b).replace("->", " → ") : "–"}</td>
                      <td className="small">{p.agreement || "–"}</td>
                      <td>{tested.length ? <StatusPill status={ok ? "tested_ok" : "failed"} /> : "–"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------- Treatment Effect
/** Anderson-Rubin confidence set: one [lo, hi] or several; a null end is unbounded. */
function arSet(v) {
  const parts = (v.length && Array.isArray(v[0]) ? v : v.length >= 2 ? [v] : []).filter((p) => Array.isArray(p) && p.length >= 2);
  if (!parts.length) return "empty (no value fits)";
  return parts.map((p) => (p[0] == null && p[1] == null ? "unbounded (any value fits)"
    : `${p[0] == null ? "(−∞" : `[${fmt(p[0])}`}, ${p[1] == null ? "∞)" : `${fmt(p[1])}]`}`)).join(" or ");
}

function critique(rep) {
  const out = [];
  if (rep.verdict?.caveat) out.push(`Only if ${rep.verdict.caveat}.`);
  (rep.estimate?.extra?.notes || []).forEach((n) => out.push(n));
  (rep.estimate?.comparisons || []).forEach((c) => c.note && out.push(`${c.method}: ${c.note}`));
  checksOf(rep).filter((s) => s.status === "failed" || s.status === "tested_concern").forEach((s) => out.push(`${s.title}: ${s.summary}`));
  return out;
}

export function EstimatePane({ r }) {
  const rep = r.report;
  const e = rep.estimate;
  if (!e || e.refused || e.estimate == null) {
    const why = [...(rep.verdict?.reasons || []), ...checksOf(rep).filter((s) => s.status === "failed" || s.status === "tested_concern")
      .map((s) => `${s.title}: ${s.summary}`)];
    return (
      <>
        <Card title="Not estimated" right={<StatusPill status="failed" text="Not estimated" />}>
          <div className="banner-warn card-banner"><b>We refused to give a number.</b> {e?.refusal_reason || "The effect could not be estimated from this data."}</div>
        </Card>
        <Card title="Critique of the data collection">
          {why.length ? <ul className="list">{why.map((w, i) => <li key={i}>{w}</li>)}</ul> : <p className="small muted">No further problems were recorded.</p>}
        </Card>
      </>
    );
  }
  const items = [{ name: e.method, estimate: e.estimate, ci: e.ci95, primary: true },
    ...(e.comparisons || []).filter((c) => c.estimate != null).map((c) => ({ name: c.method, estimate: c.estimate, ci: c.ci95 }))];
  const claimed = r.claimed_effect;
  const designCharts = (e.charts || []).filter((c) => c.kind !== "forest"); // the forest is drawn above, with the claim
  const extra = Object.entries(e.extra || {}).filter(([k, v]) => k !== "covariates" && k !== "notes" && !k.startsWith("ar_")
    && (typeof v !== "object" || v === null));
  const lv = e.extra?.ci_levels || {};
  const levels = ["90", "95", "98", "99"].map((L) => [L, lv[L] || (L === "90" ? e.ci90 : L === "95" ? e.ci95 : null)])
    .filter(([, p]) => Array.isArray(p) && p.length >= 2 && p[0] != null && p[1] != null);
  const ar = [["90", e.extra?.ar_ci90], ["95", e.extra?.ar_ci95], ["98", e.extra?.ar_ci98], ["99", e.extra?.ar_ci99]]
    .filter(([, v]) => Array.isArray(v));
  const notes = critique(rep);
  return (
    <>
      <Card title={e.method}>
        <p className="small muted">Estimand: {e.target} ({TARGET[e.target] || e.target}), for {SCOPE[e.scope] || e.scope}.</p>
        <Chart><Forest items={items} claim={claimed} /></Chart>
        <Legend items={[["var(--treat)", "main estimate"], items.length > 1 && ["var(--ctrl)", "other methods"],
          claimed != null && ["var(--warn)", `claimed effect (${fmt(claimed)})`]]} />
      </Card>
      <div className="grid2">
        <Card title="Diagnostics">
          <KV rows={[
            ["Estimate", <><b>{signed(e.estimate)}</b>{e.se != null && <span className="muted"> (SE {fmt(e.se)})</span>}</>],
            ["p-value", fmtP(e.p_value)], ["Rows used", e.n?.toLocaleString()],
            ...extra.map(([k, v]) => [label(k), typeof v === "number" ? fmt(v) : typeof v === "boolean" ? (v ? "yes" : "no") : String(v)]),
            ["Adjusted for", (e.extra?.covariates || []).join(", ") || null],
            ["Comparison-group mean", e.baseline_mean != null && fmt(e.baseline_mean)], ["Outcome SD", e.outcome_sd != null && fmt(e.outcome_sd)],
          ]} />
          {levels.length > 0 && (
            <>
              <h4 className="sub-h">Intervals at each confidence level</h4>
              <div className="tbl-wrap">
                <table className="tbl ci-tbl">
                  <thead><tr><th>Level</th><th className="num">Interval</th><th>Zero</th></tr></thead>
                  <tbody>{levels.map(([L, p]) => (
                    <tr key={L} className={L === "95" ? "ci-main" : ""}>
                      <td>{L}%</td><td className="num">{ci(p)}</td>
                      <td>{p[0] > 0 || p[1] < 0 ? <span className="zero ex">● excludes zero</span> : <span className="zero in">○ includes zero</span>}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
              <p className="small muted">The conclusion uses the 95% interval.</p>
            </>
          )}
          {ar.length > 0 && (
            <>
              <h4 className="sub-h">Valid even with a weak instrument (Anderson–Rubin)</h4>
              <KV rows={ar.map(([L, v]) => [`${L}%`, arSet(v)])} />
              {e.extra?.ar_method && <p className="small muted">{e.extra.ar_method}</p>}
            </>
          )}
        </Card>
        <Card title="Critique & caveats">
          {notes.length ? <ul className="list">{notes.map((n, i) => <li key={i}>{n}</li>)}</ul> : <p className="small muted">No major problems flagged for this estimate.</p>}
        </Card>
      </div>
      {designCharts.map((c, i) => (
        <Card key={i} title={c.title}><Html html={R.chartsHtml([{ ...c, title: "" }])} /></Card>
      ))}
      {(e.comparisons || []).length > 0 && (
        <Card title="Other ways of estimating it">
          <div className="tbl-wrap">
            <table className="tbl">
              <thead><tr><th>Method</th><th className="num">Estimate</th><th className="num">95% interval</th><th>Note</th></tr></thead>
              <tbody>
                <tr><td><b>{e.method}</b> (main)</td><td className="num">{signed(e.estimate)}</td><td className="num">{ci(e.ci95)}</td><td /></tr>
                {e.comparisons.map((c, i) => (
                  <tr key={i}><td>{c.method}</td><td className="num">{signed(c.estimate)}</td><td className="num">{ci(c.ci95)}</td><td className="small">{c.note || ""}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------- Literature
const refUrl = (ref) => ref.url || (ref.doi ? `https://doi.org/${ref.doi}` : null);

export function LiteraturePane({ r, job }) {
  const s = literatureOf(r.report);
  const m = s?.metrics || {};
  const refs = [...(m.references || [])].sort((a, b) => (a.source === "curated" ? 0 : 1) - (b.source === "curated" ? 0 : 1));
  return (
    <>
      {s && (
        <Card title={s.title} right={<span className="tag">Does not affect the grade</span>}>
          <p className="small muted">Source: {m.online ? "OpenAlex" : "curated reading list"}{m.query ? ` · query “${m.query}”` : ""}</p>
          <ul className="list">
            {s.summary && <li>{s.summary}</li>}
            {(s.warnings || []).map((w, i) => <li key={i}>{w}</li>)}
          </ul>
          {s.error && <div className="banner-error card-banner">{s.error}</div>}
        </Card>
      )}
      <Card title="Studies">
        {refs.length === 0 ? <p className="small muted">No papers came back.</p> : (
          <ul className="studies">
            {refs.map((ref, i) => {
              const url = refUrl(ref);
              return (
                <li key={i}>
                  {url ? <a href={url} target="_blank" rel="noopener noreferrer" className="study-title">{ref.title}</a> : <span className="study-title">{ref.title}</span>}
                  <div className="study-meta">{[ref.year, ref.venue, ref.citations != null && `${ref.citations.toLocaleString()} citations`, ref.authors]
                    .filter(Boolean).join(" · ")}</div>
                </li>
              );
            })}
          </ul>
        )}
      </Card>
      <Card title="Go beyond the data">
        <p className="small">
          The reading list never changes the grade. To find causes your dataset could not contain, search the literature for
          what drives any variable on the Investigation page. Findings appear as <b>hypothetical</b> variables with a
          Reliability Score.
        </p>
        {r.influence?.nodes?.length > 0 && <a className="btn sm primary" href={`#/investigate/${job.id}`}>Open the Investigation page →</a>}
      </Card>
    </>
  );
}

// ------------------------------------------------------------------------------------------- Future Design
const PRIORITY = { failed: "high", tested_concern: "medium", assumed: "low" };
const PRIO_ORDER = { high: 0, medium: 1, low: 2 };

/** v1's recommendation rows, built from what this engine found: each problem or untested assumption, then the next study. */
function recommendations(rep) {
  const recs = [];
  if (rep.estimate?.refused) recs.push({ area: "Estimate", priority: "high", text: `No number was given: ${rep.estimate.refusal_reason}` });
  checksOf(rep).filter((s) => PRIORITY[s.status]).forEach((s) => {
    const untested = (s.assumptions || []).find((a) => a.status === "assumed" && a.detail && /testable|would become|could be/i.test(a.detail));
    recs.push({ area: shortTitle(s), priority: PRIORITY[s.status], text: s.summary + (untested ? ` ${untested.detail}` : "") });
  });
  const sug = rep.suggestion;
  if (sug?.summary) recs.push({ area: "Next study", priority: rep.verdict?.label === "Supported" ? "medium" : "high", text: sug.summary });
  (sug?.warnings || []).forEach((w) => recs.push({ area: "Power", priority: "medium", text: w }));
  return recs.sort((a, b) => PRIO_ORDER[a.priority] - PRIO_ORDER[b.priority]);
}

export function FutureDesignPane({ r }) {
  const rep = r.report;
  const s = rep.suggestion;
  const m = s?.metrics || {};
  const recs = recommendations(rep);
  const charts = useMemo(() => (s ? R.chartsHtml(s.charts) : ""), [s]);
  const assumptions = Object.entries(m.assumptions || {}).filter(([, v]) => v !== null && v !== undefined);
  return (
    <>
      <Card title="Suggestions for future design">
        {recs.length === 0 ? <p className="small muted">No problems were found that a new study would need to fix.</p>
          : recs.map((x, i) => (
            <div key={i} className="rec">
              <div><b>{x.area}</b><div className={`prio ${x.priority}`}>{x.priority}</div></div>
              <div>{x.text}</div>
            </div>
          ))}
      </Card>
      {s && (
        <Card title="How big the next study needs to be">
          <p><b>{m.design || "Suggested design"}</b>{m.n_total == null && (m.n_by_effect_size || []).length > 0 && <span className="muted"> · the claim states no size, so the number needed depends on the effect that matters:</span>}</p>
          {m.n_total != null && (
            <div className="stats">
              <div className="stat"><span className="stat-n">{m.n_per_arm != null ? m.n_per_arm.toLocaleString() : "–"}</span><span className="stat-k">per arm</span></div>
              <div className="stat"><span className="stat-n">{m.n_total.toLocaleString()}</span><span className="stat-k">in total</span></div>
            </div>
          )}
          {(m.n_by_effect_size || []).length > 0 && (
            <div className="tbl-wrap">
              <table className="tbl">
                <thead><tr><th className="num">Effect (SD)</th><th className="num">Effect (outcome units)</th><th className="num">Per arm</th><th className="num">Total</th></tr></thead>
                <tbody>{m.n_by_effect_size.map((row, i) => (
                  <tr key={i}><td className="num">{fmt(row.effect_sd)}</td><td className="num">{fmt(row.effect_units)}</td>
                    <td className="num">{row.n_per_arm?.toLocaleString()}</td><td className="num">{row.n_total?.toLocaleString()}</td></tr>
                ))}</tbody>
              </table>
            </div>
          )}
          {assumptions.length > 0 && (
            <details className="why">
              <summary>Assumptions behind these numbers</summary>
              <KV rows={assumptions.map(([k, v]) => [label(k), typeof v === "number" ? fmt(v) : String(v)])} />
            </details>
          )}
          <Html html={charts} />
        </Card>
      )}
    </>
  );
}

// ------------------------------------------------------------------------------------------- Python Code
const py = (v) => (v == null ? "None" : JSON.stringify(v, null, 4)
  .replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None"));

/** The analysis as runnable Python, one block per pipeline step (the same calls, in the same order, as
    backend/causality/pipeline.py). Run together, the blocks reproduce the report. */
export function pythonSteps(job, rep) {
  const p = job.params || {};
  const name = job.filename || "data.csv";
  const ext = name.split(".").pop().toLowerCase();
  const reader = { xlsx: "read_excel", xls: "read_excel", json: "read_json", parquet: "read_parquet" }[ext] || "read_csv";
  const sep = ext === "tsv" ? ', sep="\\t"' : "";
  const roles = Object.fromEntries(Object.entries(rep?.data?.roles || p.roles || {}).filter(([, v]) => v !== null && !(Array.isArray(v) && !v.length) && v !== false));
  const audit = rep?.mode !== "explore";
  const steps = [];
  steps.push(["load", `"""Reproduce the Causal Audit analysis "${job.title}" step by step.
Put ${name} next to this script; run it from the project's backend/ folder (or add backend/ to PYTHONPATH)."""
import pandas as pd

from causality import data, pipeline
from causality.schema import Roles, Section, Status

raw = pd.${reader}(${JSON.stringify(name)}${sep})
roles = pipeline.parse_roles(${py(roles)})  # the roles the analysis used`]);
  if (audit) {
    steps.push(["claim", `from causality.claim import parse_claim

# columns are matched on a first preparation (declared roles only)
prep0 = data.prepare(raw, Roles(**{**roles.__dict__, "post_treatment": list(roles.post_treatment)}))
claim = parse_claim(${py(job.claim || "")}, prep0)
print("treatment:", claim.treatment_phrase, "| outcome:", claim.outcome_phrase, "| direction:", claim.direction, "| size:", claim.size)`]);
  }
  steps.push(["preprocess", `prep = data.prepare(raw, roles)   # types, missing values, duplicates (kept), one-hot encoding
${audit ? "claim.treatment_col, claim.outcome_col = roles.treatment, roles.outcome\n" : ""}summary = data.summary(prep)
print(summary["n_rows"], "rows x", summary["n_columns"], "columns; adjust for:", summary["adjustment_set"])`]);
  if (!audit) {
    steps.push(["explore", `from causality import explore

ex = explore.run(prep)            # FCI (main graph), PC and LiNGAM, stability-checked
print(ex.summary)
for pair in ex.metrics.get("pairs", []):
    print(pair.get("drawn"), "| FCI", pair.get("fci"), "| PC", pair.get("pc"), "| LiNGAM", pair.get("lingam"))`]);
    return steps;
  }
  steps.push(["design", `design = data.resolve_design(prep, ${py(p.design || null)})
print("design:", design.kind, "| declared:", design.declared)`]);
  steps.push(["balance", `from causality.checks import balance

bal = balance.run(prep, design, claim)
print(bal.status, bal.summary)
# under randomization, columns the treatment strongly predicts were most likely measured after it
sus = (bal.metrics or {}).get("suspected_post_treatment") or []
if sus and prep.roles.randomized:
    data.mark_post_treatment(prep, list(sus), "most likely measured after treatment")`]);
  steps.push(["checks", `from causality import design_assess
from causality.checks import attrition, interference, literature, overlap

ov = overlap.run(prep, design, claim)
att = attrition.run(prep, design, claim)
inter = interference.run(prep, design, claim)
lit = literature.run(prep, design, claim, online=${py(!!p.online_literature)})
design.assumptions, design_sections = design_assess.assess(prep, design, claim)
for s in [ov, att, inter, *design_sections]:
    print(s.title, "|", s.status, "|", s.summary)`]);
  steps.push(["estimate", `from causality import estimate as est_mod

est = est_mod.estimate(prep, design, claim)
if est.refused:
    print("No estimate:", est.refusal_reason)
else:
    print(f"{est.method}: {est.estimate:.4g}, 95% interval {est.ci95}, p = {est.p_value:.3g}")
    for c in est.comparisons:
        print("  also:", c["method"], c["estimate"])`]);
  steps.push(["sensitivity", `from causality.checks import sensitivity

sens = sensitivity.run(prep, design, claim, est)
if sens.status == Status.PASS:    # hidden confounding is never a pass
    sens.status = Status.ASSUMED
print(sens.summary)`]);
  steps.push(["verdict", `from causality.verdict import make_verdict

sections = [bal, ov, sens, att, inter, *design_sections]
est_card = Section(key="estimate", title="Estimate", error=est.refusal_reason if est.refused else None)
design.grade, design.grade_reasons = design_assess.grade(design, sections + [est_card])
claim.estimand = pipeline._estimand(prep, design, claim, est)
verdict = make_verdict(claim, design, est, sections)
print("grade:", design.grade, "| verdict:", verdict.label, "-", verdict.sentence)
for a in pipeline.assumed_panel(design, sections):
    print("assumed, not tested:", a.name)`]);
  steps.push(["next study & diagram", `from causality.suggest import suggest_study

suggestion = suggest_study(prep, design, claim, est)
print(suggestion.summary)
dag = design_assess.dag(prep, design, claim)    # the assumed causal diagram (nodes and edges)
print([(e["from"], e["to"]) for e in dag.data["edges"]])`]);
  return steps;
}

function CopyButton({ text }) {
  const [copied, setCopied] = useState(false);
  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch { /* clipboard blocked: the download still works */ }
  }
  return <button className="btn sm" onClick={copy} aria-live="polite">{copied ? "Copied" : "Copy"}</button>;
}

export function CodePane({ r, job }) {
  const steps = useMemo(() => pythonSteps(job, r.report), [job, r]);
  const all = steps.map(([k, c]) => `# ---- ${k} ${"-".repeat(Math.max(4, 72 - k.length))}\n${c}\n`).join("\n");
  return (
    <>
      <Card title="Generated Python for every step"
        right={<button className="btn sm" onClick={() => download(`reproduce_${job.id}.py`, all, "text/x-python")}>Download .py</button>}>
        <p className="small muted">Reproduce the analysis outside the app: the same calls to the core analysis, in the same order. Run the blocks in order.</p>
      </Card>
      {steps.map(([k, code]) => (
        <Card key={k} title={k} right={<CopyButton text={code} />}>
          <pre className="code"><code>{code}</code></pre>
        </Card>
      ))}
    </>
  );
}

// ------------------------------------------------------------------------------------------- Data
export function DataPane({ r }) {
  const d = r.report.data || {};
  const stats = r.column_stats || {};
  const cols = d.columns || [];
  const n = d.n_rows || 0;
  return (
    <Card title={`${fmt(d.n_rows)} rows × ${fmt(d.n_columns)} columns`}
      right={d.n_duplicates ? <span className="small muted">{fmt(d.n_duplicates)} duplicate rows, kept</span> : null}>
      <div className={`tbl-wrap${cols.length > 20 ? " tall" : ""}`}>
        <table className="tbl data-tbl">
          <thead><tr><th>Column</th><th>Type</th><th className="num">Missing</th><th className="num">Unique</th><th className="num">Mean</th></tr></thead>
          <tbody>
            {cols.map((c) => {
              const miss = stats[c.name]?.pct_missing ?? (n ? (100 * c.n_missing) / n : null);
              const mean = stats[c.name]?.mean;
              return (
                <tr key={c.name} title={c.note || undefined}>
                  <td>{c.name}{(c.encoded_as || []).length > 1 && <span className="small muted"> → {(c.encoded_as || []).length} columns</span>}</td>
                  <td><span className="tag">{c.kind}</span></td>
                  <td className="num" title={`${c.n_missing.toLocaleString()} missing`}>{miss != null ? pct(miss) : c.n_missing.toLocaleString()}</td>
                  <td className="num">{c.n_unique.toLocaleString()}</td>
                  <td className="num">{["continuous", "count", "binary"].includes(c.kind) && mean != null ? fmt(mean) : "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// ------------------------------------------------------------------------------------------- Pipeline
const STEP_PILLS = { balance: ["balance"], sensitivity: ["hidden_confounding"] };

/** The finished run: one row per step; the checks step opens into a dashed box with each check's result (v1). */
export function PipelinePane({ r, job }) {
  const rep = r.report || {};
  const sec = Object.fromEntries((rep.sections || []).map((s) => [s.key, s]));
  const inChecks = (rep.sections || []).filter((s) => !["balance", "hidden_confounding"].includes(s.key));
  const total = job.elapsed != null ? `${fmt(job.elapsed, 1)}s` : "";
  const Row = ({ s, pill }) => (
    <li className={`pstep ${s.status}`}>
      <span className="dot" aria-hidden="true" />
      <span className="step-label">{s.label}</span>
      {pill && <StatusPill status={pill.status} />}
      <span className="note">{s.seconds != null ? `${s.seconds}s` : s.status === "skipped" ? "skipped" : s.status === "error" ? "failed" : ""}</span>
      {s.note && <span className="step-note err small">{s.note}</span>}
    </li>
  );
  return (
    <section className="card pane-card pipeline-pane">
      <div className="card-title-row"><h3>Pipeline</h3><span className="small muted">{rep.mode || job.mode}{total && ` · ${total}`}</span></div>
      <ol className="psteps">
        {(job.steps || []).map((s) => (s.key === "checks" && inChecks.length ? (
          <li key={s.key} className="par">
            <div className="par-title"><span>{s.label}</span><span className="note">{s.seconds != null ? `${s.seconds}s` : ""}</span></div>
            <ul className="par-grid">
              {inChecks.map((c) => (
                <li key={c.key} className="pstep done">
                  <span className="dot" aria-hidden="true" />
                  <span className="step-label">{c.title}</span>
                  {STATUS[c.status] ? <StatusPill status={c.status} /> : <span className="tag">info</span>}
                </li>
              ))}
            </ul>
          </li>
        ) : <Row key={s.key} s={s} pill={(STEP_PILLS[s.key] || []).map((k) => sec[k]).find(Boolean)} />))}
      </ol>
    </section>
  );
}

// ------------------------------------------------------------------------------------------- Claim only
const DIRECTION = { supports: "pass", contradicts: "fail", mixed: "warn" };

export function ClaimOverviewPane({ r, job }) {
  const c = r.claim_only || {};
  return (
    <>
      <SummaryCard r={r} />
      <Card title="The claim">
        <div className="disclaimer" role="note">
          <b>No data, so no verdict.</b> This is what published research reports about the claim, found by an AI web
          search. It may be incomplete or wrong. Upload a dataset to test the claim from data.
        </div>
        <KV rows={[["Claim", job.claim], ["Cause", c.treatment || "not stated"], ["Effect", c.outcome || "not stated"], ["Population", c.population || "not stated"]]} />
      </Card>
    </>
  );
}

export function ClaimLiteraturePane({ r }) {
  const ev = r.claim_only?.evidence || [];
  return (
    <Card title="What the research says">
      {!ev.length && <p className="small muted">No findings with verifiable sources came back.</p>}
      <ul className="studies">
        {ev.map((e, i) => (
          <li key={i}>
            <div className="card-title-row"><span>{e.finding}</span><span className={`status ${DIRECTION[e.direction] || "na"}`}>{e.direction}</span></div>
            <ul className="sources">
              {e.sources.map((s, j) => (
                <li key={j}><a href={s.url} target="_blank" rel="noopener noreferrer">{s.title}</a>{s.year ? <span className="muted"> · {s.year}</span> : null}</li>
              ))}
            </ul>
          </li>
        ))}
      </ul>
    </Card>
  );
}

export function ClaimDesignPane({ r }) {
  const c = r.claim_only || {};
  return (
    <div className="grid2">
      <Card title="What would bias a naive comparison">
        <ul className="list">{(c.confounders || []).map((x, i) => <li key={i}><b>{x.name}</b>: {x.why}</li>)}</ul>
      </Card>
      <Card title="Designs that could test it">
        <ul className="list">{(c.designs || []).map((x, i) => <li key={i}><b>{x.design}</b>: {x.why}</li>)}</ul>
        <h4 className="sub-h">Data you would need</h4>
        <ul className="list">{(c.data_needed || []).map((x, i) => <li key={i}>{x}</li>)}</ul>
      </Card>
    </div>
  );
}
