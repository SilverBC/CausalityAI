/* eslint-disable */
/* Report -> HTML builders from the original CausalityAI page. Pure (no DOM, no fetch), every value escaped, and they
   encode the rule "assumed is never shown as a pass". The React tabs place these pieces; the standalone HTML export
   inlines this exact function with Function#toString, so a downloaded report renders the same offline. */
import Charts, { defineCharts } from "./charts.js";

function defineRender(Charts) {
  "use strict";

  // ------------------------------------------------------------------ helpers
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function num(v) { return typeof v === "number" && isFinite(v); }
  function fmt(v) {
    if (v == null) return "–";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (!num(v)) return esc(v);
    if (v === 0) return "0";
    if (Number.isInteger(v) && Math.abs(v) < 1e6) return String(v);
    var a = Math.abs(v);
    if (a >= 1e6 || a < 1e-3) return v.toExponential(2);
    return String(+v.toPrecision(Math.max(3, Math.floor(Math.log(a) / Math.LN10) + 1)));
  }
  function arr(v) { return Array.isArray(v) ? v : []; }
  function human(k) {
    k = String(k).replace(/_/g, " ");
    return k.charAt(0).toUpperCase() + k.slice(1);
  }
  function ci(t) {
    return Array.isArray(t) && t.length >= 2 ? "[" + fmt(t[0]) + ", " + fmt(t[1]) + "]" : "–";
  }

  var STATUS = {
    tested_ok: { cls: "ok", text: "Tested: OK", icon: "✓" },
    tested_concern: { cls: "concern", text: "Tested: concern", icon: "!" },
    failed: { cls: "fail", text: "Failed", icon: "✕" },
    assumed: { cls: "assumed", text: "Assumed (not tested)", icon: "?" }
  };
  function pill(status) {
    var s = STATUS[status];
    if (!s) return '<span class="pill info">Information</span>';
    return '<span class="pill ' + s.cls + '" title="' + esc(s.text) + '"><span class="pill-icon" aria-hidden="true">' +
      s.icon + "</span>" + esc(s.text) + "</span>";
  }

  function verdictClass(label) {
    var l = String(label || "").toLowerCase();
    if (l === "supported") return "v-ok";
    if (l.indexOf("partly") === 0) return "v-concern";
    if (l === "contradicted") return "v-fail";
    if (l === "no evidence") return "v-none";
    return "v-neutral";
  }
  function subClass(label) {
    var l = String(label || "").toLowerCase();
    if (l === "supported") return "v-ok";
    if (l === "rejected" || l === "contradicted") return "v-fail";
    if (l === "capped" || l === "inconclusive" || l.indexOf("partly") >= 0) return "v-concern";
    return "v-neutral";
  }
  function gradeClass(g) {
    g = String(g || "").toUpperCase();
    return { A: "g-a", B: "g-b", C: "g-c", D: "g-d", E: "g-d", F: "g-f" }[g] || "g-none";
  }

  function list(items, cls) {
    items = arr(items).filter(function (x) { return x != null && x !== ""; });
    if (!items.length) return "";
    return '<ul class="' + (cls || "plain-list") + '">' + items.map(function (x) {
      return "<li>" + esc(typeof x === "object" ? JSON.stringify(x) : x) + "</li>";
    }).join("") + "</ul>";
  }

  // ------------------------------------------------------------------ metrics
  function isNumArray(v) { return Array.isArray(v) && v.every(function (x) { return x == null || typeof x === "number"; }); }
  function isRecordArray(v) {
    return Array.isArray(v) && v.length > 0 && v.every(function (x) { return x && typeof x === "object" && !Array.isArray(x); });
  }
  function valueHtml(v, depth) {
    if (v == null) return "–";
    if (typeof v === "number" || typeof v === "boolean") return fmt(v);
    if (typeof v === "string") return esc(v.length > 400 ? v.slice(0, 400) + "…" : v);
    if (Array.isArray(v)) {
      if (!v.length) return "none";
      if (isNumArray(v)) return v.length <= 8 ? "[" + v.map(fmt).join(", ") + "]" : '<span class="muted">' + v.length + " values</span>";
      if (isRecordArray(v)) return depth < 2 ? recordTable(v, 25) : '<span class="muted">' + v.length + " rows</span>";
      if (v.length > 20) return '<span class="muted">' + v.length + " items</span>";
      return v.map(function (x) { return valueHtml(x, depth + 1); }).join(", ");
    }
    if (typeof v === "object") {
      if (depth >= 2) return '<span class="muted">…</span>';
      return kvTable(v, depth + 1);
    }
    return esc(v);
  }
  function kvTable(obj, depth) {
    var keys = Object.keys(obj || {});
    if (!keys.length) return "";
    var rows = keys.map(function (k) {
      return '<tr><th scope="row">' + esc(human(k)) + "</th><td>" + valueHtml(obj[k], depth || 0) + "</td></tr>";
    }).join("");
    return '<div class="table-wrap"><table class="kv">' + rows + "</table></div>";
  }
  function recordTable(rows, cap, cols) {
    rows = arr(rows);
    if (!rows.length) return "";
    if (!cols) {
      cols = [];
      rows.forEach(function (r) { Object.keys(r || {}).forEach(function (k) { if (cols.indexOf(k) < 0) cols.push(k); }); });
    }
    var shown = rows.slice(0, cap || 50);
    var head = "<tr>" + cols.map(function (c) { return '<th scope="col">' + esc(human(c)) + "</th>"; }).join("") + "</tr>";
    var body = shown.map(function (r) {
      return "<tr>" + cols.map(function (c) {
        var v = r ? r[c] : null;
        if (STATUS[v]) return "<td>" + pill(v) + "</td>";
        return "<td>" + (v != null && typeof v === "object" ? (Array.isArray(v) && isNumArray(v) && v.length <= 4 ? "[" + v.map(fmt).join(", ") + "]" : '<span class="muted">…</span>') : fmt(v)) + "</td>";
      }).join("") + "</tr>";
    }).join("");
    var more = rows.length > shown.length ? '<p class="muted small">Showing ' + shown.length + " of " + rows.length + " rows.</p>" : "";
    return '<div class="table-wrap"><table class="records"><thead>' + head + "</thead><tbody>" + body + "</tbody></table></div>" + more;
  }

  // ------------------------------------------------------------------ pieces
  function chartsHtml(charts, opts) {
    return arr(charts).map(function (c) {
      if (!c) return "";
      return '<figure class="chart-fig">' +
        (c.title ? '<figcaption class="chart-title">' + esc(c.title) + "</figcaption>" : "") +
        Charts.render(c, opts) +
        (c.kind === "dag" && c.data && arr(c.data.nodes).length ? dagLegend(c) : "") +
        (c.note ? '<p class="chart-note">' + esc(c.note) + "</p>" : "") +
        "</figure>";
    }).join("");
  }

  function dagLegend(chart) {
    var types = {};
    arr(chart && chart.data && chart.data.edges).forEach(function (e) { if (e) types[e.type || "directed"] = true; });
    var items = [
      ["directed", '<svg viewBox="0 0 44 12" class="leg-svg" aria-hidden="true"><line x1="2" y1="6" x2="36" y2="6" style="stroke:var(--edge);stroke-width:2"/><path d="M34 1L42 6L34 11z" style="fill:var(--edge)"/></svg>', "arrow: direction the data supports (or the direction we assume)"],
      ["partial", '<svg viewBox="0 0 44 12" class="leg-svg" aria-hidden="true"><circle cx="6" cy="6" r="4" style="fill:var(--bg-card);stroke:var(--edge);stroke-width:1.6"/><line x1="10" y1="6" x2="36" y2="6" style="stroke:var(--edge);stroke-width:2"/><path d="M34 1L42 6L34 11z" style="fill:var(--edge)"/></svg>', "circle then arrow: the tail end is uncertain"],
      ["undirected", '<svg viewBox="0 0 44 12" class="leg-svg" aria-hidden="true"><circle cx="6" cy="6" r="4" style="fill:var(--bg-card);stroke:var(--edge);stroke-width:1.6"/><line x1="10" y1="6" x2="34" y2="6" style="stroke:var(--edge);stroke-width:2"/><circle cx="38" cy="6" r="4" style="fill:var(--bg-card);stroke:var(--edge);stroke-width:1.6"/></svg>', "circles / plain line: linked, direction uncertain"],
      ["bidirected", '<svg viewBox="0 0 44 12" class="leg-svg" aria-hidden="true"><path d="M10 1L2 6L10 11z" style="fill:var(--edge)"/><line x1="8" y1="6" x2="36" y2="6" style="stroke:var(--edge);stroke-width:2;stroke-dasharray:5 3"/><path d="M34 1L42 6L34 11z" style="fill:var(--edge)"/></svg>', "dashed, arrows both ends: a hidden common cause"]
    ];
    var present = items.filter(function (it) { return types[it[0]]; });
    if (!present.length) present = items.slice(0, 1);
    return '<div class="dag-legend"><p class="dag-hypo"><strong>Every role in this diagram is a hypothesis</strong>, not something the data proved.</p><ul>' +
      present.map(function (it) { return "<li>" + it[1] + "<span>" + esc(it[2]) + "</span></li>"; }).join("") +
      '</ul><ul class="node-legend">' +
      '<li><span class="swatch sw-t"></span>treatment</li><li><span class="swatch sw-o"></span>outcome</li>' +
      '<li><span class="swatch sw-c"></span>other measured column</li><li><span class="swatch sw-i"></span>instrument / running variable</li>' +
      '<li><span class="swatch sw-p"></span>measured after treatment</li><li><span class="swatch sw-h"></span>hidden (not in the data)</li></ul></div>';
  }

  function assumptionItem(a) {
    if (!a) return "";
    return '<li class="assumption">' + pill(a.status) +
      '<div class="a-body"><span class="a-name">' + esc(a.name || a.key) + "</span>" +
      (a.detail ? '<span class="a-detail">' + esc(a.detail) + "</span>" : "") +
      (a.testable === false ? '<span class="a-tag">cannot be tested from data</span>' : "") +
      "</div></li>";
  }

  function errorBox(msg) {
    if (!msg) return "";
    return '<div class="banner banner-error" role="alert"><strong>Something went wrong here:</strong> ' + esc(msg) + "</div>";
  }

  // opts.numbersOpen (default true): whether the "Numbers" disclosure starts open
  function sectionCard(sec, extraClass, opts) {
    var numbersOpen = !(opts && opts.numbersOpen === false);
    if (!sec) return "";
    var tags = "";
    if (sec.caps_verdict) tags += '<span class="tag tag-cap" title="Because of this check, the verdict cannot be better than Inconclusive">Caps the verdict</span>';
    if (sec.affects_grade === false) tags += '<span class="tag">Does not affect the grade</span>';
    var metrics = sec.metrics && typeof sec.metrics === "object" ? sec.metrics : {};
    var big = {}, small = {};
    Object.keys(metrics).forEach(function (k) {
      if (isRecordArray(metrics[k]) && metrics[k].length > 1) big[k] = metrics[k]; else small[k] = metrics[k];
    });
    var bigHtml = Object.keys(big).map(function (k) {
      return '<h4 class="sub-h">' + esc(human(k)) + "</h4>" + recordTable(big[k], 60);
    }).join("");
    var assumptions = arr(sec.assumptions);
    // explore: the diagram (and its legend) comes first, before the tables
    var isExplore = sec.key === "explore";
    var chartBlock = chartsHtml(sec.charts, isExplore ? { layout: "circle" } : null);
    return '<section class="card section-card ' + (extraClass || "") + '" id="sec-' + esc(sec.key) + '">' +
      '<header class="card-head"><h3>' + esc(sec.title || human(sec.key)) + "</h3>" + pill(sec.status) + tags + "</header>" +
      (sec.summary ? '<p class="summary">' + esc(sec.summary) + "</p>" : "") +
      errorBox(sec.error) +
      (isExplore ? chartBlock : "") +
      (Object.keys(small).length ? '<details class="metrics"' + (numbersOpen ? " open" : "") + '><summary>Numbers</summary>' + kvTable(small, 0) + "</details>" : "") +
      bigHtml +
      (assumptions.length ? '<h4 class="sub-h">Assumptions</h4><ul class="assumptions">' + assumptions.map(assumptionItem).join("") + "</ul>" : "") +
      (arr(sec.warnings).length ? '<div class="warnings"><h4 class="sub-h">Warnings</h4>' + list(sec.warnings, "warn-list") + "</div>" : "") +
      (isExplore ? "" : chartBlock) +
      "</section>";
  }

  // ------------------------------------------------------------------ blocks
  function errorsBanner(report) {
    var errs = arr(report.errors);
    if (!errs.length) return "";
    return '<div class="banner banner-error" role="alert"><strong>Some parts could not run.</strong>' + list(errs) + "</div>";
  }

  function needsInputBlock(report, interactive) {
    var qs = arr(report.needs_input);
    if (!qs.length) return "";
    return '<section class="card needs-input" aria-live="polite"><h2>We need your answer before we can go on</h2>' +
      qs.map(function (q, i) {
        var opts = arr(q.options);
        var control = interactive
          ? '<select data-ni="' + i + '" data-kind="' + esc(q.kind) + '" data-role="' + esc(q.role || "") + '">' +
            opts.map(function (o) { return '<option value="' + esc(o) + '">' + esc(o) + "</option>"; }).join("") + "</select>"
          : list(opts);
        return '<div class="ni-q"><label><span class="ni-text">' + esc(q.question) + "</span>" + control + "</label></div>";
      }).join("") +
      (interactive ? '<button type="button" class="btn primary" id="ni-submit">Use these answers and analyze again</button>' : "") +
      "</section>";
  }

  function verdictBand(report) {
    var v = report.verdict, d = report.design || {}, est = report.estimand;
    var grade = d.grade;
    var reasons = arr(d.grade_reasons);
    var gradeHtml = grade
      ? '<details class="grade ' + gradeClass(grade) + '"><summary title="' + esc(reasons.join(" \n")) + '"><span class="grade-letter">' + esc(grade) +
        '</span><span class="grade-cap">Design grade<br><span class="small">why?</span></span></summary>' +
        (reasons.length ? list(reasons, "grade-reasons") : '<p class="small">No reasons given.</p>') + "</details>"
      : "";
    var claim = report.claim && report.claim.raw ? '<p class="claim-echo">Claim: <q>' + esc(report.claim.raw) + "</q></p>" : "";
    var main = v
      ? '<div class="verdict-label ' + verdictClass(v.label) + '">' + esc(v.label) + "</div>" +
        '<p class="verdict-sentence">' + esc(v.sentence) + "</p>" +
        (v.caveat ? '<p class="verdict-caveat"><strong>Only if:</strong> ' + esc(v.caveat) + "</p>" : "")
      : '<div class="verdict-label v-neutral">No verdict</div><p class="verdict-sentence">The analysis did not reach a verdict.</p>';
    return '<section class="card verdict-band">' +
      '<div class="verdict-main">' + claim + main +
      (est && est.text ? '<p class="estimand"><span class="k">What was estimated:</span> ' + esc(est.text) + "</p>" : "") +
      (v && arr(v.reasons).length ? '<details class="v-reasons"><summary>Why this verdict</summary>' + list(v.reasons) + "</details>" : "") +
      "</div>" + gradeHtml + "</section>";
  }

  function subVerdicts(report) {
    var subs = arr(report.verdict && report.verdict.sub).filter(function (s) { return s && typeof s === "object"; });
    if (!subs.length) return "";
    return '<div class="sub-verdicts">' + subs.map(function (s) {
      return '<div class="sub-card ' + subClass(s.label) + '"><div class="sub-aspect">' + esc(human(s.aspect)) + "</div>" +
        '<div class="sub-label">' + esc(human(s.label || "")) + "</div>" +
        (s.detail ? '<div class="sub-detail">' + esc(s.detail) + "</div>" : "") + "</div>";
    }).join("") + "</div>";
  }

  function assumedPanel(report) {
    var items = arr(report.assumed_panel).filter(function (a) { return a && typeof a === "object"; });
    if (!items.length) return "";
    return '<section class="card assumed-panel" aria-labelledby="assumed-h">' +
      '<h3 id="assumed-h"><span class="q-icon" aria-hidden="true">?</span>Assumed, not tested</h3>' +
      '<p class="assumed-lead">These cannot be checked from this data. The result above holds only if each of them is true. ' +
      "Nothing here has passed a test.</p>" +
      '<ul class="assumed-list">' + items.map(function (a) {
        return '<li><span class="a-name">' + esc(a.name || a.key) + "</span>" +
          (a.detail ? '<span class="a-detail">' + esc(a.detail) + "</span>" : "") +
          (a.source ? '<span class="a-src">from: ' + esc(a.source) + "</span>" : "") + "</li>";
      }).join("") + "</ul></section>";
  }

  var SCOPE = {
    full: "everyone in the data", treated: "the treated units", compliers: "units whose treatment followed the instrument",
    local_cutoff: "units close to the cutoff", overlap_trimmed: "units with comparable counterparts (extremes trimmed)"
  };

  function estimateCard(report) {
    var e = report.estimate;
    if (!e) return "";
    var rows = [];
    var add = function (k, v) { rows.push('<tr><th scope="row">' + esc(k) + "</th><td>" + v + "</td></tr>"); };
    add("Method", esc(e.method));
    add("Target", esc(e.target));
    add("Who it applies to", esc(SCOPE[e.scope] || e.scope));
    if (!e.refused) {
      add("Estimate", '<strong class="big-num">' + fmt(e.estimate) + "</strong>" + (num(e.se) ? ' <span class="muted">(standard error ' + fmt(e.se) + ")</span>" : ""));
      add("90% interval", ci(e.ci90));
      add("95% interval", ci(e.ci95));
      if (num(e.p_value)) add("p-value", fmt(e.p_value));
    }
    add("Rows used", fmt(e.n));
    if (num(e.baseline_mean)) add("Comparison-group mean", fmt(e.baseline_mean));
    if (num(e.outcome_sd)) add("Outcome spread (SD)", fmt(e.outcome_sd));

    var frows = [];
    if (!e.refused && num(e.estimate)) {
      frows.push({ label: e.method || "main estimate", est: e.estimate, lo: e.ci95 ? e.ci95[0] : null, hi: e.ci95 ? e.ci95[1] : null });
    }
    arr(e.comparisons).forEach(function (c) {
      if (!c) return;
      var c95 = c.ci95 || [];
      frows.push({ label: c.method || "other", est: c.estimate, lo: c95[0], hi: c95[1] });
    });
    var forest = frows.length > 1 ? Charts.render({ kind: "forest", title: "Main estimate next to other methods (95% intervals)", data: { rows: frows, ref: 0 }, x_label: "effect on the outcome" }) : "";
    var notes = arr(e.comparisons).filter(function (c) { return c && c.note; }).map(function (c) { return (c.method || "") + ": " + c.note; });
    return '<section class="card estimate-card" id="sec-estimate"><header class="card-head"><h3>The estimate</h3>' +
      (e.refused ? '<span class="pill fail"><span class="pill-icon" aria-hidden="true">✕</span>Not estimated</span>' : "") + "</header>" +
      (e.refused ? '<div class="banner banner-warn"><strong>We refused to give a number.</strong> ' + esc(e.refusal_reason || "No reason given.") + "</div>" : "") +
      '<div class="table-wrap"><table class="kv">' + rows.join("") + "</table></div>" +
      (forest ? '<figure class="chart-fig"><figcaption class="chart-title">Other ways of estimating the same thing</figcaption>' + forest + "</figure>" : "") +
      (notes.length ? list(notes, "plain-list small") : "") +
      (e.extra && Object.keys(e.extra).length ? '<details class="metrics"><summary>More numbers</summary>' + kvTable(e.extra, 0) + "</details>" : "") +
      chartsHtml(e.charts) + "</section>";
  }

  function claimCard(report) {
    var c = report.claim;
    if (!c) return "";
    var rows = {};
    if (c.treatment_phrase || c.treatment_col) rows["Cause (treatment)"] = (c.treatment_phrase ? "“" + c.treatment_phrase + "” → " : "") + (c.treatment_col || "not matched") + (num(c.treatment_score) && c.treatment_col ? " (match " + fmt(c.treatment_score) + ")" : "");
    if (c.outcome_phrase || c.outcome_col) rows["Effect (outcome)"] = (c.outcome_phrase ? "“" + c.outcome_phrase + "” → " : "") + (c.outcome_col || "not matched") + (num(c.outcome_score) && c.outcome_col ? " (match " + fmt(c.outcome_score) + ")" : "");
    if (c.direction) rows["Direction"] = (c.negated ? "not " : "") + c.direction;
    if (num(c.size)) rows["Size"] = (c.bound ? c.bound.replace(/_/g, " ") + " " : "") + fmt(c.size) + " " + (c.size_unit || "");
    if (c.shape) rows["Shape"] = c.shape;
    if (c.population_phrase) rows["Population"] = c.population_phrase + (c.population_filter ? " → " + c.population_filter.column + " " + c.population_filter.op + " " + c.population_filter.value : "");
    if (arr(c.hedges).length) rows["Hedges"] = c.hedges.join(", ");
    var tr = Object.keys(rows).map(function (k) { return '<tr><th scope="row">' + esc(k) + "</th><td>" + esc(rows[k]) + "</td></tr>"; }).join("");
    return '<section class="card claim-card"><header class="card-head"><h3>How we read your claim</h3></header>' +
      (c.testable === false ? '<div class="banner banner-warn"><strong>This claim cannot be tested with this kind of data:</strong> ' + esc(c.untestable_reason || "") + "</div>" : "") +
      (tr ? '<div class="table-wrap"><table class="kv">' + tr + "</table></div>" : "") +
      list(c.notes, "plain-list small") + "</section>";
  }

  var DESIGN_NAMES = { experiment: "Randomized experiment", did: "Difference-in-differences", iv: "Instrumental variable", rd: "Regression discontinuity", observational: "Observational comparison" };
  function designCard(report) {
    var d = report.design;
    if (!d) return "";
    var scores = d.scores || {};
    var sk = Object.keys(scores).filter(function (k) { return num(scores[k]); });
    var chart = sk.length > 1 ? Charts.render({ kind: "bar", title: "How well each study design fits the data", data: { labels: sk.map(function (k) { return DESIGN_NAMES[k] || k; }), values: sk.map(function (k) { return scores[k]; }) }, x_label: "fit score (0 to 1)" }) : "";
    var conflicts = arr(d.conflicts), as = arr(d.assumptions);
    return '<section class="card design-card"><header class="card-head"><h3>Study design: ' + esc(DESIGN_NAMES[d.kind] || d.kind) + "</h3>" +
      '<span class="tag">' + (d.declared ? "you said so" : "inferred from the data") + "</span></header>" +
      (d.kind !== "experiment" ? '<p class="small muted">We never call a study an experiment unless you tell us treatment was randomized.</p>' : "") +
      list(d.reasons) +
      (conflicts.length ? '<h4 class="sub-h">What you declared vs. what the data shows</h4><ul class="assumptions">' + conflicts.map(assumptionItem).join("") + "</ul>" : "") +
      (as.length ? '<h4 class="sub-h">What this design relies on</h4><ul class="assumptions">' + as.map(assumptionItem).join("") + "</ul>" : "") +
      (chart ? '<figure class="chart-fig">' + chart + "</figure>" : "") + "</section>";
  }

  function dagCard(report) {
    if (!report.dag) return "";
    return '<section class="card dag-card" id="sec-dag"><header class="card-head"><h3>' + esc(report.dag.title || "Assumed causal diagram") + "</h3></header>" +
      chartsHtml([Object.assign({}, report.dag, { title: "" })], { layout: report.mode === "explore" ? "circle" : "layered" }) + "</section>";
  }

  function dataCard(report) {
    var d = report.data;
    if (!d || typeof d !== "object" || !Object.keys(d).length) return "";
    var top = '<div class="stat-row">' +
      (d.n_rows != null ? '<div class="stat"><span class="stat-n">' + fmt(d.n_rows) + '</span><span class="stat-k">rows</span></div>' : "") +
      (d.n_columns != null ? '<div class="stat"><span class="stat-n">' + fmt(d.n_columns) + '</span><span class="stat-k">columns</span></div>' : "") +
      (d.n_duplicates != null ? '<div class="stat"><span class="stat-n">' + fmt(d.n_duplicates) + '</span><span class="stat-k">duplicate rows — kept, not deleted</span></div>' : "") +
      "</div>";
    var cols = arr(d.columns);
    var colTable = cols.length ? '<details class="metrics"><summary>Column types (' + cols.length + ")</summary>" +
      recordTable(cols.map(function (c) {
        return { column: c.name, kind: c.kind, missing: c.n_missing, distinct: c.n_unique, became: arr(c.encoded_as).join(", "), note: c.note || "" };
      }), 300) + "</details>" : "";
    var adj = arr(d.adjustment_set);
    var roles = d.roles && typeof d.roles === "object" ? d.roles : null;
    var roleRows = {};
    if (roles) Object.keys(roles).forEach(function (k) {
      var v = roles[k];
      if (v == null || v === false || (Array.isArray(v) && !v.length)) return;
      roleRows[k] = v;
    });
    var known = { n_rows: 1, n_columns: 1, n_duplicates: 1, columns: 1, adjustment_set: 1, notes: 1, roles: 1 };
    var rest = {};
    Object.keys(d).forEach(function (k) { if (!known[k]) rest[k] = d[k]; });
    return '<section class="card data-card" id="sec-data"><header class="card-head"><h3>How the data was prepared</h3></header>' + top +
      list(d.notes) +
      '<h4 class="sub-h">Adjustment set (the columns every check controls for)</h4>' +
      (adj.length ? '<p class="chips">' + adj.map(function (c) { return '<span class="chip">' + esc(c) + "</span>"; }).join("") + "</p>" : '<p class="muted small">None.</p>') +
      (Object.keys(roleRows).length ? '<h4 class="sub-h">Column roles used</h4>' + kvTable(roleRows, 1) : "") +
      colTable + (Object.keys(rest).length ? kvTable(rest, 0) : "") + "</section>";
  }

  function exploreIntro(report) {
    return '<section class="card verdict-band explore-band"><div class="verdict-main">' +
      '<div class="verdict-label v-neutral">Exploring</div>' +
      '<p class="verdict-sentence">No claim was given, so we looked for what may cause what. These are leads to study, not conclusions.</p>' +
      "</div></section>";
  }

  // ------------------------------------------------------------------ main
  function render(report, opts) {
    opts = opts || {};
    if (!report || typeof report !== "object") return errorBox("The result could not be read.");
    var h = "";
    h += errorsBanner(report);
    h += needsInputBlock(report, !!opts.interactive);
    var sections = arr(report.sections);
    if (report.mode === "explore") {
      h += exploreIntro(report);
      if (report.explore) h += sectionCard(report.explore, "explore-card prominent");
      if (report.dag && !(report.explore && arr(report.explore.charts).some(function (c) { return c && c.kind === "dag"; }))) h += dagCard(report);
      if (sections.length) h += '<h2 class="group-h">Checks</h2><div class="cards">' + sections.map(function (s) { return sectionCard(s); }).join("") + "</div>";
      if (report.suggestion) h += sectionCard(report.suggestion, "suggestion-card");
      h += dataCard(report);
    } else {
      h += verdictBand(report);
      h += subVerdicts(report);
      h += assumedPanel(report);
      h += '<div class="cards two">' + claimCard(report) + designCard(report) + "</div>";
      h += estimateCard(report);
      if (sections.length) h += '<h2 class="group-h">Checks</h2><div class="cards">' + sections.map(function (s) { return sectionCard(s); }).join("") + "</div>";
      if (report.suggestion) h += sectionCard(report.suggestion, "suggestion-card");
      if (report.explore) h += sectionCard(report.explore, "explore-card");
      h += dagCard(report);
      h += dataCard(report);
    }
    return '<div class="report">' + h + "</div>";
  }

  return {
    render: render, esc: esc, fmt: fmt, pill: pill,
    // the individual pieces, for pages that lay the report out in tabs
    errorsBanner: errorsBanner, verdictBand: verdictBand, subVerdicts: subVerdicts, assumedPanel: assumedPanel,
    claimCard: claimCard, designCard: designCard, estimateCard: estimateCard, dagCard: dagCard, dataCard: dataCard,
    sectionCard: sectionCard, exploreIntro: exploreIntro, chartsHtml: chartsHtml, list: list,
    kvTable: kvTable, recordTable: recordTable, isRecordArray: isRecordArray, human: human
  };
}

const R = defineRender(Charts);
export default R;
export { defineRender, defineCharts };
