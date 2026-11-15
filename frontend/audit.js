/* CausalityAI: "Audit with Opus" mode.
 *
 * The model (Opus) reads the claim, picks the columns, the design and the covariates, calls the
 * evidence tools and judges what the numbers mean. Code only computes numbers and applies three
 * floor rules. This page shows that report as it is:
 *   - evidence items are neutral; only the model's reading of them is coloured;
 *   - "Assumed, not tested" is never styled as a pass;
 *   - when a floor rule changed the verdict, the model's first answer and each rule are shown.
 *
 * Talks to POST /api/audit (multipart: file, claim, context) and
 * POST /api/audit/{session_id}/answer (JSON {"answers": {id: text}}) and
 * POST /api/audit/{session_id}/followup (JSON {"question": text}) after a verdict.
 * Charts the model chose (report.screen) are drawn with the shared charts.js (window.CausalCharts).
 * Does not change app.js: it intercepts the shared form's submit (capture phase) in audit mode.
 */
(function () {
  "use strict";
  if (typeof document === "undefined") return;
  var form = document.getElementById("analyze-form");
  var out = document.getElementById("audit-results");
  if (!form || !out) return;

  var $ = function (id) { return document.getElementById(id); };
  var btn = $("audit-btn");
  var S = { report: null, busy: false, timer: null };

  // ------------------------------------------------------------------ helpers
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function arr(v) { return Array.isArray(v) ? v : []; }
  function obj(v) { return v && typeof v === "object" && !Array.isArray(v) ? v : null; }
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
  function human(k) {
    k = String(k).replace(/_/g, " ");
    return k.charAt(0).toUpperCase() + k.slice(1);
  }
  function safeId(id) { return "ev-" + String(id == null ? "" : id).replace(/[^A-Za-z0-9_-]/g, "_"); }

  var DESIGN_NAMES = {
    experiment: "Randomized experiment", did: "Difference-in-differences", iv: "Instrumental variable",
    rd: "Regression discontinuity", observational: "Observational comparison"
  };
  var TOOL_NAMES = {
    table_facts: "Facts about the table", general_checks: "General checks", structure_facts: "How treatment is laid out",
    design_tests: "Tests of the design", estimate_effect: "Effect estimate", sensitivity: "Sensitivity to hidden factors",
    explore: "Exploratory search", layout_facts: "How treatment is laid out", custom_check: "A check written for this question",
    custom_checks: "Checks written for this question"
  };
  var RULE_NAMES = {
    assumed_is_not_tested: "What is assumed is not tested",
    needs_a_real_interval: "A verdict needs a real interval",
    no_more_than_the_interval: "No more than the interval shows",
    unreadable_draft: "The verdict could not be read"
  };
  var RULE_HELP = {
    assumed_is_not_tested: "Everything the checks say no data can test must be listed as assumed, and a sensitivity number can never count as support.",
    needs_a_real_interval: "Any label other than Inconclusive must point at an effect estimate with a finite 95% interval, computed for the same treatment, outcome and design.",
    no_more_than_the_interval: "Supported and Not supported need the 95% interval entirely on one side of zero; No effect needs it close around zero.",
    unreadable_draft: "A verdict with no readable label is treated as Inconclusive."
  };
  var ROLE_NAMES = {
    treatment: "Treatment (the cause)", outcome: "Outcome (the effect)", unit: "Unit id", time: "Time",
    instrument: "Instrument", running: "Running variable", cutoff: "Cutoff", cluster: "Cluster", strata: "Strata"
  };

  function readingClass(r) {
    r = String(r || "").toLowerCase();
    if (r === "supports") return "r-supports";
    if (r === "concern") return "r-concern";
    if (r === "fails") return "r-fails";
    return "r-neutral";
  }
  function readingText(r) {
    r = String(r || "").toLowerCase();
    return { supports: "Supports the claim", concern: "A concern", fails: "Fails", "not informative": "Not informative" }[r] || (r ? human(r) : "No reading given");
  }
  function labelClass(l) {
    l = String(l || "").toLowerCase();
    if (l === "supported") return "v-ok";
    if (l === "not supported") return "v-fail";
    return "v-neutral";
  }

  // escape, then turn evidence ids like e5 into links to that item
  function cite(text, ids) {
    var h = esc(text);
    if (!ids) return h;
    return h.replace(/\b(e\d+)\b/g, function (m) {
      return ids[m] ? '<a class="cite" href="#' + safeId(m) + '">' + m + "</a>" : m;
    });
  }
  function citeList(items, ids, cls) {
    items = arr(items).filter(function (x) { return x != null && x !== ""; });
    if (!items.length) return "";
    return '<ul class="' + (cls || "plain-list") + '">' + items.map(function (x) {
      return "<li>" + cite(typeof x === "object" ? JSON.stringify(x) : x, ids) + "</li>";
    }).join("") + "</ul>";
  }
  function chips(cols, cls) {
    return '<span class="chips">' + arr(cols).map(function (c) {
      return '<span class="chip ' + (cls || "") + '">' + esc(c) + "</span>";
    }).join("") + "</span>";
  }

  // ------------------------------------------------------------------ numbers (expandable)
  function isNumArray(v) { return Array.isArray(v) && v.every(function (x) { return x == null || typeof x === "number"; }); }
  function isRecordArray(v) { return Array.isArray(v) && v.length > 0 && v.every(function (x) { return obj(x); }); }
  function valueHtml(v, depth) {
    if (v == null) return "–";
    if (typeof v === "number" || typeof v === "boolean") return fmt(v);
    if (typeof v === "string") return esc(v.length > 500 ? v.slice(0, 500) + "…" : v);
    if (Array.isArray(v)) {
      if (!v.length) return "none";
      if (isNumArray(v)) return v.length <= 10 ? "[" + v.map(fmt).join(", ") + "]" : '<span class="muted">' + v.length + " values</span>";
      if (isRecordArray(v)) return depth < 3 ? recordTable(v, 40, depth) : '<span class="muted">' + v.length + " rows</span>";
      if (v.every(function (x) { return typeof x === "string"; })) return '<ul class="plain-list">' + v.slice(0, 60).map(function (x) { return "<li>" + esc(x) + "</li>"; }).join("") + "</ul>";
      return v.slice(0, 40).map(function (x) { return valueHtml(x, depth + 1); }).join(", ");
    }
    if (typeof v === "object") return depth >= 3 ? '<span class="muted">…</span>' : kvTable(v, depth + 1);
    return esc(v);
  }
  function kvTable(o, depth) {
    var keys = Object.keys(o || {});
    if (!keys.length) return '<span class="muted">none</span>';
    return '<div class="table-wrap"><table class="kv">' + keys.map(function (k) {
      return '<tr><th scope="row">' + esc(human(k)) + "</th><td>" + valueHtml(o[k], depth || 0) + "</td></tr>";
    }).join("") + "</table></div>";
  }
  function recordTable(rows, cap, depth) {
    var cols = [];
    rows.forEach(function (r) { Object.keys(r || {}).forEach(function (k) { if (cols.indexOf(k) < 0) cols.push(k); }); });
    var shown = rows.slice(0, cap || 40);
    var cell = function (v) { return inline(v, 0); };
    return '<div class="table-wrap"><table class="records"><thead><tr>' +
      cols.map(function (c) { return '<th scope="col">' + esc(human(c)) + "</th>"; }).join("") + "</tr></thead><tbody>" +
      shown.map(function (r) { return "<tr>" + cols.map(function (c) { return "<td>" + cell(r ? r[c] : null) + "</td>"; }).join("") + "</tr>"; }).join("") +
      "</tbody></table></div>" +
      (rows.length > shown.length ? '<p class="muted small">Showing ' + shown.length + " of " + rows.length + " rows.</p>" : "");
  }

  // top level of an evidence item's numbers: small values in a key/value table, lists of records as full-width tables
  function numbersHtml(n) {
    var small = {}, big = [];
    Object.keys(n).forEach(function (k) { if (isRecordArray(n[k])) big.push(k); else small[k] = n[k]; });
    return (Object.keys(small).length ? kvTable(small, 0) : "") + big.map(function (k) {
      return '<h4 class="sub-h">' + esc(human(k)) + "</h4>" + recordTable(n[k], 60, 1);
    }).join("");
  }

  // compact text for a value inside a table cell: no nested tables
  function inline(v, depth) {
    if (v == null) return "–";
    if (typeof v !== "object") return typeof v === "string" ? esc(v.length > 200 ? v.slice(0, 200) + "…" : v) : fmt(v);
    if (Array.isArray(v)) {
      if (!v.length) return "none";
      if (isNumArray(v)) return v.length <= 6 ? "[" + v.map(fmt).join(", ") + "]" : '<span class="muted">' + v.length + " values</span>";
      if (depth >= 1 || v.length > 8) return '<span class="muted">' + v.length + " items</span>";
      return v.map(function (x) { return inline(x, depth + 1); }).join("; ");
    }
    if (depth >= 1) return '<span class="muted">…</span>';
    return '<span class="inline-kv">' + Object.keys(v).map(function (k) {
      return "<span>" + esc(human(k)) + ": <strong>" + inline(v[k], depth + 1) + "</strong></span>";
    }).join("") + "</span>";
  }

  function argsText(args) {
    var a = obj(args);
    if (!a || !Object.keys(a).length) return "no arguments";
    return Object.keys(a).map(function (k) {
      var v = a[k], s;
      if (v == null) s = "every column without a role";
      else if (Array.isArray(v)) s = v.length ? v.join(", ") : "none";
      else if (typeof v === "object") s = Object.keys(v).map(function (rk) { return rk + "=" + v[rk]; }).join(", ");
      else s = String(v);
      return esc(human(k)) + ": <code>" + esc(s) + "</code>";
    }).join(" · ");
  }

  // ------------------------------------------------------------------ blocks
  function evidenceIndex(report) {
    var ids = {};
    arr(report.evidence).forEach(function (e) { if (obj(e) && e.id != null) ids[String(e.id)] = e; });
    return ids;
  }

  function errorBlock(report) {
    var msg = report.error || "The audit stopped without saying why.";
    return '<div class="banner banner-error" role="alert"><strong>The audit could not finish.</strong> ' + esc(msg) + "</div>";
  }

  function questionsBlock(report) {
    var qs = arr(report.questions).filter(obj);
    if (!qs.length) return "";
    return '<section class="card needs-input" id="audit-questions"><h2>Opus needs your answer before it can go on</h2>' +
      '<p class="small muted">These are things the data cannot show. Pick an option or write your own answer.</p>' +
      qs.map(function (q, i) {
        var qid = q.id != null ? String(q.id) : "q" + (i + 1);
        var inputId = "aq-in-" + i;
        return '<div class="aq" data-qid="' + esc(qid) + '">' +
          '<p class="aq-text" id="aq-t-' + i + '">' + esc(q.question) + "</p>" +
          (arr(q.options).length ? '<div class="aq-opts" role="group" aria-labelledby="aq-t-' + i + '">' + arr(q.options).map(function (o) {
            return '<button type="button" class="btn aq-opt" aria-pressed="false" data-for="' + inputId + '" data-value="' + esc(o) + '">' + esc(o) + "</button>";
          }).join("") + "</div>" : "") +
          '<label class="field" for="' + inputId + '"><span class="field-label">Your answer</span>' +
          '<input type="text" id="' + inputId + '" class="aq-input" autocomplete="off" placeholder="Choose above or type here"></label></div>';
      }).join("") +
      '<div class="actions"><button type="button" class="btn primary" id="aq-send"><span class="spinner" aria-hidden="true"></span><span class="btn-text">Send answers</span></button></div>' +
      "</section>";
  }

  function verdictBlock(report, ids) {
    var v = obj(report.verdict);
    var claim = report.claim ? '<p class="claim-echo">Claim: <q>' + esc(report.claim) + "</q></p>" : "";
    if (!v) {
      return '<section class="card verdict-band audit-verdict"><div class="verdict-main">' + claim +
        '<div class="verdict-label v-neutral">No verdict</div><p class="verdict-sentence">The model did not submit a verdict.</p></div></section>';
    }
    var est = "";
    var e = v.estimate_evidence_id != null ? ids[String(v.estimate_evidence_id)] : null;
    var n = e && obj(e.numbers);
    if (n && num(n.estimate)) {
      var c = obj(n.ci) && Array.isArray(n.ci["95"]) ? n.ci["95"] : null;
      est = '<p class="estimand"><span class="k">Estimate used:</span> ' + fmt(n.estimate) +
        (c && c.length >= 2 ? ", 95% interval " + fmt(c[0]) + " to " + fmt(c[1]) : "") +
        (n.method ? " (" + esc(n.method) + ")" : "") + " " + cite(String(v.estimate_evidence_id), ids) + "</p>";
    } else if (v.estimate_evidence_id != null) {
      est = '<p class="estimand"><span class="k">Estimate cited:</span> ' + cite(String(v.estimate_evidence_id), ids) + "</p>";
    }
    var dir = v.claimed_direction ? '<p class="small muted">Direction the claim states: ' +
      esc({ increase: "an increase", decrease: "a decrease", none: "no effect" }[v.claimed_direction] || v.claimed_direction) + "</p>" : "";
    return '<section class="card verdict-band audit-verdict"><div class="verdict-main">' + claim +
      '<div class="verdict-label ' + labelClass(v.label) + '">' + esc(v.label || "No label") + "</div>" +
      '<p class="verdict-sentence">' + cite(v.sentence, ids) + "</p>" + dir + est +
      floorBlock(report, ids) +
      (arr(v.reasons).length ? '<details class="v-reasons" open><summary>Why, in the model\'s words</summary>' + citeList(v.reasons, ids) + "</details>" : "") +
      (arr(v.caveats).length ? '<h4 class="sub-h">Caveats</h4>' + citeList(v.caveats, ids) : "") +
      "</div></section>";
  }

  function floorBlock(report, ids) {
    var f = obj(report.floor);
    if (!f || !f.changed) return "";
    var d = obj(report.draft_verdict);
    var v = obj(report.verdict) || {};
    var vs = arr(f.violations).filter(obj);
    // rule 1 alone only adds to "assumed" or re-reads a judgment: the label and sentence stay the model's own
    var relabelled = d && (d.label !== v.label || d.sentence !== v.sentence);
    return '<div class="floor-box" role="note"><h3>' + (relabelled ? "The fixed rules changed what the model said" : "The fixed rules added to what the model said") + "</h3>" +
      (relabelled ? '<div class="floor-draft"><span class="small muted">The model first said:</span><br><span class="d-label">' + esc(d.label || "No label") +
        '</span><p class="small">' + cite(d.sentence, ids) + "</p></div>" : "") +
      (vs.length ? '<p class="small"><strong>' + (vs.length === 1 ? "This rule" : "These rules") + " changed it:</strong></p>" +
        '<ul class="floor-rules">' + vs.map(function (x) {
          return '<li><span class="r-name">' + esc(RULE_NAMES[x.rule] || human(x.rule || "rule")) + "</span>" +
            (RULE_HELP[x.rule] ? '<span class="small muted">' + esc(RULE_HELP[x.rule]) + "</span>" : "") +
            (x.detail ? '<span class="small">' + cite(x.detail, ids) + "</span>" : "") +
            (x.action ? '<span class="r-action">' + esc(x.action) + "</span>" : "") + "</li>";
        }).join("") + "</ul>" : '<p class="small">No rule details were given.</p>') +
      "</div>";
  }

  // never green, never a tick: dashed amber panel with a question mark
  function assumedBlock(report) {
    var v = obj(report.verdict);
    var items = arr(v && v.assumed).filter(function (x) { return x != null && x !== ""; });
    if (!items.length) return "";
    return '<section class="card assumed-panel" aria-labelledby="aud-assumed-h">' +
      '<h2 id="aud-assumed-h"><span class="q-icon" aria-hidden="true">?</span>Assumed, not tested</h2>' +
      '<p class="assumed-lead">No data can check these. The verdict above holds only if each of them is true. Nothing here has passed a test.</p>' +
      '<ul class="assumed-list">' + items.map(function (a) {
        return '<li><span class="a-detail">' + esc(typeof a === "object" ? JSON.stringify(a) : a) + "</span></li>";
      }).join("") + "</ul></section>";
  }

  function readingBlock(report) {
    var r = obj(report.reading);
    if (!r) return "";
    var rows = [];
    var add = function (k, v) { rows.push('<tr><th scope="row">' + esc(k) + "</th><td>" + v + "</td></tr>"); };
    add("Treatment (the cause)", r.treatment ? '<span class="chip">' + esc(r.treatment) + "</span>" : '<span class="muted">not chosen</span>');
    add("Outcome (the effect)", r.outcome ? '<span class="chip">' + esc(r.outcome) + "</span>" : '<span class="muted">not chosen</span>');
    add("Study design", r.design ? "<strong>" + esc(DESIGN_NAMES[r.design] || r.design) + "</strong>" : '<span class="muted">not chosen</span>');
    if (r.design_reason) add("Why this design", esc(r.design_reason));
    var roles = obj(r.roles) || {};
    var rk = Object.keys(roles).filter(function (k) { return roles[k] != null && roles[k] !== "" && k !== "treatment" && k !== "outcome"; });
    if (rk.length) add("Other roles", rk.map(function (k) {
      var val = Array.isArray(roles[k]) ? roles[k].join(", ") : roles[k];
      return esc(ROLE_NAMES[k] || human(k)) + ': <span class="chip">' + (typeof val === "number" ? fmt(val) : esc(val)) + "</span>";
    }).join("<br>"));
    var cov = r.covariates;
    add("Adjusted for", cov == null ? '<span class="muted">every column without a role (the model did not narrow the list)</span>'
      : (arr(cov).length ? chips(cov) : '<span class="muted">nothing</span>'));
    var ex = arr(r.excluded).filter(obj);
    add("Left out, and why", ex.length ? '<ul class="excluded-list">' + ex.map(function (x) {
      return '<li><span class="chip chip-out">' + esc(x.column) + '</span><span class="x-reason">' + esc(x.reason || "No reason given.") + "</span></li>";
    }).join("") + "</ul>" : '<span class="muted">nothing left out</span>');
    return '<section class="card reading-card"><header class="card-head"><h3>How Opus read the claim and the data</h3></header>' +
      '<div class="table-wrap"><table class="kv">' + rows.join("") + "</table></div>" +
      (report.context ? '<details class="metrics"><summary>Study description you gave</summary><p class="small">' + esc(report.context) + "</p></details>" : "") +
      "</section>";
  }

  function judgmentsFor(report) {
    var v = obj(report.verdict);
    var by = {};
    arr(v && v.judgments).filter(obj).forEach(function (j) {
      var k = String(j.evidence_id);
      (by[k] = by[k] || []).push(j);
    });
    return by;
  }

  function evidenceItem(e, judg, ids) {
    var id = String(e.id);
    var n = obj(e.numbers);
    var code = n && typeof n.code === "string" && n.code.trim() ? n.code : "";
    var secs = n && num(n.seconds) ? n.seconds : null;
    if (n && ("code" in n || "seconds" in n)) {
      n = Object.assign({}, n);
      delete n.code;
      delete n.seconds;
    }
    var hasNums = n && Object.keys(n).length;
    var js = judg[id] || [];
    var q = e.tool === "custom_check" && obj(e.args) && e.args.question ? e.args.question : "";
    return '<li class="ev-item" id="' + safeId(id) + '"><div class="ev-main">' +
      '<div class="ev-head"><span class="ev-id">' + esc(id) + '</span><span class="ev-tool">' + esc(TOOL_NAMES[e.tool] || human(e.tool || "tool")) +
      '</span><span class="ev-name">' + esc(e.tool) + "</span></div>" +
      (q ? '<p class="ev-question">' + esc(q) + "</p>" : '<p class="ev-args">' + argsText(e.args) + "</p>") +
      (secs != null ? '<p class="small muted ev-secs">Ran in ' + fmt(secs) + " s</p>" : "") +
      (e.ok === false ? '<p class="ev-error"><strong>This call could not run:</strong> ' + esc(e.error || "no reason given") + "</p>" : "") +
      (arr(e.lines).length ? '<ul class="ev-lines">' + arr(e.lines).map(function (l) { return "<li>" + esc(l) + "</li>"; }).join("") + "</ul>" : "") +
      (hasNums ? '<details class="metrics"><summary>Numbers</summary>' + numbersHtml(n) + "</details>" : "") +
      (code ? '<details class="metrics code-block"><summary>The code that computed this</summary><pre><code>' + esc(code) + "</code></pre></details>" : "") +
      '</div><div class="ev-side"><h4>The model\'s reading</h4>' +
      (js.length ? js.map(function (j) {
        return '<div class="judg ' + readingClass(j.reading) + '"><div class="judg-reading">' + esc(readingText(j.reading)) + "</div>" +
          (j.why ? '<p class="judg-why">' + cite(j.why, ids) + "</p>" : "") + "</div>";
      }).join("") : '<p class="ev-nojudg">The model did not cite this item.</p>') +
      "</div></li>";
  }

  function evidenceBlock(report, ids, title, collapsed) {
    var ev = arr(report.evidence).filter(obj);
    var judg = judgmentsFor(report);
    var orphans = [];
    Object.keys(judg).forEach(function (k) { if (!ids[k]) orphans = orphans.concat(judg[k]); });
    if (!ev.length && !orphans.length) return "";
    var body = '<p class="small muted">Each tool call in the order the model made it. The numbers themselves carry no pass or fail mark; the coloured notes on the right are the model\'s own reading.</p>' +
      '<ol class="ev-list">' + ev.map(function (e) { return evidenceItem(e, judg, ids); }).join("") + "</ol>" +
      (orphans.length ? '<div class="banner banner-warn"><strong>Judgments that cite evidence not in this report:</strong><ul>' + orphans.map(function (j) {
        return "<li>" + esc(j.evidence_id) + ": " + esc(readingText(j.reading)) + (j.why ? ", " + esc(j.why) : "") + "</li>";
      }).join("") + "</ul></div>" : "");
    var head = esc(title || "The evidence") + " (" + ev.length + ")";
    if (collapsed) return '<details class="card"><summary><strong>' + head + "</strong></summary>" + body + "</details>";
    return '<h2 class="group-h">' + head + "</h2>" + body;
  }

  // the pictures the model chose to show, in its order; drawn by the shared charts.js
  function screenBlock(report, ids) {
    var items = arr(report.screen).filter(function (x) { return obj(x) && obj(x.chart); });
    if (!items.length) return "";
    var Charts = window.CausalCharts;
    return '<section class="card screen-card" aria-labelledby="aud-screen-h"><header class="card-head"><h3 id="aud-screen-h">What the data look like</h3></header>' +
      '<p class="small muted">The pictures Opus chose to show alongside the verdict.</p>' +
      items.map(function (it) {
        var c = it.chart;
        var src = it.evidence_id != null && it.evidence_id !== "" ? ' <span class="small muted">from ' + cite(String(it.evidence_id), ids) + "</span>" : "";
        var notes = [it.note, c.note].filter(function (x, i, a) { return x && a.indexOf(x) === i; });
        return '<figure class="chart-fig">' +
          (c.title || src ? '<figcaption class="chart-title">' + esc(c.title || "") + src + "</figcaption>" : "") +
          (Charts && Charts.render ? Charts.render(c) : '<p class="small muted">The chart code did not load, so this picture cannot be drawn.</p>') +
          notes.map(function (x) { return '<p class="chart-note">' + cite(x, ids) + "</p>"; }).join("") +
          "</figure>";
      }).join("") + "</section>";
  }

  // ask Opus a further question about the same data; earlier questions are listed
  function followupBlock(report) {
    var asked = arr(report.followups).map(function (f) { return obj(f) ? f.question : f; })
      .filter(function (q) { return typeof q === "string" && q.trim(); });
    return '<section class="card followup-card" id="audit-followup" aria-labelledby="aud-fu-h"><h2 id="aud-fu-h">Ask a follow-up</h2>' +
      (asked.length ? '<h4 class="sub-h">Asked so far</h4><ol class="fu-asked">' + asked.map(function (q) { return "<li>" + esc(q) + "</li>"; }).join("") + "</ol>" : "") +
      '<label class="field" for="fu-input"><span class="field-label">Your question</span>' +
      '<textarea id="fu-input" rows="2" placeholder="For example: what if we drop 2020? Does it hold for the larger stores only?"></textarea></label>' +
      '<p class="small muted">Opus may run more checks and then gives the verdict again; the same fixed rules apply. This can take a minute or more.</p>' +
      '<div class="actions"><button type="button" class="btn primary" id="fu-send"><span class="spinner" aria-hidden="true"></span><span class="btn-text">Ask Opus</span></button></div>' +
      '<div id="fu-status" aria-live="polite"></div></section>';
  }

  function transcriptBlock(report) {
    var t = arr(report.transcript).filter(function (x) { return obj(x) && x.text; });
    if (!t.length) return "";
    return '<details class="card transcript"><summary><strong>What the model wrote along the way</strong></summary>' +
      t.map(function (x) {
        return '<p><span class="t-role">' + esc(x.role === "user" ? "You" : "Opus") + "</span><br>" + esc(x.text) + "</p>";
      }).join("") + "</details>";
  }

  function usageBlock(report) {
    var u = obj(report.usage);
    if (!u) return "";
    var parts = [];
    if (u.model) parts.push("model " + esc(u.model));
    if (num(u.turns)) parts.push(fmt(u.turns) + " turns");
    if (num(u.input_tokens)) parts.push(fmt(u.input_tokens) + " tokens in");
    if (num(u.output_tokens)) parts.push(fmt(u.output_tokens) + " tokens out");
    return parts.length ? '<p class="small muted audit-usage">' + parts.join(" · ") + "</p>" : "";
  }

  function render(report) {
    if (!obj(report)) return '<div class="banner banner-error" role="alert"><strong>The result could not be read.</strong></div>';
    var ids = evidenceIndex(report);
    var h = "";
    var status = report.status || "done";
    if (status === "error") {
      h += errorBlock(report);
      h += evidenceBlock(report, ids, "Evidence gathered before it stopped", true);
      h += transcriptBlock(report);
    } else if (status === "needs_input") {
      h += questionsBlock(report);
      h += screenBlock(report, ids);
      h += evidenceBlock(report, ids, "Evidence gathered so far", true);
      h += transcriptBlock(report);
    } else {
      h += verdictBlock(report, ids);
      h += followupBlock(report);
      h += assumedBlock(report);
      h += screenBlock(report, ids);
      h += readingBlock(report);
      h += evidenceBlock(report, ids);
      h += transcriptBlock(report);
    }
    h += usageBlock(report);
    return '<div class="report audit-report">' + h + "</div>";
  }

  // ------------------------------------------------------------------ app
  function message(msg, kind) {
    out.innerHTML = '<div class="banner ' + (kind === "info" ? "banner-info" : "banner-error") + '" role="alert">' + esc(msg) + "</div>";
  }

  function show(report, note) {
    S.report = report;
    try {
      out.innerHTML = (note ? '<div class="banner banner-info">' + esc(note) + "</div>" : "") + render(report);
    } catch (err) {
      message("The report arrived but could not be shown: " + plain(err));
      if (window.console) console.error(err);
      return;
    }
    wireQuestions();
    wireFollowup();
    out.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function wireQuestions() {
    Array.prototype.forEach.call(out.querySelectorAll(".aq-opt"), function (b) {
      b.addEventListener("click", function () {
        var input = $(b.getAttribute("data-for"));
        if (input) input.value = b.getAttribute("data-value");
        Array.prototype.forEach.call(b.parentNode.querySelectorAll(".aq-opt"), function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
      });
    });
    Array.prototype.forEach.call(out.querySelectorAll(".aq-input"), function (inp) {
      inp.addEventListener("input", function () {
        var box = inp.closest(".aq");
        if (box) Array.prototype.forEach.call(box.querySelectorAll(".aq-opt"), function (o) {
          o.setAttribute("aria-pressed", o.getAttribute("data-value") === inp.value ? "true" : "false");
        });
      });
      inp.addEventListener("keydown", function (e) { if (e.key === "Enter") { e.preventDefault(); sendAnswers(); } });
    });
    var send = $("aq-send");
    if (send) send.addEventListener("click", sendAnswers);
  }

  function readJson(resp) {
    return resp.text().then(function (t) {
      var body = null;
      try { body = t ? JSON.parse(t) : null; } catch (e) { body = null; }
      return { ok: resp.ok, status: resp.status, body: body, text: t };
    });
  }
  function plain(err) {
    var m = (err && err.message ? err.message : String(err)).replace(/\.+$/, "");
    if (/Failed to fetch|NetworkError|Load failed/i.test(m)) return "the server could not be reached";
    return m;
  }
  // a report, or the plain error message the server gave
  function handle(res) {
    var b = res.body;
    var rep = obj(b) && obj(b.report) ? b.report : b;
    if (obj(rep) && typeof rep.status === "string" && ("evidence" in rep || "verdict" in rep || "questions" in rep || rep.status === "error")) {
      show(rep);
      return;
    }
    var msg = obj(b) && (b.error || b.detail) ? (typeof (b.error || b.detail) === "string" ? (b.error || b.detail) : JSON.stringify(b.error || b.detail)) : null;
    throw new Error(msg || "The server answered with status " + res.status + " and no report");
  }

  function setBusy(on, label) {
    S.busy = on;
    [btn, $("aq-send"), $("fu-send")].forEach(function (b) {
      if (!b) return;
      b.disabled = on;
      b.classList.toggle("busy", on);
    });
    if (btn) btn.querySelector(".btn-text").textContent = on ? "Opus is working…" : "Audit with Opus";
    form.setAttribute("aria-busy", on ? "true" : "false");
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
    if (on) {
      var t0 = Date.now();
      var wait = '<section class="card audit-wait" role="status"><span class="spinner" aria-hidden="true"></span><div>' +
        "<h2>" + esc(label || "Opus is reading the data") + "</h2>" +
        '<p class="small">It picks the columns and the design, runs the checks it needs and judges the numbers. A run usually takes a minute or more. Please keep this page open.</p>' +
        '<p class="small muted">Waiting <span class="audit-elapsed" id="audit-elapsed">0 s</span></p></div></section>';
      if (label && out.querySelector("#audit-questions")) {
        out.querySelector("#audit-questions").insertAdjacentHTML("afterend", wait);
      } else {
        out.innerHTML = wait;
      }
      S.timer = setInterval(function () {
        var el = $("audit-elapsed");
        if (!el) return;
        var s = Math.round((Date.now() - t0) / 1000);
        el.textContent = s < 60 ? s + " s" : Math.floor(s / 60) + " min " + (s % 60) + " s";
      }, 1000);
    }
  }

  function currentFile() {
    var app = window.CausalApp;
    if (app && app.file) return app.file;
    var fi = $("file");
    return fi && fi.files && fi.files[0] ? fi.files[0] : null;
  }

  function runAudit() {
    if (S.busy) return;
    var f = currentFile();
    var claim = ($("claim").value || "").trim();
    var context = ($("audit-context").value || "").trim();
    if (!f) { message("Please choose a data file first (CSV, TSV, Excel or JSON)."); return; }
    if (!claim) { message("Please write the claim you want audited, for example: \"The program increased weekly sales.\""); $("claim").focus(); return; }
    var fd = new FormData();
    fd.append("file", f);
    fd.append("claim", claim);
    fd.append("context", context);
    setBusy(true);
    fetch("/api/audit", { method: "POST", body: fd }).then(readJson).then(function (res) {
      setBusy(false);
      handle(res);
    }).catch(function (err) {
      setBusy(false);
      message("We could not finish the audit: " + plain(err) + ".");
    });
  }

  function sendAnswers() {
    if (S.busy || !S.report) return;
    var answers = {}, any = false;
    Array.prototype.forEach.call(out.querySelectorAll(".aq"), function (box) {
      var inp = box.querySelector(".aq-input");
      var v = inp ? inp.value.trim() : "";
      if (v) { answers[box.getAttribute("data-qid")] = v; any = true; }
    });
    var warn = out.querySelector(".aq-warn");
    if (warn) warn.remove();
    var actions = $("aq-send") && $("aq-send").parentNode;
    if (!any) {
      if (actions) actions.insertAdjacentHTML("beforebegin", '<p class="aq-warn banner banner-warn">Please answer at least one question.</p>');
      return;
    }
    var sid = S.report.session_id;
    if (!sid || sid === "sample") {
      if (actions) actions.insertAdjacentHTML("beforebegin", '<p class="aq-warn banner banner-warn">This is a bundled sample, not a live session, so the answers cannot be sent. Upload a file and run an audit to answer for real.</p>');
      return;
    }
    setBusy(true, "Opus is going on with your answers");
    fetch("/api/audit/" + encodeURIComponent(sid) + "/answer", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ answers: answers })
    }).then(readJson).then(function (res) {
      setBusy(false);
      var b = res.body;
      var rep = obj(b) && obj(b.report) ? b.report : b;
      if (obj(rep) && rep.status === "error" && S.report && S.report.status === "needs_input") {
        // The server keeps the audit paused when a resume fails, so keep the questions and typed answers
        // on screen and let the same answers be sent again.
        throw new Error((rep.error || "the audit stopped without saying why").replace(/\.+$/, "") +
          ". Your answers are kept; send them again");
      }
      handle(res);
    }).catch(function (err) {
      setBusy(false);
      var w = out.querySelector(".audit-wait");
      if (w) w.remove();
      var a = $("aq-send") && $("aq-send").parentNode;
      var msg = "We could not send the answers: " + plain(err) + ".";
      if (a) a.insertAdjacentHTML("beforebegin", '<p class="aq-warn banner banner-error">' + esc(msg) + "</p>");
      else message(msg);
    });
  }

  function wireFollowup() {
    var send = $("fu-send"), inp = $("fu-input");
    if (send) send.addEventListener("click", sendFollowup);
    if (inp) inp.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) { e.preventDefault(); sendFollowup(); }
    });
  }

  function followupStatus(html) {
    var st = $("fu-status");
    if (st) st.innerHTML = html || "";
  }

  // busy state for the follow-up only: the current report stays on screen while Opus works
  function setFollowBusy(on) {
    S.busy = on;
    var send = $("fu-send"), inp = $("fu-input");
    [btn, $("aq-send"), send].forEach(function (b) {
      if (!b) return;
      b.disabled = on;
      b.classList.toggle("busy", on);
    });
    if (send) send.querySelector(".btn-text").textContent = on ? "Opus is working…" : "Ask Opus";
    if (inp) inp.disabled = on;
    var card = $("audit-followup");
    if (card) card.setAttribute("aria-busy", on ? "true" : "false");
    if (S.timer) { clearInterval(S.timer); S.timer = null; }
    if (!on) { followupStatus(""); return; }
    var t0 = Date.now();
    followupStatus('<div class="audit-wait fu-wait" role="status"><span class="spinner" aria-hidden="true"></span><div>' +
      '<p class="small"><strong>Opus is working on your question.</strong> It may run more checks before it answers; this often takes a minute or more. Please keep this page open.</p>' +
      '<p class="small muted">Waiting <span class="audit-elapsed" id="fu-elapsed">0 s</span></p></div></div>');
    S.timer = setInterval(function () {
      var el = $("fu-elapsed");
      if (!el) return;
      var s = Math.round((Date.now() - t0) / 1000);
      el.textContent = s < 60 ? s + " s" : Math.floor(s / 60) + " min " + (s % 60) + " s";
    }, 1000);
  }

  function sendFollowup() {
    if (S.busy || !S.report) return;
    var inp = $("fu-input");
    var question = inp ? inp.value.trim() : "";
    if (!question) {
      followupStatus('<p class="banner banner-warn">Please write a question first.</p>');
      if (inp) inp.focus();
      return;
    }
    var sid = S.report.session_id;
    if (!sid || sid === "sample") {
      followupStatus('<p class="banner banner-warn">This is a bundled sample, not a live session, so the question cannot be sent. Upload a file and run an audit to ask for real.</p>');
      return;
    }
    setFollowBusy(true);
    fetch("/api/audit/" + encodeURIComponent(sid) + "/followup", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: question })
    }).then(readJson).then(function (res) {
      setFollowBusy(false);
      var b = res.body;
      var rep = obj(b) && obj(b.report) ? b.report : b;
      if (obj(rep) && rep.status === "error") {
        // keep the verdict already on screen: it still stands; only the follow-up failed
        throw new Error((rep.error || "the follow-up stopped without saying why").replace(/\.+$/, ""));
      }
      handle(res);
      var box = $("audit-followup");
      if (box) box.scrollIntoView({ behavior: "smooth", block: "start" });
    }).catch(function (err) {
      setFollowBusy(false);
      followupStatus('<p class="banner banner-error" role="alert">' + esc("We could not get an answer to the follow-up: " + plain(err) +
        ". The verdict above is unchanged and your question is kept, so you can send it again.") + "</p>");
    });
  }

  function loadSample() {
    setMode("audit");
    fetch("sample_audit.json").then(function (r) { if (!r.ok) throw new Error("status " + r.status); return r.json(); })
      .then(function (rep) { show(rep, "Showing the bundled sample audit, not a live run."); })
      .catch(function (err) { message("Could not load the sample (" + plain(err) + "). Serve this folder over http, for example with python -m http.server."); });
  }

  // ------------------------------------------------------------------ mode switch
  function getMode() { return document.body.classList.contains("audit-mode") ? "audit" : "analyze"; }
  function setMode(m) {
    var audit = m === "audit";
    document.body.classList.toggle("audit-mode", audit);
    Array.prototype.forEach.call(document.querySelectorAll('input[name="mode"]'), function (r) { r.checked = r.value === (audit ? "audit" : "analyze"); });
    var claim = $("claim");
    if (claim) claim.placeholder = audit ? "Joining the program increased weekly sales" : "X increases Y by 15% among Z";
    try { localStorage.setItem("causality-mode", audit ? "audit" : "analyze"); } catch (e) { /* storage may be blocked */ }
  }

  Array.prototype.forEach.call(document.querySelectorAll('input[name="mode"]'), function (r) {
    r.addEventListener("change", function () { if (r.checked) setMode(r.value); });
  });
  // runs before app.js's own submit handler on the form; in audit mode it takes over
  document.addEventListener("submit", function (e) {
    if (e.target !== form || getMode() !== "audit") return;
    e.preventDefault();
    e.stopPropagation();
    runAudit();
  }, true);
  var sampleLink = $("audit-sample");
  if (sampleLink) sampleLink.addEventListener("click", function (e) { e.preventDefault(); loadSample(); });

  var start = "analyze";
  try { start = localStorage.getItem("causality-mode") || "analyze"; } catch (e) { start = "analyze"; }
  if (/[?&]mode=audit\b/.test(location.search)) start = "audit";
  if (/[?&]sample=(audit|explore)\b/.test(location.search)) start = "analyze";
  setMode(start);
  if (/[?&]sample=opus\b/.test(location.search)) loadSample();

  // dev: CausalAudit.show(report) renders any report object; CausalAudit.render(report) returns HTML
  window.CausalAudit = { show: show, render: render, esc: esc, loadSample: loadSample, setMode: setMode };
})();
