/* eslint-disable */
/* CausalityAI charts: hand-written SVG, no dependencies.
 *
 * Everything lives inside defineCharts() so the HTML export can inline this exact code
 * with defineCharts.toString(). Each chart function takes a Chart object
 * ({kind, title, data, x_label, y_label, note}) and returns an SVG string.
 * Colors come from CSS variables (--s1..--s6, --fg, --muted, --grid, --ref, --bg-card).
 * Null / missing numbers are skipped; empty data gives a small "nothing to draw" box.
 */
function defineCharts() {
  "use strict";
  var uid = 0;
  var W = 560;          // viewBox width; the svg scales to its container
  var FONT = 12;

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function num(v) { return typeof v === "number" && isFinite(v); }
  function fmt(v) {
    if (!num(v)) return "–";
    if (v === 0) return "0";
    var a = Math.abs(v);
    if (a >= 1e6 || a < 1e-3) return v.toExponential(2);
    return String(+v.toPrecision(Math.max(3, Math.floor(Math.log(a) / Math.LN10) + 1)));
  }
  function arr(v) { return Array.isArray(v) ? v : []; }
  function clip(s, n) { s = String(s == null ? "" : s); return s.length > n ? s.slice(0, n - 1) + "…" : s; }
  function color(i) { return "var(--s" + ((i % 6) + 1) + ")"; }

  function extent(vals) {
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < vals.length; i++) {
      var v = vals[i];
      if (num(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
    }
    return lo === Infinity ? null : [lo, hi];
  }
  function padded(e, frac) {
    var lo = e[0], hi = e[1];
    if (lo === hi) { var d = Math.abs(lo) * 0.1 || 1; return [lo - d, hi + d]; }
    var p = (hi - lo) * (frac == null ? 0.05 : frac);
    return [lo - p, hi + p];
  }
  function ticks(lo, hi, n) {
    if (!num(lo) || !num(hi) || hi <= lo) return [];
    n = n || 5;
    var span = hi - lo;
    var step = Math.pow(10, Math.floor(Math.log(span / n) / Math.LN10));
    var err = span / n / step;
    if (err >= 7.5) step *= 10; else if (err >= 3.5) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [];
    for (var t = Math.ceil(lo / step) * step; t <= hi + step * 1e-9; t += step) {
      out.push(+t.toPrecision(12));
      if (out.length > 50) break;
    }
    return out;
  }
  function lin(d0, d1, r0, r1) {
    var k = (d1 - d0) === 0 ? 0 : (r1 - r0) / (d1 - d0);
    return function (v) { return r0 + (v - d0) * k; };
  }
  function r(v) { return Math.round(v * 10) / 10; }

  function open(h, chart, extraClass) {
    var id = "ch" + (++uid);
    var title = chart && chart.title ? chart.title : (chart && chart.kind ? chart.kind + " chart" : "chart");
    return '<svg class="chart ' + (extraClass || "") + '" viewBox="0 0 ' + W + " " + r(h) +
      '" role="img" aria-labelledby="' + id + '" preserveAspectRatio="xMidYMid meet" font-size="' + FONT + '">' +
      '<title id="' + id + '">' + esc(title) + "</title>";
  }
  function empty(chart, msg) {
    return open(60, chart) +
      '<rect x="0.5" y="0.5" width="' + (W - 1) + '" height="59" rx="6" style="fill:none;stroke:var(--grid);stroke-dasharray:4 3"/>' +
      '<text x="' + W / 2 + '" y="34" text-anchor="middle" style="fill:var(--muted)">' + esc(msg || "No data to draw") + "</text></svg>";
  }
  function text(x, y, s, anchor, style, extra) {
    return '<text x="' + r(x) + '" y="' + r(y) + '"' + (anchor ? ' text-anchor="' + anchor + '"' : "") +
      ' style="' + (style || "fill:var(--muted)") + '"' + (extra || "") + ">" + esc(s) + "</text>";
  }
  function line(x1, y1, x2, y2, style) {
    return '<line x1="' + r(x1) + '" y1="' + r(y1) + '" x2="' + r(x2) + '" y2="' + r(y2) + '" style="' + style + '"/>';
  }

  // Axes + grid. box = {l, r, t, b} pixel bounds. xs/ys = scales; xt/yt = tick arrays.
  function axes(box, xs, ys, xt, yt, xlabel, ylabel) {
    var s = "";
    (yt || []).forEach(function (t) {
      var y = ys(t);
      s += line(box.l, y, box.r, y, "stroke:var(--grid);stroke-width:1");
      s += text(box.l - 6, y + 4, fmt(t), "end");
    });
    (xt || []).forEach(function (t) {
      var x = xs(t);
      s += line(x, box.t, x, box.b, "stroke:var(--grid);stroke-width:1");
      s += text(x, box.b + 15, fmt(t), "middle");
    });
    s += line(box.l, box.b, box.r, box.b, "stroke:var(--axis);stroke-width:1");
    s += line(box.l, box.t, box.l, box.b, "stroke:var(--axis);stroke-width:1");
    if (xlabel) s += text((box.l + box.r) / 2, box.b + 32, clip(xlabel, 70), "middle", "fill:var(--fg)");
    if (ylabel) s += text(14, (box.t + box.b) / 2, clip(ylabel, 40), "middle", "fill:var(--fg)",
      ' transform="rotate(-90 14 ' + r((box.t + box.b) / 2) + ')"');
    return s;
  }
  function legend(names, x, y) {
    var s = "", cx = x;
    names.forEach(function (n, i) {
      var label = clip(n, 24);
      var w = 18 + label.length * 6.6 + 12;
      if (cx + w > W - 8 && cx > x) { cx = x; y += 16; }
      s += '<rect x="' + r(cx) + '" y="' + r(y - 9) + '" width="10" height="10" rx="2" style="fill:' + color(i) + '"/>';
      s += text(cx + 14, y, label, "start", "fill:var(--fg)");
      cx += w;
    });
    return { svg: s, bottom: y + 8 };
  }
  function labelWidth(labels) {
    var m = 0;
    labels.forEach(function (l) { m = Math.max(m, String(l == null ? "" : l).length); });
    return Math.min(190, Math.max(50, Math.min(m, 28) * 6.6 + 14));
  }

  // ---------------------------------------------------------------------- bar
  // data = {labels, values, ref, errors, ref_band?: [lo, hi]}  -- horizontal bars.
  function bar(chart) {
    var d = chart.data || {};
    var labels = arr(d.labels), vals = arr(d.values), errs = Array.isArray(d.errors) ? d.errors : null;
    var rows = [];
    var n = Math.max(labels.length, vals.length);
    for (var i = 0; i < n; i++) rows.push({ label: labels[i] != null ? labels[i] : "", v: vals[i], e: errs ? errs[i] : null });
    if (!rows.some(function (x) { return num(x.v); })) return empty(chart);
    var band = Array.isArray(d.ref_band) && num(d.ref_band[0]) && num(d.ref_band[1]) ? d.ref_band : null;
    var pool = [0];
    rows.forEach(function (x) {
      if (!num(x.v)) return;
      pool.push(x.v);
      if (num(x.e)) { pool.push(x.v - Math.abs(x.e)); pool.push(x.v + Math.abs(x.e)); }
    });
    if (num(d.ref)) pool.push(d.ref);
    if (band) { pool.push(band[0]); pool.push(band[1]); }
    var dom = padded(extent(pool), 0.08);
    var lw = labelWidth(rows.map(function (x) { return x.label; }));
    var rowH = 24, top = 12;
    var box = { l: lw + 8, r: W - 46, t: top, b: top + rows.length * rowH + 4 };
    var h = box.b + (chart.x_label ? 42 : 26);
    var xs = lin(dom[0], dom[1], box.l, box.r);
    var s = open(h, chart);
    if (band) s += '<rect x="' + r(xs(band[0])) + '" y="' + box.t + '" width="' + r(Math.max(0, xs(band[1]) - xs(band[0]))) +
      '" height="' + r(box.b - box.t) + '" style="fill:var(--axis);opacity:0.14"/>';
    s += axes(box, xs, null, ticks(dom[0], dom[1], 5), [], chart.x_label, "");
    var x0 = xs(0);
    rows.forEach(function (row, i) {
      var cy = box.t + i * rowH + rowH / 2;
      s += text(box.l - 8, cy + 4, clip(row.label, 28), "end", "fill:var(--fg)");
      if (!num(row.v)) { s += text(x0 + 4, cy + 4, "n/a", "start"); return; }
      var x1 = xs(row.v);
      s += '<rect x="' + r(Math.min(x0, x1)) + '" y="' + r(cy - 8) + '" width="' + r(Math.max(1, Math.abs(x1 - x0))) +
        '" height="16" rx="2" style="fill:var(--s1)"><title>' + esc(row.label + ": " + fmt(row.v)) + "</title></rect>";
      if (num(row.e)) {
        var a = xs(row.v - Math.abs(row.e)), b = xs(row.v + Math.abs(row.e));
        s += line(a, cy, b, cy, "stroke:var(--fg);stroke-width:1.5");
        s += line(a, cy - 5, a, cy + 5, "stroke:var(--fg);stroke-width:1.5");
        s += line(b, cy - 5, b, cy + 5, "stroke:var(--fg);stroke-width:1.5");
      }
      var ee = num(row.e) ? Math.abs(row.e) : 0;
      if (row.v >= 0) {
        var tx = xs(row.v + ee);
        s += text(Math.min(tx + 4, W - 4), cy + 4, fmt(row.v), tx + 40 > W ? "end" : "start", "fill:var(--fg)");
      } else {
        var tl = xs(row.v - ee);
        if (tl - 40 >= box.l) s += text(tl - 4, cy + 4, fmt(row.v), "end", "fill:var(--fg)");
        else if (x0 - x1 > 40) s += text(x1 + 4, cy + 4, fmt(row.v), "start", "fill:var(--bg-card)");
        else s += text(x0 + 4, cy + 4, fmt(row.v), "start", "fill:var(--fg)");
      }
    });
    if (num(d.ref)) {
      s += line(xs(d.ref), box.t - 4, xs(d.ref), box.b, "stroke:var(--ref);stroke-width:1.5;stroke-dasharray:5 4");
    }
    return s + "</svg>";
  }

  // ---------------------------------------------------------------------- hist
  // data = {bins: edges (n+1), series: [{name, counts}]}  -- overlaid, semi-transparent.
  function hist(chart) {
    var d = chart.data || {};
    var bins = arr(d.bins), series = arr(d.series).filter(function (s) { return s && Array.isArray(s.counts); });
    if (bins.length < 2 || !series.length) return empty(chart);
    var be = extent(bins);
    var maxc = 0;
    series.forEach(function (s) { s.counts.forEach(function (c) { if (num(c) && c > maxc) maxc = c; }); });
    if (!be || maxc <= 0) return empty(chart);
    var names = series.map(function (s, i) { return s.name || "series " + (i + 1); });
    var leg = legend(names, 56, 14);
    var box = { l: 52, r: W - 12, t: leg.bottom + 8, b: leg.bottom + 8 + 190 };
    var h = box.b + (chart.x_label ? 42 : 26);
    var xs = lin(be[0], be[1], box.l, box.r), ys = lin(0, maxc * 1.05, box.b, box.t);
    var s = open(h, chart);
    s += axes(box, xs, ys, ticks(be[0], be[1], 6), ticks(0, maxc * 1.05, 4), chart.x_label, chart.y_label || "count");
    series.forEach(function (ser, si) {
      for (var i = 0; i < bins.length - 1; i++) {
        var c = ser.counts[i];
        if (!num(c) || !num(bins[i]) || !num(bins[i + 1]) || c <= 0) continue;
        var x = xs(bins[i]), x2 = xs(bins[i + 1]), y = ys(c);
        s += '<rect x="' + r(x) + '" y="' + r(y) + '" width="' + r(Math.max(0.5, x2 - x - 0.5)) + '" height="' + r(box.b - y) +
          '" style="fill:' + color(si) + ';fill-opacity:0.45;stroke:' + color(si) + ';stroke-width:0.8"><title>' +
          esc(names[si] + ": " + fmt(bins[i]) + " to " + fmt(bins[i + 1]) + " – " + c) + "</title></rect>";
      }
    });
    s += leg.svg;
    return s + "</svg>";
  }

  // ---------------------------------------------------------------------- scatter
  // data = {points: [{x, y, group}], lines: [{x: [], y: [], name}]}
  function scatter(chart) {
    var d = chart.data || {};
    var pts = arr(d.points).filter(function (p) { return p && num(p.x) && num(p.y); });
    var lines = arr(d.lines).filter(function (l) { return l && Array.isArray(l.x) && Array.isArray(l.y); });
    var xsAll = pts.map(function (p) { return p.x; }), ysAll = pts.map(function (p) { return p.y; });
    lines.forEach(function (l) { xsAll = xsAll.concat(l.x); ysAll = ysAll.concat(l.y); });
    var ex = extent(xsAll), ey = extent(ysAll);
    if (!ex || !ey) return empty(chart);
    var groups = [];
    pts.forEach(function (p) { var g = p.group == null ? "" : String(p.group); if (groups.indexOf(g) < 0) groups.push(g); });
    var named = groups.filter(function (g) { return g !== ""; });
    var legendNames = named.concat(lines.map(function (l, i) { return l.name || "line " + (i + 1); }));
    var leg = legendNames.length ? legend(legendNames, 56, 14) : { svg: "", bottom: 4 };
    var dx = padded(ex), dy = padded(ey);
    var box = { l: 52, r: W - 12, t: leg.bottom + 8, b: leg.bottom + 8 + 220 };
    var h = box.b + (chart.x_label ? 42 : 26);
    var xs = lin(dx[0], dx[1], box.l, box.r), ys = lin(dy[0], dy[1], box.b, box.t);
    var s = open(h, chart);
    s += axes(box, xs, ys, ticks(dx[0], dx[1], 6), ticks(dy[0], dy[1], 5), chart.x_label, chart.y_label);
    pts.forEach(function (p) {
      var g = p.group == null ? "" : String(p.group);
      var ci = g === "" ? 0 : named.indexOf(g);
      s += '<circle cx="' + r(xs(p.x)) + '" cy="' + r(ys(p.y)) + '" r="3" style="fill:' + color(ci) + ';fill-opacity:0.55"/>';
    });
    lines.forEach(function (l, li) {
      var ci = named.length + li;
      s += pathSegments(l.x, l.y, xs, ys).map(function (pd) {
        return '<path d="' + pd + '" style="fill:none;stroke:' + color(ci) + ';stroke-width:2.2"/>';
      }).join("");
    });
    s += leg.svg;
    return s + "</svg>";
  }

  function pathSegments(x, y, xs, ys) {
    var segs = [], cur = "";
    for (var i = 0; i < Math.min(x.length, y.length); i++) {
      if (num(x[i]) && num(y[i])) {
        cur += (cur ? "L" : "M") + r(xs(x[i])) + " " + r(ys(y[i]));
      } else if (cur) { segs.push(cur); cur = ""; }
    }
    if (cur) segs.push(cur);
    return segs;
  }

  // ---------------------------------------------------------------------- line
  // data = {series: [{name, x, y, lo, hi}], vline, hline}
  function lineChart(chart) {
    var d = chart.data || {};
    var series = arr(d.series).filter(function (s) { return s && Array.isArray(s.x) && Array.isArray(s.y); });
    var xsAll = [], ysAll = [];
    series.forEach(function (s) {
      xsAll = xsAll.concat(s.x); ysAll = ysAll.concat(s.y);
      if (Array.isArray(s.lo)) ysAll = ysAll.concat(s.lo);
      if (Array.isArray(s.hi)) ysAll = ysAll.concat(s.hi);
    });
    if (num(d.vline)) xsAll.push(d.vline);
    if (num(d.hline)) ysAll.push(d.hline);
    var ex = extent(xsAll), ey = extent(ysAll);
    if (!ex || !ey || !series.length) return empty(chart);
    var names = series.map(function (s, i) { return s.name || "series " + (i + 1); });
    var leg = series.length > 1 || series[0].name ? legend(names, 56, 14) : { svg: "", bottom: 4 };
    var dx = padded(ex, 0.03), dy = padded(ey);
    var box = { l: 52, r: W - 12, t: leg.bottom + 8, b: leg.bottom + 8 + 210 };
    var h = box.b + (chart.x_label ? 42 : 26);
    var xs = lin(dx[0], dx[1], box.l, box.r), ys = lin(dy[0], dy[1], box.b, box.t);
    var s = open(h, chart);
    s += axes(box, xs, ys, ticks(dx[0], dx[1], 6), ticks(dy[0], dy[1], 5), chart.x_label, chart.y_label);
    series.forEach(function (ser, si) {
      if (Array.isArray(ser.lo) && Array.isArray(ser.hi)) {
        // ribbon, broken at nulls
        var seg = [];
        var flush = function () {
          if (seg.length > 1) {
            var top = seg.map(function (p) { return r(xs(p[0])) + " " + r(ys(p[2])); });
            var bot = seg.slice().reverse().map(function (p) { return r(xs(p[0])) + " " + r(ys(p[1])); });
            s += '<path d="M' + top.join("L") + "L" + bot.join("L") + 'Z" style="fill:' + color(si) + ';fill-opacity:0.18;stroke:none"/>';
          }
          seg = [];
        };
        for (var i = 0; i < ser.x.length; i++) {
          if (num(ser.x[i]) && num(ser.lo[i]) && num(ser.hi[i])) seg.push([ser.x[i], ser.lo[i], ser.hi[i]]);
          else flush();
        }
        flush();
      }
    });
    if (num(d.hline)) s += line(box.l, ys(d.hline), box.r, ys(d.hline), "stroke:var(--ref);stroke-width:1.5;stroke-dasharray:5 4");
    if (num(d.vline)) s += line(xs(d.vline), box.t, xs(d.vline), box.b, "stroke:var(--ref);stroke-width:1.5;stroke-dasharray:5 4");
    series.forEach(function (ser, si) {
      s += pathSegments(ser.x, ser.y, xs, ys).map(function (pd) {
        return '<path d="' + pd + '" style="fill:none;stroke:' + color(si) + ';stroke-width:2.2;stroke-linejoin:round"/>';
      }).join("");
      if (ser.x.length <= 30) {
        for (var i = 0; i < ser.x.length; i++) {
          if (num(ser.x[i]) && num(ser.y[i])) s += '<circle cx="' + r(xs(ser.x[i])) + '" cy="' + r(ys(ser.y[i])) +
            '" r="2.6" style="fill:' + color(si) + '"><title>' + esc(fmt(ser.x[i]) + ", " + fmt(ser.y[i])) + "</title></circle>";
        }
      }
    });
    s += leg.svg;
    return s + "</svg>";
  }

  // ---------------------------------------------------------------------- forest
  // data = {rows: [{label, est, lo, hi}], ref}
  function forest(chart) {
    var d = chart.data || {};
    var rows = arr(d.rows).filter(function (x) { return x; });
    var pool = [];
    rows.forEach(function (x) { pool.push(x.est, x.lo, x.hi); });
    if (num(d.ref)) pool.push(d.ref);
    var e = extent(pool);
    if (!rows.length || !e) return empty(chart);
    var dom = padded(e, 0.08);
    var lw = labelWidth(rows.map(function (x) { return x.label; }));
    var rowH = 26, top = 10;
    var box = { l: lw + 8, r: W - 132, t: top, b: top + rows.length * rowH + 4 };
    var h = box.b + (chart.x_label ? 42 : 26);
    var xs = lin(dom[0], dom[1], box.l, box.r);
    var s = open(h, chart);
    s += axes(box, xs, null, ticks(dom[0], dom[1], 5), [], chart.x_label, "");
    if (num(d.ref)) s += line(xs(d.ref), box.t - 4, xs(d.ref), box.b, "stroke:var(--ref);stroke-width:1.5;stroke-dasharray:5 4");
    rows.forEach(function (row, i) {
      var cy = box.t + i * rowH + rowH / 2;
      s += text(box.l - 8, cy + 4, clip(row.label, 28), "end", "fill:var(--fg)");
      if (num(row.lo) && num(row.hi)) {
        s += line(xs(row.lo), cy, xs(row.hi), cy, "stroke:var(--s1);stroke-width:2");
        s += line(xs(row.lo), cy - 5, xs(row.lo), cy + 5, "stroke:var(--s1);stroke-width:2");
        s += line(xs(row.hi), cy - 5, xs(row.hi), cy + 5, "stroke:var(--s1);stroke-width:2");
      }
      if (num(row.est)) s += '<rect x="' + r(xs(row.est) - 5) + '" y="' + r(cy - 5) + '" width="10" height="10" style="fill:var(--s1);stroke:var(--bg-card);stroke-width:1"/>';
      var lbl = num(row.est) ? fmt(row.est) + (num(row.lo) && num(row.hi) ? " [" + fmt(row.lo) + ", " + fmt(row.hi) + "]" : "") : "not estimated";
      s += text(box.r + 10, cy + 4, lbl, "start", "fill:var(--fg)");
    });
    return s + "</svg>";
  }

  // ---------------------------------------------------------------------- dag
  // data = {nodes: [{id, role}], edges: [{from, to, type, label, confidence}]}
  var TOP = ["confounder", "covariate", "hidden", "latent", "unobserved", "unmeasured", "common_cause", "time", "unit", "other", "variable", ""];
  var BOTTOM = ["collider", "post_treatment", "cluster", "strata", "saturation", "descendant", "selection"];
  function roleOf(n) {
    // map descriptive roles ("instrument (assumed)", "running variable", "post-treatment (excluded)",
    // "unmeasured confounder", "fixed effect") onto the layout keys
    var r = String(n.role || "").toLowerCase();
    if (/unmeasured|hidden|latent|unobserved/.test(r)) return "hidden";
    if (/post.?treatment/.test(r)) return "post_treatment";
    if (/^instrument/.test(r)) return "instrument";
    if (/^running/.test(r)) return "running";
    if (/^confounder/.test(r)) return "confounder";
    if (/fixed effect|^cluster/.test(r)) return "cluster";
    return r.replace(/\s*\(.*\)\s*$/, "").replace(/[\s-]+/g, "_");
  }
  function isHidden(role) { return /hidden|latent|unobserved|unmeasured/.test(role); }
  function nodeStyle(role) {
    if (role === "treatment") return "fill:var(--node-t);stroke:var(--s1)";
    if (role === "outcome") return "fill:var(--node-o);stroke:var(--s2)";
    if (isHidden(role)) return "fill:var(--bg-card);stroke:var(--muted);stroke-dasharray:4 3";
    if (role === "instrument" || role === "running") return "fill:var(--node-i);stroke:var(--s3)";
    if (role === "post_treatment" || role === "collider" || role === "mediator") return "fill:var(--node-p);stroke:var(--s4)";
    return "fill:var(--node-c);stroke:var(--axis)";
  }

  function dag(chart, opts) {
    opts = opts || {};
    var d = chart.data || {};
    var seen = {};
    var nodes = arr(d.nodes).filter(function (n) {
      if (!n || n.id == null || seen[n.id]) return false;
      seen[n.id] = true; return true;
    }).map(function (n) {
      var label = clip(n.label || n.id, 22);
      return { id: String(n.id), full: String(n.label || n.id), label: label, role: roleOf(n), w: Math.max(56, label.length * 7 + 18), h: 30 };
    });
    if (!nodes.length) return empty(chart, "No diagram to draw");
    var byId = {};
    nodes.forEach(function (n) { byId[n.id] = n; });
    var edges = arr(d.edges).filter(function (e) { return e && byId[String(e.from)] && byId[String(e.to)] && String(e.from) !== String(e.to); });

    var hasT = nodes.some(function (n) { return n.role === "treatment"; });
    var hasO = nodes.some(function (n) { return n.role === "outcome"; });
    var circular = opts.layout === "circle" || (opts.layout !== "layered" && !(hasT && hasO));
    var H;
    if (circular) {
      var n = nodes.length;
      var maxw = 0; nodes.forEach(function (x) { maxw = Math.max(maxw, x.w); });
      var rx = Math.max(60, W / 2 - maxw / 2 - 12);
      var ry = n <= 2 ? 40 : Math.max(90, Math.min(220, n * 18));
      H = 2 * ry + 70;
      nodes.forEach(function (x, i) {
        var a = -Math.PI / 2 + (2 * Math.PI * i) / n;
        x.x = W / 2 + (n === 1 ? 0 : rx * Math.cos(a));
        x.y = H / 2 + (n === 1 ? 0 : ry * Math.sin(a));
      });
    } else {
      var COL = { instrument: 0, running: 0, treatment: 1, mediator: 2, outcome: 3 };
      var mid = [], top = [], bot = [];
      nodes.forEach(function (x) {
        if (x.role in COL) mid.push(x);
        else if (BOTTOM.indexOf(x.role) >= 0) bot.push(x);
        else top.push(x);
      });
      var used = [];
      mid.forEach(function (x) { if (used.indexOf(COL[x.role]) < 0) used.push(COL[x.role]); });
      used.sort();
      var nc = used.length;
      var edgeMargin = 12;
      mid.forEach(function (x) { edgeMargin = Math.max(edgeMargin, x.w / 2 + 8); });
      var colX = function (c) {
        var k = used.indexOf(c);
        return nc <= 1 ? W / 2 : edgeMargin + (k * (W - 2 * edgeMargin)) / (nc - 1);
      };
      mid.forEach(function (x) { x.x = colX(COL[x.role]); });
      // order the top / bottom rows by where their neighbours sit, so edges cross less
      var nb = {};
      arr(d.edges).forEach(function (e) {
        if (!e) return;
        var f = String(e.from), t = String(e.to);
        (nb[f] = nb[f] || []).push(t); (nb[t] = nb[t] || []).push(f);
      });
      var bary = function (x) {
        var xs_ = (nb[x.id] || []).map(function (id) { return byId[id]; }).filter(function (m) { return m && m.x != null && mid.indexOf(m) >= 0; });
        if (!xs_.length) return W / 2;
        return xs_.reduce(function (a, m) { return a + m.x; }, 0) / xs_.length;
      };
      [top, bot].forEach(function (list) {
        list.forEach(function (x) { x.b = bary(x); });
        list.sort(function (a, b) { return a.b - b.b; });
      });
      var GAP = 16;
      var wrapRows = function (list) {
        var rows = [], cur = [], wsum = 0;
        list.forEach(function (x) {
          if (cur.length && wsum + GAP + x.w > W - 16) { rows.push(cur); cur = []; wsum = 0; }
          wsum += (cur.length ? GAP : 0) + x.w; cur.push(x);
        });
        if (cur.length) rows.push(cur);
        return rows;
      };
      var placeRows = function (rows, startY, gap) {
        rows.forEach(function (row, ri) {
          var total = row.reduce(function (a, x) { return a + x.w; }, 0) + GAP * (row.length - 1);
          var free = W - 16 - total;
          // spread items evenly when the row has room, otherwise pack tightly
          var g = row.length > 1 ? GAP + Math.min(free / (row.length - 1), 60) : 0;
          var used_ = total + (g - GAP) * (row.length - 1);
          var cx = (W - used_) / 2;
          row.forEach(function (x) { x.x = cx + x.w / 2; x.y = startY + ri * gap; cx += x.w + g; });
        });
      };
      var midCounts = {};
      mid.forEach(function (x) { midCounts[COL[x.role]] = (midCounts[COL[x.role]] || 0) + 1; });
      var midRows = 1;
      for (var k in midCounts) midRows = Math.max(midRows, midCounts[k]);
      var rowGap = 56, bandGap = 96;
      var y0 = 24;
      var topRows = wrapRows(top), botRows = wrapRows(bot);
      placeRows(topRows, y0, rowGap);
      var midY = topRows.length ? y0 + (topRows.length - 1) * rowGap + bandGap : y0;
      var midIdx = {};
      mid.forEach(function (x) {
        var c = COL[x.role];
        var k = midIdx[c] = (midIdx[c] || 0) + 1;
        x.y = midY + (k - 1) * rowGap;
      });
      var botY = midY + (midRows - 1) * rowGap + bandGap;
      placeRows(botRows, botY, rowGap);
      H = (botRows.length ? botY + (botRows.length - 1) * rowGap : midY + (midRows - 1) * rowGap) + 30;
    }

    var id = ++uid;
    var mk = function (name) { return "dag" + id + name; };
    var s = open(H, chart, "dag");
    s += "<defs>" +
      '<marker id="' + mk("a") + '" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">' +
      '<path d="M0 0L10 5L0 10z" style="fill:var(--edge)"/></marker>' +
      '<marker id="' + mk("o") + '" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" orient="auto">' +
      '<circle cx="5" cy="5" r="3.6" style="fill:var(--bg-card);stroke:var(--edge);stroke-width:1.6"/></marker>' +
      "</defs>";

    function boundary(n, tx, ty) {
      var dx = tx - n.x, dy = ty - n.y;
      if (dx === 0 && dy === 0) return [n.x, n.y];
      var hw = n.w / 2 + 3, hh = n.h / 2 + 3;
      var t = Math.min(dx === 0 ? Infinity : hw / Math.abs(dx), dy === 0 ? Infinity : hh / Math.abs(dy));
      return [n.x + dx * t, n.y + dy * t];
    }
    function hitsOther(a, b) {
      return nodes.some(function (n) {
        if (n === a || n === b) return false;
        // sample the segment
        for (var k = 1; k < 20; k++) {
          var t = k / 20, x = a.x + (b.x - a.x) * t, y = a.y + (b.y - a.y) * t;
          if (Math.abs(x - n.x) < n.w / 2 + 4 && Math.abs(y - n.y) < n.h / 2 + 4) return true;
        }
        return false;
      });
    }
    // pair counter so two edges between the same pair bend apart
    var pairCount = {};
    var edgeSvg = "", labelSvg = "";
    edges.forEach(function (e) {
      var a = byId[String(e.from)], b = byId[String(e.to)];
      var type = String(e.type || "directed");
      var key = [a.id, b.id].sort().join("\u0000");
      var k = pairCount[key] = (pairCount[key] || 0) + 1;
      var bend = 0;
      if (type === "bidirected") bend = 34;
      if (hitsOther(a, b)) bend = bend || 46;
      if (k > 1) bend = bend + 26 * (k - 1);
      var mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
      var dx = b.x - a.x, dy = b.y - a.y, len = Math.sqrt(dx * dx + dy * dy) || 1;
      // bend away from the picture centre when possible
      var nx = -dy / len, ny = dx / len;
      if (bend && (mx + nx - W / 2) * nx + (my + ny - H / 2) * ny < 0) { nx = -nx; ny = -ny; }
      var cx = mx + nx * bend, cy = my + ny * bend;
      var p0 = boundary(a, bend ? cx : b.x, bend ? cy : b.y);
      var p1 = boundary(b, bend ? cx : a.x, bend ? cy : a.y);
      var dpath = bend ? "M" + r(p0[0]) + " " + r(p0[1]) + "Q" + r(cx) + " " + r(cy) + " " + r(p1[0]) + " " + r(p1[1])
        : "M" + r(p0[0]) + " " + r(p0[1]) + "L" + r(p1[0]) + " " + r(p1[1]);
      var start = "", end = "", dash = "";
      if (type === "directed") end = mk("a");
      else if (type === "bidirected") { start = mk("a"); end = mk("a"); dash = ";stroke-dasharray:6 4"; }
      else if (type === "partial") { start = mk("o"); end = mk("a"); }
      else { start = mk("o"); end = mk("o"); } // undirected
      var conf = num(e.confidence) ? e.confidence : null;
      var width = conf != null ? 1.2 + 1.6 * Math.max(0, Math.min(1, conf)) : 1.8;
      var tip = a.full + " " + ({ directed: "→", bidirected: "↔", partial: "o→", undirected: "o–o" }[type] || "–") + " " + b.full +
        (e.label ? " (" + e.label + ")" : "") + (conf != null ? ", confidence " + fmt(conf) : "");
      edgeSvg += '<path d="' + dpath + '" style="fill:none;stroke:var(--edge);stroke-width:' + r(width) + dash + '"' +
        (start ? ' marker-start="url(#' + start + ')"' : "") + (end ? ' marker-end="url(#' + end + ')"' : "") +
        "><title>" + esc(tip) + "</title></path>";
      var lbl = e.label ? clip(e.label, 18) : "";
      if (conf != null) lbl = (lbl ? lbl + " " : "") + fmt(conf);
      if (lbl) {
        var lx = bend ? 0.25 * p0[0] + 0.5 * cx + 0.25 * p1[0] : (p0[0] + p1[0]) / 2;
        var ly = bend ? 0.25 * p0[1] + 0.5 * cy + 0.25 * p1[1] : (p0[1] + p1[1]) / 2;
        labelSvg += text(lx, ly - 3, lbl, "middle", "fill:var(--muted);font-size:11px;paint-order:stroke;stroke:var(--bg-card);stroke-width:4px;stroke-linejoin:round");
      }
    });
    s += edgeSvg + labelSvg;
    nodes.forEach(function (n) {
      s += '<g><title>' + esc(n.full + (n.role ? " (" + n.role.replace(/_/g, " ") + ")" : "")) + "</title>" +
        '<rect x="' + r(n.x - n.w / 2) + '" y="' + r(n.y - n.h / 2) + '" width="' + r(n.w) + '" height="' + n.h +
        '" rx="8" style="' + nodeStyle(n.role) + ';stroke-width:1.6"/>' +
        text(n.x, n.y + 4, n.label, "middle", "fill:var(--fg)" + (n.role === "treatment" || n.role === "outcome" ? ";font-weight:600" : "")) +
        "</g>";
    });
    return s + "</svg>";
  }

  function render(chart, opts) {
    if (!chart || typeof chart !== "object") return "";
    try {
      switch (chart.kind) {
        case "bar": return bar(chart);
        case "hist": return hist(chart);
        case "scatter": return scatter(chart);
        case "line": return lineChart(chart);
        case "forest": return forest(chart);
        case "dag": return dag(chart, opts);
        default: return empty(chart, "Unknown chart kind: " + chart.kind);
      }
    } catch (err) {
      return empty(chart, "This chart could not be drawn");
    }
  }

  return { render: render, bar: bar, hist: hist, scatter: scatter, line: lineChart, forest: forest, dag: dag, fmt: fmt, esc: esc };
}

export { defineCharts };
const Charts = defineCharts();
export default Charts;
