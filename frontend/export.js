/* CausalityAI exports.
 *   downloadJSON(report)  -- the report as a .json file.
 *   downloadHTML(report)  -- one standalone .html file that renders the same results offline:
 *       inlined style.css, defineCharts() and defineRender() (taken with Function#toString),
 *       and the report embedded in <script type="application/json">.
 */
(function () {
  function save(blob, name) {
    var a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); a.remove(); }, 1000);
  }

  function stamp() {
    var d = new Date();
    var p = function (n) { return (n < 10 ? "0" : "") + n; };
    return d.getFullYear() + p(d.getMonth() + 1) + p(d.getDate()) + "-" + p(d.getHours()) + p(d.getMinutes());
  }

  function downloadJSON(report) {
    save(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }), "causality-report-" + stamp() + ".json");
  }

  // CSS: read it from the loaded stylesheet; fall back to fetching the file.
  function getCss() {
    var css = "";
    try {
      Array.prototype.forEach.call(document.styleSheets, function (sh) {
        if (sh.href && !/style\.css(\?|$)/.test(sh.href)) return;
        Array.prototype.forEach.call(sh.cssRules || [], function (r) { css += r.cssText + "\n"; });
      });
    } catch (e) { css = ""; }
    if (css) return Promise.resolve(css);
    return fetch("style.css").then(function (r) { return r.ok ? r.text() : ""; }).catch(function () { return ""; });
  }

  // Keep "</script" and "<!--" out of inline script bodies.
  function safeScript(s) { return String(s).replace(/<\/(script)/gi, "<\\/$1").replace(/<!--/g, "<\\!--"); }

  function buildHTML(report, css) {
    var json = JSON.stringify(report).replace(/</g, "\\u003c")
      .replace(new RegExp(String.fromCharCode(0x2028), "g"), "\\u2028").replace(new RegExp(String.fromCharCode(0x2029), "g"), "\\u2029");
    var claim = report && report.claim && report.claim.raw ? report.claim.raw : "";
    var title = "CausalityAI report" + (claim ? " – " + claim.slice(0, 80) : "");
    var esc = function (s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); };
    var boot =
      "var CausalCharts = (" + defineCharts.toString() + ")();\n" +
      "var CausalRender = (" + defineRender.toString() + ")(CausalCharts);\n" +
      "(function(){var out=document.getElementById('results');try{" +
      "var rep=JSON.parse(document.getElementById('report-data').textContent);" +
      "out.innerHTML=CausalRender.render(rep,{interactive:false});" +
      "}catch(e){out.textContent='This report could not be shown: '+e.message;}})();\n";
    return "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n" +
      "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
      "<title>" + esc(title) + "</title>\n<style>\n" + css.replace(/<\/style/gi, "<\\/style") + "\n</style>\n</head>\n<body>\n" +
      "<header class=\"site-head\"><div class=\"wrap\"><h1 class=\"brand\">Causality<span>AI</span></h1>" +
      "<p class=\"tagline\">Saved report · exported " + esc(new Date().toLocaleString()) + "</p></div></header>\n" +
      "<main class=\"wrap\"><div id=\"results\"></div></main>\n" +
      "<script type=\"application/json\" id=\"report-data\">" + json + "</script>\n" +
      "<script>\n" + safeScript(boot) + "</script>\n</body>\n</html>\n";
  }

  function downloadHTML(report) {
    return getCss().then(function (css) {
      var html = buildHTML(report, css);
      save(new Blob([html], { type: "text/html" }), "causality-report-" + stamp() + ".html");
      return html;
    });
  }

  window.CausalExport = { downloadJSON: downloadJSON, downloadHTML: downloadHTML, buildHTML: buildHTML, getCss: getCss };
})();
