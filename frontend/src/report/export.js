/* Downloads.
   downloadJSON(result)  the full result (report, graph, summary) as .json
   downloadHTML(report)  one standalone .html file that shows the report offline: the original stylesheet,
                         defineCharts() and defineRender() inlined with Function#toString, the report embedded as JSON. */
import standaloneCss from "./standalone.css?raw";
import { defineCharts, defineRender } from "./render.js";
import { download } from "../format.js";

function stamp() {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}
const slug = (s) => (s || "report").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "report";
// keep "</script" and "<!--" out of inline script bodies
const safeScript = (s) => String(s).replace(/<\/(script)/gi, "<\\/$1").replace(/<!--/g, "<\\!--");
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export function downloadJSON(result, title) {
  download(`bluff-or-buff-${slug(title)}-${stamp()}.json`, JSON.stringify(result, null, 2), "application/json");
}

export function buildHTML(report, title) {
  const LS = new RegExp(String.fromCharCode(0x2028), "g"), PS = new RegExp(String.fromCharCode(0x2029), "g");
  const json = JSON.stringify(report).replace(/</g, "\\u003c").replace(LS, "\\u2028").replace(PS, "\\u2029");
  const boot =
    `var CausalCharts = (${defineCharts.toString()})();\n` +
    `var CausalRender = (${defineRender.toString()})(CausalCharts);\n` +
    "(function(){var out=document.getElementById('results');try{" +
    "var rep=JSON.parse(document.getElementById('report-data').textContent);" +
    "out.innerHTML=CausalRender.render(rep,{interactive:false});" +
    "}catch(e){out.textContent='This report could not be shown: '+e.message;}})();\n";
  return "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n" +
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" +
    `<title>${esc(`Bluff-or-Buff report: ${title || "analysis"}`)}</title>\n` +
    `<style>\n${standaloneCss.replace(/<\/style/gi, "<\\/style")}\n</style>\n</head>\n<body>\n` +
    `<header class="site-head"><div class="wrap"><h1 class="brand">Causal <span>Audit</span></h1>` +
    `<p class="tagline">${esc(title || "")} · exported ${esc(new Date().toLocaleString())}</p></div></header>\n` +
    "<main class=\"wrap\"><div id=\"results\"></div></main>\n" +
    `<script type="application/json" id="report-data">${json}</script>\n` +
    `<script>\n${safeScript(boot)}</script>\n</body>\n</html>\n`;
}

export function downloadHTML(report, title) {
  download(`bluff-or-buff-${slug(title)}-${stamp()}.html`, buildHTML(report, title), "text/html");
}
