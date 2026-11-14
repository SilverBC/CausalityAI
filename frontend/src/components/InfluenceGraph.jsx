/* Analysis page: how the variables push on each other. Node size = influence, edge width = standardized effect. */
import { useMemo, useState } from "react";
import { fmt } from "../format.js";
import GraphCanvas from "./graph/GraphCanvas.jsx";
import { buildGraph } from "./graph/model.js";
import { ROLE_LABEL, useThemeColors } from "./graph/theme.js";

export function GraphLegend({ data, colors }) {
  const roles = [...new Set(data.nodes.map((n) => n.role))];
  const has = (f) => data.links.some(f);
  return (
    <div className="legend">
      {roles.map((r) => (
        <span key={r}><i className={data.nodes.some((n) => n.role === r && n.observed === false) ? "hollow" : ""}
          style={{ "--c": colors.role[r] || colors.node }} />{ROLE_LABEL[r] || r}</span>
      ))}
      <span><i style={{ "--c": colors.pos }} />positive effect</span>
      <span><i style={{ "--c": colors.neg }} />negative effect</span>
      {has((l) => l.weight != null && !l.signed) && <span><i style={{ "--c": colors.neutral }} />effect of a category (no sign)</span>}
      <span><b className="legend-arrow" aria-hidden="true">→</b>cause → effect</span>
      {has((l) => !l.directed) && <span><b className="legend-arrow" aria-hidden="true">—</b>no arrow: direction unknown</span>}
      {has((l) => l.kind === "latent" || l.kind === "uncertain") && <span><b className="legend-arrow" aria-hidden="true">┄</b>dashed: hidden cause or uncertain end</span>}
    </div>
  );
}

export default function InfluenceGraph({ result, running }) {
  const colors = useThemeColors();
  const [highlight, setHighlight] = useState(null);
  const inf = result?.influence;
  const ok = !!(inf && !inf.error && inf.nodes?.length);
  const data = useMemo(() => (ok ? buildGraph(inf) : null), [inf, ok]);
  const ranked = useMemo(() => (ok ? inf.nodes.filter((n) => n.role !== "outcome" && n.observed).sort((a, b) => b.influence - a.influence) : []), [inf, ok]);
  const maxInf = ranked[0]?.influence || 1;

  const empty = running ? "The graph appears when the analysis finishes."
    : inf?.error ? `The graph could not be built: ${inf.error}`
    : result ? (result.mode === "claim-only" ? "No graph without a dataset." : "No graph for this analysis.")
    : "Run an analysis to map how the variables influence each other.";

  return (
    <section className="card graph-card" aria-labelledby="graph-title">
      <div className="graph-head">
        <div>
          <h2 id="graph-title">Influence graph</h2>
          <p className="small muted">
            How the variables in your data push on each other. Bigger nodes move the rest more. Drag to pin a node, scroll
            to zoom, hover to focus.
          </p>
        </div>
        <div className="row">
          {ok && <span className="tag">{inf.source === "discovery" ? "discovered from the data" : "assumed diagram, weighted by the data"}</span>}
          {ok && <span className="btn sm disabled" aria-disabled="true" title="The Investigation page arrives in the next stage">Investigate further →</span>}
        </div>
      </div>

      <GraphCanvas data={data} colors={colors} highlightId={highlight} empty={empty} />

      {ok && (
        <div className="graph-foot">
          <GraphLegend data={data} colors={colors} />
          {ranked.length > 0 && (
            <div className="rank">
              <div className="small muted">Most influential</div>
              {ranked.slice(0, 6).map((n) => (
                <div key={n.id} className="rank-row" onMouseEnter={() => setHighlight(n.id)} onMouseLeave={() => setHighlight(null)}
                  onFocus={() => setHighlight(n.id)} onBlur={() => setHighlight(null)} tabIndex={0}>
                  <span className="rank-name">{n.label}</span>
                  <span className="rank-bar"><span style={{ width: `${(100 * n.influence) / maxInf}%`, background: colors.role[n.role] || colors.node }} /></span>
                  <span className="rank-val">{fmt(n.influence, 2)}</span>
                </div>
              ))}
            </div>
          )}
          <p className="small muted">
            Node size = influence: the sum of the standardized effects (in SD units) a variable has on the variables it points
            to. Edge width = that standardized effect, from regressing each variable on its parents. Right-click a pinned node
            to release it.
          </p>
        </div>
      )}
    </section>
  );
}
