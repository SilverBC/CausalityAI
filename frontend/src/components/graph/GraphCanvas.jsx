/* Force-directed canvas, Obsidian-like. Shared by the Analysis page and (next stage) the Investigation page. */
import { useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { fmt } from "../../format.js";
import { ROLE_LABEL, useWidth, withAlpha } from "./theme.js";

const CURVE = 0.08; // link curvature, shared by the library's lines and our arrowheads
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

export default function GraphCanvas({ data, colors, height = 460, highlightId = null, selectedId = null, onSelect = null, empty = null }) {
  const wrapRef = useRef(null);
  const fgRef = useRef(null);
  const width = useWidth(wrapRef);
  const [hoverNode, setHoverNode] = useState(null);
  const fitted = useRef(false);

  const hover = hoverNode || (highlightId && data?.nodes.find((n) => n.id === highlightId)) || null;
  const active = (n) => !hover || hover.id === n.id || hover.neighbors.has(n.id);
  const linkActive = (l) => !hover || l.source.id === hover.id || l.target.id === hover.id;

  // forces, and a refit when the set of nodes changes
  const nodeCount = data?.nodes.length || 0;
  const lastCount = useRef(0);
  useEffect(() => {
    if (nodeCount !== lastCount.current) fitted.current = false;
    lastCount.current = nodeCount;
    const fg = fgRef.current;
    if (!fg || !data) return;
    fg.d3Force("charge")?.strength(-320);
    fg.d3Force("link")?.distance((l) => 70 + (l.source.r || 0) + (l.target.r || 0));
    fg.d3ReheatSimulation();
  }, [data, width > 0]); // eslint-disable-line react-hooks/exhaustive-deps

  // refit when the canvas height changes, once the layout has settled
  const lastHeight = useRef(height);
  useEffect(() => {
    if (lastHeight.current === height) return;
    lastHeight.current = height;
    if (!fitted.current) return;
    const t = setTimeout(() => fgRef.current?.zoomToFit(300, 30), 60);
    return () => clearTimeout(t);
  }, [height]);

  // the canvas only repaints on interaction: repaint once when the theme colours change
  useEffect(() => { fgRef.current?.refresh?.(); }, [colors]);

  const linkWidth = (l) => (l.weight == null ? 1 : 0.8 + 9 * Math.min(1, Math.abs(l.weight)));
  const linkColor = (l, head = false) => {
    const base = l.weight == null ? colors.muted : !l.signed ? colors.neutral : l.weight >= 0 ? colors.pos : colors.neg;
    return withAlpha(base, linkActive(l) ? (hover ? 0.95 : head ? 0.85 : 0.6) : 0.1);
  };

  function drawNode(n, ctx, scale) {
    const col = colors.role[n.role] || colors.node;
    ctx.globalAlpha = active(n) ? 1 : 0.18;
    if (hover?.id === n.id || selectedId === n.id) {
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 5, 0, 2 * Math.PI);
      ctx.fillStyle = withAlpha(col, 0.25);
      ctx.fill();
    }
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, 2 * Math.PI);
    if (n.observed !== false) {
      ctx.fillStyle = col;
      ctx.fill();
    } else { // unmeasured or grouped: hollow, dashed
      ctx.fillStyle = withAlpha(col, 0.12);
      ctx.fill();
      ctx.setLineDash([3, 2]);
      ctx.lineWidth = 1.6;
      ctx.strokeStyle = col;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (n.found) { // added by the data search
      ctx.lineWidth = 2;
      ctx.strokeStyle = colors.accent;
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 3, 0, 2 * Math.PI);
      ctx.stroke();
    }
    if (n.role === "outcome") {
      ctx.lineWidth = 2;
      ctx.strokeStyle = colors.panel;
      ctx.beginPath();
      ctx.arc(n.x, n.y, Math.max(1, n.r - 3), 0, 2 * Math.PI);
      ctx.stroke();
    }
    // labels fade in with zoom; always shown for key nodes and the hovered neighbourhood
    const key = n.role === "treatment" || n.role === "outcome" || n.found || selectedId === n.id;
    if (hover ? active(n) : data.nodes.length <= 25 || scale > 0.9 || key) {
      const fs = Math.max(11 / scale, 3);
      ctx.font = `${hover?.id === n.id ? 600 : 500} ${fs}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = colors.ink;
      ctx.fillText(n.label, n.x, n.y + n.r + 3);
    }
    ctx.globalAlpha = 1;
  }

  function paintHitArea(n, color, ctx) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r + 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  // Arrowheads drawn by hand: the library places its arrows by a default node size, which hides them under custom
  // sized nodes. Follow the same quadratic curve and put the tip just outside the target's circle (and ring).
  function drawArrow(l, ctx) {
    if (!l.directed) return; // undirected: direction unknown, so no arrowhead
    const s = l.source, t = l.target;
    if (s.x == null || t.x == null) return;
    const dx = t.x - s.x, dy = t.y - s.y, len = Math.hypot(dx, dy);
    if (len < 1e-6) return;
    const a = Math.atan2(dy, dx), d = len * CURVE;
    const cx = (s.x + t.x) / 2 + d * Math.cos(a - Math.PI / 2), cy = (s.y + t.y) / 2 + d * Math.sin(a - Math.PI / 2);
    const at = (u) => [(1 - u) ** 2 * s.x + 2 * (1 - u) * u * cx + u * u * t.x, (1 - u) ** 2 * s.y + 2 * (1 - u) * u * cy + u * u * t.y];
    const stop = (t.r || 0) + (t.found ? 4 : 1.5);
    if (len <= stop + (s.r || 0)) return;
    let lo = 0, hi = 1; // where the curve leaves the target's circle
    for (let i = 0; i < 18; i++) {
      const m = (lo + hi) / 2, [x, y] = at(m);
      if (Math.hypot(x - t.x, y - t.y) > stop) lo = m; else hi = m;
    }
    const [x, y] = at(lo);
    const tx = 2 * (1 - lo) * (cx - s.x) + 2 * lo * (t.x - cx), ty = 2 * (1 - lo) * (cy - s.y) + 2 * lo * (t.y - cy);
    const L = 5 + Math.min(7, linkWidth(l) * 0.55), W = L * 0.62;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(Math.atan2(ty, tx));
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(-L, W / 2);
    ctx.lineTo(-L * 0.78, 0);
    ctx.lineTo(-L, -W / 2);
    ctx.closePath();
    ctx.fillStyle = linkColor(l, true);
    ctx.fill();
    ctx.restore();
  }

  const nodeTooltip = (n) => {
    const lines = [`<b>${esc(n.label)}</b> <span>${esc(ROLE_LABEL[n.role] || n.role)}</span>`];
    if (n.observed === false) lines.push(n.role === "group" ? `${(n.members || []).length} covariates, drawn as one node` : "not measured in the data");
    else {
      lines.push(`influence ${fmt(n.influence)}`);
      if (n.total_effect_on_outcome != null) lines.push(`total effect on the outcome ${fmt(n.total_effect_on_outcome)} SD`);
    }
    if (n.depth != null) lines.push(n.depth === 0 ? "the outcome" : `${n.depth} step${n.depth === 1 ? "" : "s"} from the outcome`);
    if (n.found) lines.push("<i>found by the data search</i>");
    return `<div class="fg-tip">${lines.join("<br>")}</div>`;
  };
  const linkTooltip = (l) => {
    const what = l.kind === "latent" ? "hidden common cause (assumed)"
      : l.weight == null ? "strength not estimable"
      : l.signed ? `standardized effect ${fmt(l.weight)}` : `joint effect size ${fmt(l.weight)} (a category: no single sign)`;
    const how = l.found ? "found by the data search (direction assumed upstream)" : l.kind === "uncertain" ? "one end uncertain" : l.kind === "assumed" ? "assumed by the design" : "";
    return `<div class="fg-tip">${esc(l.source.label)} ${l.directed ? "→" : "—"} ${esc(l.target.label)}<br>${what}${how ? `<br><i>${how}</i>` : ""}</div>`;
  };

  const ok = data && data.nodes.length > 0;
  return (
    <div ref={wrapRef} className="graph-canvas" style={{ height: ok ? height : "auto" }}>
      {ok && width > 0 ? (
        <ForceGraph2D
          ref={fgRef}
          width={width}
          height={height}
          graphData={data}
          backgroundColor="rgba(0,0,0,0)"
          nodeId="id"
          nodeCanvasObject={drawNode}
          nodePointerAreaPaint={paintHitArea}
          nodeLabel={nodeTooltip}
          linkLabel={linkTooltip}
          linkColor={(l) => linkColor(l)}
          linkWidth={linkWidth}
          linkLineDash={(l) => (l.kind === "latent" || l.kind === "uncertain" ? [4, 3] : null)}
          linkCanvasObjectMode={() => "after"}
          linkCanvasObject={drawArrow}
          linkDirectionalParticles={(l) => (hover && linkActive(l) && l.directed && l.kind !== "latent" ? 2 : 0)}
          linkDirectionalParticleWidth={(l) => Math.max(2, linkWidth(l) * 0.6)}
          linkDirectionalParticleColor={() => colors.ink}
          linkCurvature={CURVE}
          onNodeHover={(n) => setHoverNode(n || null)}
          onNodeClick={onSelect ? (n) => onSelect(n) : undefined}
          onBackgroundClick={onSelect ? () => onSelect(null) : undefined}
          onNodeDragEnd={(n) => { n.fx = n.x; n.fy = n.y; }}
          onNodeRightClick={(n) => { n.fx = undefined; n.fy = undefined; fgRef.current?.d3ReheatSimulation(); }}
          onEngineStop={() => { if (!fitted.current) { fitted.current = true; fgRef.current?.zoomToFit(400, 50); } }}
          cooldownTicks={150}
        />
      ) : (
        <div className="graph-empty">{empty}</div>
      )}
    </div>
  );
}
