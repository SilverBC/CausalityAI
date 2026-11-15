/* Force-directed canvas, Obsidian-like. Shared by the Analysis page and the Investigation page. */
import { useEffect, useRef, useState } from "react";
import ForceGraph2D from "react-force-graph-2d";
import { fmt } from "../../format.js";
import { ROLE_LABEL, useWidth, withAlpha } from "./theme.js";

const CURVE = 0.08; // link curvature, shared by the library's lines and our arrowheads
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* Pointer gestures, handled here instead of by the library (its drag starts on press, and any mouse movement while
   pressed cancels the click, so clicks on nodes were often lost or nudged the node):
   - click: press and release within TOLERANCE_PX, however long it takes. Selects the node (or clears on background).
   - hold to move: press a node and keep still for HOLD_MS; a ring fills while you hold, then the node follows the
     pointer and stays pinned where it is dropped. Right-click unpins (library).
   - moving before the hold completes pans the canvas (library zoom/pan). */
const HOLD_MS = { mouse: 350, pen: 450, touch: 450 };
const TOLERANCE_PX = 6;
const PIN_PX = 17;                                                // the pin badge's size on screen
const pinOffset = (n, scale) => (n.r || 4) * 0.72 + (PIN_PX / scale) * 0.15;  // from the centre, up and right

/** "839 citations · peer-reviewed journal" for a scored source. */
export const sourceLine = (s) =>
  s.verified ? `${(s.citations ?? 0).toLocaleString()} citation${s.citations === 1 ? "" : "s"} · ${s.venue_kind}`
    : `not found in OpenAlex · ${s.venue_kind || "unverified"}`;

export default function GraphCanvas({
  data, colors, height = 460, highlightId = null, selectedId = null, onSelect = null, empty = null,
  pendingIds = null, queuedIds = null, names = null,
}) {
  // display-only names (user renames): used for drawing; n.label stays the original that all logic uses
  const nameOf = (n) => (names && names[n.id]) || n.label;
  const wrapRef = useRef(null);
  const fgRef = useRef(null);
  const width = useWidth(wrapRef);
  const [hoverNode, setHoverNode] = useState(null);
  const gesture = useRef(null);            // the pointer press in progress
  const press = useRef(null);              // {id, t0, ms}: the hold ring being drawn
  const [mode, setMode] = useState(null);  // null | "press" | "drag": continuous redraw and the cursor
  const fitted = useRef(false);
  const pulsing = !!pendingIds?.size && !reducedMotion();
  const dataRef = useRef(data);
  dataRef.current = data;
  const onSelectRef = useRef(onSelect);
  onSelectRef.current = onSelect;

  // Hold-to-move only where nodes open details (Investigation page, phone graph); elsewhere the library's own
  // immediate drag is enough.
  const holdToMove = !!onSelect;
  // while a node is held it stays the hovered one: the library's hover test lags a frame behind the moving node and
  // would flip between it and nothing, dimming and undimming the whole graph (flashing)
  const dragged = mode === "drag" ? gesture.current?.node || null : null;
  const hover = dragged || hoverNode || (highlightId && data?.nodes.find((n) => n.id === highlightId)) || null;
  const hoverRef = useRef(null);
  hoverRef.current = hover;
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

  // clicks and hold-to-move (see the comment at the top)
  useEffect(() => {
    const el = wrapRef.current;
    if (!el || !holdToMove) return;
    const local = (ev) => {
      const r = el.querySelector("canvas")?.getBoundingClientRect();
      return r ? [ev.clientX - r.left, ev.clientY - r.top] : [0, 0];
    };
    const nodeAt = (ev) => {
      const fg = fgRef.current;
      if (!fg || !dataRef.current) return null;
      const { x, y } = fg.screen2GraphCoords(...local(ev));
      const slop = (ev.pointerType === "mouse" ? 3 : 10) / (fg.zoom() || 1); // fingers are less precise than a mouse
      let best = null, bd = Infinity;
      for (const n of dataRef.current.nodes) {
        if (n.x == null) continue;
        const d = Math.hypot(n.x - x, n.y - y);
        if (d <= (n.r || 4) + 3 + slop && d < bd) { best = n; bd = d; }
      }
      return best;
    };
    // the pin badge of a pinned node under the pointer (clicking it releases the node; phones have no right-click)
    const pinAt = (ev) => {
      const fg = fgRef.current;
      if (!fg || !dataRef.current) return null;
      const [px, py] = local(ev), k = fg.zoom() || 1, hit = PIN_PX * 0.62 + (ev.pointerType === "mouse" ? 2 : 8);
      return dataRef.current.nodes.find((n) => {
        if (n.fx == null || n.x == null) return false;
        const off = pinOffset(n, k), p = fg.graph2ScreenCoords(n.x + off, n.y - off);
        return Math.hypot(p.x - px, p.y - py) <= hit;
      }) || null;
    };
    const reset = () => {
      const g = gesture.current;
      if (g) clearTimeout(g.timer);
      gesture.current = null;
      press.current = null;
      setMode(null);
    };
    const startDrag = (g) => {
      if (gesture.current !== g || g.moved) return;
      g.dragging = true;
      g.wasPinned = g.node.fx != null;
      g.node.fx = g.node.x;
      g.node.fy = g.node.y;
      g.lastHeat = performance.now();
      press.current = null;
      setMode("drag");
      fgRef.current?.d3ReheatSimulation();
      navigator.vibrate?.(8);
    };
    const down = (ev) => {
      if (ev.button !== 0 || !ev.isPrimary) return;
      reset();
      const pin = pinAt(ev);
      const node = pin ? null : nodeAt(ev);   // on a pin badge: a click releases, no hold
      const g = { id: ev.pointerId, x0: ev.clientX, y0: ev.clientY, node, pin, moved: false, dragging: false, dragMoved: false };
      gesture.current = g;
      if (node) {
        const ms = HOLD_MS[ev.pointerType] ?? HOLD_MS.mouse;
        press.current = { id: node.id, t0: performance.now(), ms };
        g.timer = setTimeout(() => startDrag(g), ms);
        setMode("press");
      }
    };
    const move = (ev) => {
      const g = gesture.current;
      if (!g || ev.pointerId !== g.id) return;
      const dist = Math.hypot(ev.clientX - g.x0, ev.clientY - g.y0);
      if (g.dragging) {
        if (dist > TOLERANCE_PX) g.dragMoved = true;
        // keep the node on the canvas: dropped past an edge, it would be lost until the next zoom-to-fit
        const r = el.getBoundingClientRect(), [px, py] = local(ev), m = 12;
        const { x, y } = fgRef.current.screen2GraphCoords(Math.min(Math.max(px, m), r.width - m), Math.min(Math.max(py, m), r.height - m));
        g.node.fx = g.node.x = x;
        g.node.fy = g.node.y = y;
        const now = performance.now();
        if (now - g.lastHeat > 400) { g.lastHeat = now; fgRef.current?.d3ReheatSimulation(); } // neighbours follow
      } else if (dist > TOLERANCE_PX && !g.moved) {
        g.moved = true; // a pan, not a click or a hold
        clearTimeout(g.timer);
        press.current = null;
        setMode(null);
      }
    };
    const up = (ev) => {
      const g = gesture.current;
      if (!g || ev.pointerId !== g.id) return;
      reset();
      if (g.dragging && g.dragMoved) { fgRef.current?.d3ReheatSimulation(); return; } // pinned where it was dropped
      if (g.dragging && !g.wasPinned) { g.node.fx = undefined; g.node.fy = undefined; } // a slow click: no pin
      if (g.pin && !g.moved) { // released: back into the layout
        g.pin.fx = undefined;
        g.pin.fy = undefined;
        fgRef.current?.d3ReheatSimulation();
        return;
      }
      if (!g.moved) onSelectRef.current?.(g.node || null);
    };
    // while a node is held, the library's pan must not move the canvas as well (d3-zoom listens to mouse/touch
    // events; these capture listeners on window run before it and swallow them)
    const swallow = (ev) => {
      if (!gesture.current?.dragging) return;
      ev.stopImmediatePropagation();
      if (ev.cancelable && ev.type === "touchmove") ev.preventDefault();
    };
    const noMenu = (ev) => { if (gesture.current && ev.pointerType !== "mouse") ev.preventDefault(); }; // long-press menu
    el.addEventListener("pointerdown", down, true);
    window.addEventListener("pointermove", move, true);
    window.addEventListener("pointerup", up, true);
    window.addEventListener("pointercancel", reset, true);
    window.addEventListener("mousemove", swallow, true);
    window.addEventListener("touchmove", swallow, { capture: true, passive: false });
    el.addEventListener("contextmenu", noMenu);
    return () => {
      reset();
      el.removeEventListener("pointerdown", down, true);
      window.removeEventListener("pointermove", move, true);
      window.removeEventListener("pointerup", up, true);
      window.removeEventListener("pointercancel", reset, true);
      window.removeEventListener("mousemove", swallow, true);
      window.removeEventListener("touchmove", swallow, { capture: true });
      el.removeEventListener("contextmenu", noMenu);
    };
  }, [holdToMove]);

  // for tests and debugging: where a node is on screen, whether it is pinned, and which node is highlighted
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    el.__graph = {
      screenOf: (id) => {
        const n = dataRef.current?.nodes.find((x) => x.id === id);
        const r = el.querySelector("canvas")?.getBoundingClientRect();
        if (!n || n.x == null || !r || !fgRef.current) return null;
        const fg = fgRef.current, p = fg.graph2ScreenCoords(n.x, n.y);
        const out = { x: r.left + p.x, y: r.top + p.y, pinned: n.fx != null };
        if (out.pinned) {
          const off = pinOffset(n, fg.zoom() || 1), q = fg.graph2ScreenCoords(n.x + off, n.y - off);
          out.pin = { x: r.left + q.x, y: r.top + q.y };
        }
        return out;
      },
      hoverId: () => hoverRef.current?.id ?? null,
    };
    return () => { delete el.__graph; };
  }, []);

  // the canvas repaints only on interaction: repaint when colours, names or queue state change
  useEffect(() => { fgRef.current?.refresh?.(); }, [colors, names, pendingIds, queuedIds, selectedId]);

  const ring = (n) => (n.found || n.suggested ? 4 : 1.5);
  const linkWidth = (l) => (l.hypothetical ? 1.4 : l.weight == null ? 1 : 0.8 + 9 * Math.min(1, Math.abs(l.weight)));
  const linkColor = (l, head = false) => {
    if (l.hypothetical) return withAlpha(colors.hypo, linkActive(l) ? (hover || head ? 0.95 : 0.7) : 0.08);
    const base = l.weight == null ? colors.muted : !l.signed ? colors.neutral : l.weight >= 0 ? colors.pos : colors.neg;
    return withAlpha(base, linkActive(l) ? (hover ? 0.95 : head ? 0.85 : 0.6) : 0.1);
  };

  function circle(ctx, n, r) {
    ctx.beginPath();
    ctx.arc(n.x, n.y, r, 0, 2 * Math.PI);
  }

  function drawNode(n, ctx, scale) {
    const col = n.hypothetical ? colors.hypo : colors.role[n.role] || colors.node;
    ctx.globalAlpha = active(n) ? 1 : 0.18;

    if (pendingIds?.has(n.id)) { // being investigated: a pulsing halo (a still ring with reduced motion)
      const t = reducedMotion() ? 0.35 : (Date.now() % 1400) / 1400;
      circle(ctx, n, n.r + 4 + t * 10);
      ctx.strokeStyle = withAlpha(colors.hypo, 0.8 * (1 - t));
      ctx.lineWidth = 2.5;
      ctx.stroke();
    } else if (queuedIds?.has(n.id)) { // waiting in the queue: a quiet dashed ring
      circle(ctx, n, n.r + 4);
      ctx.setLineDash([2, 2.5]);
      ctx.lineWidth = 1.4;
      ctx.strokeStyle = withAlpha(colors.hypo, 0.75);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    const held = gesture.current?.dragging && gesture.current.node === n;
    if (press.current?.id === n.id) { // holding: a ring fills up until the node can be moved
      const p = Math.min(1, (performance.now() - press.current.t0) / press.current.ms);
      ctx.beginPath();
      ctx.arc(n.x, n.y, n.r + 8, -Math.PI / 2, -Math.PI / 2 + 2 * Math.PI * p);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = withAlpha(colors.ink, 0.65);
      ctx.stroke();
    }
    if (held) { // lifted: a soft halo while it follows the pointer
      circle(ctx, n, n.r + 9);
      ctx.fillStyle = withAlpha(col, 0.18);
      ctx.fill();
    }
    if (hover?.id === n.id || selectedId === n.id) {
      circle(ctx, n, n.r + 5);
      ctx.fillStyle = withAlpha(col, 0.25);
      ctx.fill();
    }

    circle(ctx, n, n.r);
    if (n.hypothetical || n.observed === false) { // not in the data: hollow, dashed
      ctx.fillStyle = withAlpha(col, n.hypothetical ? 0.16 : 0.12);
      ctx.fill();
      ctx.setLineDash([3, 2]);
      ctx.lineWidth = 1.8;
      ctx.strokeStyle = col;
      ctx.stroke();
      ctx.setLineDash([]);
    } else {
      ctx.fillStyle = col;
      ctx.fill();
    }
    if (n.found || n.suggested) { // found by the data search / pointed to by the literature
      circle(ctx, n, n.r + 3);
      ctx.lineWidth = 2;
      ctx.strokeStyle = n.found ? colors.accent : colors.hypo;
      ctx.stroke();
    }
    if (n.role === "outcome") {
      circle(ctx, n, Math.max(1, n.r - 3));
      ctx.lineWidth = 2;
      ctx.strokeStyle = colors.panel;
      ctx.stroke();
    }
    if (selectedId === n.id) {
      circle(ctx, n, n.r + 6);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = colors.ink;
      ctx.stroke();
    }

    if (n.fx != null && n.fy != null) pinBadge(ctx, n, scale);

    // labels fade in with zoom; always shown for key nodes and the hovered neighbourhood
    const key = n.role === "treatment" || n.role === "outcome" || n.found || n.literature || selectedId === n.id;
    if (hover ? active(n) : data.nodes.length <= 25 || scale > 0.9 || key) {
      const fs = Math.max(11 / scale, 3);
      ctx.font = `${hover?.id === n.id || selectedId === n.id ? 600 : 500} ${fs}px ui-sans-serif, system-ui, sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "top";
      ctx.fillStyle = n.hypothetical ? colors.hypo : colors.ink;
      ctx.fillText(nameOf(n), n.x, n.y + n.r + 3);
    }
    ctx.globalAlpha = 1;
  }

  /** A small push pin on the node's upper right: it stays where it was dropped (right-click releases it). Constant
      size on screen, whatever the zoom. */
  function pinBadge(ctx, n, scale) {
    const s = PIN_PX / scale;
    const off = pinOffset(n, scale);
    ctx.save();
    ctx.translate(n.x + off, n.y - off);
    ctx.beginPath();
    ctx.arc(0, 0, s * 0.62, 0, 2 * Math.PI);
    ctx.fillStyle = colors.panel;
    ctx.fill();
    ctx.lineWidth = Math.max(0.8 / scale, s * 0.07);
    ctx.strokeStyle = withAlpha(colors.ink, 0.35);
    ctx.stroke();
    ctx.rotate(Math.PI / 5);                                 // the classic tilt
    ctx.beginPath();                                         // head, collar and body
    ctx.moveTo(-0.15 * s, -0.4 * s);
    ctx.lineTo(0.15 * s, -0.4 * s);
    ctx.lineTo(0.11 * s, -0.1 * s);
    ctx.lineTo(0.24 * s, 0.0 * s);
    ctx.lineTo(-0.24 * s, 0.0 * s);
    ctx.lineTo(-0.11 * s, -0.1 * s);
    ctx.closePath();
    ctx.fillStyle = colors.ink;
    ctx.fill();
    ctx.beginPath();                                         // needle
    ctx.moveTo(0, 0);
    ctx.lineTo(0, 0.38 * s);
    ctx.lineWidth = s * 0.09;
    ctx.lineCap = "round";
    ctx.strokeStyle = colors.ink;
    ctx.stroke();
    ctx.restore();
  }

  function paintHitArea(n, color, ctx) {
    ctx.fillStyle = color;
    circle(ctx, n, n.r + 3);
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
    const stop = (t.r || 0) + ring(t);
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
    const lines = [`<b>${esc(nameOf(n))}</b> <span>${esc(ROLE_LABEL[n.role] || n.role)}</span>`];
    if (nameOf(n) !== n.label) lines.push(`<span>originally “${esc(n.label)}”</span>`);
    if (n.hypothetical) {
      lines.push("not in your data · suggested by published research");
      lines.push(n.reliability != null ? `<b>Reliability Score ${n.reliability}/100</b>` : "Reliability Score: not scored");
      if (n.bestSource) lines.push(esc(sourceLine(n.bestSource)));
    } else if (n.observed === false) {
      lines.push(n.role === "group" ? `${(n.members || []).length} covariates, drawn as one node` : "not measured in the data");
    } else {
      lines.push(`influence ${fmt(n.influence)}`);
      if (n.total_effect_on_outcome != null) lines.push(`total effect on the outcome ${fmt(n.total_effect_on_outcome)} SD`);
      if (n.suggested) lines.push("<i>also suggested by the literature</i>");
    }
    if (n.fx != null) lines.push(`<b>pinned</b> · ${holdToMove ? "click the pin or right-click" : "right-click"} to release`);
    if (n.depth != null) lines.push(n.depth === 0 ? "the outcome" : `${n.depth} step${n.depth === 1 ? "" : "s"} from the outcome`);
    if (n.found) lines.push("<i>found by the data search</i>");
    lines.push(`<span>${holdToMove ? "click for details · hold to move" : "drag to move"}</span>`);
    return `<div class="fg-tip">${lines.join("<br>")}</div>`;
  };
  const linkTooltip = (l) => {
    const head = `${esc(nameOf(l.source))} ${l.directed ? "→" : "—"} ${esc(nameOf(l.target))}`;
    if (l.hypothetical) return `<div class="fg-tip">${head}<br><i>hypothetical link from the literature</i><br>${esc((l.explanation || "").slice(0, 160))}</div>`;
    const what = l.kind === "latent" ? "hidden common cause (assumed)"
      : l.weight == null ? "strength not estimable"
      : l.signed ? `standardized effect ${fmt(l.weight)}` : `joint effect size ${fmt(l.weight)} (a category: no single sign)`;
    const how = l.found ? "found by the data search (direction assumed upstream)" : l.kind === "uncertain" ? "one end uncertain" : l.kind === "assumed" ? "assumed by the design" : "";
    return `<div class="fg-tip">${head}<br>${what}${how ? `<br><i>${how}</i>` : ""}</div>`;
  };

  const ok = data && data.nodes.length > 0;
  return (
    <div ref={wrapRef} className={`graph-canvas${mode === "drag" ? " dragging" : hoverNode ? " over-node" : ""}`}
      style={{ height: ok ? height : "auto" }}>
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
          linkLineDash={(l) => (l.hypothetical || l.kind === "latent" || l.kind === "uncertain" ? [4, 3] : null)}
          linkCanvasObjectMode={() => "after"}
          linkCanvasObject={drawArrow}
          linkDirectionalParticles={(l) => (hover && linkActive(l) && l.directed && !l.hypothetical && l.kind !== "latent" ? 2 : 0)}
          linkDirectionalParticleWidth={(l) => Math.max(2, linkWidth(l) * 0.6)}
          linkDirectionalParticleColor={() => colors.ink}
          linkCurvature={CURVE}
          autoPauseRedraw={!pulsing && !mode}
          enableNodeDrag={!holdToMove}
          onNodeDragEnd={holdToMove ? undefined : (n) => { n.fx = n.x; n.fy = n.y; }}
          onNodeHover={(n) => { if (!gesture.current?.dragging) setHoverNode(n || null); }}
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
