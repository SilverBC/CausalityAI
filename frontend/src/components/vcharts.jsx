/* SVG charts for the results tabs, drawn in the style of the previous app (Causality v1, components/charts.jsx):
   Love plot, mirrored propensity histogram, forest plot with square markers, pill-node DAGs. Colours come from the
   design tokens, so light and dark both work. Everything else still uses the original renderer (src/report/charts.js). */
import { fmt } from "../format.js";

// ------------------------------------------------------------------------------------------- helpers
export const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);

/** About `n` round tick values covering [lo, hi]. */
export function ticks(lo, hi, n = 5) {
  if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [lo];
  const raw = (hi - lo) / Math.max(1, n);
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw) || raw;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : +v.toPrecision(12));
  return out;
}

const Svg = ({ w, h, label, children }) => (
  <svg viewBox={`0 0 ${w} ${h}`} role="img" aria-label={label}>{children}</svg>
);

function AxisX({ sx, y, t, w0, w1, f = fmt }) {
  return (
    <>
      <line x1={w0} x2={w1} y1={y} y2={y} stroke="var(--line)" />
      {t.map((v, i) => (
        <g key={i}>
          <line x1={sx(v)} x2={sx(v)} y1={y} y2={y + 4} stroke="var(--line)" />
          <text x={sx(v)} y={y + 16} textAnchor="middle">{f(v)}</text>
        </g>
      ))}
    </>
  );
}

export function Legend({ items }) {
  return (
    <div className="legend">
      {items.filter(Boolean).map(([color, text, cls], i) => (
        <span key={i}>{color && <i className={cls || "sq"} style={{ background: color }} />}{text}</span>
      ))}
    </div>
  );
}

export const Chart = ({ children }) => <div className="vchart">{children}</div>;

// ------------------------------------------------------------------------------------------- balance
/** Standardized differences per covariate; the shaded band is what chance alone produces (`band`). */
export function LovePlot({ rows, band = 0.1 }) {
  rows = (rows || []).filter((r) => r.smd != null && isFinite(r.smd))
    .sort((a, b) => Math.abs(b.smd) - Math.abs(a.smd)).slice(0, 20);
  if (!rows.length) return null;
  const W = 440, rowH = 22, top = 10, left = 130, right = 14, H = top + rows.length * rowH + 34;
  const m = Math.max(0.3, band * 1.3, ...rows.map((r) => Math.abs(r.smd))) * 1.1;
  const sx = lin(-m, m, left, W - right);
  const bandH = rows.length * rowH;
  return (
    <Svg w={W} h={H} label="Standardized differences between the groups, per covariate">
      <rect x={sx(-band)} y={top} width={sx(band) - sx(-band)} height={bandH} fill="var(--pass-bg)" opacity=".75" />
      <line x1={sx(0)} x2={sx(0)} y1={top} y2={top + bandH} stroke="var(--muted)" />
      {rows.map((r, i) => {
        const y = top + i * rowH + rowH / 2;
        const bad = Math.abs(r.smd) > band;
        return (
          <g key={i}>
            <text x={left - 8} y={y + 4} textAnchor="end">{String(r.variable).slice(0, 20)}</text>
            <line x1={sx(0)} x2={sx(r.smd)} y1={y} y2={y} stroke="var(--line)" strokeWidth="2" />
            <circle cx={sx(r.smd)} cy={y} r="5" fill={bad ? "var(--fail)" : "var(--pass)"}>
              <title>{`${r.variable}: standardized difference ${fmt(r.smd)}${r.p != null ? `, p ${fmt(r.p)}` : ""}`}</title>
            </circle>
          </g>
        );
      })}
      <AxisX sx={sx} y={top + bandH + 4} t={ticks(-m, m, 4)} w0={left} w1={W - right} />
    </Svg>
  );
}

// ------------------------------------------------------------------------------------------- overlap
/** Propensity scores: treated bars up, control bars down, on a shared 0–1 axis. */
export function MirrorHist({ bins, treated, control }) {
  if (!treated?.length || !control?.length) return null;
  const W = 440, H = 230, L = 14, R = 14, mid = 105;
  const lo = bins?.[0] ?? 0, hi = bins?.[bins.length - 1] ?? 1;
  const mx = Math.max(1, ...treated, ...control);
  const n = treated.length, bw = (W - L - R) / n;
  const sy = lin(0, mx, 0, 88);
  const sx = lin(lo, hi, L, W - R);
  return (
    <Svg w={W} h={H + 6} label="Propensity score by group: treated above the line, control below">
      <line x1={L} x2={W - R} y1={mid} y2={mid} stroke="var(--muted)" />
      {treated.map((t, i) => {
        const x = L + i * bw;
        const c = control[i] || 0;
        return (
          <g key={i}>
            <rect x={x + 1} y={mid - sy(t)} width={Math.max(0, bw - 2)} height={sy(t)} fill="var(--treat)" opacity=".85"><title>{`treated ${t}`}</title></rect>
            <rect x={x + 1} y={mid} width={Math.max(0, bw - 2)} height={sy(c)} fill="var(--ctrl)" opacity=".85"><title>{`control ${c}`}</title></rect>
          </g>
        );
      })}
      <AxisX sx={sx} y={mid + 96} t={ticks(lo, hi, 5)} w0={L} w1={W - R} />
      <text x={L} y={12}>treated ↑</text>
      <text x={L} y={mid + 90}>control ↓</text>
    </Svg>
  );
}

// ------------------------------------------------------------------------------------------- estimates
/** Estimates with 95% intervals: main (--treat), other methods (--ctrl), zero line, claimed effect (--warn). */
export function Forest({ items, claim }) {
  items = items.filter((i) => i.estimate != null && isFinite(i.estimate));
  if (!items.length) return null;
  const hasClaim = claim != null && isFinite(claim);
  const W = 640, rowH = 30, L = 250, R = 20, T = 8, H = T + items.length * rowH + (hasClaim ? 40 : 28);
  const vals = items.flatMap((i) => (i.ci ? [i.ci[0], i.ci[1], i.estimate] : [i.estimate])).concat(hasClaim ? [claim, 0] : [0])
    .filter((v) => v != null && isFinite(v));
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const p = (hi - lo) * 0.08 || 1;
  lo -= p; hi += p;
  const sx = lin(lo, hi, L, W - R);
  const bottom = T + items.length * rowH;
  return (
    <Svg w={W} h={H} label="Estimates with 95% intervals">
      <line x1={sx(0)} x2={sx(0)} y1={T} y2={bottom} stroke="var(--muted)" />
      {hasClaim && (
        <>
          <line x1={sx(claim)} x2={sx(claim)} y1={T} y2={bottom} stroke="var(--warn)" strokeWidth="2" strokeDasharray="5 3" />
          <text x={sx(claim)} y={bottom + 32} textAnchor="middle" style={{ fill: "var(--warn)" }}>claimed</text>
        </>
      )}
      {items.map((it, i) => {
        const y = T + i * rowH + rowH / 2;
        const col = it.primary ? "var(--treat)" : "var(--ctrl)";
        const name = String(it.name);
        return (
          <g key={i}>
            <text x={L - 10} y={y + 4} textAnchor="end">{name.length > 40 ? `${name.slice(0, 39)}…` : name}</text>
            {it.ci && <line x1={sx(it.ci[0])} x2={sx(it.ci[1])} y1={y} y2={y} stroke={col} strokeWidth="2.5" />}
            <rect x={sx(it.estimate) - 5} y={y - 5} width="10" height="10" fill={col}>
              <title>{`${name}: ${fmt(it.estimate)}${it.ci ? ` [${fmt(it.ci[0])}, ${fmt(it.ci[1])}]` : ""}`}</title>
            </rect>
          </g>
        );
      })}
      <AxisX sx={sx} y={bottom + 2} t={ticks(lo, hi, 6)} w0={L} w1={W - R} />
    </Svg>
  );
}

// ------------------------------------------------------------------------------------------- graphs
const isHidden = (role) => /hidden|latent|unobserved|unmeasured/.test(role || "");
/** Map the engine's role strings onto the five drawn kinds. */
export function roleKind(role) {
  const r = String(role || "");
  if (r === "treatment" || r === "outcome") return r;
  if (isHidden(r)) return "latent";
  if (r === "instrument" || r === "running") return "instrument";
  if (/post_treatment|mediator|collider|after/.test(r)) return "mediator";
  return "other";
}
const ROLE_COLOR = { treatment: "var(--treat)", outcome: "var(--ctrl)", latent: "var(--latent)", instrument: "var(--warn)",
  mediator: "var(--fail)", other: "var(--ink2)" };

function NodeBox({ x, y, label, role, w }) {
  const kind = roleKind(role);
  const col = ROLE_COLOR[kind];
  const solid = kind === "treatment" || kind === "outcome";
  const text = String(label);
  return (
    <g className="dag-node">
      <rect x={x - w / 2} y={y - 15} width={w} height="30" rx="15" fill={solid ? col : "var(--panel)"} stroke={col} strokeWidth="1.8"
        strokeDasharray={kind === "latent" ? "4 3" : undefined} />
      <text x={x} y={y + 4} textAnchor="middle" style={{ fill: solid ? "var(--panel)" : "var(--ink)", fontSize: 12, fontWeight: 600 }}>
        {text.length > 22 ? `${text.slice(0, 21)}…` : text}
      </text>
      <title>{`${text} (${role})`}</title>
    </g>
  );
}

const MARKERS = ["ink2", "latent", "muted", "treat"];
const ArrowDefs = ({ id }) => (
  <defs>
    {MARKERS.map((c) => (
      <marker key={c} id={`${id}-${c}`} viewBox="0 0 10 10" refX="9" refY="5" markerWidth="11" markerHeight="11" markerUnits="userSpaceOnUse" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" fill={`var(--${c})`} />
      </marker>
    ))}
  </defs>
);

/** Quadratic path from p to q that stops at the (approximate) pill border. */
function edgePath(p, q, w, curve = 0) {
  const dx = q.x - p.x, dy = q.y - p.y, len = Math.hypot(dx, dy) || 1;
  const t = Math.min(0.5, 1 / Math.max(Math.abs(dx) / (w / 2 + 4), Math.abs(dy) / 19));
  const a = { x: p.x + dx * t, y: p.y + dy * t }, b = { x: q.x - dx * t, y: q.y - dy * t };
  const mx = (a.x + b.x) / 2 - (dy / len) * curve, my = (a.y + b.y) / 2 + (dx / len) * curve;
  return `M${a.x},${a.y} Q${mx},${my} ${b.x},${b.y}`;
}

let uid = 0;
const nextId = () => `vdag${++uid}`;

/** The assumed DAG of an audit, laid out as in v1: measured causes on the left, treatment in the middle, outcome on
   the right; instruments left of the treatment, post-treatment columns between treatment and outcome, hidden causes
   on top. Edge labels are left out (every arrow here is a hypothesis; the legend says so). */
export function Dag({ nodes, edges }) {
  if (!nodes?.length) return null;
  const id = nextId();
  const W = 820, NW = 150;
  const by = (k) => nodes.filter((n) => roleKind(n.role) === k);
  const T = by("treatment")[0], Y = by("outcome")[0];
  const instr = by("instrument"), mids = by("mediator"), latent = by("latent"), left = by("other");
  const leftCols = left.length > 12 ? 2 : 1;
  const perCol = Math.ceil(left.length / leftCols) || 1;
  const top = latent.length ? 70 : 20;
  const H = Math.max(240, top + 30 + perCol * 44, top + 60 + mids.length * 60 + 60, top + 40 + instr.length * 50);
  const cy = top + (H - top) / 2 - 10;
  const pos = {};
  const colX = leftCols === 2 ? [90, 250] : [100];
  const tx = leftCols === 2 ? 470 : 400;
  if (T) pos[T.id] = { x: tx, y: cy };
  if (Y) pos[Y.id] = { x: 730, y: cy };
  left.forEach((n, i) => {
    const c = Math.floor(i / perCol), k = i % perCol;
    pos[n.id] = { x: colX[c], y: top + ((k + 0.5) * (H - top - 20)) / perCol };
  });
  instr.forEach((n, i) => (pos[n.id] = { x: tx - 170, y: cy + (i - (instr.length - 1) / 2) * 50 + (left.length ? 0 : 0) }));
  if (instr.length && left.length) left.forEach((n) => (pos[n.id].x -= 20));
  mids.forEach((n, i) => (pos[n.id] = { x: (tx + 730) / 2, y: cy + 70 + i * 60 }));
  latent.forEach((n, i) => (pos[n.id] = { x: (tx + 730) / 2 + i * 170 - (latent.length - 1) * 85, y: 30 }));
  return (
    <Svg w={W} h={H} label="Assumed causal diagram">
      <ArrowDefs id={id} />
      {(edges || []).map((e, i) => {
        const p = pos[e.from], q = pos[e.to];
        if (!p || !q) return null;
        const lat = e.type === "bidirected" || isHidden(nodes.find((n) => n.id === e.from)?.role);
        const main = T && Y && e.from === T.id && e.to === Y.id;
        const c = lat ? "latent" : main ? "treat" : "ink2";
        // long edges from the left column into the outcome bow away from the treatment pill
        const intoY = !main && Y && e.to === Y.id && T && p.x < pos[T.id].x;
        const bend = intoY ? (p.y < cy ? 30 : -30) : main && mids.length ? -30 : 0;
        // fan the arrows into the outcome out over its left end instead of one point
        const qq = intoY ? { x: q.x, y: q.y + Math.max(-11, Math.min(11, (p.y - cy) / 12)) } : q;
        return (
          <path key={i} d={edgePath(p, qq, NW, bend)} fill="none" stroke={`var(--${c})`} strokeWidth={main ? 3 : 1.4}
            strokeDasharray={lat ? "5 4" : undefined} opacity={main || lat ? 1 : 0.55}
            markerEnd={`url(#${id}-${c})`} markerStart={e.type === "bidirected" ? `url(#${id}-${c})` : undefined}>
            <title>{`${e.from} → ${e.to}${e.label ? ` (${e.label})` : ""}`}</title>
          </path>
        );
      })}
      {nodes.map((n) => pos[n.id] && <NodeBox key={n.id} x={pos[n.id].x} y={pos[n.id].y} label={n.label || n.id} role={n.role} w={NW} />)}
    </Svg>
  );
}

/** Discovered graph on a circle: line weight = how many algorithms found the edge, dashed purple = latent cause. */
export function Discovery({ nodes, edges }) {
  if (!nodes?.length) return null;
  const id = nextId();
  const n = nodes.length, W = 760, H = n <= 3 ? 320 : 520, cx = W / 2, cy = H / 2, r = Math.min(W, H) / 2 - 60;
  const pos = {};
  nodes.forEach((nd, i) => {
    const a = (i / n) * Math.PI * 2 - Math.PI / 2;
    pos[nd.id] = { x: cx + r * 1.25 * Math.cos(a), y: cy + r * Math.sin(a) };
  });
  return (
    <Svg w={W} h={H} label="Discovered causal graph">
      <ArrowDefs id={id} />
      {(edges || []).map((e, i) => {
        const p = pos[e.from], q = pos[e.to];
        if (!p || !q) return null;
        const lat = e.type === "bidirected";
        const directed = e.type === "directed" || e.type === "partial";
        const support = e.support ?? 2;
        const c = lat ? "latent" : directed ? "ink2" : "muted";
        return (
          <path key={i} d={edgePath(p, q, 120, 0)} fill="none" stroke={`var(--${c})`} strokeWidth={0.8 + support * 0.7}
            strokeDasharray={lat ? "5 4" : undefined} opacity={0.35 + 0.2 * support}
            markerEnd={directed || lat ? `url(#${id}-${c})` : undefined} markerStart={lat ? `url(#${id}-${c})` : undefined}>
            <title>{`${e.from} ${lat ? "↔" : directed ? "→" : "—"} ${e.to}${e.label ? ` · ${e.label}` : ""}`}</title>
          </path>
        );
      })}
      {nodes.map((nd) => <NodeBox key={nd.id} x={pos[nd.id].x} y={pos[nd.id].y} label={nd.label || nd.id} role={nd.role} w={120} />)}
    </Svg>
  );
}

export const DAG_LEGEND = [
  ["var(--treat)", "treatment"], ["var(--ctrl)", "outcome"], ["var(--warn)", "instrument / running variable"],
  ["var(--fail)", "measured after treatment"], ["var(--latent)", "hidden (not in the data)"],
];
