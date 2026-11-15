/* Turn the backend's influence graph (+ finished literature investigations) into the mutable objects
   react-force-graph expects. */

export const MIN_R = 5, MAX_R = 24, LIT_R = 9;
const REL_MIN_R = 6, REL_MAX_R = 18; // hypothetical nodes are sized by their Reliability Score (0-100)
const reliabilityRadius = (s) => (s == null ? LIT_R : REL_MIN_R + ((REL_MAX_R - REL_MIN_R) * Math.max(0, Math.min(100, s))) / 100);
const bestSource = (sources = []) => [...sources].sort((a, b) => (b.source_score ?? 0) - (a.source_score ?? 0))[0] || null;

/**
 * @param inf             influence graph from the backend (data nodes carry influence; edges carry weights)
 * @param investigations  only "done" ones add nodes and links
 * @param prevNodes       previously drawn nodes, to keep positions stable when the graph is rebuilt
 * @param opts.hideHypothetical  leave out literature factors that are not in the dataset
 * @param opts.maxDepth          leave out literature nodes further than this from the outcome
 * @returns {nodes, links, litDepth}  litDepth: literature node id -> steps from the outcome (null if unreachable)
 */
export function buildGraph(inf, investigations = [], prevNodes = null, { hideHypothetical = false, maxDepth = null } = {}) {
  const maxInf = Math.max(1e-9, ...inf.nodes.map((n) => n.influence || 0));
  const nodes = inf.nodes.map((n) => ({
    ...n,
    r: n.role === "outcome" ? (MIN_R + MAX_R) / 2 : MIN_R + (MAX_R - MIN_R) * Math.sqrt((n.influence || 0) / maxInf),
    neighbors: new Set(),
    lit: [],
  }));
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const links = inf.edges.filter((e) => byId[e.source] && byId[e.target]).map((e) => ({ ...e }));
  const linked = new Set(links.map((l) => `${l.source}->${l.target}`));

  // literature findings: hypothetical nodes and links, no strengths
  for (const inv of investigations) {
    if (inv.status !== "done" || !inv.result) continue;
    for (const f of inv.result.factors) {
      const note = { from: inv.label, fromId: inv.node, explanation: f.explanation, sources: f.sources, affects: f.affects, reliability: f.reliability };
      const existing = byId[f.id];
      if (existing) {
        existing.lit.push(note);
        if (existing.hypothetical && (f.reliability ?? -1) > (existing.reliability ?? -1)) {
          // found again by another investigation, with stronger sources
          Object.assign(existing, { reliability: f.reliability, relFactor: f, bestSource: bestSource(f.sources), r: reliabilityRadius(f.reliability) });
        }
        if (!existing.hypothetical) existing.suggested = true; // a column the literature also points to
        continue;
      }
      if (hideHypothetical && !f.in_data) continue;
      const node = {
        id: f.id, label: f.label, role: f.in_data ? "literature_data" : "hypothetical",
        hypothetical: !f.in_data, observed: f.in_data, literature: true, suggested: f.in_data,
        reliability: f.reliability ?? null, relFactor: f, bestSource: bestSource(f.sources),
        r: f.in_data ? LIT_R : reliabilityRadius(f.reliability), neighbors: new Set(), lit: [note],
      };
      nodes.push(node);
      byId[f.id] = node;
    }
    for (const e of inv.result.edges) {
      const key = `${e.source}->${e.target}`;
      if (!byId[e.source] || !byId[e.target] || linked.has(key)) continue;
      linked.add(key);
      const factor = inv.result.factors.find((f) => f.id === e.source);
      links.push({ source: e.source, target: e.target, directed: true, hypothetical: true, weight: null,
        explanation: e.explanation, sources: factor?.sources || [], from: inv.label });
    }
  }

  // steps upstream from the outcome over the combined graph
  const outcome = inf.outcome || nodes.find((n) => n.role === "outcome")?.id;
  const dist = distancesToOutcome(links, outcome);
  const litDepth = new Map();
  nodes.forEach((n) => {
    if (!n.literature) return;
    n.depth = dist.get(n.id) ?? null;
    litDepth.set(n.id, n.depth);
  });

  // data nodes are already pruned by the backend's depth; literature nodes are filtered here (unreachable ones stay)
  const inDepth = new Set(nodes.filter((n) => !n.literature || maxDepth == null || n.depth == null || n.depth <= maxDepth).map((n) => n.id));
  const kept = links.filter((l) => inDepth.has(l.source) && inDepth.has(l.target));
  kept.forEach((l) => { byId[l.source].neighbors.add(l.target); byId[l.target].neighbors.add(l.source); });
  // a literature node whose links were all filtered out is hidden too
  const visible = nodes.filter((n) => inDepth.has(n.id) && (!n.literature || n.neighbors.size > 0));

  if (prevNodes?.length) {
    const prev = Object.fromEntries(prevNodes.map((n) => [n.id, n]));
    visible.forEach((n) => {
      const p = prev[n.id] || [...n.neighbors].map((id) => prev[id]).find(Boolean);
      if (p?.x == null) return;
      const jitter = prev[n.id] ? 0 : 50; // a new node starts next to a neighbour
      n.x = p.x + (Math.random() - 0.5) * jitter;
      n.y = p.y + (Math.random() - 0.5) * jitter;
      if (prev[n.id]) { n.fx = prev[n.id].fx; n.fy = prev[n.id].fy; }
    });
  }
  return { nodes: visible, links: kept, litDepth };
}

/** BFS upstream from the outcome; undirected links count both ways. */
function distancesToOutcome(links, outcome) {
  const dist = new Map();
  if (!outcome) return dist;
  const upstream = new Map();
  const add = (to, from) => (upstream.get(to) || upstream.set(to, []).get(to)).push(from);
  links.forEach((l) => { add(l.target, l.source); if (l.directed === false) add(l.source, l.target); });
  dist.set(outcome, 0);
  let frontier = [outcome];
  while (frontier.length) {
    const next = [];
    for (const n of frontier)
      for (const p of upstream.get(n) || [])
        if (!dist.has(p)) { dist.set(p, dist.get(n) + 1); next.push(p); }
    frontier = next;
  }
  return dist;
}
