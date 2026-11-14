/* Turn the backend's influence graph into the mutable objects react-force-graph expects. */

export const MIN_R = 5, MAX_R = 24;

/**
 * @param inf        influence graph from the backend (nodes carry influence; edges carry standardized weights)
 * @param prevNodes  previously drawn nodes, to keep positions stable when the graph is rebuilt
 */
export function buildGraph(inf, prevNodes = null) {
  const maxInf = Math.max(1e-9, ...inf.nodes.map((n) => n.influence || 0));
  const nodes = inf.nodes.map((n) => ({
    ...n,
    // area grows with influence; the outcome gets a fixed mid size
    r: n.role === "outcome" ? (MIN_R + MAX_R) / 2 : MIN_R + (MAX_R - MIN_R) * Math.sqrt((n.influence || 0) / maxInf),
    neighbors: new Set(),
  }));
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const links = inf.edges.filter((e) => byId[e.source] && byId[e.target]).map((e) => ({ ...e }));
  links.forEach((l) => { byId[l.source].neighbors.add(l.target); byId[l.target].neighbors.add(l.source); });

  if (prevNodes?.length) {
    const prev = Object.fromEntries(prevNodes.map((n) => [n.id, n]));
    nodes.forEach((n) => {
      const p = prev[n.id] || [...n.neighbors].map((id) => prev[id]).find(Boolean);
      if (p?.x == null) return;
      const jitter = prev[n.id] ? 0 : 50; // a new node starts next to a neighbour
      n.x = p.x + (Math.random() - 0.5) * jitter;
      n.y = p.y + (Math.random() - 0.5) * jitter;
      if (prev[n.id]) { n.fx = prev[n.id].fx; n.fy = prev[n.id].fy; }
    });
  }
  return { nodes, links };
}
