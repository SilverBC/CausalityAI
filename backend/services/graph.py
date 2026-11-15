"""Influence graph: the analysis's causal diagram, weighted with standardized effects from the data.

Base graph
    Audit mode    the assumed causal diagram (`report.dag`) the core drew from the design and roles.
    Explore mode  the diagram causal discovery found (the dag chart in `report.explore`).
Edge weight
    the standardized OLS coefficient of the parent when the child is regressed on all its observed parents.
    A parent that is a category (several 0/1 columns) gets the size of its joint effect, sqrt(sum of squared betas),
    and no sign. Undirected discovery edges get the correlation; edges touching unmeasured nodes get none.
Node influence = sum of |weights| on its outgoing edges.
Total effect on the outcome = sum over directed paths of the products of edge weights (only when acyclic).

Depth (steps upstream from the outcome)
    depth == default  the base graph
    depth <  default  pruned to nodes within `depth` steps of the outcome
    depth >  default  (depth - default) rounds of searching the dataset for further upstream causes: for each variable
                      added in the previous round (round 1: every graph variable), columns that predict it jointly with
                      its known parents (p < 0.01, |std beta| >= 0.1) join as its parents. A new variable that reaches
                      both the treatment and the outcome (not through the treatment) is labelled a confounder.
The direction of a found edge is an assumption of the upstream search, and found edges are marked as such.
Ported from v1's influence.py / expand.py, adapted to this core's diagram format and encoded columns.
"""
from __future__ import annotations

from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

MAX_EXTRA_ROUNDS = 3   # the slider goes at most this far past the default depth
MAX_DEPTH = 8          # absolute ceiling
MAX_NODES = 40
PARENTS_PER_NODE = 3
MAX_CANDIDATES = 30    # pre-screened by |correlation| before the joint regression
ROW_SAMPLE = 5000
P_THRESHOLD = 0.01
MIN_STD_BETA = 0.1
MAX_MISSING = 0.3

ROLE_MAP = {
    "treatment": "treatment", "outcome": "outcome", "confounder (assumed)": "confounder",
    "instrument (assumed)": "instrument", "instrument": "instrument", "running variable": "running",
    "fixed effect": "fixed_effect", "post-treatment (excluded)": "post_treatment",
    "unmeasured confounder": "latent", "variable": "variable",
}


# --------------------------------------------------------------------------------------------- data
class Frame:
    """The prepared numeric data, plus how original columns map onto its (one-hot encoded) columns."""

    def __init__(self, df: pd.DataFrame, encoding: dict, kinds: dict, skip):
        self.df, self.encoding, self.kinds, self.skip = df, encoding or {}, kinds or {}, set(skip or ())
        self._z: Optional[pd.DataFrame] = None

    def cols(self, node: str) -> list[str]:
        if node in self.df.columns:
            return [node]
        return [c for c in self.encoding.get(node, []) if c in self.df.columns]

    @property
    def z(self) -> pd.DataFrame:
        """Standardized numeric frame (sampled for speed)."""
        if self._z is None:
            df = self.df if len(self.df) <= ROW_SAMPLE else self.df.sample(ROW_SAMPLE, random_state=0)
            out = {}
            for c in df.columns:
                x = pd.to_numeric(df[c], errors="coerce").astype(float)
                sd = x.std()
                if x.notna().sum() >= 10 and sd and np.isfinite(sd):
                    out[c] = (x - x.mean()) / sd
            self._z = pd.DataFrame(out, index=df.index)
        return self._z

    def candidates(self) -> list[str]:
        """Original columns a data search may add: single numeric columns that are not id/time/cluster roles."""
        out = []
        for orig, kind in self.kinds.items():
            if orig in self.skip or kind not in ("binary", "continuous", "count"):
                continue
            cs = self.cols(orig)
            if len(cs) == 1 and cs[0] in self.z.columns and self.df[cs[0]].isna().mean() <= MAX_MISSING:
                out.append(orig)
        return out


def frame_meta(report: dict) -> dict:
    """What the graph needs to know about the prepared data; stored with the result."""
    data = report.get("data") or {}
    roles = data.get("roles") or {}
    cols = data.get("columns") or []
    return {"encoding": {c["name"]: c.get("encoded_as") or [] for c in cols},
            "kinds": {c["name"]: c.get("kind") for c in cols},
            "skip": [roles[r] for r in ("unit", "time", "cluster", "strata") if roles.get(r)]}


# --------------------------------------------------------------------------------------------- base graph
def base_graph(report: dict) -> Optional[dict]:
    """Nodes and edges in the influence-graph vocabulary, from the core report's diagram."""
    chart, source = report.get("dag"), "claim"
    if not (chart and (chart.get("data") or {}).get("nodes")):
        chart = next((c for c in (report.get("explore") or {}).get("charts") or [] if c.get("kind") == "dag"), None)
        source = "discovery"
    if not chart or not (chart.get("data") or {}).get("nodes"):
        return None
    d = chart["data"]
    nodes = []
    for n in d["nodes"]:
        role = "group" if n.get("members") else ROLE_MAP.get(n.get("role", "variable"), "variable")
        nodes.append({"id": n["id"], "label": n["id"], "role": role, "members": n.get("members"),
                      "latent": bool(n.get("dashed")) or role == "latent"})
    ids = {n["id"] for n in nodes}
    latent_ids = [n["id"] for n in nodes if n["latent"]]
    edges: list[dict] = []
    for e in d.get("edges") or []:
        s, t, typ = e["from"], e["to"], e.get("type", "directed")
        if s not in ids or t not in ids:
            continue
        if typ == "bidirected":  # a hidden common cause: through the dashed U node when there is one
            if latent_ids and source == "claim":
                for u in latent_ids:
                    for x in (s, t):
                        if not any(ed["source"] == u and ed["target"] == x for ed in edges):
                            edges.append({"source": u, "target": x, "directed": True, "kind": "latent",
                                          "label": e.get("label")})
            else:
                edges.append({"source": s, "target": t, "directed": False, "kind": "latent", "label": e.get("label")})
            continue
        edges.append({"source": s, "target": t, "directed": typ in ("directed", "partial"),
                      "kind": "uncertain" if typ == "partial" else ("data" if source == "discovery" else "assumed"),
                      "label": e.get("label"), "confidence": e.get("confidence")})
    linked = {x for e in edges for x in (e["source"], e["target"])}
    nodes = [n for n in nodes if n["id"] in linked or not n["latent"]]
    return {"source": source, "nodes": nodes, "edges": edges}


# --------------------------------------------------------------------------------------------- weights
def _ols(Z: pd.DataFrame, target: str, parents: list[str]) -> dict[str, float]:
    sub = Z[[target, *parents]].dropna()
    if len(sub) < len(parents) + 10:
        return {}
    X = np.column_stack([np.ones(len(sub)), sub[parents].to_numpy()])
    beta, *_ = np.linalg.lstsq(X, sub[target].to_numpy(), rcond=None)
    return {p: float(b) for p, b in zip(parents, beta[1:])}


def influence(fr: Frame, graph: dict) -> dict:
    nodes, edges, Z = graph["nodes"], graph["edges"], fr.z
    cols = {n["id"]: fr.cols(n["id"]) for n in nodes}
    observed = {i for i, cs in cols.items() if cs and all(c in Z.columns for c in cs)}

    parents: dict[str, list[str]] = {}
    for e in edges:
        if e.get("directed", True) and e.get("kind") != "latent" and e["source"] in observed and e["target"] in observed:
            parents.setdefault(e["target"], []).append(e["source"])

    weight: dict[tuple[str, str], tuple[float, bool]] = {}   # (parent, child) -> (weight, has a sign)
    for child, ps in parents.items():
        if len(cols[child]) != 1:
            continue  # a category as the child has no single standardized coefficient
        ps = sorted(set(ps))
        betas = _ols(Z, cols[child][0], [c for p in ps for c in cols[p]])
        for p in ps:
            bs = [betas[c] for c in cols[p] if c in betas]
            if len(bs) == 1:
                weight[(p, child)] = (bs[0], True)
            elif bs:
                weight[(p, child)] = (float(np.sqrt(np.sum(np.square(bs)))), False)

    out_edges = []
    for e in edges:
        s, t = e["source"], e["target"]
        w, signed = None, True
        if e.get("kind") != "latent" and s in observed and t in observed:
            if e.get("directed", True):
                w, signed = weight.get((s, t), (None, True))
            elif len(cols[s]) == 1 and len(cols[t]) == 1:
                pair = Z[[cols[s][0], cols[t][0]]].dropna()
                if len(pair) > 10:
                    w = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        ok = w is not None and np.isfinite(w)
        out_edges.append({**e, "weight": round(w, 4) if ok else None, "signed": signed,
                          "latent": e.get("kind") == "latent", "found": bool(e.get("found"))})

    strength = {n["id"]: 0.0 for n in nodes}
    for e in out_edges:
        if e["weight"] is not None:
            strength[e["source"]] += abs(e["weight"])
            if not e["directed"]:
                strength[e["target"]] += abs(e["weight"])

    Y = next((n["id"] for n in nodes if n["role"] == "outcome"), None)
    total: dict[str, float] = {}
    G = nx.DiGraph([(e["source"], e["target"], {"w": e["weight"]}) for e in out_edges
                    if e["directed"] and e["weight"] is not None and e["signed"]])
    if Y and Y in G and nx.is_directed_acyclic_graph(G):
        total[Y] = 1.0
        for n in reversed(list(nx.topological_sort(G))):
            if n != Y:
                total[n] = sum(G[n][m]["w"] * total.get(m, 0.0) for m in G.successors(n))

    return {
        "source": graph.get("source"), "outcome": Y,
        "treatment": next((n["id"] for n in nodes if n["role"] == "treatment"), None),
        "nodes": [{"id": n["id"], "label": n.get("label", n["id"]), "role": n["role"],
                   "observed": n["id"] in observed and not n.get("latent"), "influence": round(strength[n["id"]], 4),
                   "total_effect_on_outcome": round(total[n["id"]], 4) if n["id"] in total and n["id"] != Y else None,
                   "depth": n.get("depth"), "found": bool(n.get("found")), "members": n.get("members")}
                  for n in nodes],
        "edges": out_edges,
    }


# --------------------------------------------------------------------------------------------- depth
def distances(nodes: list[dict], edges: list[dict], outcome: Optional[str]) -> dict[str, int]:
    """Steps upstream from the outcome (undirected edges count both ways)."""
    if not outcome:
        return {}
    up: dict[str, set[str]] = {n["id"]: set() for n in nodes}
    for e in edges:
        s, t = e["source"], e["target"]
        if s in up and t in up:
            up[t].add(s)
            if not e.get("directed", True):
                up[s].add(t)
    dist, frontier = {outcome: 0}, [outcome]
    while frontier:
        nxt = []
        for n in frontier:
            for p in up.get(n, ()):
                if p not in dist:
                    dist[p] = dist[n] + 1
                    nxt.append(p)
        frontier = nxt
    return dist


def depth_info(graph: dict) -> dict:
    Y = next((n["id"] for n in graph["nodes"] if n["role"] == "outcome"), None)
    if not Y:  # exploration without an outcome: no depth to search
        return {"default_depth": 1, "max_depth": 1, "searchable": False}
    default = max(list(distances(graph["nodes"], graph["edges"], Y).values()) + [1])
    return {"default_depth": default, "max_depth": min(MAX_DEPTH, default + MAX_EXTRA_ROUNDS), "searchable": True}


def _significant(Z: pd.DataFrame, target: str, controls: list[str], tests: list[str]) -> dict[str, float]:
    import statsmodels.api as sm

    cols = [c for c in dict.fromkeys([target, *controls, *tests]) if c in Z.columns]
    tests = [c for c in tests if c in cols and c != target]
    controls = [c for c in controls if c in cols and c != target and c not in tests]
    if not tests:
        return {}
    sub = Z[cols].dropna()
    if len(sub) < len(cols) + 30:
        return {}
    try:
        m = sm.OLS(sub[target], sm.add_constant(sub[[*controls, *tests]], has_constant="add")).fit()
    except Exception:  # noqa: BLE001
        return {}
    return {c: float(m.params[c]) for c in tests if m.pvalues[c] < P_THRESHOLD and abs(m.params[c]) >= MIN_STD_BETA}


def graph_at_depth(fr: Frame, base: dict, depth: int) -> dict:
    nodes = [dict(n) for n in base["nodes"]]
    edges = [dict(e) for e in base["edges"]]
    info = depth_info(base)
    Y = next((n["id"] for n in nodes if n["role"] == "outcome"), None)
    T = next((n["id"] for n in nodes if n["role"] == "treatment"), None)
    default = info["default_depth"]
    depth = max(1, min(int(depth), info["max_depth"]))
    dist = distances(nodes, edges, Y)
    for n in nodes:
        n["depth"] = dist.get(n["id"])
    meta = {"depth": depth, "found_nodes": 0, "found_edges": 0, "exhausted": False}

    if depth < default:
        keep = {i for i, d in dist.items() if d <= depth}
        return {**base, "nodes": [n for n in nodes if n["id"] in keep],
                "edges": [e for e in edges if e["source"] in keep and e["target"] in keep], **meta}
    if depth == default or not Y:
        return {**base, "nodes": nodes, "edges": edges, **meta}

    Z = fr.z
    col = {n["id"]: (fr.cols(n["id"]) or [None])[0] for n in nodes}
    by_id = {n["id"]: n for n in nodes}
    G = nx.DiGraph()
    G.add_nodes_from(by_id)
    G.add_edges_from((e["source"], e["target"]) for e in edges if e.get("directed", True))
    pool = [c for c in fr.candidates() if c not in by_id]

    def zcol(i: str) -> Optional[str]:
        c = col.get(i)
        return c if c in Z.columns else None

    def observed_parents(v: str) -> list[str]:
        return [zcol(u) for u in G.predecessors(v) if zcol(u)]

    to_search = [n["id"] for n in nodes if zcol(n["id"]) and n["id"] != Y]
    found, exhausted, base_edges = 0, False, len(edges)
    for _ in range(depth - default):
        new_nodes: list[str] = []
        for F in to_search:
            if len(by_id) >= MAX_NODES:
                break
            desc = nx.descendants(G, F) | {F}
            # candidates exclude F's descendants, so c -> F can never close a cycle
            cands = [c for c in pool if c not in desc and c not in by_id and not G.has_edge(c, F)]
            cands += [i for i in by_id if by_id[i].get("found") and i not in desc and zcol(i) and not G.has_edge(i, F)]
            ccols = {fr.cols(c)[0]: c for c in cands if fr.cols(c) and fr.cols(c)[0] in Z.columns}
            if not ccols:
                continue
            corr = Z[list(ccols)].corrwith(Z[zcol(F)]).abs().dropna().sort_values(ascending=False)
            top = list(corr.index[:MAX_CANDIDATES])
            hits = _significant(Z, zcol(F), observed_parents(F), top)
            for c_col, _b in sorted(hits.items(), key=lambda kv: -abs(kv[1]))[:PARENTS_PER_NODE]:
                c = ccols[c_col]
                if c not in by_id:
                    if len(by_id) >= MAX_NODES:
                        break
                    by_id[c] = {"id": c, "label": c, "role": "upstream", "found": True, "latent": False}
                    col[c] = c_col
                    nodes.append(by_id[c])
                    G.add_node(c)
                    new_nodes.append(c)
                if not G.has_edge(c, F):
                    G.add_edge(c, F)
                    edges.append({"source": c, "target": F, "directed": True, "kind": "found", "found": True})
        for V in new_nodes:  # a new variable may also cause other nodes already on the graph
            for N in list(G.nodes):
                if N == V or not zcol(N) or G.has_edge(V, N) or nx.has_path(G, N, V):
                    continue
                if _significant(Z, zcol(N), observed_parents(N), [zcol(V)]):
                    G.add_edge(V, N)
                    edges.append({"source": V, "target": N, "directed": True, "kind": "found", "found": True})
        found += len(new_nodes)
        if not new_nodes:
            exhausted = True
            break
        to_search = new_nodes

    if T and Y and T in G:
        G_noT = G.copy()
        G_noT.remove_node(T)
        for n in nodes:
            if n.get("found") and nx.has_path(G, n["id"], T) and Y in G_noT and nx.has_path(G_noT, n["id"], Y):
                n["role"] = "confounder"
    dist = distances(nodes, edges, Y)
    for n in nodes:
        n["depth"] = dist.get(n["id"], n.get("depth"))
    return {**base, "nodes": nodes, "edges": edges, "depth": depth, "found_nodes": found,
            "found_edges": len(edges) - base_edges, "exhausted": exhausted}


def influence_at_depth(fr: Frame, base: dict, depth: Optional[int] = None) -> dict:
    info = depth_info(base)
    g = graph_at_depth(fr, base, info["default_depth"] if depth is None else depth)
    out = influence(fr, g)
    out.update(info)
    out.update({k: g[k] for k in ("depth", "found_nodes", "found_edges", "exhausted")})
    return out
