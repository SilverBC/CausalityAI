"""Explore mode: what causes what, with no claim.

Main graph: FCI (allows hidden common causes) on standardized numeric variables, Fisher-z
tests, alpha 0.05. Second views: PC and DirectLiNGAM. Every edge is re-checked by FCI on
random subsamples; only stable tail-arrow edges are drawn as arrows. Uncertain links are
drawn without arrowheads. Exploration never "passes": the section is informational.

causal-learn graph convention (verified in tests/test_explore.py):
    G.graph[i, j] is the endpoint mark AT NODE i on the edge between i and j
    (-1 = tail, 1 = arrowhead, 2 = circle, 0 = no edge).
    So graph[i, j] == -1 and graph[j, i] == 1 means  i --> j.
"""
from __future__ import annotations

import contextlib
import io
import itertools
import time
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .data import words
from .schema import Assumption, Chart, Prepared, Section, Status

KEY = "explore"
TITLE = "What causes what (exploration)"

ALPHA = 0.05
MAX_VARS = 15
MAX_ROWS = 3000
MIN_ROWS = 30
N_SUB = 20
SUB_FRAC = (0.5, 0.8)
ADJ_STABLE = 0.5        # adjacency drawn only if present in >= 50% of subsamples
ORIENT_STABLE = 0.7     # an arrowhead/tail claim kept only if seen in >= 70% of subsamples
TIME_BUDGET = 30.0      # seconds for the subsample FCI runs
NEAR_DUP = 0.95
LINGAM_MIN_COEF = 0.1   # standardized coefficient below which a LiNGAM edge is ignored
SEED = 0

TAIL, ARROW, CIRCLE = -1, 1, 2
_SYM = {TAIL: "-", ARROW: ">", CIRCLE: "o"}
TIME_WORDS = {"year", "yr", "month", "day", "date", "time", "week", "period", "quarter",
              "timestamp", "wave", "round"}
ID_WORDS = {"id", "uuid", "index", "idx", "key", "code", "number", "no"}


# ----------------------------------------------------------------------------- helpers
def _quiet(fn, *a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*a, **k)


def _fci_graph(X: np.ndarray) -> np.ndarray:
    from causallearn.search.ConstraintBased.FCI import fci
    G, _ = _quiet(fci, X, "fisherz", ALPHA, verbose=False, show_progress=False)
    return np.asarray(G.graph, dtype=int)


def _pc_graph(X: np.ndarray) -> np.ndarray:
    from causallearn.search.ConstraintBased.PC import pc
    cg = _quiet(pc, X, ALPHA, "fisherz", show_progress=False)
    return np.asarray(cg.G.graph, dtype=int)


def pag_mark(g: np.ndarray, i: int, j: int) -> tuple[int, int]:
    """(mark at i, mark at j) for the edge i *-* j; (0, 0) if no edge."""
    return int(g[i, j]), int(g[j, i])


def edge_string(mi: int, mj: int) -> Optional[str]:
    """Marks at i and j -> e.g. 'o->' (i o-> j), '-->' (i --> j), '<->', 'o-o', '<-o'."""
    if mi == 0 or mj == 0:
        return None
    left = {TAIL: "-", ARROW: "<", CIRCLE: "o"}[mi]
    right = {TAIL: "-", ARROW: ">", CIRCLE: "o"}[mj]
    return f"{left}-{right}"


def edge_type(mi: int, mj: int) -> Optional[str]:
    """Plain type of an edge from its marks (before stability)."""
    if mi == 0 or mj == 0:
        return None
    if (mi, mj) in ((TAIL, ARROW), (ARROW, TAIL)):
        return "directed"
    if (mi, mj) == (ARROW, ARROW):
        return "bidirected"
    if (mi, mj) in ((CIRCLE, ARROW), (ARROW, CIRCLE)):
        return "partial"
    return "undirected"   # o-o, ---, --o: no arrowhead claim we draw


def _std(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)


def _hsic_p(x: np.ndarray, y: np.ndarray, n_perm: int = 100, seed: int = 0) -> float:
    """HSIC independence test (Gaussian kernels, median bandwidth, permutation p-value)."""
    rng = np.random.default_rng(seed)
    n = len(x)

    def gram(v):
        d = (v[:, None] - v[None, :]) ** 2
        med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
        K = np.exp(-d / med)
        H = np.eye(n) - 1.0 / n
        return H @ K @ H

    Kc, Lc = gram(x), gram(y)
    stat = float(np.sum(Kc * Lc))
    count = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if float(np.sum(Kc * Lc[np.ix_(p, p)])) >= stat:
            count += 1
    return (count + 1) / (n_perm + 1)


def _direction_hint(x: np.ndarray, y: np.ndarray, seed: int) -> dict:
    """Regress each on the other; the direction whose residual is independent of the
    regressor (HSIC) is the one a linear non-Gaussian model prefers. Only a hint."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x), size=min(300, len(x)), replace=False)
    x, y = x[idx], y[idx]

    def resid(dep, reg):
        b = np.polyfit(reg, dep, 1)
        return dep - np.polyval(b, reg)

    p_xy = _hsic_p(x, resid(y, x), seed=seed)      # model x -> y
    p_yx = _hsic_p(y, resid(x, y), seed=seed + 1)  # model y -> x
    if p_xy > ALPHA and p_yx <= ALPHA:
        pref = "a->b"
    elif p_yx > ALPHA and p_xy <= ALPHA:
        pref = "b->a"
    else:
        pref = "unclear"
    return {"p_resid_indep_a_to_b": p_xy, "p_resid_indep_b_to_a": p_yx, "prefers": pref}


def _partial_corr(Z: np.ndarray) -> Optional[np.ndarray]:
    C = np.corrcoef(Z, rowvar=False)
    try:
        if np.linalg.cond(C) > 1e10:
            return None
        P = np.linalg.inv(C)
    except np.linalg.LinAlgError:
        return None
    d = np.sqrt(np.diag(P))
    pc = -P / np.outer(d, d)
    np.fill_diagonal(pc, 1.0)
    return pc


def _r_pvalue(r: float, n: int, k: int) -> Optional[float]:
    """Fisher-z p-value of a (partial) correlation with k conditioning variables."""
    if r is None or not np.isfinite(r) or n - k - 3 <= 0:
        return None
    r = float(np.clip(r, -0.999999, 0.999999))
    z = 0.5 * np.log((1 + r) / (1 - r)) * np.sqrt(n - k - 3)
    return float(2 * stats.norm.sf(abs(z)))


def _err(msg: str, **metrics) -> Section:
    return Section(key=KEY, title=TITLE, status=None, error=msg, metrics=metrics)


# ----------------------------------------------------------------------------- variables
def _select_variables(prep: Prepared) -> tuple[pd.DataFrame, dict]:
    info: dict = {"left_out": [], "time_like": [], "id_like": [], "dropped_duplicates": []}
    r = prep.roles
    candidates = []
    for c in prep.columns:
        reason = None
        if c.kind in ("id", "text", "constant"):
            reason = {"id": "identifier", "text": "free text", "constant": "constant"}[c.kind]
        elif c.kind == "categorical":
            reason = "category with several levels (one-hot columns are not used for discovery)"
        elif c.name in (r.unit, r.cluster):
            reason = "unit/cluster label"
        elif c.name not in prep.df.columns:
            reason = "not available as a number"
        if reason:
            info["left_out"].append({"column": c.name, "reason": reason})
            continue
        w = set(words(c.name))
        if c.kind == "datetime" or c.name == r.time or w & TIME_WORDS:
            info["time_like"].append(c.name)
        if w & ID_WORDS and c.kind != "binary":
            info["id_like"].append(c.name)
        candidates.append(c.name)

    df = prep.df[candidates].apply(pd.to_numeric, errors="coerce").astype(float)
    # zero variance after encoding
    for c in list(df.columns):
        if df[c].dropna().nunique() <= 1:
            info["left_out"].append({"column": c, "reason": "constant"})
            df = df.drop(columns=c)

    if df.shape[1] > MAX_VARS:
        nonmiss = df.notna().mean()
        nuniq = df.nunique()
        order = sorted(df.columns, key=lambda c: (-nonmiss[c], -nuniq[c], list(df.columns).index(c)))
        keep = order[:MAX_VARS]
        for c in order[MAX_VARS:]:
            info["left_out"].append({"column": c, "reason": f"over the {MAX_VARS}-variable limit "
                                     "(kept the most complete, most varied columns)"})
        df = df[[c for c in df.columns if c in keep]]
    return df, info


def _drop_exact_duplicates(df: pd.DataFrame, info: dict) -> pd.DataFrame:
    """Columns that are an exact linear copy of an earlier column make the correlation
    matrix singular; keep the first."""
    if df.shape[1] < 2:
        return df
    C = df.corr().abs().to_numpy()
    cols = list(df.columns)
    drop = set()
    for i, j in itertools.combinations(range(len(cols)), 2):
        if cols[i] in drop or cols[j] in drop:
            continue
        if np.isfinite(C[i, j]) and C[i, j] > 0.9999:
            drop.add(cols[j])
            info["dropped_duplicates"].append({"column": cols[j], "copy_of": cols[i]})
    return df.drop(columns=sorted(drop))


# ----------------------------------------------------------------------------- main
def run(prep: Prepared) -> Section:
    try:
        return _run(prep)
    except Exception as e:  # noqa: BLE001
        return _err(f"The exploration could not be completed ({type(e).__name__}: {e}). "
                    "No causal links are reported.")


def _run(prep: Prepared) -> Section:
    t0 = time.time()
    warnings: list[str] = []
    df, vinfo = _select_variables(prep)
    if df.shape[1] < 3:
        return _err(f"Only {df.shape[1]} usable numeric variable(s) were found; exploring what causes "
                    "what needs at least 3 (yes/no, numeric or count columns).",
                    variables=list(df.columns), left_out=vinfo["left_out"])

    n_all = len(df)
    df = df.dropna()
    n_dropped = n_all - len(df)
    if n_dropped:
        warnings.append(f"{n_dropped} row(s) with missing values were left out of the discovery.")
    if len(df) < MIN_ROWS:
        return _err(f"Only {len(df)} complete row(s) are available; at least {MIN_ROWS} are needed "
                    "to explore what causes what.", variables=list(df.columns))

    # constants after dropping rows, then exact duplicates
    for c in list(df.columns):
        if df[c].nunique() <= 1:
            vinfo["left_out"].append({"column": c, "reason": "constant in the complete rows"})
            df = df.drop(columns=c)
    df = _drop_exact_duplicates(df, vinfo)
    for d in vinfo["dropped_duplicates"]:
        warnings.append(f"'{d['column']}' is an exact copy (up to scale) of '{d['copy_of']}' and was left out.")
    if df.shape[1] < 3:
        return _err(f"After removing constant and duplicate columns only {df.shape[1]} variable(s) remain; "
                    "at least 3 are needed.", variables=list(df.columns))

    rng = np.random.default_rng(SEED)
    if len(df) > MAX_ROWS:
        df = df.iloc[np.sort(rng.choice(len(df), MAX_ROWS, replace=False))]
        warnings.append(f"A random subsample of {MAX_ROWS} rows was used for speed.")
    if len(df) < 100:
        warnings.append(f"Only {len(df)} rows: independence tests have little power, so links may be missed.")

    names = list(df.columns)
    p = len(names)
    Z = _std(df).to_numpy()
    n = len(Z)
    binaries = [c for c in names if df[c].nunique() == 2]
    if binaries:
        warnings.append("Yes/no variables (" + ", ".join(binaries) + ") are treated as numbers by the "
                        "linear tests; links involving them are rougher.")

    pcorr = _partial_corr(Z)
    if pcorr is None:
        return _err("The variables are almost perfectly predictable from one another (the correlation "
                    "matrix is singular), so independence tests cannot be run. Remove redundant columns.",
                    variables=names)
    corr = np.corrcoef(Z, rowvar=False)

    # ---------------- main FCI
    from causallearn.search.ConstraintBased.FCI import fci  # noqa: F401  (import before timing)
    t_fci = time.time()
    G = _fci_graph(Z)
    t_fci = time.time() - t_fci

    # ---------------- stability
    n_sub = N_SUB
    if t_fci * N_SUB > TIME_BUDGET:
        n_sub = max(5, int(TIME_BUDGET / max(t_fci, 1e-3)))
        warnings.append(f"Stability used {n_sub} subsamples (instead of {N_SUB}) to keep the run time short.")
    adj_cnt = np.zeros((p, p))
    mark_cnt = {m: np.zeros((p, p)) for m in (TAIL, ARROW, CIRCLE)}   # mark_cnt[m][i,j]: mark at i on i-j
    str_cnt: dict[tuple[int, int], dict[str, int]] = {}
    sub_rng = np.random.default_rng(SEED + 1)
    done = 0
    t_sub = time.time()
    for _ in range(n_sub):
        if time.time() - t_sub > TIME_BUDGET * 1.3:
            break
        frac = sub_rng.uniform(*SUB_FRAC)
        idx = sub_rng.choice(n, size=max(MIN_ROWS, int(frac * n)), replace=False)
        Xs = Z[idx]
        Xs = (Xs - Xs.mean(0)) / np.where(Xs.std(0) > 0, Xs.std(0), 1)
        try:
            g = _fci_graph(Xs)
        except Exception:  # noqa: BLE001
            continue
        done += 1
        for i, j in itertools.combinations(range(p), 2):
            mi, mj = pag_mark(g, i, j)
            if mi == 0:
                continue
            adj_cnt[i, j] += 1
            adj_cnt[j, i] += 1
            mark_cnt[mi][i, j] += 1
            mark_cnt[mj][j, i] += 1
            s = edge_string(mi, mj)
            str_cnt.setdefault((i, j), {})
            str_cnt[(i, j)][s] = str_cnt[(i, j)].get(s, 0) + 1
    if done == 0:
        return _err("Every subsample run of the discovery failed, so no link could be checked for stability.",
                    variables=names)
    adj_f = adj_cnt / done
    mark_f = {m: v / done for m, v in mark_cnt.items()}

    # ---------------- second views
    Gpc = None
    try:
        Gpc = _pc_graph(Z)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"The PC algorithm failed ({type(e).__name__}); no PC view.")
    lingam_B, lingam_gauss, lingam_assump = None, None, None
    try:
        lingam_B, lingam_gauss, lingam_assump = _lingam(Z, names, rng)
    except Exception as e:  # noqa: BLE001
        warnings.append(f"LiNGAM failed ({type(e).__name__}); no LiNGAM view.")
    lingam_reliable_global = lingam_assump is not None and lingam_assump.status == Status.PASS

    # ---------------- per pair
    pairs, drawn, unstable = [], [], []
    pc_edges, lingam_edges, fci_edges = [], [], []
    hint_seed = 100
    for i, j in itertools.combinations(range(p), 2):
        a, b = names[i], names[j]
        mi, mj = pag_mark(G, i, j)
        fci_s = edge_string(mi, mj)
        if fci_s:
            fci_edges.append({"a": a, "b": b, "edge": f"{a} {fci_s} {b}", "type": edge_type(mi, mj)})
        pc_s = None
        if Gpc is not None:
            pmi, pmj = pag_mark(Gpc, i, j)
            pc_s = edge_string(pmi, pmj)
            if pc_s:
                pc_edges.append({"from": a, "to": b, "edge": f"{a} {pc_s} {b}"})
        lg_s = None
        if lingam_B is not None:
            bij, bji = lingam_B[j, i], lingam_B[i, j]   # B[to, from]
            if abs(bij) >= LINGAM_MIN_COEF and abs(bij) >= abs(bji):
                lg_s = "-->"
                lingam_edges.append(_lingam_edge(a, b, bij, lingam_gauss, i, j, lingam_reliable_global))
            elif abs(bji) >= LINGAM_MIN_COEF:
                lg_s = "<--"
                lingam_edges.append(_lingam_edge(b, a, bji, lingam_gauss, j, i, lingam_reliable_global))

        af = float(adj_f[i, j])
        if not (fci_s or pc_s or lg_s or af >= 0.2):
            continue

        # stability-adjusted type
        final_type, frm, to, conf = None, a, b, None
        status_note = ""
        if fci_s and af >= ADJ_STABLE:
            t = edge_type(mi, mj)
            head_j, head_i = mark_f[ARROW][j, i], mark_f[ARROW][i, j]
            if t == "directed":
                frm, to = (a, b) if mj == ARROW else (b, a)
                # frequency of exactly this tail-arrow orientation
                f_dir = (_count(str_cnt, i, j, "-->") if mj == ARROW else _count(str_cnt, i, j, "<--")) / done
                f_head = head_j if mj == ARROW else head_i
                if f_dir >= ORIENT_STABLE:
                    final_type, conf = "directed", f_dir
                elif f_head >= ORIENT_STABLE:
                    final_type, conf = "partial", f_head
                    status_note = "FCI gave a full arrow, but the tail was not stable across subsamples"
                else:
                    final_type, conf, frm, to = "undirected", af, a, b
                    status_note = "FCI gave an arrow, but its direction was not stable across subsamples"
            elif t == "bidirected":
                f_bi = _count(str_cnt, i, j, "<->") / done
                if f_bi >= ORIENT_STABLE:
                    final_type, conf = "bidirected", f_bi
                else:
                    final_type, conf = "undirected", af
                    status_note = "the hidden-common-cause reading was not stable across subsamples"
            elif t == "partial":
                frm, to = (a, b) if mj == ARROW else (b, a)
                f_head = head_j if mj == ARROW else head_i
                if f_head >= ORIENT_STABLE:
                    final_type, conf = "partial", f_head
                else:
                    final_type, conf, frm, to = "undirected", af, a, b
                    status_note = "the arrowhead was not stable across subsamples"
            else:
                final_type, conf = "undirected", af
        elif fci_s:
            status_note = f"found on the full data but only in {af:.0%} of subsamples; not drawn"
        elif af >= ADJ_STABLE:
            status_note = f"seen in {af:.0%} of subsamples but not on the full data; not drawn"

        # sanity checks
        checks = []
        r = float(corr[i, j])
        prv = float(pcorr[i, j])
        p_r = _r_pvalue(r, n, 0)
        p_pr = _r_pvalue(prv, n, p - 2)
        checks.append({"name": "marginal correlation", "value": r, "p_value": p_r,
                       "ok": p_r is not None and p_r < ALPHA})
        checks.append({"name": "partial correlation given all other variables", "value": prv, "p_value": p_pr,
                       "ok": p_pr is not None and p_pr < ALPHA})
        flags = []
        if abs(r) > NEAR_DUP:
            flags.append("near_duplicate")
            checks.append({"name": "near-duplicate", "value": abs(r), "ok": False,
                           "detail": "these two columns are almost the same quantity; no causal reading"})
        tl = [v for v in (a, b) if v in vinfo["time_like"] or v in vinfo["id_like"]]
        if tl:
            flags.append("time_or_id")
            checks.append({"name": "time/id-like variable", "value": None, "ok": False,
                           "detail": ", ".join(tl) + " looks like time or an identifier; links to it "
                                     "usually reflect trends or bookkeeping, not causation"})
        hint = None
        if final_type is not None:
            hint = _direction_hint(Z[:, i], Z[:, j], hint_seed)
            hint_seed += 2
            checks.append({"name": "residual independence direction hint (HSIC)", **hint,
                           "detail": "regress each on the other; under a linear model with non-Gaussian "
                                     "noise only the true direction leaves residuals independent of the "
                                     "regressor. A hint only; it never changes the drawn edge."})
        lg_note = None
        if lg_s:
            le = lingam_edges[-1]
            lg_note = f"{le['from']} -> {le['to']}" + ("" if le["reliable"] else " (unreliable)")

        reading = _reading(a, b, final_type, frm, to, conf, flags, status_note, af)
        rec = {"a": a, "b": b,
               "fci": f"{a} {fci_s} {b}" if fci_s else None,
               "pc": f"{a} {pc_s} {b}" if pc_s else None,
               "lingam": lg_note,
               "drawn": {"type": final_type, "from": frm, "to": to} if final_type else None,
               "stability": {"adjacency": af, "orientations": {k: v / done for k, v in str_cnt.get((i, j), {}).items()},
                             "confidence": conf},
               "corr": r, "partial_corr": prv, "checks": checks, "flags": flags,
               "agreement": _agreement(final_type, frm, to, a, b, pc_s, lg_note),
               "reading": reading}
        pairs.append(rec)
        if final_type:
            drawn.append({"from": frm, "to": to, "type": final_type,
                          "label": _label(final_type, conf, flags), "confidence": conf})
        elif fci_s or af >= ADJ_STABLE:
            unstable.append({"a": a, "b": b, "adjacency_stability": af, "note": status_note})

    # ---------------- assemble
    n_dir = sum(e["type"] == "directed" for e in drawn)
    n_unc = len(drawn) - n_dir
    isolated = [v for v in names if not any(v in (e["from"], e["to"]) for e in drawn)]
    summary = (f"Among {p} variables, {len(drawn)} stable link(s) were found: {n_dir} supported arrow(s) and "
               f"{n_unc} link(s) whose direction is uncertain. Uncertain links are shown without arrowheads "
               "(or with a circle end where only one side is known).")
    if unstable:
        summary += f" {len(unstable)} other link(s) were not stable across subsamples and are not drawn."
    summary += " These are patterns the data allow, not proven effects."

    assumptions = [
        Assumption("explore_faithfulness", "Hidden common causes allowed; faithfulness assumed", Status.ASSUMED,
                   "Causal sufficiency is NOT assumed by FCI (hidden common causes allowed), but faithfulness is "
                   "assumed: every independence in the data reflects the causal structure, not cancelling effects.",
                   testable=False, source=KEY),
        Assumption("explore_acyclic_no_selection", "No selection bias / feedback loops (acyclicity)", Status.ASSUMED,
                   "The rows are not selected on a common effect of the variables, and no variable causes "
                   "itself through a loop.", testable=False, source=KEY),
        Assumption("explore_ci_tests", "Conditional independence tests are adequate (linear-Gaussian Fisher-z)",
                   Status.ASSUMED, "Links are found with Fisher-z tests, which only detect linear relations "
                   "and assume roughly Gaussian data; non-linear links can be missed.", testable=False, source=KEY),
    ]
    if lingam_assump is not None:
        assumptions.append(lingam_assump)

    nodes = [{"id": v, "role": "variable"} for v in names]
    charts = [Chart(kind="dag", title="What may cause what (FCI, stability-checked)",
                    data={"nodes": nodes, "edges": drawn},
                    note="Arrow = stable cause->effect; both arrowheads = hidden common cause; circle end "
                         "(partial) = that end is not known; plain line = linked, direction unknown. "
                         "Confidence = share of subsamples agreeing.")]
    if pairs:
        ranked = sorted(pairs, key=lambda q: -q["stability"]["adjacency"])[:30]
        charts.append(Chart(kind="bar", title="How often each link appeared across subsamples",
                            data={"labels": [f"{q['a']} – {q['b']}" for q in ranked],
                                  "values": [q["stability"]["adjacency"] for q in ranked],
                                  "ref": ADJ_STABLE, "errors": None},
                            x_label="link", y_label="share of subsamples",
                            note=f"Links below {ADJ_STABLE:.0%} are not drawn."))

    metrics = {
        "variables": names, "n_rows": n, "n_rows_dropped_missing": n_dropped,
        "left_out": vinfo["left_out"], "dropped_duplicates": vinfo["dropped_duplicates"],
        "time_like": vinfo["time_like"], "id_like": vinfo["id_like"],
        "alpha": ALPHA, "test": "fisherz",
        "fci_edges": fci_edges, "pc_edges": pc_edges, "lingam_edges": lingam_edges,
        "drawn_edges": drawn, "unstable_edges": unstable, "isolated": isolated,
        "n_directed": n_dir, "n_uncertain": n_unc,
        "stability_rule": (f"adjacency drawn if in >= {ADJ_STABLE:.0%} of {done} subsamples "
                           f"({SUB_FRAC[0]:.0%}-{SUB_FRAC[1]:.0%} of rows); an arrow only if FCI gives a "
                           f"tail-arrow and that exact orientation appears in >= {ORIENT_STABLE:.0%}; "
                           "otherwise downgraded to partial/undirected"),
        "n_subsamples": done, "pairs": pairs, "runtime_s": round(time.time() - t0, 2),
    }
    return Section(key=KEY, title=TITLE, status=None, summary=summary, metrics=metrics,
                   assumptions=assumptions, charts=charts, warnings=warnings)


def _count(str_cnt, i, j, s) -> int:
    return str_cnt.get((i, j), {}).get(s, 0)


def _lingam(Z: np.ndarray, names: list[str], rng):
    import lingam
    X = Z if len(Z) <= 2000 else Z[rng.choice(len(Z), 2000, replace=False)]
    model = lingam.DirectLiNGAM(random_state=SEED)
    model.fit(X)
    B = np.asarray(model.adjacency_matrix_)          # B[i, j]: effect of j on i
    resid = X - X @ B.T
    gauss = []
    pvals = {}
    for k, v in enumerate(names):
        e = resid[:, k]
        pv = float(stats.shapiro(e[:2000]).pvalue) if np.std(e) > 0 else 1.0
        pvals[v] = pv
        gauss.append(pv > ALPHA)
    n_g = int(sum(gauss))
    gnames = [v for v, g in zip(names, gauss) if g]
    if n_g <= 1:
        a = Assumption("explore_lingam_nongaussian", "Non-Gaussian noise (needed for LiNGAM directions)",
                       Status.PASS, "Shapiro-Wilk on LiNGAM residuals: " +
                       (f"only '{gnames[0]}' looks Gaussian; " if gnames else "none look Gaussian; ") +
                       "LiNGAM's directions are identifiable.", testable=True, source=KEY)
    else:
        a = Assumption("explore_lingam_nongaussian", "Non-Gaussian noise (needed for LiNGAM directions)",
                       Status.CONCERN, f"Shapiro-Wilk on LiNGAM residuals: {n_g} residuals look Gaussian "
                       f"({', '.join(gnames)}). LiNGAM cannot tell direction between Gaussian-noise variables, "
                       "so its directions involving them are shown as unreliable.", testable=True, source=KEY)
    a_extra = {"shapiro_p": pvals}
    a.detail += f" (p-values: {', '.join(f'{k}={v:.2g}' for k, v in a_extra['shapiro_p'].items())})"
    return B, gauss, a


def _lingam_edge(frm, to, coef, gauss, i_from, i_to, global_ok) -> dict:
    reliable = not (gauss[i_from] and gauss[i_to]) and (global_ok or not (gauss[i_from] or gauss[i_to]))
    return {"from": frm, "to": to, "coef_std": float(coef), "reliable": bool(reliable)}


def _agreement(t, frm, to, a, b, pc_s, lg_note) -> str:
    if t is None:
        return "not drawn"
    bits = []
    if pc_s is None:
        bits.append("PC finds no link")
    else:
        pc_dir = {"-->": (a, b), "<--": (b, a)}.get(pc_s)
        if pc_dir is None:
            bits.append("PC finds a link without direction")
        elif t in ("directed", "partial") and pc_dir == (to, frm):
            bits.append(f"PC points the other way ({to} -> {frm})")
        elif t in ("directed", "partial"):
            bits.append("PC agrees on the direction")
        else:
            bits.append(f"PC orients it {pc_dir[0]} -> {pc_dir[1]}")
    if lg_note is None:
        bits.append("LiNGAM finds no link")
    else:
        lf, lt = [s.strip() for s in lg_note.replace(" (unreliable)", "").split("->")]
        if t in ("directed", "partial") and (lt, lf) == (frm, to):
            bits.append(f"LiNGAM points the other way ({lg_note})")
        elif t in ("directed", "partial"):
            bits.append(f"LiNGAM agrees ({lg_note})")
        else:
            bits.append(f"LiNGAM suggests {lg_note}")
    return "; ".join(bits)


def _label(t, conf, flags) -> str:
    base = {"directed": "stable arrow", "partial": "one end uncertain", "bidirected": "hidden common cause",
            "undirected": "direction unknown"}[t]
    if "near_duplicate" in flags:
        base = "near-duplicate"
    return f"{base} ({conf:.0%})" if conf is not None else base


def _reading(a, b, t, frm, to, conf, flags, note, af) -> str:
    if "near_duplicate" in flags:
        return (f"{a} and {b} are almost the same measurement (correlation above {NEAR_DUP}); "
                "this link says nothing about causation.")
    if t is None:
        return f"{a} and {b}: no stable link ({note or f'linked in only {af:.0%} of subsamples'})."
    pct = f"{conf:.0%}" if conf is not None else "?"
    if t == "directed":
        s = (f"{frm} appears to cause {to} (same orientation in {pct} of subsamples), if the listed "
             "assumptions hold.")
    elif t == "partial":
        s = (f"{to} does not appear to cause {frm}; whether {frm} causes {to} or a hidden factor drives both "
             f"is uncertain ({pct} of subsamples).")
    elif t == "bidirected":
        s = f"{a} and {b} look driven by a hidden common cause; neither seems to cause the other ({pct})."
    else:
        s = (f"{a} and {b} are linked, but the data cannot say which way or whether a hidden factor drives "
             f"both (linked in {pct} of subsamples).")
    if note:
        s += f" Note: {note}."
    if "time_or_id" in flags:
        s += " One of them looks like time or an identifier, so treat this link as a trend, not a cause."
    return s
