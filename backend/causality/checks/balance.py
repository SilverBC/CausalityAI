"""Balance between groups.

Per-column standardized mean differences (binary comparison) or correlations (continuous
comparison), shown against what chance alone would produce, and one joint permutation test
of "do the covariates predict the comparison variable?" (strata: permute within strata;
cluster: permute whole clusters).

The comparison depends on the design:
  experiment / observational  -> treatment
  iv                          -> the instrument (it should look as-if random)
  rd                          -> side of the cutoff, in a window around it
  did                         -> ever-treated vs never-treated units (one row per unit)

Under stated randomization, a column that treatment predicts very strongly is reported as
suspected post-treatment (metrics["suspected_post_treatment"]); never otherwise.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..schema import Assumption, Chart, Claim, Design, Prepared, Section, Status

KEY = "balance"
TITLE = "Balance between groups"
N_PERM = 1000
MAX_BARS = 40


# --------------------------------------------------------------------------- public
def run(prep: Prepared, design: Design, claim: Optional[Claim] = None) -> Section:
    try:
        return _run(prep, design, claim)
    except Exception as e:  # noqa: BLE001
        return Section(key=KEY, title=TITLE, error=f"The balance check could not be completed ({type(e).__name__}: {e}).",
                       metrics={"suspected_post_treatment": []})


# --------------------------------------------------------------------------- helpers
def _post_cols(prep: Prepared) -> list[str]:
    r = prep.roles
    skip = {r.treatment, r.outcome}
    out = []
    for orig in r.post_treatment:
        for c in prep.encoding.get(orig, [orig]):
            if c in prep.df and c not in skip and c not in out and pd.api.types.is_numeric_dtype(prep.df[c]):
                out.append(c)
    return out


def smd_and_p(x: np.ndarray, g: np.ndarray) -> tuple[float, float]:
    a, b = x[g == 1], x[g == 0]
    if len(a) < 2 or len(b) < 2:
        return 0.0, 1.0
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if not np.isfinite(sd) or sd <= 0:
        return 0.0, 1.0
    d = float((a.mean() - b.mean()) / sd)
    with np.errstate(all="ignore"):
        p = stats.ttest_ind(a, b, equal_var=False).pvalue
    return d, float(p) if np.isfinite(p) else 1.0


def auc_single(x: np.ndarray, g: np.ndarray) -> float:
    """AUC of one column separating the groups (direction-free: >= 0.5)."""
    n1, n0 = int((g == 1).sum()), int((g == 0).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    r = stats.rankdata(x)
    auc = (r[g == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return float(max(auc, 1 - auc))


def _corr_and_p(x: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    if np.std(x) == 0 or np.std(t) == 0:
        return 0.0, 1.0
    r, p = stats.pearsonr(x, t)
    return float(r), float(p)


def _residualize_on_groups(M: np.ndarray, groups: Optional[np.ndarray]) -> np.ndarray:
    """Demean each column within groups (or overall)."""
    if groups is None:
        return M - M.mean(axis=0, keepdims=True)
    out = M.astype(float).copy()
    for gv in np.unique(groups):
        idx = groups == gv
        out[idx] -= out[idx].mean(axis=0, keepdims=True)
    return out


def joint_test(X: np.ndarray, g: np.ndarray, strata: Optional[np.ndarray] = None,
               cluster: Optional[np.ndarray] = None, n_perm: int = N_PERM, seed: int = 0) -> dict:
    """Permutation test of the omnibus statistic n*R^2 from regressing g on X (after removing
    strata means). Permutes g within strata, or whole clusters if g is constant within clusters
    (otherwise within clusters). Returns {"stat", "p", "method"}."""
    rng = np.random.default_rng(seed)
    n = len(g)
    Xr = _residualize_on_groups(X, strata)
    keep = Xr.std(axis=0) > 1e-12
    Xr = Xr[:, keep]
    if Xr.shape[1] == 0:
        return {"stat": 0.0, "p": 1.0, "method": "no covariates vary"}
    Q, R = np.linalg.qr(Xr)
    rank = int((np.abs(np.diag(R)) > 1e-9 * max(1.0, np.abs(R).max())).sum())
    Q = Q[:, np.abs(np.diag(R)) > 1e-9 * max(1.0, np.abs(R).max())]

    def stat_of(gm: np.ndarray) -> np.ndarray:  # gm: n x B
        gc = _residualize_on_groups(gm, strata)
        ss = (gc ** 2).sum(axis=0)
        proj = ((Q.T @ gc) ** 2).sum(axis=0)
        with np.errstate(all="ignore"):
            return np.where(ss > 0, n * proj / ss, 0.0)

    obs = float(stat_of(g.reshape(-1, 1).astype(float))[0])

    method = "permutation test of all covariates together"
    cluster_level = False
    if cluster is not None:
        const = pd.Series(g).groupby(cluster).nunique().max() <= 1
        cluster_level = bool(const)
        method += ", permuting whole clusters" if cluster_level else ", permuting within clusters"
    elif strata is not None:
        method += ", permuting within strata"

    # group index arrays for permutation
    perm_groups = None
    if cluster is not None and not cluster_level:
        perm_groups = cluster
    elif strata is not None:
        perm_groups = strata
    if cluster_level:
        cl_codes, cl_uni = pd.factorize(pd.Series(cluster))
        cl_g = pd.Series(g).groupby(cl_codes).first().to_numpy()
        cl_strata = None
        if strata is not None:
            cl_strata = pd.Series(strata).groupby(cl_codes).first().to_numpy()
    idx_groups = None
    if perm_groups is not None and not cluster_level:
        idx_groups = [np.flatnonzero(perm_groups == v) for v in np.unique(perm_groups)]

    chunk = max(1, min(200, int(4e6 // max(n, 1))))
    count, done = 0, 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        if cluster_level:
            G = np.tile(cl_g, (b, 1))
            if cl_strata is None:
                G = rng.permuted(G, axis=1)
            else:
                for v in np.unique(cl_strata):
                    ii = np.flatnonzero(cl_strata == v)
                    G[:, ii] = rng.permuted(G[:, ii], axis=1)
            gm = G[:, cl_codes].T
        elif idx_groups is not None:
            G = np.tile(g.astype(float), (b, 1))
            for ii in idx_groups:
                G[:, ii] = rng.permuted(G[:, ii], axis=1)
            gm = G.T
        else:
            gm = rng.permuted(np.tile(g.astype(float), (b, 1)), axis=1).T
        s = stat_of(gm)
        count += int((s >= obs - 1e-12).sum())
        done += b
    p = (1 + count) / (1 + n_perm)
    # asymptotic reference (chi-square with rank df) for very small permutation p
    p_asym = float(stats.chi2.sf(obs, rank)) if rank > 0 else 1.0
    return {"stat": obs, "p": float(p), "p_asymptotic": p_asym, "df": rank, "method": method}


def _status_from_p(p: float) -> Status:
    if p >= 0.05:
        return Status.PASS
    if p >= 0.01:
        return Status.CONCERN
    return Status.FAIL


# --------------------------------------------------------------------------- main
def _run(prep: Prepared, design: Design, claim: Optional[Claim]) -> Section:
    r = prep.roles
    df = prep.df
    kind = design.kind if design else "observational"
    if not r.treatment or r.treatment not in df:
        return Section(key=KEY, title=TITLE, error="No treatment column is set, so groups cannot be compared.",
                       metrics={"suspected_post_treatment": []})

    adj = [c for c in prep.adjustment_set if c in df and pd.api.types.is_numeric_dtype(df[c])]
    post = [c for c in _post_cols(prep) if c not in adj]
    warnings: list[str] = []
    notes: list[str] = []

    # ---- choose the comparison variable and rows
    data = df[df[r.treatment].notna()].copy()
    comp_col = r.treatment
    comparison = "treatment"
    if kind == "iv" and r.instrument and r.instrument in df:
        comp_col = r.instrument
        comparison = "instrument"
        data = df[df[r.instrument].notna()].copy()
        adj = [c for c in adj if c != r.instrument]
    elif kind == "rd" and r.running and r.running in df and r.cutoff is not None:
        x = df[r.running]
        d = df[x.notna()].copy()
        dist = (d[r.running] - r.cutoff).abs()
        h = float(np.quantile(dist, 0.3))
        win = d[dist <= h].copy()
        side = (win[r.running] >= r.cutoff).astype(float)
        if min(side.sum(), (1 - side).sum()) < 10:
            win = d.copy()
            side = (win[r.running] >= r.cutoff).astype(float)
            h = float(dist.max())
        win["__side__"] = side
        data = win
        comp_col = "__side__"
        comparison = "side of the cutoff"
        notes.append(f"Units within {h:.4g} of the cutoff {r.cutoff:g} on '{r.running}' are compared "
                     "(above vs below). Jumps in covariates at the cutoff are tested in the design checks.")
        adj = [c for c in adj if c != r.running]
    elif kind == "did" and r.unit and r.unit in df:
        d = df[df[r.treatment].notna()].copy()
        if r.time and r.time in d:
            d = d.sort_values(r.time)
        ever = d.groupby(r.unit)[r.treatment].max().rename("__ever__")
        first = d.groupby(r.unit)[adj + post].first() if (adj + post) else pd.DataFrame(index=ever.index)
        data = first.join(ever)
        data[r.unit] = data.index
        comp_col = "__ever__"
        comparison = "ever-treated vs never-treated units"
        notes.append("Units that are ever treated are compared with units that never are, using each unit's "
                     "first observed period. Level differences are allowed by difference-in-differences; "
                     "what matters is whether trends are parallel, tested in the design checks.")

    g_all = data[comp_col].to_numpy(dtype=float)
    vals = set(np.unique(g_all[~np.isnan(g_all)]))
    binary = vals <= {0.0, 1.0} and len(vals) == 2

    strata = data[r.strata].to_numpy() if (r.strata and r.strata in data and kind != "did") else None
    cluster = None
    if kind == "did":
        cluster = None
    elif r.cluster and r.cluster in data:
        cluster = data[r.cluster].to_numpy()

    n = len(data)
    metrics: dict = {"comparison": comparison, "comparison_column": comp_col if not comp_col.startswith("__") else None,
                     "n": int(n), "suspected_post_treatment": [], "post_treatment_shown": post}

    if n < 6 or len(vals) < 2:
        return Section(key=KEY, title=TITLE, status=None, metrics=metrics,
                       summary=f"The {comparison} does not vary in the data, so there are no groups to compare.",
                       assumptions=_assumptions(kind, None, None))

    rows = []  # per column
    cols = adj + post
    if binary:
        g = g_all.astype(int)
        n1, n0 = int((g == 1).sum()), int((g == 0).sum())
        band = 1.96 * np.sqrt(1 / max(n1, 1) + 1 / max(n0, 1))
        metrics.update({"n_treated": n1, "n_control": n0})
        for c in cols:
            x = data[c].to_numpy(dtype=float)
            ok = ~np.isnan(x)
            d, p = smd_and_p(x[ok], g[ok])
            auc = auc_single(x[ok], g[ok])
            rows.append({"column": c, "value": d, "p": p, "auc": auc, "post_treatment": c in post})
        stat_name = "standardized mean difference"
    else:
        t = g_all
        band = 1.96 / np.sqrt(max(n - 3, 1))
        for c in cols:
            x = data[c].to_numpy(dtype=float)
            ok = ~np.isnan(x)
            rr, p = _corr_and_p(x[ok], t[ok])
            rows.append({"column": c, "value": rr, "p": p, "auc": None, "post_treatment": c in post})
        stat_name = "correlation with the " + ("treatment" if comparison == "treatment" else comparison)

    # ---- suspected post-treatment (only under stated randomization, binary treatment, treatment comparison)
    suspected: list[str] = []
    if r.randomized and binary and comparison == "treatment":
        for row in rows:
            if row["post_treatment"]:
                continue
            strong = (abs(row["value"]) > 0.5 and row["p"] < 1e-6) or (row["auc"] is not None and row["auc"] > 0.75 and row["p"] < 1e-3)
            if strong:
                suspected.append(row["column"])
        for c in suspected:
            row = next(x for x in rows if x["column"] == c)
            warnings.append(f"'{c}' differs between treated and control far more than randomization allows "
                            f"(standardized difference {row['value']:+.2f}). It was most likely measured after "
                            "treatment, so it is left out of the adjustment set.")
    metrics["suspected_post_treatment"] = suspected

    # ---- joint test (pre-treatment covariates only, not suspected ones)
    test_cols = [c for c in adj if c not in suspected]
    jt = None
    if test_cols:
        sub = data[test_cols + [comp_col]].dropna()
        idx = sub.index
        X = sub[test_cols].to_numpy(dtype=float)
        gg = sub[comp_col].to_numpy(dtype=float)
        st = data.loc[idx, r.strata].to_numpy() if strata is not None else None
        cl = data.loc[idx, r.cluster].to_numpy() if cluster is not None else None
        n_perm = N_PERM if len(sub) <= 20000 else 400
        jt = joint_test(X, gg, st, cl, n_perm=n_perm)
        metrics.update({"joint_p": jt["p"], "joint_stat": jt["stat"], "joint_df": jt["df"],
                        "joint_p_asymptotic": jt["p_asymptotic"], "joint_test": jt["method"]})

    # ---- chance comparison
    pre_rows = [x for x in rows if not x["post_treatment"] and x["column"] not in suspected]
    n_beyond = int(sum(abs(x["value"]) > band for x in pre_rows))
    expected = 0.05 * len(pre_rows)
    metrics.update({"statistic": stat_name, "chance_band": float(band), "n_columns": len(pre_rows),
                    "n_beyond_chance": n_beyond, "expected_beyond_chance": float(expected),
                    "per_column": {x["column"]: x["value"] for x in rows},
                    "per_column_p": {x["column"]: x["p"] for x in rows},
                    "max_abs": float(max((abs(x["value"]) for x in pre_rows), default=0.0))})

    # ---- status
    status: Optional[Status]
    joint_status = _status_from_p(jt["p"]) if jt else None
    if jt is None:
        status = None
    elif kind == "experiment":
        status = joint_status
    elif kind == "iv":
        status = joint_status
    elif kind in ("rd", "did"):
        status = Status.PASS if jt["p"] >= 0.05 else Status.CONCERN
    else:  # observational: imbalance is expected; describes the comparison
        status = Status.CONCERN

    # ---- summary
    unit_word = "standardized difference" if binary else "correlation"
    top = sorted(pre_rows, key=lambda x: -abs(x["value"]))[:3]
    top_txt = ", ".join(f"'{x['column']}' ({x['value']:+.2f})" for x in top if abs(x["value"]) > band)
    parts = []
    if not pre_rows:
        parts.append("There are no measured pre-treatment covariates to compare.")
    else:
        parts.append(f"{n_beyond} of {len(pre_rows)} covariate(s) differ by more than chance alone would usually "
                     f"produce (|{unit_word}| > {band:.2f}); about {expected:.1f} would by chance.")
        if top_txt:
            parts.append(f"Largest: {top_txt}.")
        if jt:
            parts.append(f"Taken together, the covariates {'do' if jt['p'] < 0.05 else 'do not'} predict the "
                         f"{comparison} (p {_fmt_p(jt['p'])}).")
    if kind == "experiment" and jt:
        if jt["p"] >= 0.05:
            parts.append("This is what randomization should produce.")
        else:
            parts.append("This is more imbalance than randomization usually produces; check how assignment was done.")
    elif kind == "observational" and pre_rows:
        parts.append("Without randomization differences are expected; adjustment handles measured differences, "
                     "not unmeasured ones.")
    elif kind == "iv" and jt:
        parts.append("An instrument that looks as-if random should be balanced on covariates."
                     if jt["p"] >= 0.05 else
                     "Covariates differ by instrument value, so the instrument may not be as-if random.")
    if post:
        parts.append("Columns you marked as measured after treatment are shown but not used.")
    if suspected:
        parts.append(f"Suspected post-treatment: {', '.join(suspected)}.")

    # ---- chart
    chart_rows = sorted(rows, key=lambda x: -abs(x["value"]))[:MAX_BARS]
    labels = []
    for x in chart_rows:
        lab = x["column"]
        if x["post_treatment"]:
            lab += " (after treatment)"
        elif x["column"] in suspected:
            lab += " (suspected after treatment)"
        labels.append(lab)
    charts = [Chart(kind="bar", title=f"Differences between groups ({unit_word})",
                    data={"labels": labels, "values": [float(x["value"]) for x in chart_rows],
                          "ref": float(band), "band": [float(-band), float(band)], "errors": None},
                    x_label="covariate", y_label=unit_word,
                    note=f"Bars beyond ±{band:.2f} exceed what chance alone usually produces; "
                         f"about 5% of columns ({expected:.1f}) would by chance.")] if rows else []

    return Section(key=KEY, title=TITLE, status=status, summary=" ".join(notes + parts), metrics=metrics,
                   assumptions=_assumptions(kind, joint_status, jt), charts=charts, warnings=warnings)


def _fmt_p(p: float) -> str:
    return f"= {p:.3f}" if p >= 0.001 else "< 0.001"


def _assumptions(kind: str, joint_status: Optional[Status], jt: Optional[dict]) -> list[Assumption]:
    out: list[Assumption] = []
    if kind == "experiment":
        if joint_status is not None:
            out.append(Assumption("randomization_balance", "Randomization produced comparable groups", joint_status,
                                  f"Joint test that measured covariates predict treatment: p {_fmt_p(jt['p'])}.",
                                  True, "balance"))
        out.append(Assumption("randomization_carried_out", "Randomization was carried out as described", Status.ASSUMED,
                              "Balance on measured columns is consistent with randomization but cannot prove it; "
                              "how treatment was actually assigned cannot be seen in the data.", False, "balance"))
    elif kind == "iv" and joint_status is not None:
        out.append(Assumption("instrument_balance", "The instrument is unrelated to measured covariates", joint_status,
                              f"Joint test that measured covariates predict the instrument: p {_fmt_p(jt['p'])}. "
                              "This says nothing about unmeasured ones.", True, "balance"))
    return out
