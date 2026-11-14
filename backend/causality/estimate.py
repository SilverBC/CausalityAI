"""The effect estimate for the study design.

`estimate(prep, design, claim) -> Estimate`. One estimator per design, always with a
90% and a 95% interval, plus comparison estimators in `comparisons`. Never raises: any
failure comes back as a refused Estimate with a plain-words reason.

Methods (see each function for details):
  experiment     Lin (2013) covariate-adjusted regression, HC2 robust SEs, cluster-robust
                 SEs if a cluster column is given, strata fixed effects if strata given.
  did            Callaway & Sant'Anna (2021) group-time ATT(g,t), unconditional 2x2 DiD per
                 cell, never-treated controls (not-yet-treated if too few never-treated),
                 simple and event-study aggregations, unit (cluster) bootstrap SEs.
                 Two-way fixed effects as a comparison.
  iv             2SLS with robust (or cluster) SEs; Anderson-Rubin set by exact inversion
                 of the robust AR test (may be unbounded or two pieces); first-stage F.
  rd             Local linear, triangular kernel, Imbens-Kalyanaraman bandwidth; robust
                 bias-corrected interval (Calonico-Cattaneo-Titiunik idea); fuzzy if the
                 treatment share does not jump by (almost) 1.
  observational  Cross-fitted AIPW (5 folds), propensity logistic / gradient boosting,
                 outcome linear / gradient boosting (picked per nuisance by out-of-fold
                 loss), propensities clipped to [0.01, 0.99], influence-function SEs.
  continuous T   Partially linear double ML slope + adjusted spline dose-response curve
                 with shape tests.
"""
from __future__ import annotations

import math
import os
import time
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .schema import Chart, Claim, Design, Estimate, Prepared

# joblib (used by sklearn/econml) shells out to count cores on Windows; avoid the noisy failure
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))

SEED = 0
N_BOOT = 199
MIN_N = 20
MIN_GROUP = 5


# =============================================================================== public
def estimate(prep: Prepared, design: Design, claim: Optional[Claim] = None) -> Estimate:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return _estimate(prep, design, claim)
    except Exception as e:  # noqa: BLE001
        return _refuse("Estimate", _target_for(design), "full",
                       f"The estimate could not be computed ({type(e).__name__}: {e}).")


def _estimate(prep: Prepared, design: Design, claim: Optional[Claim]) -> Estimate:
    r = prep.roles
    kind = design.kind if design else "observational"
    if not r.treatment or not r.outcome or r.treatment not in prep.df or r.outcome not in prep.df:
        return _refuse("Estimate", _target_for(design), "full",
                       "A treatment column and an outcome column are both needed for an estimate.")
    d = prep.complete()
    if len(d) < MIN_N:
        return _refuse("Estimate", _target_for(design), "full",
                       f"Only {len(d)} rows have both treatment and outcome observed; at least {MIN_N} are needed.",
                       n=len(d))
    y = d[r.outcome].to_numpy(float)
    if np.nanstd(y) == 0:
        return _refuse("Estimate", _target_for(design), "full",
                       "The outcome takes a single value in the rows used, so any interval would have zero width.",
                       n=len(d))
    binary = prep.treatment_is_binary()
    if kind == "did":
        return _did(prep, d, claim)
    if kind == "iv":
        return _iv(prep, d, claim)
    if kind == "rd":
        return _rd(prep, d, claim)
    if not binary:
        return _dose(prep, d, claim, kind)
    if kind == "experiment":
        return _experiment(prep, d, claim)
    return _observational(prep, d, claim)


def _target_for(design: Optional[Design]) -> str:
    return {"experiment": "ATE", "did": "ATT", "iv": "LATE", "rd": "local_at_cutoff",
            "observational": "ATE"}.get(getattr(design, "kind", ""), "ATE")


# =============================================================================== helpers
def _f(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _refuse(method, target, scope, reason, n=0, **kw) -> Estimate:
    e = Estimate(method=method, target=target, scope=scope, n=int(n), refused=True, refusal_reason=reason)
    for k, v in kw.items():
        setattr(e, k, v)
    return e


def _crit(level: float, dfree: Optional[int]) -> float:
    q = 0.5 + level / 2
    return float(stats.t.ppf(q, dfree)) if dfree else float(stats.norm.ppf(q))


def _ci(est, se, level, dfree=None):
    c = _crit(level, dfree)
    return (float(est - c * se), float(est + c * se))


CI_LEVELS = (0.90, 0.95, 0.98, 0.99)


def _set_ci_levels(e: Estimate, centre, se, dfree=None) -> None:
    """extra["ci_levels"]: intervals at 90/95/98/99% by the same rule as ci90/ci95 (same centre,
    SE and reference distribution). The "90" and "95" entries are e.ci90 / e.ci95 themselves."""
    out = {}
    for lv in CI_LEVELS:
        key = str(int(round(lv * 100)))
        ci = e.ci90 if key == "90" else (e.ci95 if key == "95" else _ci(centre, se, lv, dfree))
        out[key] = [float(ci[0]), float(ci[1])]
    e.extra["ci_levels"] = out


def _pval(est, se, dfree=None):
    z = abs(est / se)
    return float(2 * stats.t.sf(z, dfree)) if dfree else float(2 * stats.norm.sf(z))


def _finish(e: Estimate, est, se, dfree=None) -> Estimate:
    """Fill estimate/se/CIs/p, or refuse if the interval is degenerate."""
    est_f, se_f = _f(est), _f(se)
    if est_f is None:
        e.refused, e.refusal_reason, e.estimate = True, "The estimate is not a finite number.", None
        return e
    scale = 1.0 + abs(est_f) + (e.outcome_sd or 0.0)
    if se_f is None or se_f <= 1e-10 * scale:
        e.refused = True
        e.refusal_reason = ("The uncertainty could not be measured (the interval would have zero or infinite width), "
                            "so no estimate is reported.")
        e.estimate = None
        e.extra["unreported_point_estimate"] = est_f
        return e
    e.estimate, e.se = est_f, se_f
    e.ci90, e.ci95 = _ci(est_f, se_f, 0.90, dfree), _ci(est_f, se_f, 0.95, dfree)
    e.p_value = _pval(est_f, se_f, dfree)
    _set_ci_levels(e, est_f, se_f, dfree)
    if dfree:
        e.extra["t_df"] = int(dfree)
    return e


def _comp(method, est, se, note="", dfree=None) -> dict:
    est, se = _f(est), _f(se)
    ci = _ci(est, se, 0.95, dfree) if (est is not None and se is not None and se > 0) else None
    return {"method": method, "estimate": est, "se": se, "ci95": ci, "note": note}


def _covariates(prep: Prepared, d: pd.DataFrame, drop: tuple = ()) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in prep.adjustment_set if c in d.columns and c not in drop]
    keep = []
    for c in cols:
        v = d[c]
        if v.notna().all() and v.nunique() > 1:
            keep.append(c)
    if not keep:
        return np.zeros((len(d), 0)), []
    return d[keep].to_numpy(float), keep


def _groups(d: pd.DataFrame, col: Optional[str]) -> Optional[np.ndarray]:
    if not col or col not in d.columns:
        return None
    g = d[col].to_numpy(float).copy()
    miss = ~np.isfinite(g)
    if miss.any():  # missing cluster labels: each its own cluster
        start = np.nanmax(g) + 1 if (~miss).any() else 0
        g[miss] = start + np.arange(miss.sum())
    _, codes = np.unique(g, return_inverse=True)
    return codes


def _sandwich(X: np.ndarray, resid: np.ndarray, groups=None, hc2=True):
    """OLS coefficients' robust covariance. HC2 if no groups, else CR1 cluster."""
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    if groups is None:
        if hc2:
            h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
            denom = 1 - h
            if np.any(denom <= 1e-10):
                w = resid ** 2 * n / max(n - k, 1)        # fall back to HC1
            else:
                w = resid ** 2 / denom
        else:
            w = resid ** 2 * n / max(n - k, 1)
        meat = (X * w[:, None]).T @ X
    else:
        G = int(groups.max()) + 1
        S = np.zeros((G, k))
        np.add.at(S, groups, X * resid[:, None])
        meat = S.T @ S * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return XtX_inv @ meat @ XtX_inv


def _ols(y, X, groups=None):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    V = _sandwich(X, resid, groups)
    return beta, V, resid


def _cluster_df(groups) -> Optional[int]:
    return int(groups.max()) if groups is not None else None  # G - 1


def _outcome_sd(y, t=None, binary=False) -> Optional[float]:
    if binary and t is not None and (t == 0).sum() >= 2:
        return _f(np.std(y[t == 0], ddof=1))
    return _f(np.std(y, ddof=1))


def _forest(e: Estimate, title: str, ref: float = 0.0) -> Optional[Chart]:
    rows = []
    if e.estimate is not None and e.ci95:
        rows.append({"label": e.method, "est": e.estimate, "lo": e.ci95[0], "hi": e.ci95[1]})
    for c in e.comparisons:
        if c.get("estimate") is not None and c.get("ci95"):
            rows.append({"label": c["method"], "est": c["estimate"], "lo": c["ci95"][0], "hi": c["ci95"][1]})
    if not rows:
        return None
    return Chart("forest", title, {"rows": rows, "ref": ref}, x_label="Effect (outcome units)")


def _cluster_note(groups, label) -> str:
    G = int(groups.max()) + 1
    s = f"Standard errors allow for correlation within the {G} clusters of '{label}'."
    if G < 20:
        s += f" With only {G} clusters these standard errors can be too small."
    return s


# =============================================================================== experiment
def _experiment(prep: Prepared, d: pd.DataFrame, claim) -> Estimate:
    r = prep.roles
    t = d[r.treatment].to_numpy(float)
    y = d[r.outcome].to_numpy(float)
    n = len(y)
    n1, n0 = int(t.sum()), int((1 - t).sum())
    e = Estimate(method="Covariate-adjusted comparison (Lin 2013)", target="ATE", scope="full", n=n)
    e.baseline_mean = _f(y[t == 0].mean()) if n0 else None
    e.outcome_sd = _outcome_sd(y, t, True)
    if min(n1, n0) < MIN_GROUP:
        e.refused, e.refusal_reason = True, f"Only {min(n1, n0)} unit(s) in the smaller group; at least {MIN_GROUP} are needed."
        return e
    notes = []
    groups = _groups(d, r.cluster)
    drop = (r.strata,) if r.strata else ()
    X, names = _covariates(prep, d, drop)
    if r.strata and r.strata in d.columns:
        sd = pd.get_dummies(d[r.strata].astype("category"), drop_first=True, dtype=float)
        sd = sd.loc[:, sd.nunique() > 1]
        if sd.shape[1]:
            X = np.column_stack([X, sd.to_numpy()])
            names = names + [f"stratum={c}" for c in sd.columns]
            notes.append(f"Strata fixed effects included ({sd.shape[1] + 1} strata).")
    Xc = X - X.mean(axis=0) if X.shape[1] else X
    interact = X.shape[1] > 0 and 2 * X.shape[1] + 2 <= max(n1, n0) / 2 and 2 * X.shape[1] + 2 <= n / 5
    if interact:
        D = np.column_stack([np.ones(n), t, Xc, t[:, None] * Xc])
    else:
        D = np.column_stack([np.ones(n), t, Xc]) if X.shape[1] else np.column_stack([np.ones(n), t])
        if X.shape[1]:
            notes.append("Too many covariates for the sample size to interact them with treatment; "
                         "used plain covariate adjustment instead.")
            e.method = "Covariate-adjusted regression"
    if not X.shape[1]:
        e.method = "Difference in means (regression, robust SEs)"
    beta, V, _ = _ols(y, D, groups)
    dfree = _cluster_df(groups)
    se = math.sqrt(V[1, 1]) if V[1, 1] >= 0 else float("nan")
    _finish(e, beta[1], se, dfree)
    # unadjusted comparison
    D0 = np.column_stack([np.ones(n), t])
    b0, V0, _ = _ols(y, D0, groups)
    e.comparisons.append(_comp("Unadjusted difference in means", b0[1], math.sqrt(max(V0[1, 1], 0)),
                               "No covariates; in a randomized experiment this is also unbiased.", dfree))
    se_rule = "HC2 robust"
    if groups is not None:
        se_rule = "cluster-robust (CR1) by " + r.cluster
        notes.append(_cluster_note(groups, r.cluster))
        e.extra["n_clusters"] = int(groups.max()) + 1
    e.extra.update({"covariates": names, "se_type": se_rule, "n_treated": n1, "n_control": n0,
                    "treated_mean": _f(y[t == 1].mean()), "notes": notes})
    ch = _forest(e, "Effect estimates")
    if ch:
        e.charts.append(ch)
    return e


# =============================================================================== nuisance models
def _folds(n, k=5, seed=SEED, strat=None):
    from sklearn.model_selection import KFold, StratifiedKFold
    if strat is not None:
        try:
            return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(n), strat))
        except ValueError:
            pass
    return list(KFold(k, shuffle=True, random_state=seed).split(np.zeros(n)))


def _use_gbm(n, p):
    return n >= 300 and p >= 1


def _xfit_reg(X, y, folds, gbm: bool, fit_mask=None):
    """Cross-fitted regression predictions for all rows; models fit on rows of the training
    folds with fit_mask True. Returns dict name -> predictions."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import HistGradientBoostingRegressor
    n = len(y)
    fit_mask = np.ones(n, bool) if fit_mask is None else fit_mask
    out = {"linear": np.zeros(n)}
    if gbm:
        out["gbm"] = np.zeros(n)
    for tr, te in folds:
        tr = tr[fit_mask[tr]]
        if X.shape[1] == 0 or len(tr) < X.shape[1] + 2:
            m = y[tr].mean() if len(tr) else y[fit_mask].mean()
            for k in out:
                out[k][te] = m
            continue
        out["linear"][te] = LinearRegression().fit(X[tr], y[tr]).predict(X[te])
        if gbm:
            g = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                              min_samples_leaf=20, random_state=SEED)
            out["gbm"][te] = g.fit(X[tr], y[tr]).predict(X[te])
    return out


def _xfit_clf(X, t, folds, gbm: bool):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    n = len(t)
    out = {"logistic": np.zeros(n)}
    if gbm:
        out["gbm"] = np.zeros(n)
    for tr, te in folds:
        if X.shape[1] == 0 or len(np.unique(t[tr])) < 2:
            for k in out:
                out[k][te] = t[tr].mean()
            continue
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=2000))
        out["logistic"][te] = lr.fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
        if gbm:
            g = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=15,
                                               min_samples_leaf=20, random_state=SEED)
            out["gbm"][te] = g.fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
    return out


def _pick(preds: dict, truth, mask=None, loss="mse"):
    mask = np.ones(len(truth), bool) if mask is None else mask
    best, best_l = None, np.inf
    for k, p in preds.items():
        if loss == "mse":
            l = np.mean((truth[mask] - p[mask]) ** 2)
        else:
            q = np.clip(p[mask], 1e-6, 1 - 1e-6)
            l = -np.mean(truth[mask] * np.log(q) + (1 - truth[mask]) * np.log(1 - q))
        if l < best_l - 1e-12:
            best, best_l = k, l
    return best, preds[best]


def _if_se(psi, groups=None):
    n = len(psi)
    c = psi - psi.mean()
    if groups is None:
        return float(np.sqrt(np.sum(c ** 2) / (n * (n - 1))))
    G = int(groups.max()) + 1
    s = np.bincount(groups, weights=c, minlength=G)
    return float(np.sqrt(np.sum(s ** 2) * G / max(G - 1, 1)) / n)


# =============================================================================== observational
def _observational(prep: Prepared, d: pd.DataFrame, claim) -> Estimate:
    r = prep.roles
    notes = []
    scope, target = "full", "ATE"
    trimmed = 0
    try:
        from .checks.overlap import trim_mask  # written by another module
        m = trim_mask(prep)
        m = pd.Series(m).reindex(d.index).fillna(True).astype(bool)
        trimmed = int((~m).sum())
        if trimmed > 0 and m.sum() >= MIN_N:
            d = d.loc[m]
            scope, target = "overlap_trimmed", "ATE in the overlap region"
            notes.append(f"{trimmed} row(s) with almost no comparable units in the other group were set aside; "
                         "the estimate is for the remaining rows (the overlap region), not everyone.")
        elif trimmed > 0:
            trimmed = 0
            notes.append("Trimming to the overlap region would leave too few rows, so all rows were kept.")
    except Exception:  # noqa: BLE001
        trimmed = 0
    t = d[r.treatment].to_numpy(float)
    y = d[r.outcome].to_numpy(float)
    n = len(y)
    n1, n0 = int(t.sum()), int(n - t.sum())
    e = Estimate(method="Doubly robust (AIPW, cross-fitted)", target=target, scope=scope, n=n)
    e.outcome_sd = _outcome_sd(y, t, True)
    if n < MIN_N or min(n1, n0) < MIN_GROUP:
        e.refused = True
        e.refusal_reason = f"Too few rows ({n}) or too few in the smaller group ({min(n1, n0)}) to estimate."
        return e
    groups = _groups(d, r.cluster)
    X, names = _covariates(prep, d)
    gbm = _use_gbm(n, X.shape[1])
    folds = _folds(n, 5, SEED, strat=t.astype(int))
    ps_all = _xfit_clf(X, t, folds, gbm)
    ps_name, ps = _pick(ps_all, t, loss="logloss")
    m1_all = _xfit_reg(X, y, folds, gbm, fit_mask=t == 1)
    m0_all = _xfit_reg(X, y, folds, gbm, fit_mask=t == 0)
    m1_name, m1 = _pick(m1_all, y, mask=t == 1)
    m0_name, m0 = _pick(m0_all, y, mask=t == 0)
    n_clip = int(((ps < 0.01) | (ps > 0.99)).sum())
    ps = np.clip(ps, 0.01, 0.99)
    if n_clip:
        notes.append(f"{n_clip} propensity score(s) were clipped to [0.01, 0.99].")
    a1 = m1 + t * (y - m1) / ps
    a0 = m0 + (1 - t) * (y - m0) / (1 - ps)
    psi = a1 - a0
    tau = psi.mean()
    se = _if_se(psi, groups)
    dfree = _cluster_df(groups)
    _finish(e, tau, se, dfree)
    e.baseline_mean = _f(a0.mean())
    # comparisons ----------------------------------------------------------------
    D = np.column_stack([np.ones(n), t, X - X.mean(0)]) if X.shape[1] else np.column_stack([np.ones(n), t])
    b, V, _ = _ols(y, D, groups)
    e.comparisons.append(_comp("Regression adjustment (OLS)", b[1], math.sqrt(max(V[1, 1], 0)),
                               "Assumes the outcome is linear in the covariates.", dfree))
    w1, w0 = t / ps, (1 - t) / (1 - ps)
    mu1, mu0 = np.sum(w1 * y) / np.sum(w1), np.sum(w0 * y) / np.sum(w0)
    psi_ipw = w1 * (y - mu1) / np.mean(w1) - w0 * (y - mu0) / np.mean(w0)
    e.comparisons.append(_comp("Inverse probability weighting", mu1 - mu0, _if_se(psi_ipw, groups),
                               "Relies only on the propensity model; SE ignores propensity estimation.", dfree))
    b0, V0, _ = _ols(y, np.column_stack([np.ones(n), t]), groups)
    e.comparisons.append(_comp("Naive difference in means", b0[1], math.sqrt(max(V0[1, 1], 0)),
                               "No adjustment at all; shown only for contrast.", dfree))
    if X.shape[1]:
        start = time.time()
        try:
            from econml.dml import LinearDML
            from sklearn.linear_model import LinearRegression, LogisticRegression
            dml = LinearDML(model_y=LinearRegression(), model_t=LogisticRegression(C=1e4, max_iter=2000),
                            discrete_treatment=True, cv=3, random_state=SEED)
            dml.fit(y, t.astype(int), X=None, W=X)
            lo, hi = dml.ate_interval(alpha=0.05)
            pt = float(dml.ate())
            if time.time() - start < 20:
                c = {"method": "Double ML (EconML LinearDML)", "estimate": _f(pt),
                     "se": _f((float(hi) - float(lo)) / (2 * 1.959964)),
                     "ci95": (float(lo), float(hi)), "note": "Linear nuisance models; a cross-check."}
                e.comparisons.append(c)
        except Exception:  # noqa: BLE001
            pass
    e.extra.update({
        "covariates": names, "propensity_model": ps_name, "outcome_models": {"treated": m1_name, "control": m0_name},
        "folds": 5, "n_treated": n1, "n_control": n0, "n_trimmed": trimmed, "propensity_clip": [0.01, 0.99],
        "n_clipped": n_clip, "se_type": "influence function" + (f", clustered by {r.cluster}" if groups is not None else ""),
        "mean_y1": _f(a1.mean()), "mean_y0": _f(a0.mean()), "notes": notes,
    })
    if groups is not None:
        e.extra["n_clusters"] = int(groups.max()) + 1
        notes.append(_cluster_note(groups, r.cluster))
    ch = _forest(e, "Effect estimates")
    if ch:
        e.charts.append(ch)
    return e


# =============================================================================== continuous treatment
def _rcs_basis(x, knots):
    """Restricted cubic spline nonlinear terms (Harrell), scaled."""
    k = np.asarray(knots, float)
    K = len(k)
    scale = (k[-1] - k[0]) ** 2
    cols = []
    for j in range(K - 2):
        c = (np.clip(x - k[j], 0, None) ** 3
             - np.clip(x - k[-2], 0, None) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
             + np.clip(x - k[-1], 0, None) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2]))
        cols.append(c / scale)
    return np.column_stack(cols) if cols else np.zeros((len(x), 0))


def _dose(prep: Prepared, d: pd.DataFrame, claim, kind) -> Estimate:
    r = prep.roles
    t = d[r.treatment].to_numpy(float)
    y = d[r.outcome].to_numpy(float)
    n = len(y)
    e = Estimate(method="Partially linear double ML (effect per unit of treatment)", target="dose_response",
                 scope="full", n=n)
    e.outcome_sd = _outcome_sd(y)
    e.baseline_mean = _f(y[t == 0].mean()) if (t == 0).sum() >= MIN_GROUP else _f(y[t <= np.quantile(t, 0.1)].mean())
    if np.unique(t).size < 3:
        e.refused, e.refusal_reason = True, "The treatment takes fewer than three values; a dose-response cannot be estimated."
        return e
    notes = []
    groups = _groups(d, r.cluster)
    dfree = _cluster_df(groups)
    X, names = _covariates(prep, d)
    gbm = _use_gbm(n, X.shape[1])
    folds = _folds(n, 5, SEED)
    my_name, my = _pick(_xfit_reg(X, y, folds, gbm), y)
    mt_name, mt = _pick(_xfit_reg(X, t, folds, gbm), t)
    yr, tr = y - my, t - mt
    denom = np.mean(tr ** 2)
    theta = np.sum(tr * yr) / np.sum(tr ** 2)
    psi = (yr - theta * tr) * tr / denom
    se = _if_se(psi + theta, groups)
    _finish(e, theta, se, dfree)
    # comparisons
    b, V, _ = _ols(y, np.column_stack([np.ones(n), t]), groups)
    e.comparisons.append(_comp("Naive regression slope", b[1], math.sqrt(max(V[1, 1], 0)), "No adjustment.", dfree))
    if X.shape[1]:
        b, V, _ = _ols(y, np.column_stack([np.ones(n), t, X]), groups)
        e.comparisons.append(_comp("Linear regression slope with covariates", b[1], math.sqrt(max(V[1, 1], 0)),
                                   "Assumes everything is linear.", dfree))
    # dose-response curve: Y ~ linear T + restricted cubic spline(T) + covariates ----------
    uq = np.unique(t)
    shape = {}
    curve = None
    nk = 4 if uq.size >= 8 else (3 if uq.size >= 5 else 0)
    knots = np.unique(np.quantile(t, {4: [0.05, 0.35, 0.65, 0.95], 3: [0.1, 0.5, 0.9], 0: []}[nk])) if nk else []
    if len(knots) < 3:
        knots = []
    B = _rcs_basis(t, knots) if len(knots) else np.zeros((n, 0))
    Xm = X.mean(0) if X.shape[1] else np.zeros(0)
    Dm = np.column_stack([np.ones(n), t, B, X - Xm]) if X.shape[1] else np.column_stack([np.ones(n), t, B])
    bs, Vs, _ = _ols(y, Dm, groups)
    k_nl = B.shape[1]
    lo_q, hi_q = np.quantile(t, [0.02, 0.98])
    grid = np.linspace(lo_q, hi_q, 40)

    def rows(xg):
        Bg = _rcs_basis(xg, knots) if len(knots) else np.zeros((len(xg), 0))
        R = np.column_stack([np.ones(len(xg)), xg, Bg])
        return np.column_stack([R, np.zeros((len(xg), Dm.shape[1] - R.shape[1]))])

    Rg = rows(grid)
    fg = Rg @ bs
    sg = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", Rg, Vs, Rg), 0, None))
    zc = _crit(0.95, dfree)
    curve = {"x": grid.tolist(), "y": fg.tolist(), "lo": (fg - zc * sg).tolist(), "hi": (fg + zc * sg).tolist()}
    if k_nl:
        idx = np.arange(2, 2 + k_nl)
        bnl = bs[idx]
        Vnl = Vs[np.ix_(idx, idx)]
        try:
            W = float(bnl @ np.linalg.solve(Vnl, bnl))
            p_lin = float(stats.f.sf(W / k_nl, k_nl, dfree)) if dfree else float(stats.chi2.sf(W, k_nl))
        except np.linalg.LinAlgError:
            W, p_lin = None, None
        shape["linearity_test"] = {"stat": _f(W), "df": k_nl, "p_value": _f(p_lin),
                                   "method": "robust Wald test that the spline's curved terms are zero"}
        shape["linear_rejected_5pct"] = bool(p_lin is not None and p_lin < 0.05)
    else:
        shape["linearity_test"] = None
        shape["linear_rejected_5pct"] = None
        notes.append("Too few distinct treatment values to test the shape of the dose-response.")
    # slopes in the lower and upper half (linear contrasts of the curve)
    q05, q50, q95 = np.quantile(t, [0.05, 0.5, 0.95])
    if q50 > q05 and q95 > q50:
        R3 = rows(np.array([q05, q50, q95]))
        cl = (R3[1] - R3[0]) / (q50 - q05)
        ch_ = (R3[2] - R3[1]) / (q95 - q50)
        sl, sh = cl @ bs, ch_ @ bs
        se_l, se_h = math.sqrt(max(cl @ Vs @ cl, 0)), math.sqrt(max(ch_ @ Vs @ ch_, 0))
        dd = ch_ - cl
        se_d = math.sqrt(max(dd @ Vs @ dd, 0))
        p_diff = _pval(sh - sl, se_d, dfree) if se_d > 0 else None
        shape["slope_low"] = {"est": _f(sl), "ci95": _ci(sl, se_l, 0.95, dfree) if se_l > 0 else None,
                              "range": [float(q05), float(q50)]}
        shape["slope_high"] = {"est": _f(sh), "ci95": _ci(sh, se_h, 0.95, dfree) if se_h > 0 else None,
                               "range": [float(q50), float(q95)]}
        shape["slope_change_p_value"] = _f(p_diff)
        deriv = np.diff(fg)
        tol = 0.05 * np.max(np.abs(deriv)) if np.max(np.abs(deriv)) > 0 else 0
        pos, neg = np.mean(deriv > tol), np.mean(deriv < -tol)
        mono = "increasing" if neg == 0 and pos > 0 else ("decreasing" if pos == 0 and neg > 0 else "not monotone")
        shape["monotone_hint"] = mono
        lin_ok = not shape.get("linear_rejected_5pct")
        lo_zero = shape["slope_low"]["ci95"] is not None and shape["slope_low"]["ci95"][0] <= 0 <= shape["slope_low"]["ci95"][1]
        hi_nonzero = shape["slope_high"]["ci95"] is not None and not (shape["slope_high"]["ci95"][0] <= 0 <= shape["slope_high"]["ci95"][1])
        if lin_ok:
            hint = "linear (no evidence of curvature)"
        elif mono == "not monotone":
            hint = "nonmonotone"
        elif lo_zero and hi_nonzero:
            hint = "threshold (flat at low doses, then changes)"
        elif abs(sh) < abs(sl) and (p_diff is not None and p_diff < 0.05):
            hint = "diminishing (the slope flattens at higher doses)"
        elif abs(sh) > abs(sl) and (p_diff is not None and p_diff < 0.05):
            hint = "increasing returns (the slope steepens at higher doses)"
        else:
            hint = "curved, but the pattern is not clear"
        shape["shape_hint"] = hint
    e.extra.update({
        "dose_curve": curve, "shape_tests": shape, "covariates": names,
        "nuisance_models": {"outcome": my_name, "treatment": mt_name}, "folds": 5,
        "spline_knots": [float(k) for k in knots],
        "se_type": "influence function" + (f", clustered by {r.cluster}" if groups is not None else ""),
        "treatment_range": [float(t.min()), float(t.max())], "notes": notes,
        "curve_method": "OLS of outcome on treatment + restricted cubic spline of treatment + covariates (robust SEs); "
                        "curve shown at the covariate means, pointwise 95% intervals.",
    })
    if groups is not None:
        e.extra["n_clusters"] = int(groups.max()) + 1
    e.charts.append(Chart("line", "Dose-response (adjusted)", {
        "series": [{"name": "Predicted outcome", "x": curve["x"], "y": curve["y"], "lo": curve["lo"], "hi": curve["hi"]}],
        "vline": None, "hline": None}, x_label=r.treatment, y_label=r.outcome,
        note="Pointwise 95% intervals; covariates held at their averages."))
    return e


# =============================================================================== DiD
def _did(prep: Prepared, d: pd.DataFrame, claim) -> Estimate:
    r = prep.roles
    method = "Callaway-Sant'Anna group-time DiD"
    if not r.unit or not r.time or r.unit not in d.columns or r.time not in d.columns:
        return _refuse(method, "ATT", "treated",
                       "Difference-in-differences needs both a unit column and a time column; at least one is missing.")
    if not prep.treatment_is_binary():
        return _refuse(method, "ATT", "treated",
                       "This difference-in-differences estimator needs an on/off treatment; the treatment has more than two values.")
    notes = []
    dd = d[[r.unit, r.time, r.treatment, r.outcome] + ([r.cluster] if r.cluster and r.cluster in d.columns else [])].dropna(subset=[r.unit, r.time])
    if dd.duplicated(subset=[r.unit, r.time]).any():
        notes.append("Some unit-period pairs appear more than once; their outcomes were averaged.")
    times = np.sort(dd[r.time].unique())
    T_ = len(times)
    if T_ < 2:
        return _refuse(method, "ATT", "treated", "Only one time period is present.")
    tpos = {v: i for i, v in enumerate(times)}
    dd = dd.assign(_tp=dd[r.time].map(tpos))
    units = np.sort(dd[r.unit].unique())
    upos = {v: i for i, v in enumerate(units)}
    dd = dd.assign(_up=dd[r.unit].map(upos))
    U = len(units)
    Y = np.full((U, T_), np.nan)
    agg = dd.groupby(["_up", "_tp"])[r.outcome].mean()
    Y[agg.index.get_level_values(0), agg.index.get_level_values(1)] = agg.to_numpy()
    D = np.full((U, T_), np.nan)
    aggd = dd.groupby(["_up", "_tp"])[r.treatment].max()
    D[aggd.index.get_level_values(0), aggd.index.get_level_values(1)] = aggd.to_numpy()
    # first treated period (position), inf = never
    G = np.full(U, np.inf)
    for i in range(U):
        on = np.where(D[i] == 1)[0]
        if on.size:
            G[i] = on[0]
            after = D[i, on[0]:]
            if np.any(after[np.isfinite(after)] == 0):
                pass
    switch_off = sum(1 for i in range(U) if np.isfinite(G[i]) and np.any(D[i, int(G[i]):] == 0))
    if switch_off:
        notes.append(f"{switch_off} unit(s) turn treatment off again; they are treated as adopting at their first "
                     "treated period (the method assumes treatment stays on).")
    always = np.isfinite(G) & (G == 0)
    if always.any():
        notes.append(f"{int(always.sum())} unit(s) are treated from the first period, so they have no before-period "
                     "and are left out.")
    never = ~np.isfinite(G)
    cohorts = sorted(int(g) for g in np.unique(G[np.isfinite(G) & (G > 0)]))
    if not cohorts:
        return _refuse(method, "ATT", "treated", "No unit switches from untreated to treated during the data, so there is no before/after comparison.")
    use_never = never.sum() >= MIN_GROUP
    control_rule = "never treated" if use_never else "not yet treated"
    if not use_never:
        notes.append(f"Only {int(never.sum())} never-treated unit(s); units not yet treated serve as the comparison group.")
    # cluster for bootstrap: cluster column if constant within unit, else unit
    boot_groups = np.arange(U)
    boot_label = r.unit
    if r.cluster and r.cluster in dd.columns and r.cluster != r.unit:
        cu = dd.groupby("_up")[r.cluster].nunique()
        if (cu <= 1).all():
            cl = dd.groupby("_up")[r.cluster].first().reindex(range(U)).to_numpy(float)
            _, boot_groups = np.unique(np.nan_to_num(cl, nan=-1), return_inverse=True)
            boot_label = r.cluster
    # cells ---------------------------------------------------------------------
    cells = []  # (g, t, e, dvec, treat_mask, ctrl_mask)
    for g in cohorts:
        for tt in range(T_):
            if tt == g - 1:
                continue
            base = g - 1 if tt >= g else tt - 1
            if base < 0:
                continue
            diff = Y[:, tt] - Y[:, base]
            tm = (G == g) & np.isfinite(diff)
            if use_never:
                cm = never & np.isfinite(diff)
            else:
                cm = (G > max(tt, base)) & (G != g) & np.isfinite(diff)
            if tm.sum() < 1 or cm.sum() < 2:
                continue
            cells.append((g, tt, tt - g, np.nan_to_num(diff), tm.astype(float), cm.astype(float)))
    post = [c for c in cells if c[1] >= c[0]]
    if not post:
        return _refuse(method, "ATT", "treated", "No treated cohort has a usable comparison group after treatment starts.")
    gsize = {g: float(((G == g)).sum()) for g in cohorts}

    def compute(w):
        """w: weights per unit (len U). Returns (att_cells, overall, event dict)."""
        atts = np.empty(len(cells))
        for j, (g, tt, ev, dv, tm, cm) in enumerate(cells):
            wt, wc = w * tm, w * cm
            st, sc = wt.sum(), wc.sum()
            atts[j] = (wt @ dv) / st - (wc @ dv) / sc if st > 0 and sc > 0 else np.nan
        gw = {g: float((w * (G == g)).sum()) for g in cohorts}
        num = den = 0.0
        ev_num, ev_den = {}, {}
        for j, (g, tt, ev, *_r) in enumerate(cells):
            if not np.isfinite(atts[j]):
                continue
            if tt >= g:
                num += gw[g] * atts[j]
                den += gw[g]
            ev_num[ev] = ev_num.get(ev, 0.0) + gw[g] * atts[j]
            ev_den[ev] = ev_den.get(ev, 0.0) + gw[g]
        overall = num / den if den > 0 else np.nan
        evs = {k: ev_num[k] / ev_den[k] for k in ev_num if ev_den[k] > 0}
        return atts, overall, evs

    w0 = np.ones(U)
    atts, overall, evs = compute(w0)
    # bootstrap by cluster (unit)
    rng = np.random.default_rng(SEED)
    Gc = int(boot_groups.max()) + 1
    boots, boot_cells, boot_ev = [], [], {k: [] for k in evs}
    for _ in range(N_BOOT):
        cw = rng.multinomial(Gc, np.full(Gc, 1.0 / Gc)).astype(float)
        w = cw[boot_groups]
        a, o, ev = compute(w)
        boots.append(o)
        boot_cells.append(a)
        for k in boot_ev:
            boot_ev[k].append(ev.get(k, np.nan))
    boots = np.array(boots)
    se = float(np.nanstd(boots, ddof=1)) if np.isfinite(boots).sum() > 20 else float("nan")
    e = Estimate(method=method + f" ({control_rule} comparison)", target="ATT", scope="treated",
                 n=int(len(dd)))
    yall = d[r.outcome].to_numpy(float)
    tall = d[r.treatment].to_numpy(float)
    e.outcome_sd = _outcome_sd(yall, tall, True)
    _finish(e, overall, se)
    # baseline: treated units' untreated-outcome mean in post cells = observed mean - ATT(g,t)
    bnum = bden = 0.0
    for j, (g, tt, ev, dv, tm, cm) in enumerate(cells):
        if tt >= g and np.isfinite(atts[j]):
            obs = np.nanmean(Y[(G == g) & np.isfinite(Y[:, tt]), tt])
            bnum += gsize[g] * (obs - atts[j])
            bden += gsize[g]
    e.baseline_mean = _f(bnum / bden) if bden else None
    # event study
    ev_keys = sorted(evs)
    ev_rows = []
    zc = _crit(0.95, None)
    for k in ev_keys:
        sb = np.array(boot_ev[k])
        s = float(np.nanstd(sb, ddof=1)) if np.isfinite(sb).sum() > 20 else float("nan")
        ev_rows.append({"event_time": int(k), "att": _f(evs[k]), "se": _f(s),
                        "lo": _f(evs[k] - zc * s), "hi": _f(evs[k] + zc * s)})
    boot_cells = np.array(boot_cells)
    gt_rows = []
    for j, (g, tt, ev, *_r) in enumerate(cells):
        s = float(np.nanstd(boot_cells[:, j], ddof=1))
        gt_rows.append({"group": _f(times[g]), "time": _f(times[tt]), "event_time": int(ev),
                        "att": _f(atts[j]), "se": _f(s)})
    pre = [row for row in ev_rows if row["event_time"] < 0 and row["se"]]
    pre_test = None
    if pre:
        # joint test of pre-period ATTs = 0 using bootstrap covariance
        idx = [ev_keys.index(row["event_time"]) for row in pre]
        M = np.array([boot_ev[ev_keys[i]] for i in idx]).T
        M = M[np.isfinite(M).all(1)]
        v = np.array([row["att"] for row in pre])
        try:
            S = np.cov(M, rowvar=False).reshape(len(v), len(v))
            W = float(v @ np.linalg.solve(S, v))
            pre_test = {"stat": W, "df": len(v), "p_value": float(stats.chi2.sf(W, len(v))),
                        "note": "Joint test that all before-treatment effects are zero (a check on parallel trends)."}
        except Exception:  # noqa: BLE001
            pre_test = None
    # TWFE comparison ------------------------------------------------------------
    twfe = _twfe(dd, r)
    if twfe:
        b, s, Gt = twfe
        note = "Two-way fixed effects (unit + period), clustered by unit."
        if e.estimate is not None and s and e.se and abs(b - e.estimate) > 1.96 * math.sqrt(s ** 2 + e.se ** 2) * 0.5:
            note += (" It differs noticeably from the Callaway-Sant'Anna estimate: with staggered adoption and effects "
                     "that change over time or across cohorts, two-way fixed effects can weight some comparisons "
                     "negatively and be biased.")
        e.comparisons.append(_comp("Two-way fixed effects", b, s, note, Gt - 1))
    e.extra.update({
        "control_group": control_rule, "n_units": U, "n_periods": T_, "cohorts": [_f(times[g]) for g in cohorts],
        "cohort_sizes": {str(_f(times[g])): int(gsize[g]) for g in cohorts}, "n_never_treated": int(never.sum()),
        "group_time_att": gt_rows, "event_study": ev_rows, "pretrend_test": pre_test,
        "aggregation": "simple: average of post-treatment ATT(g,t) weighted by cohort size",
        "se_type": f"bootstrap over {boot_label} ({N_BOOT} reps, seed {SEED})", "n_clusters": int(Gc),
        "covariates": [], "notes": notes + ["Comparisons are unconditional 2x2 differences (no covariates), so parallel "
                                            "trends is assumed without covariate adjustment."],
    })
    if ev_rows:
        e.charts.append(Chart("line", "Effect by time since treatment started", {
            "series": [{"name": "ATT", "x": [row["event_time"] for row in ev_rows],
                        "y": [row["att"] for row in ev_rows], "lo": [row["lo"] for row in ev_rows],
                        "hi": [row["hi"] for row in ev_rows]}],
            "vline": -0.5, "hline": 0.0}, x_label="Periods since treatment started", y_label=f"Effect on {r.outcome}",
            note="Negative times are before treatment and should be near zero if trends were parallel."))
    ch = _forest(e, "Effect estimates")
    if ch:
        e.charts.append(ch)
    return e


def _twfe(dd: pd.DataFrame, r) -> Optional[tuple]:
    try:
        y = dd[r.outcome].to_numpy(float)
        t = dd[r.treatment].to_numpy(float)
        u = dd["_up"].to_numpy(int)
        p = dd["_tp"].to_numpy(int)
        yd, td = y.copy(), t.copy()
        for _ in range(100):
            prev = td.copy()
            for code in (u, p):
                for v in (yd, td):
                    m = np.bincount(code, weights=v) / np.maximum(np.bincount(code), 1)
                    v -= m[code]
            if np.max(np.abs(td - prev)) < 1e-10:
                break
        stt = np.sum(td ** 2)
        if stt <= 0:
            return None
        b = np.sum(td * yd) / stt
        res = yd - b * td
        G = int(u.max()) + 1
        s = np.bincount(u, weights=td * res, minlength=G)
        V = np.sum(s ** 2) / stt ** 2 * G / max(G - 1, 1)
        return float(b), float(math.sqrt(V)), G
    except Exception:  # noqa: BLE001
        return None


# =============================================================================== IV
def _resid_on(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(W, A, rcond=None)
    return A - W @ coef


def _ar_set(yt, dt, zt, groups, level, c_scale):
    """Robust Anderson-Rubin confidence set for one endogenous regressor and one instrument,
    by exact inversion (the test statistic is a ratio of quadratics in beta).
    Inputs are already residualized on the exogenous regressors. Returns list of [lo, hi]."""
    S = zt @ zt
    a1, b1 = zt @ yt, zt @ dt
    ry = yt - (a1 / S) * zt
    rd = dt - (b1 / S) * zt
    if groups is None:
        s_yy, s_yd, s_dd = np.sum(zt ** 2 * ry ** 2), np.sum(zt ** 2 * ry * rd), np.sum(zt ** 2 * rd ** 2)
    else:
        G = int(groups.max()) + 1
        gy = np.bincount(groups, weights=zt * ry, minlength=G)
        gd = np.bincount(groups, weights=zt * rd, minlength=G)
        s_yy, s_yd, s_dd = gy @ gy, gy @ gd, gd @ gd
    k = float(stats.chi2.ppf(level, 1)) * c_scale
    qa = b1 ** 2 - k * s_dd
    qb = -2 * a1 * b1 + 2 * k * s_yd
    qc = a1 ** 2 - k * s_yy
    tol = 1e-12 * max(abs(b1 ** 2), k * s_dd, 1e-300)
    inf = float("inf")
    if abs(qa) <= tol:
        if abs(qb) < 1e-300:
            return [[-inf, inf]] if qc <= 0 else []
        root = -qc / qb
        return [[-inf, root]] if qb > 0 else [[root, inf]]
    disc = qb ** 2 - 4 * qa * qc
    if qa > 0:
        if disc < 0:
            return []
        s = math.sqrt(disc)
        r1, r2 = sorted([(-qb - s) / (2 * qa), (-qb + s) / (2 * qa)])
        return [[r1, r2]]
    if disc < 0:
        return [[-inf, inf]]
    s = math.sqrt(disc)
    r1, r2 = sorted([(-qb - s) / (2 * qa), (-qb + s) / (2 * qa)])
    return [[-inf, r1], [r2, inf]]


def _describe_set(sets) -> str:
    if not sets:
        return "empty (the instrument model itself looks rejected)"
    if any(math.isinf(a) or math.isinf(b) for a, b in sets):
        if len(sets) == 1 and math.isinf(sets[0][0]) and math.isinf(sets[0][1]):
            return "unbounded: every value is consistent with the data (the instrument is too weak to say anything)"
        return "unbounded: " + " or ".join(f"[{a:.4g}, {b:.4g}]" for a, b in sets) + " (the instrument is weak)"
    return " or ".join(f"[{a:.4g}, {b:.4g}]" for a, b in sets)


def _iv(prep: Prepared, d: pd.DataFrame, claim) -> Estimate:
    r = prep.roles
    method = "Two-stage least squares"
    if not r.instrument or r.instrument not in d.columns:
        return _refuse(method, "LATE", "compliers", "The instrument design needs an instrument column; none was given.")
    d = d.dropna(subset=[r.instrument])
    y = d[r.outcome].to_numpy(float)
    dv = d[r.treatment].to_numpy(float)
    z = d[r.instrument].to_numpy(float)
    n = len(y)
    e = Estimate(method=method, target="LATE", scope="compliers", n=n)
    binary_t = prep.treatment_is_binary()
    e.outcome_sd = _outcome_sd(y, dv, binary_t)
    if n < MIN_N:
        e.refused, e.refusal_reason = True, f"Only {n} usable rows; at least {MIN_N} are needed."
        return e
    if np.unique(z).size < 2:
        e.refused, e.refusal_reason = True, "The instrument takes a single value in the rows used."
        return e
    if set(np.unique(z)) <= {0, 1} and min((z == 1).sum(), (z == 0).sum()) < MIN_GROUP:
        e.refused, e.refusal_reason = True, "Too few units in one instrument group (fewer than 5)."
        return e
    notes = []
    groups = _groups(d, r.cluster)
    dfree = _cluster_df(groups)
    X, names = _covariates(prep, d, drop=(r.instrument,))
    W = np.column_stack([np.ones(n), X]) if X.shape[1] else np.ones((n, 1))
    Xm = np.column_stack([dv, W])
    Zm = np.column_stack([z, W])
    k = Xm.shape[1]
    # first stage
    fb, fV, _ = _ols(dv, Zm, groups)
    fs_F = float(fb[0] ** 2 / fV[0, 0]) if fV[0, 0] > 0 else float("nan")
    # 2SLS
    pi, *_ = np.linalg.lstsq(Zm, Xm, rcond=None)
    Xh = Zm @ pi
    beta, *_ = np.linalg.lstsq(Xh, y, rcond=None)
    u = y - Xm @ beta
    V = _sandwich(Xh, u, groups, hc2=False)
    se = math.sqrt(V[0, 0]) if V[0, 0] >= 0 else float("nan")
    _finish(e, beta[0], se, dfree)
    # AR sets
    yt, dt, zt = _resid_on(y, W), _resid_on(dv, W), _resid_on(z, W)
    c_scale = 1.0 / (n / max(n - Zm.shape[1], 1))  # small-sample scale applied to the variance
    if groups is not None:
        Gc = int(groups.max()) + 1
        c_scale = 1.0 / ((Gc / max(Gc - 1, 1)) * ((n - 1) / max(n - Zm.shape[1], 1)))
    ar95 = _ar_set(yt, dt, zt, groups, 0.95, 1.0 / c_scale)
    ar90 = _ar_set(yt, dt, zt, groups, 0.90, 1.0 / c_scale)
    ar98 = _ar_set(yt, dt, zt, groups, 0.98, 1.0 / c_scale)
    ar99 = _ar_set(yt, dt, zt, groups, 0.99, 1.0 / c_scale)
    bounded = bool(ar95) and not any(math.isinf(a) or math.isinf(b) for a, b in ar95)
    if fs_F < 10:
        notes.append(f"The instrument is weak (first-stage F = {fs_F:.1f} < 10); trust the Anderson-Rubin interval, "
                     "not the 2SLS interval.")
    if not bounded:
        notes.append("The Anderson-Rubin 95% set is unbounded" if ar95 else "The Anderson-Rubin 95% set is empty")
    # comparisons
    ob, oV, _ = _ols(y, np.column_stack([dv, W]), groups)
    e.comparisons.append(_comp("Naive regression (OLS)", ob[0], math.sqrt(max(oV[0, 0], 0)),
                               "Ignores the instrument; biased if treatment take-up is related to unmeasured factors.", dfree))
    rf_b, rf_V, _ = _ols(y, Zm, groups)
    # baseline: complier untreated mean (Abadie) for binary instrument & treatment, else untreated mean
    base = None
    if binary_t and set(np.unique(z)) <= {0, 1}:
        den = np.mean(1 - dv[z == 0]) - np.mean(1 - dv[z == 1])
        if abs(den) > 0.05:
            base = (np.mean(y[z == 0] * (1 - dv[z == 0])) - np.mean(y[z == 1] * (1 - dv[z == 1]))) / den
            e.extra["baseline_rule"] = "estimated untreated outcome mean among compliers"
    if base is None and binary_t and (dv == 0).sum():
        base = y[dv == 0].mean()
        e.extra["baseline_rule"] = "outcome mean among untreated units"
    e.baseline_mean = _f(base)
    e.extra.update({
        "first_stage_F": _f(fs_F), "first_stage_coef": _f(fb[0]), "first_stage_se": _f(math.sqrt(max(fV[0, 0], 0))),
        "weak_instrument": bool(not (fs_F >= 10)),
        "reduced_form": {"est": _f(rf_b[0]), "se": _f(math.sqrt(max(rf_V[0, 0], 0)))},
        "ar_ci95": ar95, "ar_ci90": ar90, "ar_ci95_bounded": bounded, "ar_ci95_text": _describe_set(ar95),
        "ar_ci90_text": _describe_set(ar90), "ar_ci98": ar98, "ar_ci99": ar99,
        "ar_method": "robust Anderson-Rubin test inverted exactly (the statistic is a ratio of quadratics in the effect)",
        "covariates": names, "instrument": r.instrument,
        "se_type": "heteroskedasticity-robust" + (f", clustered by {r.cluster}" if groups is not None else ""),
        "notes": notes,
    })
    if groups is not None:
        e.extra["n_clusters"] = int(groups.max()) + 1
        notes.append(_cluster_note(groups, r.cluster))
    rows = []
    if e.estimate is not None:
        rows.append({"label": "2SLS", "est": e.estimate, "lo": e.ci95[0], "hi": e.ci95[1]})
        if bounded and len(ar95) == 1:
            rows.append({"label": "Anderson-Rubin 95% set", "est": e.estimate, "lo": ar95[0][0], "hi": ar95[0][1]})
    for c in e.comparisons:
        if c["ci95"]:
            rows.append({"label": c["method"], "est": c["estimate"], "lo": c["ci95"][0], "hi": c["ci95"][1]})
    if rows:
        e.charts.append(Chart("forest", "Effect estimates", {"rows": rows, "ref": 0.0}, x_label="Effect (outcome units)"))
    return e


# =============================================================================== RD
def _wls_weights(X, w):
    """Rows of (X'WX)^-1 X'W (linear smoother weights for each coefficient)."""
    XtW = X.T * w
    return np.linalg.pinv(XtW @ X) @ XtW


def _tri(x, h):
    return np.clip(1 - np.abs(x) / h, 0, None)


def _ik_bandwidth(x, y):
    """Imbens & Kalyanaraman (2012) bandwidth for sharp RD, triangular kernel (C_K = 3.4375)."""
    N = len(x)
    left, right = x < 0, x >= 0
    h1 = 1.84 * np.std(x) * N ** (-0.2)
    l1, r1 = left & (x >= -h1), right & (x <= h1)
    if l1.sum() < 3 or r1.sum() < 3:
        h1 = np.quantile(np.abs(x), 0.5)
        l1, r1 = left & (x >= -h1), right & (x <= h1)
    f = (l1.sum() + r1.sum()) / (2 * N * h1)
    var = (np.sum((y[l1] - y[l1].mean()) ** 2) + np.sum((y[r1] - y[r1].mean()) ** 2)) / max(l1.sum() + r1.sum(), 1)
    # third derivative from a global cubic between the medians on each side
    ml, mr = np.median(x[left]), np.median(x[right])
    sel = (x >= ml) & (x <= mr)
    xs = x[sel]
    Xc = np.column_stack([np.ones(sel.sum()), (xs >= 0), xs, xs ** 2, xs ** 3])
    g, *_ = np.linalg.lstsq(Xc, y[sel], rcond=None)
    m3 = 6 * g[4]
    rng = max(np.max(np.abs(x)), 1e-12)
    m3 = m3 if abs(m3) > 1e-12 else 1e-12

    def h2(Ns):
        return 3.56 * (var / (f * m3 ** 2)) ** (1 / 7) * Ns ** (-1 / 7)

    out = {}
    for name, side in (("r", right), ("l", left)):
        hh = min(h2(side.sum()), rng)
        s = side & (np.abs(x) <= hh)
        if s.sum() < 5:
            s = side
        xs = x[s]
        Xq = np.column_stack([np.ones(s.sum()), xs, xs ** 2])
        c, *_ = np.linalg.lstsq(Xq, y[s], rcond=None)
        out[name] = (2 * c[2], s.sum(), hh)
    (m2r, n2r, h2r), (m2l, n2l, h2l) = out["r"], out["l"]
    rr = 2160 * var / (n2r * h2r ** 4)
    rl = 2160 * var / (n2l * h2l ** 4)
    h = 3.4375 * (2 * var / (f * ((m2r - m2l) ** 2 + rr + rl))) ** 0.2 * N ** (-0.2)
    return float(h)


def _rd_fit(x, y, tvec, h, b, groups, fuzzy, cutoff_side_treated):
    """Local linear at h (conventional) and robust bias-corrected (local quadratic bias at b >= h).
    Returns dict with conventional and robust estimates/SEs and side limits."""
    res = {}
    n_tot = len(x)
    omega_ll = np.zeros(n_tot)
    omega_bc = np.zeros(n_tot)
    lim = {}
    resid_y = np.zeros(n_tot)
    resid_t = np.zeros(n_tot)
    resid_y_ll = np.zeros(n_tot)
    resid_t_ll = np.zeros(n_tot)
    n_eff = 0
    for sgn, side in ((1, x >= 0), (-1, x < 0)):
        sh = side & (np.abs(x) < h)
        sb = side & (np.abs(x) < b)
        if sh.sum() < 4 or sb.sum() < 5:
            return None
        n_eff += int(sh.sum())
        xh = x[sh]
        Mll = _wls_weights(np.column_stack([np.ones(sh.sum()), xh]), _tri(xh, h))
        xb = x[sb]
        Mq = _wls_weights(np.column_stack([np.ones(sb.sum()), xb, xb ** 2]), _tri(xb, b))
        L = Mll[0] @ (xh ** 2)            # bias factor of the LL intercept per unit of the x^2 coefficient
        idx_h, idx_b = np.where(sh)[0], np.where(sb)[0]
        omega_ll[idx_h] += sgn * Mll[0]
        omega_bc[idx_h] += sgn * Mll[0]
        omega_bc[idx_b] -= sgn * L * Mq[2]
        lim[sgn] = float(Mll[0] @ y[sh])
        # residuals
        cq = Mq @ y[sb]
        resid_y[idx_b] = y[sb] - np.column_stack([np.ones(sb.sum()), xb, xb ** 2]) @ cq
        cl = Mll @ y[sh]
        resid_y_ll[idx_h] = y[sh] - np.column_stack([np.ones(sh.sum()), xh]) @ cl
        if fuzzy:
            cqt = Mq @ tvec[sb]
            resid_t[idx_b] = tvec[sb] - np.column_stack([np.ones(sb.sum()), xb, xb ** 2]) @ cqt
            clt = Mll @ tvec[sh]
            resid_t_ll[idx_h] = tvec[sh] - np.column_stack([np.ones(sh.sum()), xh]) @ clt
        lim[f"t{sgn}"] = float(Mll[0] @ tvec[sh])
    jy_ll, jy_bc = omega_ll @ y, omega_bc @ y
    jt_ll, jt_bc = omega_ll @ tvec, omega_bc @ tvec
    scale = n_eff / max(n_eff - 4, 1)

    def var(omega, ry, rt, tau, jt):
        infl = omega * (ry - tau * rt) / jt
        if groups is None:
            return float(np.sum(infl ** 2) * scale)
        G = int(groups.max()) + 1
        s = np.bincount(groups, weights=infl, minlength=G)
        return float(np.sum(s ** 2) * G / max(G - 1, 1))

    if fuzzy:
        if abs(jt_ll) < 1e-8 or abs(jt_bc) < 1e-8:
            return None
        tau_ll, tau_bc = jy_ll / jt_ll, jy_bc / jt_bc
        v_ll = var(omega_ll, resid_y_ll, resid_t_ll, tau_ll, jt_ll)
        v_bc = var(omega_bc, resid_y, resid_t, tau_bc, jt_bc)
    else:
        sgn = 1.0 if cutoff_side_treated >= 0 else -1.0
        tau_ll, tau_bc = sgn * jy_ll, sgn * jy_bc
        v_ll = var(omega_ll, resid_y_ll, np.zeros(n_tot), 0.0, 1.0)
        v_bc = var(omega_bc, resid_y, np.zeros(n_tot), 0.0, 1.0)
    res.update(tau=float(tau_ll), se=math.sqrt(v_ll), tau_bc=float(tau_bc), se_rb=math.sqrt(v_bc),
               left=lim[-1], right=lim[1], t_left=lim["t-1"], t_right=lim["t1"], n_eff=n_eff,
               jump_y=float(jy_ll), jump_t=float(jt_ll))
    return res


def _rd(prep: Prepared, d: pd.DataFrame, claim) -> Estimate:
    r = prep.roles
    method = "Local linear regression discontinuity"
    if not r.running or r.running not in d.columns or r.cutoff is None:
        return _refuse(method, "local_at_cutoff", "local_cutoff",
                       "The cutoff design needs a running variable and a cutoff value.")
    d = d.dropna(subset=[r.running])
    c0 = float(r.cutoff)
    x = d[r.running].to_numpy(float) - c0
    y = d[r.outcome].to_numpy(float)
    tv = d[r.treatment].to_numpy(float)
    groups = _groups(d, r.cluster)
    n = len(y)
    e = Estimate(method=method, target="local_at_cutoff", scope="local_cutoff", n=n)
    if (x < 0).sum() < MIN_GROUP or (x >= 0).sum() < MIN_GROUP or n < MIN_N:
        e.refused, e.refusal_reason = True, "Too few units on one side of the cutoff."
        return e
    notes = []
    h = _ik_bandwidth(x, y)
    xr = max(np.max(x), -np.min(x))
    if not np.isfinite(h) or h <= 0:
        h = 0.5 * xr
    # at least 10 obs per side within h
    for _ in range(50):
        if ((x < 0) & (x > -h)).sum() >= 10 and ((x >= 0) & (x < h)).sum() >= 10:
            break
        h *= 1.25
    h = float(min(h, xr * 1.0001))
    b = float(min(max(h * n ** (2 / 35), h), xr * 1.0001))
    # sharp or fuzzy
    near = np.abs(x) < h
    share_jump = tv[near & (x >= 0)].mean() - tv[near & (x < 0)].mean()
    fuzzy = abs(share_jump) < 0.99
    if fuzzy:
        e.method = "Fuzzy regression discontinuity (local linear, ratio of jumps)"
        notes.append(f"Treatment share changes by {share_jump:+.2f} at the cutoff (not a full switch), so the effect is "
                     "the outcome jump divided by the treatment jump: it applies to units whose treatment the cutoff changed.")
    fit = _rd_fit(x, y, tv, h, b, groups, fuzzy, share_jump)
    if fit is None:
        e.refused, e.refusal_reason = True, "Too few units near the cutoff to fit a line on each side."
        return e
    e.n = int(fit["n_eff"])
    treated_right = share_jump >= 0
    e.baseline_mean = _f(fit["left"] if treated_right else fit["right"])
    ys = y[near & ((x < 0) if treated_right else (x >= 0))]
    e.outcome_sd = _f(np.std(ys, ddof=1)) if ys.size > 2 else _outcome_sd(y)
    # point = conventional; interval = robust bias-corrected
    dfree = _cluster_df(groups)
    _finish(e, fit["tau"], fit["se_rb"], dfree)
    if e.estimate is not None:
        e.ci90 = _ci(fit["tau_bc"], fit["se_rb"], 0.90, dfree)
        e.ci95 = _ci(fit["tau_bc"], fit["se_rb"], 0.95, dfree)
        e.p_value = _pval(fit["tau_bc"], fit["se_rb"], dfree)
        _set_ci_levels(e, fit["tau_bc"], fit["se_rb"], dfree)
    for mult, lab in ((0.5, "Half bandwidth"), (2.0, "Double bandwidth")):
        hm = min(h * mult, xr * 1.0001)
        bm = min(max(hm * n ** (2 / 35), hm), xr * 1.0001)
        f2 = _rd_fit(x, y, tv, hm, bm, groups, fuzzy, share_jump)
        if f2:
            e.comparisons.append(_comp(f"{lab} ({hm:.4g})", f2["tau"], f2["se_rb"],
                                       "Robust bias-corrected SE; a stable estimate across bandwidths is reassuring.", dfree))
        else:
            e.comparisons.append({"method": f"{lab} ({hm:.4g})", "estimate": None, "se": None, "ci95": None,
                                  "note": "Too few units within this bandwidth."})
    e.extra.update({
        "cutoff": c0, "bandwidth": h, "pilot_bandwidth": b, "kernel": "triangular",
        "bandwidth_rule": "Imbens-Kalyanaraman (2012) MSE-optimal rule; bias pilot b = h * n^(2/35) (>= h)",
        "fuzzy": bool(fuzzy), "treatment_share_jump": _f(share_jump), "first_stage_jump": _f(fit["jump_t"]),
        "conventional_estimate": _f(fit["tau"]), "conventional_se": _f(fit["se"]),
        "conventional_ci95": _ci(fit["tau"], fit["se"], 0.95, dfree) if fit["se"] > 0 else None,
        "bias_corrected_estimate": _f(fit["tau_bc"]), "robust_se": _f(fit["se_rb"]),
        "left_limit": _f(fit["left"]), "right_limit": _f(fit["right"]),
        "n_left": int(((x < 0) & (x > -h)).sum()), "n_right": int(((x >= 0) & (x < h)).sum()),
        "se_type": "robust bias-corrected (CCT idea)" + (f", clustered by {r.cluster}" if groups is not None else ""),
        "covariates": [], "notes": notes + [
            "The point estimate is the conventional local linear jump; the intervals are robust bias-corrected "
            "(centered on the bias-corrected estimate, with variance that includes the bias estimate)."],
    })
    if groups is not None:
        e.extra["n_clusters"] = int(groups.max()) + 1
    # chart: binned means + fitted lines
    pts = []
    for side in (x < 0, x >= 0):
        xs, ysd = x[side], y[side]
        if xs.size == 0:
            continue
        edges = np.linspace(xs.min(), xs.max() + 1e-9, 21)
        idx = np.clip(np.digitize(xs, edges) - 1, 0, 19)
        for k in range(20):
            m = idx == k
            if m.any():
                pts.append({"x": float(xs[m].mean() + c0), "y": float(ysd[m].mean()), "group": "binned mean"})
    lines = []
    for sgn, side in ((-1, x < 0), (1, x >= 0)):
        sh = side & (np.abs(x) < h)
        if sh.sum() >= 2:
            cf = _wls_weights(np.column_stack([np.ones(sh.sum()), x[sh]]), _tri(x[sh], h)) @ y[sh]
            xx = np.array([0.0, sgn * h])
            lines.append({"x": (xx + c0).tolist(), "y": (cf[0] + cf[1] * xx).tolist(),
                          "name": "fit below cutoff" if sgn < 0 else "fit above cutoff"})
    e.charts.append(Chart("scatter", "Outcome around the cutoff", {"points": pts, "lines": lines, "vline": c0},
                          x_label=r.running, y_label=r.outcome,
                          note=f"Dots are bin averages; lines are local linear fits within the bandwidth ({h:.3g}) of the cutoff {c0:g}."))
    return e
