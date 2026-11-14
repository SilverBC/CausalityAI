"""Overlap (positivity): could every kind of unit have been in either group?

Two cross-fitted propensity models (standardized logistic regression and gradient boosting),
their AUCs, common support by percentiles, Crump-style trimming ([0.05, 0.95] and inside common
support), and a plain-words description of where overlap breaks down.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..schema import Assumption, Chart, Claim, Design, Prepared, Section, Status

KEY = "overlap"
TITLE = "Overlap"
TRIM_LO, TRIM_HI = 0.05, 0.95
MAX_GB_TRAIN = 20000
_CACHE: dict = {}


# --------------------------------------------------------------------------- propensity
def _key(prep: Prepared):
    t = prep.roles.treatment
    tv = prep.df[t] if t in prep.df else None
    sig = (float(np.nansum(tv.to_numpy(dtype=float) * np.arange(1, len(tv) + 1))) if tv is not None else 0.0)
    return (id(prep.df), t, tuple(prep.adjustment_set), prep.df.shape, sig,
            tuple(prep.df.index[:3]), tuple(prep.df.index[-3:]))


def _fit(prep: Prepared) -> dict:
    """Cross-fitted propensity scores. Returns dict with ps (len(df), NaN where T missing),
    ps_logit, ps_gb, auc_logit, auc_gb, rows (bool mask of rows used)."""
    k = _key(prep)
    if k in _CACHE:
        return _CACHE[k]
    df = prep.df
    t = prep.roles.treatment
    rows = df[t].notna().to_numpy()
    T = df.loc[rows, t].to_numpy(dtype=int)
    cols = [c for c in prep.adjustment_set if c in df]
    n_all = len(df)
    out = {"rows": rows, "cols": cols}
    ps_l = np.full(n_all, np.nan)
    ps_g = np.full(n_all, np.nan)
    if not cols:
        m = T.mean()
        ps_l[rows] = m
        ps_g[rows] = m
        out.update(ps_logit=ps_l, ps_gb=ps_g, ps=ps_l.copy(), auc_logit=0.5, auc_gb=0.5)
        _CACHE[k] = out
        return out
    X = df.loc[rows, cols].to_numpy(dtype=float)
    n = len(T)
    n_splits = int(min(5, max(2, min((T == 1).sum(), (T == 0).sum()))))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    pl = np.zeros(n)
    pg = np.zeros(n)
    rng = np.random.default_rng(0)
    for tr, te in skf.split(X, T):
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        lr.fit(X[tr], T[tr])
        pl[te] = lr.predict_proba(X[te])[:, 1]
        tr_g = tr if len(tr) <= MAX_GB_TRAIN else rng.choice(tr, MAX_GB_TRAIN, replace=False)
        if len(np.unique(T[tr_g])) < 2:
            tr_g = tr
        gb = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=40, l2_regularization=1.0,
                                            early_stopping=False, random_state=0)
        gb.fit(X[tr_g], T[tr_g])
        pg[te] = gb.predict_proba(X[te])[:, 1]
    ps_l[rows] = pl
    ps_g[rows] = pg
    out.update(ps_logit=ps_l, ps_gb=ps_g, ps=(ps_l + ps_g) / 2,
               auc_logit=float(roc_auc_score(T, pl)), auc_gb=float(roc_auc_score(T, pg)))
    if len(_CACHE) > 16:
        _CACHE.clear()
    _CACHE[k] = out
    return out


def propensity(prep: Prepared) -> np.ndarray:
    """Cross-fitted propensity score (average of the logistic and boosting models), one value
    per row of prep.df; NaN where treatment is missing or not binary."""
    try:
        if not prep.treatment_is_binary():
            return np.full(len(prep.df), np.nan)
        return _fit(prep)["ps"].copy()
    except Exception:  # noqa: BLE001
        return np.full(len(prep.df), np.nan)


def _support(ps: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    p1, p0 = ps[T == 1], ps[T == 0]
    lo = max(np.percentile(p1, 1), np.percentile(p0, 1))
    hi = min(np.percentile(p1, 99), np.percentile(p0, 99))
    return float(lo), float(hi)


def trim_alpha(n: int, k: int) -> float:
    """Trimming bound: keep propensity scores in [alpha, 1 - alpha].

    A fixed, pre-stated rule that depends only on the sample size n and the number of
    covariates k (never on the outcome or the estimate):
        alpha = clip(0.5 * sqrt((k + 1) * ln(n) / n), 0.01, 0.10)
    The noise in an estimated propensity score shrinks like sqrt(k / n), so a large, simple
    sample can tell a score of 0.02 from 0 and may keep it, while a small or
    high-dimensional one cannot and must trim more. n=3000, k=3 gives about 0.05."""
    n, k = max(int(n), 2), max(int(k), 0)
    return float(np.clip(0.5 * np.sqrt((k + 1) * np.log(n) / n), 0.01, 0.10))


def auc_lower(auc: float, n1: int, n0: int) -> float:
    """Lower 95% bound of a cross-fitted AUC (simple binomial-style approximation on the
    smaller group), so a small sample is not rejected on noise alone."""
    m = max(min(int(n1), int(n0)), 1)
    return float(auc - 1.96 * np.sqrt(max(auc * (1 - auc), 0.0) / m))


def _keep(ps: np.ndarray, lo: float, hi: float, alpha: float = TRIM_LO) -> np.ndarray:
    return (ps >= max(alpha, lo) - 1e-12) & (ps <= min(1 - alpha, hi) + 1e-12)


def trim_mask(prep: Prepared) -> pd.Series:
    """Boolean Series aligned to prep.df.index; True = keep (propensity in [0.05, 0.95] and
    inside common support). Rows with missing treatment are False. Non-binary treatment or no
    covariates: all rows with treatment observed are kept."""
    try:
        t = prep.roles.treatment
        if not t or t not in prep.df:
            return pd.Series(True, index=prep.df.index)
        has_t = prep.df[t].notna()
        if not prep.treatment_is_binary() or not [c for c in prep.adjustment_set if c in prep.df]:
            return has_t.astype(bool)
        f = _fit(prep)
        rows = f["rows"]
        ps = f["ps"][rows]
        T = prep.df.loc[rows, t].to_numpy(dtype=int)
        lo, hi = _support(ps, T)
        keep = np.zeros(len(prep.df), dtype=bool)
        keep[rows] = _keep(ps, lo, hi, trim_alpha(len(ps), len([c for c in prep.adjustment_set if c in prep.df])))
        return pd.Series(keep, index=prep.df.index)
    except Exception:  # noqa: BLE001
        return pd.Series(True, index=prep.df.index)


# --------------------------------------------------------------------------- run
def run(prep: Prepared, design: Design, claim: Optional[Claim] = None) -> Section:
    try:
        return _run(prep, design, claim)
    except Exception as e:  # noqa: BLE001
        return Section(key=KEY, title=TITLE, error=f"The overlap check could not be completed ({type(e).__name__}: {e}).")


def _smd(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return float((a.mean() - b.mean()) / sd) if sd > 0 and np.isfinite(sd) else 0.0


def _run(prep: Prepared, design: Design, claim: Optional[Claim]) -> Section:
    r = prep.roles
    df = prep.df
    t = r.treatment
    if not t or t not in df:
        return Section(key=KEY, title=TITLE, error="No treatment column is set, so overlap cannot be checked.")
    cols = [c for c in prep.adjustment_set if c in df]
    if df[t].dropna().nunique() < 2:
        return Section(key=KEY, title=TITLE, status=None,
                       summary="The treatment takes a single value, so there are no groups to overlap.")
    if not cols:
        return Section(key=KEY, title=TITLE, status=None, metrics={"n_covariates": 0},
                       summary="There are no measured covariates in the adjustment set, so there is nothing to "
                               "check overlap on. This does not show that groups are comparable on anything.")
    if not prep.treatment_is_binary():
        return _continuous(prep, cols)

    f = _fit(prep)
    rows = f["rows"]
    ps = f["ps"][rows]
    T = df.loc[rows, t].to_numpy(dtype=int)
    X = df.loc[rows, cols].to_numpy(dtype=float)
    n1, n0 = int((T == 1).sum()), int((T == 0).sum())
    lo, hi = _support(ps, T)
    out_t = float(np.mean((ps[T == 1] < lo) | (ps[T == 1] > hi)))
    out_c = float(np.mean((ps[T == 0] < lo) | (ps[T == 0] > hi)))
    alpha = trim_alpha(len(ps), len(cols))
    keep = _keep(ps, lo, hi, alpha)
    trim_t = float(1 - keep[T == 1].mean())
    trim_c = float(1 - keep[T == 0].mean())
    trim_all = float(1 - keep.mean())
    kept_t, kept_c = int(keep[T == 1].sum()), int(keep[T == 0].sum())
    auc_l, auc_g = f["auc_logit"], f["auc_gb"]
    auc_max = max(auc_l, auc_g)
    auc_lb = auc_lower(auc_max, n1, n0)   # rejection uses the lower bound: sample-size aware

    # where overlap breaks down: describe each tail separately (high vs low propensity),
    # since trimmed units on both sides can average out to look like the kept ones
    where = []
    where_parts = []
    mid = float(np.median(ps[keep])) if keep.any() else 0.5
    for side, mask, who in (("high", ~keep & (ps > mid), "almost always treated"),
                            ("low", ~keep & (ps <= mid), "almost never treated")):
        if mask.sum() < 2 or keep.sum() < 2:
            continue
        diffs = []
        for j, c in enumerate(cols):
            d = _smd(X[mask, j], X[keep, j])
            diffs.append((c, d, float(X[mask, j].mean()), float(X[keep, j].mean())))
        diffs.sort(key=lambda z: -abs(z[1]))
        top = [z for z in diffs[:3] if abs(z[1]) >= 0.2]
        if not top:
            continue
        where += [{"tail": who, "column": c, "smd": d, "trimmed_mean": a, "kept_mean": b} for c, d, a, b in top]
        bits = [f"{'high' if d > 0 else 'low'} '{c}' (average {a:.3g} vs {b:.3g} among kept units)" for c, d, a, b in top]
        where_parts.append(f"units {who} ({int(mask.sum())}) have " + ", ".join(bits))
    where_txt = ("Where overlap is thin: " + "; ".join(where_parts) + ".") if where_parts else ""

    # status
    few = min(kept_t, kept_c) < 10 or min(kept_t / max(n1, 1), kept_c / max(n0, 1)) < 0.2
    if trim_all > 0.20 or auc_lb > 0.9 or few:
        status = Status.FAIL
    elif trim_all >= 0.05 or auc_lb >= 0.8:
        status = Status.CONCERN
    else:
        status = Status.PASS

    warnings = []
    if r.randomized and status != Status.PASS:
        warnings.append("Treatment was said to be randomized, yet the covariates predict it (or groups do not "
                        "overlap); under randomization overlap should be near perfect. Check the assignment "
                        "and whether some columns were measured after treatment.")
    if few:
        warnings.append("One group has almost no units comparable to the other group.")

    summary = (f"Covariates predict treatment with AUC {auc_l:.2f} (logistic) and {auc_g:.2f} (boosting); "
               f"0.5 means no better than a coin flip. Common support of the propensity score runs from "
               f"{lo:.2f} to {hi:.2f}. Trimming to scores in [{alpha:.3f}, {1 - alpha:.3f}] (a fixed rule from the sample size and the "
               f"number of covariates) inside that range drops "
               f"{trim_t:.0%} of treated and {trim_c:.0%} of control units ({trim_all:.0%} overall).")
    if where_txt:
        summary += " " + where_txt
    if status == Status.PASS:
        summary += " Treated and control units overlap well on the measured covariates."
    elif status == Status.CONCERN:
        summary += " Some units have no comparable counterparts; estimates for them lean on extrapolation."
    else:
        summary += " Many units have no comparable counterparts; an effect for everyone cannot be estimated " \
                   "without extrapolating, so only the overlap region can be studied."

    detail = (f"AUC {auc_max:.2f} (lower bound {auc_lb:.2f}); {trim_all:.0%} of units fall outside the region where both groups are present.")
    assumptions = [Assumption("overlap", "Overlap / positivity: every kind of unit could have been in either group",
                              status, detail, True, "overlap")]

    bins = np.linspace(0, 1, 21)
    hist = Chart(kind="hist", title="Propensity score by group",
                 data={"bins": bins.tolist(), "series": [
                     {"name": "Treated", "counts": np.histogram(ps[T == 1], bins)[0].astype(int).tolist()},
                     {"name": "Control", "counts": np.histogram(ps[T == 0], bins)[0].astype(int).tolist()}]},
                 x_label="estimated probability of treatment", y_label="units",
                 note=f"Common support: {lo:.2f} to {hi:.2f}.")
    bar = Chart(kind="bar", title="Share of units trimmed for lack of overlap",
                data={"labels": ["Treated", "Control", "All"], "values": [trim_t, trim_c, trim_all],
                      "ref": 0.05, "errors": None},
                x_label="group", y_label="share trimmed", note="Line at 5%: above it, overlap is a concern.")

    metrics = {"auc_logit": auc_l, "auc_boosting": auc_g, "common_support": [lo, hi],
               "outside_support_treated": out_t, "outside_support_control": out_c,
               "trimmed_share_treated": trim_t, "trimmed_share_control": trim_c, "trimmed_share": trim_all,
               "n_kept_treated": kept_t, "n_kept_control": kept_c, "n_treated": n1, "n_control": n0,
               "trim_rule": f"propensity in [{alpha:.3f}, {1 - alpha:.3f}] and inside common support (1st-99th percentiles)",
               "trim_alpha": alpha, "trim_alpha_rule": "clip(0.5*sqrt((k+1)*ln(n)/n), 0.01, 0.10)",
               "auc_lower_bound": auc_lb, "propensity_used": "average of the logistic and boosting models (cross-fitted)",
               "where_overlap_fails": where,
               "n_covariates": len(cols)}
    return Section(key=KEY, title=TITLE, status=status, summary=summary, metrics=metrics,
                   assumptions=assumptions, charts=[hist, bar], warnings=warnings)


def _continuous(prep: Prepared, cols: list[str]) -> Section:
    t = prep.roles.treatment
    d = prep.df[prep.df[t].notna()]
    X = d[cols].to_numpy(dtype=float)
    T = d[t].to_numpy(dtype=float)
    lr = LinearRegression().fit(X, T)
    res = T - lr.predict(X)
    r2 = float(1 - res.var() / T.var()) if T.var() > 0 else 0.0
    rsd = float(res.std(ddof=min(len(cols) + 1, len(T) - 1)))
    sd_t = float(T.std(ddof=1))
    warnings = []
    if r2 > 0.9:
        warnings.append(f"The covariates explain {r2:.0%} of the variation in the treatment; little variation is "
                        "left to compare units with similar covariates, so estimates rely on extrapolation.")
    summary = (f"For a continuous treatment, overlap is shown as how much the treatment still varies among units "
               f"with similar covariates (generalized propensity). The covariates explain {r2:.0%} of the treatment's "
               f"variation; the remaining spread has SD {rsd:.3g} (overall SD {sd_t:.3g}).")
    bins = np.linspace(res.min(), res.max(), 21) if res.max() > res.min() else np.array([res.min() - 1, res.min() + 1])
    chart = Chart(kind="hist", title="Treatment variation left after accounting for covariates",
                  data={"bins": bins.tolist(),
                        "series": [{"name": "Residual treatment", "counts": np.histogram(res, bins)[0].astype(int).tolist()}]},
                  x_label=f"{t} minus its prediction from covariates", y_label="units")
    return Section(key=KEY, title=TITLE, status=None, summary=summary,
                   metrics={"treatment_r2": r2, "residual_sd": rsd, "treatment_sd": sd_t,
                            "flag_low_variation": r2 > 0.9, "n_covariates": len(cols)},
                   charts=[chart], warnings=warnings)
