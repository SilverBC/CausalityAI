"""Missing outcomes (attrition).

Overall and differential attrition are compared with the What Works Clearinghouse (WWC)
attrition boundaries; whether missingness is predicted by measured covariates is tested
with a logistic regression; and the effect is bounded with Lee (2009) trimming bounds and
Manski worst-case bounds. Whether outcomes are missing for reasons tied to the outcome
itself can never be tested from data, so that assumption is always ASSUMED when any
outcome is missing.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..schema import Assumption, Chart, Claim, Design, Prepared, Section, Status

KEY = "missing_outcomes"
TITLE = "Missing outcomes"

# ---------------------------------------------------------------------------------------
# WWC attrition boundaries: maximum differential attrition (percentage points) allowed at
# each level of overall attrition (percent, 0..65 in 1-point steps) for a study to count
# as "low attrition". "cautious" = WWC conservative boundary, "optimistic" = WWC liberal
# boundary.
#
# Source: What Works Clearinghouse Standards Handbook (v4.0/4.1, 2017-2020; carried into
# v5.0, 2022), attrition boundary table. These values were transcribed from memory, NOT
# copied from the handbook, so they are APPROXIMATE (believed to be within a few tenths of a
# point of the published table). Shape is right: the liberal boundary starts at 10.0 pp,
# peaks near 11 pp at 10-20% overall and falls to ~2.4 pp at 65%; the conservative
# boundary starts at 5.7 pp, peaks near 6.3 pp at ~8-10% and reaches 0 near 60%.
# Beyond the end of a list the boundary is treated as 0 (any differential = high attrition).
BOUNDARY_SOURCE = ("What Works Clearinghouse Standards Handbook (v4.x / v5.0) attrition boundaries; "
                   "values approximate (transcribed from memory, not the published table), "
                   "accurate to roughly +/-0.3 percentage points.")

OPTIMISTIC = [  # liberal boundary, overall 0..65
    10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.7, 10.8, 10.9, 10.9,
    10.9, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 11.0, 10.9, 10.9,
    10.9, 10.8, 10.8, 10.7, 10.6, 10.5, 10.4, 10.3, 10.2, 10.1,
    9.9, 9.8, 9.6, 9.5, 9.3, 9.1, 9.0, 8.8, 8.6, 8.4,
    8.2, 8.0, 7.8, 7.6, 7.4, 7.2, 7.0, 6.7, 6.5, 6.3,
    6.0, 5.8, 5.6, 5.3, 5.1, 4.9, 4.6, 4.4, 4.1, 3.9,
    3.6, 3.4, 3.1, 2.9, 2.6, 2.4,
]
CAUTIOUS = [  # conservative boundary, overall 0..65
    5.7, 5.8, 5.9, 5.9, 6.0, 6.1, 6.2, 6.3, 6.3, 6.3,
    6.3, 6.2, 6.2, 6.1, 6.0, 5.9, 5.9, 5.8, 5.7, 5.5,
    5.4, 5.3, 5.2, 5.1, 4.9, 4.8, 4.7, 4.5, 4.4, 4.3,
    4.1, 4.0, 3.8, 3.6, 3.5, 3.3, 3.2, 3.1, 2.9, 2.8,
    2.6, 2.5, 2.3, 2.1, 2.0, 1.8, 1.6, 1.5, 1.3, 1.2,
    1.0, 0.9, 0.7, 0.5, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
]
MAX_OVERALL = 65.0


def max_differential(overall_pct: float, boundary: str = "cautious") -> float:
    """Largest differential attrition (pp) still counted as low attrition, linearly
    interpolated between the 1-point table steps. Overall above 65% -> -inf (never low)."""
    table = CAUTIOUS if boundary == "cautious" else OPTIMISTIC
    if overall_pct > MAX_OVERALL:
        return float("-inf")
    x = max(0.0, float(overall_pct))
    lo = int(np.floor(x))
    hi = min(lo + 1, len(table) - 1)
    w = x - lo
    return float(table[lo] * (1 - w) + table[hi] * w)


def within(overall_pct: float, diff_pct: float, boundary: str) -> bool:
    if overall_pct > MAX_OVERALL:
        return False
    return abs(diff_pct) <= max_differential(overall_pct, boundary) + 1e-9


# ---------------------------------------------------------------------------------------
def _weighted_trimmed_mean(y: np.ndarray, p: float, drop: str) -> float:
    """Mean after dropping a share p of the values from the top ('top') or bottom
    ('bottom'), with fractional weight at the boundary so ties are handled exactly."""
    y = np.sort(np.asarray(y, dtype=float))
    if drop == "top":
        y = y[::-1]
    n = len(y)
    keep = n * (1 - p)                 # values kept (fractional)
    # drop the first n*p of y (after sorting so the dropped side comes first)
    idx = np.arange(n, dtype=float)
    w = np.clip(idx + 1 - np.maximum(idx, n * p), 0.0, 1.0)
    if keep <= 0 or w.sum() <= 0:
        return float("nan")
    return float(np.sum(w * y) / w.sum())


def lee_bounds(y: np.ndarray, t: np.ndarray, obs: np.ndarray) -> Optional[dict]:
    """Lee (2009) trimming bounds for the effect among units whose outcome would be observed
    under either arm. Trims the arm with the higher observation rate."""
    q1 = obs[t == 1].mean()
    q0 = obs[t == 0].mean()
    if q1 <= 0 or q0 <= 0:
        return None
    y1 = y[(t == 1) & obs]
    y0 = y[(t == 0) & obs]
    if q1 >= q0:
        p = (q1 - q0) / q1
        lower = _weighted_trimmed_mean(y1, p, "top") - y0.mean()
        upper = _weighted_trimmed_mean(y1, p, "bottom") - y0.mean()
        trimmed = "treated"
    else:
        p = (q0 - q1) / q0
        lower = y1.mean() - _weighted_trimmed_mean(y0, p, "bottom")
        upper = y1.mean() - _weighted_trimmed_mean(y0, p, "top")
        trimmed = "control"
    return {"lower": float(lower), "upper": float(upper), "trim_share": float(p), "trimmed_group": trimmed}


def manski_bounds(y: np.ndarray, t: np.ndarray, obs: np.ndarray, lo: float, hi: float) -> dict:
    out = {}
    for g in (0, 1):
        m = t == g
        q = obs[m].mean()
        mean_obs = y[m & obs].mean() if (m & obs).any() else 0.0
        out[g] = (q * mean_obs + (1 - q) * lo, q * mean_obs + (1 - q) * hi)
    return {"lower": float(out[1][0] - out[0][1]), "upper": float(out[1][1] - out[0][0]),
            "outcome_min": float(lo), "outcome_max": float(hi)}


def _missingness_test(prep: Prepared, d: pd.DataFrame, miss: np.ndarray) -> dict:
    """Logistic regression of the missing indicator on treatment + adjustment set,
    likelihood-ratio test against the intercept-only model."""
    import statsmodels.api as sm

    cols = [prep.roles.treatment] + [c for c in prep.adjustment_set if c in d.columns]
    X = d[cols].astype(float)
    X = X.loc[:, X.nunique() > 1]
    if X.shape[1] == 0 or miss.sum() < 5 or (len(miss) - miss.sum()) < 5:
        return {"p_value": None, "note": "too few missing (or observed) outcomes to test"}
    # standardize for numerical stability; cap columns for speed
    X = X.iloc[:, :50]
    X = (X - X.mean()) / X.std(ddof=0).replace(0, 1)
    X = sm.add_constant(X, has_constant="add")
    try:
        fit = sm.Logit(miss.astype(float), X).fit(disp=0, maxiter=200)
        p = float(fit.llr_pvalue)
        top = fit.params.drop("const").abs().sort_values(ascending=False)
        pv = fit.pvalues.drop("const")
        sig = [c for c in top.index if pv[c] < 0.05][:5]
        return {"p_value": p, "lr_stat": float(fit.llr), "df": int(fit.df_model),
                "significant_predictors": sig}
    except Exception as e:  # noqa: BLE001  (separation, singular matrix ...)
        try:
            fit = sm.Logit(miss.astype(float), X).fit_regularized(disp=0, alpha=1e-4)
            null_ll = sm.Logit(miss.astype(float), X[["const"]]).fit(disp=0).llf
            lr = 2 * (fit.llf - null_ll)
            p = float(stats.chi2.sf(lr, X.shape[1] - 1))
            return {"p_value": p, "lr_stat": float(lr), "df": int(X.shape[1] - 1),
                    "note": "fitted with a small penalty (the plain fit did not converge)"}
        except Exception:  # noqa: BLE001
            return {"p_value": None, "note": f"the test could not be fitted ({type(e).__name__})"}


# ---------------------------------------------------------------------------------------
def run(prep: Prepared, design: Design, claim: Optional[Claim]) -> Section:
    try:
        return _run(prep, design, claim)
    except Exception as e:  # noqa: BLE001
        return Section(key=KEY, title=TITLE, status=None, caps_verdict=False,
                       error=f"The missing-outcomes check could not run ({type(e).__name__}: {e}).")


def _run(prep: Prepared, design: Design, claim: Optional[Claim]) -> Section:
    r = prep.roles
    if not r.treatment or not r.outcome or r.treatment not in prep.df or r.outcome not in prep.df:
        return Section(key=KEY, title=TITLE, error="No treatment or outcome column is set, so missing outcomes "
                                                    "cannot be checked.")
    d = prep.df[prep.df[r.treatment].notna()].copy()
    n = len(d)
    if n == 0:
        return Section(key=KEY, title=TITLE, error="The treatment is missing in every row.")
    y = d[r.outcome].to_numpy(dtype=float)
    t = d[r.treatment].to_numpy(dtype=float)
    miss = np.isnan(y)
    obs = ~miss
    n_miss = int(miss.sum())
    overall = n_miss / n
    metrics: dict = {"n": n, "n_missing": n_miss, "overall_attrition": overall,
                     "boundary_used": "cautious", "boundary_source": BOUNDARY_SOURCE}
    if n_miss < (prep.df[r.outcome].isna().sum()):
        metrics["n_missing_with_missing_treatment"] = int(prep.df[r.outcome].isna().sum() - n_miss)

    no_missing = Assumption("no_missing_outcomes", "No outcomes are missing", Status.PASS,
                            f"All {n} rows with a treatment value have an observed outcome.", True, "attrition")
    if n_miss == 0:
        return Section(key=KEY, title=TITLE, status=Status.PASS,
                       summary=f"No outcomes are missing among the {n} rows with a treatment value.",
                       metrics=metrics, assumptions=[no_missing], caps_verdict=False)

    mnar = Assumption(
        "missing_not_outcome_related", "Outcomes are not missing for reasons tied to the outcome itself",
        Status.ASSUMED,
        f"{n_miss} outcome(s) ({overall:.1%}) are missing. Whether people dropped out because of what their "
        "outcome would have been can never be checked from the data; the bounds below show how far the answer "
        "could move if it did.", False, "attrition")
    assumptions: list[Assumption] = [mnar]
    warnings: list[str] = []
    charts: list[Chart] = []

    # covariates predicting missingness (a tested piece)
    mt = _missingness_test(prep, d, miss)
    metrics["missingness_predicted_p"] = mt.get("p_value")
    metrics["missingness_test"] = mt
    if mt.get("p_value") is None:
        mt_status = None
    elif mt["p_value"] < 0.05:
        mt_status = Status.CONCERN
        preds = ", ".join(mt.get("significant_predictors") or []) or "treatment / covariates jointly"
        assumptions.append(Assumption(
            "missingness_unrelated_to_covariates", "Who is missing does not depend on measured characteristics",
            Status.CONCERN, f"Missingness is predicted by measured columns (likelihood-ratio p = {mt['p_value']:.3g}; "
            f"notably: {preds}). The people with outcomes differ from those without.", True, "attrition"))
    else:
        mt_status = Status.PASS
        assumptions.append(Assumption(
            "missingness_unrelated_to_covariates", "Who is missing does not depend on measured characteristics",
            Status.PASS, f"Treatment and the measured columns do not predict missingness (likelihood-ratio "
            f"p = {mt['p_value']:.3g}). This says nothing about unmeasured reasons.", True, "attrition"))

    if prep.treatment_is_binary():
        status, summary = _binary(prep, d, y, t, obs, overall, metrics, assumptions, charts, warnings)
    else:
        status, summary = _continuous(d, t, miss, overall, metrics, assumptions, warnings)

    statuses = [status] + ([mt_status] if mt_status else [])
    final = Status.worst(statuses) or Status.CONCERN
    if final == Status.PASS:
        summary += (" The check of rates passes, but the result still rests on the untestable assumption that "
                    "outcomes are not missing because of their values.")
    elif mt_status == Status.CONCERN and status == Status.PASS:
        summary += " Missingness is, however, predicted by measured columns, so complete cases may not be representative."
    caps = final in (Status.CONCERN, Status.FAIL)
    return Section(key=KEY, title=TITLE, status=final, summary=summary, metrics=metrics,
                   assumptions=assumptions, charts=charts, warnings=warnings, caps_verdict=caps)


def _binary(prep, d, y, t, obs, overall, metrics, assumptions, charts, warnings):
    miss = ~obs
    a1 = miss[t == 1].mean()
    a0 = miss[t == 0].mean()
    diff = a1 - a0
    n1, n0 = int((t == 1).sum()), int((t == 0).sum())
    # two-proportion z-test for differential attrition
    pooled = miss.mean()
    se = np.sqrt(pooled * (1 - pooled) * (1 / max(n1, 1) + 1 / max(n0, 1)))
    p_diff = float(2 * stats.norm.sf(abs(diff) / se)) if se > 0 else None
    o_pct, d_pct = overall * 100, abs(diff) * 100
    in_c, in_o = within(o_pct, d_pct, "cautious"), within(o_pct, d_pct, "optimistic")
    metrics.update({
        "attrition_treated": float(a1), "attrition_control": float(a0), "differential_attrition": float(diff),
        "differential_p": p_diff, "max_differential_cautious": max_differential(o_pct, "cautious"),
        "max_differential_optimistic": max_differential(o_pct, "optimistic"),
        "within_cautious": in_c, "within_optimistic": in_o,
    })
    sig = p_diff is not None and p_diff < 0.05
    rate = (f"{overall:.1%} of outcomes are missing ({a1:.1%} treated vs {a0:.1%} untreated, a gap of "
            f"{diff*100:+.1f} percentage points")
    rate += f", p = {p_diff:.2g})." if p_diff is not None else ")."
    if not in_o:
        status = Status.FAIL
        summary = rate + (" This is high attrition even under the optimistic What Works Clearinghouse boundary; "
                          "the comparison of complete cases may be badly biased.")
    elif not in_c:
        status = Status.CONCERN
        summary = rate + (" This is beyond the cautious What Works Clearinghouse boundary (used by default) though "
                          "within the optimistic one.")
    elif sig:
        status = Status.CONCERN
        summary = rate + (" The rates are within the cautious What Works Clearinghouse boundary, but the two groups "
                          "lose outcomes at clearly different rates.")
    else:
        status = Status.PASS
        summary = rate + " This is within the cautious What Works Clearinghouse boundary (low attrition)."
    assumptions.append(Assumption(
        "low_attrition", "Attrition is low and similar across groups", status,
        f"Overall {o_pct:.1f}%, differential {d_pct:.1f} pp; the cautious boundary allows up to "
        f"{max(metrics['max_differential_cautious'], 0):.1f} pp at this overall rate"
        + ("" if o_pct <= MAX_OVERALL else " (overall attrition above 65% is always high)") + ".",
        True, "attrition"))

    # complete-case difference
    y1, y0 = y[(t == 1) & obs], y[(t == 0) & obs]
    rows = []
    if len(y1) >= 2 and len(y0) >= 2:
        cc = float(y1.mean() - y0.mean())
        cc_se = float(np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0)))
        metrics["complete_case_difference"] = cc
        metrics["complete_case_ci95"] = (cc - 1.96 * cc_se, cc + 1.96 * cc_se)
        rows.append({"label": "Complete cases (difference in means)", "est": cc,
                     "lo": cc - 1.96 * cc_se, "hi": cc + 1.96 * cc_se})
    lee = lee_bounds(y, t, obs)
    if lee:
        metrics["lee_bounds"] = lee
        rows.append({"label": "Lee trimming bounds", "est": (lee["lower"] + lee["upper"]) / 2,
                     "lo": lee["lower"], "hi": lee["upper"]})
    else:
        warnings.append("Lee bounds could not be computed (one group has no observed outcomes).")
    yo = y[obs]
    binary_y = set(np.unique(yo)) <= {0.0, 1.0}
    lo, hi = (0.0, 1.0) if binary_y else (float(yo.min()), float(yo.max()))
    man = manski_bounds(y, t, obs, lo, hi)
    man["range_note"] = ("outcome is 0/1, so these are true worst-case bounds" if binary_y else
                         "worst-case bounds assuming missing outcomes lie within the observed range "
                         f"[{lo:.4g}, {hi:.4g}]; values outside that range would widen them")
    metrics["manski_bounds"] = man
    rows.append({"label": "Worst-case (Manski) bounds", "est": (man["lower"] + man["upper"]) / 2,
                 "lo": man["lower"], "hi": man["upper"]})
    if rows:
        charts.append(Chart(
            kind="forest", title="Effect under different assumptions about missing outcomes",
            data={"rows": rows, "ref": 0.0}, x_label="Difference in outcome (treated minus untreated)",
            note=("Complete cases assume missing outcomes are like observed ones. Lee bounds cover units whose "
                  "outcome would be observed either way (point bounds, no sampling error). Worst-case bounds "
                  "assume nothing about the missing outcomes" + ("" if binary_y else " beyond the observed range") + ".")))
    if lee and lee["lower"] <= 0 <= lee["upper"]:
        summary += " The Lee bounds include zero, so attrition alone could explain the difference."
    return status, summary


def _continuous(d, t, miss, overall, metrics, assumptions, warnings):
    o_pct = overall * 100
    try:
        r_pb, p = stats.pointbiserialr(miss.astype(float), t)
        r_pb, p = float(r_pb), float(p)
    except Exception:  # noqa: BLE001
        r_pb, p = None, None
    metrics.update({"missing_treatment_correlation": r_pb, "missing_treatment_p": p,
                    "lee_bounds": None,
                    "lee_note": "Lee bounds need a two-group (0/1) treatment; not computed for this treatment."})
    warnings.append("Lee and worst-case bounds are only computed for a two-group (0/1) treatment.")
    sig = p is not None and p < 0.05
    rate = f"{overall:.1%} of outcomes are missing"
    if r_pb is not None:
        rate += f"; missingness correlates with the treatment level at r = {r_pb:+.2f} (p = {p:.2g})"
    rate += "."
    # with no groups, the boundary can only be read at zero differential
    if o_pct > MAX_OVERALL:
        status, extra = Status.FAIL, " Over 65% of outcomes are missing, which is high attrition by any standard."
    elif not within(o_pct, 0.0, "cautious"):
        status, extra = Status.CONCERN, " Overall attrition is beyond the cautious What Works Clearinghouse boundary."
    elif sig:
        status, extra = Status.CONCERN, " Units with different treatment levels lose outcomes at different rates."
    else:
        status, extra = Status.PASS, " Missingness is not related to the treatment level."
    assumptions.append(Assumption("low_attrition", "Attrition is low and unrelated to treatment", status,
                                  rate + extra, True, "attrition"))
    return status, rate + extra
