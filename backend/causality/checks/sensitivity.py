"""Hidden confounding: how strong would an unmeasured confounder have to be?

* Robustness value (Cinelli & Hazlett 2020) from an OLS of Y on T + the adjustment set:
  RV_q (bring the estimate to (1-q) of itself; q=1 -> zero) and RV_{q,alpha}.
* Benchmark bounds (C&H section 4.4): the adjusted estimate if a hidden confounder were
  k times as strong as an observed covariate (k = 1, 2, 3), for the top 3 covariates.
* E-value (VanderWeele & Ding 2017) for the point estimate and the CI limit closest to the null.

These are sensitivity numbers only: the status is never PASS.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from ..schema import Assumption, Chart, Claim, Design, Estimate, Prepared, Section, Status

KEY = "hidden_confounding"
TITLE = "Hidden confounding (sensitivity)"


# --------------------------------------------------------------------------- formulas
def partial_r2(t_stat: float, dof: float) -> float:
    return float(t_stat ** 2 / (t_stat ** 2 + dof))


def robustness_value(t_stat: float, dof: float, q: float = 1.0, alpha: Optional[float] = None) -> float:
    """Cinelli & Hazlett (2020) robustness value. With alpha=None: RV_q (eq. 18); with alpha:
    RV_{q,alpha} (eq. 19 / sensemakr), the strength that makes the CI include (1-q)*estimate."""
    f = abs(t_stat) / math.sqrt(dof)
    fq = q * f
    if alpha is None:
        return float(0.5 * (math.sqrt(fq ** 4 + 4 * fq ** 2) - fq ** 2))
    f_crit = abs(stats.t.ppf(alpha / 2, dof - 1)) / math.sqrt(dof - 1)
    fqa = fq - f_crit
    if fqa < 0:
        return 0.0
    rv = 0.5 * (math.sqrt(fqa ** 4 + 4 * fqa ** 2) - fqa ** 2)
    if fq > 1 / f_crit:   # extreme case: confounder explains all residual outcome variance
        rv = (fq ** 2 - f_crit ** 2) / (1 + fq ** 2)
    return float(rv)


def bias_bound(se: float, dof: float, r2yz_dx: float, r2dz_x: float) -> float:
    """Maximum absolute bias from a confounder with the given partial R2s (C&H eq. 11)."""
    if r2dz_x >= 1:
        return float("inf")
    return float(se * math.sqrt(dof) * math.sqrt(r2yz_dx * r2dz_x / (1 - r2dz_x)))


def benchmark_partial_r2(r2dxj_x: float, r2yxj_dx: float, kd: float, ky: Optional[float] = None
                         ) -> Optional[tuple[float, float]]:
    """Bounds on (R2_{Y~Z|D,X}, R2_{D~Z|X}) for a confounder kd (ky) times as strong as covariate
    X_j (C&H 2020, section 4.4; sensemakr::ovb_partial_r2_bound). None if impossible."""
    ky = kd if ky is None else ky
    if r2dxj_x >= 1 or r2yxj_dx >= 1:
        return None
    r2dz_x = kd * (r2dxj_x / (1 - r2dxj_x))
    if r2dz_x >= 1:
        return None
    denom = (1 - kd * r2dxj_x) * (1 - r2dxj_x)
    if denom <= 0:
        return None
    r2zxj_xd = kd * r2dxj_x ** 2 / denom
    if r2zxj_xd >= 1:
        return None
    r2yz_dx = ((math.sqrt(ky) + math.sqrt(r2zxj_xd)) / math.sqrt(1 - r2zxj_xd)) ** 2 * (r2yxj_dx / (1 - r2yxj_dx))
    return float(min(r2yz_dx, 1.0)), float(r2dz_x)


def evalue_rr(rr: float) -> float:
    """E-value for a risk ratio (VanderWeele & Ding 2017)."""
    if rr is None or not np.isfinite(rr) or rr <= 0:
        return float("nan")
    rr = rr if rr >= 1 else 1 / rr
    return float(rr + math.sqrt(rr * (rr - 1)))


def _evalue_ci(rr_lo: float, rr_hi: float, rr: float) -> float:
    """E-value for the CI limit closest to the null (1 if the CI includes 1)."""
    if rr_lo <= 1 <= rr_hi:
        return 1.0
    return evalue_rr(rr_lo if rr > 1 else rr_hi)


# --------------------------------------------------------------------------- run
def run(prep: Prepared, design: Design, claim: Optional[Claim] = None, estimate: Optional[Estimate] = None) -> Section:
    try:
        return _run(prep, design, claim, estimate)
    except Exception as e:  # noqa: BLE001
        return Section(key=KEY, title=TITLE, status=Status.ASSUMED,
                       error=f"The sensitivity analysis could not be completed ({type(e).__name__}: {e}).",
                       assumptions=[_main_assumption(design.kind if design else "observational", None)])


def _fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%" if x < 0.995 else f"{100 * x:.0f}%"


def _main_assumption(kind: str, detail_numbers: Optional[str]) -> Assumption:
    nums = (" " + detail_numbers) if detail_numbers else ""
    if kind == "experiment":
        return Assumption("no_hidden_confounding", "Randomization rules out hidden confounders, if carried out as described",
                          Status.ASSUMED, "If assignment really was random, no unmeasured factor can drive both "
                          "treatment and outcome. This cannot be checked in the data." + nums, False, "sensitivity")
    if kind in ("iv", "rd", "did"):
        rel = {"iv": "the instrument being unrelated to the outcome except through treatment",
               "rd": "nothing else changing at the cutoff",
               "did": "parallel trends"}[kind]
        return Assumption("no_hidden_confounding", "No hidden confounders of the design's comparison", Status.ASSUMED,
                          f"This design does not rely on measuring every confounder; it relies on {rel} instead "
                          "(see the design checks). The numbers below are for a plain adjusted regression, shown "
                          "for reference." + nums, False, "sensitivity")
    return Assumption("no_hidden_confounding", "No hidden confounders", Status.ASSUMED,
                      "No unmeasured factor drives both treatment and outcome. This can never be tested from data."
                      + nums, False, "sensitivity")


def _run(prep: Prepared, design: Design, claim: Optional[Claim], est: Optional[Estimate]) -> Section:
    kind = design.kind if design else "observational"
    r = prep.roles
    t, y = r.treatment, r.outcome
    if not t or not y or t not in prep.df or y not in prep.df:
        return Section(key=KEY, title=TITLE, status=Status.ASSUMED,
                       error="Treatment or outcome is not set, so sensitivity cannot be computed.",
                       assumptions=[_main_assumption(kind, None)])
    d = prep.complete()
    cols = [c for c in prep.adjustment_set if c in d and d[c].nunique() > 1]
    n = len(d)
    if n < len(cols) + 5:
        return Section(key=KEY, title=TITLE, status=Status.ASSUMED,
                       error="Too few complete rows for the sensitivity regression.",
                       assumptions=[_main_assumption(kind, None)])
    Y = d[y].to_numpy(dtype=float)
    T = d[t].to_numpy(dtype=float)
    X = d[cols].to_numpy(dtype=float) if cols else np.zeros((n, 0))
    Z = sm.add_constant(np.column_stack([T, X]), has_constant="add")
    fit = sm.OLS(Y, Z).fit()
    b, se, tstat = float(fit.params[1]), float(fit.bse[1]), float(fit.tvalues[1])
    dof = float(fit.df_resid)
    warnings: list[str] = []
    notes: list[str] = []

    r2yd = partial_r2(tstat, dof)
    rv = robustness_value(tstat, dof, 1.0)
    rva = robustness_value(tstat, dof, 1.0, 0.05)

    # ---- benchmarks
    bench_rows = []
    if cols:
        fit_d = sm.OLS(T, sm.add_constant(X, has_constant="add")).fit()
        r2d = [partial_r2(float(fit_d.tvalues[j + 1]), float(fit_d.df_resid)) for j in range(len(cols))]
        r2y = [partial_r2(float(fit.tvalues[j + 2]), dof) for j in range(len(cols))]
        order = [j for j in np.argsort(r2d)[::-1] if np.isfinite(r2d[j]) and np.isfinite(r2y[j])][:3]
        for j in order:
            for k in (1, 2, 3):
                bnd = benchmark_partial_r2(r2d[j], r2y[j], k)
                if bnd is None:
                    continue
                r2yz, r2dz = bnd
                bias = bias_bound(se, dof, r2yz, r2dz)
                adj = b - math.copysign(bias, b) if b != 0 else -bias
                adj_se = se * math.sqrt((1 - r2yz) / (1 - r2dz)) * math.sqrt(dof / (dof - 1)) if r2dz < 1 else float("nan")
                bench_rows.append({"covariate": cols[j], "k": k, "r2dz_x": r2dz, "r2yz_dx": r2yz,
                                   "r2_treatment_covariate": r2d[j], "r2_outcome_covariate": r2y[j],
                                   "adjusted_estimate": adj, "adjusted_se": adj_se,
                                   "adjusted_lo": adj - 1.96 * adj_se, "adjusted_hi": adj + 1.96 * adj_se})

    # ---- estimate used for E-value
    use_est = est is not None and not est.refused and est.estimate is not None and np.isfinite(est.estimate)
    if use_est:
        e_val = float(est.estimate)
        if est.ci95:
            lo_ci, hi_ci = float(est.ci95[0]), float(est.ci95[1])
        elif est.se:
            lo_ci, hi_ci = e_val - 1.96 * est.se, e_val + 1.96 * est.se
        else:
            lo_ci, hi_ci = b - 1.96 * se, b + 1.96 * se
        src = f"the main estimate ({est.method})"
    else:
        e_val, lo_ci, hi_ci = b, b - 1.96 * se, b + 1.96 * se
        why = "the estimate was refused" if (est is not None and est.refused) else "no estimate was available"
        src = "an adjusted regression of the outcome on treatment and the adjustment set"
        notes.append(f"E-values use {src} because {why}.")

    binary_y = set(np.unique(Y)) <= {0.0, 1.0}
    binary_t = set(np.unique(T)) <= {0.0, 1.0}
    sd_y = float(np.std(Y, ddof=1))
    scale = 1.0 if binary_t else float(np.std(T, ddof=1))   # continuous T: per one-SD change
    if binary_y:
        p0 = None
        if use_est and est.baseline_mean is not None:
            p0 = float(est.baseline_mean)
        elif binary_t:
            p0 = float(Y[T == 0].mean())
        else:
            p0 = float(Y.mean())
        if p0 is not None and p0 > 0:
            rr = (p0 + e_val * scale) / p0
            rr_lo, rr_hi = (p0 + lo_ci * scale) / p0, (p0 + hi_ci * scale) / p0
        else:
            rr, rr_lo, rr_hi = float("nan"), float("nan"), float("nan")
        e_method = "risk ratio from treated vs control outcome rates"
    else:
        sdy = float(est.outcome_sd) if (use_est and est.outcome_sd) else sd_y
        dd = lambda v: v * scale / sdy if sdy > 0 else float("nan")  # noqa: E731
        rr, rr_lo, rr_hi = math.exp(0.91 * dd(e_val)), math.exp(0.91 * dd(lo_ci)), math.exp(0.91 * dd(hi_ci))
        e_method = "approximate risk ratio exp(0.91 x standardized effect) for a continuous outcome"
    rr_lo, rr_hi = (min(rr_lo, rr_hi), max(rr_lo, rr_hi)) if np.isfinite(rr_lo) and np.isfinite(rr_hi) else (rr_lo, rr_hi)
    if np.isfinite(rr) and rr <= 0 or (np.isfinite(rr_lo) and rr_lo <= 0):
        ev, ev_ci = float("nan"), float("nan")
        warnings.append("The risk ratio implied by the estimate is not positive, so no E-value is given.")
    else:
        ev = evalue_rr(rr)
        ev_ci = _evalue_ci(rr_lo, rr_hi, rr) if np.isfinite(rr_lo) else float("nan")
    if not binary_t:
        notes.append(f"For the continuous treatment, E-values refer to a one-SD change in '{t}' ({scale:.3g}).")

    # ---- words
    rv_words = (f"A hidden confounder would need to explain {_fmt_pct(rv)} of the remaining variance of both "
                f"treatment and outcome to remove the effect, and {_fmt_pct(rva)} to make it no longer "
                "statistically distinguishable from zero (5% level).")
    ev_words = ""
    if np.isfinite(ev):
        ev_words = (f" On the risk-ratio scale (E-value), a confounder associated with both treatment and outcome "
                    f"by a risk ratio of {ev:.2f} each could explain away the estimate"
                    + (f", and {ev_ci:.2f} each could move the confidence interval to include no effect." if np.isfinite(ev_ci) else "."))
    bench_words = ""
    if bench_rows:
        first = [row for row in bench_rows if row["k"] == 1]
        if first:
            strongest = first[0]
            bench_words = (f" If a hidden confounder were as strong as '{strongest['covariate']}' (the measured covariate "
                           f"most related to treatment), the estimate would be about {strongest['adjusted_estimate']:.3g} "
                           f"(now {b:.3g}).")

    # ---- status
    status = Status.ASSUMED
    fragile = rv < 0.05
    if kind == "observational" and fragile:
        status = Status.CONCERN
        warnings.append(f"The effect is fragile: a weak hidden confounder (explaining {_fmt_pct(rv)} of the remaining "
                        "variance of treatment and outcome) would remove it.")
    if kind in ("iv", "rd", "did"):
        notes.append("This design relies on its own assumption rather than on measuring every confounder; the "
                     "numbers are for a plain adjusted regression and are shown for reference.")
    elif kind == "experiment":
        notes.append("Randomization rules out hidden confounders if carried out as described; numbers shown for reference.")

    assumption = _main_assumption(kind, rv_words + ev_words)

    # ---- charts
    charts = []
    if bench_rows:
        labels = [f"{row['k']}x {row['covariate']}" for row in bench_rows]
        charts.append(Chart(kind="bar", title="Estimate if a hidden confounder were as strong as a measured covariate",
                            data={"labels": ["observed"] + labels,
                                  "values": [b] + [row["adjusted_estimate"] for row in bench_rows],
                                  "ref": 0.0, "errors": None},
                            x_label="hidden confounder strength", y_label="adjusted estimate",
                            note="Bars that cross 0 mean a confounder that strong would remove the effect."))
    grid = np.linspace(0, min(0.6, max(0.1, 2 * rv)), 25)
    adj_line = [b - math.copysign(bias_bound(se, dof, g, g), b) if g < 1 else float("nan") for g in grid]
    charts.append(Chart(kind="line", title="Estimate vs hidden confounder strength",
                        data={"series": [{"name": "adjusted estimate", "x": grid.tolist(), "y": adj_line,
                                          "lo": None, "hi": None}],
                              "vline": rv, "hline": 0.0},
                        x_label="share of remaining variance of treatment and outcome explained by the confounder",
                        y_label="adjusted estimate",
                        note="Where the line crosses 0 is the robustness value."))

    summary = " ".join([rv_words + ev_words + bench_words] + notes +
                       ["These numbers describe how strong hidden confounding would have to be; they cannot show "
                        "that there is none."])
    metrics = {"regression_estimate": b, "regression_se": se, "t_stat": tstat, "dof": dof,
               "partial_r2_treatment_outcome": r2yd, "rv_q": rv, "rv_q_alpha": rva, "q": 1.0, "alpha": 0.05,
               "evalue_estimate": ev, "evalue_ci": ev_ci, "evalue_rr": rr, "evalue_method": e_method,
               "evalue_source": src, "fragile": fragile, "benchmarks": bench_rows, "n": n}
    return Section(key=KEY, title=TITLE, status=status, summary=summary, metrics=metrics,
                   assumptions=[assumption], charts=charts, warnings=warnings)
