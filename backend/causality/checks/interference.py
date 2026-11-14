"""Spillovers between units (interference).

Interference can only be tested when the share of treated units per cluster was itself
randomized (a randomized-saturation / two-stage design). Then untreated units in clusters
with more treated neighbours should look like untreated units elsewhere if there is no
spillover. Otherwise "no interference" is an untestable assumption (ASSUMED).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from ..schema import Assumption, Chart, Claim, Design, Prepared, Section, Status

KEY = "interference"
TITLE = "Spillovers between units"
NAME = "No interference between units (one unit's treatment does not affect another's outcome)"


def run(prep: Prepared, design: Design, claim: Optional[Claim]) -> Section:
    try:
        return _run(prep, design, claim)
    except Exception as e:  # noqa: BLE001
        return Section(key=KEY, title=TITLE, status=None,
                       error=f"The spillover check could not run ({type(e).__name__}: {e}).")


def _assumed(prep: Prepared) -> Section:
    r = prep.roles
    detail = ("Nothing in this design lets the data show whether treating one unit changes another unit's outcome "
              "(for example through neighbours, classmates or markets). It would become testable if the share of "
              "treated units within each group (village, school, market) were itself randomized, so that untreated "
              "units in heavily treated groups could be compared with untreated units in lightly treated groups.")
    notes = []
    if r.cluster and not (r.saturation and r.randomized):
        notes.append(f"The cluster column '{r.cluster}' is used for standard errors only; it does not test spillovers, "
                     "because the treated share per cluster was not randomized.")
    if r.saturation and not r.randomized:
        notes.append(f"A treated-share column '{r.saturation}' was given, but treatment was not declared randomized, "
                     "so differences across shares could reflect how clusters chose their share.")
    elif r.saturation and not r.cluster:
        notes.append(f"A treated-share column '{r.saturation}' was given but no cluster column, so the test cannot run.")
    summary = ("Spillovers between units were not tested: this design cannot show them, so the result assumes "
               "there are none.")
    if notes:
        summary += " " + notes[0]
    return Section(key=KEY, title=TITLE, status=Status.ASSUMED, summary=summary,
                   metrics={"tested": False, "notes": notes},
                   assumptions=[Assumption("no_interference", NAME, Status.ASSUMED, detail + (" " + " ".join(notes) if notes else ""),
                                           False, "interference")])


def _cluster_ols(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    import statsmodels.api as sm

    return sm.OLS(y, sm.add_constant(X, has_constant="add")).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})


def _run(prep: Prepared, design: Design, claim: Optional[Claim]) -> Section:
    r = prep.roles
    if not (r.saturation and r.randomized and r.cluster):
        return _assumed(prep)
    for c in (r.treatment, r.outcome, r.saturation, r.cluster):
        if not c or c not in prep.df:
            return Section(key=KEY, title=TITLE, error=f"The column '{c}' needed for the spillover test is not available.")
    if not prep.treatment_is_binary():
        s = _assumed(prep)
        s.summary = ("Spillovers were not tested: the randomized-saturation test needs a two-group (0/1) treatment. "
                     + s.summary)
        return s

    d = prep.complete()[[r.treatment, r.outcome, r.saturation, r.cluster]].dropna()
    t, y, sat, cl = (d[r.treatment].to_numpy(float), d[r.outcome].to_numpy(float),
                     d[r.saturation].to_numpy(float), d[r.cluster].to_numpy())
    u = t == 0
    n_clusters_u = len(np.unique(cl[u]))
    if u.sum() < 10 or n_clusters_u < 4 or len(np.unique(sat[u])) < 2:
        return Section(key=KEY, title=TITLE, status=Status.ASSUMED,
                       summary="Too few untreated units or clusters across treated shares to test for spillovers.",
                       metrics={"tested": False, "n_untreated": int(u.sum()), "n_clusters_untreated": n_clusters_u},
                       assumptions=[Assumption("no_interference", NAME, Status.ASSUMED,
                                               "The saturation design was declared but there is too little variation "
                                               "among untreated units to test it.", False, "interference")])

    fit = _cluster_ols(y[u], sat[u].reshape(-1, 1), cl[u])
    slope, se, p = float(fit.params[1]), float(fit.bse[1]), float(fit.pvalues[1])
    ci = tuple(float(v) for v in fit.conf_int()[1])
    metrics: dict = {"tested": True, "spillover_slope": slope, "spillover_se": se, "spillover_ci95": ci,
                     "spillover_p": p, "n_untreated": int(u.sum()), "n_clusters": int(len(np.unique(cl))),
                     "slope_meaning": "change in untreated units' mean outcome when the treated share of their "
                                      "cluster goes from 0 to 1 (all others treated)"}
    warnings: list[str] = []
    if len(np.unique(cl)) < 20:
        warnings.append(f"Only {len(np.unique(cl))} clusters; cluster-robust standard errors may be too small.")

    # direct effect by saturation level, mean outcomes by arm and level
    levels = sorted(np.unique(sat))
    by_level = []
    mean_t, mean_u = [], []
    for s in levels:
        m = sat == s
        mt = float(y[m & (t == 1)].mean()) if (m & (t == 1)).any() else None
        mu = float(y[m & (t == 0)].mean()) if (m & (t == 0)).any() else None
        mean_t.append(mt)
        mean_u.append(mu)
        row = {"saturation": float(s), "n": int(m.sum()), "mean_treated": mt, "mean_untreated": mu,
               "direct_effect": None, "ci95": None}
        if (m & (t == 1)).sum() >= 2 and (m & (t == 0)).sum() >= 2 and len(np.unique(cl[m])) >= 2:
            try:
                f = _cluster_ols(y[m], t[m].reshape(-1, 1), cl[m])
                row["direct_effect"] = float(f.params[1])
                row["ci95"] = tuple(float(v) for v in f.conf_int()[1])
            except Exception:  # noqa: BLE001
                pass
        by_level.append(row)
    # treated units vs untreated units in clusters where nobody was treated (pure control):
    # free of within-cluster spillover onto the comparison group
    pure = (sat == 0) & (t == 0)
    for row in by_level:
        row["effect_vs_pure_control"] = (row["mean_treated"] - float(y[pure].mean())
                                         if pure.any() and row["mean_treated"] is not None else None)
    metrics["by_saturation"] = by_level
    metrics["by_saturation_note"] = ("direct_effect compares treated and untreated units in clusters with the same "
                                     "treated share (it nets out spillover onto the untreated); effect_vs_pure_control "
                                     "compares treated units with untreated units in clusters where nobody was treated.")

    series = []
    xt = [float(s) for s, v in zip(levels, mean_t) if v is not None]
    series.append({"name": "Treated", "x": xt, "y": [v for v in mean_t if v is not None], "lo": None, "hi": None})
    xu = [float(s) for s, v in zip(levels, mean_u) if v is not None]
    series.append({"name": "Untreated", "x": xu, "y": [v for v in mean_u if v is not None], "lo": None, "hi": None})
    chart = Chart(kind="line", title="Mean outcome by share of the cluster treated",
                  data={"series": series, "vline": None, "hline": None},
                  x_label="Share of cluster treated (randomized)", y_label="Mean outcome",
                  note="If untreated units do better (or worse) as more of their cluster is treated, treatment spills over.")

    scope = ("This test only covers spillovers within clusters; spillovers between clusters (for example between "
             "neighbouring villages) are still assumed away.")
    if p < 0.05:
        status = Status.CONCERN
        summary = (f"Spillover detected: untreated units' outcomes change by {slope:+.3g} (95% CI {ci[0]:.3g} to "
                   f"{ci[1]:.3g}, p = {p:.2g}) as their cluster goes from none to all treated. A simple treated-vs-untreated "
                   "comparison mixes the direct effect with this spillover, so it may be biased; look at the direct "
                   "effect within each treated share.")
        a = Assumption("no_interference", NAME, Status.CONCERN,
                       f"Untreated units are affected by the treated share of their cluster (slope {slope:+.3g}, p = {p:.2g}). "
                       + scope, True, "interference")
        warnings.append("The direct-effect estimate may be biased by spillovers to untreated units.")
    else:
        status = Status.PASS
        summary = (f"No spillover detected within clusters: untreated units' outcomes change by {slope:+.3g} "
                   f"(95% CI {ci[0]:.3g} to {ci[1]:.3g}, p = {p:.2g}) across randomized treated shares. " + scope)
        a = Assumption("no_interference_within_clusters", "No spillovers within clusters", Status.PASS,
                       f"Slope {slope:+.3g}, p = {p:.2g}; the confidence interval shows the spillover sizes still "
                       "consistent with the data.", True, "interference")
    assumptions = [a]
    assumptions.append(Assumption("no_interference_between_clusters", "No spillovers between clusters",
                       Status.ASSUMED, scope, False, "interference"))
    return Section(key=KEY, title=TITLE, status=status, summary=summary, metrics=metrics,
                   assumptions=assumptions, charts=[chart], warnings=warnings)
