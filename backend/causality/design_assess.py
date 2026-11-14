"""Design-specific assumptions, the design's own tests, the A-F grade and the assumed DAG.

assess(prep, design, claim) -> (assumptions, sections)
    Lists the identifying assumptions of the design with a status each, and runs the tests
    the design allows (one Section per test, keys "design_<name>"):
      did : design_pretrends        event study with unit + time fixed effects, joint Wald
                                    test of the leads (reference period -1), no-anticipation
                                    check (lead -2), staggered-adoption cohorts in metrics.
      iv  : design_first_stage      robust first-stage F (single instrument: F = t^2, which
                                    equals the Montiel Olea-Pflueger effective F).
            design_instrument_balance  joint test that covariates do not predict the instrument.
      rd  : design_bunching         McCrary (2008) density test: local linear regression of
                                    histogram heights on each side, triangular kernel, test of
                                    the log-density jump. A binomial count test in a narrow
                                    symmetric window is reported alongside in metrics.
            design_covariate_jumps  local linear discontinuity in each covariate, Bonferroni.
      did : design_placebo_time     placebo in time: pre-treatment data only, adoption moved k
                                    periods earlier, two-way fixed-effects DiD (should be zero).
            design_event_study      only when the treatment is a PULSE (on for one period / a
                                    short run, then off): finance-style event-study tests
                                    (Boehmer-Musumeci-Poulsen 1991, Corrado 1989 / Corrado-Zivney
                                    1992). Informational, never graded.
      iv  : design_endogeneity      Durbin-Wu-Hausman (control-function) test. Informational.
      any design with a time role:
            design_functional_form  Ramsey RESET on the linear two-way fixed-effects model.
                                    A diagnostic of that comparison model only; never graded.
    Untestable assumptions are always ASSUMED (testable=False). ASSUMED is never a pass.
    Cards with affects_grade=False never add tested assumptions to the returned list.

grade(design, sections) -> (grade, reasons)      see the docstring of `grade`.
dag(prep, design, claim) -> Chart (kind "dag")   the ASSUMED causal diagram.

Nothing here raises: errors become Section.error in plain words.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .data import words
from .schema import Assumption, Chart, Claim, Design, Prepared, Section, Status

PULSE_MAX_RUN = 3            # a "pulse" is on for at most this many periods in a row ...
PULSE_MAX_SHARE = 0.25       # ... and for at most this share of the unit's periods, then off again
EST_WINDOW_MIN = 10
EST_WINDOW_MAX = 120
MARKET_WORDS = {"market", "mkt", "index", "benchmark"}
MAX_EVENT_LEADS = 5
MAX_EVENT_LAGS = 5
MAX_COVARIATE_TESTS = 20
MAX_DAG_CONFOUNDERS = 12


# =================================================================================== helpers
def _A(key, name, status, detail="", testable=True, src="design") -> Assumption:
    return Assumption(key=key, name=name, status=status, detail=detail, testable=testable, source=src)


def _observed(prep: Prepared, col: str) -> pd.Series:
    """True where `col` holds a real (not median-imputed) value."""
    m = prep.df[col].notna()
    flag = f"{col}__missing"
    if flag in prep.df:
        m &= prep.df[flag] != 1
    return m


def _orig_of(prep: Prepared) -> dict:
    """df column -> original column name (one-hot children and missing flags collapse)."""
    inv = {}
    for orig, cols in prep.encoding.items():
        for c in cols:
            inv[c] = orig
    return inv


def _cluster_vcov(X: np.ndarray, u: np.ndarray, groups: np.ndarray, XtX_inv: np.ndarray, n_absorbed: int = 0):
    """Cluster-robust (CR1) covariance."""
    n, k = X.shape
    _, g_idx = np.unique(groups, return_inverse=True)
    G = g_idx.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, g_idx, X * u[:, None])
    meat = S.T @ S
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return adj * XtX_inv @ meat @ XtX_inv, int(G)


def _demean_twoway(M: np.ndarray, a: np.ndarray, b: np.ndarray, tol=1e-10, max_iter=500) -> np.ndarray:
    """Remove unit (a) and time (b) fixed effects by alternating projections
    (exact for balanced and unbalanced panels at convergence)."""
    M = M.astype(float).copy()
    na, nb = a.max() + 1, b.max() + 1
    ca = np.bincount(a, minlength=na).astype(float)
    cb = np.bincount(b, minlength=nb).astype(float)
    for _ in range(max_iter):
        old = M.copy()
        for idx, cnt, ng in ((a, ca, na), (b, cb, nb)):
            sums = np.zeros((ng, M.shape[1]))
            np.add.at(sums, idx, M)
            M -= (sums / np.maximum(cnt, 1)[:, None])[idx]
        if np.max(np.abs(M - old)) < tol * (1 + np.max(np.abs(M))):
            break
    return M


def _ols_hc1(y: np.ndarray, X: np.ndarray):
    """OLS with HC1 robust covariance. X includes the constant if wanted."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    u = y - X @ b
    n, k = X.shape
    meat = (X * (u ** 2)[:, None]).T @ X
    V = XtX_inv @ meat @ XtX_inv * n / max(n - k, 1)
    return b, V


def _wls_hc1(y, X, w):
    sw = np.sqrt(w)
    return _ols_hc1(y * sw, X * sw[:, None])


def _wald(b: np.ndarray, V: np.ndarray, df_denom: Optional[int] = None) -> tuple[float, float]:
    """Joint test b = 0. Returns (statistic, p). F(q, df_denom) if df_denom else chi2(q)."""
    q = len(b)
    W = float(b @ np.linalg.pinv(V) @ b)
    if df_denom:
        F = W / q
        return F, float(stats.f.sf(F, q, df_denom))
    return W, float(stats.chi2.sf(W, q))


def _status_from_p(p: Optional[float], fail=0.05, concern=0.2) -> Status:
    if p is None or not np.isfinite(p):
        return Status.ASSUMED
    if p < fail:
        return Status.FAIL
    if p < concern:
        return Status.CONCERN
    return Status.PASS


def _f(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if np.isfinite(x) else None


# =================================================================================== assess
def assess(prep: Prepared, design: Design, claim: Optional[Claim]) -> tuple[list[Assumption], list[Section]]:
    """Design assumptions (with statuses) + design-specific test cards. Never raises."""
    assumptions: list[Assumption] = []
    sections: list[Section] = []
    kind = getattr(design, "kind", "observational")
    src = f"design.{kind}"
    try:
        if kind == "experiment":
            assumptions += _experiment_assumptions(prep, src)
        elif kind == "did":
            sec = _safe(_pretrends, prep, "design_pretrends", "Parallel trends before treatment")
            sections.append(sec)
            assumptions += sec.assumptions
            assumptions.append(_A("did_parallel_post", "Parallel trends would have continued after treatment (untestable)",
                                  Status.ASSUMED,
                                  "Without treatment, treated and comparison units would have kept moving in parallel "
                                  "after treatment started. Matching trends before treatment make this more plausible "
                                  "but can never show it.", False, src))
            assumptions.append(_A("did_composition", "Same units over time (no change in who is observed)",
                                  Status.ASSUMED,
                                  "The groups compared keep the same make-up before and after treatment.", False, src))
        elif kind == "iv":
            fs = _safe(_first_stage, prep, "design_first_stage", "Strong first stage")
            bal = _safe(_instrument_balance, prep, "design_instrument_balance", "Covariates balanced across instrument values")
            sections += [fs, bal]
            assumptions += fs.assumptions + bal.assumptions
            assumptions.append(_A("iv_independence", "Instrument as good as random",
                                  Status.ASSUMED,
                                  "The instrument is unrelated to everything else that affects the outcome. Balance "
                                  "of measured covariates is only a partial check; it cannot show this for unmeasured ones.",
                                  False, src))
            assumptions.append(_A("iv_exclusion", "Exclusion restriction: the instrument affects the outcome only through the treatment",
                                  Status.ASSUMED,
                                  "No data can show that the instrument has no other path to the outcome.", False, src))
            assumptions.append(_A("iv_monotonicity", "Monotonicity (no defiers)",
                                  Status.ASSUMED,
                                  "Nobody takes the treatment because the instrument pushed them away from it. "
                                  "Under this, the estimate is the effect for compliers only (a local effect).", False, src))
        elif kind == "rd":
            bunch = _safe(_bunching, prep, "design_bunching", "No sorting around the cutoff")
            cov = _safe(_covariate_jumps, prep, "design_covariate_jumps", "Covariates continuous at the cutoff")
            sections += [bunch, cov]
            assumptions += bunch.assumptions + cov.assumptions
            assumptions.append(_A("rd_continuity", "Continuity of potential outcomes at the cutoff",
                                  Status.ASSUMED,
                                  "Units just below and just above the cutoff would have had the same average outcome "
                                  "without treatment. Smooth covariates and no bunching make this more plausible but "
                                  "cannot show it. The effect applies only to units near the cutoff.", False, src))
        else:  # observational (and anything unknown)
            assumptions.append(_A("obs_pre_treatment_covariates",
                                  "Covariates were measured before treatment (except those you marked)",
                                  Status.ASSUMED,
                                  "Adjusting for something the treatment changed would bias the estimate. The data "
                                  "cannot tell when each column was measured.", False, src))
            assumptions.append(_A("obs_consistency", "Correct treatment timing / consistency",
                                  Status.ASSUMED,
                                  "Treatment came before the outcome, and 'treated' means the same thing for everyone "
                                  "who has it.", False, src))
    except Exception as e:  # noqa: BLE001 -- never raise
        sections.append(Section(key=f"design_{kind}", title="Design checks",
                                error=f"The design checks could not be run ({type(e).__name__}: {e})."))
    try:
        r = prep.roles
        if kind == "did":
            pl = _safe(_placebo_time, prep, "design_placebo_time", "Placebo test: no effect before treatment started")
            sections.append(pl)
            assumptions += pl.assumptions
        if kind == "iv":
            # informational, not graded: its assumptions are NOT added to the returned list
            sections.append(_safe(_endogeneity, prep, "design_endogeneity",
                                  "Do the plain and the instrument estimates differ? (Durbin-Wu-Hausman)",
                                  affects_grade=False))
        if r.time and r.treatment and r.outcome:
            sections.append(_safe(_functional_form, prep, "design_functional_form",
                                  "Functional form of the linear comparison model (RESET)", affects_grade=False))
        if r.unit and r.time and r.treatment and r.outcome:
            ev = _safe(_event_study, prep, "design_event_study",
                       "Event study tests (abnormal outcome on the event date)", affects_grade=False)
            if ev is not None:
                sections.append(ev)
                assumptions += [a for a in ev.assumptions if a.status == Status.ASSUMED]
    except Exception as e:  # noqa: BLE001 -- never raise
        sections.append(Section(key=f"design_{kind}_extra", title="Additional design checks", affects_grade=False,
                                error=f"The additional design checks could not be run ({type(e).__name__}: {e})."))
    assumptions += list(getattr(design, "conflicts", []) or [])
    return assumptions, sections


def _safe(fn, prep, key, title, affects_grade: bool = True) -> Optional[Section]:
    try:
        sec = fn(prep, key, title)
    except Exception as e:  # noqa: BLE001
        sec = Section(key=key, title=title, error=f"This test could not be run ({type(e).__name__}: {e}).")
    if sec is not None and not affects_grade:
        sec.affects_grade = False
    return sec


# ----------------------------------------------------------------------------- experiment
def _experiment_assumptions(prep: Prepared, src: str) -> list[Assumption]:
    r = prep.roles
    detail = "You said treatment was assigned at random. The data cannot confirm how it was assigned."
    if r.cluster and r.cluster in prep.df:
        k = int(prep.df[r.cluster].nunique())
        detail += (f" Treatment was declared as assigned by cluster ('{r.cluster}', {k} clusters): the analysis "
                   "must be at the cluster level (cluster-robust errors), and the effective sample size is "
                   "closer to the number of clusters than to the number of rows.")
    if r.strata:
        detail += f" Randomization within strata ('{r.strata}') should be reflected by comparing within strata."
    return [
        _A("exp_randomized", "Treatment was randomized as described", Status.ASSUMED, detail, False, src),
        _A("exp_stable_treatment", "Stable treatment (everyone treated got the same thing)", Status.ASSUMED,
           "Every treated unit received the same version of the treatment, and it was delivered as planned.", False, src),
    ]


# ----------------------------------------------------------------------------- DiD
def _pretrends(prep: Prepared, key: str, title: str) -> Section:
    r = prep.roles
    src = "design.did"
    for role in ("unit", "time", "treatment", "outcome"):
        if not getattr(r, role):
            return Section(key=key, title=title, error=f"The {role} column is not set, so the event study cannot be run.")
    d = prep.complete()
    d = d[_observed(prep, r.unit).loc[d.index] & _observed(prep, r.time).loc[d.index]]
    d = d[[r.unit, r.time, r.treatment, r.outcome]].dropna()
    if len(d) < 20:
        return Section(key=key, title=title, error="Too few complete rows to run the event study.")
    unit = pd.factorize(d[r.unit])[0]
    per = (d[r.time].rank(method="dense") - 1).astype(int).to_numpy()
    tr = (d[r.treatment].to_numpy() > 0).astype(float)
    y = d[r.outcome].to_numpy(float)

    # adoption period per unit = first period with (mostly) treated
    g = pd.DataFrame({"u": unit, "p": per, "t": tr}).groupby(["u", "p"])["t"].mean().reset_index()
    on = g[g["t"] >= 0.5]
    adopt = on.groupby("u")["p"].min()
    adopt_arr = np.full(unit.max() + 1, -1)
    adopt_arr[adopt.index.to_numpy()] = adopt.to_numpy()
    first_per = g.groupby("u")["p"].min()
    always = [u for u, a in adopt.items() if a <= first_per.loc[u]]
    # reversals: treated then untreated later
    gg = g.merge(adopt.rename("a"), left_on="u", right_index=True)
    reversals = int(gg[(gg["p"] > gg["a"]) & (gg["t"] < 0.5)]["u"].nunique())

    pulse = _pulse_info(unit, per, tr)
    n_never = int((adopt_arr < 0).sum())
    n_units = int(unit.max() + 1)
    staggered = len(set(adopt.drop(always, errors="ignore").to_numpy())) > 1

    rel = np.where(adopt_arr[unit] >= 0, per - adopt_arr[unit], np.nan)
    treated_rows = ~np.isnan(rel)
    if not treated_rows.any():
        return Section(key=key, title=title, error="No unit ever becomes treated, so there is no event to study.")
    min_rel = int(np.nanmin(rel))
    max_rel = int(np.nanmax(rel))
    K = min(MAX_EVENT_LEADS, -min_rel)
    L = min(MAX_EVENT_LAGS, max_rel)
    metrics = {"n_units": n_units, "n_periods": int(per.max() + 1), "n_never_treated": n_never,
               "adoption_cohorts": [int(c) for c in sorted(set(adopt.to_numpy()))],
               "staggered_adoption": bool(staggered), "n_always_treated": len(always),
               "n_units_switching_off": reversals, "pulse_treatment": bool(pulse["detected"])}
    warnings = []
    if pulse["detected"]:
        warnings.append(f"The treatment is a one-off event, not a lasting treatment: {pulse['n_pulse']} of "
                        f"{pulse['n_ever']} treated unit(s) are treated for a short spell and then untreated again. "
                        "Event time here is counted from each unit's event period. Points right of the line show the "
                        "outcome after the event (the treatment is off again by then), so they are not effects of "
                        "ongoing treatment; only the points left of the line are used for the test of trends "
                        "before the event.")
    elif staggered:
        warnings.append(f"Treatment starts at {len(set(adopt.to_numpy()))} different times (staggered adoption). "
                        "With effects that change over time or across cohorts, a plain two-way fixed-effects event "
                        "study mixes comparisons; treat the coefficients as a diagnostic, not as effects.")
    if reversals and not pulse["detected"]:
        warnings.append(f"{reversals} unit(s) stop being treated after starting; the event study assumes treatment stays on.")
    if always:
        warnings.append(f"{len(always)} unit(s) are treated from the first period, so they have no pre-treatment data.")

    if K < 2:
        sec = Section(key=key, title=title, status=None, metrics=metrics, warnings=warnings,
                      summary="There is at most one period before treatment, so trends before treatment cannot be "
                              "compared. Parallel trends is assumed, not tested.")
        sec.assumptions = [
            _A("did_parallel_pre", "Parallel trends before treatment", Status.ASSUMED,
               "Fewer than two pre-treatment periods: nothing to test.", True, src),
            _A("did_no_anticipation", "No anticipation (outcomes did not move just before treatment)", Status.ASSUMED,
               "Not enough pre-treatment periods to check.", True, src)]
        return sec

    # event-time dummies, endpoints binned, -1 is the reference
    ks = [k for k in range(-K, L + 1) if k != -1]
    cols, names = [], []
    for k in ks:
        if k == -K:
            m = treated_rows & (rel <= -K)
        elif k == L:
            m = treated_rows & (rel >= L)
        else:
            m = treated_rows & (rel == k)
        if m.sum() == 0:
            continue
        cols.append(m.astype(float))
        names.append(k)
    X = np.column_stack(cols)
    Xd = _demean_twoway(np.column_stack([y, X]), unit, per)
    yd, Xd = Xd[:, 0], Xd[:, 1:]
    # drop columns made redundant by the fixed effects (e.g. no never-treated group)
    keep = list(range(Xd.shape[1]))
    while keep and np.linalg.matrix_rank(Xd[:, keep]) < len(keep):
        # drop the most distant lead (or lag) endpoint first
        far = max(keep, key=lambda j: (abs(names[j]) if names[j] < 0 else abs(names[j]) - 0.5))
        keep.remove(far)
    if not keep:
        return Section(key=key, title=title, error="The event-study terms cannot be separated from the fixed effects.")
    Xd = Xd[:, keep]
    names = [names[j] for j in keep]
    XtX_inv = np.linalg.pinv(Xd.T @ Xd)
    b = XtX_inv @ Xd.T @ yd
    u = yd - Xd @ b
    V, G = _cluster_vcov(Xd, u, unit, XtX_inv)
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    z = stats.t.ppf(0.975, max(G - 1, 1))

    lead_idx = [j for j, k in enumerate(names) if k <= -2]
    metrics.update({"event_times": names, "coefficients": [float(v) for v in b], "se": [float(v) for v in se],
                    "reference_period": -1, "n_clusters": G, "n_obs": int(len(y)),
                    "endpoints_binned": {"lead": -K, "lag": L}})
    assumptions = []
    if not lead_idx:
        status = Status.ASSUMED
        p = None
        assumptions.append(_A("did_parallel_pre", "Parallel trends before treatment", Status.ASSUMED,
                              "The pre-treatment terms could not be estimated separately from the fixed effects.", True, src))
    else:
        F, p = _wald(b[lead_idx], V[np.ix_(lead_idx, lead_idx)], max(G - 1, 1))
        status = _status_from_p(p, 0.05, 0.2)
        metrics.update({"pretrend_F": F, "pretrend_p": p, "pretrend_df": [len(lead_idx), max(G - 1, 1)]})
        word = {Status.PASS: "no sign of different trends", Status.CONCERN: "weak signs of different trends",
                Status.FAIL: "clear signs of different trends"}[status]
        assumptions.append(_A("did_parallel_pre", "Parallel trends before treatment", status,
                              f"Joint test that the {len(lead_idx)} pre-treatment coefficients are all zero "
                              f"(period -1 is the reference): p = {p:.3g}; {word} before treatment. Passing this does "
                              "not show that trends would have stayed parallel afterwards.", True, src))
    # no anticipation: change from lead -2 to lead -1 (the reference)
    if -2 in names:
        j = names.index(-2)
        tval = b[j] / se[j] if se[j] > 0 else np.nan
        p2 = float(2 * stats.t.sf(abs(tval), max(G - 1, 1))) if np.isfinite(tval) else None
        st2 = Status.ASSUMED if p2 is None else (Status.CONCERN if p2 < 0.05 else Status.PASS)
        metrics.update({"anticipation_lead_minus2": float(b[j]), "anticipation_p": p2})
        assumptions.append(_A("did_no_anticipation", "No anticipation (outcomes did not move just before treatment)", st2,
                              (f"The outcome changed by {-b[j]:+.3g} between period -2 and period -1 relative to "
                               f"comparison units (p = {p2:.3g}). " if p2 is not None else "") +
                              "This is a partial check: anticipation earlier than period -2, or inside the reference "
                              "period itself, would not show here.", True, src))
    else:
        assumptions.append(_A("did_no_anticipation", "No anticipation (outcomes did not move just before treatment)",
                              Status.ASSUMED, "Period -2 is not available, so anticipation cannot be checked.", True, src))

    # chart: coefficients with 95% CIs, reference at -1 drawn as 0
    pts = sorted(list(zip(names, b, se)) + [(-1, 0.0, 0.0)])
    chart = Chart(kind="line", title="Event study: outcome relative to the period before treatment",
                  data={"series": [{"name": "Estimate (95% CI)", "x": [float(k) for k, _, _ in pts],
                                    "y": [float(v) for _, v, _ in pts],
                                    "lo": [float(v - z * s) for _, v, s in pts],
                                    "hi": [float(v + z * s) for _, v, s in pts]}],
                        "vline": 0.0, "hline": 0.0},
                  x_label="Periods since the event" if pulse["detected"] else "Periods since treatment started",
                  y_label="Difference vs period -1",
                  note=f"Unit and time fixed effects; errors clustered by unit ({G} units). End points pool "
                       f"periods {-K} and earlier / {L} and later. Points left of the line test trends before treatment.")
    worst = Status.worst([a.status for a in assumptions])
    if worst is None:
        summary = "Trends before treatment could not be tested."
    elif status == Status.FAIL:
        summary = f"Treated and comparison units were already moving differently before treatment (p = {p:.3g})."
    elif status == Status.CONCERN:
        summary = f"Some sign that treated and comparison units moved differently before treatment (p = {p:.3g})."
    else:
        summary = (f"No clear difference in trends before treatment (p = {p:.3g}). Whether they would have stayed "
                   "parallel afterwards is still an assumption.")
    return Section(key=key, title=title, status=worst, summary=summary, metrics=metrics,
                   assumptions=assumptions, charts=[chart], warnings=warnings)


# ----------------------------------------------------------------------------- panel helpers
def _pulse_info(unit: np.ndarray, per: np.ndarray, tr: np.ndarray) -> dict:
    """Is the treatment a PULSE (an event) rather than something that stays on?

    A treated unit counts as a pulse unit when its first treated spell lasts at most
    PULSE_MAX_RUN periods, it is observed untreated again afterwards, and it is treated in at
    most PULSE_MAX_SHARE of its periods. Detected when more than half of the ever-treated
    units (and at least 3) are pulse units. `event` maps pulse unit -> first treated period."""
    out = {"detected": False, "n_pulse": 0, "n_ever": 0, "n_multi": 0, "event": {}}
    g = pd.DataFrame({"u": unit, "p": per, "t": tr}).groupby(["u", "p"])["t"].mean().reset_index()
    g["on"] = g["t"] >= 0.5
    on = g[g["on"]]
    if on.empty:
        return out
    first_on = on.groupby("u")["p"].min()
    g2 = g.merge(first_on.rename("e"), left_on="u", right_index=True)
    off_after = g2[(~g2["on"]) & (g2["p"] > g2["e"])].groupby("u")["p"].min()
    g2 = g2.merge(off_after.rename("o"), left_on="u", right_index=True, how="left")
    run = g2[g2["on"] & (g2["p"] < g2["o"].fillna(np.inf))].groupby("u").size()
    n_on = on.groupby("u").size()
    n_obs = g.groupby("u").size()
    info = pd.DataFrame({"e": first_on, "run": run, "n_on": n_on}).join(n_obs.rename("n_obs"))
    info["back_off"] = info.index.isin(off_after.index)
    is_pulse = info["back_off"] & (info["run"] <= PULSE_MAX_RUN) & (info["n_on"] / info["n_obs"] <= PULSE_MAX_SHARE)
    n_pulse, n_ever = int(is_pulse.sum()), int(len(info))
    out.update({"n_pulse": n_pulse, "n_ever": n_ever,
                "n_multi": int((is_pulse & (info["n_on"] > info["run"])).sum()),
                "max_run": int(info.loc[is_pulse, "run"].max()) if n_pulse else 0,
                "event": {int(u): int(e) for u, e in info.loc[is_pulse, "e"].items()},
                "detected": bool(n_pulse >= 3 and n_pulse > 0.5 * n_ever)})
    return out


def _panel(prep: Prepared, extra: list[str] = ()) -> Optional[dict]:
    """Complete rows of the panel with unit codes, dense period numbers and each unit's
    first treated period (-1 = never treated). None if there are too few rows."""
    r = prep.roles
    d = prep.complete()
    d = d[_observed(prep, r.unit).loc[d.index] & _observed(prep, r.time).loc[d.index]]
    cols = list(dict.fromkeys([r.unit, r.time, r.treatment, r.outcome] + [c for c in extra if c in d]))
    d = d[cols].dropna()
    if len(d) < 20:
        return None
    unit = pd.factorize(d[r.unit])[0]
    per = (d[r.time].rank(method="dense") - 1).astype(int).to_numpy()
    tr = (d[r.treatment].to_numpy() > 0).astype(float)
    g = pd.DataFrame({"u": unit, "p": per, "t": tr}).groupby(["u", "p"])["t"].mean().reset_index()
    adopt = g[g["t"] >= 0.5].groupby("u")["p"].min()
    adopt_arr = np.full(unit.max() + 1, -1)
    adopt_arr[adopt.index.to_numpy()] = adopt.to_numpy()
    first_per = np.full(unit.max() + 1, per.max() + 1)
    np.minimum.at(first_per, unit, per)
    return {"d": d, "unit": unit, "per": per, "tr": tr, "y": d[r.outcome].to_numpy(float),
            "adopt": adopt_arr, "first_per": first_per}


def _demean_oneway(M: np.ndarray, a: np.ndarray) -> np.ndarray:
    M = M.astype(float)
    sums = np.zeros((a.max() + 1, M.shape[1]))
    np.add.at(sums, a, M)
    return M - (sums / np.maximum(np.bincount(a, minlength=a.max() + 1), 1)[:, None])[a]


def _fit(y: np.ndarray, X: np.ndarray, groups: Optional[np.ndarray] = None):
    """OLS; cluster-robust covariance if `groups` is given, else HC1. Returns (b, V, df)."""
    if groups is None:
        b, V = _ols_hc1(y, X)
        return b, V, max(len(y) - X.shape[1], 1)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ X.T @ y
    V, G = _cluster_vcov(X, y - X @ b, groups, XtX_inv)
    return b, V, max(G - 1, 1)


# ----------------------------------------------------------------------------- RESET
def _functional_form(prep: Prepared, key: str, title: str) -> Section:
    """Ramsey RESET on the linear fixed-effects regression: outcome on treatment + the
    adjustment set with unit and time fixed effects (time only without a unit role). Powers
    (2, then 2 and 3) of the standardized fitted values, fixed effects included, are added
    and tested jointly with a Wald test (clustered by unit when there is a unit role)."""
    r = prep.roles
    for role in ("time", "treatment", "outcome"):
        if not getattr(r, role):
            return Section(key=key, title=title, error=f"The {role} column is not set, so this test cannot be run.")
    df = prep.complete()
    m = _observed(prep, r.time).loc[df.index]
    has_unit = bool(r.unit and r.unit in df)
    if has_unit:
        m &= _observed(prep, r.unit).loc[df.index]
    adj = [c for c in prep.adjustment_set if c in df and c not in (r.treatment, r.outcome, r.time, r.unit)]
    cols = list(dict.fromkeys([r.outcome, r.treatment, r.time] + ([r.unit] if has_unit else []) + adj))
    d = df.loc[m, cols].dropna()
    if len(d) < 30:
        return Section(key=key, title=title, error="Too few complete rows to test the functional form.")
    per = pd.factorize(d[r.time])[0]
    unit = pd.factorize(d[r.unit])[0] if has_unit else None
    y = d[r.outcome].to_numpy(float)
    X = np.column_stack([d[c].to_numpy(float) for c in [r.treatment] + adj])

    def within(M):
        return _demean_twoway(M, unit, per) if has_unit else _demean_oneway(M, per)

    Z = within(np.column_stack([y, X]))
    yd, Xd = Z[:, 0], Z[:, 1:]
    scale = np.sqrt((Xd ** 2).sum(0))
    raw_scale = np.sqrt(((X - X.mean(0)) ** 2).sum(0))
    keepc = scale > 1e-8 * np.maximum(raw_scale, 1e-300)
    Xd = Xd[:, keepc]
    if Xd.shape[1] == 0:
        return Section(key=key, title=title,
                       error="The treatment and covariates do not vary once the fixed effects are removed.")
    b = np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ yd
    resid = yd - Xd @ b
    fitted = y - resid                      # fitted values including the fixed effects
    sd = fitted.std()
    if not np.isfinite(sd) or sd <= 0:
        return Section(key=key, title=title, error="The fitted values do not vary, so the test cannot be run.")
    z = (fitted - fitted.mean()) / sd

    def reset(powers):
        P = within(np.column_stack([z ** k for k in powers]))
        # the part of the powers not already explained by the linear terms must vary
        Pr = P - Xd @ (np.linalg.pinv(Xd.T @ Xd) @ Xd.T @ P)
        if np.any((Pr ** 2).sum(0) < 1e-8 * np.maximum((P ** 2).sum(0), 1e-300)):
            return None
        Xa = np.column_stack([Xd, P])
        ba, Va, dfd = _fit(yd, Xa, unit)
        q = len(powers)
        if dfd <= q:
            return None
        F, p = _wald(ba[-q:], Va[-q:, -q:], dfd)
        return {"F": _f(F), "p": _f(p), "df": [q, int(dfd)]} if _f(p) is not None else None

    t2, t23 = reset([2]), reset([2, 3])
    main = t23 or t2
    fe_words = "unit and time fixed effects" if has_unit else "time fixed effects only (no unit column)"
    se_words = f"errors clustered by '{r.unit}'" if has_unit else "heteroskedasticity-robust errors"
    metrics = {"method": f"Ramsey RESET on the linear regression with {fe_words}; Wald test, {se_words}",
               "reset_degree2": t2, "reset_degree2_3": t23,
               "p": main["p"] if main else None, "n_obs": int(len(d)), "n_covariates": int(Xd.shape[1] - 1),
               "fixed_effects": "unit+time" if has_unit else "time",
               "n_clusters": int(unit.max() + 1) if has_unit else None}
    scope = ("This speaks only to the linear two-way fixed-effects comparison model, not to the main "
             "Callaway-Sant'Anna estimate, which does not rely on that model.")
    if main is None:
        return Section(key=key, title=title, status=None, metrics=metrics,
                       summary="The functional-form test could not be computed (the added terms cannot be separated "
                               "from the model). Nothing is concluded from it. " + scope)
    p = main["p"]
    parts = []
    if t2:
        parts.append(f"squared fitted values: p = {t2['p']:.3g}")
    if t23:
        parts.append(f"squared and cubed together: p = {t23['p']:.3g}")
    if p >= 0.05:
        status = Status.PASS
        summary = (f"No sign that the straight-line model with fixed effects misses curvature ({'; '.join(parts)}). "
                   + scope + " Passing it does not show that the design is valid, and it says nothing about "
                   "parallel trends or hidden confounders.")
    else:
        status = Status.CONCERN
        summary = (f"The straight-line model with fixed effects seems to miss curvature ({'; '.join(parts)}): the "
                   "outcome may be better compared on another scale (for example in logs), or effects may differ "
                   "with the level of the outcome. " + scope + " It is a warning about that comparison model, "
                   "not evidence for or against a causal effect.")
    # binned residuals against fitted values
    charts = []
    try:
        nb = int(min(20, max(4, len(z) // 20)))
        order = np.argsort(z, kind="stable")
        chunks = np.array_split(order, nb)
        pts = [{"x": float(fitted[c].mean()), "y": float(resid[c].mean()), "group": None} for c in chunks if len(c)]
        charts.append(Chart(kind="scatter", title="What the linear model leaves over, by fitted value",
                            data={"points": pts, "lines": [], "hline": 0.0},
                            x_label="Fitted value (bin average)", y_label="Average leftover (residual)",
                            note="Each point averages one slice of the data ordered by fitted value. A curve or "
                                 "U-shape means the straight-line model misses something; a flat cloud around zero "
                                 "does not prove the model right."))
    except Exception:  # noqa: BLE001
        pass
    return Section(key=key, title=title, status=status, summary=summary, metrics=metrics, charts=charts)


# ----------------------------------------------------------------------------- placebo in time
PLACEBO_GROUP_NOTE = ("A placebo GROUP test (the same comparison on a group known to be unaffected by the "
                      "treatment) was not run: it needs you to name such a group, which the data cannot do.")


def _placebo_time(prep: Prepared, key: str, title: str) -> Section:
    r = prep.roles
    src = "design.did"
    for role in ("unit", "time", "treatment", "outcome"):
        if not getattr(r, role):
            return Section(key=key, title=title, error=f"The {role} column is not set, so the placebo test cannot be run.")
    P = _panel(prep)
    if P is None:
        return Section(key=key, title=title, error="Too few complete rows to run the placebo test.")
    unit, per, y, adopt, first_per = P["unit"], P["per"], P["y"], P["adopt"], P["first_per"]
    if not (adopt >= 0).any():
        return Section(key=key, title=title, error="No unit ever becomes treated, so there is no placebo test to run.")
    metrics: dict = {"placebo_group_note": PLACEBO_GROUP_NOTE}

    def not_run(why: str) -> Section:
        return Section(key=key, title=title, status=None, metrics=metrics,
                       summary=f"The placebo-in-time test could not be run: {why} Nothing is concluded from it. "
                               + PLACEBO_GROUP_NOTE)

    switchers = (adopt >= 0) & (adopt > first_per)
    if not switchers.any():
        return not_run("every treated unit is treated from its first period, so there is no pre-treatment data.")
    a_min = int(adopt[switchers].min())
    n_pre = int((np.unique(per) < a_min).sum())
    metrics["n_pre_periods_earliest_cohort"] = n_pre
    if n_pre < 2:
        return not_run(f"the earliest treated units have only {n_pre} period(s) before treatment; at least 2 are "
                       "needed (one before and one after the pretend start).")
    k = max(1, n_pre // 2)
    keep = (adopt[unit] < 0) | (per < adopt[unit])
    u, p_, yy = unit[keep], per[keep], y[keep]
    D = ((adopt[u] >= 0) & (p_ >= adopt[u] - k)).astype(float)
    uc, pc = pd.factorize(u)[0], pd.factorize(p_)[0]
    Z = _demean_twoway(np.column_stack([yy, D]), uc, pc)
    yd, Dd = Z[:, 0], Z[:, 1]
    sdd = float(Dd @ Dd)
    if D.sum() == 0 or sdd < 1e-8 * max(len(D), 1):
        return not_run("in the periods before treatment the pretend treatment cannot be separated from the unit "
                       "and time effects (no comparison units are left).")
    b, V, dfd = _fit(yd, Dd[:, None], uc)
    est, se = float(b[0]), float(np.sqrt(max(V[0, 0], 0)))
    if not se > 0:
        return not_run("the standard error of the placebo effect is zero.")
    pval = float(2 * stats.t.sf(abs(est / se), dfd))
    tc = float(stats.t.ppf(0.975, dfd))
    ci = (est - tc * se, est + tc * se)
    sd_y = float(np.std(yy))
    metrics.update({"placebo_effect": est, "se": se, "p": pval, "ci95": [ci[0], ci[1]], "shift_periods": int(k),
                    "n_obs": int(len(yy)), "n_clusters": int(dfd + 1), "outcome_sd_pre": sd_y,
                    "n_placebo_treated_units": int(np.unique(u[D == 1]).size),
                    "n_comparison_only_units": int((adopt[np.unique(u)] < 0).sum()),
                    "method": "two-way fixed effects (unit and period) on pre-treatment observations only, "
                              "errors clustered by unit; no covariates"})
    status = Status.FAIL if pval < 0.01 else (Status.CONCERN if pval < 0.05 else Status.PASS)
    how = (f"Using only data from before each unit's real start, and pretending treatment started {k} period(s) "
           f"earlier, the 'effect' is {est:+.3g} (95% CI {ci[0]:+.3g} to {ci[1]:+.3g}, p = {pval:.3g}). "
           "It should be zero, because nothing had happened yet.")
    if status == Status.PASS:
        tail = (" No effect shows up before treatment. This does not show that trends would have stayed parallel "
                "afterwards; that stays an assumption.")
    elif status == Status.CONCERN:
        tail = " Some 'effect' shows up before treatment started, which points to different trends or anticipation."
    else:
        tail = (" A clear 'effect' shows up before treatment started: treated and comparison units were already "
                "moving apart, so the real estimate cannot be read as the effect of treatment alone.")
    a = _A("did_placebo_time", "No effect before treatment started (placebo in time)", status, how + tail, True, src)
    chart = Chart(kind="forest", title="Pretend effect before treatment started (should be zero)",
                  data={"rows": [{"label": f"Start moved {k} period(s) earlier", "est": est, "lo": ci[0], "hi": ci[1]}],
                        "ref": 0.0},
                  x_label=f"Change in {r.outcome}",
                  note="Estimated only on observations from before each unit's real start. An interval away from zero "
                       "is a warning sign; an interval around zero is not proof of parallel trends.")
    return Section(key=key, title=title, status=status, summary=how + tail + " " + PLACEBO_GROUP_NOTE,
                   metrics=metrics, assumptions=[a], charts=[chart])


# ----------------------------------------------------------------------------- event study (pulse)
def _market_covariate(prep: Prepared) -> Optional[str]:
    for c in prep.adjustment_set:
        if c in prep.df and not c.endswith("__missing") and set(words(c)) & MARKET_WORDS:
            return c
    return None


def _event_study(prep: Prepared, key: str, title: str) -> Optional[Section]:
    """Finance-style event-study tests, only when the treatment is a pulse. Returns None when
    it is not (no card). Day 0 = the first treated period of each unit.

    BMP (Boehmer, Musumeci & Poulsen 1991): SAR_i = AR_i0 / (s_i * sqrt(C_i)), with s_i the
    estimation-window residual SD and C_i the forecast-error correction (market model:
    1 + 1/M_i + (Rm_0 - mean Rm)^2 / sum (Rm_t - mean Rm)^2; mean-adjusted: 1 + 1/M_i);
    t = mean(SAR) / (sd(SAR) / sqrt(N)) against t(N-1).
    Corrado (1989) rank test with the Corrado & Zivney (1992) standardized ranks
    U_it = rank(AR_it) / (1 + L_i) over estimation window + day 0 (L_i observations):
    z = [N^-1/2 sum_i (U_i0 - 1/2)] / S, S^2 = mean over window days t of
    [N_t^-1/2 sum_i (U_it - 1/2)]^2, against the standard normal."""
    r = prep.roles
    mkt = _market_covariate(prep)
    P = _panel(prep, [mkt] if mkt else [])
    if P is None:
        return None
    unit, per, y = P["unit"], P["per"], P["y"]
    pulse = _pulse_info(unit, per, P["tr"])
    if not pulse["detected"]:
        return None
    src = "design.event_study"
    event = pulse["event"]
    frame = pd.DataFrame({"u": unit, "p": per, "y": y})
    if mkt:
        frame["m"] = P["d"][mkt].to_numpy(float)
    frame = frame[frame["u"].isin(list(event))].groupby(["u", "p"], sort=True).mean().reset_index()

    W = EST_WINDOW_MAX
    ar0, sars, cars, ev_periods = [], [], [], []
    U = []                       # per unit: standardized ranks, last entry = day 0
    prof_sum, prof_n = np.zeros(21), np.zeros(21)
    n_short = n_flat = 0
    for u_id, grp in frame.groupby("u", sort=False):
        e = event[int(u_id)]
        pp = grp["p"].to_numpy()
        yy = grp["y"].to_numpy(float)
        i0 = int(np.searchsorted(pp, e))
        pre = np.arange(max(0, i0 - W), i0)
        M = len(pre)
        if M < EST_WINDOW_MIN or i0 >= len(pp) or pp[i0] != e:
            n_short += 1
            continue
        if mkt:
            mm = grp["m"].to_numpy(float)
            xm, ym = mm[pre].mean(), yy[pre].mean()
            sxx = float(((mm[pre] - xm) ** 2).sum())
            if sxx <= 0:
                n_flat += 1
                continue
            beta = float(((mm[pre] - xm) * (yy[pre] - ym)).sum() / sxx)
            ar = yy - (ym - beta * xm) - beta * mm
            s2 = float((ar[pre] ** 2).sum() / (M - 2))
            C = 1 + 1 / M + (mm[i0] - xm) ** 2 / sxx
        else:
            ar = yy - yy[pre].mean()
            s2 = float((ar[pre] ** 2).sum() / (M - 1))
            C = 1 + 1 / M
        if not s2 > 0:
            n_flat += 1
            continue
        ar0.append(float(ar[i0]))
        sars.append(float(ar[i0] / np.sqrt(s2 * C)))
        ev_periods.append(e)
        comb = np.r_[ar[pre], ar[i0]]
        U.append(stats.rankdata(comb) / (len(comb) + 1))
        pos = {int(q): j for j, q in enumerate(pp)}
        if e - 1 in pos and e + 1 in pos:
            cars.append(float(ar[pos[e - 1]] + ar[i0] + ar[pos[e + 1]]))
        for kk in range(-10, 11):
            j = pos.get(e + kk)
            if j is not None:
                prof_sum[kk + 10] += ar[j]
                prof_n[kk + 10] += 1

    N = len(ar0)
    model = "market model" if mkt else "mean-adjusted"
    metrics = {"n_events": N, "model": model, "market_column": mkt, "mean_abnormal": None, "bmp_stat": None,
               "bmp_p": None, "corrado_stat": None, "corrado_p": None, "car_m1_p1": None,
               "n_pulse_units": pulse["n_pulse"], "n_treated_units": pulse["n_ever"],
               "n_units_skipped_short_window": n_short, "n_units_skipped_no_variation": n_flat,
               "estimation_window": f"up to {EST_WINDOW_MAX} periods before the event (at least {EST_WINDOW_MIN}), "
                                    "event period excluded",
               "event_window": "the event period only (day 0)",
               "mean_abnormal_se": (float(np.std(ar0, ddof=1) / np.sqrt(N)) if N >= 2 else None)}
    assumed = _A("event_no_confounding_events", "Nothing else happened to these units on the event date",
                 Status.ASSUMED,
                 "An abnormal outcome on the event date is attributed to the event only if no other news or shock "
                 "hit the same units on the same date. The data cannot show this.", False, src)
    if N < 3:
        return Section(key=key, title=title, status=None, metrics=metrics, assumptions=[assumed],
                       summary=f"The treatment looks like a one-off event, but only {N} unit(s) have at least "
                               f"{EST_WINDOW_MIN} periods before their event, so the event-study tests were not run.")
    ar0a, sara = np.array(ar0), np.array(sars)
    mean_ar = float(ar0a.mean())
    sd_sar = float(sara.std(ddof=1))
    bmp = float(sara.mean() / (sd_sar / np.sqrt(N))) if sd_sar > 0 else None
    bmp_p = float(2 * stats.t.sf(abs(bmp), N - 1)) if bmp is not None else None
    # Corrado / Corrado-Zivney: align windows on the event (last column = day 0)
    Lmax = max(len(v) for v in U)
    Um = np.full((N, Lmax), np.nan)
    for i, v in enumerate(U):
        Um[i, Lmax - len(v):] = v
    dev = Um - 0.5
    n_t = (~np.isnan(dev)).sum(0)
    A = np.nansum(dev, axis=0) / np.sqrt(np.maximum(n_t, 1))
    S = float(np.sqrt(np.mean(A[n_t > 0] ** 2)))
    corr = float(A[-1] / S) if S > 0 else None
    corr_p = float(2 * stats.norm.sf(abs(corr))) if corr is not None else None
    car = float(np.mean(cars)) if cars else None
    car_p = None
    if len(cars) >= 3 and np.std(cars, ddof=1) > 0:
        car_p = float(stats.ttest_1samp(cars, 0.0).pvalue)
    vals, counts = np.unique(ev_periods, return_counts=True)
    top_share = float(counts.max() / N)
    metrics.update({"mean_abnormal": mean_ar, "median_abnormal": float(np.median(ar0a)),
                    "share_positive": float((ar0a > 0).mean()), "bmp_stat": bmp, "bmp_p": bmp_p,
                    "corrado_stat": corr, "corrado_p": corr_p, "car_m1_p1": car, "car_m1_p1_n": len(cars),
                    "car_m1_p1_p": car_p, "n_event_dates": int(len(vals)), "max_share_same_date": top_share,
                    "mean_estimation_window": float(np.mean([len(v) - 1 for v in U]))})
    warnings = []
    if top_share > 0.2:
        warnings.append(f"{top_share:.0%} of the events fall in the same period. Units hit on the same date move "
                        "together, and the Boehmer-Musumeci-Poulsen test then rejects too often (Kolari & Pynnonen "
                        "2010); read its p-value as too small.")
    if pulse["n_multi"]:
        warnings.append(f"{pulse['n_multi']} unit(s) have more than one event; only the first is used; later "
                        "events are ignored.")
    if pulse.get("max_run", 1) > 1:
        warnings.append("For some units the treatment lasts more than one period; day 0 is the first of them.")
    if n_short or n_flat:
        warnings.append(f"{n_short + n_flat} treated unit(s) were left out (fewer than {EST_WINDOW_MIN} periods "
                        "before the event, or no variation to fit the expected-outcome model).")
    if pulse["n_ever"] > pulse["n_pulse"]:
        warnings.append(f"{pulse['n_ever'] - pulse['n_pulse']} treated unit(s) do not look like one-off events "
                        "(treatment stays on) and are not part of these tests.")
    if N < 30:
        warnings.append(f"Only {N} events: both tests lean on large-sample approximations.")
    model_words = (f"a market model (each unit's outcome regressed on '{mkt}' over its estimation window)" if mkt else
                   "the mean-adjusted model (each unit's own average over its estimation window; no market-like "
                   "column was found)")

    def ptxt(p):
        return "not computable" if p is None else f"p = {p:.3g}"

    summary = (f"The treatment is a one-off event for {pulse['n_pulse']} unit(s). On the event date the outcome was "
               f"on average {mean_ar:+.3g} away from what {model_words} expected ({N} events). "
               f"Boehmer-Musumeci-Poulsen test: {ptxt(bmp_p)}; Corrado rank test: {ptxt(corr_p)}."
               + (f" Summed over the periods -1 to +1 the average is {car:+.3g}." if car is not None else "")
               + " These tests ask whether the outcome moved unusually on the event date. They do not show that "
               "the event caused the move: anything else that happened to these units on the same date is counted in.")
    ks = [k for k in range(-10, 11) if prof_n[k + 10] > 0]
    chart = Chart(kind="line", title="Average abnormal outcome around the event",
                  data={"series": [{"name": "Average abnormal outcome", "x": [float(k) for k in ks],
                                    "y": [float(prof_sum[k + 10] / prof_n[k + 10]) for k in ks],
                                    "lo": None, "hi": None}],
                        "vline": 0.0, "hline": 0.0},
                  x_label="Periods relative to the event", y_label=f"Abnormal {r.outcome}",
                  note=f"Abnormal = observed minus expected under {model}. Periods before the event are inside the "
                       "estimation window, so they average close to zero by construction.")
    return Section(key=key, title=title, status=None, summary=summary, metrics=metrics, assumptions=[assumed],
                   charts=[chart], warnings=warnings)


# ----------------------------------------------------------------------------- IV: Durbin-Wu-Hausman
def _endogeneity(prep: Prepared, key: str, title: str) -> Section:
    """Regression-based (control-function) Durbin-Wu-Hausman test: the first-stage residual is
    added to the outcome regression and its coefficient is t-tested with robust errors."""
    r = prep.roles
    for role in ("instrument", "treatment", "outcome"):
        if not getattr(r, role):
            return Section(key=key, title=title, error=f"The {role} column is not set, so this test cannot be run.")
    df = prep.df
    mask = _observed(prep, r.instrument) & df[r.treatment].notna() & df[r.outcome].notna()
    adj = [c for c in prep.adjustment_set if c in df and c not in (r.instrument, r.treatment, r.outcome)]
    d = df.loc[mask, list(dict.fromkeys([r.instrument, r.treatment, r.outcome] + adj))].dropna()
    adj = [c for c in adj if d[c].nunique() > 1]
    if len(d) < 20 or d[r.instrument].nunique() < 2 or d[r.treatment].nunique() < 2:
        return Section(key=key, title=title,
                       error="Too few complete rows, or the instrument or the treatment does not vary.")
    groups = None
    se_type = "heteroskedasticity-robust (HC1)"
    if r.cluster and r.cluster in prep.df and prep.df.loc[d.index, r.cluster].notna().all():
        groups = pd.factorize(prep.df.loc[d.index, r.cluster])[0]
        se_type = f"cluster-robust by '{r.cluster}' ({int(groups.max() + 1)} clusters)"
    one = np.ones(len(d))
    Zv, T, Y = (d[c].to_numpy(float) for c in (r.instrument, r.treatment, r.outcome))
    Xc = [d[c].to_numpy(float) for c in adj]
    X1 = np.column_stack([one, Zv] + Xc)
    b1, V1, _ = _fit(T, X1, groups)
    se1 = float(np.sqrt(max(V1[1, 1], 0)))
    F = float((b1[1] / se1) ** 2) if se1 > 0 else None
    v = T - X1 @ b1
    b_ols, _, _ = _fit(Y, np.column_stack([one, T] + Xc), groups)
    X2 = np.column_stack([one, T] + Xc + [v])
    if np.linalg.matrix_rank(X2) < X2.shape[1]:
        return Section(key=key, title=title,
                       error="The instrument does not move the treatment at all in this sample, so the test "
                             "cannot be computed.")
    b2, V2, dfd = _fit(Y, X2, groups)
    se = float(np.sqrt(max(V2[-1, -1], 0)))
    if not se > 0:
        return Section(key=key, title=title, error="The test statistic could not be computed.")
    tval = float(b2[-1] / se)
    p = float(2 * stats.t.sf(abs(tval), dfd)) if groups is not None else float(2 * stats.norm.sf(abs(tval)))
    metrics = {"method": "Durbin-Wu-Hausman, regression (control-function) version: first-stage residual added "
                         "to the outcome regression",
               "residual_coef": float(b2[-1]), "residual_se": se, "t": tval, "p": p, "se_type": se_type,
               "ols_estimate": float(b_ols[1]), "iv_estimate": float(b2[1]),
               "difference_ols_minus_iv": float(b_ols[1] - b2[1]), "first_stage_F": F, "n": int(len(d)),
               "n_covariates": len(adj)}
    warnings = []
    if F is None or F < 10:
        warnings.append("The first stage is weak (F " + (f"= {F:.1f}" if F is not None else "not computable")
                        + " < 10). With a weak instrument this test is uninformative: a large p-value does not "
                          "mean the treatment is free of confounding.")
    gap = f"plain regression {b_ols[1]:+.3g} vs instrument estimate {b2[1]:+.3g}"
    if p < 0.05:
        lead = (f"The plain regression and the instrument estimate disagree ({gap}; p = {p:.3g}): the treatment "
                "looks tied to unmeasured drivers of the outcome, so the instrument estimate is the one to read.")
    else:
        lead = (f"No detectable difference between the plain regression and the instrument estimate ({gap}; "
                f"p = {p:.3g}). That is not evidence that the treatment is free of hidden confounding; the test "
                "may simply lack power.")
    caveat = (" This test ASSUMES the instrument is valid. It cannot test the exclusion restriction or whether "
              "the instrument is as good as random, and it tells you nothing when the first stage is weak.")
    return Section(key=key, title=title, status=None, summary=lead + caveat, metrics=metrics, warnings=warnings)


# ----------------------------------------------------------------------------- IV
def _iv_frame(prep: Prepared, need_treatment=True) -> tuple[pd.DataFrame, list[str]]:
    r = prep.roles
    d = prep.df
    mask = _observed(prep, r.instrument)
    if need_treatment:
        mask &= d[r.treatment].notna()
    adj = [c for c in prep.adjustment_set if c in d and c != r.instrument]
    cols = [r.instrument] + ([r.treatment] if need_treatment else []) + adj
    out = d.loc[mask, cols].dropna()
    adj = [c for c in adj if out[c].nunique() > 1]
    return out, adj


def _first_stage(prep: Prepared, key: str, title: str) -> Section:
    r = prep.roles
    src = "design.iv"
    if not r.instrument or not r.treatment:
        return Section(key=key, title=title, error="The instrument or the treatment column is not set.")
    d, adj = _iv_frame(prep)
    if len(d) < 20 or d[r.instrument].nunique() < 2:
        return Section(key=key, title=title, error="Too few rows with an observed instrument, or the instrument does not vary.")
    y = d[r.treatment].to_numpy(float)
    X = np.column_stack([np.ones(len(d)), d[r.instrument].to_numpy(float)] + [d[c].to_numpy(float) for c in adj])
    if r.cluster and r.cluster in prep.df:
        XtX_inv = np.linalg.pinv(X.T @ X)
        b = XtX_inv @ X.T @ y
        V, G = _cluster_vcov(X, y - X @ b, prep.df.loc[d.index, r.cluster].to_numpy(), XtX_inv)
        se_type = f"cluster-robust by '{r.cluster}' ({G} clusters)"
    else:
        b, V = _ols_hc1(y, X)
        se_type = "heteroskedasticity-robust (HC1)"
    coef, se = float(b[1]), float(np.sqrt(max(V[1, 1], 0)))
    F = (coef / se) ** 2 if se > 0 else float("inf")
    metrics = {"first_stage_coef": coef, "first_stage_se": se, "robust_F": F, "effective_F": F,
               "n": int(len(d)), "se_type": se_type, "n_covariates": len(adj)}
    zv = d[r.instrument]
    if set(zv.unique()) <= {0, 1}:
        s1 = float(d.loc[zv == 1, r.treatment].mean())
        s0 = float(d.loc[zv == 0, r.treatment].mean())
        metrics.update({"treated_share_instrument_1": s1, "treated_share_instrument_0": s0, "complier_share": s1 - s0})
    if F < 10:
        status = Status.FAIL
        word = "weak: the instrument barely moves the treatment, so the estimate can be badly biased"
    elif F < 20:
        status = Status.CONCERN
        word = "borderline"
    else:
        status = Status.PASS
        word = "strong"
    detail = (f"Robust first-stage F = {F:.1f} ({word}). For a single instrument this equals the effective F of "
              "Montiel Olea and Pflueger. Rule of thumb: below 10 is weak. Lee et al. (2022) show the usual "
              "t-test needs F above 104.7 to have its stated 5% size"
              + (" (this F is below that)" if F < 104.7 else " (this F is above that)") +
              "; the Anderson-Rubin interval reported with the estimate stays valid however weak the instrument is.")
    a = _A("iv_first_stage", "Strong first stage", status, detail, True, src)
    summary = {Status.FAIL: f"Weak instrument (F = {F:.1f} < 10): the instrument barely shifts the treatment.",
               Status.CONCERN: f"Borderline first stage (F = {F:.1f}, between 10 and 20).",
               Status.PASS: f"The instrument strongly shifts the treatment (F = {F:.1f})."}[status]
    return Section(key=key, title=title, status=status, summary=summary, metrics=metrics, assumptions=[a])


def _instrument_balance(prep: Prepared, key: str, title: str) -> Section:
    r = prep.roles
    src = "design.iv"
    if not r.instrument:
        return Section(key=key, title=title, error="The instrument column is not set.")
    d, adj = _iv_frame(prep, need_treatment=False)
    adj = adj[:MAX_COVARIATE_TESTS]
    if not adj:
        a = _A("iv_covariate_balance", "Covariates balanced across instrument values", Status.ASSUMED,
               "There are no covariates to compare, so this was not tested.", True, src)
        return Section(key=key, title=title, status=None, summary="No covariates to check balance on.", assumptions=[a])
    zv = d[r.instrument].to_numpy(float)
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in adj])
    b, V = _ols_hc1(zv, X)
    W, p = _wald(b[1:], V[1:, 1:], max(len(d) - X.shape[1], 1))
    metrics = {"balance_F": W, "balance_p": p, "n_covariates": len(adj), "n": int(len(d))}
    smds = {}
    if set(np.unique(zv)) <= {0.0, 1.0}:
        for c in adj:
            a1, a0 = d.loc[zv == 1, c], d.loc[zv == 0, c]
            sd = np.sqrt((a1.var() + a0.var()) / 2)
            smds[c] = float((a1.mean() - a0.mean()) / sd) if sd > 0 else 0.0
        metrics["smd"] = smds
    max_smd = max((abs(v) for v in smds.values()), default=None)
    metrics["max_abs_smd"] = max_smd
    big = max_smd is None or max_smd >= 0.1
    if p < 0.01 and big:
        status = Status.FAIL
    elif p < 0.05:
        status = Status.CONCERN
    else:
        status = Status.PASS
    detail = (f"Joint test that the {len(adj)} covariate(s) do not predict the instrument: p = {p:.3g}"
              + (f", largest standardized difference {max_smd:.2f}" if max_smd is not None else "") + ". "
              "This only checks measured covariates; the instrument being as good as random stays an assumption.")
    a = _A("iv_covariate_balance", "Covariates balanced across instrument values", status, detail, True, src)
    summary = {Status.PASS: "Measured covariates look similar across instrument values.",
               Status.CONCERN: "Some imbalance in measured covariates across instrument values.",
               Status.FAIL: "Measured covariates differ across instrument values: the instrument does not look random."}[status]
    charts = []
    if smds:
        top = sorted(smds.items(), key=lambda kv: -abs(kv[1]))[:20]
        charts.append(Chart(kind="bar", title="Standardized differences by instrument value",
                            data={"labels": [k for k, _ in top], "values": [v for _, v in top], "ref": 0.1, "errors": None},
                            y_label="Standardized difference (instrument 1 vs 0)",
                            note="Differences beyond 0.1 are usually called imbalanced."))
    return Section(key=key, title=title, status=status, summary=summary, metrics=metrics,
                   assumptions=[a], charts=charts)


# ----------------------------------------------------------------------------- RD
def _rd_frame(prep: Prepared, extra: list[str] = ()) -> Optional[pd.DataFrame]:
    r = prep.roles
    if not r.running or r.cutoff is None or r.running not in prep.df:
        return None
    m = _observed(prep, r.running)
    return prep.df.loc[m, [r.running] + [c for c in extra if c in prep.df]]


def _rd_bandwidth(x: np.ndarray, c: float) -> float:
    """Rule-of-thumb bandwidth 1.84 * sd * n^(-1/5) (uniform-kernel Silverman rule), widened
    if needed so each side has at least 30 points."""
    n = len(x)
    h = 1.84 * np.std(x) * n ** (-0.2)
    for _ in range(30):
        if ((x >= c - h) & (x < c)).sum() >= 30 and ((x >= c) & (x <= c + h)).sum() >= 30:
            break
        h *= 1.25
    return float(h)


def _mccrary(x: np.ndarray, c: float) -> dict:
    """McCrary (2008) density test: histogram with bins aligned to the cutoff, local linear
    fits of bin heights on each side with a triangular kernel, theta = log f+ - log f-."""
    n = len(x)
    sd = np.std(x)
    b = 2 * sd * n ** -0.5
    h = max(1.84 * sd * n ** -0.2, 10 * b)
    lo, hi = x.min(), x.max()
    out = {"bin_width": float(b), "bandwidth": float(h)}
    fits = {}
    for side in ("left", "right"):
        if side == "left":
            j = np.arange(1, int(np.floor((c - lo) / b)) + 1)
            edges_hi = c - (j - 1) * b
            edges_lo = c - j * b
        else:
            j = np.arange(1, int(np.floor((hi - c) / b)) + 1)
            edges_lo = c + (j - 1) * b
            edges_hi = c + j * b
        mids = (edges_lo + edges_hi) / 2
        sel = np.abs(mids - c) <= h
        edges_lo, edges_hi, mids = edges_lo[sel], edges_hi[sel], mids[sel]
        if len(mids) < 3:
            return {**out, "error": "too few bins on one side of the cutoff"}
        if side == "left":
            counts = np.array([((x >= a) & (x < e)).sum() for a, e in zip(edges_lo, edges_hi)])
        else:
            counts = np.array([((x >= a) & (x < e)).sum() for a, e in zip(edges_lo, edges_hi)])
        heights = counts / (n * b)
        w = np.clip(1 - np.abs(mids - c) / h, 0, None)
        X = np.column_stack([np.ones(len(mids)), mids - c])
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(X * sw[:, None], heights * sw, rcond=None)[0]
        fits[side] = float(coef[0])
    fl, fr = fits["left"], fits["right"]
    out.update({"density_left": fl, "density_right": fr})
    if fl <= 0 or fr <= 0:
        return {**out, "error": "estimated density at the cutoff is not positive on one side"}
    theta = np.log(fr) - np.log(fl)
    se = np.sqrt((1 / (n * h)) * (24 / 5) * (1 / fr + 1 / fl))
    p = float(2 * stats.norm.sf(abs(theta / se)))
    out.update({"log_density_jump": float(theta), "se": float(se), "p": p})
    return out


def _binomial_window(x: np.ndarray, c: float, w: float) -> dict:
    below = int(((x >= c - w) & (x < c)).sum())
    above = int(((x >= c) & (x < c + w)).sum())
    if below + above == 0:
        return {"window": w, "below": below, "above": above, "p": None}
    p = float(stats.binomtest(above, below + above, 0.5).pvalue)
    return {"window": float(w), "below": below, "above": above, "p": p}


def _bunching(prep: Prepared, key: str, title: str) -> Section:
    r = prep.roles
    src = "design.rd"
    d = _rd_frame(prep)
    if d is None:
        return Section(key=key, title=title, error="The running variable or the cutoff is not set.")
    x = d[r.running].dropna().to_numpy(float)
    c = float(r.cutoff)
    if (x < c).sum() < 10 or (x >= c).sum() < 10:
        return Section(key=key, title=title, error="Too few units on one side of the cutoff to test for sorting.")
    mc = _mccrary(x, c)
    binom = _binomial_window(x, c, max(mc["bin_width"] * 3, mc["bandwidth"] / 5))
    metrics = {"method": "McCrary (2008) local linear density test; binomial count test in a narrow "
                         "symmetric window reported as a cross-check",
               "mccrary": mc, "binomial": binom, "cutoff": c, "n": int(len(x))}
    warnings = []
    n_unique_near = int(np.unique(x[np.abs(x - c) <= mc["bandwidth"]]).size)
    if n_unique_near < 20:
        warnings.append(f"The running variable takes only {n_unique_near} distinct values near the cutoff; "
                        "density tests are rough with heaped or discrete scores.")
    # sharp vs fuzzy: share treated each side (within the bandwidth used for covariates)
    if r.treatment and r.treatment in prep.df:
        h = _rd_bandwidth(x, c)
        dd = prep.df.loc[d.index, [r.running, r.treatment]].dropna()
        near = dd[(dd[r.running] - c).abs() <= h]
        sb = _f(near.loc[near[r.running] < c, r.treatment].mean())
        sa = _f(near.loc[near[r.running] >= c, r.treatment].mean())
        metrics.update({"share_treated_below": sb, "share_treated_above": sa, "share_window": h})
        if sb is not None and sa is not None:
            sharp = (sb <= 0.02 and sa >= 0.98) or (sb >= 0.98 and sa <= 0.02)
            metrics["rd_type"] = "sharp" if sharp else "fuzzy"
    p = mc.get("p")
    if p is None:
        p = binom.get("p")
        metrics["test_used"] = "binomial"
        warnings.append(f"The density test could not be computed ({mc.get('error')}); the binomial count test is used.")
    else:
        metrics["test_used"] = "mccrary"
    if p is None:
        status = Status.ASSUMED
    elif p < 0.05:
        status = Status.FAIL
    elif p < 0.1:
        status = Status.CONCERN
    else:
        status = Status.PASS
    ratio = (binom["above"] / binom["below"]) if binom["below"] else None
    detail = (f"Density of '{r.running}' just above vs just below {c:g}: "
              + (f"log jump {mc['log_density_jump']:+.2f}, p = {mc['p']:.3g} (McCrary test). " if mc.get("p") is not None else "")
              + f"Within {binom['window']:.3g} of the cutoff: {binom['below']} below, {binom['above']} above"
              + (f" (binomial p = {binom['p']:.3g})." if binom.get("p") is not None else ".")
              + " Bunching on one side suggests units could push themselves across the cutoff.")
    a = _A("rd_no_sorting", "No sorting around the cutoff", status, detail, True, src)
    summary = {Status.FAIL: "More units sit on one side of the cutoff than expected: signs of sorting (manipulation).",
               Status.CONCERN: "Weak signs of bunching at the cutoff.",
               Status.PASS: "No sign of bunching at the cutoff.",
               Status.ASSUMED: "Bunching at the cutoff could not be tested."}[status]
    if ratio is not None:
        metrics["count_ratio_above_below"] = ratio
    # histogram around the cutoff, bins aligned so the cutoff is an edge
    span = min(3 * mc["bandwidth"], max(c - x.min(), x.max() - c))
    nb = 20
    bw = span / nb
    edges = np.concatenate([c - bw * np.arange(nb, 0, -1), c + bw * np.arange(0, nb + 1)])
    counts, _ = np.histogram(x, bins=edges)
    below_counts = [int(v) if e < c else 0 for v, e in zip(counts, edges[:-1])]
    above_counts = [int(v) if e >= c else 0 for v, e in zip(counts, edges[:-1])]
    chart = Chart(kind="hist", title=f"'{r.running}' around the cutoff",
                  data={"bins": [float(e) for e in edges],
                        "series": [{"name": "below cutoff", "counts": below_counts},
                                   {"name": "at or above cutoff", "counts": above_counts}],
                        "vline": c},
                  x_label=r.running, y_label="Number of units",
                  note=f"The cutoff ({c:g}) is a bin edge. A spike on one side suggests sorting.")
    return Section(key=key, title=title, status=None if status == Status.ASSUMED else status, summary=summary,
                   metrics=metrics, assumptions=[a], charts=[chart], warnings=warnings)


def _covariate_jumps(prep: Prepared, key: str, title: str) -> Section:
    r = prep.roles
    src = "design.rd"
    covs = [c for c in prep.adjustment_set if c in prep.df and not c.endswith("__missing")][:MAX_COVARIATE_TESTS]
    d = _rd_frame(prep, covs)
    if d is None:
        return Section(key=key, title=title, error="The running variable or the cutoff is not set.")
    if not covs:
        a = _A("rd_covariates_smooth", "Covariates continuous at the cutoff", Status.ASSUMED,
               "There are no covariates to check, so this was not tested.", True, src)
        return Section(key=key, title=title, status=None, summary="No covariates to check.", assumptions=[a])
    c = float(r.cutoff)
    x_all = d[r.running].to_numpy(float)
    h = _rd_bandwidth(x_all, c)
    near = d[(d[r.running] - c).abs() <= h]
    xc = near[r.running].to_numpy(float) - c
    D = (xc >= 0).astype(float)
    w = np.clip(1 - np.abs(xc) / h, 0, None)
    X = np.column_stack([np.ones(len(xc)), D, xc, D * xc])
    results = {}
    for cov in covs:
        y = near[cov].to_numpy(float)
        if np.nanstd(y) == 0:
            continue
        b, V = _wls_hc1(y, X, w)
        se = np.sqrt(max(V[1, 1], 0))
        p = float(2 * stats.norm.sf(abs(b[1] / se))) if se > 0 else None
        sd = np.std(y)
        results[cov] = {"jump": float(b[1]), "se": float(se), "p": p, "jump_sd": float(b[1] / sd) if sd > 0 else None}
    tested = {k: v for k, v in results.items() if v["p"] is not None}
    if not tested:
        a = _A("rd_covariates_smooth", "Covariates continuous at the cutoff", Status.ASSUMED,
               "No covariate varied near the cutoff, so this was not tested.", True, src)
        return Section(key=key, title=title, status=None, summary="No covariate could be checked.", assumptions=[a])
    k = len(tested)
    worst_cov = min(tested, key=lambda cc: tested[cc]["p"])
    p_min = tested[worst_cov]["p"]
    p_bonf = min(1.0, p_min * k)
    if p_bonf < 0.01:
        status = Status.FAIL
    elif p_bonf < 0.05:
        status = Status.CONCERN
    else:
        status = Status.PASS
    metrics = {"bandwidth": h, "bandwidth_rule": "1.84 * sd * n^(-1/5), triangular kernel, local linear each side",
               "n_in_window": int(len(near)), "tests": results, "min_p": p_min, "bonferroni_p": p_bonf,
               "n_tests": k, "worst_covariate": worst_cov}
    detail = (f"Local linear jump at {c:g} in each of {k} covariate(s) within {h:.3g} of the cutoff. Largest jump: "
              f"'{worst_cov}' (p = {p_min:.3g}; Bonferroni-adjusted p = {p_bonf:.3g}). Units just either side of the "
              "cutoff should look alike.")
    a = _A("rd_covariates_smooth", "Covariates continuous at the cutoff", status, detail, True, src)
    summary = {Status.PASS: "Covariates change smoothly through the cutoff.",
               Status.CONCERN: f"'{worst_cov}' jumps at the cutoff (Bonferroni p = {p_bonf:.3g}).",
               Status.FAIL: f"'{worst_cov}' clearly jumps at the cutoff (Bonferroni p = {p_bonf:.3g}): units on each "
                            "side differ, not just in treatment."}[status]
    # binned means of the worst covariate
    span = min(3 * h, max(c - x_all.min(), x_all.max() - c))
    win = d[(d[r.running] - c).abs() <= span]
    nb = 20
    bw = span / nb
    edges = np.concatenate([c - bw * np.arange(nb, 0, -1), c + bw * np.arange(0, nb + 1)])
    idx = np.clip(np.searchsorted(edges, win[r.running].to_numpy(float), side="right") - 1, 0, len(edges) - 2)
    grp = win.groupby(idx)[worst_cov].mean()
    points = [{"x": float((edges[i] + edges[i + 1]) / 2), "y": float(v),
               "group": "at or above cutoff" if edges[i] >= c else "below cutoff"} for i, v in grp.items()]
    lines = []
    for side, m in (("below cutoff", xc < 0), ("at or above cutoff", xc >= 0)):
        if m.sum() >= 3:
            Xs = np.column_stack([np.ones(m.sum()), xc[m]])
            bs = np.linalg.lstsq(Xs * np.sqrt(w[m])[:, None], near[worst_cov].to_numpy(float)[m] * np.sqrt(w[m]), rcond=None)[0]
            xs = np.array([xc[m].min(), xc[m].max()])
            lines.append({"x": [float(v + c) for v in xs], "y": [float(bs[0] + bs[1] * v) for v in xs], "name": side})
    chart = Chart(kind="scatter", title=f"'{worst_cov}' around the cutoff (binned means)",
                  data={"points": points, "lines": lines, "vline": c},
                  x_label=r.running, y_label=f"mean of {worst_cov}",
                  note="Shown for the covariate with the largest jump. Lines are local linear fits within the bandwidth.")
    return Section(key=key, title=title, status=status, summary=summary, metrics=metrics,
                   assumptions=[a], charts=[chart])


# =================================================================================== grade
KEY_SECTIONS = {"balance", "overlap", "missing_outcomes"}
MINOR_CONCERN_KEYS = {"decl_cluster", "decl_panel", "iv_first_stage", "did_no_anticipation"}
NO_VARIATION_KEYS = {"decl_treatment", "decl_outcome", "decl_did_groups", "decl_instrument", "decl_cutoff",
                     "decl_cutoff_jump"}


def grade(design: Design, sections: list[Section]) -> tuple[str, list[str]]:
    """Grade the design A-F by its WORST problem, never by an average.

    F  a declaration conflict (any FAIL from source "declarations": e.g. the treatment does
       not vary, no comparison group, the cutoff does not assign treatment), or the
       estimate card ("estimate*") errored.
    D  any FAIL in a design test or a check whose card affects the grade (weak instrument,
       pre-trends, bunching, covariate jumps, balance, overlap, missing outcomes,
       interference ...).
    C  any non-minor CONCERN in a key card (balance, overlap, missing_outcomes, any
       design_* test), a key card that could not be run, or 2+ concerns of any kind; and the
       observational design is capped at C regardless (its key assumption, no hidden
       confounders, can never be tested).
    B  exactly one concern elsewhere or one minor concern (few clusters, borderline first
       stage F 10-20, anticipation signal, duplicate unit-periods), or an IV / RD / DiD with
       every test passing (their identifying assumption is untestable and the effect is
       local / for the treated).
    A  a randomized experiment with every tested assumption PASS (and a passing balance
       card; with no balance card at all the experiment gets B).

    ASSUMED never lowers or raises the grade by itself. The hidden-confounding card is never
    PASS by contract, so only its FAIL counts (for observational data, which is capped at
    C anyway). Cards with affects_grade=False (literature) are ignored. Errors in other
    graded cards count as concerns (an untested check is not a pass).
    """
    try:
        return _grade(design, sections)
    except Exception as e:  # noqa: BLE001
        return "F", [f"The grade could not be computed ({type(e).__name__}: {e}), so no grade better than F is given."]


def _grade(design: Design, sections: list[Section]) -> tuple[str, list[str]]:
    kind = getattr(design, "kind", "observational")
    fails: list[str] = []          # D-level
    key_concerns: list[str] = []   # C-level
    minor: list[str] = []          # B-level (one is fine, two -> C)
    other: list[str] = []          # B-level
    f_level: list[str] = []

    seen = set()

    def visit(a: Assumption, where: str, key_card: bool):
        ident = (a.key, a.source, a.status)
        if ident in seen:
            return
        seen.add(ident)
        label = f"{a.name}{(': ' + a.detail) if a.detail else ''}"
        if a.source == "declarations":
            if a.status == Status.FAIL:
                f_level.append(f"Declaration conflict: {label}")
            elif a.status == Status.CONCERN:
                (minor if a.key in MINOR_CONCERN_KEYS else other).append(label)
            return
        if a.status == Status.FAIL:
            fails.append(f"{where}: {label}")
        elif a.status == Status.CONCERN:
            if a.key in MINOR_CONCERN_KEYS:
                minor.append(f"{where}: {label}")
            elif key_card:
                key_concerns.append(f"{where}: {label}")
            else:
                other.append(f"{where}: {label}")

    for a in list(design.conflicts or []) + list(design.assumptions or []):
        visit(a, "Design", a.source.startswith("design") if a.source else False)

    for s in sections or []:
        if not s.affects_grade:
            continue
        title = s.title or s.key
        is_estimate = s.key.startswith("estimate")
        key_card = s.key.startswith("design_") or (s.key in KEY_SECTIONS and (s.key != "balance" or kind == "experiment"))
        if s.error:
            if is_estimate:
                f_level.append(f"The effect could not be estimated: {s.error}")
            elif key_card:
                key_concerns.append(f"{title} could not be run: {s.error}")
            else:
                other.append(f"{title} could not be run: {s.error}")
            continue
        if s.key == "hidden_confounding":
            if s.status == Status.FAIL:
                fails.append(f"{title}: {s.summary}".strip())
            continue
        n_before = len(fails) + len(key_concerns) + len(minor) + len(other)
        for a in s.assumptions:
            visit(a, title, key_card)
        n_after = len(fails) + len(key_concerns) + len(minor) + len(other)
        if n_after == n_before and s.status in (Status.FAIL, Status.CONCERN):
            # card-level status without a matching assumption
            msg = f"{title}: {s.summary}".rstrip(": ")
            if s.status == Status.FAIL:
                fails.append(msg)
            elif key_card:
                key_concerns.append(msg)
            else:
                other.append(msg)

    n_concerns = len(key_concerns) + len(minor) + len(other)
    reasons: list[str] = []
    if f_level:
        g = "F"
        reasons += f_level + fails + key_concerns + minor + other
    elif fails:
        g = "D"
        reasons += fails + key_concerns + minor + other
    elif key_concerns or n_concerns >= 2:
        g = "C"
        reasons += key_concerns + minor + other
        if not key_concerns:
            reasons.insert(0, f"{n_concerns} separate concerns add up.")
    elif n_concerns == 1:
        g = "B"
        reasons += minor + other
    elif kind in ("iv", "rd", "did"):
        g = "B"
        reasons.append({"iv": "Every testable check passed, but the instrument's exclusion restriction can never be "
                              "tested, and the effect applies to compliers only.",
                        "rd": "Every testable check passed, but continuity at the cutoff can never be tested, and the "
                              "effect applies only to units near the cutoff.",
                        "did": "Every testable check passed, but parallel trends after treatment can never be tested, "
                               "and the effect is for the treated units."}[kind])
    elif kind == "experiment" and not any(s.key == "balance" and s.status == Status.PASS and not s.error
                                          for s in sections or []):
        g = "B"
        reasons.append("Randomized experiment, but covariate balance was not tested, so randomization was not "
                       "checked against the data at all.")
    elif kind == "experiment":
        g = "A"
        reasons.append("Randomized experiment with every tested assumption passing. Randomization itself is taken "
                       "from your description.")
    else:
        g = "C"

    if kind not in ("experiment", "iv", "rd", "did") and g in ("A", "B"):
        g = "C"
    if kind not in ("experiment", "iv", "rd", "did"):
        reasons.append("Observational data: without a design that rules out hidden confounders, the grade is capped "
                       "at C however clean the checks look.")
    return g, reasons


# =================================================================================== DAG
def dag(prep: Prepared, design: Design, claim: Optional[Claim]) -> Chart:
    """The ASSUMED causal diagram. Every node role and edge is a hypothesis."""
    try:
        return _dag(prep, design)
    except Exception as e:  # noqa: BLE001
        return Chart(kind="dag", title="Assumed causal diagram", data={"nodes": [], "edges": []},
                     note=f"The diagram could not be drawn ({type(e).__name__}: {e}).")


G_TREATMENT = "treatment"
G_OUTCOME = "outcome"
G_CONFOUNDERS = "confounders (assumed)"
G_INSTRUMENT = "instrument (assumed)"
G_RUNNING = "running variable"
G_FIXED = "fixed effects (unit and time)"
G_UNMEASURED = "unmeasured (not in the data)"
G_POST = "measured after treatment"
G_COLLIDER = "possible colliders (not adjusted for)"
ROLE_GROUP = {"treatment": G_TREATMENT, "outcome": G_OUTCOME, "confounder (assumed)": G_CONFOUNDERS,
              "instrument (assumed)": G_INSTRUMENT, "running variable": G_RUNNING, "fixed effect": G_FIXED,
              "unmeasured confounder": G_UNMEASURED, "post-treatment (excluded)": G_POST,
              "possible collider (excluded)": G_COLLIDER}
GROUP_NOTES = {
    G_TREATMENT: "The cause being examined.",
    G_OUTCOME: "The result being examined.",
    G_CONFOUNDERS: "Columns adjusted for, on the assumption that they were measured before treatment and may "
                   "drive both treatment and outcome. That they are confounders is assumed, not shown.",
    G_INSTRUMENT: "Assumed to shift the treatment and to affect the outcome in no other way (untestable).",
    G_RUNNING: "The score that decides treatment at the cutoff.",
    G_FIXED: "Unit and time identifiers; differences between units and between periods are removed.",
    G_UNMEASURED: "Anything not in the data that could drive both treatment and outcome. It cannot be checked.",
    G_POST: "Columns you marked (or that were flagged) as measured after treatment. They are not adjusted for, "
            "because adjusting for a consequence of treatment biases the estimate.",
    G_COLLIDER: "Columns marked as measured after treatment that are also associated with both the treatment and "
                "the outcome in the data. They may be common effects of the two (colliders); this is a hypothesis "
                "from that pattern plus the marking, not a finding. Adjusting for one would open a non-causal "
                "path, so they are left out.",
}
COLLIDER_ALPHA = 0.01


def _assoc_p(a: pd.Series, b: pd.Series) -> Optional[float]:
    d = pd.concat([a, b], axis=1).dropna()
    if len(d) < 10 or d.iloc[:, 0].nunique() < 2 or d.iloc[:, 1].nunique() < 2:
        return None
    return _f(stats.pearsonr(d.iloc[:, 0].to_numpy(float), d.iloc[:, 1].to_numpy(float))[1])


def _collider_evidence(prep: Prepared, orig: str) -> dict:
    """Smallest correlation p-value of a post-treatment column (any one-hot child) with the
    treatment and with the outcome, on rows with both observed."""
    r = prep.roles
    out = {"p_treatment": None, "p_outcome": None, "possible_collider": False}
    if not r.treatment or not r.outcome:
        return out
    d = prep.complete()
    children = [c for c in prep.encoding.get(orig, [orig])
                if c in d and not c.endswith("__missing") and c not in (r.treatment, r.outcome)]
    for c in children:
        if not pd.api.types.is_numeric_dtype(d[c]):
            continue
        for role_col, k in ((r.treatment, "p_treatment"), (r.outcome, "p_outcome")):
            p = _assoc_p(d[c], d[role_col])
            if p is not None and (out[k] is None or p < out[k]):
                out[k] = p
    out["possible_collider"] = bool(out["p_treatment"] is not None and out["p_outcome"] is not None
                                    and out["p_treatment"] < COLLIDER_ALPHA and out["p_outcome"] < COLLIDER_ALPHA)
    return out


def _dag(prep: Prepared, design: Design) -> Chart:
    r = prep.roles
    kind = getattr(design, "kind", "observational")
    nodes: list[dict] = []
    edges: list[dict] = []
    ids = set()

    def node(i, role, **kw):
        if i and i not in ids:
            ids.add(i)
            nodes.append({"id": i, "role": role, "group": ROLE_GROUP.get(role, role), **kw})

    def edge(a, b, typ="directed", label="hypothesis"):
        if a in ids and b in ids:
            edges.append({"from": a, "to": b, "type": typ, "label": label, "confidence": None})

    T, Y = r.treatment, r.outcome
    node(T, "treatment")
    node(Y, "outcome")
    edge(T, Y, label="hypothesis (the claim)")

    inv = _orig_of(prep)
    origs: list[str] = []
    for c in prep.adjustment_set:
        o = inv.get(c, c[:-len("__missing")] if c.endswith("__missing") else c)
        if o not in origs:
            origs.append(o)
    shown = origs if len(origs) <= MAX_DAG_CONFOUNDERS else origs[:MAX_DAG_CONFOUNDERS - 1]
    for o in shown:
        node(o, "confounder (assumed)")
        if kind != "experiment":
            edge(o, T)
        edge(o, Y)
    if len(origs) > len(shown):
        grp = f"other covariates ({len(origs) - len(shown)})"
        node(grp, "confounder (assumed)", members=origs[len(shown):])
        if kind != "experiment":
            edge(grp, T)
        edge(grp, Y)

    missing_edges = []
    U = "U (unmeasured)"
    if kind == "iv" and r.instrument:
        node(r.instrument, "instrument (assumed)")
        edge(r.instrument, T)
        node(U, "unmeasured confounder", dashed=True)
        edge(U, T, label="hypothesis (allowed: the instrument handles it)")
        edge(U, Y, label="hypothesis (allowed: the instrument handles it)")
        missing_edges.append({"from": r.instrument, "to": Y,
                              "label": "assumed absent: exclusion restriction (an assumption, untestable)"})
    if kind == "rd" and r.running:
        node(r.running, "running variable")
        edge(r.running, T, label=f"hypothesis (treatment switches at {r.cutoff:g})" if r.cutoff is not None else "hypothesis")
        edge(r.running, Y)
    elif r.running:
        node(r.running, "running variable")
        edge(r.running, T)
    for fe in (r.unit, r.time):
        if fe and fe not in ids:
            node(fe, "fixed effect")
            edge(fe, T)
            edge(fe, Y)
    colliders = []
    for p in r.post_treatment:
        if p in ids:
            continue
        try:
            ev = _collider_evidence(prep, p)
        except Exception:  # noqa: BLE001 -- the diagram is still drawn
            ev = {"p_treatment": None, "p_outcome": None, "possible_collider": False}
        if ev["possible_collider"]:
            colliders.append(p)
            node(p, "possible collider (excluded)", p_treatment=ev["p_treatment"], p_outcome=ev["p_outcome"])
            edge(T, p)
            edge(Y, p)
        else:
            node(p, "post-treatment (excluded)", p_treatment=ev["p_treatment"], p_outcome=ev["p_outcome"])
            edge(T, p, label="hypothesis (measured after treatment; not adjusted for)")
    if kind == "observational":
        node(U, "unmeasured confounder", dashed=True)
        edges.append({"from": T, "to": Y, "type": "bidirected", "label": U, "confidence": None})
        missing_edges.append({"from": U, "to": f"{T}, {Y}",
                              "label": "assumed absent after adjustment: no hidden confounders (untestable)"})
    if kind == "experiment":
        missing_edges.append({"from": "covariates", "to": T, "label": "absent because treatment was randomized (as you said)"})

    groups = []
    for n in nodes:
        g = next((x for x in groups if x["name"] == n["group"]), None)
        if g is None:
            g = {"name": n["group"], "members": [], "note": GROUP_NOTES.get(n["group"], "")}
            groups.append(g)
        g["members"].append(n["id"])

    note = ("Every role, group and arrow here is a hypothesis taken from the declared design and roles, not "
            "something the data showed.")
    if kind == "iv":
        note += (f" There is deliberately no arrow from '{r.instrument}' to '{Y}': that is the exclusion "
                 "restriction, an untestable assumption.")
    if kind == "observational":
        note += " The dashed U stands for anything unmeasured that drives both treatment and outcome; it is assumed away."
    if colliders:
        note += (" Possible colliders (" + ", ".join(f"'{c}'" for c in colliders) + ") are a hypothesis from the "
                 "data pattern (associated with both treatment and outcome) plus your marking them as measured "
                 "after treatment; the same pattern also fits a column that treatment changes and that in turn "
                 "affects the outcome. Adjusting for a collider would open a non-causal path between treatment "
                 "and outcome, which is why they are excluded.")
    note += (" A collider measured before treatment cannot be detected from data, so one may be sitting among "
             "the columns adjusted for.")
    return Chart(kind="dag", title="Assumed causal diagram",
                 data={"nodes": nodes, "edges": edges, "missing_edges": missing_edges, "groups": groups}, note=note)
