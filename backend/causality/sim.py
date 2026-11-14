"""Simulated datasets where the truth is known. Each returns (DataFrame, truth dict).

truth keys: effect (true effect in outcome units, for the estimand named in `target`),
target, design, plus anything design-specific. Used by every test and the acceptance run.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rct(n=2000, effect=1.5, seed=0, attrition=0.0, differential=0.0, binary_outcome=False):
    """Randomized experiment. Outcome missing for `attrition` share overall, with
    treated units `differential` more likely to drop (outcome-related if > 0)."""
    rng = np.random.default_rng(seed)
    age = rng.normal(40, 10, n)
    female = rng.binomial(1, 0.5, n)
    region = rng.choice(["north", "south", "east"], n)
    income = rng.lognormal(10, 0.5, n)
    treated = rng.binomial(1, 0.5, n)
    y = 10 + 0.05 * age + 1.0 * female + 0.00002 * income + effect * treated + rng.normal(0, 3, n)
    if binary_outcome:
        y = (y > np.median(y)).astype(float)
    p_drop = np.clip(attrition + differential * treated * (y > np.median(y)), 0, 0.95)
    y = np.where(rng.random(n) < p_drop, np.nan, y)
    df = pd.DataFrame({"treated": treated, "age": age, "female": female, "region": region,
                       "income": income, "score": y})
    return df, {"effect": effect, "target": "ATE", "design": "experiment", "treatment": "treated", "outcome": "score",
                "control_mean": 10 + 0.05 * 40 + 0.5 + 0.00002 * np.exp(10.125)}


def rct_with_post_treatment(n=2000, effect=1.0, seed=1):
    """RCT where `engagement` is caused by treatment (a post-treatment column)."""
    df, truth = rct(n=n, effect=effect, seed=seed)
    rng = np.random.default_rng(seed + 100)
    df["engagement"] = 2.0 * df["treated"] + rng.normal(0, 0.5, n)
    truth["post_treatment"] = ["engagement"]
    return df, truth


def observational(n=3000, effect=2.0, seed=2, confounding=1.0):
    """Treatment depends on measured x1, x2. No hidden confounding (truth recoverable)."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    x3 = rng.binomial(1, 0.4, n)
    p = 1 / (1 + np.exp(-(confounding * x1 - 0.5 * x2 + 0.3 * x3)))
    t = rng.binomial(1, p)
    y = 5 + effect * t + 2.0 * x1 + 1.0 * x2 + 0.5 * x3 + rng.normal(0, 1.5, n)
    df = pd.DataFrame({"program": t, "x1": x1, "x2": x2, "member": x3, "earnings": y})
    return df, {"effect": effect, "target": "ATE", "design": "observational", "treatment": "program", "outcome": "earnings"}


def observational_hidden(n=3000, effect=0.0, seed=3):
    """Hidden confounder u drives both treatment and outcome; the true effect is 0."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0, 1, n)
    x1 = rng.normal(0, 1, n)
    t = rng.binomial(1, 1 / (1 + np.exp(-(1.5 * u + 0.5 * x1))))
    y = 3 + effect * t + 2.0 * u + 0.5 * x1 + rng.normal(0, 1, n)
    df = pd.DataFrame({"coffee": t, "x1": x1, "health": y})
    return df, {"effect": effect, "target": "ATE", "design": "observational", "treatment": "coffee", "outcome": "health",
                "hidden_confounding": True}


def no_overlap(n=2000, seed=4):
    """Treatment almost determined by x: poor overlap in the tails."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    t = rng.binomial(1, 1 / (1 + np.exp(-6 * x)))
    y = 1 + 1.0 * t + 2 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"t": t, "x": x, "y": y}), {"effect": 1.0, "target": "ATE", "design": "observational",
                                                   "treatment": "t", "outcome": "y"}


def did_staggered(units=200, periods=8, effect=2.0, seed=5, pretrend=0.0):
    """Staggered adoption panel. Cohorts adopt at period 3, 5 or never. Effect constant
    (so TWFE is fine) unless `pretrend` != 0 adds a differential pre-trend to adopters."""
    rng = np.random.default_rng(seed)
    cohort = rng.choice([3, 5, 0], units, p=[0.3, 0.3, 0.4])
    fe = rng.normal(0, 2, units)
    rows = []
    for i in range(units):
        for t in range(periods):
            d = int(cohort[i] > 0 and t >= cohort[i])
            trend = pretrend * t * (cohort[i] > 0)
            y = 10 + fe[i] + 0.5 * t + trend + effect * d + rng.normal(0, 1)
            rows.append((f"unit{i:03d}", 2010 + t, d, y, rng.normal(0, 1)))
    df = pd.DataFrame(rows, columns=["state", "year", "policy", "sales", "noise_cov"])
    return df, {"effect": effect, "target": "ATT", "design": "did", "treatment": "policy", "outcome": "sales",
                "unit": "state", "time": "year"}


def iv(n=4000, effect=1.0, seed=1, strength=0.5):
    # (seed 6 draws a sample ~2.5 SE from the truth: ordinary noise, but a poor default demo)
    """Binary instrument `lottery` shifts take-up; hidden u confounds take-up and outcome."""
    rng = np.random.default_rng(seed)
    z = rng.binomial(1, 0.5, n)
    u = rng.normal(0, 1, n)
    x = rng.normal(0, 1, n)
    d = (strength * z + 0.4 * u + 0.2 * x + rng.normal(0, 0.5, n) > 0.25).astype(int)
    y = 2 + effect * d + 1.5 * u + 0.5 * x + rng.normal(0, 1, n)
    df = pd.DataFrame({"lottery": z, "enrolled": d, "x": x, "test_score": y})
    return df, {"effect": effect, "target": "LATE", "design": "iv", "treatment": "enrolled", "outcome": "test_score",
                "instrument": "lottery"}


def rd(n=4000, effect=3.0, seed=7, cutoff=50.0, manipulate=False):
    """Sharp cutoff: treated if exam score >= cutoff. Optional bunching above the cutoff."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(0, 100, n)
    if manipulate:
        idx = (s > cutoff - 5) & (s < cutoff) & (rng.random(n) < 0.6)
        s[idx] = cutoff + rng.uniform(0, 2, idx.sum())
    t = (s >= cutoff).astype(int)
    y = 20 + 0.3 * s + effect * t + rng.normal(0, 2, n)
    df = pd.DataFrame({"exam_score": s, "scholarship": t, "gpa_later": y, "female": rng.binomial(1, 0.5, n)})
    return df, {"effect": effect, "target": "local_at_cutoff", "design": "rd", "treatment": "scholarship",
                "outcome": "gpa_later", "running": "exam_score", "cutoff": cutoff}


def dose(n=3000, slope=0.5, seed=8):
    """Continuous treatment with measured confounder; linear dose-response."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    hours = np.clip(5 + 2 * x + rng.normal(0, 2, n), 0, None)
    y = 1 + slope * hours + 1.0 * x + rng.normal(0, 1, n)
    return pd.DataFrame({"study_hours": hours, "x": x, "grade": y}), {
        "effect": slope, "target": "dose_response", "design": "observational", "treatment": "study_hours", "outcome": "grade"}


def cluster_saturation(clusters=60, per=40, direct=1.0, spill=0.8, seed=9):
    """Two-stage randomized saturation design: each cluster's treated share is randomized
    from {0, 0.3, 0.7}; untreated units in high-saturation clusters get a spillover."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(clusters):
        sat = rng.choice([0.0, 0.3, 0.7])
        ce = rng.normal(0, 0.5)
        for _ in range(per):
            t = int(rng.random() < sat)
            y = 5 + ce + direct * t + spill * sat * (1 - t) + rng.normal(0, 1)
            rows.append((f"v{c}", sat, t, y, rng.normal(0, 1)))
    df = pd.DataFrame(rows, columns=["village", "saturation", "treated", "consumption", "hh_size"])
    return df, {"effect": direct, "spillover": spill, "target": "ATE", "design": "experiment", "treatment": "treated",
                "outcome": "consumption", "cluster": "village", "saturation": "saturation"}


def chain(n=3000, seed=10):
    """Explore mode truth: A -> B -> C, D -> C, E independent noise (non-Gaussian errors)."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(-2, 2, n)
    b = 1.2 * a + rng.uniform(-1, 1, n)
    d = rng.uniform(-2, 2, n)
    c = 0.8 * b - 1.0 * d + rng.uniform(-1, 1, n)
    e = rng.uniform(-2, 2, n)
    return pd.DataFrame({"A": a, "B": b, "C": c, "D": d, "E": e}), {
        "edges": [("A", "B"), ("B", "C"), ("D", "C")], "independent": ["E"], "design": "explore"}


def event_study(firms=120, days=80, effect=0.02, seed=12):
    """Finance-style event study: each firm has ONE event day (a pulse, not a lasting
    treatment) somewhere in the last 20 days; its return jumps by `effect` on that day only.
    Returns follow a market model with firm-specific beta and volatility."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0005, 0.01, days)
    rows = []
    for i in range(firms):
        beta, vol = rng.uniform(0.6, 1.4), rng.uniform(0.01, 0.03)
        ev = int(rng.integers(days - 20, days - 3))
        for d in range(days):
            is_ev = int(d == ev)
            r = 0.0002 + beta * market[d] + effect * is_ev + rng.normal(0, vol)
            rows.append((f"firm{i:03d}", d, is_ev, r, market[d]))
    df = pd.DataFrame(rows, columns=["firm", "day", "announcement", "ret", "market_ret"])
    return df, {"effect": effect, "target": "event_day_abnormal_return", "design": "did", "treatment": "announcement",
                "outcome": "ret", "unit": "firm", "time": "day"}


def null_rct(n=4000, seed=11):
    """RCT with exactly zero effect (for 'does not affect' claims / equivalence)."""
    return rct(n=n, effect=0.0, seed=seed)


ALL = {
    "rct": rct, "rct_post": rct_with_post_treatment, "observational": observational,
    "observational_hidden": observational_hidden, "no_overlap": no_overlap, "did": did_staggered,
    "iv": iv, "rd": rd, "dose": dose, "cluster_saturation": cluster_saturation, "chain": chain,
    "null_rct": null_rct, "event_study": event_study,
}
