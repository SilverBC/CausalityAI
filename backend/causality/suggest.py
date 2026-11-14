"""What study would settle the question, and how big it must be.

Power is ALWAYS based on the CLAIMED effect (units.claimed_effect), never on the observed
estimate: powering on the observed effect is circular and overstates what a new study can do.

Two-arm trial, two-sided alpha 0.05, 80% power:
    n per arm = 2 * (z_{1-alpha/2} + z_{power})^2 * sd^2 / delta^2
then x design effect 1 + (m - 1) * ICC when a cluster role exists, then / (1 - attrition),
then rounded up once at the end.

No-effect claims: equivalence test (TOST, two one-sided 5% tests), true effect 0, margin
0.1 SD:  n per arm = 2 * (z_{1-alpha} + z_{1-beta/2})^2 * sd^2 / margin^2.
Negated claims ("does not reduce"): one-sided non-inferiority test, true effect 0, margin =
|claimed size| or 0.1 SD:  n per arm = 2 * (z_{1-alpha} + z_{power})^2 * sd^2 / margin^2.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .schema import Chart, Claim, Design, Estimate, Prepared, Section
from .units import claimed_effect

ALPHA = 0.05
POWER = 0.80
RD_FACTOR = 3.0   # rough: a local-linear RD needs ~3x the sample of a trial for the same precision


def n_per_arm(delta: float, sd: float, alpha: float = ALPHA, power: float = POWER, kind: str = "superiority") -> float:
    """Unrounded n per arm for a two-arm comparison with equal arms."""
    if kind == "equivalence":
        z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(1 - (1 - power) / 2)
    elif kind == "noninferiority":
        z = stats.norm.ppf(1 - alpha) + stats.norm.ppf(power)
    else:
        z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return 2.0 * z ** 2 * sd ** 2 / delta ** 2


def power_at(n_total: float, delta: float, sd: float, inflate: float, alpha: float = ALPHA,
             kind: str = "superiority") -> float:
    """Power for a total enrolled n, given the inflation factor (design effect / retention)."""
    n_arm = max(n_total / 2.0 / inflate, 1e-9)
    ncp = abs(delta) / (sd * math.sqrt(2.0 / n_arm))
    if kind == "equivalence":
        return float(max(0.0, 2 * stats.norm.cdf(ncp - stats.norm.ppf(1 - alpha)) - 1))
    if kind == "noninferiority":
        return float(stats.norm.cdf(ncp - stats.norm.ppf(1 - alpha)))
    return float(stats.norm.cdf(ncp - stats.norm.ppf(1 - alpha / 2)))


def icc_oneway(y: pd.Series, g: pd.Series) -> Optional[float]:
    """One-way ANOVA intraclass correlation, clipped to [0, 1]."""
    d = pd.DataFrame({"y": y, "g": g}).dropna()
    if d.empty or d["g"].nunique() < 2:
        return None
    k = d["g"].nunique()
    n = len(d)
    sizes = d.groupby("g").size()
    means = d.groupby("g")["y"].mean()
    grand = d["y"].mean()
    ssb = float((sizes * (means - grand) ** 2).sum())
    ssw = float(((d["y"] - d["g"].map(means)) ** 2).sum())
    if n - k <= 0:
        return None
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - (sizes ** 2).sum() / n) / (k - 1)
    den = msb + (m0 - 1) * msw
    if den <= 0:
        return None
    return float(min(1.0, max(0.0, (msb - msw) / den)))


def _design_choice(prep: Prepared, design: Optional[Design], est: Optional[Estimate]) -> tuple[str, str, float]:
    """(name, plain explanation, extra sample factor relative to a simple trial)."""
    kind = design.kind if design else "observational"
    cont = not prep.treatment_is_binary() if prep is not None and prep.T is not None else False
    if kind == "iv":
        comp = None
        extra = (est.extra or {}) if est else {}
        for k in ("compliance", "first_stage_coef", "first_stage"):
            v = extra.get(k)
            if isinstance(v, (int, float)) and 0 < abs(v) <= 1:
                comp = abs(float(v))
                break
        if comp:
            f = 1.0 / comp ** 2
            return ("Larger encouragement design",
                    f"Randomize the encouragement (the instrument) to more people. Only about {comp:.0%} of people "
                    f"change their treatment because of the encouragement, so the study needs about {f:.1f} times "
                    "the size of a trial that assigns treatment directly.", f)
        return ("Larger encouragement design",
                "Randomize the encouragement (the instrument) to more people. Because not everyone follows the "
                "encouragement, the study needs more people than the trial size below (divide by the share who "
                "follow it, squared).", 1.0)
    if kind == "rd":
        return ("More data near the cutoff (or a randomized trial near it)",
                "Collect many more units just above and just below the cutoff; if possible, randomize treatment "
                f"among units close to the cutoff. A cutoff comparison needs roughly {RD_FACTOR:g} times the size "
                "of the trial below for the same precision.", RD_FACTOR)
    if kind == "experiment":
        return ("Pre-registered randomized experiment (replication)",
                "Repeat the randomized experiment, stating the claimed effect and the analysis in advance.", 1.0)
    extra = (" With a continuous treatment, compare two groups whose treatment differs by one unit "
             "(the claimed effect is per unit).") if cont else ""
    return ("Randomized experiment",
            "Randomly assign the treatment, so nothing else can differ systematically between the groups. "
            "If it cannot be randomized, look for a natural experiment (a lottery, a rule with a cutoff, or a "
            "staggered rollout)." + extra, 1.0)


def _cluster_info(prep: Prepared) -> tuple[float, Optional[float], Optional[float]]:
    """(design effect, mean cluster size, ICC)."""
    c = prep.roles.cluster if prep is not None else None
    y = prep.Y if prep is not None else None
    if not c or y is None or c not in prep.df:
        return 1.0, None, None
    g = prep.df[c]
    ok = y.notna() & g.notna()
    if ok.sum() < 4:
        return 1.0, None, None
    m = float(ok.sum() / g[ok].nunique())
    icc = icc_oneway(y[ok], g[ok])
    if icc is None:
        return 1.0, m, None
    return 1.0 + (m - 1.0) * icc, m, icc


def _attrition(prep: Prepared) -> float:
    if prep is None or prep.Y is None:
        return 0.0
    t = prep.T
    y = prep.Y
    rows = t.notna() if t is not None else pd.Series(True, index=y.index)
    if rows.sum() == 0:
        return 0.0
    return float(min(0.9, y[rows].isna().mean()))


def suggest_study(prep: Prepared, design: Optional[Design], claim: Optional[Claim],
                  estimate: Optional[Estimate]) -> Section:
    try:
        return _suggest(prep, design, claim, estimate)
    except Exception as e:  # never raise on bad data
        return Section(key="next_study", title="What study would settle this", status=None,
                       error=f"The study size could not be worked out: {e}")


def _suggest(prep, design, claim, est) -> Section:
    sec = Section(key="next_study", title="What study would settle this", status=None)
    name, how, factor = _design_choice(prep, design, est)

    sd = est.outcome_sd if est is not None and est.outcome_sd else None
    if not sd and prep is not None and prep.Y is not None:
        v = float(prep.Y.std())
        sd = v if math.isfinite(v) and v > 0 else None

    deff, m, icc = _cluster_info(prep)
    att = _attrition(prep)
    inflate = deff / (1.0 - att)

    direction = claim.direction if claim else None
    negated = bool(claim and claim.negated and direction in ("increase", "decrease"))
    if claim is not None and (direction == "no_effect" or (claim.negated and not negated)):
        kind = "equivalence"
    elif negated:
        kind = "noninferiority"
    else:
        kind = "superiority"

    c = claimed_effect(claim, est) if claim is not None else None
    sd_txt = f"{sd:.3g}" if sd else "unknown"
    assumptions = {"alpha": ALPHA, "power": POWER, "outcome_sd": sd, "design_effect": round(deff, 3),
                   "mean_cluster_size": m, "icc": icc, "attrition_rate": round(att, 3),
                   "test": kind, "extra_factor_for_design": factor,
                   "note": "Powered on the CLAIMED effect, never the observed one."}

    # the effect / margin to power for --------------------------------------------
    delta: Optional[float] = None
    delta_txt = ""
    if kind == "equivalence":
        if sd:
            delta = 0.1 * sd
            delta_txt = f"an equivalence margin of 0.1 SD ({delta:.3g}), assuming the true effect is zero"
    elif kind == "noninferiority":
        if c not in (None, 0):
            delta = abs(c)
            delta_txt = f"a margin equal to the claimed size ({delta:.3g})"
        elif sd:
            delta = 0.1 * sd
            delta_txt = f"a margin of 0.1 SD ({delta:.3g}), assuming the true effect is zero"
    elif c not in (None, 0):
        delta = abs(c)
        delta_txt = f"the claimed effect of {c:.3g} outcome units"

    if not sd:
        sec.error = "The outcome's spread (standard deviation) is unknown, so the study size cannot be worked out."
        sec.summary = f"Suggested design: {name}. {how}"
        sec.metrics = {"design": name, "n_per_arm": None, "n_total": None, "assumptions": assumptions}
        return sec

    def finish(npa_raw: float) -> tuple[int, int]:
        per = math.ceil(npa_raw * inflate * factor - 1e-9)
        return per, 2 * per

    adj_txt = []
    if deff > 1.0:
        adj_txt.append(f"x{deff:.2f} for clustering (about {m:.0f} per cluster, ICC {icc:.2f})")
    if att > 0:
        adj_txt.append(f"/{1 - att:.2f} for the {att:.0%} of outcomes that went missing here")
    if factor != 1.0:
        adj_txt.append(f"x{factor:.1f} for the {name.lower()}")
    adj = (" (" + ", ".join(adj_txt) + ")") if adj_txt else ""

    if delta is None:
        # no size claimed (or not convertible): show a range of effect sizes
        why = ("The claim states no size" if not (claim and claim.size is not None)
               else "The claimed size could not be turned into outcome units")
        table = []
        series = []
        for es in (0.1, 0.2, 0.3):
            d = es * sd
            per, tot = finish(n_per_arm(d, sd))
            table.append({"effect_sd": es, "effect_units": d, "n_per_arm": per, "n_total": tot})
            xs = np.unique(np.linspace(max(4, tot * 0.1), tot * 2.0, 40).round())
            series.append({"name": f"{es:g} SD effect", "x": xs.tolist(),
                           "y": [power_at(x / factor, d, sd, inflate) for x in xs], "lo": None, "hi": None})
        sec.summary = (f"Suggested design: {name}. {how} {why}, so the study size depends on how big an effect "
                       f"matters: about {table[0]['n_total']:,} people in total for a 0.1 SD effect, "
                       f"{table[1]['n_total']:,} for 0.2 SD, {table[2]['n_total']:,} for 0.3 SD{adj}. "
                       "State the size of the effect you expect to settle this.")
        sec.metrics = {"design": name, "n_per_arm": None, "n_total": None, "n_by_effect_size": table,
                       "assumptions": assumptions}
        sec.warnings.append("A size must be stated in the claim to say how big a study would settle it.")
        sec.charts.append(Chart(kind="line", title="Chance of detecting the effect vs study size",
                                data={"series": series, "vline": None, "hline": POWER},
                                x_label="total people enrolled", y_label="power"))
        return sec

    npa_raw = n_per_arm(delta, sd, kind=kind)
    per, tot = finish(npa_raw)
    xs = np.unique(np.linspace(max(4, tot * 0.1), tot * 2.0, 60).round())
    ys = [power_at(x / factor, delta, sd, inflate, kind=kind) for x in xs]
    test_txt = {"equivalence": "show the effect is smaller than the margin (equivalence test)",
                "noninferiority": "rule out an effect beyond the margin (one-sided test)",
                "superiority": "detect it (two-sided 5% test)"}[kind]
    observed = ""
    if est is not None and est.n:
        observed = f" Your data had {est.n:,} usable rows."
    sec.summary = (f"Suggested design: {name}. {how} To {test_txt} with 80% power for {delta_txt}, "
                   f"with outcome SD {sd_txt}, it needs about {per:,} per group, {tot:,} in total{adj}.{observed} "
                   "This uses the claimed effect, not the effect seen in this data.")
    sec.metrics = {"design": name, "n_per_arm": per, "n_total": tot, "n_per_arm_unadjusted": npa_raw,
                   "effect_powered_for": delta, "assumptions": assumptions}
    sec.charts.append(Chart(kind="line", title="Chance of a clear answer vs study size",
                            data={"series": [{"name": "claimed effect", "x": xs.tolist(), "y": ys,
                                              "lo": None, "hi": None}],
                                  "vline": float(tot), "hline": POWER},
                            x_label="total people enrolled", y_label="power"))
    return sec
