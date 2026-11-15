"""Agent-facing layer: a facts card, precomputed check lines, and tools that take the design
and the column roles as INPUTS. Nothing here guesses a design or asks a question.

  facts_card(df)                         facts about the table, no conclusions
  general_checks(df, T, Y, covariates)   "Check #N: (method) - (result) - (applies when)" lines
  structure_facts(df, T, Y, ...)         hard-to-find facts about how treatment is laid out
  design_tests(df, design, roles, cov)   the declared design's own tests + conflicts with the data
  estimate_effect(df, design, roles, cov)

Every line reports what was computed and the numbers. There are no pass/fail labels: the model that
reads the lines judges what they mean.
"""
from __future__ import annotations

import itertools
import re
import warnings
from typing import Optional

import numpy as np
import pandas as pd

from . import data
from .schema import Roles, Section

warnings.filterwarnings("ignore")

ROLE_KEYS = ("treatment", "outcome", "unit", "time", "instrument", "running", "cluster", "strata")
TIME_WORDS = {"year", "yr", "date", "month", "quarter", "week", "day", "wave", "period", "time"}


def load(path: str) -> pd.DataFrame:
    with open(path, "rb") as f:
        return data.load_table(f.read(), path)


def _short(text: str, n: int = 380) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= n:
        return text
    cut = text[:n]
    stop = max(cut.rfind(". "), cut.rfind("; "))
    return (cut[:stop + 1] if stop > n * 0.5 else cut.rstrip() + " ...")


def _fmt(x) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "?"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    a = abs(float(x))
    if a == 0:
        return "0"
    return f"{x:.4g}" if (a < 1e5 and a >= 1e-3) else f"{x:.3e}"


# ------------------------------------------------------------------------------- facts card
def _keys(df: pd.DataFrame, max_cols: int = 14) -> list[str]:
    """Columns, or pairs of columns, whose values identify a row uniquely."""
    n = len(df)
    out = []
    cand = []
    for c in df.columns:
        k = df[c].nunique(dropna=False)
        if k == n:
            out.append(str(c))
        elif 1 < k < n and (not pd.api.types.is_float_dtype(df[c]) or np.allclose(df[c].dropna() % 1, 0)):
            cand.append(c)
    cand = sorted(cand, key=lambda c: -df[c].nunique())[:max_cols]
    pairs = []
    for a, b in itertools.combinations(cand, 2):
        if df[a].nunique() * df[b].nunique() < n:
            continue
        if not df.duplicated(subset=[a, b]).any():
            pairs.append(f"{a} x {b}")
    return out[:6] + pairs[:6]


def facts_card(df: pd.DataFrame, max_columns: int = 60) -> str:
    n, k = df.shape
    L = [f"Rows: {n}. Columns: {k}. Exact duplicate rows: {int(df.duplicated().sum())}."]
    keys = _keys(df)
    L.append("Row identity (values that are unique per row): " + ("; ".join(keys) if keys else
             "no single column or pair of columns identifies a row") + ".")
    L.append("")
    L.append("Columns (name | type | distinct | missing | range or most common values):")
    const, timeish = [], []
    for c in list(df.columns)[:max_columns]:
        s = df[c]
        kind = data.column_kind(s)
        nun, miss = int(s.nunique()), float(s.isna().mean())
        if kind == "constant":
            const.append(str(c))
        if pd.api.types.is_numeric_dtype(s) and kind not in ("binary", "constant"):
            v = s.dropna()
            rng = f"{_fmt(v.min())} to {_fmt(v.max())}, median {_fmt(v.median())}" if len(v) else "all missing"
            if (set(data.words(c)) & TIME_WORDS) or (len(v) and v.min() >= 1800 and v.max() <= 2100 and nun <= 300
                                                       and np.allclose(v % 1, 0)):
                timeish.append((str(c), nun, v.min() if len(v) else None, v.max() if len(v) else None))
        else:
            top = s.astype(str).where(s.notna()).value_counts().head(4)
            rng = ", ".join(f"{str(i)[:24]} ({j})" for i, j in top.items())
        L.append(f"- {c} | {kind} | {nun} | {miss:.0%} | {rng}")
    if k > max_columns:
        L.append(f"- ... and {k - max_columns} more columns not listed.")
    if timeish:
        L.append("")
        for c, nun, lo, hi in timeish[:4]:
            per = df.groupby(c).size()
            L.append(f"Time-like column '{c}': {nun} distinct values from {_fmt(lo)} to {_fmt(hi)}; rows per value "
                     f"range from {int(per.min())} to {int(per.max())}.")
    odd = []
    if const:
        odd.append("constant columns: " + ", ".join(const[:8]))
    num = df.select_dtypes("number")
    if 1 < num.shape[1] <= 80 and n >= 10:
        c = num.corr().abs()
        dup = [(a, b) for a, b in itertools.combinations(c.columns, 2) if c.loc[a, b] > 0.999]
        if dup:
            odd.append("near-identical numeric columns: " + ", ".join(f"{a} ~ {b}" for a, b in dup[:6]))
    if odd:
        L.append("")
        L.append("Oddities: " + "; ".join(odd) + ".")
    L.append("")
    L.append("Sample rows (first 5, then 5 at random):")
    show = df.iloc[:, :40]
    samp = pd.concat([show.head(5), show.sample(min(5, n), random_state=0)])
    L.append(samp.to_csv(index=False, float_format="%.5g").strip()[:3500])
    return "\n".join(L)


# ------------------------------------------------------------------------------- preparation
def _restrict(df: pd.DataFrame, roles: dict, covariates: Optional[list[str]]) -> pd.DataFrame:
    """Keep the role columns plus the chosen covariates. None = every other column."""
    if covariates is None:
        return df
    keep = [c for c in df.columns if c in set(covariates) or c in {v for v in roles.values() if isinstance(v, str)}]
    return df[keep]


def _prepare(df: pd.DataFrame, design: str, roles: dict, covariates: Optional[list[str]]):
    r = {k: roles.get(k) for k in ROLE_KEYS if roles.get(k)}
    for k, v in r.items():
        if v not in df.columns:
            raise data.InputError(f"The {k} column '{v}' is not in the table.")
    cut = roles.get("cutoff")
    R = Roles(**r, cutoff=float(cut) if cut not in (None, "") else None, randomized=(design == "experiment"))
    prep = data.prepare(_restrict(df, r, covariates), R)
    des = data.resolve_design(prep, design)
    return prep, des


def _val(v, depth: int = 0) -> Optional[str]:
    """One stored number, list or small table as text. None = nothing worth printing."""
    if v is None or isinstance(v, bool):
        return None if v is None else ("yes" if v else "no")
    if isinstance(v, (int, float, np.integer, np.floating)):
        return _fmt(float(v) if isinstance(v, (float, np.floating)) else int(v))
    if isinstance(v, str):
        return _short(v, 200)
    if isinstance(v, dict):
        if depth >= 2:
            return None
        parts = [(k, _val(x, depth + 1)) for k, x in list(v.items())[:12]]
        body = ", ".join(f"{k} {t}" for k, t in parts if t is not None)
        return ("{" + body + (", ..." if len(v) > 12 else "") + "}") if body else None
    if isinstance(v, (list, tuple)):
        if not v:
            return None
        items = [_val(x, depth + 1) for x in list(v)[:12 if depth == 0 else 6]]
        items = [t for t in items if t is not None]
        return ("[" + ", ".join(items) + (", ..." if len(v) > len(items) else "") + "]") if items else None
    return None


# Stored values that are a judgement or a display setting rather than a measurement.
_SKIP = {"fragile", "boundary_used", "test_used", "suspected_post_treatment", "post_treatment_shown"}


def _line(sec: Section) -> str:
    """The numbers a check computed, with no pass/fail label: the reader judges them."""
    if sec.error:
        return "could not run: " + _short(sec.error)
    parts = []
    for k, v in (sec.metrics or {}).items():
        if k in _SKIP:
            continue
        t = _val(v)
        if t is not None:
            parts.append(f"{k} = {t}")
    return "; ".join(parts) if parts else "no numbers were produced"


# ------------------------------------------------------------------------------- general checks
GENERAL = [
    ("Balance", "joint permutation test that the covariates predict treatment, plus standardized differences",
     "a direct treated-versus-untreated comparison (experiment, observational). Under an instrument, a cutoff or "
     "difference-in-differences the relevant balance is a different one: run the design tests."),
    ("Overlap", "average of two cross-fitted propensity models (logistic, boosting), common support, trimming",
     "a direct treated-versus-untreated comparison with a 0/1 treatment. Not meaningful for panel designs, a "
     "cutoff, or a one-off event."),
    ("Missing outcomes", "overall and differential dropout against the What Works Clearinghouse limits, with bounds",
     "any design."),
    ("Hidden-confounding sensitivity", "robustness value and E-value from a covariate-adjusted regression",
     "a comparison that relies on having measured the confounders. It is a sensitivity number, never a pass."),
]


def general_checks(df: pd.DataFrame, treatment: str, outcome: str, covariates: Optional[list[str]] = None) -> list[str]:
    from .checks import attrition, balance, overlap, sensitivity
    prep, des = _prepare(df, "observational", {"treatment": treatment, "outcome": outcome}, covariates)
    runs = [lambda: balance.run(prep, des, None), lambda: overlap.run(prep, des, None),
            lambda: attrition.run(prep, des, None), lambda: sensitivity.run(prep, des, None, None)]
    out = [f"Covariates used ({len(prep.adjustment_set)} columns after encoding): "
           + (", ".join(prep.adjustment_set[:40]) or "none") + ("" if len(prep.adjustment_set) <= 40 else ", ...")]
    for i, ((name, method, applies), fn) in enumerate(zip(GENERAL, runs), 1):
        try:
            res = _line(fn())
        except Exception as e:  # noqa: BLE001
            res = f"could not run: {type(e).__name__}: {e}"
        out.append(f"Check #{i}: {name} ({method}) - {res} - Applies when: {applies}")
    return out


# ------------------------------------------------------------------------------- structure facts
def structure_facts(df: pd.DataFrame, treatment: str, outcome: str, instruments: Optional[list[str]] = None,
                    covariates: Optional[list[str]] = None) -> list[str]:
    """Facts about how treatment is laid out in the table. No design is named."""
    out = []
    t = pd.to_numeric(df[treatment], errors="coerce") if pd.api.types.is_numeric_dtype(df[treatment]) else None
    tb = data._as_bool(df[treatment]) if t is None else t
    vals = set(pd.unique(tb.dropna())) if tb is not None else set()
    binary = vals <= {0, 1} and len(vals) == 2
    if binary:
        out.append(f"Treatment '{treatment}' is 0/1; {tb.mean():.1%} of rows are treated.")
    elif tb is not None and len(vals):
        out.append(f"Treatment '{treatment}' is numeric with {len(vals)} distinct values "
                   f"({_fmt(tb.min())} to {_fmt(tb.max())}).")
    else:
        out.append(f"Treatment '{treatment}' is not numeric ({df[treatment].nunique()} labels).")
        return out
    # within-unit variation for columns that repeat
    n = len(df)
    for c in df.columns:
        if c in (treatment, outcome):
            continue
        k = df[c].nunique()
        if not (5 <= k <= n / 2):
            continue
        if pd.api.types.is_float_dtype(df[c]) and not np.allclose(df[c].dropna() % 1, 0):
            continue
        g = pd.DataFrame({"u": df[c], "t": tb}).dropna().groupby("u")["t"]
        varies = float(g.nunique().gt(1).mean())
        if varies > 0.02:
            out.append(f"Within values of '{c}' ({k} groups, about {n / k:.1f} rows each), treatment changes in "
                       f"{varies:.0%} of groups.")
        if len(out) >= 5:
            break
    if binary:
        best = None
        for c in df.select_dtypes("number").columns:
            if c in (treatment, outcome) or df[c].nunique() < 10:
                continue
            cut, acc = data.find_cutoff(df[c], tb)
            if cut is not None and acc >= 0.9 and data._dense_at(df[c], cut) and data._sharp_at(df[c], tb, cut):
                if best is None or acc > best[2]:
                    best = (c, cut, acc)
        if best:
            out.append(f"Treatment equals 1 exactly when '{best[0]}' is at or beyond {_fmt(best[1])} in {best[2]:.0%} "
                       "of rows, with data on both sides of that value.")
        else:
            out.append("No numeric column was found on which treatment switches at a single threshold.")
    for z in (instruments or [])[:6]:
        if z not in df.columns or z in (treatment, outcome):
            continue
        try:
            import statsmodels.api as sm
            cols = [c for c in (covariates or []) if c in df.columns and c != z
                    and pd.api.types.is_numeric_dtype(df[c])][:25]
            d = pd.DataFrame({"t": tb, "z": pd.to_numeric(df[z], errors="coerce")}).join(df[cols]).dropna()
            if len(d) < 30 or d["z"].nunique() < 2:
                continue
            fit = sm.OLS(d["t"], sm.add_constant(d[["z"] + cols])).fit(cov_type="HC1")
            f = float(fit.tvalues["z"] ** 2)
            out.append(f"'{z}' (proposed as a possible instrument) predicts treatment with first-stage F = {_fmt(f)} "
                       f"given {len(cols)} covariates (below 10 is weak). Relevance only: whether it affects the "
                       "outcome solely through treatment cannot be tested.")
        except Exception:  # noqa: BLE001
            continue
    return out


# ------------------------------------------------------------------------------- design tools
def design_tests(df: pd.DataFrame, design: str, roles: dict, covariates: Optional[list[str]] = None) -> list[str]:
    from . import design_assess
    prep, des = _prepare(df, design, roles, covariates)
    out = []
    for a in des.conflicts:
        out.append(f"Declared design or roles do not match the data: {a.name}. {_short(a.detail, 260)}")
    if not des.conflicts:
        out.append("The declared roles exist and have the layout this design needs. This does not show the design is "
                   "the right one.")
    assumptions, sections = design_assess.assess(prep, des, None)
    if design in ("experiment", "observational"):
        from .checks import balance
        sections = [balance.run(prep, des, None)] + list(sections)
    for s in sections:
        out.append(f"Test of '{s.title}': {_line(s)}")
    if not sections:
        out.append("This design has no tests of its own; see the general checks.")
    assumed = [a.name for a in assumptions if not a.testable]
    if assumed:
        out.append("Not tested, because no data can test them: " + "; ".join(dict.fromkeys(assumed)) + ".")
    return out


def estimate_effect(df: pd.DataFrame, design: str, roles: dict, covariates: Optional[list[str]] = None) -> list[str]:
    from . import estimate as est_mod
    prep, des = _prepare(df, design, roles, covariates)
    e = est_mod.estimate(prep, des, None)
    if e.refused or e.estimate is None:
        return [f"No estimate: {e.refusal_reason or 'the estimator refused'}"]
    lv = e.extra.get("ci_levels") or {}
    out = [f"Method: {e.method}. Target: {e.target}; speaks for: {e.scope}. n = {e.n}.",
           f"Estimate {_fmt(e.estimate)}, standard error {_fmt(e.se)}, p = {_fmt(e.p_value)}.",
           "Intervals: " + "; ".join(f"{k}% {_fmt(v[0])} to {_fmt(v[1])}" for k, v in lv.items())
           if lv else f"95% interval {_fmt(e.ci95[0])} to {_fmt(e.ci95[1])}."]
    if e.extra.get("ar_ci95_text"):
        out.append("Anderson-Rubin 95% set: " + str(e.extra["ar_ci95_text"]))
    for c in (e.comparisons or [])[:4]:
        ci = c.get("ci95")
        out.append(f"Other method: {c.get('method')}: {_fmt(c.get('estimate'))}"
                   + (f" (95% {_fmt(ci[0])} to {_fmt(ci[1])})" if ci and None not in ci else ""))
    for nline in (e.extra.get("notes") or [])[:3]:
        out.append("Note: " + _short(str(nline), 220))
    return out
