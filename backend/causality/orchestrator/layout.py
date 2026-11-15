"""Tool `layout_facts`: design-free facts about how treatment is laid out in the table.

The model reads these to form its own hypotheses about how treatment was assigned when the
description is thin (for example a table whose columns are named c01, c02, ...). Nothing here
names a design as the answer or says which columns to adjust for; lines are plain sentences
with numbers.

    run(df, {"treatment": str, "outcome": str}) -> (lines, numbers)
"""
from __future__ import annotations

import itertools
import math
from typing import Optional

import numpy as np
import pandas as pd

from .. import data

TOOL_NAME = "layout_facts"

TOOL_SPEC = {
    "name": TOOL_NAME,
    "description": (
        "Facts about how treatment is laid out in the table, without assuming any design: numeric "
        "columns on which a 0/1 treatment switches at a single value; pairs of columns that identify "
        "a row and within which treatment changes over time; how well the other columns predict "
        "treatment (cross-validated); columns that move treatment while being unrelated to every "
        "other column; and the columns most strongly associated with treatment. Call it early, "
        "especially when the description of the data is thin or the column names carry no meaning."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "treatment": {"type": "string", "description": "Treatment column, as named in the table."},
            "outcome": {"type": "string", "description": "Outcome column, as named in the table."},
        },
        "required": ["treatment", "outcome"],
        "additionalProperties": False,
    },
}

MAX_FIT_ROWS = 5000        # rows used for each cross-validated model (sampled, fixed seed)
MAX_PANEL_CANDIDATES = 12  # repeated columns tried as unit or time
MAX_INSTRUMENT_TRIES = 6   # columns whose own predictability is checked
ASSOC_P = 1e-3             # "strongly associated": p below this ...
ASSOC_SIZE = 0.1           # ... and |standardized difference| or |correlation| at least this


# --------------------------------------------------------------------------- formatting
def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(x):
        return str(x)
    if x == int(x) and abs(x) < 1e7:
        return str(int(x))
    a = abs(x)
    return f"{x:.4g}" if 1e-3 <= a < 1e6 else f"{x:.3e}"


def _fmt_p(p) -> str:
    if p is None or not np.isfinite(p):
        return "p = n/a"
    if p < 1e-300:
        return "p < 1e-300"
    if p < 1e-4:
        return f"p = {p:.1e}"
    return f"p = {p:.3g}"


def _num(x) -> Optional[float]:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _plain(v):
    """numpy scalars -> plain Python values (JSON-safe)."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return _num(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


# --------------------------------------------------------------------------- column helpers
def _treatment_series(s: pd.Series) -> tuple[Optional[pd.Series], bool, str]:
    """Numeric version of the treatment, whether it is 0/1, and a note on any recoding."""
    b = data._as_bool(s)
    if b is not None:
        return b, True, ""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float), True, ""
    if pd.api.types.is_numeric_dtype(s):
        t = pd.to_numeric(s, errors="coerce").astype(float)
        vals = set(pd.unique(t.dropna()))
        if vals <= {0.0, 1.0} and len(vals) == 2:
            return t, True, ""
        if len(vals) == 2:
            lo, hi = sorted(vals)
            note = f"its two values {_fmt(lo)} and {_fmt(hi)} are read as 0 and 1"
            return (t == hi).astype(float).where(t.notna()), True, note
        return t, False, ""
    labels = sorted(s.dropna().astype(str).unique())
    if len(labels) == 2:
        note = f"its two labels '{labels[0]}' and '{labels[1]}' are read as 0 and 1"
        return (s.astype(str) == labels[1]).astype(float).where(s.notna()), True, note
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.95:
        return _treatment_series(num)
    return None, False, ""


def _is_row_id(s: pd.Series) -> bool:
    """A column whose values are unique per row and look like labels or integer ids."""
    nn = s.dropna()
    if len(nn) < 20 or nn.nunique() != len(s):
        return False
    if pd.api.types.is_numeric_dtype(s):
        return bool(np.allclose(nn % 1, 0))
    return True


def _integerish(s: pd.Series) -> bool:
    if not pd.api.types.is_numeric_dtype(s):
        return True
    nn = s.dropna()
    return bool(len(nn)) and bool(np.allclose(nn % 1, 0))


def _encode(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Numeric design matrix: numbers as they are (median-filled), few-label text one-hot."""
    parts = []
    for c in cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(float)
        if pd.api.types.is_numeric_dtype(s):
            x = pd.to_numeric(s, errors="coerce").astype(float)
            if x.isna().any():
                parts.append(x.isna().astype(float).rename(f"{c}__missing"))
                x = x.fillna(x.median() if x.notna().any() else 0.0)
            parts.append(x.rename(c))
        else:
            k = s.nunique()
            if 2 <= k <= 30:
                d = pd.get_dummies(s.astype(str).where(s.notna(), "__missing__"), prefix=c,
                                   drop_first=True, dtype=float)
                parts.append(d)
    if not parts:
        return pd.DataFrame(index=df.index)
    return pd.concat(parts, axis=1)


def _cv_predict(X: pd.DataFrame, y: pd.Series, binary: bool, seed: int = 0) -> Optional[float]:
    """Cross-validated AUC (binary y) or R^2 of a boosted-tree model. None when not computable."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.metrics import r2_score, roc_auc_score
    from sklearn.model_selection import KFold, StratifiedKFold

    ok = y.notna()
    X, y = X.loc[ok], y.loc[ok]
    if X.shape[1] == 0 or len(y) < 30 or y.nunique() < 2:
        return None
    if len(y) > MAX_FIT_ROWS:
        idx = np.random.default_rng(seed).choice(len(y), MAX_FIT_ROWS, replace=False)
        X, y = X.iloc[idx], y.iloc[idx]
    Xa, ya = X.to_numpy(dtype=float), y.to_numpy(dtype=float)
    pred = np.full(len(ya), np.nan)
    try:
        if binary:
            if min(np.sum(ya == 1), np.sum(ya == 0)) < 6:
                return None
            folds = StratifiedKFold(3, shuffle=True, random_state=seed).split(Xa, ya)
            for tr, te in folds:
                m = HistGradientBoostingClassifier(max_iter=60, learning_rate=0.1, max_leaf_nodes=15,
                                                   early_stopping=False, random_state=seed)
                m.fit(Xa[tr], ya[tr])
                pred[te] = m.predict_proba(Xa[te])[:, 1]
            return float(roc_auc_score(ya, pred))
        folds = KFold(3, shuffle=True, random_state=seed).split(Xa)
        for tr, te in folds:
            m = HistGradientBoostingRegressor(max_iter=60, learning_rate=0.1, max_leaf_nodes=15,
                                              early_stopping=False, random_state=seed)
            m.fit(Xa[tr], ya[tr])
            pred[te] = m.predict(Xa[te])
        return float(r2_score(ya, pred))
    except Exception:  # noqa: BLE001
        return None


def _association(x: pd.Series, t: pd.Series, t_binary: bool) -> Optional[dict]:
    """Size and p-value of the association between one column and treatment."""
    from scipy import stats

    if pd.api.types.is_bool_dtype(x):
        x = x.astype(float)
    if pd.api.types.is_numeric_dtype(x):
        d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "t": t}).dropna()
        if len(d) < 10 or d["x"].nunique() < 2:
            return None
        if t_binary:
            a, b = d.loc[d["t"] == 1, "x"], d.loc[d["t"] == 0, "x"]
            if len(a) < 2 or len(b) < 2:
                return None
            sd = math.sqrt((a.var() + b.var()) / 2)
            size = (a.mean() - b.mean()) / sd if sd > 0 else 0.0
            p = stats.ttest_ind(a, b, equal_var=False).pvalue
            return {"value": _num(size), "p": _num(p), "measure": "standardized difference"}
        r, p = stats.pearsonr(d["x"], d["t"])
        return {"value": _num(r), "p": _num(p), "measure": "correlation"}
    k = x.nunique()
    if not 2 <= k <= 30:
        return None
    d = pd.DataFrame({"x": x.astype(str), "t": t}).dropna()
    if t_binary:
        tab = pd.crosstab(d["x"], d["t"])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            return None
        chi2, p, _, _ = stats.chi2_contingency(tab)
        v = math.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
        return {"value": _num(v), "p": _num(p), "measure": "Cramer's V"}
    groups = [g["t"].to_numpy() for _, g in d.groupby("x") if len(g) > 1]
    if len(groups) < 2:
        return None
    f, p = stats.f_oneway(*groups)
    ss_b = sum(len(g) * (g.mean() - d["t"].mean()) ** 2 for g in groups)
    ss_t = ((d["t"] - d["t"].mean()) ** 2).sum()
    return {"value": _num(math.sqrt(ss_b / ss_t)) if ss_t > 0 else 0.0, "p": _num(p),
            "measure": "correlation ratio"}


# --------------------------------------------------------------------------- the five facts
def _thresholds(df, t, skip) -> list[dict]:
    out = []
    for c in df.select_dtypes("number").columns:
        if c in skip or df[c].nunique() < 10:
            continue
        x = pd.to_numeric(df[c], errors="coerce").astype(float)
        cut, acc = data.find_cutoff(x, t)
        if cut is None or acc < 0.9 or not data._sharp_at(x, t, cut):
            continue
        d = pd.DataFrame({"x": x, "t": t}).dropna()
        window = float(0.25 * d["x"].std())
        near = d[(d["x"] - cut).abs() < window]
        above = d["x"] >= cut
        treated_above = d.loc[above, "t"].mean() >= d.loc[~above, "t"].mean()
        out.append({
            "column": str(c), "cutoff": _num(cut), "share_consistent": _num(acc),
            "n_below_near": int((near["x"] < cut).sum()), "n_above_near": int((near["x"] >= cut).sum()),
            "window": _num(window),
            "treated_side": "at or above" if treated_above else "below",
            "share_treated_below": _num(d.loc[~above, "t"].mean()),
            "share_treated_above": _num(d.loc[above, "t"].mean()),
            "data_on_both_sides": bool(data._dense_at(x, cut)),
        })
    out.sort(key=lambda r: -r["share_consistent"])
    return out[:5]


def _panel_stats(df, t, unit, time, t_binary) -> Optional[dict]:
    d = pd.DataFrame({"u": df[unit], "time": df[time], "t": t}).dropna()
    if len(d) < 20:
        return None
    d = d.sort_values(["u", "time"], kind="stable")
    u = d["u"]
    diff = d.groupby("u", sort=False)["t"].diff()
    up = (diff > 0).groupby(u).any()
    down = (diff < 0).groupby(u).any()
    n_changes = (diff.fillna(0) != 0).groupby(u).sum()
    n_units = int(u.nunique())
    if float((up | down).mean()) <= 0.02:
        return None
    base = 0.0 if t_binary else float(d["t"].min())
    on = d["t"] != base
    never = (~on).groupby(u).all()
    # first period with treatment on, for units that start untreated and later switch on
    starts_off = ~on.groupby(u).first()
    first_time = d.loc[on].groupby("u")["time"].first()
    first_time = first_time[first_time.index.isin(starts_off[starts_off].index)]
    counts = first_time.value_counts().sort_index()
    return {
        "unit": str(unit), "time": str(time), "n_units": n_units, "n_periods": int(d["time"].nunique()),
        "share_units_switching_on": _num(up.mean()), "share_units_switching_off": _num(down.mean()),
        "n_never_treated": int(never.sum()),
        "start_times": [_plain(v) for v in list(counts.index)[:20]],
        "units_per_start_time": [int(v) for v in list(counts.values)[:20]],
        "share_units_with_one_switch": _num((n_changes == 1).mean()),
    }


def _panels(df, t, t_binary, skip) -> list[dict]:
    n = len(df)
    cand = []
    for c in df.columns:
        if c in skip:
            continue
        k = df[c].nunique()
        if 2 <= k <= n / 2 and _integerish(df[c]):
            cand.append(c)
    cand = sorted(cand, key=lambda c: -df[c].nunique())[:MAX_PANEL_CANDIDATES]
    found = []
    for a, b in itertools.combinations(cand, 2):
        if df[a].nunique() * df[b].nunique() < n * 0.98 or df.duplicated(subset=[a, b]).any():
            continue
        best = None
        for unit, time in ((a, b), (b, a)):
            if not (pd.api.types.is_numeric_dtype(df[time]) or pd.api.types.is_datetime64_any_dtype(df[time])):
                continue
            s = _panel_stats(df, t, unit, time, t_binary)
            if s is None:
                continue
            key = (s["share_units_with_one_switch"] or 0.0, s["n_units"])
            if best is None or key > best[0]:
                best = (key, s)
        if best:
            found.append(best[1])
    found.sort(key=lambda s: (-(s["share_units_with_one_switch"] or 0.0), -s["n_units"]))
    return found[:3]


def _instrument_candidates(df, t, y, t_binary, assoc, feats, skip_cols) -> list[dict]:
    import statsmodels.api as sm
    from scipy import stats

    tries = []
    for c, a in assoc.items():
        if c in skip_cols or a is None or not pd.api.types.is_numeric_dtype(df[c]):
            continue
        d = pd.DataFrame({"t": t, "z": pd.to_numeric(df[c], errors="coerce")}).dropna()
        if len(d) < 30 or d["z"].nunique() < 2:
            continue
        try:
            fit = sm.OLS(d["t"], sm.add_constant(d[["z"]])).fit(cov_type="HC1")
            f = float(fit.tvalues["z"] ** 2)
        except Exception:  # noqa: BLE001
            continue
        if f >= 10:
            tries.append((f, c))
    tries.sort(reverse=True)
    out = []
    for f, c in tries[:MAX_INSTRUMENT_TRIES]:
        others = [k for k in feats if k != c]
        X = _encode(df, others)
        z = pd.to_numeric(df[c], errors="coerce")
        z_binary = z.nunique() == 2
        if z_binary:
            z = (z == z.max()).astype(float).where(z.notna())
        pred = _cv_predict(X, z, z_binary, seed=1)
        if pred is None and X.shape[1] > 0:
            continue
        low = pred is None or (pred < 0.6 if z_binary else pred < 0.1)
        if not low:
            continue
        dy = pd.DataFrame({"z": pd.to_numeric(df[c], errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
        r, p = (stats.pearsonr(dy["z"], dy["y"]) if len(dy) > 3 and dy["z"].nunique() > 1
                and dy["y"].nunique() > 1 else (np.nan, np.nan))
        by_value = []
        zr = pd.to_numeric(df[c], errors="coerce")
        if zr.nunique() <= 5:
            g = pd.DataFrame({"z": zr, "t": t}).dropna().groupby("z")["t"]
            by_value = [{"value": _plain(v), "share_rows": _num(k / g.size().sum()), "mean_treatment": _num(m)}
                        for (v, k), m in zip(g.size().items(), g.mean().values)]
        out.append({
            "column": str(c), "first_stage_f": _num(f), "predictability_from_others": _num(pred),
            "predictability_metric": "auc" if z_binary else "r2",
            "assoc_with_outcome": _num(r), "p_value_outcome": _num(p),
            "n_values": int(df[c].nunique()), "column_kind": data.column_kind(df[c]),
            "treatment_by_value": by_value,
        })
    return out


# --------------------------------------------------------------------------- run
def run(df: pd.DataFrame, args: dict) -> tuple[list[str], dict]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("The table is empty.")
    args = args or {}
    T, Y = args.get("treatment"), args.get("outcome")
    if not T or not isinstance(T, str):
        raise ValueError("Give the treatment column name.")
    if not Y or not isinstance(Y, str):
        raise ValueError("Give the outcome column name.")
    for name, c in (("treatment", T), ("outcome", Y)):
        if c not in df.columns:
            raise ValueError(f"The {name} column '{c}' is not in the table. Columns: "
                             + ", ".join(map(str, list(df.columns)[:40])) + ".")
    if T == Y:
        raise ValueError("Treatment and outcome must be different columns.")

    numbers = {"thresholds": [], "panels": [], "treatment_predictability": None,
               "instrument_candidates": [], "associated_columns": []}
    lines: list[str] = []
    t, t_binary, note = _treatment_series(df[T])
    n = len(df)
    if t is None:
        lines.append(f"Treatment '{T}' is not numeric and has {df[T].nunique()} labels; the layout facts below "
                     "need a numeric or two-valued treatment, so none were computed.")
        return lines, numbers
    if t.nunique() < 2:
        lines.append(f"Treatment '{T}' takes a single value in this table; there is no variation to describe.")
        return lines, numbers
    if t_binary:
        lines.append(f"Treatment '{T}' is 0/1" + (f" ({note})" if note else "")
                     + f"; {t.mean():.1%} of {int(t.notna().sum())} rows are treated.")
    else:
        lines.append(f"Treatment '{T}' is numeric with {t.nunique()} distinct values "
                     f"({_fmt(t.min())} to {_fmt(t.max())}).")

    others = [c for c in df.columns if c not in (T, Y) and df[c].nunique(dropna=True) > 1]
    row_ids = [c for c in others if _is_row_id(df[c])]

    # 1. thresholds
    if t_binary:
        th = _thresholds(df, t, set(row_ids) | {T, Y})
        numbers["thresholds"] = th
        for r in th:
            side = r["treated_side"]
            lines.append(
                f"Treatment switches at a single value of '{r['column']}': rows with '{r['column']}' {side} "
                f"{_fmt(r['cutoff'])} are treated and the others are not in {r['share_consistent']:.1%} of rows "
                f"(treated share {r['share_treated_below']:.1%} below {_fmt(r['cutoff'])}, "
                f"{r['share_treated_above']:.1%} at or above). Within {_fmt(r['window'])} of {_fmt(r['cutoff'])} "
                f"there are {r['n_below_near']} rows below and {r['n_above_near']} at or above"
                + ("." if r["data_on_both_sides"] else
                   "; the number of rows right at the value is lower than elsewhere in the column on at least one side."))
        if not th:
            lines.append("No numeric column was found on which treatment switches at a single value.")

    # 2. panels
    pn = _panels(df, t, t_binary, set(row_ids) | {T, Y})
    numbers["panels"] = pn
    for p in pn:
        starts = ", ".join(f"{_fmt(s)} ({k} units)" for s, k in zip(p["start_times"][:8], p["units_per_start_time"][:8]))
        lines.append(
            f"Each row is identified by the pair '{p['unit']}' ({p['n_units']} values) x '{p['time']}' "
            f"({p['n_periods']} values, read as time). Within '{p['unit']}', treatment goes from off to on over "
            f"'{p['time']}' in {p['share_units_switching_on']:.0%} of units and from on to off in "
            f"{p['share_units_switching_off']:.0%}; {p['n_never_treated']} units are never treated"
            + (f"; first treated '{p['time']}': {starts}." if starts else "."))
    if not pn:
        lines.append("No pair of columns was found that identifies each row and within which treatment changes "
                     "over time.")
    panel_cols = {c for p in pn for c in (p["unit"], p["time"])}

    # 3. predictability
    feats = [c for c in others if c not in row_ids and c not in panel_cols]
    assoc = {c: _association(df[c], t, t_binary) for c in feats}
    X = _encode(df, feats)
    value = _cv_predict(X, t, t_binary)
    per_col = [{"column": str(c), "std_diff_or_corr": a["value"], "p_value": a["p"], "measure": a["measure"]}
               for c, a in assoc.items() if a is not None]
    per_col.sort(key=lambda r: -abs(r["std_diff_or_corr"] or 0.0))
    numbers["treatment_predictability"] = {"value": _num(value), "metric": "auc" if t_binary else "r2",
                                           "n_columns": len(feats), "per_column": per_col}
    left_out = row_ids + sorted(panel_cols)
    excl = (" and " + ", ".join(f"'{c}'" for c in left_out[:6])) if left_out else ""
    if value is None:
        lines.append(f"Treatment could not be predicted from the other columns ({len(feats)} usable columns).")
    elif t_binary:
        lines.append(f"From the other {len(feats)} columns (leaving out treatment, outcome{excl}), a cross-validated "
                     f"model predicts treatment with AUC {value:.2f} (0.5 is chance level, 1.0 is perfect).")
    else:
        lines.append(f"From the other {len(feats)} columns (leaving out treatment, outcome{excl}), a cross-validated "
                     f"model explains R^2 = {value:.2f} of the variation in treatment.")

    # 4. instrument candidates
    th_cols = {r["column"] for r in numbers["thresholds"]}
    ic = _instrument_candidates(df, t, df[Y], t_binary, assoc, feats, th_cols)
    numbers["instrument_candidates"] = ic
    for r in ic:
        metric = "AUC" if r["predictability_metric"] == "auc" else "R^2"
        pred = "n/a (no other columns)" if r["predictability_from_others"] is None else \
            f"{r['predictability_from_others']:.2f}"
        tv = r["treatment_by_value"]
        share = ""
        if tv and t_binary:
            share = " Treated share by its value: " + ", ".join(
                f"{_fmt(v['value'])}: {v['mean_treatment']:.1%} ({v['share_rows']:.0%} of rows)" for v in tv) + "."
        lines.append(
            f"'{r['column']}' ({r['n_values']} distinct values) moves with treatment (first-stage F = "
            f"{_fmt(r['first_stage_f'])} from '{r['column']}' alone), yet the other columns do not predict it "
            f"(cross-validated {metric} {pred}).{share} Its correlation with the outcome is "
            f"{_fmt(r['assoc_with_outcome'])} ({_fmt_p(r['p_value_outcome'])}). A column with this pattern can be a "
            "source of variation in treatment unrelated to everything else measured (for example a random "
            "encouragement), a common cause of treatment and outcome, or a consequence of treatment; the numbers "
            "cannot tell these apart, and whether it affects the outcome only through treatment cannot be tested "
            "from data.")
    if not ic:
        lines.append("No column was found that moves treatment (first-stage F of at least 10) while being "
                     "unpredictable from the other columns.")

    # 5. associated columns
    strong = [r for r in per_col if r["p_value"] is not None and r["p_value"] < ASSOC_P
              and abs(r["std_diff_or_corr"] or 0.0) >= ASSOC_SIZE]
    numbers["associated_columns"] = [{"column": r["column"], "std_diff_or_corr": r["std_diff_or_corr"],
                                      "p_value": r["p_value"], "measure": r["measure"]} for r in strong[:10]]
    if strong:
        parts = "; ".join(f"'{r['column']}' ({r['measure']} {_fmt(r['std_diff_or_corr'])}, {_fmt_p(r['p_value'])})"
                          for r in strong[:8])
        lines.append(
            f"Columns strongly associated with treatment: {parts}. The numbers cannot tell whether such a column is "
            "a cause of treatment or a consequence of it (something treatment itself changed); that depends on when "
            "and how it was measured, which is a question for the user.")
    else:
        lines.append(f"No column is strongly associated with treatment (p below {ASSOC_P:g} and size at least "
                     f"{ASSOC_SIZE:g}).")
    return lines, numbers
