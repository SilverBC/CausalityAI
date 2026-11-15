"""Charts the model can put on the audit page (part H of ORCHESTRATOR_MERGED.md).

TOOL_SPEC              the Anthropic tool definition for `show_chart`.
charts_for(item, df)   the standard charts for one evidence item; never raises, [] when none.
draw(df, args)         a custom chart computed from columns; raises ValueError with a plain message.

Every chart is a plain dict in the shape of `schema.Chart` (kind, title, data, x_label, y_label,
note), with plain JSON values only (no numpy types, no NaN or infinity) and no array longer than
MAX_POINTS. The `data` shape of each kind is the one `frontend/charts.js` draws.
"""
from __future__ import annotations

import math
import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd

MAX_POINTS = 400          # no array in a chart is longer than this
MAX_GROUPS = 6            # charts.js has six series colors
MAX_BARS = 40             # more bars than this cannot be read
HIST_BINS = 30
SCATTER_BINS = 40         # binned means per group when a scatter is too large to draw raw
LINE_BINS = 50
Z95 = 1.959963984540054

KINDS = ("hist", "scatter", "line", "bar")

_NULLABLE_STR = {"anyOf": [{"type": "string"}, {"type": "null"}]}

TOOL_SPEC: dict = {
    "name": "show_chart",
    "description": (
        "Put one picture on the audit page so a reader who is not a statistician can follow the verdict. "
        "Either give evidence_id (an id such as e3 from an earlier tool call; the standard charts for that "
        "item are shown: the estimate with its intervals and other methods, the outcome around the cutoff, "
        "outcome over time for ever-treated and never-treated units, outcome and treatment by instrument "
        "value, event-study coefficients, standardized differences of covariates) and leave kind, x, y and "
        "group null; or leave evidence_id null and build a chart from columns: kind 'hist' (distribution of "
        "x, one series per value of group), 'scatter' (y against x; binned means when the table is large), "
        "'line' (mean of y by x, one line per value of group) or 'bar' (mean of y for each value of x; "
        "row counts when y is null). Use two to four charts in all. title and note are shown to the reader "
        "in plain words; note says what to look at."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "evidence_id": {"anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "An evidence id for its standard charts, or null for a chart from columns."},
            "kind": {"anyOf": [{"type": "string", "enum": list(KINDS)}, {"type": "null"}],
                     "description": "Chart kind for a chart from columns; null with an evidence id."},
            "x": {**_NULLABLE_STR, "description": "Column on the horizontal axis (or the categories of a bar chart)."},
            "y": {**_NULLABLE_STR, "description": "Column on the vertical axis; null for hist, or for row counts in a bar."},
            "group": {**_NULLABLE_STR, "description": "Column whose values split the data into series; or null."},
            "title": {"type": "string", "description": "Short plain title using the real column names."},
            "note": {"type": "string", "description": "One plain sentence: what the reader should look at."},
        },
        "required": ["evidence_id", "kind", "x", "y", "group", "title", "note"],
        "additionalProperties": False,
    },
}


# --------------------------------------------------------------------------------- helpers
def _f(x) -> Optional[float]:
    """A finite float, or None."""
    if x is None or isinstance(x, bool):
        return None if x is None else float(x)
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _clean(x: Any, depth: int = 0) -> Any:
    """Plain JSON values: no numpy types, NaN/inf -> None, arrays cut at MAX_POINTS."""
    if depth > 10:
        return None
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _f(x)
    if isinstance(x, dict):
        return {str(k): _clean(v, depth + 1) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray, pd.Series, pd.Index)):
        return [_clean(v, depth + 1) for v in list(x)[:MAX_POINTS]]
    return str(x)


def _chart(kind: str, title: str, data: dict, x_label: str = "", y_label: str = "", note: str = "") -> dict:
    return _clean({"kind": kind, "title": str(title or ""), "data": data, "x_label": str(x_label or ""),
                   "y_label": str(y_label or ""), "note": str(note or "")})


def _fmt(v) -> str:
    v = _f(v)
    if v is None:
        return "n/a"
    if v == int(v) and abs(v) < 1e9:
        return str(int(v))
    return f"{v:.4g}"


def _label(v) -> str:
    """A category value as a short plain label."""
    if isinstance(v, (float, np.floating)) and _f(v) is not None:
        return _fmt(v)
    if isinstance(v, (np.integer,)):
        return str(int(v))
    return str(v)


def _numeric(s: pd.Series) -> Optional[pd.Series]:
    """The column as floats (bools and dates converted, infinity as missing), or None when it is not
    numeric-like."""
    v = _numeric_raw(s)
    return None if v is None else v.replace([np.inf, -np.inf], np.nan)


def _numeric_raw(s: pd.Series) -> Optional[pd.Series]:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").astype(float)
    if pd.api.types.is_datetime64_any_dtype(s):
        return _decimal_year(s)
    v = pd.to_numeric(s, errors="coerce")
    ok = s.notna()
    if ok.sum() and v[ok].notna().mean() >= 0.95:
        return v.astype(float)
    if s.dtype == object and ok.sum():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = pd.to_datetime(s, errors="coerce")
        except Exception:  # noqa: BLE001
            return None
        if d[ok].notna().mean() >= 0.95:
            return _decimal_year(d)
    return None


def _decimal_year(d: pd.Series) -> pd.Series:
    d = pd.to_datetime(d, errors="coerce")
    try:
        d = d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        pass
    out = d.dt.year + (d.dt.day_of_year - 1) / 365.25
    return out.astype(float)


def _top_groups(s: pd.Series, k: int = MAX_GROUPS) -> list:
    return list(s.dropna().value_counts().index[:k])


def _col(df: pd.DataFrame, name, what: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"The {what} must be a column name.")
    if name not in df.columns:
        near = [c for c in map(str, df.columns) if c.lower() == name.lower()]
        hint = f" Did you mean '{near[0]}'?" if near else ""
        raise ValueError(f"The {what} column '{name}' is not in the table.{hint}")
    return name


def _need_numeric(df: pd.DataFrame, name: str, what: str) -> pd.Series:
    v = _numeric(df[name])
    if v is None or v.notna().sum() == 0:
        raise ValueError(f"The {what} column '{name}' has no numbers to draw.")
    return v


def _bin_edges(v: np.ndarray, nbins: int) -> np.ndarray:
    lo, hi = float(np.min(v)), float(np.max(v))
    if lo == hi:
        d = abs(lo) * 0.05 or 0.5
        lo, hi = lo - d, hi + d
    return np.linspace(lo, hi, nbins + 1)


def _binned_means(x: np.ndarray, y: np.ndarray, nbins: int, quantile: bool = True):
    """(mean x, mean y, n) per bin of x; equal-count bins when quantile, else equal width."""
    if len(x) == 0:
        return [], [], []
    if quantile:
        edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
        if len(edges) < 2:
            edges = _bin_edges(x, 1)
    else:
        edges = _bin_edges(x, nbins)
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    mx, my, n = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            mx.append(float(x[m].mean()))
            my.append(float(y[m].mean()))
            n.append(int(m.sum()))
    return mx, my, n


# --------------------------------------------------------------------------------- standard charts
def _forest(numbers: dict) -> Optional[dict]:
    rows = []
    est = _f(numbers.get("estimate"))
    ci = (numbers.get("ci") or {}).get("95") if isinstance(numbers.get("ci"), dict) else None
    method = str(numbers.get("method") or "Main estimate")
    if est is not None and not numbers.get("refused"):
        lo = _f(ci[0]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
        hi = _f(ci[1]) if isinstance(ci, (list, tuple)) and len(ci) == 2 else None
        rows.append({"label": f"{method} (main)", "est": est, "lo": lo, "hi": hi})
    for c in numbers.get("comparisons") or []:
        if not isinstance(c, dict):
            continue
        e = _f(c.get("estimate"))
        if e is None:
            continue
        c95 = c.get("ci95")
        lo = _f(c95[0]) if isinstance(c95, (list, tuple)) and len(c95) == 2 else None
        hi = _f(c95[1]) if isinstance(c95, (list, tuple)) and len(c95) == 2 else None
        rows.append({"label": str(c.get("method") or "Other method"), "est": e, "lo": lo, "hi": hi})
    if not rows:
        return None
    roles = numbers.get("roles") or {}
    t, y = roles.get("treatment", "treatment"), roles.get("outcome", "outcome")
    note = ("Each row is one way of estimating the effect, with its 95% interval. The dashed line is zero "
            "(no effect).")
    if (numbers.get("design") == "iv") and numbers.get("ci_method"):
        note += " The main row's interval is the Anderson-Rubin interval when it is a single bounded range."
    return _chart("forest", f"Estimated effect of {t} on {y}", {"rows": rows[:MAX_POINTS], "ref": 0.0},
                  x_label=f"Change in {y}", note=note)


def _rd_scatter(numbers: dict, df: pd.DataFrame) -> Optional[dict]:
    roles = numbers.get("roles") or {}
    run, y = roles.get("running"), roles.get("outcome")
    cut = _f(numbers.get("cutoff_used"))
    if cut is None:
        cut = _f(roles.get("cutoff"))
    if not run or not y or run not in df.columns or y not in df.columns or cut is None:
        return None
    x = _numeric(df[run])
    yy = _numeric(df[y])
    if x is None or yy is None:
        return None
    ok = x.notna() & yy.notna()
    xv, yv = x[ok].to_numpy(float), yy[ok].to_numpy(float)
    if len(xv) == 0:
        return None
    points, lines = [], []
    sides = [("below the cutoff", xv < cut), ("at or above the cutoff", xv >= cut)]
    for name, m in sides:
        if m.sum() == 0:
            continue
        mx, my, _ = _binned_means(xv[m], yv[m], 20, quantile=False)
        points += [{"x": a, "y": b, "group": name} for a, b in zip(mx, my)]
    for name, m in sides:
        if m.sum() >= 3 and np.ptp(xv[m]) > 0:
            k, c = np.polyfit(xv[m], yv[m], 1)
            ends = [float(xv[m].min()), float(xv[m].max())]
            lines.append({"x": ends, "y": [float(k * e + c) for e in ends], "name": f"straight-line fit {name}"})
    if not points:
        return None
    ys = [p["y"] for p in points]
    lines.append({"x": [cut, cut], "y": [min(ys), max(ys)], "name": f"cutoff ({_fmt(cut)})"})
    return _chart("scatter", f"Average {y} along {run}, with the cutoff at {_fmt(cut)}",
                  {"points": points, "lines": lines}, x_label=run, y_label=f"Average {y}",
                  note=(f"Each dot is the average {y} for rows with similar {run}. A jump at the cutoff "
                        "is what the estimate measures. The straight lines use all rows on each side and are "
                        "only a guide; the estimate itself uses rows near the cutoff."))


def _did_line(numbers: dict, df: pd.DataFrame) -> Optional[dict]:
    roles = numbers.get("roles") or {}
    u, t, tr, y = roles.get("unit"), roles.get("time"), roles.get("treatment"), roles.get("outcome")
    if not all(c and c in df.columns for c in (u, t, tr, y)):
        return None
    tv = _numeric(df[t])
    yv = _numeric(df[y])
    trv = _numeric(df[tr])
    if tv is None or yv is None or trv is None:
        return None
    d = pd.DataFrame({"u": df[u], "t": tv, "d": trv, "y": yv}).dropna()
    if d.empty:
        return None
    ever = d.groupby("u")["d"].max() > 0
    d["ever"] = d["u"].map(ever)
    series = []
    periods = np.sort(d["t"].unique())
    binned = len(periods) > MAX_POINTS
    for flag, name in ((True, f"units ever given {tr}"), (False, f"units never given {tr}")):
        g = d[d["ever"] == flag]
        if g.empty:
            continue
        if binned:
            mx, my, _ = _binned_means(g["t"].to_numpy(float), g["y"].to_numpy(float), LINE_BINS, quantile=False)
        else:
            m = g.groupby("t")["y"].mean().sort_index()
            mx, my = m.index.tolist(), m.tolist()
        series.append({"name": name, "x": mx, "y": my, "lo": None, "hi": None})
    if not series:
        return None
    start = d.loc[d["d"] > 0, "t"]
    vline = float(start.min()) if len(start) else None
    note = (f"Average {y} in each period. Before {tr} starts, the two lines should move in parallel for the "
            "comparison to be fair; the dashed line marks the first period in which any unit has it.")
    if binned:
        note += f" Periods are grouped into {LINE_BINS} bins."
    return _chart("line", f"Average {y} over {t}: units ever given {tr} versus never given it",
                  {"series": series, "vline": vline, "hline": None}, x_label=t, y_label=f"Average {y}", note=note)


def _iv_bars(numbers: dict, df: pd.DataFrame) -> list[dict]:
    roles = numbers.get("roles") or {}
    z, tr, y = roles.get("instrument"), roles.get("treatment"), roles.get("outcome")
    if not all(c and c in df.columns for c in (z, tr, y)):
        return []
    trv = _numeric(df[tr])
    yv = _numeric(df[y])
    if trv is None or yv is None:
        return []
    zs = df[z]
    zn = _numeric(zs)
    if zn is not None and zn.nunique() > 12:
        try:
            cats = pd.qcut(zn, 5, duplicates="drop")
        except ValueError:
            return []
        labels_of = {c: f"{z} {_fmt(c.left)} to {_fmt(c.right)}" for c in cats.cat.categories}
        key = cats
    else:
        key = zs
        labels_of = None
    d = pd.DataFrame({"k": key, "d": trv, "y": yv}).dropna()
    if d.empty:
        return []
    out = []
    for col, what, ttl in (("d", f"Share given {tr}" if trv.dropna().isin([0, 1]).all() else f"Average {tr}",
                            f"{tr} by value of {z}"),
                           ("y", f"Average {y}", f"{y} by value of {z}")):
        g = d.groupby("k", observed=True)[col].agg(["mean", "std", "count"]).head(MAX_BARS)
        labels = [labels_of[k] if labels_of else f"{z} = {_label(k)}" for k in g.index]
        se = [(_f(s / math.sqrt(n)) if n > 1 else None) for s, n in zip(g["std"], g["count"])]
        out.append(_chart("bar", f"{what} for each value of {z}",
                          {"labels": labels, "values": g["mean"].tolist(), "ref": None, "errors": se},
                          x_label=what,
                          note=(f"If {z} works as an instrument, it should move {tr}; the difference in {y} "
                                f"across values of {z} is what the estimate scales up. Bars show plus or minus "
                                "one standard error.")))
    return out


def _event_lines(numbers: dict) -> list[dict]:
    out = []
    for test in numbers.get("tests") or []:
        if not isinstance(test, dict):
            continue
        m = test.get("metrics") or {}
        times, coef, se = m.get("event_times"), m.get("coefficients"), m.get("se")
        if not isinstance(times, list) or not isinstance(coef, list) or len(times) != len(coef) or not times:
            continue
        se = se if isinstance(se, list) and len(se) == len(coef) else [None] * len(coef)
        pts = []
        for k, b, s in zip(times, coef, se):
            kk, bb, ss = _f(k), _f(b), _f(s)
            if kk is None or bb is None:
                continue
            pts.append((kk, bb, None if ss is None else bb - Z95 * ss, None if ss is None else bb + Z95 * ss))
        ref = _f(m.get("reference_period"))
        if ref is not None and all(p[0] != ref for p in pts):
            pts.append((ref, 0.0, 0.0, 0.0))
        if not pts:
            continue
        pts.sort(key=lambda p: p[0])
        pts = pts[:MAX_POINTS]
        has_ci = all(p[2] is not None for p in pts)
        roles = numbers.get("roles") or {}
        y = roles.get("outcome", "the outcome")
        pulse = bool(m.get("pulse_treatment"))
        refp = _fmt(ref if ref is not None else -1)
        out.append(_chart(
            "line", f"Change in {y} before and after treatment, relative to period {refp}",
            {"series": [{"name": "estimate with 95% interval" if has_ci else "estimate",
                         "x": [p[0] for p in pts], "y": [p[1] for p in pts],
                         "lo": [p[2] for p in pts] if has_ci else None,
                         "hi": [p[3] for p in pts] if has_ci else None}],
             "vline": 0.0, "hline": 0.0},
            x_label="Periods since the event" if pulse else "Periods since treatment started",
            y_label=f"Difference in {y} versus period {refp}",
            note=("Points left of the vertical line are before treatment: if they sit near zero, treated and "
                  "comparison units moved alike before treatment. Points to the right show the change after.")))
    return out


def _balance_bar(numbers: dict) -> list[dict]:
    out = []
    t = numbers.get("treatment") or "treatment"
    for chk in numbers.get("checks") or []:
        if not isinstance(chk, dict):
            continue
        m = chk.get("metrics") or {}
        pc = m.get("per_column")
        if not isinstance(pc, dict) or not pc:
            continue
        rows = [(str(k), _f(v)) for k, v in pc.items() if _f(v) is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: -abs(r[1]))
        shown = rows[:MAX_BARS]
        band = _f(m.get("chance_band"))
        data = {"labels": [r[0] for r in shown], "values": [r[1] for r in shown], "ref": 0.0, "errors": None}
        note = (f"How far apart those with and without {t} are on each column, in standard deviations. Zero "
                "means the groups look alike on that column.")
        if band is not None and band > 0:
            data["ref_band"] = [-band, band]
            note += f" The shaded band (plus or minus {_fmt(band)}) is how far apart chance alone would usually put them."
        if len(rows) > len(shown):
            note += f" The {len(shown)} largest of {len(rows)} columns are shown."
        out.append(_chart("bar", f"Differences between those with and without {t}, column by column", data,
                          x_label="Standardized difference (treated minus untreated, in standard deviations)",
                          note=note))
    return out


def charts_for(item: dict, df) -> list[dict]:
    """The standard charts for one evidence item. Never raises; [] when there are none."""
    try:
        if not isinstance(item, dict) or not item.get("ok"):
            return []
        numbers = item.get("numbers") or {}
        if not isinstance(numbers, dict):
            return []
        tool = item.get("tool")
        have_df = isinstance(df, pd.DataFrame) and not df.empty
        builders = []
        if tool == "estimate_effect":
            builders.append(lambda: _forest(numbers))
            design = numbers.get("design")
            if have_df and design == "rd":
                builders.append(lambda: _rd_scatter(numbers, df))
            elif have_df and design == "did":
                builders.append(lambda: _did_line(numbers, df))
            elif have_df and design == "iv":
                builders.append(lambda: _iv_bars(numbers, df))
        elif tool == "design_tests":
            builders.append(lambda: _event_lines(numbers))
        elif tool == "general_checks":
            builders.append(lambda: _balance_bar(numbers))
        out: list[dict] = []
        for b in builders:
            try:
                r = b()
            except Exception:  # noqa: BLE001 - one failed picture must not take the others with it
                continue
            if isinstance(r, dict):
                out.append(r)
            elif isinstance(r, list):
                out += [c for c in r if isinstance(c, dict)]
        return out
    except Exception:  # noqa: BLE001
        return []


# --------------------------------------------------------------------------------- custom charts
def _opt_str(args: dict, key: str) -> Optional[str]:
    v = args.get(key)
    if v is None or (isinstance(v, str) and not v.strip()):
        return None
    if not isinstance(v, str):
        raise ValueError(f"{key} must be a column name or null.")
    return v


def _groups(df: pd.DataFrame, group: Optional[str]):
    """[(name, row mask)] for the series; one series over all rows when group is None."""
    if group is None:
        return [(None, np.ones(len(df), dtype=bool))], ""
    g = df[group]
    top = _top_groups(g)
    if not top:
        raise ValueError(f"The group column '{group}' has no values.")
    out = [(f"{group} = {_label(v)}", (g == v).to_numpy()) for v in top]
    n_all = g.nunique(dropna=True)
    note = f" Only the {len(top)} most common values of {group} are shown." if n_all > len(top) else ""
    return out, note


def _draw_hist(df, x, group, title, note):
    xv = _need_numeric(df, x, "x")
    groups, gnote = _groups(df, group)
    ok = xv.notna().to_numpy()
    allv = xv.to_numpy(float)[ok]
    edges = _bin_edges(allv, HIST_BINS)
    series = []
    for name, m in groups:
        v = xv.to_numpy(float)[m & ok]
        counts, _ = np.histogram(v, bins=edges)
        series.append({"name": name or f"all rows ({len(v)})", "counts": counts.tolist()})
    return _chart("hist", title or f"How {x} is spread" + (f", by {group}" if group else ""),
                  {"bins": edges.tolist(), "series": series}, x_label=x, y_label="Number of rows",
                  note=(note + gnote).strip())


def _draw_scatter(df, x, y, group, title, note):
    if y is None:
        raise ValueError("A scatter chart needs a y column.")
    xv = _need_numeric(df, x, "x")
    yv = _need_numeric(df, y, "y")
    groups, gnote = _groups(df, group)
    ok = (xv.notna() & yv.notna()).to_numpy()
    xa, ya = xv.to_numpy(float), yv.to_numpy(float)
    n_rows = int(sum(int((m & ok).sum()) for _, m in groups))
    points = []
    if n_rows <= MAX_POINTS:
        for name, m in groups:
            mm = m & ok
            points += [{"x": a, "y": b, "group": name} for a, b in zip(xa[mm], ya[mm])]
        extra = ""
    else:
        per = max(5, min(SCATTER_BINS, MAX_POINTS // len(groups)))
        for name, m in groups:
            mm = m & ok
            mx, my, _ = _binned_means(xa[mm], ya[mm], per, quantile=True)
            points += [{"x": a, "y": b, "group": name} for a, b in zip(mx, my)]
        extra = (f" The table has {n_rows} rows, so each dot is the average of rows with similar {x} "
                 f"(about {per} groups of equal size" + (" per series" if group else "") + ").")
    if not points:
        raise ValueError(f"No row has numbers in both '{x}' and '{y}'.")
    return _chart("scatter", title or f"{y} against {x}" + (f", by {group}" if group else ""),
                  {"points": points[:MAX_POINTS], "lines": []}, x_label=x,
                  y_label=y if n_rows <= MAX_POINTS else f"Average {y}", note=(note + extra + gnote).strip())


def _draw_line(df, x, y, group, title, note):
    if y is None:
        raise ValueError("A line chart needs a y column.")
    xv = _need_numeric(df, x, "x")
    yv = _need_numeric(df, y, "y")
    groups, gnote = _groups(df, group)
    ok = (xv.notna() & yv.notna()).to_numpy()
    xa, ya = xv.to_numpy(float), yv.to_numpy(float)
    per_cap = max(2, MAX_POINTS // len(groups))
    series, binned = [], False
    for name, m in groups:
        mm = m & ok
        if not mm.any():
            continue
        s = pd.Series(ya[mm]).groupby(xa[mm]).mean().sort_index()
        if len(s) > per_cap:
            mx, my, _ = _binned_means(xa[mm], ya[mm], min(LINE_BINS, per_cap), quantile=False)
            binned = True
        else:
            mx, my = s.index.tolist(), s.tolist()
        series.append({"name": name or f"average {y}", "x": mx, "y": my, "lo": None, "hi": None})
    if not series:
        raise ValueError(f"No row has numbers in both '{x}' and '{y}'.")
    extra = f" {x} has many values, so it is grouped into bins of equal width." if binned else ""
    return _chart("line", title or f"Average {y} by {x}" + (f", by {group}" if group else ""),
                  {"series": series, "vline": None, "hline": None}, x_label=x, y_label=f"Average {y}",
                  note=(note + extra + gnote).strip())


def _draw_bar(df, x, y, group, title, note):
    if group is not None:
        raise ValueError("A bar chart takes no group column; use a line chart or a histogram to split by group.")
    xs = df[x]
    if xs.nunique(dropna=True) == 0:
        raise ValueError(f"The x column '{x}' has no values.")
    xn = _numeric(xs)
    extra = ""
    if xn is not None and xs.nunique(dropna=True) > MAX_BARS:
        key = pd.qcut(xn, 10, duplicates="drop")
        labels_of = {c: f"{_fmt(c.left)} to {_fmt(c.right)}" for c in key.cat.categories}
        extra = f" {x} has many values, so it is grouped into ranges with about the same number of rows."
    else:
        key, labels_of = xs, None
    if y is None:
        counts = key.value_counts(sort=labels_of is None)
        if labels_of is not None:
            counts = counts.sort_index()
        counts = counts.head(MAX_BARS)
        labels = [labels_of[k] if labels_of else _label(k) for k in counts.index]
        data = {"labels": labels, "values": counts.astype(int).tolist(), "ref": None, "errors": None}
        x_label, ttl = "Number of rows", f"Number of rows for each value of {x}"
    else:
        yv = _need_numeric(df, y, "y")
        d = pd.DataFrame({"k": key, "y": yv}).dropna()
        if d.empty:
            raise ValueError(f"No row has both a value of '{x}' and a number in '{y}'.")
        g = d.groupby("k", observed=True)["y"].agg(["mean", "std", "count"])
        if labels_of is None:
            g = g.sort_values("count", ascending=False)
        g = g.head(MAX_BARS)
        labels = [labels_of[k] if labels_of else _label(k) for k in g.index]
        se = [(_f(s / math.sqrt(n)) if n > 1 else None) for s, n in zip(g["std"], g["count"])]
        data = {"labels": labels, "values": g["mean"].tolist(), "ref": None, "errors": se}
        x_label, ttl = f"Average {y}", f"Average {y} for each value of {x}"
        extra += " Bars show plus or minus one standard error."
    if xs.nunique(dropna=True) > MAX_BARS and labels_of is None:
        extra += f" Only the {MAX_BARS} most common values of {x} are shown."
    return _chart("bar", title or ttl, data, x_label=x_label, y_label=x, note=(note + extra).strip())


def draw(df, args: dict) -> dict:
    """A custom chart from columns. Raises ValueError with a plain message on bad arguments."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("No table is loaded, so there is nothing to draw.")
    if not isinstance(args, dict):
        raise ValueError("The chart arguments must be an object.")
    kind = args.get("kind")
    if kind is not None and (not isinstance(kind, str) or kind.strip().lower() not in KINDS):
        raise ValueError(f"Unknown chart kind {kind!r}. Choose one of: {', '.join(KINDS)}.")
    kind = kind.strip().lower() if isinstance(kind, str) else None
    x = _opt_str(args, "x")
    y = _opt_str(args, "y")
    group = _opt_str(args, "group")
    if x is None:
        raise ValueError("A chart from columns needs an x column.")
    x = _col(df, x, "x")
    if y is not None:
        y = _col(df, y, "y")
    if group is not None:
        group = _col(df, group, "group")
    title = args.get("title")
    title = title.strip() if isinstance(title, str) else ""
    note = args.get("note")
    note = note.strip() if isinstance(note, str) else ""
    if kind is None:
        if y is None:
            kind = "hist" if _numeric(df[x]) is not None else "bar"
        else:
            kind = "scatter" if _numeric(df[x]) is not None and df[x].nunique() > MAX_BARS else "bar"
    if kind == "hist":
        if y is not None:
            raise ValueError("A histogram shows one column (x); leave y null, or use a scatter or line chart.")
        return _draw_hist(df, x, group, title, note)
    if kind == "scatter":
        return _draw_scatter(df, x, y, group, title, note)
    if kind == "line":
        return _draw_line(df, x, y, group, title, note)
    return _draw_bar(df, x, y, group, title, note)
