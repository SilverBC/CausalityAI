"""Reading the input: load a table, type its columns, encode, impute with flags,
resolve and check declared roles, and infer the study design.

Nothing here deletes rows quietly. Duplicates are counted and reported; rows are kept.
"""
from __future__ import annotations

import io
import re
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from .schema import Assumption, ColumnInfo, Design, Prepared, Roles, Status

TRUE_WORDS = {"true", "t", "yes", "y", "1", "treated", "on"}
FALSE_WORDS = {"false", "f", "no", "n", "0", "control", "untreated", "off"}
MAX_ONEHOT_LEVELS = 10
DESIGNS = ["experiment", "did", "iv", "rd", "observational"]


class InputError(ValueError):
    """Raised with a plain-words message the page shows as-is."""


# --------------------------------------------------------------------------- loading
def load_table(content: bytes, filename: str = "data.csv") -> pd.DataFrame:
    name = (filename or "").lower()
    try:
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(content))
        elif name.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
        elif name.endswith(".json"):
            df = pd.read_json(io.BytesIO(content))
        else:
            text = content.decode("utf-8-sig", errors="replace")
            sep = "\t" if name.endswith(".tsv") or text.count("\t") > text.count(",") else None
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python")
    except Exception as e:  # noqa: BLE001
        raise InputError(f"Could not read the file as a table ({type(e).__name__}: {e}).") from e
    if df.shape[0] < 10:
        raise InputError(f"The table has only {df.shape[0]} rows; at least 10 are needed.")
    if df.shape[1] < 2:
        raise InputError("The table needs at least two columns.")
    df.columns = [str(c).strip() for c in df.columns]
    if len(set(df.columns)) != len(df.columns):
        raise InputError("Two or more columns have the same name; please rename them.")
    return df


# --------------------------------------------------------------------------- typing
def _as_bool(s: pd.Series) -> Optional[pd.Series]:
    if s.dtype == bool:
        return s.astype(float)
    if not pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_datetime64_any_dtype(s):
        low = s.dropna().astype(str).str.strip().str.lower()
        vals = set(low.unique())
        if vals and vals <= (TRUE_WORDS | FALSE_WORDS) and len(vals) == 2:
            mapped = s.astype(str).str.strip().str.lower().map(lambda v: 1.0 if v in TRUE_WORDS else (0.0 if v in FALSE_WORDS else np.nan))
            mapped[s.isna()] = np.nan
            return mapped
    return None


def column_kind(s: pd.Series) -> str:
    nn = s.dropna()
    n_unique = nn.nunique()
    if n_unique <= 1:
        return "constant"
    if _as_bool(s) is not None:
        return "binary"
    if pd.api.types.is_numeric_dtype(s):
        if n_unique == 2:
            return "binary"
        if np.allclose(nn, np.round(nn)) and nn.min() >= 0 and n_unique <= 30:
            return "count"
        if np.allclose(nn, np.round(nn)) and n_unique == len(nn) and len(nn) > 20 and _is_sequential(nn):
            return "id"
        return "continuous"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    # object: maybe numeric stored as text, maybe date
    num = pd.to_numeric(nn, errors="coerce")
    if num.notna().mean() > 0.95:
        return column_kind(pd.to_numeric(s, errors="coerce"))
    if nn.astype(str).str.match(r"^\d{4}-\d{1,2}(-\d{1,2})?").mean() > 0.9:
        return "datetime"
    if n_unique <= MAX_ONEHOT_LEVELS:
        return "categorical"
    if n_unique > 0.5 * len(nn):
        avg_len = nn.astype(str).str.len().mean()
        return "text" if avg_len > 25 else "id"
    return "id"   # many labels: usable as a unit/cluster role, too many levels to one-hot


def _is_sequential(nn: pd.Series) -> bool:
    v = np.sort(nn.to_numpy())
    return bool(np.all(np.diff(v) == 1))


def words(name: str) -> list[str]:
    """Whole-word tokens of a column name (snake, kebab, camel, spaces)."""
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [w for w in re.split(r"[^A-Za-z0-9]+", s.lower()) if w]


# --------------------------------------------------------------------------- prepare
def prepare(raw: pd.DataFrame, roles: Optional[Roles] = None) -> Prepared:
    """Type, encode and impute. Role columns are kept as numeric but never imputed
    (treatment/outcome) and never one-hot encoded. Returns a Prepared whose adjustment
    set is computed by `finalize_adjustment_set`."""
    roles = roles or Roles()
    _check_role_columns_exist(raw, roles)
    notes: list[str] = []
    df = pd.DataFrame(index=raw.index)
    infos: list[ColumnInfo] = []
    encoding: dict[str, list[str]] = {}
    role_cols = {roles.treatment, roles.outcome, roles.unit, roles.time, roles.instrument,
                 roles.running, roles.cluster, roles.strata, roles.saturation} - {None}
    no_impute = set(role_cols)   # role columns are never filled in (a filled running variable fakes bunching)

    n_dup = int(raw.duplicated().sum())
    if n_dup:
        notes.append(f"{n_dup} row(s) are exact duplicates of another row. They were kept, not deleted; "
                     "check whether they are real repeated observations or data-entry copies.")

    for col in raw.columns:
        s = raw[col]
        kind = column_kind(s)
        n_missing = int(s.isna().sum())
        out: list[str] = []
        note = ""
        if kind == "binary":
            b = _as_bool(s)
            if b is None:
                num = pd.to_numeric(s, errors="coerce")
                vals = sorted(num.dropna().unique())
                if set(vals) <= {0, 1}:
                    b = num.astype(float)
                else:
                    b = (num == vals[1]).astype(float).where(num.notna())
                    note = f"coded {vals[1]!r} as 1 and {vals[0]!r} as 0"
            df[col] = b
            out = [col]
        elif kind in ("continuous", "count"):
            df[col] = pd.to_numeric(s, errors="coerce").astype(float)
            out = [col]
        elif kind == "datetime":
            d = pd.to_datetime(s, errors="coerce")
            if col in role_cols:
                # periods ranked 0..k-1 so time can be used as a number
                df[col] = d.rank(method="dense") - 1
                note = "dates turned into period numbers (0 = earliest)"
            else:
                df[col] = (d - d.min()).dt.days.astype(float)
                note = "dates turned into days since the earliest date"
            out = [col]
        elif kind == "categorical":
            if col in role_cols:
                codes, uniques = pd.factorize(s)
                df[col] = np.where(codes < 0, np.nan, codes).astype(float)
                note = "labels turned into codes: " + ", ".join(f"{u}={i}" for i, u in enumerate(uniques[:10]))
                out = [col]
            else:
                filled = s.astype(object).where(s.notna(), "(missing)")
                levels = sorted(str(v) for v in s.dropna().unique())
                if s.isna().any():
                    levels.append("(missing)")   # never the dropped reference level
                cat = pd.Categorical(filled.astype(str), categories=levels)
                dummies = pd.get_dummies(cat, prefix=col, prefix_sep="=", drop_first=True, dtype=float)
                dummies.index = df.index
                df = pd.concat([df, dummies], axis=1)
                out = list(dummies.columns)
                note = f"one-hot encoded ({s.nunique()} levels, first level is the reference)"
        elif kind in ("id", "text", "constant"):
            if col in role_cols:
                codes, _ = pd.factorize(s)
                df[col] = np.where(codes < 0, np.nan, codes).astype(float)
                out = [col]
            else:
                note = {"id": "looks like an identifier; not used as a covariate",
                        "text": "free text; not used",
                        "constant": "takes a single value; not used"}[kind]
                if kind == "id" and raw[col].nunique() < 0.5 * raw[col].notna().sum():
                    note = f"label with {raw[col].nunique()} levels (too many to one-hot); usable as a unit or cluster"
        infos.append(ColumnInfo(col, kind, n_missing, int(s.nunique()), out, note))
        encoding[col] = out

    # impute covariates (never treatment / outcome) with missing-value flags
    for info in infos:
        if info.name in no_impute or info.kind == "categorical":
            continue
        for c in info.encoded_as:
            if df[c].isna().any():
                flag = f"{c}__missing"
                df[flag] = df[c].isna().astype(float)
                fill = df[c].median()
                df[c] = df[c].fillna(fill)
                encoding[info.name] = encoding[info.name] + [flag]
                info.encoded_as = encoding[info.name]
                notes.append(f"'{c}': {int(df[flag].sum())} missing value(s) filled with the median "
                             f"({fill:.4g}); a flag column '{flag}' records which rows were filled.")

    for role in ("treatment", "unit", "time", "instrument", "running", "cluster", "strata", "saturation"):
        c = getattr(roles, role)
        if c and c in df and df[c].isna().any():
            notes.append(f"The {role} '{c}' is missing for {int(df[c].isna().sum())} row(s); those rows "
                         "cannot be used for estimation (they are kept in the table and counted).")
    if roles.outcome and df[roles.outcome].isna().any():
        notes.append(f"Outcome '{roles.outcome}' is missing for {int(df[roles.outcome].isna().sum())} row(s); "
                     "this is examined in the missing-outcomes check, not filled in.")

    df = df.copy()
    prep = Prepared(df=df, raw=raw, columns=infos, roles=roles, adjustment_set=[], notes=notes,
                    n_duplicates=n_dup, encoding=encoding)
    finalize_adjustment_set(prep)
    return prep


def _check_role_columns_exist(raw: pd.DataFrame, roles: Roles) -> None:
    for r in ("treatment", "outcome", "unit", "time", "instrument", "running", "cluster", "strata", "saturation"):
        c = getattr(roles, r)
        if c is not None and c not in raw.columns:
            raise InputError(f"The column '{c}' given as the {r} is not in the table. "
                             f"Columns are: {', '.join(map(str, raw.columns[:30]))}.")
    for c in roles.post_treatment:
        if c not in raw.columns:
            raise InputError(f"The column '{c}' marked as measured after treatment is not in the table.")


def finalize_adjustment_set(prep: Prepared) -> list[str]:
    """THE one adjustment set. Excludes treatment, outcome, every role column and anything
    marked (or suspected) post-treatment, including their flags and one-hot children."""
    r = prep.roles
    # strata are a role too: estimators add one dummy per stratum themselves, so the
    # factor codes must not enter as a single linear covariate
    excluded_orig = {r.treatment, r.outcome, r.unit, r.time, r.instrument, r.running,
                     r.cluster, r.saturation, r.strata} - {None}
    excluded_orig |= set(r.post_treatment)
    excluded = set()
    for orig in excluded_orig:
        excluded |= set(prep.encoding.get(orig, [orig]))
        excluded.add(orig)
    adj = []
    for info in prep.columns:
        if info.name in excluded_orig:
            continue
        for c in info.encoded_as:
            if c in excluded or c not in prep.df:
                continue
            if prep.df[c].nunique(dropna=True) <= 1:
                continue
            adj.append(c)
    prep.adjustment_set = adj
    return adj


def mark_post_treatment(prep: Prepared, df_cols: list[str], reason: str) -> None:
    """Add df columns (or originals) to post_treatment and recompute the adjustment set."""
    inv = {c: orig for orig, cols in prep.encoding.items() for c in cols}
    for c in df_cols:
        orig = inv.get(c, c)
        if orig not in prep.roles.post_treatment:
            prep.roles.post_treatment.append(orig)
            prep.notes.append(f"'{orig}' is treated as measured after treatment: {reason}")
    finalize_adjustment_set(prep)


def subset(prep: Prepared, mask: pd.Series, note: str) -> Prepared:
    """A Prepared restricted to rows in `mask` (used for a claimed population)."""
    df = prep.df.loc[mask].copy()
    new = Prepared(df=df, raw=prep.raw.loc[mask].copy(), columns=prep.columns, roles=prep.roles,
                   adjustment_set=[], notes=prep.notes + [note], n_duplicates=prep.n_duplicates,
                   encoding=prep.encoding)
    finalize_adjustment_set(new)
    return new


# --------------------------------------------------------------------------- declarations
def check_declarations(prep: Prepared, design_kind: Optional[str]) -> list[Assumption]:
    """Check each declared role against the data. Conflicts come back as FAILED assumptions."""
    r, df = prep.roles, prep.df
    out: list[Assumption] = []

    def fail(key, name, detail):
        out.append(Assumption(key, name, Status.FAIL, detail, True, "declarations"))

    if design_kind == "experiment" and not r.randomized:
        fail("decl_randomized", "Treatment was randomized",
             "The design was declared an experiment but randomization was not confirmed.")
    if r.treatment and r.treatment in df:
        t = df[r.treatment].dropna()
        if t.nunique() < 2:
            fail("decl_treatment", "Treatment varies", f"'{r.treatment}' takes only one value, so nothing can be compared.")
        elif prep.treatment_is_binary():
            share = t.mean()
            if min(share, 1 - share) * len(t) < 5:
                fail("decl_treatment", "Enough treated and untreated units",
                     f"Only {int(min(share, 1-share)*len(t))} unit(s) in the smaller treatment group.")
    if r.outcome and r.outcome in df and df[r.outcome].dropna().nunique() < 2:
        fail("decl_outcome", "Outcome varies", f"'{r.outcome}' takes only one value.")
    if r.treatment and r.outcome and r.treatment == r.outcome:
        fail("decl_roles", "Treatment and outcome differ", "The same column was given as treatment and outcome.")

    if design_kind == "did" or (r.unit and r.time):
        if not r.time:
            fail("decl_time", "A time column is given", "Difference-in-differences needs a time column.")
        elif df[r.time].nunique() < 2:
            fail("decl_time", "At least two time periods", f"'{r.time}' has a single value.")
        if design_kind == "did" and not r.unit:
            fail("decl_unit", "A unit column is given", "Difference-in-differences needs a unit (or group) column.")
        if r.unit and r.time:
            dup = df.duplicated(subset=[r.unit, r.time]).sum()
            if dup and design_kind == "did":
                out.append(Assumption("decl_panel", "One row per unit and period", Status.CONCERN,
                                      f"{int(dup)} unit-period pair(s) appear more than once; treated as repeated cross-sections.",
                                      True, "declarations"))
            if r.treatment and design_kind == "did":
                ever = df.groupby(r.unit)[r.treatment].max()
                if ever.nunique() < 2 and df.groupby(r.time)[r.treatment].mean().nunique() < 2:
                    fail("decl_did_groups", "Some units are treated and some are not (or not yet)",
                         "Treatment does not vary across units or over time, so there is no comparison group.")
    if design_kind == "iv" or r.instrument:
        if not r.instrument:
            fail("decl_instrument", "An instrument is given", "The instrument design needs an instrument column.")
        elif r.instrument in (r.treatment, r.outcome):
            fail("decl_instrument", "Instrument differs from treatment and outcome",
                 "The instrument is the same column as the treatment or the outcome.")
        elif df[r.instrument].nunique() < 2:
            fail("decl_instrument", "Instrument varies", f"'{r.instrument}' takes a single value.")
    if design_kind == "rd" or r.running:
        if not r.running:
            fail("decl_running", "A running variable is given", "The cutoff design needs a running variable.")
        elif r.cutoff is None:
            fail("decl_cutoff", "A cutoff is given", "The cutoff design needs the cutoff value.")
        else:
            x = df[r.running].dropna()
            below, above = (x < r.cutoff).sum(), (x >= r.cutoff).sum()
            if below < 5 or above < 5:
                fail("decl_cutoff", "Units on both sides of the cutoff",
                     f"Only {int(below)} unit(s) below and {int(above)} at or above the cutoff {r.cutoff:g}.")
            elif r.treatment and prep.treatment_is_binary():
                d = df[[r.running, r.treatment]].dropna()
                side = (d[r.running] >= r.cutoff).astype(float)
                jump = d.loc[side == 1, r.treatment].mean() - d.loc[side == 0, r.treatment].mean()
                if abs(jump) < 0.2:
                    fail("decl_cutoff_jump", "Treatment changes at the cutoff",
                         f"The share treated changes by only {jump:+.2f} at {r.cutoff:g}; the cutoff does not seem to assign treatment.")
                else:
                    best_c, best_acc = find_cutoff(d[r.running], d[r.treatment])
                    decl_acc = max(np.mean(side == d[r.treatment]), np.mean(side != d[r.treatment]))
                    if best_c is not None and best_acc - decl_acc > 0.1:
                        fail("decl_cutoff_fit", "The declared cutoff is where treatment changes",
                             f"Treatment follows a cutoff at {best_c:.4g} ({best_acc:.0%} agreement) much better than "
                             f"the declared {r.cutoff:g} ({decl_acc:.0%}).")
    if r.cluster:
        k = df[r.cluster].nunique()
        if k < 2:
            fail("decl_cluster", "Several clusters", f"'{r.cluster}' has a single value.")
        elif k < 20:
            out.append(Assumption("decl_cluster", "Enough clusters for cluster-robust errors", Status.CONCERN,
                                  f"Only {k} clusters; cluster-robust standard errors can be too small.", True, "declarations"))
    if r.strata and df[r.strata].nunique() < 2:
        fail("decl_strata", "Several strata", f"'{r.strata}' has a single value.")
    if r.treatment and r.treatment in r.post_treatment:
        fail("decl_post", "Treatment is not marked post-treatment", "The treatment column was marked as measured after treatment.")
    return out


# --------------------------------------------------------------------------- design inference
HINTS = {
    "unit": {"id", "unit", "firm", "state", "county", "school", "person", "individual", "household", "country", "city", "region", "store"},
    "time": {"year", "time", "period", "month", "week", "day", "date", "quarter", "wave", "t"},
    "instrument": {"instrument", "lottery", "offer", "offered", "assigned", "assignment", "draw", "voucher", "eligible", "eligibility", "encouragement", "iv", "z"},
    "running": {"score", "running", "age", "distance", "margin", "index", "rank", "income", "vote", "share"},
    "cluster": {"cluster", "school", "village", "class", "classroom", "site", "clinic", "group", "cohort"},
    "strata": {"strata", "stratum", "block", "pair"},
}


def hint(col: str, role: str) -> float:
    return 1.0 if set(words(col)) & HINTS.get(role, set()) else 0.0


def _dense_at(x: pd.Series, c: float) -> bool:
    """True if the column has a normal amount of data near c (no hole at the threshold)."""
    x = x.dropna().to_numpy()
    h = 0.1 * np.std(x)
    below = np.mean((x >= c - h) & (x < c))
    above = np.mean((x >= c) & (x < c + h))
    # compare each side with the average density over the central range
    lo, hi = np.quantile(x, [0.1, 0.9])
    typical = np.mean((x > lo) & (x < hi)) * h / max(hi - lo, 1e-12)
    return below >= 0.4 * typical and above >= 0.4 * typical


def _sharp_at(x: pd.Series, t: pd.Series, c: float) -> bool:
    """True if treatment follows the threshold closely right at the cutoff (within 0.25 SD)."""
    d = pd.DataFrame({"x": x, "t": t}).dropna()
    w = d[(d["x"] - c).abs() < 0.25 * d["x"].std()]
    if len(w) < 20:
        return False
    side = (w["x"] >= c).astype(float)
    if side.nunique() < 2:
        return False
    # the treated share must really jump at the line (a rare treatment agrees with
    # "nobody treated" almost everywhere without any cutoff being involved)
    jump = abs(w.loc[side == 1, "t"].mean() - w.loc[side == 0, "t"].mean())
    return jump >= 0.5 and max(np.mean(side == w["t"]), np.mean(side != w["t"])) >= 0.8


def find_cutoff(x: pd.Series, t: pd.Series) -> tuple[Optional[float], float]:
    """Threshold c maximising agreement of t with 1{x>=c} (or x<c), searched exactly over
    every observed value of x (between its 2nd and 98th percentiles). Returns (c, accuracy)."""
    d = pd.DataFrame({"x": x, "t": t}).dropna()
    if d["x"].nunique() < 10 or d["t"].nunique() != 2:
        return None, 0.0
    d = d.sort_values("x")
    xs, ts = d["x"].to_numpy(), d["t"].to_numpy().astype(float)
    n = len(xs)
    # threshold at a new value xs[k]: rows < k are "below", rows >= k "at or above"
    first = np.r_[True, xs[1:] != xs[:-1]]
    ks = np.flatnonzero(first)
    lo_q, hi_q = np.quantile(xs, [0.02, 0.98])
    ks = ks[(xs[ks] >= lo_q) & (xs[ks] <= hi_q)]
    if len(ks) == 0:
        return None, 0.0
    ones_below = np.r_[0.0, np.cumsum(ts)][ks]
    zeros_below = ks - ones_below
    ones_above = ts.sum() - ones_below
    acc_up = (zeros_below + ones_above) / n                       # treated at/above
    acc_down = (ones_below + (n - ks - ones_above)) / n          # treated below
    acc = np.maximum(acc_up, acc_down)
    i = int(np.argmax(acc))
    return float(xs[ks[i]]), float(acc[i])

def infer_design(prep: Prepared, design_hint: Optional[str] = None) -> Design:
    """Data first, column names only as whole-word hints. Never returns 'experiment'
    unless the user said treatment was randomized."""
    r, df = prep.roles, prep.df
    t = r.treatment
    scores = {k: 0.0 for k in DESIGNS}
    reasons: list[str] = []
    scores["observational"] = 0.5
    proposed: dict[str, dict] = {}
    if r.randomized:
        scores["experiment"] = 0.9
        reasons.append("You said treatment was randomized.")

    cand_cols = [c.name for c in prep.columns if c.encoded_as and c.name not in (t, r.outcome)]
    all_cols = [c.name for c in prep.columns if c.name not in (t, r.outcome) and c.kind != "constant"]

    # panel / DiD: a repeated unit column and a time column, treatment varying over time
    if t and t in df:
        unit_c = r.unit or _best_unit(prep, all_cols)
        time_c = r.time or _best_time(prep, all_cols, unit_c)
        if unit_c and time_c and unit_c != time_c:
            g = pd.DataFrame({"u": prep.raw[unit_c], "tm": prep.raw[time_c], "t": df[t]})
            panel = g.groupby("u")["tm"].nunique().mean() >= 2 and g["u"].nunique() >= 5
            switches = g.groupby("u")["t"].nunique().gt(1).mean() if panel else 0
            never = (g.groupby("u")["t"].max() == 0).mean() if panel else 0
            if panel and switches > 0.05:
                s = 0.6 + 0.2 * (never > 0.05) + 0.1 * (hint(unit_c, "unit") + hint(time_c, "time")) / 2
                scores["did"] = min(s, 0.95)
                reasons.append(f"'{unit_c}' repeats over '{time_c}' and treatment switches on within units.")
                proposed["did"] = {"unit": r.unit or unit_c, "time": r.time or time_c}

        if prep.treatment_is_binary():
            # cutoff: treatment is (nearly) a step function of some continuous column
            best = (None, None, 0.0)
            for c in cand_cols:
                info = next(i for i in prep.columns if i.name == c)
                if info.kind not in ("continuous", "count") or c == r.unit or c == r.time:
                    continue
                if c not in df:
                    continue
                cut, acc = find_cutoff(df[c], df[t])
                if cut is not None and not (_dense_at(df[c], cut) and _sharp_at(df[c], df[t], cut)):
                    # a gap, or a gradual change, at the threshold: more likely a consequence of
                    # treatment (or a strong predictor) than a running variable with a rule
                    continue
                if cut is not None and acc > best[2]:
                    best = (c, cut, acc)
            if r.running and r.cutoff is not None:
                best = (r.running, r.cutoff, find_cutoff(df[r.running], df[t])[1])
            if best[0] and best[2] >= 0.9:
                scores["rd"] = min(0.4 + 0.5 * (best[2] - 0.9) / 0.1 + 0.1 * hint(best[0], "running"), 0.95)
                reasons.append(f"Treatment switches at {best[1]:.4g} on '{best[0]}' for {best[2]:.0%} of rows.")
                proposed["rd"] = {"running": r.running or best[0],
                                  "cutoff": r.cutoff if r.cutoff is not None else best[1]}

            # instrument: binary column that predicts treatment, is not treatment, and is
            # unrelated to the other covariates (looks as-if random)
            inst_cands = [r.instrument] if r.instrument else [
                c for c in cand_cols
                if next(i for i in prep.columns if i.name == c).kind == "binary" and c != r.running]
            best_iv = (None, 0.0)
            for z in inst_cands:
                if z not in df:
                    continue
                d = df[[z, t]].dropna()
                if d[z].nunique() < 2:
                    continue
                fs = d.loc[d[z] == 1, t].mean() - d.loc[d[z] == 0, t].mean()
                agree = np.mean(d[z] == d[t])
                if abs(fs) < 0.15 or agree > 0.97:   # too weak, or just a copy of treatment
                    continue
                others = [c for c in prep.adjustment_set if c not in prep.encoding.get(z, [z])][:20]
                imb = _max_abs_smd(df, z, others)
                s = 0.35 + 0.25 * min(abs(fs) / 0.5, 1) + 0.15 * (imb < 0.1) + 0.2 * hint(z, "instrument")
                if s > best_iv[1]:
                    best_iv = (z, s)
            if best_iv[0] and best_iv[1] >= 0.6:
                scores["iv"] = min(best_iv[1], 0.95)
                reasons.append(f"'{best_iv[0]}' strongly shifts treatment and looks unrelated to the other columns "
                               "(a possible instrument).")
                proposed["iv"] = {"instrument": r.instrument or best_iv[0]}

    # the user's own wording is a hint: it can tip a design the data already supports,
    # never create one the data do not show (and never an experiment)
    if design_hint in ("did", "iv", "rd") and scores.get(design_hint, 0) > 0:
        scores[design_hint] = min(scores[design_hint] + 0.25, 0.97)
        reasons.append(f"Your wording points to {_design_name(design_hint)}, and the data fit it.")
    elif design_hint in ("did", "iv", "rd"):
        reasons.append(f"Your wording points to {_design_name(design_hint)}, but the data show no such structure "
                       "(declare the roles if it is there).")
    elif design_hint == "observational" and not r.randomized:
        scores["observational"] = max(scores["observational"], 0.98)
        reasons.append("Your wording says the data are observational.")
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    kind, top = ranked[0]
    # roles found by inference are applied only for the design that wins
    for role, val in proposed.get(kind, {}).items():
        setattr(r, role, val)
        if role != "cutoff":
            ensure_role_column(prep, val)
    scores = {k: float(v) for k, v in scores.items()}
    second_kind, second = ranked[1]
    ambiguous = (top - second) < 0.1 and second >= 0.6 and "observational" not in (kind, second_kind)
    question = None
    if ambiguous:
        question = (f"The data fit two designs about equally well: {_design_name(kind)} and "
                    f"{_design_name(second_kind)}. Which one describes how treatment was assigned?")
    if kind == "observational":
        reasons.append("No design-based source of variation was found, so the data are treated as observational.")
        if not r.randomized and t and prep.treatment_is_binary():
            reasons.append("Even if groups look balanced, this is not called an experiment unless you say treatment was randomized.")
    return Design(kind=kind, declared=False, scores=scores, ambiguous=ambiguous, question=question, reasons=reasons)


def _design_name(k: str) -> str:
    return {"experiment": "a randomized experiment", "did": "difference-in-differences", "iv": "an instrument",
            "rd": "a cutoff (regression discontinuity)", "observational": "observational"}[k]


def _best_unit(prep: Prepared, cols: list[str]) -> Optional[str]:
    df = prep.df
    best, score = None, 0.0
    for c in cols:
        info = next(i for i in prep.columns if i.name == c)
        src = prep.raw[c]
        k = src.nunique()
        if k < 5 or k > len(src) / 2:
            continue
        s = 0.5 + 0.5 * hint(c, "unit") + (0.2 if info.kind == "id" else 0)
        if s > score:
            best, score = c, s
    return best


def _best_time(prep: Prepared, cols: list[str], unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    best, score = None, 0.0
    for c in cols:
        if c == unit:
            continue
        info = next(i for i in prep.columns if i.name == c)
        if info.kind not in ("count", "continuous", "datetime", "binary"):
            continue
        k = prep.raw[c].nunique()
        if k < 2 or k > 500:
            continue
        per_unit = prep.raw.groupby(unit)[c].nunique().mean()
        if per_unit < 2:
            continue
        s = 0.5 * (per_unit / k) + 0.5 * hint(c, "time")
        if s > score:
            best, score = c, s
    return best


def ensure_role_column(prep: Prepared, col: Optional[str]) -> None:
    """Make sure a role column inferred after preparation exists in df as a single numeric
    column (codes for labels, period numbers for dates) and leaves the adjustment set."""
    if not col:
        return
    if col not in prep.df or prep.encoding.get(col) != [col]:
        for c in prep.encoding.get(col, []):
            if c in prep.df and c != col:
                prep.df.drop(columns=c, inplace=True)
        s = prep.raw[col]
        info = next(i for i in prep.columns if i.name == col)
        if info.kind == "datetime":
            prep.df[col] = pd.to_datetime(s, errors="coerce").rank(method="dense") - 1
        elif pd.api.types.is_numeric_dtype(s):
            prep.df[col] = s.astype(float)
        else:
            codes, _ = pd.factorize(s)
            prep.df[col] = np.where(codes < 0, np.nan, codes).astype(float)
        prep.encoding[col] = [col]
        info.encoded_as = [col]
    finalize_adjustment_set(prep)


def _max_abs_smd(df: pd.DataFrame, group: str, cols: list[str]) -> float:
    g = df[group]
    m = 0.0
    for c in cols:
        a, b = df.loc[g == 1, c], df.loc[g == 0, c]
        sd = np.sqrt((a.var() + b.var()) / 2)
        if sd and np.isfinite(sd) and sd > 0:
            m = max(m, abs(a.mean() - b.mean()) / sd)
    return m


def resolve_design(prep: Prepared, declared: Optional[str], hint: Optional[str] = None) -> Design:
    """Use the declared design if given (checked against the data), else infer it."""
    if declared:
        declared = declared.lower().strip()
        aliases = {"rct": "experiment", "randomized": "experiment", "diff-in-diff": "did",
                   "difference-in-differences": "did", "instrument": "iv", "instrumental": "iv",
                   "cutoff": "rd", "regression discontinuity": "rd", "rdd": "rd", "obs": "observational"}
        declared = aliases.get(declared, declared)
        if declared not in DESIGNS:
            raise InputError(f"Unknown design '{declared}'. Choose one of: experiment, did, iv, rd, observational.")
        if declared == "experiment":
            prep.roles.randomized = True   # declaring an experiment = saying treatment was randomized
        if declared == "rd" and prep.roles.running and prep.roles.cutoff is None and prep.roles.treatment:
            prep.roles.cutoff, _ = find_cutoff(prep.df[prep.roles.running], prep.df[prep.roles.treatment])
        d = Design(kind=declared, declared=True, reasons=["You declared this design."])
    else:
        d = infer_design(prep, hint)
    d.conflicts = check_declarations(prep, d.kind)
    finalize_adjustment_set(prep)
    return d


# --------------------------------------------------------------------------- small helpers
def smd(a: pd.Series, b: pd.Series) -> float:
    sd = np.sqrt((np.nanvar(a, ddof=1) + np.nanvar(b, ddof=1)) / 2)
    return float((np.nanmean(a) - np.nanmean(b)) / sd) if sd > 0 else 0.0


def normal_ci(est: float, se: float, level: float) -> tuple[float, float]:
    z = stats.norm.ppf(0.5 + level / 2)
    return (est - z * se, est + z * se)


def summary(prep: Prepared) -> dict:
    return {
        "n_rows": int(len(prep.raw)),
        "n_columns": int(prep.raw.shape[1]),
        "n_duplicates": prep.n_duplicates,
        "columns": prep.columns,
        "adjustment_set": prep.adjustment_set,
        "notes": prep.notes,
        "roles": prep.roles,
    }
