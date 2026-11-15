"""Evidence tools the model calls (part A of ORCHESTRATOR.md).

TOOL_SPECS                      Anthropic tool definitions for contract.EVIDENCE_TOOLS.
call(name, args, df, id)        runs one tool and returns contract.evidence_item(...). Never raises.

Each result has `lines` (plain sentences with the numbers, for the model to read; no pass/fail
words) and `numbers` (plain JSON for the floor rules and the page). Nothing here picks a design,
reads the claim, or labels a result as good or bad: the model judges.
"""
from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))  # quiet joblib's core count probe

import numpy as np
import pandas as pd

from .. import agent, data
from ..schema import Section, to_jsonable
from . import contract

MAX_ITEMS = 400          # no list or dict in `numbers` longer than this
MAX_LINES_EXPLORE = 30

# --------------------------------------------------------------------------------- tool specs
_COLS = {"type": ["array", "null"], "items": {"type": "string"},
         "description": ("Columns to adjust for. Leave out or null to use every column that is not a role column; "
                         "an empty list adjusts for nothing. Leave out columns that treatment could have changed.")}
_ROLES = {
    "type": "object",
    "description": ("Which column plays which part. treatment and outcome are needed. unit and time for "
                    "panel data (did), instrument for iv, running (and optionally cutoff) for rd, cluster when "
                    "treatment was assigned to groups, strata when it was assigned within strata."),
    "properties": {**{k: {"type": "string", "description": f"name of the {k} column"} for k in contract.ROLE_KEYS},
                   "cutoff": {"type": "number", "description": "the threshold on the running column (rd only); "
                              "left out = found from the data"}},
    "required": ["treatment", "outcome"],
    "additionalProperties": False,
}
_DESIGN = {"type": "string", "enum": list(contract.DESIGNS),
           "description": ("experiment = treatment was assigned at random (only if the user or the description says "
                           "so); did = units observed over time, some start treatment later (difference-in-differences); "
                           "iv = an instrument shifts treatment; rd = treatment switches at a threshold of a running "
                           "column; observational = none of these, compare treated and untreated given covariates.")}

TOOL_SPECS: list[dict] = [
    {"name": "table_facts",
     "description": ("Facts about the uploaded table with no conclusions: number of rows and columns, duplicate rows, "
                     "which columns identify a row, each column's type, distinct values, missing share and range, "
                     "time-like columns, constant or near-identical columns, and sample rows. Call this first."),
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "general_checks",
     "description": ("Three checks of a direct treated-versus-untreated comparison: covariate balance (joint "
                     "permutation test and standardized differences), overlap of propensity scores, and missing "
                     "outcomes (overall and by group, with bounds). Each line says when the check applies. Works for "
                     "any 0/1 or numeric treatment. Hidden-confounding sensitivity is the separate sensitivity tool."),
     "input_schema": {"type": "object", "properties": {
         "treatment": {"type": "string"}, "outcome": {"type": "string"}, "covariates": _COLS},
         "required": ["treatment", "outcome"], "additionalProperties": False}},
    {"name": "structure_facts",
     "description": ("How treatment is laid out in the table, without naming a design: share treated, columns within "
                     "which treatment changes (a sign of panel data), whether treatment switches at a threshold of a "
                     "numeric column, and, for each proposed instrument, how strongly it predicts treatment "
                     "(first-stage F). Use it to see which designs the data could support."),
     "input_schema": {"type": "object", "properties": {
         "treatment": {"type": "string"}, "outcome": {"type": "string"},
         "instruments": {"type": ["array", "null"], "items": {"type": "string"},
                         "description": "columns to try as instruments"},
         "covariates": _COLS},
         "required": ["treatment", "outcome"], "additionalProperties": False}},
    {"name": "design_tests",
     "description": ("The tests that belong to one design, run on the roles you give: for did, trends before "
                     "treatment, a placebo in time and a functional-form test; for iv, first-stage strength, "
                     "covariate balance across the instrument and a Durbin-Wu-Hausman comparison; for rd, a density "
                     "test for sorting at the cutoff and jumps in covariates at the cutoff; for experiment and "
                     "observational, covariate balance. Also lists any mismatch between the declared roles and the "
                     "data, and the assumptions no data can test."),
     "input_schema": {"type": "object", "properties": {"design": _DESIGN, "roles": _ROLES, "covariates": _COLS},
                      "required": ["design", "roles"], "additionalProperties": False}},
    {"name": "estimate_effect",
     "description": ("The effect of treatment on the outcome under the design you give: method, what it estimates "
                     "and for whom, the estimate, standard error, p-value and intervals at 90, 95, 98 and 99%, plus "
                     "other methods for comparison. It may decline to give a number and say why."),
     "input_schema": {"type": "object", "properties": {"design": _DESIGN, "roles": _ROLES, "covariates": _COLS},
                      "required": ["design", "roles"], "additionalProperties": False}},
    {"name": "sensitivity",
     "description": ("How strong an unmeasured confounder would have to be to remove the effect: robustness value "
                     "(Cinelli and Hazlett), E-value (VanderWeele and Ding) and what the estimate would be if a "
                     "hidden confounder were as strong as each of the measured covariates most related to treatment. "
                     "It shows how strong hidden confounding would have to be, never whether there is any."),
     "input_schema": {"type": "object", "properties": {"design": _DESIGN, "roles": _ROLES, "covariates": _COLS},
                      "required": ["design", "roles"], "additionalProperties": False}},
    {"name": "explore",
     "description": ("Causal discovery over the numeric columns (FCI with stability over subsamples, cross-checked "
                     "with PC and LiNGAM): which links appear stable and which way they may point. Patterns the data "
                     "allow, not proven effects. Slow on wide tables; use when the claim names no clear treatment "
                     "or to look for other paths."),
     "input_schema": {"type": "object", "properties": {}, "additionalProperties": False}},
]

# --------------------------------------------------------------------------------- helpers


class _Bad(Exception):
    """Bad arguments: the message is shown to the model as is."""


def _jsonable(x: Any, depth: int = 0) -> Any:
    """Plain JSON values only: no numpy types, no NaN/inf, small lists."""
    if depth > 12:
        return None
    if x is None or isinstance(x, (bool, str)):
        return x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        return f if math.isfinite(f) else None
    if isinstance(x, dict):
        items = list(x.items())[:MAX_ITEMS]
        return {str(k): _jsonable(v, depth + 1) for k, v in items}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v, depth + 1) for v in list(x)[:MAX_ITEMS]]
    if isinstance(x, np.ndarray):
        return [_jsonable(v, depth + 1) for v in x.ravel().tolist()[:MAX_ITEMS]]
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return None
    y = to_jsonable(x)
    if y is x:
        return str(x)
    return _jsonable(y, depth + 1)


def _num(x) -> Optional[float]:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _pair(v) -> Optional[list]:
    if v is None:
        return None
    try:
        lo, hi = _num(v[0]), _num(v[1])
    except (TypeError, IndexError, KeyError):
        return None
    return [lo, hi] if lo is not None and hi is not None else None


def _col(df: pd.DataFrame, v, what: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise _Bad(f"The {what} must be given as a column name.")
    if v not in df.columns:
        near = [c for c in map(str, df.columns) if c.lower() == v.lower()]
        hint = f" Did you mean '{near[0]}'?" if near else ""
        raise _Bad(f"The {what} column '{v}' is not in the table.{hint}")
    return v


def _col_list(df: pd.DataFrame, v, what: str) -> Optional[list[str]]:
    if v is None:
        return None
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, (list, tuple)):
        raise _Bad(f"The {what} must be a list of column names, or left out.")
    out = []
    for c in v:
        out.append(_col(df, c, what[:-1] if what.endswith("s") else what))
    return list(dict.fromkeys(out))


def _design(v) -> str:
    d = v.strip().lower() if isinstance(v, str) else None
    if d not in contract.DESIGNS:
        raise _Bad(f"Unknown design {v!r}. Choose one of: {', '.join(contract.DESIGNS)}.")
    return d


def _roles(df: pd.DataFrame, v) -> dict:
    if not isinstance(v, dict):
        raise _Bad("roles must be an object such as {\"treatment\": \"<column>\", \"outcome\": \"<column>\"}.")
    allowed = set(contract.ROLE_KEYS) | {"cutoff"}
    extra = [k for k in v if k not in allowed]
    if extra:
        raise _Bad(f"Unknown role(s): {', '.join(map(str, extra))}. Roles can be: "
                   f"{', '.join(contract.ROLE_KEYS)}, cutoff.")
    out: dict = {}
    for k in contract.ROLE_KEYS:
        if v.get(k) not in (None, ""):
            out[k] = _col(df, v[k], k)
    for k in ("treatment", "outcome"):
        if k not in out:
            raise _Bad(f"roles needs a {k} column.")
    if out["treatment"] == out["outcome"]:
        raise _Bad("The treatment and the outcome must be different columns.")
    if v.get("cutoff") not in (None, ""):
        c = _num(v["cutoff"])
        if c is None:
            raise _Bad(f"The cutoff must be a number; got {v['cutoff']!r}.")
        out["cutoff"] = c
    return out


def _metrics(sec: Section) -> dict:
    return _jsonable({k: v for k, v in (sec.metrics or {}).items() if k not in agent._SKIP})


def _untestable(assumptions) -> list[str]:
    return list(dict.fromkeys(a.name for a in (assumptions or []) if not getattr(a, "testable", True)))


# Phrases from the older modules (agent.py, estimate.py, design_assess.py) that attach a code-made
# verdict or a threshold reading to a number. Every evidence line goes through these before the model
# sees it, so lines state numbers and leave the judging to the model.
_REWORD: list[tuple[re.Pattern, str]] = [
    (re.compile(r"It is a sensitivity number, never a pass\."),
     "It is a sensitivity number: it shows how strong hidden confounding would have to be, not whether there is any."),
    (re.compile(r"\s*\(below 10 is weak\)"), "; the usual rule of thumb is 10"),
    (re.compile(r"The instrument is weak \(first-stage F = ([^)<]*?)\s*<\s*10\); trust the Anderson-Rubin interval, "
                r"not the 2SLS interval\."),
     r"First-stage F = \1; the usual rule of thumb is 10. The Anderson-Rubin set is valid however strong or weak the "
     r"instrument is; the 2SLS interval is not."),
    (re.compile(r"\s*\(the instrument is too weak to say anything\)"), ""),
    (re.compile(r"\s*\(the instrument is weak\)"), ""),
    (re.compile(r"\(the instrument model itself looks rejected\)"),
     "(no effect value is consistent with the instrument model at this level)"),
]

# Test titles from design_assess that state the hoped-for result; shown under neutral names.
_TITLES = {
    "design_first_stage": "First-stage strength",
    "design_instrument_balance": "Covariate balance across instrument values",
    "design_bunching": "Sorting around the cutoff (density test)",
    "design_covariate_jumps": "Covariate jumps at the cutoff",
    "design_pretrends": "Trends before treatment",
    "design_placebo_time": "Placebo test before treatment started",
}


def _reword(text: str) -> str:
    for pat, rep in _REWORD:
        text = pat.sub(rep, text)
    return text


_applies = _reword


def _prep_lines(prep) -> list[str]:
    return [f"Data preparation: {agent._short(str(n), 220)}" for n in (prep.notes or [])[:3]]


# --------------------------------------------------------------------------------- tools
def _table_facts(df: pd.DataFrame, args: dict):
    card = agent.facts_card(df)
    lines = [ln for ln in card.split("\n") if ln.strip()]
    cols = []
    for c in list(df.columns)[:MAX_ITEMS]:
        s = df[c]
        cols.append({"name": str(c), "kind": data.column_kind(s), "distinct": int(s.nunique()),
                     "missing_share": float(s.isna().mean())})
    return lines, {"n_rows": int(len(df)), "n_columns": int(df.shape[1]),
                   "n_duplicate_rows": int(df.duplicated().sum()), "columns": cols}


def _general_checks(df: pd.DataFrame, args: dict):
    # Balance, overlap and missing outcomes. The hidden-confounding sensitivity numbers are left to the
    # `sensitivity` tool, so every sensitivity reading is its own evidence item (floor rule 1 reads it there).
    from ..checks import attrition, balance, overlap
    t = _col(df, args.get("treatment"), "treatment")
    y = _col(df, args.get("outcome"), "outcome")
    if t == y:
        raise _Bad("The treatment and the outcome must be different columns.")
    cov = _col_list(df, args.get("covariates"), "covariates")
    prep, des = agent._prepare(df, "observational", {"treatment": t, "outcome": y}, cov)
    runs = [lambda: balance.run(prep, des, None), lambda: overlap.run(prep, des, None),
            lambda: attrition.run(prep, des, None)]
    adj = prep.adjustment_set
    lines = [f"Covariates used ({len(adj)} columns after encoding): " + (", ".join(adj[:40]) or "none")
             + ("" if len(adj) <= 40 else ", ...")]
    checks, untest = [], []
    for i, ((name, method, applies), fn) in enumerate(zip(agent.GENERAL, runs), 1):
        try:
            sec = fn()
            res = agent._line(sec)
            checks.append({"key": sec.key, "title": sec.title, "metrics": _metrics(sec),
                           "error": sec.error})
            untest += _untestable(sec.assumptions)
        except Exception as e:  # noqa: BLE001
            res = f"could not run: {type(e).__name__}: {e}"
            checks.append({"key": name.lower().replace(" ", "_"), "title": name, "metrics": {}, "error": res})
        lines.append(f"Check #{i}: {name} ({method}) - {res} - Applies when: {_applies(applies)}")
    lines.append("Hidden-confounding sensitivity (robustness value, E-value) is not part of these checks; "
                 "it comes from the sensitivity tool.")
    return lines, {"treatment": t, "outcome": y, "covariates": cov, "adjustment_set": _jsonable(adj),
                   "checks": checks, "untestable": list(dict.fromkeys(untest))}


def _structure_facts(df: pd.DataFrame, args: dict):
    t = _col(df, args.get("treatment"), "treatment")
    y = _col(df, args.get("outcome"), "outcome")
    if t == y:
        raise _Bad("The treatment and the outcome must be different columns.")
    inst = _col_list(df, args.get("instruments"), "instruments")
    cov = _col_list(df, args.get("covariates"), "covariates")
    if cov is None:
        cov = [c for c in df.columns if c not in {t, y} | set(inst or [])]
    lines = agent.structure_facts(df, t, y, inst, cov)
    return lines, {"treatment": t, "outcome": y, "instruments": inst, "covariates": _jsonable(cov)}


# statistic and p-value of each test card, by key: (statistic path, p-value path); paths go into metrics.
_TEST_STATS = {
    "balance": (("joint_stat",), ("joint_p",)),
    "design_pretrends": (("pretrend_F",), ("pretrend_p",)),
    "design_first_stage": (("robust_F",), None),
    "design_instrument_balance": (("balance_F",), ("balance_p",)),
    "design_bunching": (("mccrary", "log_density_jump"), ("mccrary", "p")),
    "design_covariate_jumps": (None, ("bonferroni_p",)),
    "design_placebo_time": (("placebo_effect",), ("p",)),
    "design_endogeneity": (("t",), ("p",)),
    "design_functional_form": (None, ("p",)),
    "design_event_study": (("bmp_stat",), ("bmp_p",)),
}


def _dig(m: dict, path) -> Optional[float]:
    if not path:
        return None
    v: Any = m
    for k in path:
        if not isinstance(v, dict):
            return None
        v = v.get(k)
    return _num(v)


def _test_row(sec: Section) -> dict:
    m = sec.metrics or {}
    sp, pp = _TEST_STATS.get(sec.key, (None, ("p_value",)))
    p = _dig(m, pp)
    if p is None and sec.key == "design_bunching":
        p = _dig(m, ("binomial", "p"))
    return {"key": sec.key, "title": _TITLES.get(sec.key, sec.title), "statistic": _dig(m, sp), "p_value": p,
            "metrics": _metrics(sec), "error": sec.error}


def _design_args(df: pd.DataFrame, args: dict):
    design = _design(args.get("design"))
    roles = _roles(df, args.get("roles"))
    cov = _col_list(df, args.get("covariates"), "covariates")
    return design, roles, cov


def _design_tests(df: pd.DataFrame, args: dict):
    from .. import design_assess
    from ..checks import balance
    design, roles, cov = _design_args(df, args)
    prep, des = agent._prepare(df, design, roles, cov)
    lines, conflicts = [], []
    for a in des.conflicts:
        conflicts.append(f"{a.name}. {agent._short(a.detail, 260)}".strip())
        lines.append(f"Declared design or roles do not match the data: {a.name}. {agent._short(a.detail, 260)}")
    if not des.conflicts:
        lines.append("The declared roles exist and have the layout this design needs. This does not show the design "
                     "is the right one.")
    assumptions, sections = design_assess.assess(prep, des, None)
    sections = [s for s in sections if s is not None]
    if design in ("experiment", "observational"):
        sections = [balance.run(prep, des, None)] + list(sections)
    tests = []
    for s in sections:
        lines.append(f"Test of '{_TITLES.get(s.key, s.title)}': {agent._line(s)}")
        tests.append(_test_row(s))
    if not sections:
        lines.append("This design has no tests of its own; see the general checks.")
    untest = _untestable(list(assumptions) + [a for s in sections for a in (s.assumptions or [])])
    if untest:
        lines.append("Not tested, because no data can test them: " + "; ".join(untest) + ".")
    lines += _prep_lines(prep)
    return lines, {"design": design, "roles": roles, "covariates": cov, "tests": tests, "conflicts": conflicts,
                   "untestable": untest, "cutoff_used": _num(prep.roles.cutoff)}


def _ci_levels(e) -> dict:
    lv = (e.extra or {}).get("ci_levels") or {}
    out = {k: _pair(lv.get(k)) for k in ("90", "95", "98", "99")}
    if out["90"] is None:
        out["90"] = _pair(e.ci90)
    if out["95"] is None:
        out["95"] = _pair(e.ci95)
    est, se = _num(e.estimate), _num(e.se)
    if est is not None and se is not None and se > 0:
        from scipy import stats
        for k in out:
            if out[k] is None:
                z = float(stats.norm.ppf(0.5 + int(k) / 200))
                out[k] = [est - z * se, est + z * se]
    if e.refused or est is None:
        out = {k: None for k in out}
    return out


_LEVELS = ("90", "95", "98", "99")


def _ar_sets(e) -> dict:
    """The Anderson-Rubin sets at each level as plain lists of [lo, hi] pieces (None for an infinite end)."""
    ex = e.extra or {}
    out = {}
    for k in _LEVELS:
        s = ex.get(f"ar_ci{k}")
        out[k] = ([[_num(p[0]), _num(p[1])] for p in s if isinstance(p, (list, tuple)) and len(p) == 2]
                  if isinstance(s, (list, tuple)) else None)
    return out


def _ar_interval(pieces) -> Optional[list]:
    """One bounded interval, or None when the set is missing, empty, unbounded or in pieces."""
    if not pieces or len(pieces) != 1:
        return None
    lo, hi = pieces[0]
    return [lo, hi] if lo is not None and hi is not None else None


def _describe_pieces(pieces) -> str:
    if pieces is None:
        return "not computed"
    if not pieces:
        return "empty (no effect value is consistent with the instrument model at this level)"
    if len(pieces) == 1 and pieces[0][0] is None and pieces[0][1] is None:
        return "unbounded: every effect value is consistent with the data"
    txt = " or ".join(f"[{'-inf' if a is None else agent._fmt(a)}, {'inf' if b is None else agent._fmt(b)}]"
                      for a, b in pieces)
    return ("unbounded: " + txt) if any(a is None or b is None for a, b in pieces) else txt


def _design_untestable(prep, des, design: str) -> list[str]:
    """The identifying assumptions of `design` that no data can test (design_assess), plus, for an
    observational comparison, that there are no hidden confounders (checks/sensitivity's wording)."""
    from .. import design_assess
    assumptions = []
    try:
        assumptions += list(design_assess.assess(prep, des, None)[0])
    except Exception:  # noqa: BLE001
        pass
    if design == "observational":
        try:
            from ..checks import sensitivity
            assumptions.append(sensitivity._main_assumption("observational", None))
        except Exception:  # noqa: BLE001
            pass
    return _untestable(assumptions)


def _estimate_effect(df: pd.DataFrame, args: dict):
    from .. import estimate as est_mod
    design, roles, cov = _design_args(df, args)
    prep, des = agent._prepare(df, design, roles, cov)
    e = est_mod.estimate(prep, des, None)
    ci = _ci_levels(e)
    ci_wald = ar = None
    if design == "iv":
        # The 2SLS (Wald) interval is not valid when the instrument is weak; the Anderson-Rubin set is valid
        # at any strength. ci holds the Anderson-Rubin interval (None when the set is unbounded, empty or in
        # pieces, so no label can rest on it); the Wald interval is kept apart for comparison.
        ci_wald = ci
        ar = _ar_sets(e)
        ci = {k: (None if (e.refused or e.estimate is None) else _ar_interval(ar[k])) for k in _LEVELS}
    untest = _design_untestable(prep, des, design)
    sd = _num(e.outcome_sd)
    if sd is None:
        yv = pd.to_numeric(prep.complete()[prep.roles.outcome], errors="coerce").dropna()
        sd = _num(yv.std(ddof=1)) if len(yv) > 1 else None
    refused = bool(e.refused or e.estimate is None)
    comps = [{"method": str(c.get("method")), "estimate": _num(c.get("estimate")), "ci95": _pair(c.get("ci95"))}
             for c in (e.comparisons or [])]
    notes = [agent._short(_reword(str(n)), 300) for n in ((e.extra or {}).get("notes") or [])][:6]
    numbers = {"design": design, "roles": roles, "covariates": cov, "method": e.method, "target": e.target,
               "scope": e.scope, "n": int(e.n or 0), "estimate": None if refused else _num(e.estimate),
               "se": None if refused else _num(e.se), "p_value": None if refused else _num(e.p_value), "ci": ci,
               "refused": refused, "refusal_reason": e.refusal_reason if refused else None,
               "outcome_sd": sd, "baseline_mean": _num(e.baseline_mean), "comparisons": _jsonable(comps),
               "adjustment_set": _jsonable(prep.adjustment_set), "cutoff_used": _num(prep.roles.cutoff),
               "anderson_rubin_95": _reword(str((e.extra or {}).get("ar_ci95_text"))) if ar else None,
               "untestable": untest, "notes": notes}
    if design == "iv":
        numbers["ci_wald"] = ci_wald
        numbers["anderson_rubin"] = ar
        numbers["first_stage_F"] = _num((e.extra or {}).get("first_stage_F"))
        numbers["ci_method"] = "Anderson-Rubin"
    if refused:
        lines = [f"No estimate ({e.method}): {e.refusal_reason or 'the estimator declined to give a number.'}"]
    else:
        lines = [f"Method: {e.method}. Target: {e.target}; speaks for: {e.scope}. n = {e.n}.",
                 f"Estimate {agent._fmt(e.estimate)}, standard error {agent._fmt(e.se)}, p = {agent._fmt(e.p_value)}."]
        if design == "iv":
            lines.append("Anderson-Rubin sets (valid however strong or weak the instrument is; these are the "
                         "intervals in ci): " + "; ".join(f"{k}% {_describe_pieces(ar[k])}" for k in _LEVELS) + ".")
            if any(ci_wald.values()):
                lines.append("2SLS Wald intervals (estimate plus or minus a multiple of the standard error): "
                             + "; ".join(f"{k}% {agent._fmt(v[0])} to {agent._fmt(v[1])}"
                                         for k, v in ci_wald.items() if v) + ".")
            if numbers["first_stage_F"] is not None:
                lines.append(f"First-stage F = {agent._fmt(numbers['first_stage_F'])}.")
        else:
            lines.append("Intervals: " + "; ".join(f"{k}% {agent._fmt(v[0])} to {agent._fmt(v[1])}"
                                                   for k, v in ci.items() if v))
        for c in comps[:4]:
            c95 = c["ci95"]
            lines.append(f"Other method: {c['method']}: {agent._fmt(c['estimate'])}"
                         + (f" (95% {agent._fmt(c95[0])} to {agent._fmt(c95[1])})" if c95 else ""))
    if sd is not None:
        lines.append(f"Standard deviation of the outcome in the rows used: {agent._fmt(sd)}.")
    if design == "rd" and prep.roles.cutoff is not None:
        lines.append(f"Cutoff used: {agent._fmt(float(prep.roles.cutoff))}.")
    for n in notes[:3]:
        lines.append("Note: " + agent._short(n, 220))
    if untest:
        lines.append("Assumed by this design, because no data can test it: " + "; ".join(untest) + ".")
    lines += _prep_lines(prep)
    return lines, numbers


def _sensitivity(df: pd.DataFrame, args: dict):
    from .. import estimate as est_mod
    from ..checks import sensitivity
    design, roles, cov = _design_args(df, args)
    prep, des = agent._prepare(df, design, roles, cov)
    e = est_mod.estimate(prep, des, None)
    sec = sensitivity.run(prep, des, None, e)
    m = sec.metrics or {}
    untest = _untestable(sec.assumptions)
    if sec.error:
        lines = ["The sensitivity analysis could not run: " + agent._short(sec.error)]
    else:
        lines = [agent._short(sec.summary, 1400)]
        lines.append("Numbers: " + agent._line(sec))
    if untest:
        lines.append("Not tested, because no data can test it: " + "; ".join(untest) + ".")
    bench = [{k: row.get(k) for k in ("covariate", "k", "r2dz_x", "r2yz_dx", "adjusted_estimate", "adjusted_lo",
                                      "adjusted_hi")} for row in (m.get("benchmarks") or [])]
    numbers = {"design": design, "roles": roles, "covariates": cov,
               "robustness_value": _num(m.get("rv_q")), "robustness_value_alpha": _num(m.get("rv_q_alpha")),
               "e_value": _num(m.get("evalue_estimate")), "e_value_ci": _num(m.get("evalue_ci")),
               "e_value_method": m.get("evalue_method"), "e_value_source": m.get("evalue_source"),
               "regression_estimate": _num(m.get("regression_estimate")), "regression_se": _num(m.get("regression_se")),
               "partial_r2_treatment_outcome": _num(m.get("partial_r2_treatment_outcome")),
               "n": m.get("n"), "benchmarks": _jsonable(bench), "untestable": untest, "error": sec.error}
    return lines, numbers


def _explore(df: pd.DataFrame, args: dict):
    from .. import explore
    from ..schema import Roles
    prep = data.prepare(df, Roles())
    sec = explore.run(prep)
    m = sec.metrics or {}
    untest = _untestable(sec.assumptions)
    if sec.error:
        lines = ["The exploration could not run: " + agent._short(sec.error)]
    else:
        drawn = m.get("drawn_edges") or []
        n_dir = sum(1 for x in drawn if x.get("type") == "directed")
        lines = [f"Variables used ({len(m.get('variables') or [])}): " + ", ".join(map(str, m.get("variables") or []))
                 + f". Rows used: {m.get('n_rows')}.",
                 f"Links drawn: {len(drawn)} ({n_dir} with a stable direction, {len(drawn) - n_dir} with an uncertain "
                 f"direction); links that did not hold across subsamples: {len(m.get('unstable_edges') or [])}. "
                 "These are patterns the data allow, not proven effects."]
        pairs = [p for p in (m.get("pairs") or []) if p.get("drawn")]
        pairs.sort(key=lambda p: -((p.get("stability") or {}).get("adjacency") or 0))
        for p in pairs[:MAX_LINES_EXPLORE]:
            lines.append(agent._short(str(p.get("reading")), 300))
        if m.get("left_out"):
            lines.append("Left out: " + "; ".join(f"{x.get('column')} ({x.get('reason')})"
                                                   for x in m["left_out"][:12]) + ".")
        if m.get("stability_rule"):
            lines.append("Rule for drawing a link: " + str(m["stability_rule"]) + ".")
    if untest:
        lines.append("Assumed, because no data can test it: " + "; ".join(untest) + ".")
    numbers = {k: _jsonable(m.get(k)) for k in ("variables", "n_rows", "left_out", "time_like", "id_like",
                                                 "drawn_edges", "unstable_edges", "isolated", "n_directed",
                                                 "n_uncertain", "n_subsamples")}
    numbers["untestable"] = untest
    numbers["error"] = sec.error
    return lines, numbers


_TOOLS = {"table_facts": _table_facts, "general_checks": _general_checks, "structure_facts": _structure_facts,
          "design_tests": _design_tests, "estimate_effect": _estimate_effect, "sensitivity": _sensitivity,
          "explore": _explore}


def _clean_lines(lines) -> list[str]:
    out = []
    for ln in lines or []:
        s = re.sub(r"[ \t]+", " ", _reword(str(ln))).strip()
        if s:
            out.append(s)
    return out


def call(name: str, args: dict, df: pd.DataFrame, id: str) -> dict:  # noqa: A002 - name fixed by the contract
    """Run one evidence tool. Never raises; failures give ok=False and a plain error."""
    try:
        safe_args = _jsonable(args) if isinstance(args, dict) else {}
    except Exception:  # noqa: BLE001
        safe_args = {}
    try:
        fn = _TOOLS.get(name)
        if fn is None:
            raise _Bad(f"There is no tool called {name!r}. The evidence tools are: {', '.join(contract.EVIDENCE_TOOLS)}.")
        if args is not None and not isinstance(args, dict):
            raise _Bad("The arguments must be an object.")
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise _Bad("No table is loaded, so nothing can be computed.")
        lines, numbers = fn(df, dict(args or {}))
        return contract.evidence_item(id, name, safe_args, ok=True, lines=_clean_lines(lines),
                                      numbers=_jsonable(numbers))
    except _Bad as e:
        return contract.evidence_item(id, name, safe_args, ok=False, error=str(e))
    except data.InputError as e:
        return contract.evidence_item(id, name, safe_args, ok=False, error=str(e))
    except Exception as e:  # noqa: BLE001
        return contract.evidence_item(id, name, safe_args, ok=False,
                                      error=f"The {name} tool could not finish ({type(e).__name__}: {e}).")
