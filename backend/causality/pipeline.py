"""The orchestrator: runs every part in order and assembles a Report.

Audit mode (a claim is given) and explore mode (no claim) share the same preparation and
checks. Each stage is isolated: a failing stage becomes a card with a plain-words error and
never takes the whole report down.
"""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from . import data
from .data import InputError
from .schema import (Assumption, Chart, Claim, Design, Estimand, Estimate, Prepared, Report,
                     Roles, Section, Status)

CHECK_TITLES = {
    "balance": "Balance between groups",
    "overlap": "Overlap",
    "hidden_confounding": "Hidden confounding (sensitivity)",
    "missing_outcomes": "Missing outcomes",
    "interference": "Spillovers between units",
    "literature": "Reading list",
}


def _safe(report: Report, key: str, title: str, fn: Callable[[], Any], default: Any = "section"):
    """Run a stage; on failure record a plain-words error and return a stub."""
    try:
        return fn()
    except InputError as e:
        report.errors.append(str(e))
    except Exception as e:  # noqa: BLE001
        msg = f"The step '{title}' could not run ({type(e).__name__}: {e})."
        report.errors.append(msg)
        traceback.print_exc()
        if default == "section":
            return Section(key=key, title=title, error=msg)
    return None if default == "section" else default


def parse_roles(roles: Optional[dict]) -> Roles:
    roles = dict(roles or {})
    clean: dict[str, Any] = {}
    for k in ("treatment", "outcome", "unit", "time", "instrument", "running", "cluster", "strata", "saturation"):
        v = roles.get(k)
        clean[k] = v if v not in ("", None) else None
    cut = roles.get("cutoff")
    if cut not in (None, ""):
        try:
            clean["cutoff"] = float(cut)
        except (TypeError, ValueError):
            raise InputError(f"The cutoff '{cut}' is not a number.")
    clean["post_treatment"] = [c for c in (roles.get("post_treatment") or []) if c]
    clean["randomized"] = bool(roles.get("randomized"))
    return Roles(**clean)


# ------------------------------------------------------------------------------- claim
def _original_of(prep: Prepared, df_col: Optional[str]) -> Optional[str]:
    if df_col is None:
        return None
    if df_col in prep.raw.columns:
        return df_col
    for orig, cols in prep.encoding.items():
        if df_col in cols:
            return orig
    return df_col


def _apply_claim_roles(report: Report, claim: Claim, roles: Roles, confirm: dict, prep0: Prepared) -> bool:
    """Fill treatment/outcome from the claim unless declared. Returns False if the user must
    confirm something first (questions are added to report.needs_input)."""
    ok = True
    cols = [c.name for c in prep0.columns if c.kind not in ("text", "constant")]
    for role in ("treatment", "outcome"):
        declared = getattr(roles, role)
        if declared:
            setattr(claim, f"{role}_col", declared)
            setattr(claim, f"{role}_score", 1.0)
            continue
        if confirm.get(role):
            setattr(claim, f"{role}_col", confirm[role])
            setattr(claim, f"{role}_score", 1.0)
            setattr(roles, role, _original_of(prep0, confirm[role]))
            continue
        col, score = getattr(claim, f"{role}_col"), getattr(claim, f"{role}_score")
        if col is None or score < 0.8:
            ok = False
            cands = [c for c, _ in (claim.candidates.get(role) or [])]
            phrase = getattr(claim, f"{role}_phrase") or "(not found)"
            q = (f"Which column is the {role}? The claim says \"{phrase}\""
                 + (f"; the best match is '{col}' (match score {score:.2f}, below 0.8)." if col else
                    ", and no column matched it."))
            report.needs_input.append({"kind": "confirm_column", "role": role, "question": q,
                                       "options": list(dict.fromkeys(cands + cols))})
        else:
            setattr(roles, role, _original_of(prep0, col))
    claim.needs_confirmation = not ok
    return ok


def _population_mask(prep: Prepared, filt: dict) -> pd.Series:
    col, op, val = filt["column"], filt.get("op", "=="), filt.get("value")
    src = prep.raw if filt.get("raw") or col not in prep.df else prep.df
    if col not in src:
        raise InputError(f"The population column '{col}' is not in the table.")
    s = src[col]
    if op in (">", ">=", "<", "<=") or isinstance(val, (int, float)) and not filt.get("raw"):
        s = pd.to_numeric(s, errors="coerce")
        val = float(val)
    ops = {"==": s == val, "!=": s != val, ">": s > val, ">=": s >= val, "<": s < val, "<=": s <= val}
    if op not in ops:
        raise InputError(f"Unknown population operator '{op}'.")
    return ops[op].fillna(False).astype(bool)


# ------------------------------------------------------------------------------- run
def run(raw: pd.DataFrame, claim_text: str = "", design_decl: Optional[str] = None,
        roles: Optional[dict] = None, confirm: Optional[dict] = None, online_literature: bool = False) -> Report:
    claim_text = (claim_text or "").strip()
    confirm = {k: v for k, v in (confirm or {}).items() if v}
    roles_obj = parse_roles(roles)
    report = Report(mode="audit" if claim_text else "explore")

    # 1. preparation with declared roles (needed for column matching)
    prep0 = data.prepare(raw, Roles(**{**roles_obj.__dict__, "post_treatment": list(roles_obj.post_treatment)}))

    claim: Optional[Claim] = None
    if claim_text:
        from .claim import parse_claim
        claim = _safe(report, "claim", "Reading the claim", lambda: parse_claim(claim_text, prep0), default=None)
        if claim is None:
            claim = Claim(raw=claim_text, testable=False, untestable_reason="the claim could not be read")
        report.claim = claim
        if not _apply_claim_roles(report, claim, roles_obj, confirm, prep0):
            report.data = data.summary(prep0)
            report.errors.append("Please confirm the columns the claim refers to before the analysis runs.")
            return report

    # 2. full preparation with resolved roles
    for role in ("treatment", "outcome"):
        col = getattr(roles_obj, role)
        if col and col not in raw.columns:
            raise InputError(f"The {role} must be a single 0/1 or numeric column; '{col}' is a category level. "
                             "Please make a 0/1 column for it.")
    if roles_obj.treatment:
        kind = next((c.kind for c in prep0.columns if c.name == roles_obj.treatment), None)
        if kind in ("categorical", "id", "text"):
            raise InputError(f"The treatment '{roles_obj.treatment}' has many labels; please give a 0/1 or numeric "
                             "treatment column.")
    prep = data.prepare(raw, roles_obj)
    if claim is not None:
        claim.treatment_col, claim.outcome_col = roles_obj.treatment, roles_obj.outcome

    # 3. population
    if claim is not None and claim.population_phrase:
        filt = claim.population_filter
        if confirm.get("population"):
            filt = _confirmed_population(prep, confirm["population"], filt)
            claim.population_filter, claim.population_score = filt, 1.0
        if filt and claim.population_score >= 0.8:
            mask = _safe(report, "population", "Selecting the population", lambda: _population_mask(prep, filt), default=None)
            if mask is not None:
                if mask.sum() < 20:
                    report.errors.append(f"Only {int(mask.sum())} rows match the population \"{claim.population_phrase}\"; "
                                         "too few to analyse, so everyone is used instead.")
                    claim.population_score = 0.0
                else:
                    prep = data.subset(prep, mask, f"Analysis restricted to {int(mask.sum())} of {len(mask)} rows: "
                                                   f"{claim.population_phrase}.")
        elif filt is None or claim.population_score < 0.8:
            opts = [c for c, _ in (claim.candidates.get("population") or [])]
            report.needs_input.append({
                "kind": "confirm_column", "role": "population",
                "question": f"Which column describes the population \"{claim.population_phrase}\"? "
                            "(Leave it unanswered to analyse everyone; the verdict will say the population was not matched.)",
                "options": opts + [c.name for c in prep.columns if c.kind in ("binary", "categorical")]})

    report.data = data.summary(prep)

    has_treatment = bool(prep.roles.treatment and prep.roles.outcome)
    if not has_treatment:
        report.mode = "explore"
        report.explore = _safe(report, "explore", "Exploration", lambda: _explore(prep))
        report.data = data.summary(prep)
        return report

    # 4. design
    design = data.resolve_design(prep, design_decl or None)
    report.design = design
    if design.ambiguous and not design_decl:
        report.needs_input.append({"kind": "choose_design", "role": None, "question": design.question,
                                   "options": [k for k, _ in sorted(design.scores.items(), key=lambda kv: -kv[1])[:2]]})
        report.errors.append("Please choose the study design before the analysis runs.")
        return report

    # 5. balance first; suspected post-treatment columns leave the (single) adjustment set
    from .checks import balance
    bal = _safe(report, "balance", CHECK_TITLES["balance"], lambda: balance.run(prep, design, claim))
    sus = (bal.metrics or {}).get("suspected_post_treatment") or [] if bal else []
    if sus and prep.roles.randomized:
        data.mark_post_treatment(prep, list(sus), "under randomization, treatment strongly predicts it, so it was "
                                                  "most likely measured after treatment.")
    report.data = data.summary(prep)  # adjustment set is now frozen

    # 6. other checks + design tests
    from .checks import attrition, interference, literature, overlap
    ov = _safe(report, "overlap", CHECK_TITLES["overlap"], lambda: overlap.run(prep, design, claim))
    att = _safe(report, "missing_outcomes", CHECK_TITLES["missing_outcomes"], lambda: attrition.run(prep, design, claim))
    inter = _safe(report, "interference", CHECK_TITLES["interference"], lambda: interference.run(prep, design, claim))
    lit = _safe(report, "literature", CHECK_TITLES["literature"],
                lambda: literature.run(prep, design, claim, online=online_literature))
    if lit is not None:
        lit.affects_grade = False

    from . import design_assess
    res = _safe(report, "design", "Design checks", lambda: design_assess.assess(prep, design, claim), default=None)
    design_assumptions, design_sections = res if res else (list(design.conflicts), [])
    design.assumptions = design_assumptions

    # 7. estimate, then sensitivity to hidden confounding
    from . import estimate as est_mod
    est = _safe(report, "estimate", "Estimation", lambda: est_mod.estimate(prep, design, claim), default=None)
    if est is None:
        est = Estimate(method="none", target="ATE", refused=True,
                       refusal_reason="The estimate could not be computed (see the errors above).")
    report.estimate = est
    from .checks import sensitivity
    sens = _safe(report, "hidden_confounding", CHECK_TITLES["hidden_confounding"],
                 lambda: sensitivity.run(prep, design, claim, est))
    if sens is not None and sens.status == Status.PASS:   # never a pass, whatever a module says
        sens.status = Status.ASSUMED

    sections = [s for s in [bal, ov, sens, att, inter] if s is not None] + list(design_sections)
    if lit is not None:
        sections.append(lit)
    report.sections = sections

    # 8. grade, estimand, verdict, next study, diagram
    est_card = Section(key="estimate", title="Estimate",
                       error=est.refusal_reason if est.refused else None)
    graded = _safe(report, "grade", "Grading", lambda: design_assess.grade(design, sections + [est_card]), default=None)
    if graded:
        design.grade, design.grade_reasons = graded
    else:
        design.grade, design.grade_reasons = "F", ["The grade could not be computed."]

    report.estimand = _estimand(prep, design, claim, est)
    if claim is not None:
        claim.estimand = report.estimand
        from .verdict import make_verdict
        report.verdict = _safe(report, "verdict", "Verdict", lambda: make_verdict(claim, design, est, sections), default=None)
    from .suggest import suggest_study
    report.suggestion = _safe(report, "next_study", "What study would settle it",
                              lambda: suggest_study(prep, design, claim, est))
    report.dag = _safe(report, "dag", "Assumed causal diagram", lambda: design_assess.dag(prep, design, claim), default=None)

    if report.mode == "explore":
        report.explore = _safe(report, "explore", "Exploration", lambda: _explore(prep))

    report.assumed_panel = assumed_panel(design, sections)
    report.data = data.summary(prep)
    return report


def _confirmed_population(prep: Prepared, col: str, filt: Optional[dict]) -> dict:
    """User picked a column for the population; binary -> ==1 unless the parser said otherwise."""
    if filt and filt.get("column") == col:
        return filt
    value = 1
    if filt and filt.get("value") in (0, 1):
        value = filt["value"]
    return {"column": col, "op": "==", "value": value}


def _explore(prep: Prepared) -> Section:
    from . import explore
    return explore.run(prep)


TARGET_WORDS = {
    "ATE": "average effect over everyone analysed",
    "ATT": "average effect on the units that were treated",
    "LATE": "average effect on compliers (units whose treatment follows the instrument)",
    "local_at_cutoff": "effect for units right at the cutoff",
    "dose_response": "change in the outcome per unit of treatment",
}


def _estimand(prep: Prepared, design: Design, claim: Optional[Claim], est: Estimate) -> Estimand:
    base = claim.estimand if claim is not None and claim.estimand else Estimand(prep.roles.treatment, prep.roles.outcome)
    target = est.target or base.target
    who = TARGET_WORDS.get(target, target)
    pop = base.population
    if est.scope == "overlap_trimmed":
        who += ", among units comparable on the measured covariates (overlap region)"
    text = (f"Effect of '{prep.roles.treatment}' on '{prep.roles.outcome}': {who}"
            + (f"; population: {pop}" if pop and pop != "everyone in the data" else "") + ".")
    return Estimand(treatment=prep.roles.treatment, outcome=prep.roles.outcome, contrast=base.contrast,
                    target=target, population=pop, text=text)


def assumed_panel(design: Design, sections: list[Section]) -> list[Assumption]:
    seen, out = set(), []
    for a in list(design.assumptions) + [a for s in sections for a in s.assumptions]:
        if a.status == Status.ASSUMED and a.key not in seen:
            seen.add(a.key)
            out.append(a)
    return out
