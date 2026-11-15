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
        roles: Optional[dict] = None, confirm: Optional[dict] = None, online_literature: bool = False,
        progress: Optional[Callable[[str], None]] = None, context: str = "") -> Report:
    """`progress`, if given, is called with a stage key as each stage starts (live status for the web app).
    It never changes the result."""
    def stage(key: str) -> None:
        if progress is not None:
            try:
                progress(key)
            except Exception:  # noqa: BLE001  (a broken progress reporter must not break the analysis)
                pass

    claim_text = (claim_text or "").strip()
    confirm = {k: v for k, v in (confirm or {}).items() if v}
    hint: Optional[str] = None
    roles_obj = parse_roles(roles)
    report = Report(mode="audit" if claim_text else "explore")

    stage("prepare")
    # 1. preparation with declared roles (needed for column matching)
    prep0 = data.prepare(raw, Roles(**{**roles_obj.__dict__, "post_treatment": list(roles_obj.post_treatment)}))

    claim: Optional[Claim] = None
    stage("claim")
    if claim_text:
        from .claim import parse_claim
        claim = _safe(report, "claim", "Reading the claim", lambda: parse_claim(claim_text, prep0), default=None)
        if claim is None:
            claim = Claim(raw=claim_text, testable=False, untestable_reason="the claim could not be read")
        report.claim = claim
        _apply_context(claim, context, prep0)
        ok = _apply_claim_roles(report, claim, roles_obj, confirm, prep0)
        ok = _apply_claim_instrument(report, claim, roles_obj, confirm, prep0) and ok
        ok = _apply_randomization_hint(report, claim, roles_obj, confirm, design_decl) and ok
        if not ok:
            report.data = data.summary(prep0)
            report.errors.append("Please answer the question(s) above before the analysis runs.")
            return report
        hint = claim.design_hint if claim.design_hint in ("did", "iv", "rd", "observational") else None
        if hint == "iv" and roles_obj.instrument and not design_decl:
            design_decl = "iv"          # naming an instrument in the claim declares the design
            claim.notes.append("Because the claim names an instrument, the instrument design is used.")

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
        stage("explore")
        report.explore = _safe(report, "explore", "Exploration", lambda: _explore(prep))
        report.data = data.summary(prep)
        return report

    stage("design")
    # 4. design
    design = data.resolve_design(prep, design_decl or None, hint=hint)
    report.design = design
    if design.ambiguous and not design_decl:
        report.needs_input.append({"kind": "choose_design", "role": None, "question": design.question,
                                   "options": [k for k, _ in sorted(design.scores.items(), key=lambda kv: -kv[1])[:2]]})
        report.errors.append("Please choose the study design before the analysis runs.")
        return report

    # 5-6. identification checks; their result is then used as a reference for the design:
    # a guessed design whose own relevance test fails is dropped, and the checks are re-run
    ident = _identification(report, prep, design, claim, online_literature, stage)
    new_kind = _reclassify(design, ident)
    if new_kind:
        old = design
        design = data.resolve_design(prep, new_kind)
        design.declared = False
        design.reasons = old.reasons + [ident["reclass_reason"],
                                        "The identification checks were run again for the new design."]
        design.scores = old.scores
        report.design = design
        ident = _identification(report, prep, design, claim, online_literature, stage)
    bal, ov, att, inter, lit = ident["bal"], ident["ov"], ident["att"], ident["inter"], ident["lit"]
    design_sections = ident["design_sections"]
    _balance_reference(design, prep, bal)
    report.data = data.summary(prep)  # adjustment set is now frozen
    from . import design_assess

    stage("estimate")
    # 7. estimate, then sensitivity to hidden confounding
    from . import estimate as est_mod
    est = _safe(report, "estimate", "Estimation", lambda: est_mod.estimate(prep, design, claim), default=None)
    if est is None:
        est = Estimate(method="none", target="ATE", refused=True,
                       refusal_reason="The estimate could not be computed (see the errors above).")
    est = _event_study_setup(report, prep, design, ident, est)
    ov = ident["ov"]
    report.estimate = est
    stage("sensitivity")
    from .checks import sensitivity
    sens = _safe(report, "hidden_confounding", CHECK_TITLES["hidden_confounding"],
                 lambda: sensitivity.run(prep, design, claim, est))
    if sens is not None and sens.status == Status.PASS:   # never a pass, whatever a module says
        sens.status = Status.ASSUMED

    sections = [s for s in [bal, ov, sens, att, inter] if s is not None] + list(design_sections)
    if lit is not None:
        sections.append(lit)
    report.sections = sections

    stage("verdict")
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
        v = report.verdict
        if v is not None and est.target == "event_day_effect" and v.caveat:
            # an event study rests on a different untestable assumption than parallel trends
            new_caveat = "if nothing else happened to these units on the event date"
            v.sentence = v.sentence.replace(v.caveat.strip().rstrip("."), new_caveat)
            v.caveat = new_caveat
    from .suggest import suggest_study
    report.suggestion = _safe(report, "next_study", "What study would settle it",
                              lambda: suggest_study(prep, design, claim, est))
    report.dag = _safe(report, "dag", "Assumed causal diagram", lambda: design_assess.dag(prep, design, claim), default=None)

    if report.mode == "explore":
        stage("explore")
        report.explore = _safe(report, "explore", "Exploration", lambda: _explore(prep))

    report.assumed_panel = assumed_panel(design, sections)
    report.data = data.summary(prep)
    return report


def _identification(report: Report, prep: Prepared, design: Design, claim: Optional[Claim], online: bool,
                    stage: Callable[[str], None] = lambda key: None) -> dict:
    """Balance first (suspected post-treatment columns leave the single adjustment set), then
    overlap, missing outcomes, interference, reading list and the design's own tests.
    `stage` reports progress to the web app (see run)."""
    from . import design_assess
    from .checks import attrition, balance, interference, literature, overlap
    stage("balance")
    bal = _safe(report, "balance", CHECK_TITLES["balance"], lambda: balance.run(prep, design, claim))
    sus = ((bal.metrics or {}).get("suspected_post_treatment") or []) if bal else []
    if sus and prep.roles.randomized:
        data.mark_post_treatment(prep, list(sus), "under randomization, treatment strongly predicts it, so it was "
                                                  "most likely measured after treatment.")
    stage("checks")
    ov = _safe(report, "overlap", CHECK_TITLES["overlap"], lambda: overlap.run(prep, design, claim))
    att = _safe(report, "missing_outcomes", CHECK_TITLES["missing_outcomes"], lambda: attrition.run(prep, design, claim))
    inter = _safe(report, "interference", CHECK_TITLES["interference"], lambda: interference.run(prep, design, claim))
    lit = _safe(report, "literature", CHECK_TITLES["literature"],
                lambda: literature.run(prep, design, claim, online=online))
    if lit is not None:
        lit.affects_grade = False
    res = _safe(report, "design", "Design checks", lambda: design_assess.assess(prep, design, claim), default=None)
    design_assumptions, design_sections = res if res else (list(design.conflicts), [])
    design.assumptions = design_assumptions
    return {"bal": bal, "ov": ov, "att": att, "inter": inter, "lit": lit, "design_sections": list(design_sections)}


def _reclassify(design: Design, ident: dict) -> Optional[str]:
    """Use the identification result as a reference for the design type. Only a design the
    tool GUESSED can be dropped, and only when its relevance test fails (an instrument that
    does not move the treatment is not an instrument). Failed validity tests (pre-trends,
    bunching) never reclassify: they stay a failed design."""
    if design.declared:
        return None
    if design.kind == "iv":
        fs = next((s for s in ident["design_sections"] if s.key == "design_first_stage"), None)
        if fs is not None and fs.status == Status.FAIL:
            ident["reclass_reason"] = ("The identification check found the guessed instrument too weak to move the "
                                       "treatment, so the instrument design was dropped and the data are treated as "
                                       "observational. Declare the instrument design if it is the real one.")
            return "observational"
    return None


def _balance_reference(design: Design, prep: Prepared, bal: Optional[Section]) -> None:
    """Balance as a reference for the design: well-balanced groups are worth a remark, but
    are never called an experiment unless the user says treatment was randomized."""
    if bal is None or design.kind != "observational" or prep.roles.randomized or not prep.adjustment_set:
        return
    p = (bal.metrics or {}).get("joint_p")
    if isinstance(p, (int, float)) and p >= 0.2 and prep.treatment_is_binary():
        design.reasons.append(f"The identification check found the groups balanced on the measured columns "
                              f"(joint p = {p:.2f}), which is what randomization would produce. This is still "
                              "treated as observational: say so if treatment was randomized.")


def _apply_claim_instrument(report: Report, claim: Claim, roles: Roles, confirm: dict, prep0: Prepared) -> bool:
    """An instrument named in the claim is collected; a weak column match is confirmed first."""
    if roles.instrument:
        claim.instrument_col, claim.instrument_score = roles.instrument, 1.0
        return True
    if confirm.get("instrument"):
        roles.instrument = _original_of(prep0, confirm["instrument"])
        claim.instrument_col, claim.instrument_score = roles.instrument, 1.0
        return True
    if not claim.instrument_phrase:
        return True
    if claim.instrument_col and claim.instrument_score >= 0.8:
        roles.instrument = _original_of(prep0, claim.instrument_col)
        return True
    cands = [c for c, _ in (claim.candidates.get("instrument") or [])]
    cols = [c.name for c in prep0.columns if c.kind not in ("text", "constant")]
    q = f"Which column is the instrument? The claim says '{claim.instrument_phrase}'"
    q += (f"; the best match is '{claim.instrument_col}' (match score {claim.instrument_score:.2f}, below 0.8)."
          if claim.instrument_col else ", and no column matched it.")
    report.needs_input.append({"kind": "confirm_column", "role": "instrument", "question": q,
                               "options": list(dict.fromkeys(cands + cols))})
    return False


def _apply_randomization_hint(report: Report, claim: Claim, roles: Roles, confirm: dict, design_decl: Optional[str]) -> bool:
    """Wording such as 'randomized trial' is only a hint: the user confirms it before anything
    is called an experiment."""
    if claim.design_hint != "experiment" or roles.randomized or design_decl:
        return True
    ans = str(confirm.get("randomized", "")).strip().lower()
    if ans in ("yes", "true", "y"):
        roles.randomized = True
        claim.notes.append("You confirmed that treatment was randomized.")
        return True
    if ans in ("no", "false", "n"):
        claim.notes.append("The claim's wording suggested randomization, but you said treatment was not randomized.")
        return True
    report.needs_input.append({
        "kind": "confirm_column", "role": "randomized",
        "question": f"The claim's wording ('{claim.design_hint_phrase or 'randomized'}') suggests treatment was "
                    "randomly assigned. Was it? Nothing is called an experiment unless you say so.",
        "options": ["yes", "no"]})
    return False


def _event_study_setup(report: Report, prep: Prepared, design: Design, ident: dict, est: Estimate) -> Estimate:
    """A one-off event (a pulse) is not a lasting treatment, so three parts of the
    difference-in-differences machinery do not apply and are switched to information only:
    row-level overlap (one treated row per unit is rare by construction), the placebo-in-time
    test (it needs a lasting switch), and the staggered-adoption estimate (it assumes treatment
    stays on, which dilutes a one-day effect). The reported estimate becomes the average
    abnormal outcome on the event date; the diluted estimate is kept as a comparison."""
    es = next((s for s in ident["design_sections"] if s.key == "design_event_study" and not s.error), None)
    if es is None:
        return est
    why = "Not applicable to a one-off event: "
    for card, reason in ((ident.get("ov"), "each unit is compared with its own expected outcome on the event date, "
                                           "so row-by-row overlap between treated and untreated rows is not the "
                                           "relevant comparison."),
                         (next((s for s in ident["design_sections"] if s.key == "design_placebo_time"), None),
                          "the test pretends a lasting treatment started earlier, which a one-day event is not.")):
        if card is None:
            continue
        card.status, card.affects_grade, card.caps_verdict = None, False, False
        card.assumptions = []
        card.summary = why + reason + " Shown for information only. " + (card.summary or "")
    design.assumptions = [a for a in design.assumptions if a.key != "did_placebo_time"]

    m = es.metrics or {}
    mean, se, n = m.get("mean_abnormal"), m.get("mean_abnormal_se"), int(m.get("n_events") or 0)
    if mean is None or not se or n < 5 or not np.isfinite(se) or se <= 0:
        return est
    from scipy import stats
    dfree = n - 1
    levels = {}
    for lv in (0.90, 0.95, 0.98, 0.99):
        q = float(stats.t.ppf(0.5 + lv / 2, dfree))
        levels[str(int(round(lv * 100)))] = [mean - q * se, mean + q * se]
    y = prep.Y
    t = prep.T
    new = Estimate(
        method=f"Event study: average abnormal outcome on the event date ({m.get('model', 'expected-outcome model')})",
        target="event_day_effect", scope="treated", estimate=float(mean), se=float(se),
        ci90=tuple(levels["90"]), ci95=tuple(levels["95"]),
        p_value=float(2 * stats.t.sf(abs(mean / se), dfree)), n=n,
        baseline_mean=float(y[t == 0].mean()) if y is not None and t is not None else None,
        outcome_sd=float(y.std()) if y is not None else None,
        extra={"ci_levels": levels, "se_type": f"cross-sectional, t with {dfree} degrees of freedom",
               "bmp_p": m.get("bmp_p"), "corrado_p": m.get("corrado_p"), "n_events": n,
               "notes": ["The treatment is a one-off event, so the estimate is the average gap between each unit's "
                         "outcome on its event date and what its own earlier behaviour predicted.",
                         "The Boehmer-Musumeci-Poulsen and Corrado rank tests are in the event-study card."]},
        charts=list(es.charts))
    if est is not None and not est.refused and est.estimate is not None:
        new.comparisons.append({"method": est.method, "estimate": est.estimate, "ci95": est.ci95,
                                "note": "Assumes treatment stays on after it starts, so a one-day effect is "
                                        "averaged over all later periods and looks much smaller."})
    return new


def _apply_context(claim: Claim, context: str, prep0: Prepared) -> None:
    """An optional description of the study/dataset. It is read ONLY for design wording and a
    named instrument (the same rules as for the claim); it never sets treatment, outcome or
    population, and the claim's own wording wins when both say something."""
    context = (context or "").strip()
    if not context:
        return
    try:
        from .claim import parse_claim
        ctx = parse_claim(context, prep0)
    except Exception:  # noqa: BLE001
        return
    if claim.design_hint is None and ctx.design_hint:
        claim.design_hint, claim.design_hint_phrase = ctx.design_hint, ctx.design_hint_phrase
        claim.notes.append(f"Design hint taken from the description you gave ('{ctx.design_hint_phrase}').")
    if not claim.instrument_phrase and ctx.instrument_phrase:
        claim.instrument_phrase, claim.instrument_col = ctx.instrument_phrase, ctx.instrument_col
        claim.instrument_score = ctx.instrument_score
        if ctx.candidates.get("instrument"):
            claim.candidates["instrument"] = ctx.candidates["instrument"]
        claim.notes.append(f"Instrument taken from the description you gave ('{ctx.instrument_phrase}').")


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
    "event_day_effect": "average abnormal outcome of the treated units on their event date",
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
