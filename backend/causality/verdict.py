"""Turn the design checks and the estimate into one plain-words verdict.

The order matters and follows the one rule (never claim more than the data can show):

1. No claim / untestable claim            -> "Not testable".
2. No estimate / refused estimate         -> "Inconclusive".
3. Identification problems FIRST          -> "Inconclusive" (direction and size not assessed):
   grade D or F, any FAIL assumption (design, declaration conflicts, grade-affecting sections),
   any FAIL grade-affecting section, any section with caps_verdict (e.g. outcome dropout).
4. Direction, then size, then scope / population / shape caps.

Rule choices (documented because the spec left them open):

* Direction uses the two-sided 95% CI (equivalently p < 0.05). An insignificant result of
  either sign is "no_evidence", never a contradiction.
* Negated claims ("X does not reduce Y"): the claim is "no meaningful effect in the named
  direction". Margin m = |claimed size| if a size is given, else 0.1 x outcome SD (0 if the SD
  is unknown, which is the strictest choice). Contradicted if the 95% CI shows a significant
  effect in the named direction (checked first). Supported if the one-sided 5% test rejects a
  meaningful effect in that direction (90% CI bound on that side is inside the margin).
  Otherwise no_evidence.
* No-effect claims: TOST with margin 0.1 x outcome SD. Supported if the 90% CI lies within
  +-m. Contradicted if the 95% CI excludes 0 AND the point estimate is outside the margin.
  Otherwise inconclusive ("cannot rule out a meaningful effect").
* Size, about/exactly/unspecified bound: with c the signed claimed effect and tol = |c|/2,
  supported if the 90% CI sits inside [c - tol, c + tol]; rejected if c is outside the 95% CI;
  otherwise inconclusive (interval too wide).
* Size, "at least c" (measured along the claimed direction, x' = sign(c) * x):
  rejected if the 95% CI upper bound' < |c| (the CI rules out reaching c);
  supported if the 90% CI lower bound' >= |c|/2 AND the point estimate' >= |c|;
  otherwise inconclusive.
* Size, "at most c" (mirror): rejected if the 95% CI lower bound' > |c|;
  supported if the 90% CI upper bound' <= 1.5|c| AND the point estimate' <= |c|;
  otherwise inconclusive.
* A claimed shape that was not tested or not supported caps "Supported" at "Partly supported".
* Partial scope (treated / compliers / local_cutoff / overlap_trimmed) or an unmatched claimed
  population caps "Supported" at "Partly supported"; the sentence names the part.
* Observational, DiD, IV and RD verdicts carry the design's untestable assumption as a caveat
  at the end of the sentence ("..., if no unmeasured confounding." for observational).
"""
from __future__ import annotations

import math
from typing import Optional

from scipy import stats

from .schema import Claim, Design, Estimate, Section, Status, SubVerdict, Verdict
from .units import claimed_effect

ASPECTS = ("direction", "size", "scope", "population", "shape", "data_quality")

SCOPE_WORDS = {
    "treated": "the treated units only",
    "compliers": "compliers only (units whose treatment followed the instrument)",
    "local_cutoff": "units near the cutoff only",
    "overlap_trimmed": "units with comparable treated and untreated counterparts only",
}
SCOPE_SHORT = {
    "treated": "for the treated units only",
    "compliers": "for compliers only",
    "local_cutoff": "only for units near the cutoff",
    "overlap_trimmed": "only for units that have comparable counterparts",
}
CAVEATS = {
    "observational": "if no unmeasured confounding",
    "did": "if the groups would have kept parallel trends without the treatment",
    "iv": "if the instrument affects the outcome only through the treatment",
    "rd": "if nothing else jumps at the cutoff",
}
UNTESTABLE_TEXT = {
    "mediation": "it is about a mechanism (how the effect travels through another variable), "
                 "which this tool cannot test from one table",
    "counterfactual": "it is worded as a counterfactual about specific cases, which data cannot check",
    "comparison with another treatment": "it compares this treatment with another treatment, "
                                         "which needs both treatments in the data",
}


# ----------------------------------------------------------------------------- helpers
def _f(x: Optional[float]) -> str:
    if x is None:
        return "?"
    if x == 0:
        return "0"
    a = abs(x)
    if a >= 1000 or a < 0.01:
        return f"{x:.3g}"
    return f"{x:.2f}".rstrip("0").rstrip(".") if a < 10 else f"{x:.1f}".rstrip("0").rstrip(".")


def _ci(est: Estimate, level: float) -> Optional[tuple[float, float]]:
    ci = est.ci90 if level == 0.90 else est.ci95
    if ci is not None and len(ci) == 2 and all(v is not None and math.isfinite(v) for v in ci):
        return (float(min(ci)), float(max(ci)))
    if est.estimate is not None and est.se is not None and est.se >= 0:
        z = stats.norm.ppf(0.5 + level / 2)
        return (est.estimate - z * est.se, est.estimate + z * est.se)
    return None


def _names(claim: Claim) -> tuple[str, str]:
    t = claim.treatment_phrase or claim.treatment_col or "the treatment"
    y = claim.outcome_phrase or claim.outcome_col or "the outcome"
    return t, y


def _claim_phrase(claim: Claim) -> str:
    t, y = _names(claim)
    d = claim.direction
    if d == "no_effect":
        return f"{t} has no meaningful effect on {y}"
    if claim.negated:
        verb = {"increase": "does not raise", "decrease": "does not lower"}.get(d, "does not affect")
        return f"{t} {verb} {y}"
    verb = {"increase": "raises", "decrease": "lowers"}.get(d, "affects")
    return f"{t} {verb} {y}"


def _size_text(claim: Claim) -> str:
    if claim.size is None:
        return ""
    unit = {"percent": "%", "pp": " percentage points", "sd": " standard deviations",
            "times": " times", "absolute": ""}.get(claim.size_unit or "", "")
    bound = {"at_least": "at least ", "at_most": "at most "}.get(claim.bound or "", "")
    return f"{bound}{_f(claim.size)}{unit}"


def _est_text(est: Estimate, ci90) -> str:
    s = f"about {_f(est.estimate)}"
    if ci90:
        s += f", 90% CI {_f(ci90[0])} to {_f(ci90[1])}"
    return s


def identification_problems(design: Design, sections: list[Section]) -> list[str]:
    """Plain-words list of everything that blocks a causal reading of the estimate."""
    probs: list[str] = []
    if design is not None:
        if design.grade in ("D", "F"):
            why = "; ".join(design.grade_reasons[:3])
            probs.append(f"The design got grade {design.grade}" + (f" ({why})." if why else "."))
        for a in list(design.assumptions) + list(design.conflicts):
            if a.status == Status.FAIL:
                probs.append(f"{a.name} failed" + (f": {a.detail}" if a.detail else "."))
    for s in sections or []:
        if s.caps_verdict:
            probs.append(f"{s.title}: {s.summary or 'this check limits what the data can show.'}")
            continue
        if not s.affects_grade:
            continue
        if s.status == Status.FAIL:
            probs.append(f"{s.title} failed" + (f": {s.summary}" if s.summary else "."))
            continue
        for a in s.assumptions:
            if a.status == Status.FAIL:
                probs.append(f"{a.name} failed" + (f": {a.detail}" if a.detail else "."))
    # de-duplicate, keep order
    seen, out = set(), []
    for p in probs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ----------------------------------------------------------------------------- direction
def _direction(claim: Claim, est: Estimate, ci90, ci95, margin_sd: float) -> SubVerdict:
    d = claim.direction
    sd = est.outcome_sd
    lo95, hi95 = ci95
    lo90, hi90 = ci90
    pt = est.estimate
    sig = lo95 > 0 or hi95 < 0
    ci_txt = f"estimate {_f(pt)}, 95% CI {_f(lo95)} to {_f(hi95)}"

    if d == "no_effect" or (claim.negated and d not in ("increase", "decrease")):
        if not sd:
            return SubVerdict("direction", "inconclusive",
                              "The outcome spread is unknown, so a 'meaningful effect' margin cannot be set; "
                              "an insignificant result is not evidence of no effect.")
        m = margin_sd * sd
        if lo90 >= -m and hi90 <= m:
            return SubVerdict("direction", "supported",
                              f"Equivalence test passed: the 90% CI ({_f(lo90)} to {_f(hi90)}) lies within "
                              f"+-{_f(m)} ({margin_sd:g} SD), so any effect is small.")
        if sig and abs(pt) > m:
            return SubVerdict("direction", "contradicted",
                              f"There is a clear effect ({ci_txt}), larger than the {_f(m)} margin.")
        return SubVerdict("direction", "inconclusive",
                          f"Cannot rule out a meaningful effect: the 90% CI ({_f(lo90)} to {_f(hi90)}) is not "
                          f"inside +-{_f(m)}. Not finding an effect is not the same as showing there is none.")

    if claim.negated:
        s = 1.0 if d == "increase" else -1.0     # direction the claim says does NOT happen
        c = claimed_effect(claim, est)
        if c is not None and c != 0:
            m, mtxt = abs(c), f"the claimed {_f(abs(c))}"
        elif sd:
            m, mtxt = margin_sd * sd, f"{margin_sd:g} SD ({_f(margin_sd * sd)})"
        else:
            m, mtxt = 0.0, "zero (outcome spread unknown)"
        word = "rise" if s > 0 else "drop"
        if (s > 0 and lo95 > 0) or (s < 0 and hi95 < 0):
            return SubVerdict("direction", "contradicted", f"There is a significant {word} ({ci_txt}).")
        bound = hi90 if s > 0 else lo90           # the 90% bound on the claimed-against side
        if s * bound < m:
            return SubVerdict("direction", "supported",
                              f"A {word} bigger than {mtxt} is ruled out (one-sided test, 90% CI bound {_f(bound)}).")
        return SubVerdict("direction", "no_evidence",
                          f"A meaningful {word} (more than {mtxt}) cannot be ruled out ({ci_txt}).")

    if d == "nonzero" or d is None:
        if sig:
            return SubVerdict("direction", "supported", f"The effect is clearly different from zero ({ci_txt}).")
        return SubVerdict("direction", "no_evidence", f"The effect is not clearly different from zero ({ci_txt}).")

    s = 1.0 if d == "increase" else -1.0
    if (s > 0 and lo95 > 0) or (s < 0 and hi95 < 0):
        return SubVerdict("direction", "supported", f"Clear effect in the claimed direction ({ci_txt}).")
    if sig:
        return SubVerdict("direction", "contradicted", f"Clear effect in the OPPOSITE direction ({ci_txt}).")
    return SubVerdict("direction", "no_evidence",
                      f"Not clearly different from zero ({ci_txt}); "
                      + ("the sign is opposite to the claim, but that is not significant either."
                         if (pt is not None and s * pt < 0) else "the data cannot tell the direction."))


# ----------------------------------------------------------------------------- size
def _size(claim: Claim, est: Estimate, ci90, ci95) -> SubVerdict:
    if claim.direction == "no_effect":
        return SubVerdict("size", "not_claimed", "A no-effect claim is checked by the equivalence test.")
    if claim.size is None:
        return SubVerdict("size", "not_claimed", "The claim gives no size.")
    if claim.negated:
        return SubVerdict("size", "not_claimed",
                          f"The stated size ({_size_text(claim)}) is used as the 'meaningful effect' margin "
                          "in the direction check.")
    c = claimed_effect(claim, est)
    if c is None:
        why = {"percent": "a percent change needs a non-zero baseline (control) mean",
               "times": "a 'times' change needs a baseline (control) mean",
               "sd": "the outcome standard deviation is unknown"}.get(claim.size_unit or "",
                                                                     "the size could not be turned into outcome units")
        return SubVerdict("size", "not_checkable", f"Not checkable: {why}.")
    lo90, hi90 = ci90
    lo95, hi95 = ci95
    pt = est.estimate
    ctxt = f"the claimed {_size_text(claim)} (= {_f(c)} in outcome units)"
    est_txt = f"estimate {_f(pt)}, 90% CI {_f(lo90)} to {_f(hi90)}, 95% CI {_f(lo95)} to {_f(hi95)}"
    tol = abs(c) / 2
    bound = claim.bound
    if bound in ("at_least", "at_most") and c != 0:
        s = 1.0 if c > 0 else -1.0
        # move into "along the claimed direction" coordinates
        l90, h90 = sorted((s * lo90, s * hi90))
        l95, h95 = sorted((s * lo95, s * hi95))
        p, a = s * pt, abs(c)
        if bound == "at_least":
            if h95 < a:
                return SubVerdict("size", "rejected", f"The 95% CI rules out an effect as large as {ctxt} ({est_txt}).")
            if l90 >= a / 2 and p >= a:
                return SubVerdict("size", "supported",
                                  f"The estimate reaches {ctxt} and the 90% CI stays above half of it ({est_txt}).")
            return SubVerdict("size", "inconclusive",
                              f"Inconclusive (interval too wide or estimate below the claim) for {ctxt} ({est_txt}).")
        if l95 > a:
            return SubVerdict("size", "rejected", f"The 95% CI shows an effect bigger than {ctxt} ({est_txt}).")
        if h90 <= a + a / 2 and p <= a:
            return SubVerdict("size", "supported",
                              f"The estimate is within {ctxt} and the 90% CI stays below 1.5 times it ({est_txt}).")
        return SubVerdict("size", "inconclusive", f"Inconclusive (interval too wide) for {ctxt} ({est_txt}).")
    if lo95 > c or hi95 < c:
        return SubVerdict("size", "rejected", f"{ctxt[0].upper() + ctxt[1:]} is outside the 95% CI ({est_txt}).")
    if lo90 >= c - tol and hi90 <= c + tol:
        return SubVerdict("size", "supported",
                          f"The 90% CI sits within half of {ctxt} either way ({est_txt}).")
    return SubVerdict("size", "inconclusive",
                      f"Inconclusive (interval too wide): {ctxt} is inside the CI, but so are quite different sizes ({est_txt}).")


# ----------------------------------------------------------------------------- other aspects
def _scope(est: Estimate) -> SubVerdict:
    sc = est.scope or "full"
    if sc == "full":
        return SubVerdict("scope", "full", f"The {est.method or 'method'} speaks for everyone in the analysed data.")
    return SubVerdict("scope", "capped",
                      f"The {est.method or 'method'} only speaks for {SCOPE_WORDS.get(sc, sc)}; "
                      "it says nothing direct about everyone else.")


def _population(claim: Claim) -> SubVerdict:
    if not claim.population_phrase:
        return SubVerdict("population", "not_claimed", "The claim names no particular group.")
    if claim.population_filter and claim.population_score >= 0.8:
        f = claim.population_filter
        return SubVerdict("population", "matched",
                          f"'{claim.population_phrase}' was matched to {f.get('column')} {f.get('op')} {f.get('value')!r}.")
    return SubVerdict("population", "not_matched",
                      f"'{claim.population_phrase}' could not be matched to the data, so it was tested on everyone instead.")


def _shape(claim: Claim, est: Estimate) -> SubVerdict:
    if not claim.shape:
        return SubVerdict("shape", "not_claimed", "The claim gives no shape.")
    tests = (est.extra or {}).get("shape_tests") if est else None
    if not isinstance(tests, dict) or not tests:
        return SubVerdict("shape", "not_tested", f"The claimed {claim.shape} shape was not tested.")
    sh = claim.shape
    if sh == "linear" and tests.get("linear_p") is not None:
        p = float(tests["linear_p"])
        if p < 0.05:
            return SubVerdict("shape", "rejected", f"The dose-response curve departs from a straight line (p = {p:.3g}).")
        return SubVerdict("shape", "consistent",
                          f"No clear departure from a straight line (p = {p:.3g}); that is not proof it is linear.")
    if sh == "diminishing" and tests.get("diminishing") is not None:
        ok = bool(tests["diminishing"])
        return SubVerdict("shape", "consistent" if ok else "not_shown",
                          "The effect flattens at higher doses." if ok else "The data do not show the effect flattening.")
    if sh == "nonmonotone" and tests.get("monotone") is not None:
        mono = bool(tests["monotone"])
        return SubVerdict("shape", "not_shown" if mono else "consistent",
                          "The curve looks one-directional, not rising-then-falling." if mono
                          else "The curve changes direction, as claimed.")
    if sh in tests and tests[sh] is not None:
        ok = bool(tests[sh])
        return SubVerdict("shape", "consistent" if ok else "not_shown",
                          f"The {sh} shape is {'consistent with' if ok else 'not shown by'} the data.")
    return SubVerdict("shape", "not_tested", f"The claimed {sh} shape was not tested.")


def _data_quality(sections: list[Section]) -> SubVerdict:
    names = {"missing_outcomes": "missing outcomes", "balance": "balance between groups",
             "overlap": "overlap between groups"}
    by = {s.key: s for s in sections or []}
    bad, ok, msgs = [], [], []
    for k, label in names.items():
        s = by.get(k)
        if s is None or s.status is None:
            continue
        if s.status == Status.PASS:
            ok.append(label)
        elif s.status in (Status.CONCERN, Status.FAIL):
            bad.append(label)
            if s.summary:
                msgs.append(s.summary)
    for s in sections or []:
        if s.error and s.key in names:
            msgs.append(f"{names[s.key]} could not be checked: {s.error}")
    if bad:
        return SubVerdict("data_quality", "concerns",
                          f"Problems with {', '.join(bad)}. " + " ".join(msgs[:3]))
    if ok:
        return SubVerdict("data_quality", "ok", f"No problems found with {', '.join(ok)}. " + " ".join(msgs[:2]))
    return SubVerdict("data_quality", "unknown", "Data quality checks were not available. " + " ".join(msgs[:2]))


def _with_caveat(sentence: str, caveat: Optional[str]) -> str:
    s = sentence.rstrip().rstrip(".")
    return f"{s}, {caveat}." if caveat else f"{s}."


# ----------------------------------------------------------------------------- main
def make_verdict(claim: Optional[Claim], design: Optional[Design], estimate: Optional[Estimate],
                 sections: Optional[list[Section]], margin_sd: float = 0.1) -> Verdict:
    try:
        return _make_verdict(claim, design, estimate, sections or [], margin_sd)
    except Exception as e:  # never raise on bad data
        return Verdict(label="Inconclusive",
                       sentence="The verdict could not be worked out, so nothing is claimed.",
                       reasons=[f"Internal problem while forming the verdict: {e}"])


def _make_verdict(claim, design, est, sections, margin_sd) -> Verdict:
    # 1. testability --------------------------------------------------------------
    if claim is None:
        return Verdict(label="Not testable", sentence="No claim was given, so there is nothing to test.",
                       sub=[SubVerdict(a, "not_assessed", "No claim.") for a in ASPECTS],
                       reasons=["No claim was given (explore mode)."])
    if not claim.testable:
        why = UNTESTABLE_TEXT.get(claim.untestable_reason or "", claim.untestable_reason or "it cannot be checked from this data")
        return Verdict(label="Not testable", sentence=f"This claim cannot be tested with this data because {why}.",
                       sub=[SubVerdict(a, "not_assessed", "The claim is not testable.") for a in ASPECTS],
                       reasons=[f"Not testable: {why}."])

    reasons: list[str] = []
    if claim.hedges:
        reasons.append(f"The claim is hedged ({', '.join(claim.hedges)}); the verdict uses the same rules, "
                       "since a hedge does not change what the data must show.")
    phrase = _claim_phrase(claim)
    kind = design.kind if design else None
    caveat = CAVEATS.get(kind)
    scope = _scope(est) if est else SubVerdict("scope", "not_assessed", "No estimate.")
    pop = _population(claim)
    shape = _shape(claim, est) if est else SubVerdict("shape", "not_assessed", "No estimate.")
    dq = _data_quality(sections)
    blockers = identification_problems(design, sections) if design is not None else []

    # 2. estimate available? -------------------------------------------------------
    if est is None or est.refused:
        why = (est.refusal_reason if est is not None else None) or "no estimate could be made"
        reasons.append(f"No estimate: {why}")
        reasons.extend(blockers)
        return Verdict(label="Inconclusive",
                       sentence=f"The data cannot say whether {phrase}, because {why.rstrip('.')}.",
                       sub=[SubVerdict("direction", "not_assessed", f"No estimate: {why}"),
                            SubVerdict("size", "not_assessed", f"No estimate: {why}"),
                            scope, pop, shape, dq],
                       reasons=reasons)
    ci90, ci95 = _ci(est, 0.90), _ci(est, 0.95)
    if est.estimate is None or ci90 is None or ci95 is None:
        reasons.append("The estimate has no uncertainty interval, so nothing can be concluded from it.")
        reasons.extend(blockers)
        return Verdict(label="Inconclusive",
                       sentence=f"The data cannot say whether {phrase}, because the estimate has no uncertainty interval.",
                       sub=[SubVerdict("direction", "not_assessed", "No interval."),
                            SubVerdict("size", "not_assessed", "No interval."), scope, pop, shape, dq],
                       reasons=reasons)

    # 3. identification first ------------------------------------------------------
    if blockers:
        nums = f"(for reference only: estimate {_f(est.estimate)}, 95% CI {_f(ci95[0])} to {_f(ci95[1])})"
        reasons = ["The design cannot separate cause from other explanations here:"] + blockers + reasons
        return Verdict(label="Inconclusive",
                       sentence=f"The data cannot show whether {phrase}, because of a problem with the design "
                                f"({blockers[0][0].lower() + blockers[0][1:].rstrip('.')}).",
                       sub=[SubVerdict("direction", "not_assessed", f"not assessed (identification problem) {nums}"),
                            SubVerdict("size", "not_assessed" if claim.size is not None else "not_claimed",
                                       "not assessed (identification problem)" if claim.size is not None
                                       else "The claim gives no size."),
                            scope, pop, shape, dq],
                       reasons=reasons, caveat=caveat)

    # 4. direction and size --------------------------------------------------------
    dsub = _direction(claim, est, ci90, ci95, margin_sd)
    ssub = _size(claim, est, ci90, ci95)
    reasons.append(dsub.detail)
    if ssub.label not in ("not_claimed",):
        reasons.append(ssub.detail)

    if dsub.label == "contradicted" or ssub.label == "rejected":
        label = "Contradicted"
    elif dsub.label == "supported":
        label = "Supported" if ssub.label in ("supported", "not_claimed") else "Partly supported"
    elif dsub.label == "no_evidence":
        label = "No evidence"
    else:
        label = "Inconclusive"

    caps: list[str] = []
    if scope.label == "capped":
        caps.append(SCOPE_SHORT.get(est.scope, f"only for {est.scope}"))
        reasons.append(scope.detail)
    if pop.label == "not_matched":
        caps.append("on everyone, not the claimed group")
        reasons.append(pop.detail)
    if shape.label in ("not_tested", "not_shown", "rejected") and claim.shape:
        caps.append(f"but the claimed {claim.shape} shape is {shape.label.replace('_', ' ')}")
        reasons.append(shape.detail)
    if label == "Supported" and caps:
        label = "Partly supported"
    if dq.label == "concerns":
        reasons.append("Data quality: " + dq.detail)

    # 5. sentence -----------------------------------------------------------------
    est_txt = _est_text(est, ci90)
    size_claim = _size_text(claim)
    where = (" " + caps[0]) if caps and est.scope != "full" and label != "Contradicted" else ""
    if label in ("Supported", "Partly supported"):
        s = f"The data support that {phrase}{where}"
        if ssub.label == "supported":
            s += f", and the size ({est_txt}) is close to the claimed {size_claim}"
        elif ssub.label == "inconclusive":
            s += f", but the size ({est_txt}) is too uncertain to confirm the claimed {size_claim}"
        elif ssub.label == "not_checkable":
            s += f" ({est_txt}), but the claimed size {size_claim} could not be checked"
        else:
            s += f" ({est_txt})"
        if pop.label == "not_matched":
            s += ", though this was tested on everyone, not the claimed group"
        if claim.shape and shape.label in ("not_tested", "not_shown", "rejected"):
            s += f", and the claimed {claim.shape} shape was not confirmed"
    elif label == "Contradicted":
        if dsub.label == "contradicted":
            s = f"The data contradict that {phrase} ({dsub.detail.rstrip('.')})"
        else:
            s = f"The data contradict the claimed size {size_claim}: the effect is {est_txt}, which rules it out"
    elif label == "No evidence":
        s = f"The data show no clear evidence that {phrase}{where} ({est_txt})"
    else:
        s = (f"The data cannot show whether {phrase}{where}: a meaningful effect cannot be ruled out "
             f"({est_txt}), and not finding an effect is not evidence of no effect")

    sentence = _with_caveat(s, caveat)
    if caveat:
        reasons.append(f"This rests on an assumption the data cannot check: {caveat}.")
    sub = [dsub, ssub, scope, pop, shape, dq]
    return Verdict(label=label, sentence=sentence, sub=sub, reasons=reasons, caveat=caveat)
