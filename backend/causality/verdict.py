"""Turn the design checks and the estimate into one plain-words verdict.

The order matters and follows the one rule (never claim more than the data can show):

1. No claim / untestable claim            -> "Not testable".
2. No estimate / refused estimate         -> "Inconclusive".
3. Identification problems FIRST          -> "Inconclusive" (direction and size not assessed):
   grade D or F, any FAIL assumption (design, declaration conflicts, grade-affecting sections),
   any FAIL grade-affecting section, any section with caps_verdict (e.g. outcome dropout).
4. Direction (scored), then the scope / population / shape caps. The claimed SIZE is NOT scored:
   it never changes the label. It is reported as a note, and the sentence says it was not scored.

Rule choices (documented because the spec left them open):

* The 95% interval is the default for every conclusion. The 90%, 98% and 99% intervals are
  shown next to it (from estimate.extra["ci_levels"], falling back to ci90 / ci95; missing
  levels are simply left out). If the 95% interval excludes zero but the 99% one does not, a
  reason says so.
* Direction uses the two-sided 95% interval (equivalently p < 0.05). An insignificant result
  of either sign is "no_evidence", never a contradiction.
* Negated claims ("X does not reduce Y"): the claim is "no meaningful effect in the named
  direction". Margin m = |stated size| if a size is given (there the size DEFINES the claim, it
  is not a magnitude being scored), else 0.1 x outcome SD (0 if the SD is unknown, which is the
  strictest choice). Contradicted if the 95% interval shows a significant effect in the named
  direction (checked first). Supported if the 95% interval bound on that side is inside the
  margin. Otherwise no_evidence.
* No-effect claims: margin 0.1 x outcome SD. Supported only if the 95% interval lies within
  +-m. That is stricter than the conventional two one-sided tests at 5%, which use the 90%
  interval. Contradicted if the 95% interval excludes 0 AND the point estimate is outside the
  margin. Otherwise inconclusive ("cannot rule out a meaningful effect").
* Size (unscored note). With c the signed claimed effect in outcome units:
  "not_scored: consistent"           c is inside the 95% interval;
  "not_scored: outside the interval" c is outside the 95% interval;
  "not_scored: not checkable"        the size cannot be converted to outcome units;
  "not_claimed"                      no size (or a no-effect / negated claim).
  For "at least c" / "at most c" the label says whether the 95% interval is compatible with
  the bound: "outside the interval" only when the whole interval lies on the wrong side of the
  bound, "consistent" when it lies wholly on the right side or straddles the bound (the detail
  says which).
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
    if a >= 1000 or a < 1:
        return f"{x:.3g}"      # three significant digits, so 0.0184 is not shown as 0.02
    return f"{x:.2f}".rstrip("0").rstrip(".") if a < 10 else f"{x:.1f}".rstrip("0").rstrip(".")


LEVELS = (90, 95, 98, 99)


def _pair(ci) -> Optional[tuple[float, float]]:
    try:
        if ci is None or len(ci) != 2:
            return None
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return (min(lo, hi), max(lo, hi))


def _ci_levels(est: Optional[Estimate]) -> dict[int, tuple[float, float]]:
    """extra["ci_levels"] with integer keys; unreadable entries are dropped."""
    raw = (getattr(est, "extra", None) or {}).get("ci_levels") if est is not None else None
    out: dict[int, tuple[float, float]] = {}
    if not isinstance(raw, dict):
        return out
    for k, v in raw.items():
        try:
            lv = float(str(k).strip().rstrip("%"))
        except ValueError:
            continue
        if 0 < lv < 1:
            lv *= 100
        pair = _pair(v)
        if pair is not None and round(lv) in LEVELS:
            out[int(round(lv))] = pair
    return out


def _ci(est: Optional[Estimate], level: int) -> Optional[tuple[float, float]]:
    """Interval at `level` percent: ci_levels first, then ci90 / ci95. Only the 95% interval
    (the one conclusions rest on) is rebuilt from the standard error when nothing else exists."""
    if est is None:
        return None
    got = _ci_levels(est).get(level)
    if got is not None:
        return got
    if level in (90, 95):
        pair = _pair(est.ci90 if level == 90 else est.ci95)
        if pair is not None:
            return pair
    if level == 95 and est.estimate is not None and est.se is not None and est.se >= 0 \
            and math.isfinite(est.estimate) and math.isfinite(est.se):
        z = stats.norm.ppf(0.975)
        return (est.estimate - z * est.se, est.estimate + z * est.se)
    return None


def interval_table(estimate: Optional[Estimate]) -> list[dict]:
    """[{"level": 90, "lo": .., "hi": .., "excludes_zero": bool}, ...] for the levels available."""
    rows = []
    for lv in LEVELS:
        ci = _ci(estimate, lv)
        if ci is None:
            continue
        rows.append({"level": lv, "lo": ci[0], "hi": ci[1], "excludes_zero": bool(ci[0] > 0 or ci[1] < 0)})
    return rows


def _intervals_text(est: Estimate) -> str:
    rows = interval_table(est)
    if not rows:
        return ""
    parts = "; ".join(f"{r['level']}% {_f(r['lo'])} to {_f(r['hi'])}" for r in rows)
    return f"Intervals: {parts}. The conclusion uses the 95% interval."


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


def _est_text(est: Estimate, ci95) -> str:
    s = f"about {_f(est.estimate)}"
    if ci95:
        s += f", 95% interval {_f(ci95[0])} to {_f(ci95[1])}"
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
def _direction(claim: Claim, est: Estimate, ci95, margin_sd: float) -> SubVerdict:
    """Every conclusion here rests on the 95% interval."""
    d = claim.direction
    sd = est.outcome_sd
    lo95, hi95 = ci95
    ci90 = _ci(est, 90)
    pt = est.estimate
    sig = lo95 > 0 or hi95 < 0
    ci_txt = f"estimate {_f(pt)}, 95% interval {_f(lo95)} to {_f(hi95)}"

    if d == "no_effect" or (claim.negated and d not in ("increase", "decrease")):
        if not sd:
            return SubVerdict("direction", "inconclusive",
                              "The outcome spread is unknown, so a 'meaningful effect' margin cannot be set; "
                              "an insignificant result is not evidence of no effect.")
        m = margin_sd * sd
        strict = ("This is stricter than the conventional two one-sided tests at 5%, "
                  "which use the 90% interval.")
        if lo95 >= -m and hi95 <= m:
            return SubVerdict("direction", "supported",
                              f"The 95% interval ({_f(lo95)} to {_f(hi95)}) lies within "
                              f"+-{_f(m)} ({margin_sd:g} SD), so any effect is small. {strict}")
        if sig and abs(pt) > m:
            return SubVerdict("direction", "contradicted",
                              f"There is a clear effect ({ci_txt}), larger than the {_f(m)} margin.")
        note = ""
        if ci90 is not None and ci90[0] >= -m and ci90[1] <= m:
            note = (f" The 90% interval ({_f(ci90[0])} to {_f(ci90[1])}) is inside the margin, so the "
                    "conventional test would pass, but the 95% interval is required here.")
        return SubVerdict("direction", "inconclusive",
                          f"Cannot rule out a meaningful effect: the 95% interval ({_f(lo95)} to {_f(hi95)}) is not "
                          f"inside +-{_f(m)} ({margin_sd:g} SD). Not finding an effect is not the same as showing "
                          f"there is none. {strict}{note}")

    if claim.negated:
        s = 1.0 if d == "increase" else -1.0     # direction the claim says does NOT happen
        c = claimed_effect(claim, est)
        if c is not None and c != 0:
            m, mtxt = abs(c), f"the stated {_f(abs(c))}"
        elif sd:
            m, mtxt = margin_sd * sd, f"{margin_sd:g} SD ({_f(margin_sd * sd)})"
        else:
            m, mtxt = 0.0, "zero (outcome spread unknown)"
        word = "rise" if s > 0 else "drop"
        if (s > 0 and lo95 > 0) or (s < 0 and hi95 < 0):
            return SubVerdict("direction", "contradicted", f"There is a significant {word} ({ci_txt}).")
        bound = hi95 if s > 0 else lo95           # the 95% bound on the claimed-against side
        if s * bound < m:
            return SubVerdict("direction", "supported",
                              f"A {word} bigger than {mtxt} is ruled out (95% interval bound {_f(bound)}; {ci_txt}).")
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


def _level_note(claim: Claim, est: Estimate, dsub: SubVerdict, margin_sd: float) -> Optional[str]:
    """Would the direction conclusion differ at the 99% level?"""
    ci95, ci99 = _ci(est, 95), _ci(est, 99)
    if ci95 is None or ci99 is None or dsub.label not in ("supported", "contradicted"):
        return None
    sig95 = ci95[0] > 0 or ci95[1] < 0
    sig99 = ci99[0] > 0 or ci99[1] < 0
    by_margin = dsub.label == "supported" and (claim.direction == "no_effect" or claim.negated)
    if by_margin:
        # this conclusion rests on the interval staying inside a margin, not on excluding zero
        again = _direction(claim, Estimate(method=est.method, target=est.target, estimate=est.estimate,
                                           ci95=ci99, baseline_mean=est.baseline_mean,
                                           outcome_sd=est.outcome_sd), ci99, margin_sd)
        if again.label != dsub.label:
            return "This conclusion holds with the 95% interval but not with the wider 99% interval."
        return None
    if sig95 and not sig99:
        return "The effect is clear at the 95% level but not at the 99% level."
    return None


# ----------------------------------------------------------------------------- size (unscored note)
SIZE_TAIL = " The verdict rests on the direction only and does not confirm the claimed size."


def _size(claim: Claim, est: Estimate, ci95) -> tuple[SubVerdict, str]:
    """The size is NOT scored. Returns the note and the clause for the verdict sentence
    ("" when no size was claimed)."""
    if claim.direction == "no_effect":
        return SubVerdict("size", "not_claimed",
                          "A no-effect claim is checked against the margin in the direction check."), ""
    if claim.size is None:
        return SubVerdict("size", "not_claimed", "The claim gives no size."), ""
    if claim.negated:
        return SubVerdict("size", "not_claimed",
                          f"The stated size ({_size_text(claim)}) is used as the 'meaningful effect' margin "
                          "in the direction check; it is not scored as a size."), ""
    lo, hi = ci95
    iv = f"{_f(lo)} to {_f(hi)}"
    stxt = _size_text(claim)
    nums = f"Estimate {_f(est.estimate)}, 95% interval {iv}."
    c = claimed_effect(claim, est)
    if c is None:
        why = {"percent": "a percent change needs a non-zero baseline (control) mean",
               "times": "a 'times' change needs a baseline (control) mean",
               "sd": "the outcome standard deviation is unknown"}.get(claim.size_unit or "",
                                                                     "the size could not be turned into outcome units")
        return (SubVerdict("size", "not_scored: not checkable",
                           f"Not scored. The claimed size ({stxt}) cannot be put in outcome units: {why}. "
                           f"{nums}{SIZE_TAIL}"),
                f"; the claimed size ({stxt}) was not scored \u2014 it could not be converted to outcome units")
    if claim.size_unit == "absolute":
        bword = {"at_least": "at least ", "at_most": "at most "}.get(claim.bound or "", "")
        shown = f"{bword}{_f(c)}"
    else:
        shown = f"{stxt} \u2248 {_f(c)}"
    head = f"Not scored. The claimed size ({stxt}) is {_f(c)} in outcome units. {nums} "

    if claim.bound in ("at_least", "at_most") and c != 0:
        s = 1.0 if c > 0 else -1.0
        l, h = sorted((s * lo, s * hi))          # along the claimed direction
        a = abs(c)
        if claim.bound == "at_least":
            good, bad = l >= a, h < a
            good_txt = "lies entirely at or beyond that bound, which is compatible with it"
            bad_txt = "falls entirely short of that bound, which is not compatible with it"
        else:
            good, bad = h <= a, l > a
            good_txt = "lies entirely within that bound, which is compatible with it"
            bad_txt = "lies entirely beyond that bound, which is not compatible with it"
        if bad:
            label, what = "not_scored: outside the interval", bad_txt
        elif good:
            label, what = "not_scored: consistent", good_txt
        else:
            label, what = "not_scored: consistent", ("contains that bound, so the data neither confirm "
                                                     "nor rule it out")
        return (SubVerdict("size", label, f"{head}The 95% interval {what}.{SIZE_TAIL}"),
                f"; the claimed size ({shown}) was not scored \u2014 the 95% interval ({iv}) {what}")

    if lo <= c <= hi:
        return (SubVerdict("size", "not_scored: consistent",
                           f"{head}The claimed value is inside the 95% interval, so it is consistent with the "
                           f"data, but so is every other value in that interval.{SIZE_TAIL}"),
                f"; the claimed size ({shown}) was not scored \u2014 it is consistent with the 95% interval ({iv})")
    return (SubVerdict("size", "not_scored: outside the interval",
                       f"{head}The claimed value is outside the 95% interval.{SIZE_TAIL}"),
            f"; the claimed size ({shown}) was not scored \u2014 it lies outside the 95% interval ({iv})")


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
    ci95 = _ci(est, 95)
    if est.estimate is None or ci95 is None:
        reasons.append("The estimate has no uncertainty interval, so nothing can be concluded from it.")
        reasons.extend(blockers)
        return Verdict(label="Inconclusive",
                       sentence=f"The data cannot say whether {phrase}, because the estimate has no uncertainty interval.",
                       sub=[SubVerdict("direction", "not_assessed", "No interval."),
                            SubVerdict("size", "not_assessed", "No interval."), scope, pop, shape, dq],
                       reasons=reasons)

    # 3. identification first ------------------------------------------------------
    if blockers:
        nums = f"(for reference only: estimate {_f(est.estimate)}, 95% interval {_f(ci95[0])} to {_f(ci95[1])})"
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

    # 4. direction (scored) and size (a note only) -------------------------------------
    dsub = _direction(claim, est, ci95, margin_sd)
    core = dsub.detail                       # without the interval list, for the sentence
    ssub, size_clause = _size(claim, est, ci95)
    reasons.append(core)
    level_note = _level_note(claim, est, dsub, margin_sd)
    if level_note:
        reasons.append(level_note)
    if ssub.label.startswith("not_scored"):
        reasons.append("Size: " + ssub.detail)
    itxt = _intervals_text(est)
    if itxt:
        dsub = SubVerdict("direction", dsub.label, f"{core} {itxt}")

    # the label comes from the direction only; the claimed size never changes it
    label = {"contradicted": "Contradicted", "supported": "Supported",
             "no_evidence": "No evidence"}.get(dsub.label, "Inconclusive")

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
    est_txt = _est_text(est, ci95)
    where = (" " + caps[0]) if caps and est.scope != "full" and label != "Contradicted" else ""
    if label in ("Supported", "Partly supported"):
        s = f"The data support that {phrase}{where} ({est_txt})"
        if pop.label == "not_matched":
            s += ", though this was tested on everyone, not the claimed group"
        if claim.shape and shape.label in ("not_tested", "not_shown", "rejected"):
            s += f", and the claimed {claim.shape} shape was not confirmed"
    elif label == "Contradicted":
        s = f"The data contradict that {phrase} ({core.rstrip('.')})"
    elif label == "No evidence":
        s = f"The data show no clear evidence that {phrase}{where} ({est_txt})"
    else:
        s = (f"The data cannot show whether {phrase}{where}: a meaningful effect cannot be ruled out "
             f"({est_txt}), and not finding an effect is not evidence of no effect")
    s += size_clause                         # says the size was not scored, whenever one was claimed

    sentence = _with_caveat(s, caveat)
    if caveat:
        reasons.append(f"This rests on an assumption the data cannot check: {caveat}.")
    sub = [dsub, ssub, scope, pop, shape, dq]
    return Verdict(label=label, sentence=sentence, sub=sub, reasons=reasons, caveat=caveat)
