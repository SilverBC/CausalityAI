import math

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from causality import sim
from causality.data import prepare
from causality.schema import (Assumption, Claim, Design, Estimate, Roles, Section, Status)
from causality.suggest import n_per_arm, suggest_study
from causality.verdict import interval_table, make_verdict


def claim(**kw):
    base = dict(raw="treated raises score", treatment_phrase="treated", outcome_phrase="score",
                treatment_col="treated", outcome_col="score", direction="increase")
    base.update(kw)
    return Claim(**base)


def est(point=1.5, se=0.1, scope="full", sd=3.0, base=10.0, **kw):
    z90, z95 = stats.norm.ppf(0.95), stats.norm.ppf(0.975)
    return Estimate(method="Difference in means", target="ATE", scope=scope, estimate=point, se=se,
                    ci90=(point - z90 * se, point + z90 * se), ci95=(point - z95 * se, point + z95 * se),
                    p_value=2 * stats.norm.sf(abs(point / se)), n=2000, baseline_mean=base, outcome_sd=sd, **kw)


def levels(point, se, keep=("90", "95", "98", "99")):
    out = {str(lv): [point - stats.norm.ppf(0.5 + lv / 200) * se, point + stats.norm.ppf(0.5 + lv / 200) * se]
           for lv in (90, 95, 98, 99)}
    return {k: v for k, v in out.items() if k in keep}


def design(kind="experiment", grade="A", **kw):
    return Design(kind=kind, grade=grade, **kw)


def sub(v, aspect):
    return next(s for s in v.sub if s.aspect == aspect)


# ----------------------------------------------------------------------------- testability
def test_no_claim_and_untestable():
    assert make_verdict(None, design(), est(), []).label == "Not testable"
    v = make_verdict(claim(testable=False, untestable_reason="mediation"), design(), est(), [])
    assert v.label == "Not testable" and "mechanism" in v.sentence
    assert [s.aspect for s in v.sub] == ["direction", "size", "scope", "population", "shape", "data_quality"]


def test_refused_estimate():
    e = Estimate(method="2SLS", target="LATE", refused=True, refusal_reason="the instrument is too weak")
    v = make_verdict(claim(), design("iv"), e, [])
    assert v.label == "Inconclusive" and "too weak" in v.sentence


# ----------------------------------------------------------------------------- identification first
def test_grade_d_blocks_huge_effect():
    v = make_verdict(claim(), design(grade="D", grade_reasons=["no balance"]), est(point=50, se=0.1), [])
    assert v.label == "Inconclusive"
    assert sub(v, "direction").label == "not_assessed"
    assert "identification" in sub(v, "direction").detail


def test_caps_verdict_blocks():
    s = Section(key="missing_outcomes", title="Missing outcomes", status=Status.CONCERN,
                summary="Treated people dropped out more.", caps_verdict=True)
    v = make_verdict(claim(), design(), est(point=5, se=0.1), [s])
    assert v.label == "Inconclusive"
    assert any("dropped out" in r for r in v.reasons)


def test_fail_assumption_in_section_blocks_but_literature_does_not():
    bad = Section(key="balance", title="Balance", status=Status.CONCERN,
                  assumptions=[Assumption("x", "Groups comparable", Status.FAIL, "very different")])
    assert make_verdict(claim(), design(), est(), [bad]).label == "Inconclusive"
    lit = Section(key="literature", title="Literature", status=Status.FAIL, affects_grade=False)
    assert make_verdict(claim(), design(), est(), [lit]).label == "Supported"
    conf = design(conflicts=[Assumption("c", "Declared randomized", Status.FAIL, "groups differ")])
    assert make_verdict(claim(), conf, est(), []).label == "Inconclusive"


# ----------------------------------------------------------------------------- direction
def test_supported_direction():
    v = make_verdict(claim(), design(), est(point=1.5, se=0.1), [])
    assert v.label == "Supported" and sub(v, "direction").label == "supported"
    assert v.sentence.count(".") >= 1 and v.caveat is None


def test_insignificant_opposite_is_no_evidence():
    v = make_verdict(claim(), design(), est(point=-0.1, se=0.2), [])
    assert v.label == "No evidence"
    assert sub(v, "direction").label == "no_evidence"


def test_significant_opposite_contradicted():
    v = make_verdict(claim(), design(), est(point=-1.0, se=0.2), [])
    assert v.label == "Contradicted"


def test_nonzero():
    assert make_verdict(claim(direction="nonzero"), design(), est(point=-1, se=0.2), []).label == "Supported"
    assert make_verdict(claim(direction="nonzero"), design(), est(point=-0.1, se=0.2), []).label == "No evidence"


def test_negated_claim():
    c = claim(direction="decrease", negated=True, raw="treated does not reduce score")
    # tight around zero: decrease beyond 0.1 SD (0.3) ruled out
    assert make_verdict(c, design(), est(point=0.0, se=0.05), []).label == "Supported"
    # significantly negative
    assert make_verdict(c, design(), est(point=-1.0, se=0.1), []).label == "Contradicted"
    # wide interval
    assert make_verdict(c, design(), est(point=0.0, se=1.0), []).label == "No evidence"


def test_negated_claim_uses_95_bound():
    c = claim(direction="decrease", negated=True, raw="treated does not reduce score")
    # margin 0.3: the 90% lower bound (-0.28) is inside it, the 95% one (-0.33) is not
    e = est(point=0.0, se=0.17)
    assert e.ci90[0] > -0.3 > e.ci95[0]
    v = make_verdict(c, design(), e, [])
    assert v.label == "No evidence" and sub(v, "direction").label == "no_evidence"
    # 95% bound inside the margin -> supported, and the detail names the 95% bound
    v2 = make_verdict(c, design(), est(point=0.0, se=0.15), [])
    assert v2.label == "Supported" and "95% interval bound" in sub(v2, "direction").detail
    # a stated size is the margin for a negated claim, not a scored size
    c3 = claim(direction="decrease", negated=True, size=1, size_unit="absolute")
    v3 = make_verdict(c3, design(), e, [])
    assert v3.label == "Supported" and sub(v3, "size").label == "not_claimed"


# ----------------------------------------------------------------------------- size (never scored)
def size_checks(v, label):
    s = sub(v, "size")
    assert s.label == label and s.label.startswith("not_scored")
    assert s.detail.startswith("Not scored.")
    assert "was not scored" in v.sentence
    assert any("Not scored." in r for r in v.reasons)
    return s


def test_size_consistent_is_a_note_only():
    c = claim(size=15, size_unit="percent")          # 15% of baseline 10 = 1.5
    v = make_verdict(c, design(), est(point=1.5, se=0.1, base=10), [])
    assert v.label == "Supported" and sub(v, "direction").label == "supported"
    s = size_checks(v, "not_scored: consistent")
    assert "1.5 in outcome units" in s.detail and "95% interval 1.3 to 1.7" in s.detail
    assert "; the claimed size (15% \u2248 1.5) was not scored \u2014 it is consistent with the 95% interval (1.3 to 1.7)" in v.sentence
    assert "close to" not in v.sentence and "confirm" not in v.sentence


def test_size_far_outside_interval_does_not_change_label():
    c = claim(size=60, size_unit="percent")          # claimed 6.0, 95% interval ~1.3..1.7
    v = make_verdict(c, design(), est(point=1.5, se=0.1, base=10), [])
    assert v.label == "Supported" and sub(v, "direction").label == "supported"
    size_checks(v, "not_scored: outside the interval")
    assert "(60% \u2248 6) was not scored \u2014 it lies outside the 95% interval (1.3 to 1.7)" in v.sentence
    # same data, no size claimed: same label, nothing about size in the sentence
    v0 = make_verdict(claim(), design(), est(point=1.5, se=0.1, base=10), [])
    assert v0.label == v.label and "size" not in v0.sentence
    assert sub(v0, "size").label == "not_claimed"


def test_size_wide_interval_still_direction_only():
    c = claim(size=15, size_unit="percent")
    v = make_verdict(c, design(), est(point=1.5, se=0.6, base=10), [])
    assert v.label == "Supported"
    size_checks(v, "not_scored: consistent")


def test_size_does_not_rescue_or_sink_other_labels():
    c = claim(size=15, size_unit="percent")
    v = make_verdict(c, design(), est(point=0.1, se=0.2, base=10), [])     # insignificant
    assert v.label == "No evidence"
    size_checks(v, "not_scored: outside the interval")
    v = make_verdict(c, design(), est(point=-1.0, se=0.2, base=10), [])    # significant, opposite sign
    assert v.label == "Contradicted"
    size_checks(v, "not_scored: outside the interval")


def test_percent_conversion_uses_baseline():
    c = claim(size=15, size_unit="percent")
    # same estimate, baseline 100 -> claimed 15, far outside the interval; label unchanged
    v = make_verdict(c, design(), est(point=1.5, se=0.1, base=100), [])
    assert v.label == "Supported"
    assert "15 in outcome units" in size_checks(v, "not_scored: outside the interval").detail
    # baseline 0 -> not checkable, and that no longer downgrades the label
    v = make_verdict(c, design(), est(point=1.5, se=0.1, base=0.0), [])
    assert v.label == "Supported"
    size_checks(v, "not_scored: not checkable")
    assert "could not be converted" in v.sentence


def test_decrease_size_signed():
    c = claim(direction="decrease", size=2, size_unit="absolute")
    v = make_verdict(c, design(), est(point=-2.0, se=0.2), [])
    assert v.label == "Supported"
    assert "-2 in outcome units" in size_checks(v, "not_scored: consistent").detail


def test_at_least_and_at_most():
    c = claim(size=1.0, size_unit="absolute", bound="at_least")
    v = make_verdict(c, design(), est(point=1.5, se=0.1), [])
    assert v.label == "Supported"
    assert "compatible with it" in size_checks(v, "not_scored: consistent").detail
    v = make_verdict(c, design(), est(point=0.5, se=0.1), [])
    assert v.label == "Supported"                    # direction is clear; the bound is only a note
    s = size_checks(v, "not_scored: outside the interval")
    assert "not compatible" in s.detail and "falls entirely short" in v.sentence
    v = make_verdict(c, design(), est(point=1.0, se=0.2), [])
    assert "neither confirm nor rule it out" in size_checks(v, "not_scored: consistent").detail
    c2 = claim(size=1.0, size_unit="absolute", bound="at_most")
    v = make_verdict(c2, design(), est(point=0.8, se=0.05), [])
    assert "compatible with it" in size_checks(v, "not_scored: consistent").detail
    v = make_verdict(c2, design(), est(point=2.0, se=0.1), [])
    assert v.label == "Supported"
    assert "not compatible" in size_checks(v, "not_scored: outside the interval").detail


def test_size_not_assessed_when_identification_fails():
    c = claim(size=15, size_unit="percent")
    v = make_verdict(c, design(grade="F"), est(), [])
    assert v.label == "Inconclusive" and sub(v, "size").label == "not_assessed"


# ----------------------------------------------------------------------------- no effect
def test_equivalence():
    c = claim(direction="no_effect", raw="treated has no effect on score")
    # margin 0.1*3 = 0.3; tight 95% interval around 0 -> supported
    v = make_verdict(c, design(), est(point=0.0, se=0.05), [])
    assert v.label == "Supported"
    assert "95% interval" in sub(v, "direction").detail and "stricter" in sub(v, "direction").detail
    assert sub(v, "size").label == "not_claimed" and "not scored" not in v.sentence
    # insignificant but wide -> NOT support
    v = make_verdict(c, design(), est(point=0.0, se=0.5), [])
    assert v.label == "Inconclusive" and sub(v, "direction").label == "inconclusive"
    # clear, big effect -> contradicted
    assert make_verdict(c, design(), est(point=1.0, se=0.1), []).label == "Contradicted"


def test_equivalence_needs_the_95_interval_inside_the_margin():
    c = claim(direction="no_effect", raw="treated has no effect on score")
    e = est(point=0.0, se=0.17)                      # 90%: +-0.28 (inside 0.3); 95%: +-0.33 (not inside)
    assert e.ci90[1] < 0.3 < e.ci95[1]
    v = make_verdict(c, design(), e, [])
    assert v.label == "Inconclusive" and sub(v, "direction").label == "inconclusive"
    d = sub(v, "direction").detail
    assert "stricter" in d and "90% interval" in d
    assert make_verdict(c, design(), est(point=0.0, se=0.15), []).label == "Supported"


# ----------------------------------------------------------------------------- interval levels
def test_sentence_and_details_quote_the_95_interval():
    v = make_verdict(claim(), design(), est(point=1.5, se=0.1), [])
    assert "95% interval 1.3 to 1.7" in v.sentence and "90%" not in v.sentence
    d = sub(v, "direction").detail
    assert "Intervals: 90% 1.34 to 1.66; 95% 1.3 to 1.7." in d      # no ci_levels: falls back to ci90 / ci95
    assert d.endswith("The conclusion uses the 95% interval.")


def test_interval_table_full_partial_missing():
    full = interval_table(est(point=1.0, se=0.45, extra={"ci_levels": levels(1.0, 0.45)}))
    assert [r["level"] for r in full] == [90, 95, 98, 99]
    assert [r["excludes_zero"] for r in full] == [True, True, False, False]
    assert full[1]["lo"] == pytest.approx(1.0 - 1.959964 * 0.45) and full[3]["hi"] == pytest.approx(1.0 + 2.575829 * 0.45)
    assert set(full[0]) == {"level", "lo", "hi", "excludes_zero"}

    part = interval_table(est(point=1.0, se=0.45, extra={"ci_levels": levels(1.0, 0.45, keep=("98",))}))
    assert [r["level"] for r in part] == [90, 95, 98]             # 90 / 95 from ci90 / ci95, 99 left out
    assert part[0]["lo"] == pytest.approx(1.0 - 1.644854 * 0.45)

    assert [r["level"] for r in interval_table(est())] == [90, 95]
    bad = est(extra={"ci_levels": {"99": [None, 2.0], "98": "x", "97": [0, 1]}})
    assert [r["level"] for r in interval_table(bad)] == [90, 95]
    assert interval_table(None) == []
    assert interval_table(Estimate(method="m", target="ATE", refused=True)) == []

    v = make_verdict(claim(), design(), est(point=1.0, se=0.45, extra={"ci_levels": levels(1.0, 0.45)}), [])
    d = sub(v, "direction").detail
    assert all(f"{lv}% " in d for lv in (90, 95, 98, 99)) and "The conclusion uses the 95% interval." in d


def test_reason_when_99_disagrees():
    msg = "The effect is clear at the 95% level but not at the 99% level."
    e = est(point=1.0, se=0.45, extra={"ci_levels": levels(1.0, 0.45)})       # z = 2.2
    v = make_verdict(claim(), design(), e, [])
    assert v.label == "Supported" and msg in v.reasons
    # clear at 99% too -> no such reason
    e2 = est(point=1.5, se=0.1, extra={"ci_levels": levels(1.5, 0.1)})
    assert msg not in make_verdict(claim(), design(), e2, []).reasons
    # no 99% interval available -> nothing is said about it
    assert msg not in make_verdict(claim(), design(), est(point=1.0, se=0.45), []).reasons
    # not clear at 95% either -> no such reason
    e3 = est(point=0.5, se=0.45, extra={"ci_levels": levels(0.5, 0.45)})
    assert msg not in make_verdict(claim(), design(), e3, []).reasons


# ----------------------------------------------------------------------------- scope / population / shape
def test_scope_cap_late():
    e = est(point=1.5, se=0.1, scope="compliers")
    v = make_verdict(claim(), design("iv"), e, [])
    assert v.label == "Partly supported"
    assert sub(v, "scope").label == "capped" and "compliers" in v.sentence


def test_population_not_matched_caps():
    c = claim(population_phrase="among women")
    v = make_verdict(c, design(), est(), [])
    assert sub(v, "population").label == "not_matched" and v.label == "Partly supported"
    c2 = claim(population_phrase="among women", population_filter={"column": "female", "op": "==", "value": 1},
               population_score=0.9)
    v2 = make_verdict(c2, design(), est(), [])
    assert sub(v2, "population").label == "matched" and v2.label == "Supported"


def test_shape():
    c = claim(shape="linear")
    assert sub(make_verdict(c, design(), est(), []), "shape").label == "not_tested"
    e = est(extra={"shape_tests": {"linear_p": 0.6}})
    v = make_verdict(c, design(), e, [])
    assert sub(v, "shape").label == "consistent" and v.label == "Supported"


def test_observational_sentence_and_hedge():
    v = make_verdict(claim(hedges=["may"]), design("observational", grade="B"), est(), [])
    assert v.sentence.endswith("if no unmeasured confounding.")
    assert v.caveat == "if no unmeasured confounding"
    assert any("hedged" in r for r in v.reasons)


def test_data_quality():
    s = Section(key="overlap", title="Overlap", status=Status.CONCERN, summary="Some poor overlap.")
    v = make_verdict(claim(), design(), est(), [s])
    assert sub(v, "data_quality").label == "concerns" and v.label == "Supported"


# ----------------------------------------------------------------------------- suggest_study
@pytest.fixture(scope="module")
def rct_prep():
    df, _ = sim.rct(n=600, seed=0)
    return prepare(df, Roles(treatment="treated", outcome="score", randomized=True))


def test_n_formula_matches_hand(rct_prep):
    z = 1.959963985 + 0.841621234
    assert n_per_arm(1.5, 3.0) == pytest.approx(2 * z ** 2 * 9 / 2.25, rel=1e-6)
    s = suggest_study(rct_prep, design(), claim(size=1.5, size_unit="absolute"), est(sd=3.0))
    assert s.key == "next_study" and s.status is None and s.error is None
    assert s.metrics["n_per_arm"] == math.ceil(2 * z ** 2 * 9 / 2.25)
    assert s.metrics["n_total"] == 2 * s.metrics["n_per_arm"]
    assert s.charts and s.charts[0].kind == "line"


def test_power_uses_claimed_not_observed(rct_prep):
    c = claim(size=1.5, size_unit="absolute")
    a = suggest_study(rct_prep, design(), c, est(point=1.5, sd=3.0)).metrics["n_total"]
    b = suggest_study(rct_prep, design(), c, est(point=0.2, sd=3.0)).metrics["n_total"]
    assert a == b
    d = suggest_study(rct_prep, design(), claim(size=0.75, size_unit="absolute"), est(sd=3.0)).metrics["n_total"]
    assert d > 3 * a


def test_no_size_gives_range(rct_prep):
    s = suggest_study(rct_prep, design("observational"), claim(), est(sd=3.0))
    assert s.metrics["n_per_arm"] is None
    assert [r["effect_sd"] for r in s.metrics["n_by_effect_size"]] == [0.1, 0.2, 0.3]
    assert "size" in s.summary.lower()


def test_equivalence_power(rct_prep):
    s = suggest_study(rct_prep, design(), claim(direction="no_effect"), est(sd=3.0))
    z = stats.norm.ppf(0.95) + stats.norm.ppf(0.9)
    assert s.metrics["n_per_arm"] == math.ceil(2 * z ** 2 / 0.01)


def test_attrition_and_cluster_inflate():
    df, _ = sim.rct(n=1000, seed=1, attrition=0.2)
    p = prepare(df, Roles(treatment="treated", outcome="score", randomized=True))
    s = suggest_study(p, design(), claim(size=1.5, size_unit="absolute"), est(sd=3.0))
    base = n_per_arm(1.5, 3.0)
    att = s.metrics["assumptions"]["attrition_rate"]
    assert 0.1 < att < 0.3 and s.metrics["n_per_arm"] > base

    dfc, _ = sim.cluster_saturation()
    pc = prepare(dfc, Roles(treatment="treated", outcome="consumption", cluster="village"))
    sc = suggest_study(pc, design(), claim(size=0.5, size_unit="absolute", outcome_col="consumption"),
                       est(sd=float(dfc["consumption"].std())))
    a = sc.metrics["assumptions"]
    assert a["icc"] is not None and a["icc"] > 0.05 and a["design_effect"] > 1.5
