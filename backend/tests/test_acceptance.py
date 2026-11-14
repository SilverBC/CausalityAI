"""End-to-end known-truth simulations through the whole pipeline. A part is accepted only
when these pass."""
import json

import pytest

from causality import pipeline, sim
from causality.schema import Status, to_jsonable

GRADES = "ABCDEF"


def run(name, claim, design=None, roles=None, **kw):
    df, truth = sim.ALL[name](**kw)
    rep = pipeline.run(df, claim_text=claim, design_decl=design, roles=roles or {})
    json.dumps(to_jsonable(rep))  # must serialise
    return rep, truth


def worse_or_equal(g, ref):
    return GRADES.index(g) >= GRADES.index(ref)


def covers(est, truth):
    lo, hi = est.ci95
    return lo <= truth <= hi


def test_rct_supported():
    rep, tr = run("rct", "Being treated increases score", design="experiment")
    assert not rep.needs_input, rep.needs_input
    assert rep.design.kind == "experiment"
    assert rep.design.grade in ("A", "B")
    assert covers(rep.estimate, tr["effect"])
    assert rep.verdict.label.startswith("Supported")


def test_size_is_an_unscored_note():
    rep, tr = run("rct", "Being treated increases score by 60%", design="experiment")
    size = next(s for s in rep.verdict.sub if s.aspect == "size")
    assert size.label.startswith("not_scored")
    assert "outside" in size.label
    assert rep.verdict.label.startswith("Supported")      # the label follows direction only
    assert "not scored" in rep.verdict.sentence.lower()   # and says the size was not assessed


def test_four_interval_levels():
    rep, _ = run("rct", "Being treated increases score", design="experiment")
    lv = rep.estimate.extra["ci_levels"]
    assert set(lv) == {"90", "95", "98", "99"}
    assert lv["99"][0] < lv["98"][0] < lv["95"][0] < lv["90"][0]


def test_rct_differential_dropout_inconclusive():
    rep, _ = run("rct", "Being treated increases score", design="experiment", attrition=0.25, differential=0.25)
    assert rep.verdict.label == "Inconclusive"


def test_null_rct_equivalence_and_no_evidence():
    # "no effect" needs the 95% interval inside +-0.1 SD: stricter than the usual 90% rule,
    # so it takes a large sample to support it ...
    rep, _ = run("null_rct", "Being treated has no effect on score", design="experiment", n=20000)
    assert rep.verdict.label.startswith("Supported")
    # ... and a smaller one is honestly inconclusive, never "supported because insignificant"
    rep, _ = run("null_rct", "Being treated has no effect on score", design="experiment", n=1000)
    assert rep.verdict.label == "Inconclusive"
    rep, _ = run("null_rct", "Being treated increases score", design="experiment")
    assert rep.verdict.label == "No evidence"


def test_post_treatment_column_flagged_under_randomization():
    rep, _ = run("rct_post", "Being treated increases score", design="experiment")
    assert "engagement" not in rep.data["adjustment_set"]
    assert "engagement" in rep.data["roles"].post_treatment


def test_observational_capped_and_caveated():
    rep, tr = run("observational", "The program increases earnings by 2")
    assert rep.design.kind == "observational"
    assert worse_or_equal(rep.design.grade, "C")
    assert covers(rep.estimate, tr["effect"])
    assert rep.verdict.sentence.rstrip(".").endswith("if no unmeasured confounding")
    keys = [a.key for a in rep.assumed_panel]
    assert any("confound" in k for k in keys)
    hc = next(s for s in rep.sections if s.key == "hidden_confounding")
    assert hc.status != Status.PASS


def test_hidden_confounding_never_passes():
    rep, _ = run("observational_hidden", "Coffee improves health")
    assert worse_or_equal(rep.design.grade, "C")
    for s in rep.sections:
        if s.key == "hidden_confounding":
            assert s.status != Status.PASS
    assert all(a.status == Status.ASSUMED for a in rep.assumed_panel)


def test_did_staggered():
    rep, tr = run("did", "The policy increases sales")
    assert rep.design.kind == "did"
    assert covers(rep.estimate, tr["effect"])
    assert rep.estimate.target == "ATT"
    assert rep.verdict.label != "Supported"  # capped: speaks only for the treated
    assert "Supported" in rep.verdict.label or "Partly" in rep.verdict.label


def test_did_pretrend_inconclusive():
    rep, _ = run("did", "The policy increases sales", pretrend=0.6)
    assert rep.verdict.label == "Inconclusive"


def test_iv():
    rep, tr = run("iv", "Enrolling raises test score")
    assert rep.design.kind == "iv"
    assert covers(rep.estimate, tr["effect"])
    assert rep.estimate.target == "LATE"
    assert rep.verdict.label not in ("Supported",)  # compliers only


def test_weak_iv_inconclusive():
    rep, _ = run("iv", "Enrolling raises test score", design="iv",
                 roles={"instrument": "lottery", "treatment": "enrolled", "outcome": "test_score"}, strength=0.02)
    assert rep.verdict.label == "Inconclusive"


def test_rd():
    rep, tr = run("rd", "The scholarship increases gpa later")
    assert rep.design.kind == "rd"
    assert covers(rep.estimate, tr["effect"])


def test_rd_manipulation_inconclusive():
    rep, _ = run("rd", "The scholarship increases gpa later", design="rd",
                 roles={"running": "exam_score", "cutoff": 50, "treatment": "scholarship", "outcome": "gpa_later"},
                 manipulate=True)
    assert rep.verdict.label == "Inconclusive"


def test_dose_response():
    rep, tr = run("dose", "More study hours increase grade")
    assert covers(rep.estimate, tr["effect"])
    assert rep.verdict.label.startswith("Supported") or "Partly" in rep.verdict.label


def test_spillover_detected_with_randomized_saturation():
    rep, tr = run("cluster_saturation", "Treatment increases consumption", design="experiment",
                  roles={"cluster": "village", "saturation": "saturation", "treatment": "treated",
                         "outcome": "consumption"})
    inter = next(s for s in rep.sections if s.key == "interference")
    assert inter.status in (Status.CONCERN, Status.FAIL)


def test_interference_assumed_without_saturation():
    rep, _ = run("rct", "Being treated increases score", design="experiment")
    inter = next(s for s in rep.sections if s.key == "interference")
    assert inter.status == Status.ASSUMED


def test_untestable_claim():
    rep, _ = run("rct", "Being treated increases score through motivation", design="experiment")
    assert rep.verdict.label == "Not testable"


def test_low_match_asks_for_confirmation():
    rep, _ = run("rct", "The intervention boosts wellbeing")
    assert rep.needs_input and rep.needs_input[0]["kind"] == "confirm_column"


def test_explore_chain():
    rep, tr = run("chain", "")
    assert rep.mode == "explore"
    ex = rep.explore
    assert ex is not None and ex.error is None
    dag = next(c for c in ex.charts if c.kind == "dag")
    pairs = {frozenset((e["from"], e["to"])) for e in dag.data["edges"]}
    for a, b in tr["edges"]:
        assert frozenset((a, b)) in pairs
    assert not any("E" in p for p in pairs)


def test_assumed_never_marked_pass():
    rep, _ = run("observational", "The program increases earnings")
    for a in rep.assumed_panel:
        assert a.status == Status.ASSUMED


def test_instrument_named_in_claim():
    rep, tr = run("iv", "Enrolling raises test score, using the lottery as an instrument")
    assert not rep.needs_input, rep.needs_input
    assert rep.claim.instrument_col == "lottery"
    assert rep.design.kind == "iv"
    assert rep.data["roles"].instrument == "lottery"
    assert any(s.key == "design_endogeneity" for s in rep.sections)


def test_randomized_wording_is_confirmed_first():
    df, _ = sim.rct()
    rep = pipeline.run(df, "In this randomized trial, being treated increases score")
    assert any(q["role"] == "randomized" for q in rep.needs_input)
    assert rep.design is None
    rep = pipeline.run(df, "In this randomized trial, being treated increases score", confirm={"randomized": "yes"})
    assert rep.design.kind == "experiment"
    rep = pipeline.run(df, "In this randomized trial, being treated increases score", confirm={"randomized": "no"})
    assert rep.design.kind == "observational"


def test_guessed_weak_instrument_is_reclassified():
    from causality.schema import Design, Section
    d = Design(kind="iv", declared=False)
    ident = {"design_sections": [Section(key="design_first_stage", title="x", status=Status.FAIL)]}
    assert pipeline._reclassify(d, ident) == "observational"
    assert pipeline._reclassify(Design(kind="iv", declared=True), ident) is None
    ok = {"design_sections": [Section(key="design_first_stage", title="x", status=Status.PASS)]}
    assert pipeline._reclassify(d, ok) is None
    pre = {"design_sections": [Section(key="design_pretrends", title="x", status=Status.FAIL)]}
    assert pipeline._reclassify(Design(kind="did"), pre) is None   # a failed validity test stays a failed design


def test_panel_gets_functional_form_and_placebo_cards():
    rep, _ = run("did", "The policy increases sales")
    keys = {s.key for s in rep.sections}
    assert {"design_functional_form", "design_placebo_time", "design_pretrends"} <= keys
    ff = next(s for s in rep.sections if s.key == "design_functional_form")
    assert ff.affects_grade is False and ff.status != Status.FAIL


def test_event_study_tests():
    rep, tr = run("event_study", "The announcement raises ret")
    es = next(s for s in rep.sections if s.key == "design_event_study")
    assert es.error is None
    assert es.metrics["bmp_p"] < 0.01 and es.metrics["corrado_p"] < 0.01
    assert abs(es.metrics["mean_abnormal"] - tr["effect"]) < 0.01
    # a one-off event: the estimate is the event-date abnormal outcome, not the diluted DiD
    assert rep.estimate.target == "event_day_effect" and covers(rep.estimate, tr["effect"])
    assert rep.estimate.comparisons and abs(rep.estimate.comparisons[0]["estimate"]) < 0.01
    assert rep.verdict.label in ("Supported", "Partly supported")
    ov = next(s for s in rep.sections if s.key == "overlap")
    assert ov.status is None and ov.affects_grade is False


def test_event_study_null_is_not_supported():
    rep, _ = run("event_study", "The announcement raises ret", effect=0.0)
    assert rep.verdict.label in ("No evidence", "Inconclusive")


def test_no_literature_search_on_generic_headers(monkeypatch):
    from causality.checks import literature
    called = []
    monkeypatch.setattr(literature, "_online", lambda q: called.append(q) or [])
    df, _ = sim.observational()
    df = df.rename(columns={"program": "X1", "earnings": "X2", "x1": "X3", "x2": "X4", "member": "X5"})
    rep = pipeline.run(df, "X1 increases X2", online_literature=True)
    lit = next(s for s in rep.sections if s.key == "literature")
    assert not called and lit.metrics.get("online_skipped")
    df, _ = sim.observational()
    pipeline.run(df, "The program increases earnings", online_literature=True)
    assert called


def test_overlap_rule_depends_on_sample_size_and_covariates():
    from causality.checks.overlap import trim_alpha
    assert trim_alpha(300, 10) > trim_alpha(3000, 3) > trim_alpha(100000, 3)
    assert 0.01 <= trim_alpha(10**7, 1) and trim_alpha(50, 30) <= 0.10


def test_dag_has_groups():
    rep, _ = run("observational", "The program increases earnings")
    assert rep.dag.data.get("groups")
    assert all("group" in n for n in rep.dag.data["nodes"])
