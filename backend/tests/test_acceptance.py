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


def test_rct_size_rejected():
    rep, tr = run("rct", "Being treated increases score by 60%", design="experiment")
    size = next(s for s in rep.verdict.sub if s.aspect == "size")
    assert size.label.startswith("rejected")
    assert rep.verdict.label == "Contradicted"


def test_rct_differential_dropout_inconclusive():
    rep, _ = run("rct", "Being treated increases score", design="experiment", attrition=0.25, differential=0.25)
    assert rep.verdict.label == "Inconclusive"


def test_null_rct_equivalence_and_no_evidence():
    rep, _ = run("null_rct", "Being treated has no effect on score", design="experiment")
    assert rep.verdict.label.startswith("Supported")
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
