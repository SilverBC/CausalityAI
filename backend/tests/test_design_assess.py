from causality import sim
from causality.data import prepare, resolve_design
from causality.design_assess import assess, dag, grade
from causality.schema import Assumption, Design, Roles, Section, Status, to_jsonable


def _run(df, roles, kind):
    prep = prepare(df, roles)
    design = resolve_design(prep, kind)
    return prep, design, assess(prep, design, None)


def _sec(sections, key):
    return next(s for s in sections if s.key == key)


def _assump(assumptions, key):
    return next(a for a in assumptions if a.key == key)


# ------------------------------------------------------------------ DiD
def _did(pretrend):
    df, _ = sim.did_staggered(pretrend=pretrend)
    return _run(df, Roles(treatment="policy", outcome="sales", unit="state", time="year"), "did")


def test_did_no_pretrend_passes():
    prep, design, (assumptions, sections) = _did(0.0)
    s = _sec(sections, "design_pretrends")
    assert s.error is None
    assert _assump(s.assumptions, "did_parallel_pre").status == Status.PASS
    assert s.metrics["staggered_adoption"] is True
    post = _assump(assumptions, "did_parallel_post")
    assert post.status == Status.ASSUMED and post.testable is False
    assert s.charts[0].kind == "line" and s.charts[0].data["vline"] == 0.0
    to_jsonable(s)


def test_did_pretrend_fails():
    _, _, (_, sections) = _did(0.5)
    s = _sec(sections, "design_pretrends")
    assert _assump(s.assumptions, "did_parallel_pre").status == Status.FAIL
    assert s.status == Status.FAIL


# ------------------------------------------------------------------ IV
def _iv(strength):
    df, _ = sim.iv(strength=strength)
    return _run(df, Roles(treatment="enrolled", outcome="test_score", instrument="lottery"), "iv")


def test_iv_strong_first_stage():
    _, _, (assumptions, sections) = _iv(0.5)
    fs = _sec(sections, "design_first_stage")
    assert fs.status == Status.PASS and fs.metrics["robust_F"] > 20
    assert _assump(assumptions, "iv_exclusion").status == Status.ASSUMED
    assert _assump(assumptions, "iv_independence").status == Status.ASSUMED
    assert _sec(sections, "design_instrument_balance").status == Status.PASS


def test_iv_weak_first_stage():
    _, _, (_, sections) = _iv(0.03)
    fs = _sec(sections, "design_first_stage")
    assert fs.status == Status.FAIL and fs.metrics["robust_F"] < 10


# ------------------------------------------------------------------ RD
def _rd(manipulate):
    df, _ = sim.rd(manipulate=manipulate)
    return _run(df, Roles(treatment="scholarship", outcome="gpa_later", running="exam_score", cutoff=50.0), "rd")


def test_rd_no_bunching():
    _, _, (assumptions, sections) = _rd(False)
    b = _sec(sections, "design_bunching")
    assert _assump(b.assumptions, "rd_no_sorting").status == Status.PASS
    assert b.metrics["rd_type"] == "sharp"
    assert _assump(assumptions, "rd_continuity").status == Status.ASSUMED
    assert _sec(sections, "design_covariate_jumps").status == Status.PASS
    to_jsonable(sections)


def test_rd_bunching_fails():
    _, _, (_, sections) = _rd(True)
    assert _assump(_sec(sections, "design_bunching").assumptions, "rd_no_sorting").status == Status.FAIL


# ------------------------------------------------------------------ never raises
def test_missing_roles_give_errors_not_exceptions():
    df, _ = sim.observational()
    prep = prepare(df, Roles(treatment="program", outcome="earnings"))
    for kind in ("did", "iv", "rd"):
        a, s = assess(prep, Design(kind=kind), None)
        assert all(sec.error for sec in s)


# ------------------------------------------------------------------ grade
def _ok(key):
    return Section(key=key, title=key, status=Status.PASS,
                   assumptions=[Assumption(key + "_a", key, Status.PASS, source=key)])


def test_grade_experiment_all_pass_is_A():
    sections = [_ok("balance"), _ok("overlap"), _ok("missing_outcomes"), _ok("interference"),
                Section(key="hidden_confounding", title="h", status=Status.CONCERN),
                Section(key="literature", title="l", status=Status.FAIL, affects_grade=False)]
    d = Design(kind="experiment", assumptions=[Assumption("exp_randomized", "r", Status.ASSUMED, testable=False)])
    g, reasons = grade(d, sections)
    assert g == "A", reasons


def test_grade_observational_capped_at_C():
    g, reasons = grade(Design(kind="observational"), [_ok("balance"), _ok("overlap"), _ok("missing_outcomes")])
    assert g == "C"
    assert any("capped" in r for r in reasons)


def test_grade_fail_is_D():
    bad = Section(key="overlap", title="Overlap", status=Status.FAIL,
                  assumptions=[Assumption("overlap", "Overlap", Status.FAIL, "no overlap", source="overlap")])
    g, reasons = grade(Design(kind="experiment"), [_ok("balance"), bad])
    assert g == "D" and "Overlap" in reasons[0]


def test_grade_declaration_conflict_is_F():
    d = Design(kind="did", conflicts=[Assumption("decl_time", "At least two time periods", Status.FAIL,
                                                 "one period", source="declarations")])
    g, reasons = grade(d, [_ok("overlap")])
    assert g == "F" and "Declaration conflict" in reasons[0]


def test_grade_iv_all_pass_is_B_and_weak_is_D():
    _, design, (assumptions, sections) = _iv(0.5)
    design.assumptions = assumptions
    assert grade(design, sections)[0] == "B"
    _, design, (assumptions, sections) = _iv(0.03)
    design.assumptions = assumptions
    assert grade(design, sections)[0] == "D"


# ------------------------------------------------------------------ dag
def test_dag_iv_has_no_exclusion_edge():
    prep, design, _ = _iv(0.5)
    ch = dag(prep, design, None)
    assert ch.kind == "dag"
    e = {(x["from"], x["to"]) for x in ch.data["edges"]}
    assert ("lottery", "enrolled") in e and ("enrolled", "test_score") in e
    assert ("lottery", "test_score") not in e
    assert ("U (unmeasured)", "enrolled") in e and ("U (unmeasured)", "test_score") in e
    roles = {n["id"]: n["role"] for n in ch.data["nodes"]}
    assert roles["lottery"] == "instrument (assumed)"
    assert "hypothesis" in ch.note


def test_dag_observational_collapses_onehot_and_has_U():
    df, _ = sim.rct()
    prep = prepare(df, Roles(treatment="treated", outcome="score"))
    ch = dag(prep, Design(kind="observational"), None)
    ids = {n["id"] for n in ch.data["nodes"]}
    assert "region" in ids and not any(i.startswith("region=") for i in ids)
    assert any(e["type"] == "bidirected" for e in ch.data["edges"])
    assert "U (unmeasured)" in ids
