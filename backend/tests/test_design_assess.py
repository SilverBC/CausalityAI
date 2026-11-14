import numpy as np
import pandas as pd

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


# ------------------------------------------------------------------ functional form (RESET)
DID_ROLES = dict(treatment="policy", outcome="sales", unit="state", time="year")


def test_reset_passes_on_linear_panel_and_is_not_graded():
    _, design, (assumptions, sections) = _did(0.0)
    s = _sec(sections, "design_functional_form")
    assert s.error is None and s.status == Status.PASS
    assert s.affects_grade is False and s.assumptions == []
    assert s.metrics["reset_degree2"]["p"] >= 0.05 and s.metrics["reset_degree2_3"]["p"] >= 0.05
    assert "Callaway" in s.summary and "does not show that the design is valid" in s.summary
    assert not any("functional" in a.key for a in assumptions)
    to_jsonable(s)


def test_reset_concern_on_nonlinear_outcome_never_fail():
    df, _ = sim.did_staggered()
    df["sales"] = np.exp(df["sales"] / 4)          # multiplicative unit and time effects
    _, design, (assumptions, sections) = _run(df, Roles(**DID_ROLES), "did")
    s = _sec(sections, "design_functional_form")
    assert s.status == Status.CONCERN and s.metrics["p"] < 0.05
    assert s.affects_grade is False
    # the grade ignores the card: same grade with and without it
    design.assumptions = assumptions
    assert grade(design, sections) == grade(design, [x for x in sections if x.key != "design_functional_form"])


def test_reset_runs_with_time_role_under_another_design():
    df, _ = sim.did_staggered()
    _, _, (_, sections) = _run(df, Roles(treatment="policy", outcome="sales", time="year"), "observational")
    s = _sec(sections, "design_functional_form")
    assert s.error is None and s.status in (Status.PASS, Status.CONCERN)
    assert s.metrics["fixed_effects"] == "time" and s.affects_grade is False


# ------------------------------------------------------------------ Durbin-Wu-Hausman
def test_dwh_detects_confounded_treatment():
    _, design, (assumptions, sections) = _iv(0.5)
    s = _sec(sections, "design_endogeneity")
    assert s.error is None and s.status is None and s.affects_grade is False
    assert s.metrics["p"] < 0.01
    assert s.assumptions == [] and not any("endogen" in a.key for a in assumptions)
    assert "ASSUMES the instrument is valid" in s.summary and "exclusion restriction" in s.summary
    assert not s.warnings
    to_jsonable(s)


def test_dwh_large_p_when_treatment_is_exogenous():
    rng = np.random.default_rng(1)
    n = 4000
    z = rng.binomial(1, 0.5, n)
    x = rng.normal(0, 1, n)
    d = (0.5 * z + 0.2 * x + rng.normal(0, 0.6, n) > 0.25).astype(int)
    y = 2 + 1.0 * d + 0.5 * x + rng.normal(0, 1.8, n)
    df = pd.DataFrame({"lottery": z, "enrolled": d, "x": x, "test_score": y})
    _, _, (_, sections) = _run(df, Roles(treatment="enrolled", outcome="test_score", instrument="lottery"), "iv")
    s = _sec(sections, "design_endogeneity")
    assert s.status is None and s.metrics["p"] > 0.2
    assert "not evidence" in s.summary


def test_dwh_warns_when_first_stage_is_weak():
    _, _, (_, sections) = _iv(0.03)
    s = _sec(sections, "design_endogeneity")
    assert s.status is None and s.metrics["first_stage_F"] < 10
    assert any("uninformative" in w for w in s.warnings)


# ------------------------------------------------------------------ placebo in time
def test_placebo_time_passes_without_pretrend():
    _, design, (assumptions, sections) = _did(0.0)
    s = _sec(sections, "design_placebo_time")
    assert s.error is None and s.status == Status.PASS and s.affects_grade is True
    assert _assump(assumptions, "did_placebo_time").status == Status.PASS
    assert s.metrics["shift_periods"] == 1 and abs(s.metrics["placebo_effect"]) < 0.5
    assert "placebo GROUP" in s.summary
    design.assumptions = assumptions
    assert grade(design, sections)[0] == "B"
    to_jsonable(s)


def test_placebo_time_flags_pretrend():
    _, _, (_, sections) = _did(0.5)
    s = _sec(sections, "design_placebo_time")
    assert s.status in (Status.FAIL, Status.CONCERN) and s.metrics["p"] < 0.05


def test_placebo_time_not_run_with_one_pre_period():
    df, _ = sim.did_staggered()
    df = df[df["year"] >= 2012]                    # earliest cohort keeps a single pre-period
    _, _, (assumptions, sections) = _run(df, Roles(**DID_ROLES), "did")
    s = _sec(sections, "design_placebo_time")
    assert s.status is None and s.error is None and s.assumptions == []
    assert "could not be run" in s.summary
    assert not any(a.key == "did_placebo_time" for a in assumptions)


# ------------------------------------------------------------------ event study (pulse treatment)
EV_ROLES = dict(treatment="announcement", outcome="ret", unit="firm", time="day")


def _event(effect, seed=12, drop_market=False):
    df, _ = sim.event_study(effect=effect, seed=seed)
    if drop_market:
        df = df.drop(columns="market_ret")
    return _run(df, Roles(**EV_ROLES), "did")


def test_event_study_detects_effect():
    _, design, (assumptions, sections) = _event(0.02)
    s = _sec(sections, "design_event_study")
    m = s.metrics
    assert s.error is None and s.status is None and s.affects_grade is False
    assert m["n_events"] == 120 and m["model"] == "market model"
    assert m["bmp_p"] < 0.01 and m["corrado_p"] < 0.01
    assert abs(m["mean_abnormal"] - 0.02) < 0.006
    assert m["car_m1_p1"] is not None
    ch = s.charts[0]
    assert ch.kind == "line" and ch.data["vline"] == 0.0
    a = _assump(assumptions, "event_no_confounding_events")
    assert a.status == Status.ASSUMED and a.testable is False
    assert "do not show that the event caused" in s.summary
    to_jsonable(sections)
    # the pre-trends card survives the pulse: no error, and no FAIL caused by treatment switching off
    pre = _sec(sections, "design_pretrends")
    assert pre.error is None and pre.status != Status.FAIL and pre.metrics["pulse_treatment"] is True
    assert not any("assumes treatment stays on" in w for w in pre.warnings)
    # graded cards only: dropping the informational cards leaves the grade unchanged
    design.assumptions = assumptions
    graded = [x for x in sections if x.affects_grade]
    assert len(graded) < len(sections)
    assert grade(design, sections) == grade(design, graded)


def test_event_study_null_effect_is_not_significant():
    for seed in (12, 13, 14):
        _, _, (_, sections) = _event(0.0, seed=seed)
        m = _sec(sections, "design_event_study").metrics
        assert m["bmp_p"] > 0.05 and m["corrado_p"] > 0.05, seed
        assert abs(m["mean_abnormal"]) < 0.006


def test_event_study_mean_adjusted_without_market_column():
    _, _, (_, sections) = _event(0.02, drop_market=True)
    m = _sec(sections, "design_event_study").metrics
    assert m["model"] == "mean-adjusted" and m["bmp_p"] < 0.01 and m["corrado_p"] < 0.01


def test_event_study_warns_on_clustered_event_dates():
    df, _ = sim.event_study(effect=0.0)
    df["announcement"] = (df["day"] == 70).astype(int)     # every firm has its event on the same day
    df.loc[df["firm"].isin(df["firm"].unique()[:20]), "announcement"] = 0   # some firms never have one
    _, _, (_, sections) = _run(df, Roles(**EV_ROLES), "did")
    s = _sec(sections, "design_event_study")
    assert s.metrics["max_share_same_date"] == 1.0
    assert any("Kolari" in w for w in s.warnings)


def test_event_study_not_detected_on_lasting_treatment():
    _, _, (assumptions, sections) = _did(0.0)
    assert not any(s.key == "design_event_study" for s in sections)
    assert not any(a.key == "event_no_confounding_events" for a in assumptions)
    assert _sec(sections, "design_pretrends").metrics["pulse_treatment"] is False


# ------------------------------------------------------------------ dag groups
def test_dag_groups_and_possible_collider():
    df, _ = sim.rct_with_post_treatment()
    # engagement is associated with both the treatment and the outcome in the data
    assert abs(df["engagement"].corr(df["treated"])) > 0.5 and abs(df["engagement"].corr(df["score"])) > 0.05
    prep = prepare(df, Roles(treatment="treated", outcome="score", post_treatment=["engagement"]))
    ch = dag(prep, resolve_design(prep, "experiment"), None)
    nodes = {n["id"]: n for n in ch.data["nodes"]}
    assert all("group" in n for n in ch.data["nodes"])
    assert nodes["engagement"]["role"] == "possible collider (excluded)"
    assert nodes["engagement"]["group"] == "possible colliders (not adjusted for)"
    assert nodes["age"]["group"] == "confounders (assumed)"
    e = {(x["from"], x["to"]): x["label"] for x in ch.data["edges"]}
    assert e[("treated", "engagement")] == "hypothesis" and e[("score", "engagement")] == "hypothesis"
    groups = {g["name"]: g for g in ch.data["groups"]}
    assert groups["possible colliders (not adjusted for)"]["members"] == ["engagement"]
    assert all(g["note"] for g in ch.data["groups"])
    assert sorted(m for g in ch.data["groups"] for m in g["members"]) == sorted(nodes)
    assert "non-causal path" in ch.note and "cannot be detected from data" in ch.note
    to_jsonable(ch)


def test_dag_post_treatment_column_unrelated_to_outcome_is_not_a_collider():
    df, _ = sim.rct()
    rng = np.random.default_rng(0)
    df["clicks"] = 2.0 * df["treated"] + rng.normal(0, 0.5, len(df))
    df["score"] = rng.normal(0, 1, len(df))        # outcome unrelated to everything
    prep = prepare(df, Roles(treatment="treated", outcome="score", post_treatment=["clicks"]))
    ch = dag(prep, resolve_design(prep, "experiment"), None)
    n = next(x for x in ch.data["nodes"] if x["id"] == "clicks")
    assert n["role"] == "post-treatment (excluded)" and n["group"] == "measured after treatment"
    assert ("score", "clicks") not in {(x["from"], x["to"]) for x in ch.data["edges"]}


def test_grade_ignores_ungraded_design_cards():
    base = [_ok("balance"), _ok("overlap"), _ok("missing_outcomes")]
    extra = [Section(key="design_functional_form", title="RESET", status=Status.CONCERN, affects_grade=False),
             Section(key="design_endogeneity", title="DWH", status=None, affects_grade=False),
             Section(key="design_event_study", title="Event", status=None, affects_grade=False, error="x")]
    d = Design(kind="experiment")
    assert grade(d, base + extra) == grade(d, base) and grade(d, base)[0] == "A"
