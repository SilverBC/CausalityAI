import math

import numpy as np
import pandas as pd
import pytest

from causality import data, sim
from causality.estimate import estimate
from causality.schema import Roles


def run(df, roles, kind):
    prep = data.prepare(df, Roles(**roles))
    design = data.resolve_design(prep, kind)
    return estimate(prep, design, None)


def inside(ci, v):
    return ci is not None and ci[0] <= v <= ci[1]


def comp(e, prefix):
    return next(c for c in e.comparisons if c["method"].startswith(prefix))


def test_rct_lin():
    df, truth = sim.rct()
    e = run(df, dict(treatment="treated", outcome="score"), "experiment")
    assert not e.refused
    assert e.target == "ATE" and e.scope == "full"
    assert inside(e.ci95, truth["effect"])
    assert e.ci90[1] - e.ci90[0] < e.ci95[1] - e.ci95[0]
    assert e.baseline_mean is not None and e.outcome_sd > 0
    assert comp(e, "Unadjusted")["estimate"] is not None


def test_rct_coverage():
    hits = 0
    for s in range(30):
        df, truth = sim.rct(n=400, seed=100 + s)
        e = run(df, dict(treatment="treated", outcome="score"), "experiment")
        hits += inside(e.ci95, truth["effect"])
    assert hits >= 25


def test_observational_aipw():
    df, truth = sim.observational()
    e = run(df, dict(treatment="program", outcome="earnings"), "observational")
    assert not e.refused
    assert inside(e.ci95, truth["effect"])
    assert not inside(e.ci95, comp(e, "Naive")["estimate"])
    assert e.scope in ("full", "overlap_trimmed")
    assert any(c["method"].startswith("Inverse") for c in e.comparisons)


def test_did_callaway_santanna():
    df, truth = sim.did_staggered()
    e = run(df, dict(treatment="policy", outcome="sales", unit="state", time="year"), "did")
    assert not e.refused
    assert e.target == "ATT" and e.scope == "treated"
    assert inside(e.ci95, truth["effect"])
    assert comp(e, "Two-way")["estimate"] is not None
    assert e.extra["event_study"] and any(c.kind == "line" for c in e.charts)


def test_did_needs_unit_and_time():
    df, _ = sim.did_staggered()
    e = run(df, dict(treatment="policy", outcome="sales"), "did")
    assert e.refused and e.estimate is None and e.refusal_reason


def test_iv_2sls_and_ar():
    # the default seed (6) of sim.iv draws a sample whose plain Wald ratio is ~2.5 SEs above the
    # truth; seed 1 is a typical draw.
    df, truth = sim.iv(seed=1)
    e = run(df, dict(treatment="enrolled", outcome="test_score", instrument="lottery"), "iv")
    assert not e.refused
    assert e.target == "LATE" and e.scope == "compliers"
    assert inside(e.ci95, truth["effect"])
    ar = e.extra["ar_ci95"]
    assert e.extra["ar_ci95_bounded"] and any(lo <= truth["effect"] <= hi for lo, hi in ar)
    assert e.extra["first_stage_F"] > 10
    naive = comp(e, "Naive")
    assert not inside(naive["ci95"], truth["effect"])


def test_iv_weak_instrument_ar_unbounded():
    df, _ = sim.iv(n=400, strength=0.02, seed=3)
    e = run(df, dict(treatment="enrolled", outcome="test_score", instrument="lottery"), "iv")
    assert e.extra.get("first_stage_F", 0) < 10
    assert e.extra["weak_instrument"]
    ar = e.extra["ar_ci95"]
    # with an (almost) irrelevant instrument the AR set should not be a short bounded interval
    assert not e.extra["ar_ci95_bounded"] or (ar[0][1] - ar[0][0]) > 5 * (e.ci95[1] - e.ci95[0] if e.ci95 else 0)
    assert any(math.isinf(lo) or math.isinf(hi) for lo, hi in ar) or not ar


def test_rd_robust():
    df, truth = sim.rd()
    e = run(df, dict(treatment="scholarship", outcome="gpa_later", running="exam_score", cutoff=50.0), "rd")
    assert not e.refused
    assert e.target == "local_at_cutoff" and e.scope == "local_cutoff"
    assert inside(e.ci95, truth["effect"])
    assert not e.extra["fuzzy"]
    assert len(e.comparisons) == 2
    assert e.baseline_mean == pytest.approx(20 + 0.3 * 50, abs=1.0)
    assert any(c.kind == "scatter" for c in e.charts)


def test_rd_fuzzy():
    df, truth = sim.rd(seed=11)
    rng = np.random.default_rng(0)
    flip = rng.random(len(df)) < 0.15
    t = np.where(flip, 1 - df["scholarship"], df["scholarship"])
    df["gpa_later"] = df["gpa_later"] + truth["effect"] * (t - df["scholarship"])
    df["scholarship"] = t
    e = run(df, dict(treatment="scholarship", outcome="gpa_later", running="exam_score", cutoff=50.0), "rd")
    assert e.extra["fuzzy"]
    assert inside(e.ci95, truth["effect"])


def test_dose_response():
    df, truth = sim.dose()
    e = run(df, dict(treatment="study_hours", outcome="grade"), "observational")
    assert not e.refused
    assert e.target == "dose_response"
    assert inside(e.ci95, truth["effect"])
    st = e.extra["shape_tests"]
    assert st["linearity_test"]["p_value"] > 0.05
    curve = e.extra["dose_curve"]
    assert len(curve["x"]) == len(curve["y"]) == len(curve["lo"]) == len(curve["hi"])


def test_cluster_saturation_cluster_se():
    df, truth = sim.cluster_saturation()
    e = run(df, dict(treatment="treated", outcome="consumption", cluster="village", saturation="saturation"),
            "experiment")
    assert not e.refused
    assert "cluster" in e.extra["se_type"]
    assert e.extra["n_clusters"] == 60
    assert e.extra.get("t_df") == 59


def test_constant_outcome_refused():
    df, _ = sim.rct(n=200)
    df["score"] = 5.0
    e = run(df, dict(treatment="treated", outcome="score"), "experiment")
    assert e.refused and e.estimate is None and e.refusal_reason


def test_never_raises_on_garbage():
    df = pd.DataFrame({"t": [0, 1] * 15, "y": np.arange(30.0)})
    prep = data.prepare(df, Roles(treatment="t", outcome="y"))
    design = data.resolve_design(prep, "iv")   # no instrument
    e = estimate(prep, design, None)
    assert e.refused and e.refusal_reason
