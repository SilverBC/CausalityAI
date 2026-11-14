import math

import numpy as np
import pandas as pd
import pytest

from causality import data, sim
from causality.checks import balance, overlap, sensitivity
from causality.schema import Estimate, Roles, Status, to_jsonable


def make(gen, declared=None, randomized=False, **kw):
    df, truth = gen(**kw)
    roles = Roles(treatment=truth["treatment"], outcome=truth["outcome"], randomized=randomized,
                  instrument=truth.get("instrument"), running=truth.get("running"), cutoff=truth.get("cutoff"),
                  unit=truth.get("unit"), time=truth.get("time"))
    prep = data.prepare(df, roles)
    design = data.resolve_design(prep, declared)
    return prep, design, truth


# ------------------------------------------------------------------ balance
def test_balance_rct_pass():
    prep, design, _ = make(sim.rct, "experiment")
    s = balance.run(prep, design, None)
    assert s.error is None, s.error
    assert s.status in (Status.PASS, Status.CONCERN)
    assert s.metrics["suspected_post_treatment"] == []
    names = {a.key: a for a in s.assumptions}
    assert names["randomization_carried_out"].status == Status.ASSUMED
    assert names["randomization_carried_out"].testable is False
    assert "expected_beyond_chance" in s.metrics and "n_beyond_chance" in s.metrics
    to_jsonable(s)


def test_balance_flags_post_treatment():
    prep, design, _ = make(sim.rct_with_post_treatment, "experiment")
    s = balance.run(prep, design, None)
    assert s.error is None, s.error
    assert "engagement" in s.metrics["suspected_post_treatment"]
    for c in ("age", "female", "income"):
        assert c not in s.metrics["suspected_post_treatment"]
    assert s.warnings
    # joint test excludes the suspected column, so randomization still looks fine
    assert s.status in (Status.PASS, Status.CONCERN)


def test_balance_not_randomized_never_flags():
    df, truth = sim.rct_with_post_treatment()
    prep = data.prepare(df, Roles(treatment="treated", outcome="score"))
    design = data.resolve_design(prep, "observational")
    s = balance.run(prep, design, None)
    assert s.metrics["suspected_post_treatment"] == []
    assert s.status == Status.CONCERN


def test_balance_user_marked_post_shown():
    df, _ = sim.rct_with_post_treatment()
    prep = data.prepare(df, Roles(treatment="treated", outcome="score", randomized=True, post_treatment=["engagement"]))
    design = data.resolve_design(prep, "experiment")
    s = balance.run(prep, design, None)
    assert "engagement" in s.metrics["post_treatment_shown"]
    assert any("after treatment" in lab for lab in s.charts[0].data["labels"])
    assert s.metrics["suspected_post_treatment"] == []


def test_balance_continuous_and_iv():
    prep, design, _ = make(sim.dose, "observational")
    s = balance.run(prep, design, None)
    assert s.error is None and s.metrics["suspected_post_treatment"] == []
    prep, design, _ = make(sim.iv, "iv")
    s = balance.run(prep, design, None)
    assert s.error is None, s.error
    assert s.metrics["comparison"] == "instrument"
    prep, design, _ = make(sim.rd, "rd")
    s = balance.run(prep, design, None)
    assert s.error is None, s.error


def test_joint_test_detects_imbalance_and_strata_cluster():
    rng = np.random.default_rng(0)
    n = 600
    x = rng.normal(size=(n, 2))
    g = (x[:, 0] + rng.normal(size=n) > 0).astype(float)
    assert balance.joint_test(x, g, n_perm=200)["p"] < 0.01
    g0 = rng.binomial(1, 0.5, n).astype(float)
    strata = rng.integers(0, 4, n)
    assert balance.joint_test(x, g0, strata=strata, n_perm=200)["p"] > 0.001
    cl = np.repeat(np.arange(60), 10)
    gc = np.repeat(rng.binomial(1, 0.5, 60), 10).astype(float)
    res = balance.joint_test(x, gc, cluster=cl, n_perm=200)
    assert "whole clusters" in res["method"] and 0 < res["p"] <= 1


def test_balance_never_raises():
    prep, design, _ = make(sim.rct, "experiment")
    prep.roles.treatment = "nope"
    s = balance.run(prep, design, None)
    assert s.error


# ------------------------------------------------------------------ overlap
def test_overlap_rct_pass():
    prep, design, _ = make(sim.rct, "experiment")
    s = overlap.run(prep, design, None)
    assert s.error is None, s.error
    assert s.status == Status.PASS
    ps = overlap.propensity(prep)
    assert len(ps) == len(prep.df)
    m = overlap.trim_mask(prep)
    assert m.dtype == bool and (m.index == prep.df.index).all()


def test_overlap_no_overlap_fails():
    prep, design, _ = make(sim.no_overlap, "observational")
    s = overlap.run(prep, design, None)
    assert s.status in (Status.FAIL, Status.CONCERN)
    assert s.assumptions[0].status == s.status
    assert s.metrics["trimmed_share"] > 0.05
    assert "x" in s.summary


def test_overlap_observational():
    prep, design, _ = make(sim.observational, "observational")
    s = overlap.run(prep, design, None)
    assert s.status in (Status.PASS, Status.CONCERN)
    assert {c.kind for c in s.charts} == {"hist", "bar"}
    assert len(s.charts[0].data["bins"]) == 21


def test_overlap_continuous_and_no_covariates():
    prep, design, _ = make(sim.dose, "observational")
    s = overlap.run(prep, design, None)
    assert s.status is None and "treatment_r2" in s.metrics
    df = pd.DataFrame({"t": np.r_[np.zeros(20), np.ones(20)], "y": np.arange(40.0)})
    prep = data.prepare(df, Roles(treatment="t", outcome="y"))
    s = overlap.run(prep, data.resolve_design(prep, "observational"), None)
    assert s.status is None and s.error is None


# ------------------------------------------------------------------ sensitivity
def test_sensitivity_hidden():
    prep, design, _ = make(sim.observational_hidden, "observational")
    s = sensitivity.run(prep, design, None, None)
    assert s.error is None, s.error
    assert s.status != Status.PASS
    assert math.isfinite(s.metrics["rv_q"]) and math.isfinite(s.metrics["evalue_estimate"])
    a = s.assumptions[0]
    assert a.status == Status.ASSUMED and a.testable is False and "remaining variance" in a.detail
    to_jsonable(s)


def test_sensitivity_observational_robust():
    prep, design, _ = make(sim.observational, "observational")
    est = Estimate(method="test", target="ATE", estimate=2.0, se=0.06, ci95=(1.88, 2.12), outcome_sd=3.0)
    s = sensitivity.run(prep, design, None, est)
    assert s.metrics["rv_q"] > 0.2
    assert s.status == Status.ASSUMED
    assert any(c.kind == "bar" and "1x x1" in c.data["labels"] for c in s.charts)


def test_sensitivity_experiment_and_refused():
    prep, design, _ = make(sim.rct, "experiment", binary_outcome=True)
    s = sensitivity.run(prep, design, None, Estimate(method="x", target="ATE", refused=True))
    assert s.status == Status.ASSUMED
    assert "Randomization" in s.assumptions[0].name
    assert "refused" in s.summary


def test_sensitivity_formulas_hand():
    # RV_q: t = 5, dof = 100 -> f = 0.5, RV = 0.5*(sqrt(0.0625 + 1) - 0.25)
    assert abs(sensitivity.robustness_value(5, 100) - 0.5 * (math.sqrt(0.0625 + 1) - 0.25)) < 1e-12
    # RV_{q,alpha}: subtract critical f
    fc = abs(__import__("scipy").stats.t.ppf(0.025, 99)) / math.sqrt(99)
    fa = 0.5 - fc
    assert abs(sensitivity.robustness_value(5, 100, alpha=0.05) - 0.5 * (math.sqrt(fa ** 4 + 4 * fa ** 2) - fa ** 2)) < 1e-12
    # E-value of RR = 2 is 2 + sqrt(2); RR 0.5 the same
    assert abs(sensitivity.evalue_rr(2.0) - (2 + math.sqrt(2))) < 1e-12
    assert abs(sensitivity.evalue_rr(0.5) - (2 + math.sqrt(2))) < 1e-12
    # benchmark: k=1, r2dxj=0.1, r2yxj=0.2 (sensemakr formulas)
    r2yz, r2dz = sensitivity.benchmark_partial_r2(0.1, 0.2, 1)
    assert abs(r2dz - 0.1 / 0.9) < 1e-12
    r2zx = 0.01 / (0.9 * 0.9)
    assert abs(r2yz - ((1 + math.sqrt(r2zx)) / math.sqrt(1 - r2zx)) ** 2 * 0.25) < 1e-12
    # the RV on a regression matches the regression's t-stat
    prep, design, _ = make(sim.observational, "observational")
    s = sensitivity.run(prep, design, None, None)
    t, dof = s.metrics["t_stat"], s.metrics["dof"]
    f = abs(t) / math.sqrt(dof)
    assert abs(s.metrics["rv_q"] - 0.5 * (math.sqrt(f ** 4 + 4 * f ** 2) - f ** 2)) < 1e-9
    # bias at RV equals |estimate|
    rv = s.metrics["rv_q"]
    bias = sensitivity.bias_bound(s.metrics["regression_se"], dof, rv, rv)
    assert abs(bias - abs(s.metrics["regression_estimate"])) < 1e-6 * abs(s.metrics["regression_estimate"]) + 1e-9


def test_sensitivity_iv_and_never_raises():
    prep, design, _ = make(sim.iv, "iv")
    s = sensitivity.run(prep, design, None, None)
    assert s.status == Status.ASSUMED and s.error is None
    prep.roles.outcome = "missing_col"
    s = sensitivity.run(prep, design, None, None)
    assert s.error and s.status != Status.PASS
