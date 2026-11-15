"""Part A: evidence tools. Known-truth data from causality.sim; no network, no model."""
import json
import math
import re

import numpy as np
import pandas as pd
import pytest

from causality import sim
from causality.orchestrator import contract, evidence

STATUS_WORDS = re.compile(r"\b(pass|passed|passes|fail|failed|fails|concern|concerns|ok)\b", re.I)


def _cases():
    return {
        "experiment": (sim.rct(n=1500, seed=0), {}),
        "did": (sim.did_staggered(units=120, seed=5), {"unit": "state", "time": "year"}),
        "iv": (sim.iv(n=3000, seed=1), {"instrument": "lottery"}),
        "rd": (sim.rd(n=3000, seed=7), {"running": "exam_score"}),
        "observational": (sim.observational(n=1200, seed=2), {}),
    }


CASES = _cases()


def _roles(design):
    (df, tr), extra = CASES[design]
    return df, tr, {"treatment": tr["treatment"], "outcome": tr["outcome"], **extra}


def _finite_json(obj):
    """json.dumps must work with allow_nan=False, and every number must be a plain int/float."""
    json.dumps(obj, allow_nan=False)

    def walk(x):
        if isinstance(x, dict):
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
        else:
            assert x is None or type(x) in (bool, int, float, str), type(x)
            if isinstance(x, float):
                assert math.isfinite(x)
    walk(obj)


def _check_item(item, tool, id_="e1"):
    assert set(item) == {"id", "tool", "args", "ok", "lines", "numbers", "error"}
    assert item["id"] == id_ and item["tool"] == tool
    _finite_json(item)
    for ln in item["lines"]:
        assert isinstance(ln, str) and ln.strip()
        assert not STATUS_WORDS.search(ln), ln


def test_tool_specs_match_contract():
    names = [t["name"] for t in evidence.TOOL_SPECS]
    assert names == list(contract.EVIDENCE_TOOLS)
    for t in evidence.TOOL_SPECS:
        assert t["description"] and t["input_schema"]["type"] == "object"
        json.dumps(t)
    spec = {t["name"]: t for t in evidence.TOOL_SPECS}
    assert spec["estimate_effect"]["input_schema"]["properties"]["design"]["enum"] == list(contract.DESIGNS)
    assert set(contract.ROLE_KEYS) <= set(spec["design_tests"]["input_schema"]["properties"]["roles"]["properties"])


@pytest.mark.parametrize("design", list(contract.DESIGNS))
def test_estimate_covers_truth(design):
    df, tr, roles = _roles(design)
    item = evidence.call("estimate_effect", {"design": design, "roles": roles}, df, "e1")
    _check_item(item, "estimate_effect")
    assert item["ok"], item["error"]
    n = item["numbers"]
    for k in ("design", "roles", "method", "target", "scope", "n", "estimate", "se", "p_value", "ci", "refused",
              "refusal_reason", "outcome_sd", "comparisons"):
        assert k in n, k
    assert n["design"] == design and n["roles"]["treatment"] == tr["treatment"]
    assert n["refused"] is False and n["n"] > 0
    assert set(n["ci"]) == {"90", "95", "98", "99"}
    lo, hi = n["ci"]["95"]
    assert lo < tr["effect"] < hi, (design, n["estimate"], n["ci"]["95"], tr["effect"])
    widths = [n["ci"][k][1] - n["ci"][k][0] for k in ("90", "95", "98", "99")]
    assert all(w > 0 for w in widths) and widths == sorted(widths)
    assert n["outcome_sd"] and n["outcome_sd"] > 0
    for c in n["comparisons"]:
        assert set(c) == {"method", "estimate", "ci95"}
    assert any("95%" in ln for ln in item["lines"])


def test_estimate_null_rct_interval_near_zero():
    df, tr = sim.null_rct(n=3000, seed=11)
    item = evidence.call("estimate_effect", {"design": "experiment",
                                             "roles": {"treatment": "treated", "outcome": "score"}}, df, "e2")
    assert item["ok"]
    lo, hi = item["numbers"]["ci"]["95"]
    assert lo < 0 < hi


@pytest.mark.parametrize("design", list(contract.DESIGNS))
def test_design_tests_and_sensitivity_run(design):
    df, tr, roles = _roles(design)
    item = evidence.call("design_tests", {"design": design, "roles": roles}, df, "e3")
    _check_item(item, "design_tests", "e3")
    assert item["ok"], item["error"]
    n = item["numbers"]
    assert n["design"] == design and isinstance(n["conflicts"], list)
    assert isinstance(n["untestable"], list) and n["untestable"]
    assert n["tests"], design
    for t in n["tests"]:
        assert {"key", "title", "statistic", "p_value", "metrics"} <= set(t)
    if design == "iv":
        fs = [t for t in n["tests"] if t["key"] == "design_first_stage"][0]
        assert fs["statistic"] > 10
        assert any("Exclusion" in u for u in n["untestable"])
    if design == "did":
        assert any(t["key"] == "design_pretrends" and t["p_value"] is not None for t in n["tests"])
    if design == "rd":
        assert n["cutoff_used"] == pytest.approx(tr["cutoff"], abs=1.0)

    s = evidence.call("sensitivity", {"design": design, "roles": roles}, df, "e4")
    _check_item(s, "sensitivity", "e4")
    assert s["ok"], s["error"]
    sn = s["numbers"]
    for k in ("robustness_value", "robustness_value_alpha", "e_value", "benchmarks", "untestable"):
        assert k in sn
    assert 0 < sn["robustness_value"] <= 1 and sn["untestable"]


def test_general_and_structure_facts():
    df, tr, roles = _roles("observational")
    g = evidence.call("general_checks", {"treatment": "program", "outcome": "earnings"}, df, "e5")
    _check_item(g, "general_checks", "e5")
    assert g["ok"], g["error"]
    keys = [c["key"] for c in g["numbers"]["checks"]]
    assert len(keys) == 3 and all({"key", "title", "metrics"} <= set(c) for c in g["numbers"]["checks"])
    assert isinstance(g["numbers"]["untestable"], list)
    assert sum(ln.startswith("Check #") for ln in g["lines"]) == 3

    g2 = evidence.call("general_checks", {"treatment": "program", "outcome": "earnings", "covariates": ["x1"]}, df, "e6")
    assert g2["ok"] and g2["numbers"]["adjustment_set"] == ["x1"]

    (dfi, _), _ = CASES["iv"]
    s = evidence.call("structure_facts", {"treatment": "enrolled", "outcome": "test_score",
                                          "instruments": ["lottery"]}, dfi, "e7")
    _check_item(s, "structure_facts", "e7")
    assert s["ok"] and any("first-stage F" in ln for ln in s["lines"])

    (dfr, _), _ = CASES["rd"]
    s = evidence.call("structure_facts", {"treatment": "scholarship", "outcome": "gpa_later"}, dfr, "e8")
    assert s["ok"] and any("exam_score" in ln for ln in s["lines"])


def test_table_facts_and_explore():
    df, _ = sim.rct(n=300, seed=3)
    t = evidence.call("table_facts", {}, df, "e9")
    _check_item(t, "table_facts", "e9")
    assert t["ok"] and t["numbers"]["n_rows"] == 300 and len(t["numbers"]["columns"]) == df.shape[1]
    assert t["lines"][0].startswith("Rows: 300")

    dfc, truth = sim.chain(n=500)
    e = evidence.call("explore", {}, dfc, "e10")
    _check_item(e, "explore", "e10")
    assert e["ok"], e["error"]
    assert e["numbers"]["untestable"] and isinstance(e["numbers"]["drawn_edges"], list)
    assert len(e["numbers"]["drawn_edges"]) >= 2


def test_covariates_restrict_estimate():
    df, tr, roles = _roles("observational")
    a = evidence.call("estimate_effect", {"design": "observational", "roles": roles, "covariates": []}, df, "e1")
    b = evidence.call("estimate_effect", {"design": "observational", "roles": roles, "covariates": None}, df, "e2")
    assert a["ok"] and b["ok"]
    assert a["numbers"]["adjustment_set"] == [] and len(b["numbers"]["adjustment_set"]) >= 3
    # without the confounders the estimate is biased; with them it covers the truth
    lo, hi = a["numbers"]["ci"]["95"]
    assert not (lo < tr["effect"] < hi)


def test_messy_values_stay_json():
    df, _ = sim.rct(n=400, seed=4, attrition=0.1)
    df.loc[:5, "age"] = np.nan
    df["flag"] = df["female"].astype(bool)
    df["when"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(np.arange(len(df)), unit="D")
    for tool, args in [("table_facts", {}), ("general_checks", {"treatment": "treated", "outcome": "score"}),
                       ("estimate_effect", {"design": "experiment", "roles": {"treatment": "treated", "outcome": "score"}}),
                       ("design_tests", {"design": "experiment", "roles": {"treatment": "treated", "outcome": "score"}})]:
        item = evidence.call(tool, args, df, "e1")
        _check_item(item, tool)
        assert item["ok"], (tool, item["error"])


BAD = [
    ("no_such_tool", {}),
    ("estimate_effect", {"design": "magic", "roles": {"treatment": "treated", "outcome": "score"}}),
    ("estimate_effect", {"design": "experiment", "roles": {"treatment": "nope", "outcome": "score"}}),
    ("estimate_effect", {"design": "experiment", "roles": {"outcome": "score"}}),
    ("estimate_effect", {"design": "experiment"}),
    ("estimate_effect", {"design": "experiment", "roles": "treated"}),
    ("estimate_effect", {"design": "experiment", "roles": {"treatment": "treated", "outcome": "score", "colour": "x"}}),
    ("estimate_effect", {"design": "rd", "roles": {"treatment": "treated", "outcome": "score", "running": "age",
                                                   "cutoff": "high"}}),
    ("design_tests", {"design": "experiment", "roles": {"treatment": "treated", "outcome": "score"},
                      "covariates": ["age", "ghost"]}),
    ("sensitivity", {"design": None, "roles": {"treatment": "treated", "outcome": "score"}}),
    ("general_checks", {"treatment": "treated"}),
    ("general_checks", {"treatment": "treated", "outcome": "treated"}),
    ("structure_facts", {"treatment": 3, "outcome": "score"}),
    ("structure_facts", {"treatment": "treated", "outcome": "score", "instruments": ["ghost"]}),
]


@pytest.mark.parametrize("name,args", BAD)
def test_bad_input_gives_plain_error(name, args):
    df, _ = sim.rct(n=200, seed=5)
    item = evidence.call(name, args, df, "e1")
    assert item["ok"] is False
    assert isinstance(item["error"], str) and item["error"] and "Traceback" not in item["error"]
    _finite_json(item)


def test_bad_args_and_table_never_raise():
    df, _ = sim.rct(n=200, seed=5)
    for args in (None, [], "x", {"design": object()}):
        item = evidence.call("estimate_effect", args, df, "e1")
        assert item["ok"] is False
        json.dumps(item)
    assert evidence.call("table_facts", None, df, "e1")["ok"]
    assert evidence.call("table_facts", {}, pd.DataFrame(), "e1")["ok"] is False
    assert evidence.call("table_facts", {}, None, "e1")["ok"] is False
    # a treatment the estimator cannot use: refused or a plain error, never an exception
    df2 = df.copy()
    df2["score"] = 1.0
    item = evidence.call("estimate_effect", {"design": "experiment",
                                             "roles": {"treatment": "treated", "outcome": "score"}}, df2, "e1")
    json.dumps(item, allow_nan=False)
    if item["ok"]:
        assert item["numbers"]["refused"] is True and item["numbers"]["ci"]["95"] is None
        assert item["numbers"]["refusal_reason"]


# ------------------------------------------------------------------ review fixes

@pytest.mark.parametrize("design,expect", [
    ("observational", "No hidden confounders"),
    ("iv", "Exclusion restriction: the instrument affects the outcome only through the treatment"),
    ("rd", "Continuity of potential outcomes at the cutoff"),
    ("did", "Parallel trends would have continued after treatment (untestable)"),
    ("experiment", "Treatment was randomized as described"),
])
def test_estimate_lists_the_designs_untestable_assumptions(design, expect):
    """An estimate alone must carry the design's identifying assumptions, so floor rule 1 lists them."""
    df, _, roles = _roles(design)
    item = evidence.call("estimate_effect", {"design": design, "roles": roles}, df, "e1")
    assert item["ok"], item["error"]
    assert expect in item["numbers"]["untestable"], item["numbers"]["untestable"]
    assert any(expect in ln for ln in item["lines"])


def test_estimate_alone_puts_assumptions_in_the_verdict():
    from causality.orchestrator import floor
    df, _, roles = _roles("observational")
    e1 = evidence.call("estimate_effect", {"design": "observational", "roles": roles}, df, "e1")
    draft = contract.empty_verdict()
    draft.update({"label": contract.SUPPORTED, "sentence": "s", "estimate_evidence_id": "e1",
                  "claimed_direction": "increase", "assumed": []})
    reading = {"design": "observational", "treatment": roles["treatment"], "outcome": roles["outcome"],
               "roles": dict(roles), "covariates": None}
    v, fl = floor.apply(draft, reading, [e1])
    assert "No hidden confounders" in v["assumed"]
    assert fl["changed"] and fl["violations"][0]["rule"] == "assumed_is_not_tested"


def _weak_iv(strength, seed):
    df, tr = sim.iv(strength=strength, seed=seed, n=4000)
    roles = {"treatment": "enrolled", "outcome": "test_score", "instrument": "lottery"}
    return df, evidence.call("estimate_effect", {"design": "iv", "roles": roles}, df, "e1")


def test_iv_interval_is_anderson_rubin_not_wald():
    """With a weak instrument the 2SLS Wald interval excludes zero while the Anderson-Rubin set does not:
    ci['95'] must be the weak-instrument-robust one, so the floor cannot pass Supported on the Wald one."""
    from causality.orchestrator import floor
    _, item = _weak_iv(0.05, 3)
    n = item["numbers"]
    assert item["ok"] and not n["refused"]
    wlo, whi = n["ci_wald"]["95"]
    assert wlo > 0                                  # the Wald interval alone would back "increase"
    lo, hi = n["ci"]["95"]
    assert lo < 0 < hi                              # the AR set includes zero
    reading = {"design": "iv", "treatment": "enrolled", "outcome": "test_score",
               "roles": {"treatment": "enrolled", "outcome": "test_score", "instrument": "lottery"},
               "covariates": None}
    d = contract.empty_verdict()
    d.update({"label": contract.SUPPORTED, "sentence": "s", "estimate_evidence_id": "e1",
              "claimed_direction": "increase", "assumed": list(n["untestable"])})
    v, fl = floor.apply(d, reading, [item])
    assert v["label"] == contract.INCONCLUSIVE
    for ln in item["lines"]:
        assert not re.search(r"instrument is weak|too weak|trust the", ln), ln


def test_iv_unbounded_anderson_rubin_gives_no_interval():
    _, item = _weak_iv(0.03, 6)
    n = item["numbers"]
    assert item["ok"] and not n["refused"]
    assert n["ci"]["95"] is None
    assert n["ci_wald"]["95"] is not None
    assert any("unbounded" in ln for ln in item["lines"])


def test_iv_strong_instrument_ar_close_to_wald():
    (df, tr), extra = CASES["iv"]
    item = evidence.call("estimate_effect", {"design": "iv", "roles": {"treatment": tr["treatment"],
                                                                     "outcome": tr["outcome"], **extra}}, df, "e1")
    lo, hi = item["numbers"]["ci"]["95"]
    wlo, whi = item["numbers"]["ci_wald"]["95"]
    assert abs((hi - lo) - (whi - wlo)) < 0.5 * (whi - wlo)


def test_general_checks_leaves_sensitivity_to_its_own_tool():
    """general_checks no longer bundles the hidden-confounding sensitivity numbers, so a 'supports'
    reading of it cannot be a reading of a sensitivity analysis (floor rule 1 covers the sensitivity tool)."""
    df, _, _ = _roles("observational")
    g = evidence.call("general_checks", {"treatment": "program", "outcome": "earnings"}, df, "e2")
    assert g["ok"], g["error"]
    keys = [c["key"] for c in g["numbers"]["checks"]]
    assert "sensitivity" not in keys and len(keys) == 3
    assert not any("robustness value" in ln.lower() and ln.startswith("Check #") for ln in g["lines"])
    assert any("sensitivity" in ln for ln in g["lines"])


VERDICT_WORDS = re.compile(r"below 10 is weak|instrument is weak|too weak|trust the|Strong first stage|"
                           r"looks rejected", re.I)


def test_lines_carry_no_threshold_verdicts():
    (dfi, _), _ = CASES["iv"]
    s = evidence.call("structure_facts", {"treatment": "enrolled", "outcome": "test_score",
                                          "instruments": ["lottery"]}, dfi, "e1")
    t = evidence.call("design_tests", {"design": "iv", "roles": {"treatment": "enrolled", "outcome": "test_score",
                                                                 "instrument": "lottery"}}, dfi, "e2")
    _, w = _weak_iv(0.05, 3)
    _, u = _weak_iv(0.03, 6)
    for item in (s, t, w, u):
        assert item["ok"], item["error"]
        for ln in item["lines"]:
            assert not VERDICT_WORDS.search(ln), ln
    assert all(not VERDICT_WORDS.search(x["title"]) for x in t["numbers"]["tests"])
    assert any("first-stage F" in ln for ln in s["lines"])
