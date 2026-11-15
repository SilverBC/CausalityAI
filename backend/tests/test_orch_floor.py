import copy
import math
import random

import pytest

from causality.orchestrator import contract, floor
from causality.orchestrator.contract import (INCONCLUSIVE, NO_EFFECT, NOT_SUPPORTED, SUPPORTED,
                                             evidence_item)

READING = {"treatment": "t", "outcome": "y", "design": "observational", "design_reason": "",
           "roles": {"treatment": "t", "outcome": "y"}, "covariates": None, "excluded": []}

STRENGTH = {INCONCLUSIVE: 0, SUPPORTED: 1, NOT_SUPPORTED: 1, NO_EFFECT: 1}


def est(id="e1", lo=0.5, hi=1.5, design="observational", treatment="t", outcome="y", ok=True,
        refused=False, outcome_sd=10.0, ci=None):
    nums = {"design": design, "roles": {"treatment": treatment, "outcome": outcome},
            "estimate": None if lo is None or hi is None else (lo + hi) / 2 if not isinstance(lo, str) else None,
            "se": 0.1, "p_value": 0.01, "ci": ci if ci is not None else {"95": [lo, hi]},
            "refused": refused, "refusal_reason": "too few rows" if refused else None,
            "outcome_sd": outcome_sd}
    return evidence_item(id, "estimate_effect",
                         {"design": design, "roles": {"treatment": treatment, "outcome": outcome}},
                         ok=ok, lines=["An estimate."], numbers=nums,
                         error=None if ok else "The column was not found.")


def draft(label=SUPPORTED, eid="e1", direction="increase", **kw):
    d = contract.empty_verdict()
    d.update({"label": label, "sentence": "Model sentence.", "estimate_evidence_id": eid,
              "claimed_direction": direction})
    d.update(kw)
    return d


def rules(fl):
    return [v["rule"] for v in fl["violations"]]


# ------------------------------------------------------------------ rule 2 and 3 hold

def test_supported_increase_holds():
    v, fl = floor.apply(draft(), READING, [est(lo=0.5, hi=1.5)])
    assert v["label"] == SUPPORTED and v["sentence"] == "Model sentence."
    assert fl == {"changed": False, "violations": []}


def test_supported_decrease_holds():
    v, fl = floor.apply(draft(direction="decrease"), READING, [est(lo=-2, hi=-0.1)])
    assert v["label"] == SUPPORTED and not fl["changed"]


def test_not_supported_holds_both_directions():
    v, _ = floor.apply(draft(NOT_SUPPORTED, direction="increase"), READING, [est(lo=-2, hi=-0.1)])
    assert v["label"] == NOT_SUPPORTED
    v, _ = floor.apply(draft(NOT_SUPPORTED, direction="decrease"), READING, [est(lo=0.1, hi=2)])
    assert v["label"] == NOT_SUPPORTED


def test_inconclusive_needs_nothing():
    v, fl = floor.apply(draft(INCONCLUSIVE, eid=None, direction=None), READING, [])
    assert v["label"] == INCONCLUSIVE and not fl["changed"]


# ------------------------------------------------------------------ rule 3 violated

@pytest.mark.parametrize("label,direction,lo,hi", [
    (SUPPORTED, "increase", -0.5, 1.5),
    (SUPPORTED, "increase", 0.0, 1.5),       # touches zero
    (SUPPORTED, "decrease", -1.5, 0.0),      # touches zero
    (SUPPORTED, "decrease", 0.2, 1.0),       # wrong side
    (NOT_SUPPORTED, "increase", -1.0, 0.0),  # touches zero
    (NOT_SUPPORTED, "increase", 0.5, 1.0),   # this backs the claim
    (NOT_SUPPORTED, "decrease", 0.0, 1.0),   # touches zero
])
def test_label_beyond_interval_becomes_inconclusive(label, direction, lo, hi):
    v, fl = floor.apply(draft(label, direction=direction), READING, [est(lo=lo, hi=hi)])
    assert v["label"] == INCONCLUSIVE
    assert rules(fl) == ["no_more_than_the_interval"]
    viol = fl["violations"][0]
    assert f"from {floor._fmt(lo)} to {floor._fmt(hi)}" in viol["detail"]
    assert "Inconclusive" in viol["action"]
    assert v["sentence"] != "Model sentence." and "95% interval" in v["sentence"]


def test_supported_without_direction_is_inconclusive():
    v, fl = floor.apply(draft(direction=None), READING, [est()])
    assert v["label"] == INCONCLUSIVE and rules(fl) == ["no_more_than_the_interval"]
    v, fl = floor.apply(draft(direction="up"), READING, [est()])
    assert v["label"] == INCONCLUSIVE


def test_no_effect_with_outcome_sd():
    # band = 0.1 * 10 = 1
    v, fl = floor.apply(draft(NO_EFFECT, direction="none"), READING, [est(lo=-0.4, hi=0.6, outcome_sd=10)])
    assert v["label"] == NO_EFFECT and not fl["changed"]
    v, fl = floor.apply(draft(NO_EFFECT, direction="none"), READING, [est(lo=-0.4, hi=1.6, outcome_sd=10)])
    assert v["label"] == INCONCLUSIVE and rules(fl) == ["no_more_than_the_interval"]
    assert "1.6" in fl["violations"][0]["detail"] and "10" in fl["violations"][0]["detail"]


def test_no_effect_without_outcome_sd():
    for sd in (None, 0, -1, float("nan"), "abc"):
        v, fl = floor.apply(draft(NO_EFFECT, direction="none"), READING, [est(lo=-0.01, hi=0.01, outcome_sd=sd)])
        assert v["label"] == INCONCLUSIVE
        assert rules(fl) == ["no_more_than_the_interval"]
        assert "standard deviation" in fl["violations"][0]["detail"]


def test_claim_of_no_effect_supported_and_not_supported():
    v, _ = floor.apply(draft(SUPPORTED, direction="none"), READING, [est(lo=-0.2, hi=0.2, outcome_sd=10)])
    assert v["label"] == SUPPORTED
    v, _ = floor.apply(draft(SUPPORTED, direction="none"), READING, [est(lo=-0.2, hi=3, outcome_sd=10)])
    assert v["label"] == INCONCLUSIVE
    v, _ = floor.apply(draft(NOT_SUPPORTED, direction="none"), READING, [est(lo=2, hi=3)])
    assert v["label"] == NOT_SUPPORTED
    v, _ = floor.apply(draft(NOT_SUPPORTED, direction="none"), READING, [est(lo=-2, hi=3)])
    assert v["label"] == INCONCLUSIVE


# ------------------------------------------------------------------ rule 2 violated

def _expect_interval_violation(d, evidence, reading=READING):
    v, fl = floor.apply(d, reading, evidence)
    assert v["label"] == INCONCLUSIVE
    assert rules(fl) == ["needs_a_real_interval"], fl
    assert v["sentence"] != "Model sentence."
    assert v["sentence"]
    return fl["violations"][0]


def test_refused_estimate():
    viol = _expect_interval_violation(draft(), [est(refused=True)])
    assert "too few rows" in viol["detail"]


def test_failed_estimate():
    viol = _expect_interval_violation(draft(), [est(ok=False)])
    assert "did not run" in viol["detail"]


def test_zero_width_interval():
    viol = _expect_interval_violation(draft(), [est(lo=1.0, hi=1.0)])
    assert "no width" in viol["detail"]
    _expect_interval_violation(draft(), [est(lo=2.0, hi=1.0)])  # reversed bounds


@pytest.mark.parametrize("ci", [
    {"95": [float("nan"), 1.0]}, {"95": [0.5, float("inf")]}, {"95": [None, 1.0]}, {"95": ["a", "b"]},
    {"95": [0.5]}, {"95": None}, {}, {"90": [0.1, 0.2]}, {"95": [True, 2]},
])
def test_bad_bounds(ci):
    _expect_interval_violation(draft(), [est(ci=ci)])


def test_ci_not_a_dict():
    item = est()
    item["numbers"]["ci"] = [0.5, 1.5]
    _expect_interval_violation(draft(), [item])


def test_different_design():
    viol = _expect_interval_violation(draft(), [est(design="did")])
    assert "did" in viol["detail"] and "observational" in viol["detail"]


def test_different_columns():
    viol = _expect_interval_violation(draft(), [est(treatment="other")])
    assert "other" in viol["detail"]
    _expect_interval_violation(draft(), [est(outcome="z")])


def test_reading_missing_columns():
    _expect_interval_violation(draft(), [est()], reading={"design": "observational"})
    _expect_interval_violation(draft(), [est()], reading=None)


def _iv_est(instrument="lottery", covariates=None, **kw):
    item = est(design="iv", **kw)
    item["numbers"]["roles"]["instrument"] = instrument
    item["numbers"]["covariates"] = covariates
    item["args"]["roles"]["instrument"] = instrument
    item["args"]["covariates"] = covariates
    return item


IV_READING = {"treatment": "t", "outcome": "y", "design": "iv", "design_reason": "",
              "roles": {"treatment": "t", "outcome": "y", "instrument": "lottery"}, "covariates": None,
              "excluded": []}


def test_same_roles_and_covariates_hold():
    v, fl = floor.apply(draft(), IV_READING, [_iv_est()])
    assert v["label"] == SUPPORTED and not fl["changed"]
    r = dict(IV_READING, covariates=["b", "a"])
    v, fl = floor.apply(draft(), r, [_iv_est(covariates=["a", "b"])])
    assert v["label"] == SUPPORTED and not fl["changed"]


def test_different_instrument_is_inconclusive():
    r = dict(IV_READING, roles={"treatment": "t", "outcome": "y", "instrument": "z2"})
    viol = _expect_interval_violation(draft(), [_iv_est()], reading=r)
    assert "instrument" in viol["detail"] and "z2" in viol["detail"]


def test_role_only_in_estimate_is_inconclusive():
    r = dict(IV_READING, roles={"treatment": "t", "outcome": "y"})
    _expect_interval_violation(draft(), [_iv_est()], reading=r)


def test_different_covariates_is_inconclusive():
    r = dict(IV_READING, covariates=["x"])
    viol = _expect_interval_violation(draft(), [_iv_est(covariates=None)], reading=r)
    assert "covariates" in viol["detail"]
    _expect_interval_violation(draft(), [_iv_est(covariates=["a"])], reading=dict(IV_READING, covariates=[]))
    _expect_interval_violation(draft(), [_iv_est(covariates=[])], reading=IV_READING)


def _facts(*names):
    return contract.evidence_item("e0", "table_facts", {}, numbers={"columns": [{"name": n} for n in names]})


def test_all_columns_named_equals_every_non_role_column():
    """Naming every non-role column and leaving covariates out are the same adjustment set."""
    facts = _facts("t", "y", "lottery", "a", "b")
    v, fl = floor.apply(draft(), dict(IV_READING, covariates=["b", "a"]), [facts, _iv_est(covariates=None)])
    assert v["label"] == SUPPORTED and not fl["changed"]
    v, fl = floor.apply(draft(), IV_READING, [facts, _iv_est(covariates=["a", "b"])])
    assert v["label"] == SUPPORTED and not fl["changed"]
    # a strict subset is still a different adjustment set
    viol = _expect_interval_violation(draft(), [facts, _iv_est(covariates=None)],
                                      reading=dict(IV_READING, covariates=["a"]))
    assert "covariates" in viol["detail"]
    _expect_interval_violation(draft(), [facts, _iv_est(covariates=["a"])], reading=IV_READING)


def test_rd_cutoff_compared_with_cutoff_used():
    item = est(design="rd")
    item["numbers"]["roles"]["running"] = "s"
    item["numbers"]["cutoff_used"] = 50.0
    r = {"treatment": "t", "outcome": "y", "design": "rd", "covariates": None,
         "roles": {"treatment": "t", "outcome": "y", "running": "s", "cutoff": 50}}
    v, fl = floor.apply(draft(), r, [item])
    assert v["label"] == SUPPORTED and not fl["changed"]
    r2 = dict(r, roles={**r["roles"], "cutoff": 60})
    viol = _expect_interval_violation(draft(), [item], reading=r2)
    assert "cutoff" in viol["detail"]
    r3 = dict(r, roles={k: v for k, v in r["roles"].items() if k != "cutoff"})
    v, _ = floor.apply(draft(), r3, [item])
    assert v["label"] == SUPPORTED          # reading leaves the cutoff to the data, as the estimate did


def test_reading_columns_from_roles_only():
    reading = {"design": "observational", "roles": {"treatment": "t", "outcome": "y"}}
    v, fl = floor.apply(draft(), reading, [est()])
    assert v["label"] == SUPPORTED and not fl["changed"]


def test_missing_estimate_id():
    viol = _expect_interval_violation(draft(eid=None), [est()])
    assert "estimate" in viol["detail"]
    _expect_interval_violation(draft(eid=""), [est()])
    viol = _expect_interval_violation(draft(eid="e9"), [est()])
    assert "e9" in viol["detail"]


def test_id_points_at_other_tool():
    item = evidence_item("e1", "design_tests", {}, numbers={"ci": {"95": [1, 2]}})
    _expect_interval_violation(draft(), [item])


def test_estimate_missing_design_in_numbers_falls_back_to_args():
    item = est()
    del item["numbers"]["design"]
    del item["numbers"]["roles"]
    v, _ = floor.apply(draft(), READING, [item])
    assert v["label"] == SUPPORTED


# ------------------------------------------------------------------ rule 1

def test_untestable_assumptions_added():
    e2 = evidence_item("e2", "design_tests", {}, numbers={"untestable": ["No unmeasured cause of both.", "A"]})
    e3 = evidence_item("e3", "sensitivity", {}, numbers={"untestable": ["A", "B"]})
    v, fl = floor.apply(draft(assumed=["A"]), READING, [est(), e2, e3])
    assert v["label"] == SUPPORTED
    assert v["assumed"] == ["A", "No unmeasured cause of both.", "B"]
    assert rules(fl) == ["assumed_is_not_tested"]
    assert "2 assumption" in fl["violations"][0]["detail"]
    assert fl["changed"] is True


def test_untestable_already_listed_is_no_violation():
    e2 = evidence_item("e2", "general_checks", {}, numbers={"untestable": ["A"]})
    v, fl = floor.apply(draft(assumed=["A"]), READING, [est(), e2])
    assert v["assumed"] == ["A"] and not fl["changed"]


def test_untestable_malformed_values_ignored():
    e2 = evidence_item("e2", "general_checks", {}, numbers={"untestable": [None, 3, "", "X"]})
    e3 = evidence_item("e3", "general_checks", {}, numbers={"untestable": "Y"})
    v, _ = floor.apply(draft(assumed=None), READING, [est(), e2, e3])
    assert v["assumed"] == ["X", "Y"]


def test_sensitivity_supports_becomes_not_informative():
    sens = evidence_item("e2", "sensitivity", {}, numbers={"robustness_value": 0.3})
    other = evidence_item("e3", "design_tests", {})
    judg = [{"evidence_id": "e2", "reading": "supports", "why": "robust"},
            {"evidence_id": "e3", "reading": "supports", "why": "ok"},
            {"evidence_id": "e2", "reading": "concern", "why": "x"}]
    v, fl = floor.apply(draft(judgments=judg), READING, [est(), sens, other])
    assert [j["reading"] for j in v["judgments"]] == ["not informative", "supports", "concern"]
    assert v["label"] == SUPPORTED
    assert rules(fl) == ["assumed_is_not_tested"]
    assert "e2" in fl["violations"][0]["action"]


def test_judgments_malformed_are_kept():
    judg = ["text", None, {"reading": "supports"}, {"evidence_id": "nope", "reading": "supports"}]
    v, fl = floor.apply(draft(judgments=judg), READING, [est()])
    assert v["judgments"] == judg and not fl["changed"]


# ------------------------------------------------------------------ combinations and malformed input

def test_rule1_and_rule2_together():
    e2 = evidence_item("e2", "sensitivity", {}, numbers={"untestable": ["U"]})
    judg = [{"evidence_id": "e2", "reading": "supports", "why": ""}]
    v, fl = floor.apply(draft(eid=None, judgments=judg), READING, [e2])
    assert rules(fl) == ["assumed_is_not_tested", "assumed_is_not_tested", "needs_a_real_interval"]
    assert v["label"] == INCONCLUSIVE and v["assumed"] == ["U"]


def test_unknown_label():
    v, fl = floor.apply(draft("Very supported"), READING, [est()])
    assert v["label"] == INCONCLUSIVE
    assert rules(fl) == ["unreadable_draft"]
    assert "Very supported" in fl["violations"][0]["detail"]
    v, fl = floor.apply(draft(None), READING, [est()])
    assert v["label"] == INCONCLUSIVE and fl["changed"]


@pytest.mark.parametrize("bad", [None, "Supported", 3, [], ["Supported"], object()])
def test_draft_not_a_dict(bad):
    v, fl = floor.apply(bad, READING, [est()])
    assert v["label"] == INCONCLUSIVE and fl["changed"]
    assert set(contract.empty_verdict()) <= set(v)


def test_missing_keys_filled():
    v, fl = floor.apply({"label": INCONCLUSIVE}, READING, None)
    assert set(contract.empty_verdict()) <= set(v)
    assert not fl["changed"]


@pytest.mark.parametrize("evidence", [None, "x", 5, [None, 3, "e1"], [{"id": "e1"}], [{"numbers": None}],
                                      [{"id": "e1", "tool": "estimate_effect", "ok": True, "numbers": "x"}]])
def test_malformed_evidence(evidence):
    v, fl = floor.apply(draft(), READING, evidence)
    assert v["label"] == INCONCLUSIVE
    assert isinstance(fl["violations"], list)


def test_inputs_not_mutated():
    e2 = evidence_item("e2", "sensitivity", {}, numbers={"untestable": ["U"]})
    items = [est(lo=-1, hi=1), e2]
    d = draft(judgments=[{"evidence_id": "e2", "reading": "supports", "why": ""}], assumed=["A"])
    reading = copy.deepcopy(READING)
    before = copy.deepcopy((d, reading, items))
    v, fl = floor.apply(d, reading, items)
    assert v["label"] == INCONCLUSIVE and v["assumed"] == ["A", "U"]
    assert (d, reading, items) == before
    assert v is not d and v["judgments"] is not d["judgments"]


# ------------------------------------------------------------------ property-style

def _rand_value(rng):
    return rng.choice([None, 0, -1, 1.5, float("nan"), float("inf"), "", "x", [], {}, True, [1, 2], {"a": 1}])


def _rand_ci(rng):
    a, b = rng.uniform(-3, 3), rng.uniform(-3, 3)
    return rng.choice([{"95": [min(a, b), max(a, b)]}, {"95": [a, b]}, {"95": [a, a]},
                       {"95": [float("nan"), b]}, {"95": None}, None, "ci", {"95": [a]}])


def _rand_item(rng, i):
    tool = rng.choice(contract.EVIDENCE_TOOLS + ("junk",))
    nums = {"design": rng.choice(contract.DESIGNS + (None,)),
            "roles": rng.choice([{"treatment": "t", "outcome": "y"}, {"treatment": "q", "outcome": "y"}, None]),
            "ci": _rand_ci(rng), "refused": rng.choice([False, True, None]),
            "outcome_sd": rng.choice([10.0, 0.0, None, float("nan"), 1.0]),
            "untestable": rng.choice([None, ["A"], ["B", "C"], "D", [None, 2]])}
    if rng.random() < 0.2:
        nums = _rand_value(rng)
    item = evidence_item(f"e{i}", tool, rng.choice([{}, {"design": "observational"}, None]),
                         ok=rng.choice([True, False, None]), numbers=nums if isinstance(nums, dict) else None)
    if not isinstance(nums, dict):
        item["numbers"] = nums
    if rng.random() < 0.05:
        return _rand_value(rng)
    return item


def _rand_draft(rng, n):
    if rng.random() < 0.05:
        return _rand_value(rng)
    d = {}
    if rng.random() < 0.95:
        d["label"] = rng.choice(contract.LABELS + ("Maybe", None, 3))
    if rng.random() < 0.9:
        d["estimate_evidence_id"] = rng.choice([f"e{i}" for i in range(n + 2)] + [None, 3])
    if rng.random() < 0.9:
        d["claimed_direction"] = rng.choice(["increase", "decrease", "none", None, "sideways"])
    if rng.random() < 0.7:
        d["assumed"] = rng.choice([[], ["A"], None, "A", [3]])
    if rng.random() < 0.7:
        d["judgments"] = [rng.choice([{"evidence_id": f"e{rng.randrange(n + 1)}",
                                       "reading": rng.choice(contract.READINGS)}, None, "x"])
                          for _ in range(rng.randrange(4))]
    return d


def test_property_never_upgrades_never_raises():
    rng = random.Random(12345)
    kept = 0
    for _ in range(3000):
        n = rng.randrange(5)
        evidence = [_rand_item(rng, i) for i in range(n)]
        if n and rng.random() < 0.5:  # a well-formed estimate, so some drafts can keep their label
            a, b = rng.uniform(-3, 3), rng.uniform(-3, 3)
            evidence[0] = est("e0", lo=min(a, b), hi=max(a, b), outcome_sd=rng.choice([10.0, 40.0, None]))
        d = _rand_draft(rng, n)
        reading = rng.choice([READING, {"design": "did", "treatment": "t", "outcome": "y"}, None, {}, "x"])
        before = copy.deepcopy((d, reading, evidence)) if not _has_nan(d, evidence) else None
        v, fl = floor.apply(d, reading, evidence)
        assert isinstance(v, dict) and isinstance(fl, dict)
        assert set(fl) == {"changed", "violations"}
        assert fl["changed"] == bool(fl["violations"])
        for viol in fl["violations"]:
            assert set(viol) == {"rule", "detail", "action"}
            assert isinstance(viol["detail"], str) and viol["detail"]
            assert viol["rule"] in ("assumed_is_not_tested", "needs_a_real_interval",
                                    "no_more_than_the_interval", "unreadable_draft")
        assert v["label"] in contract.LABELS
        draft_label = d.get("label") if isinstance(d, dict) else None
        if draft_label in contract.LABELS:
            assert v["label"] in (draft_label, INCONCLUSIVE)
            kept += v["label"] == draft_label != INCONCLUSIVE
        else:
            assert v["label"] == INCONCLUSIVE
        assert STRENGTH[v["label"]] <= STRENGTH.get(draft_label, 0)
        if before is not None:
            assert (d, reading, evidence) == before
    assert kept > 0  # the generator does reach labels the floor keeps


def _has_nan(*objs):
    # NaN != NaN breaks equality comparison of the deep copy; skip those cases for the mutation check
    def walk(x):
        if isinstance(x, float) and math.isnan(x):
            return True
        if isinstance(x, dict):
            return any(walk(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return any(walk(v) for v in x)
        return False
    return any(walk(o) for o in objs)
