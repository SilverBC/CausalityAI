import io
import json
from unittest import mock

import numpy as np
import pandas as pd

from causality import data, sim
from causality.checks import attrition, interference, literature
from causality.schema import Claim, Roles, Status, to_jsonable


def _rct(**kw):
    df, truth = sim.rct(**kw)
    prep = data.prepare(df, Roles(treatment="treated", outcome="score"))
    design = data.resolve_design(prep, "experiment")
    return prep, design, truth


# ------------------------------------------------------------------ attrition
def test_attrition_none_missing_passes():
    prep, design, _ = _rct(attrition=0.0)
    s = attrition.run(prep, design, None)
    assert s.error is None and s.key == "missing_outcomes"
    assert s.status == Status.PASS and s.caps_verdict is False
    assert any(a.key == "no_missing_outcomes" and a.status == Status.PASS for a in s.assumptions)
    json.dumps(to_jsonable(s))


def test_attrition_differential_concern_or_fail():
    prep, design, _ = _rct(attrition=0.3, differential=0.15)
    s = attrition.run(prep, design, None)
    assert s.error is None
    assert s.status in (Status.CONCERN, Status.FAIL)
    assert s.caps_verdict is True
    mnar = [a for a in s.assumptions if a.key == "missing_not_outcome_related"]
    assert mnar and mnar[0].status == Status.ASSUMED and mnar[0].testable is False
    assert "lee_bounds" in s.metrics and "manski_bounds" in s.metrics
    assert s.metrics["missingness_predicted_p"] is not None
    assert "approximate" in s.metrics["boundary_source"]
    forest = [c for c in s.charts if c.kind == "forest"][0]
    assert len(forest.data["rows"]) == 3
    json.dumps(to_jsonable(s))


def test_lee_bounds_contain_truth():
    prep, design, truth = _rct(n=6000, attrition=0.15, differential=0.1, seed=3)
    s = attrition.run(prep, design, None)
    lee = s.metrics["lee_bounds"]
    assert lee["lower"] <= truth["effect"] <= lee["upper"]
    man = s.metrics["manski_bounds"]
    assert man["lower"] <= lee["lower"] and lee["upper"] <= man["upper"]


def test_attrition_binary_outcome_manski_unit_range():
    prep, design, _ = _rct(attrition=0.1, binary_outcome=True)
    s = attrition.run(prep, design, None)
    assert s.metrics["manski_bounds"]["outcome_min"] == 0.0
    assert s.metrics["manski_bounds"]["outcome_max"] == 1.0


def test_wwc_boundary_shape():
    assert attrition.max_differential(0, "cautious") < attrition.max_differential(0, "optimistic")
    assert attrition.within(10, 5, "cautious") and not attrition.within(10, 8, "cautious")
    assert attrition.within(10, 8, "optimistic")
    assert not attrition.within(70, 0, "optimistic")
    assert len(attrition.CAUTIOUS) == len(attrition.OPTIMISTIC) == 66


def test_attrition_continuous_treatment():
    df, _ = sim.dose()
    df.loc[df.sample(frac=0.2, random_state=0).index, "grade"] = np.nan
    prep = data.prepare(df, Roles(treatment="study_hours", outcome="grade"))
    design = data.resolve_design(prep, "observational")
    s = attrition.run(prep, design, None)
    assert s.error is None
    assert s.metrics["lee_bounds"] is None
    assert "missing_treatment_correlation" in s.metrics


def test_attrition_never_raises():
    prep, design, _ = _rct(attrition=0.1)
    prep.roles.outcome = "nope"
    s = attrition.run(prep, design, None)
    assert s.error


# ------------------------------------------------------------------ interference
def _cluster(saturation=True):
    df, truth = sim.cluster_saturation()
    roles = Roles(treatment="treated", outcome="consumption", cluster="village",
                  saturation="saturation" if saturation else None, randomized=True)
    prep = data.prepare(df, roles)
    design = data.resolve_design(prep, "experiment")
    return prep, design, truth


def test_spillover_detected():
    prep, design, truth = _cluster()
    s = interference.run(prep, design, None)
    assert s.error is None and s.key == "interference"
    assert s.status == Status.CONCERN
    lo, hi = s.metrics["spillover_ci95"]
    assert lo <= truth["spillover"] <= hi
    assert abs(s.metrics["spillover_slope"] - truth["spillover"]) < 0.4
    assert s.charts and s.charts[0].kind == "line"
    json.dumps(to_jsonable(s))


def test_no_saturation_is_assumed():
    prep, design, _ = _cluster(saturation=False)
    s = interference.run(prep, design, None)
    assert s.status == Status.ASSUMED
    a = s.assumptions[0]
    assert a.status == Status.ASSUMED and a.testable is False
    assert "standard errors only" in s.summary


def test_plain_rct_interference_assumed():
    prep, design, _ = _rct()
    s = interference.run(prep, design, None)
    assert s.status == Status.ASSUMED


# ------------------------------------------------------------------ literature
def _claim():
    return Claim(raw="Tutoring raises test scores", treatment_phrase="tutoring", outcome_phrase="test scores")


def test_literature_offline():
    prep, design, _ = _rct()
    with mock.patch("urllib.request.urlopen") as m:
        s = literature.run(prep, design, _claim())
        m.assert_not_called()
    assert s.key == "literature" and s.status is None and s.affects_grade is False
    refs = s.metrics["references"]
    assert any("Athey" in r["authors"] for r in refs)
    assert "does not affect the grade" in s.summary


def test_literature_online_mocked():
    payload = {"results": [{"display_name": "A study", "publication_year": 2020, "doi": "https://doi.org/10.1/x",
                            "authorships": [{"author": {"display_name": "Doe"}}], "id": "W1"}]}
    resp = mock.MagicMock()
    resp.read.return_value = json.dumps(payload).encode()
    resp.__enter__.return_value = resp
    prep, design, _ = _rct()
    with mock.patch("urllib.request.urlopen", return_value=resp) as m:
        s = literature.run(prep, design, _claim(), online=True)
    req = m.call_args[0][0]
    assert "tutoring" in req.full_url and "api.openalex.org" in req.full_url
    assert req.get_header("User-agent") == "CausalityAI"
    assert any(r["source"] == "openalex" and r["doi"] == "10.1/x" for r in s.metrics["references"])


def test_literature_online_failure():
    prep, design, _ = _rct()
    with mock.patch("urllib.request.urlopen", side_effect=OSError("down")):
        s = literature.run(prep, design, _claim(), online=True)
    assert s.error is None
    assert any("unavailable" in w.lower() for w in s.warnings)


def test_literature_designs():
    df, _ = sim.iv()
    prep = data.prepare(df, Roles(treatment="enrolled", outcome="test_score", instrument="lottery"))
    design = data.resolve_design(prep, "iv")
    s = literature.run(prep, design, None)
    assert any("Angrist" in r["authors"] for r in s.metrics["references"])
