"""Part H: charts for the audit page. Known-truth data from causality.sim; no network, no model."""
import json
import math
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
import pytest

from causality import sim
from causality.orchestrator import evidence, screen

MAX = screen.MAX_POINTS


# --------------------------------------------------------------------------------- shape checks
def _isnum(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _numlist(v, allow_null=True):
    assert isinstance(v, list), v
    assert len(v) <= MAX
    for a in v:
        assert _isnum(a) or (allow_null and a is None), a


def _walk_lists(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from _walk_lists(v)
    elif isinstance(x, list):
        yield x
        for v in x:
            yield from _walk_lists(v)


def check_chart(c):
    """The chart is plain JSON and has the `data` shape charts.js reads for its kind."""
    assert isinstance(c, dict)
    json.dumps(c, allow_nan=False)
    assert set(c) == {"kind", "title", "data", "x_label", "y_label", "note"}
    for k in ("title", "x_label", "y_label", "note"):
        assert isinstance(c[k], str)
    assert c["title"].strip()
    for lst in _walk_lists(c):
        assert len(lst) <= MAX
    d = c["data"]
    kind = c["kind"]
    if kind == "bar":
        assert isinstance(d["labels"], list) and all(isinstance(s, str) for s in d["labels"])
        _numlist(d["values"])
        assert len(d["labels"]) == len(d["values"]) and any(_isnum(v) for v in d["values"])
        assert d["ref"] is None or _isnum(d["ref"])
        if d.get("errors") is not None:
            _numlist(d["errors"])
            assert len(d["errors"]) == len(d["values"])
        if "ref_band" in d:
            assert len(d["ref_band"]) == 2 and all(_isnum(v) for v in d["ref_band"])
    elif kind == "hist":
        _numlist(d["bins"], allow_null=False)
        assert len(d["bins"]) >= 2 and d["bins"] == sorted(d["bins"])
        assert d["series"]
        for s in d["series"]:
            assert isinstance(s["name"], str)
            assert len(s["counts"]) == len(d["bins"]) - 1
            assert all(isinstance(v, int) and v >= 0 for v in s["counts"])
    elif kind == "scatter":
        assert isinstance(d["points"], list) and isinstance(d["lines"], list)
        assert d["points"] or d["lines"]
        for p in d["points"]:
            assert _isnum(p["x"]) and _isnum(p["y"])
            assert p.get("group") is None or isinstance(p["group"], str)
        for ln in d["lines"]:
            _numlist(ln["x"]); _numlist(ln["y"])
            assert len(ln["x"]) == len(ln["y"]) and isinstance(ln["name"], str)
    elif kind == "line":
        assert d["series"]
        for s in d["series"]:
            assert isinstance(s["name"], str)
            _numlist(s["x"]); _numlist(s["y"])
            assert len(s["x"]) == len(s["y"]) and s["x"]
            for k in ("lo", "hi"):
                if s.get(k) is not None:
                    _numlist(s[k])
                    assert len(s[k]) == len(s["x"])
        for k in ("vline", "hline"):
            assert d.get(k) is None or _isnum(d[k])
    elif kind == "forest":
        assert d["rows"]
        for r in d["rows"]:
            assert isinstance(r["label"], str) and _isnum(r["est"])
            assert r["lo"] is None or _isnum(r["lo"])
            assert r["hi"] is None or _isnum(r["hi"])
        assert d["ref"] is None or _isnum(d["ref"])
    else:
        raise AssertionError(f"unexpected kind {kind}")


# --------------------------------------------------------------------------------- evidence fixtures
@pytest.fixture(scope="module")
def did():
    df, _ = sim.did_staggered(units=120, seed=5)
    roles = {"treatment": "policy", "outcome": "sales", "unit": "state", "time": "year"}
    est = evidence.call("estimate_effect", {"design": "did", "roles": roles}, df, "e1")
    tests = evidence.call("design_tests", {"design": "did", "roles": roles}, df, "e2")
    return df, est, tests


@pytest.fixture(scope="module")
def rd():
    df, _ = sim.rd(n=3000)
    roles = {"treatment": "scholarship", "outcome": "gpa_later", "running": "exam_score"}
    return df, evidence.call("estimate_effect", {"design": "rd", "roles": roles}, df, "e1")


@pytest.fixture(scope="module")
def iv():
    df, _ = sim.iv(n=3000)
    roles = {"treatment": "enrolled", "outcome": "test_score", "instrument": "lottery"}
    return df, evidence.call("estimate_effect", {"design": "iv", "roles": roles}, df, "e1")


@pytest.fixture(scope="module")
def obs():
    df, _ = sim.observational(n=2000)
    gen = evidence.call("general_checks", {"treatment": "program", "outcome": "earnings"}, df, "e1")
    est = evidence.call("estimate_effect", {"design": "observational",
                                            "roles": {"treatment": "program", "outcome": "earnings"}}, df, "e2")
    return df, gen, est


def _kinds(charts):
    return [c["kind"] for c in charts]


def test_did_estimate_gives_forest_and_trend_lines(did):
    df, est, _ = did
    assert est["ok"]
    charts = screen.charts_for(est, df)
    assert _kinds(charts) == ["forest", "line"]
    for c in charts:
        check_chart(c)
    forest, line = charts
    assert len(forest["data"]["rows"]) == 1 + len(est["numbers"]["comparisons"])
    assert forest["data"]["rows"][0]["est"] == pytest.approx(est["numbers"]["estimate"])
    assert "policy" in forest["title"] and "sales" in forest["title"]
    names = [s["name"] for s in line["data"]["series"]]
    assert len(names) == 2 and "ever" in names[0] and "never" in names[1]
    assert line["data"]["series"][0]["x"] == sorted(df["year"].unique().astype(float).tolist())
    assert line["data"]["vline"] == 2013.0      # first adoption cohort starts at period 3
    assert line["x_label"] == "year"


def test_did_design_tests_give_event_study_line(did):
    df, _, tests = did
    charts = screen.charts_for(tests, df)
    assert _kinds(charts) == ["line"]
    c = charts[0]
    check_chart(c)
    s = c["data"]["series"][0]
    assert -1.0 in s["x"] and s["y"][s["x"].index(-1.0)] == 0.0
    assert s["lo"] is not None and all(lo <= y <= hi for lo, y, hi in zip(s["lo"], s["y"], s["hi"]))
    assert c["data"]["vline"] == 0.0 and c["data"]["hline"] == 0.0


def test_rd_estimate_gives_binned_scatter_with_cutoff(rd):
    df, est = rd
    charts = screen.charts_for(est, df)
    assert _kinds(charts) == ["forest", "scatter"]
    for c in charts:
        check_chart(c)
    sc = charts[1]
    cut = est["numbers"]["cutoff_used"]
    cut_line = [ln for ln in sc["data"]["lines"] if ln["name"].startswith("cutoff")]
    assert len(cut_line) == 1 and cut_line[0]["x"] == [cut, cut]
    groups = {p["group"] for p in sc["data"]["points"]}
    assert groups == {"below the cutoff", "at or above the cutoff"}
    assert all(p["x"] < cut for p in sc["data"]["points"] if p["group"] == "below the cutoff")
    assert sc["x_label"] == "exam_score" and "gpa_later" in sc["y_label"]
    # the jump is visible: the fitted lines differ by about the true effect (3) at the cutoff
    fits = {ln["name"]: ln for ln in sc["data"]["lines"] if ln["name"].startswith("straight")}
    below = fits["straight-line fit below the cutoff"]
    above = fits["straight-line fit at or above the cutoff"]
    assert above["y"][0] - below["y"][1] == pytest.approx(3.0, abs=1.0)


def test_iv_estimate_gives_bars_by_instrument(iv):
    df, est = iv
    charts = screen.charts_for(est, df)
    assert _kinds(charts) == ["forest", "bar", "bar"]
    for c in charts:
        check_chart(c)
    share, outcome = charts[1], charts[2]
    assert share["data"]["labels"] == ["lottery = 0", "lottery = 1"]
    expect = df.groupby("lottery")["enrolled"].mean().tolist()
    assert share["data"]["values"] == pytest.approx(expect)
    assert "Share given enrolled" in share["title"]
    assert outcome["data"]["values"] == pytest.approx(df.groupby("lottery")["test_score"].mean().tolist())


def test_general_checks_give_standardized_difference_bar(obs):
    df, gen, est = obs
    charts = screen.charts_for(gen, df)
    assert _kinds(charts) == ["bar"]
    c = charts[0]
    check_chart(c)
    assert c["data"]["labels"][0] == "x1"          # largest absolute difference first
    assert set(c["data"]["labels"]) == {"x1", "x2", "member"}
    assert "ref_band" in c["data"]
    # observational estimate: forest only
    charts = screen.charts_for(est, df)
    assert _kinds(charts) == ["forest"]
    check_chart(charts[0])


def test_charts_for_never_raises_and_returns_empty_when_nothing_to_show(did):
    df, est, _ = did
    assert screen.charts_for({"ok": False, "tool": "estimate_effect", "numbers": {}}, df) == []
    assert screen.charts_for(None, df) == []
    assert screen.charts_for("junk", df) == []
    assert screen.charts_for({"ok": True, "tool": "table_facts", "numbers": {"n_rows": 3}}, df) == []
    assert screen.charts_for({"ok": True, "tool": "estimate_effect", "numbers": None}, df) == []
    # refused estimate with no comparisons -> no forest
    refused = {"ok": True, "tool": "estimate_effect",
               "numbers": {"design": "observational", "refused": True, "estimate": None, "comparisons": []}}
    assert screen.charts_for(refused, df) == []
    # roles that point at missing columns: the forest still comes, the picture from the table does not
    bad = json.loads(json.dumps(est))
    bad["numbers"]["roles"]["time"] = "no_such_column"
    assert _kinds(screen.charts_for(bad, df)) == ["forest"]
    # no table: only the charts that need none
    assert _kinds(screen.charts_for(est, None)) == ["forest"]
    # garbage inside numbers
    weird = {"ok": True, "tool": "estimate_effect",
             "numbers": {"design": "rd", "estimate": float("nan"), "ci": "x", "comparisons": [1, {"estimate": "a"}],
                         "roles": 5}}
    assert screen.charts_for(weird, df) == []
    junk_tests = {"ok": True, "tool": "design_tests",
                  "numbers": {"tests": [{"metrics": {"event_times": [1, 2], "coefficients": [1.0]}}, None]}}
    assert screen.charts_for(junk_tests, df) == []


def test_nan_and_inf_are_dropped_or_nulled():
    item = {"ok": True, "tool": "estimate_effect",
            "numbers": {"design": "observational", "method": "m", "estimate": 1.0,
                        "ci": {"95": [float("-inf"), 2.0]},
                        "comparisons": [{"method": "a", "estimate": np.float64("nan"), "ci95": [0, 1]},
                                        {"method": "b", "estimate": np.float32(0.5), "ci95": [np.nan, 1]}],
                        "roles": {"treatment": "t", "outcome": "y"}}}
    charts = screen.charts_for(item, None)
    assert len(charts) == 1
    check_chart(charts[0])
    rows = charts[0]["data"]["rows"]
    assert [r["label"] for r in rows] == ["m (main)", "b"]
    assert rows[0]["lo"] is None and rows[1]["lo"] is None and type(rows[1]["est"]) is float


def test_large_event_study_is_capped():
    k = 900
    item = {"ok": True, "tool": "design_tests",
            "numbers": {"roles": {"outcome": "y"},
                        "tests": [{"metrics": {"event_times": list(range(-450, 450)),
                                               "coefficients": [0.1] * k, "se": [0.05] * k,
                                               "reference_period": -1}}]}}
    charts = screen.charts_for(item, None)
    assert len(charts) == 1
    check_chart(charts[0])


# --------------------------------------------------------------------------------- draw
@pytest.fixture(scope="module")
def big():
    rng = np.random.default_rng(0)
    n = 5000
    df = pd.DataFrame({
        "age": rng.normal(40, 10, n),
        "income": rng.lognormal(10, 1, n),
        "region": rng.choice(["north", "south", "east", "west", "centre", "isles", "far", "near"], n),
        "treated": rng.binomial(1, 0.4, n),
        "year": rng.integers(2000, 2020, n),
        "city": [f"c{i}" for i in rng.integers(0, 100, n)],
        "when": pd.Timestamp("2020-01-01") + pd.to_timedelta(rng.integers(0, 900, n), unit="D"),
    })
    df.loc[::17, "income"] = np.nan
    df.loc[::23, "age"] = np.inf
    return df


def _args(**kw):
    a = {"evidence_id": None, "kind": None, "x": None, "y": None, "group": None, "title": "", "note": ""}
    a.update(kw)
    return a


@pytest.mark.parametrize("kw,kind", [
    (dict(kind="hist", x="age"), "hist"),
    (dict(kind="hist", x="income", group="region"), "hist"),
    (dict(kind="scatter", x="age", y="income"), "scatter"),
    (dict(kind="scatter", x="age", y="income", group="treated"), "scatter"),
    (dict(kind="line", x="year", y="income", group="treated"), "line"),
    (dict(kind="line", x="age", y="income"), "line"),
    (dict(kind="line", x="when", y="income"), "line"),
    (dict(kind="bar", x="region", y="income"), "bar"),
    (dict(kind="bar", x="city", y="income"), "bar"),
    (dict(kind="bar", x="age", y="income"), "bar"),
    (dict(kind="bar", x="region"), "bar"),
    (dict(kind="bar", x="age"), "bar"),
    (dict(x="age"), "hist"),
    (dict(x="region"), "bar"),
    (dict(x="age", y="income"), "scatter"),
    (dict(x="treated", y="income"), "bar"),
])
def test_draw_kinds_are_valid_json_and_shape(big, kw, kind):
    c = screen.draw(big, _args(**kw))
    assert c["kind"] == kind
    check_chart(c)
    assert kw["x"] in c["title"] or kw["x"] in c["x_label"] or kw["x"] in c["y_label"]


def test_draw_uses_given_title_and_note(big):
    c = screen.draw(big, _args(kind="hist", x="age", title="Ages of people in the study", note="Look at the tails."))
    assert c["title"] == "Ages of people in the study"
    assert c["note"].startswith("Look at the tails.")


def test_draw_hist_counts_and_groups(big):
    c = screen.draw(big, _args(kind="hist", x="treated", group="region"))
    names = [s["name"] for s in c["data"]["series"]]
    assert len(names) == screen.MAX_GROUPS and all(n.startswith("region = ") for n in names)
    assert "most common" in c["note"]
    c = screen.draw(big, _args(kind="hist", x="treated"))
    assert sum(c["data"]["series"][0]["counts"]) == len(big)


def test_draw_scatter_small_is_raw_large_is_binned(big):
    small = big.head(150)
    c = screen.draw(small, _args(kind="scatter", x="age", y="year"))
    finite = np.isfinite(small["age"]).sum()
    assert len(c["data"]["points"]) == finite
    c = screen.draw(big, _args(kind="scatter", x="year", y="age"))
    assert len(c["data"]["points"]) <= screen.SCATTER_BINS
    assert "average" in c["note"]


def test_draw_bar_means_match(big):
    c = screen.draw(big, _args(kind="bar", x="treated", y="year"))
    m = big.groupby("treated")["year"].mean()
    got = dict(zip(c["data"]["labels"], c["data"]["values"]))
    assert got["0"] == pytest.approx(m[0]) and got["1"] == pytest.approx(m[1])
    c = screen.draw(big, _args(kind="bar", x="city", y="year"))
    assert len(c["data"]["labels"]) == screen.MAX_BARS


def test_draw_line_means_match():
    df = pd.DataFrame({"t": [1, 1, 2, 2, 3, 3], "y": [1.0, 3.0, 2.0, 4.0, 5.0, 7.0], "g": list("ababab")})
    c = screen.draw(df, _args(kind="line", x="t", y="y"))
    s = c["data"]["series"][0]
    assert s["x"] == [1.0, 2.0, 3.0] and s["y"] == [2.0, 3.0, 6.0]
    c = screen.draw(df, _args(kind="line", x="t", y="y", group="g"))
    assert [s["name"] for s in c["data"]["series"]] == ["g = a", "g = b"]
    check_chart(c)


@pytest.mark.parametrize("kw,msg", [
    (dict(kind="pie", x="age"), "Unknown chart kind"),
    (dict(kind="hist"), "needs an x column"),
    (dict(kind="hist", x="nope"), "not in the table"),
    (dict(kind="hist", x="AGE"), "Did you mean 'age'"),
    (dict(kind="hist", x="region"), "no numbers"),
    (dict(kind="hist", x="age", y="income"), "leave y null"),
    (dict(kind="scatter", x="age"), "needs a y column"),
    (dict(kind="line", x="age"), "needs a y column"),
    (dict(kind="scatter", x="age", y="region"), "no numbers"),
    (dict(kind="bar", x="region", y="income", group="treated"), "no group"),
    (dict(kind="hist", x="age", group="missing"), "not in the table"),
    (dict(kind="hist", x=5), "column name"),
])
def test_draw_bad_args_raise_plain_value_error(big, kw, msg):
    with pytest.raises(ValueError, match=msg):
        screen.draw(big, _args(**kw))


def test_draw_rejects_no_table_and_bad_args():
    with pytest.raises(ValueError):
        screen.draw(pd.DataFrame(), _args(x="a"))
    with pytest.raises(ValueError):
        screen.draw(None, _args(x="a"))
    with pytest.raises(ValueError):
        screen.draw(pd.DataFrame({"a": [1, 2]}), "x=a")
    with pytest.raises(ValueError, match="no numbers"):
        screen.draw(pd.DataFrame({"a": [np.nan, np.nan]}), _args(kind="hist", x="a"))


def test_draw_constant_column_and_all_null_y():
    df = pd.DataFrame({"a": [3.0] * 10, "b": [np.nan] * 9 + [1.0]})
    check_chart(screen.draw(df, _args(kind="hist", x="a")))
    c = screen.draw(df, _args(kind="scatter", x="a", y="b"))
    check_chart(c)
    assert len(c["data"]["points"]) == 1


# --------------------------------------------------------------------------------- tool spec
def test_tool_spec_shape():
    spec = screen.TOOL_SPEC
    assert spec["name"] == "show_chart" and spec["strict"] is True
    schema = spec["input_schema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"]) == {
        "evidence_id", "kind", "x", "y", "group", "title", "note"}
    for k in ("evidence_id", "kind", "x", "y", "group"):
        types = [o.get("type") for o in schema["properties"][k]["anyOf"]]
        assert "null" in types
    assert schema["properties"]["title"]["type"] == "string"
    json.dumps(spec, allow_nan=False)
