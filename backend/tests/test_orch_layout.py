"""Part F: layout_facts. Small simulated tables with a known layout."""
import json
import re
import time

import numpy as np
import pandas as pd
import pytest

from causality.orchestrator import layout

BANNED = re.compile(r"\b(pass|passed|passes|fail|failed|fails|concern|concerning|warning|suspicious|"
                    r"regression discontinuity|difference[- ]in[- ]differences|instrumental variable)\b", re.I)


def sig(x):
    return 1 / (1 + np.exp(-x))


def anon(df):
    return df.rename(columns={c: f"c{i + 1:02d}" for i, c in enumerate(df.columns)})


def cutoff_table(seed=1, n=3000):
    r = np.random.default_rng(seed)
    s = np.clip(r.normal(50, 12, n), 5, 95).round(1)
    t = (s >= 50).astype(int)
    y = 2 + 0.03 * s + 0.8 * t + r.normal(0, 0.6, n)
    return pd.DataFrame({"score": s, "t": t, "female": r.integers(0, 2, n), "y": y.round(3),
                         "id": r.permutation(n) + 1000, "noise": r.normal(50, 10, n).round(2)})


def panel_table(seed=2, U=100, P=8):
    r = np.random.default_rng(seed)
    treated = np.zeros(U, int)
    treated[r.permutation(U)[:50]] = 1
    start = np.where(treated == 1, r.choice([4, 6], U), 99)
    rows = []
    for u in range(U):
        for p in range(1, P + 1):
            d = int(p >= start[u])
            rows.append((u + 1, 2010 + p, d, 10 + 0.3 * p + 1.5 * d + r.normal(0, 0.7), round(r.normal(50, 10), 2)))
    df = pd.DataFrame(rows, columns=["store", "year", "t", "y", "staff"])
    df.insert(0, "row_id", r.permutation(len(df)) + 1000)
    return df


def randomized_table(seed=3, n=3000):
    r = np.random.default_rng(seed)
    t = r.integers(0, 2, n)
    return pd.DataFrame({"t": t, "age": r.normal(40, 10, n).round(0), "prior": r.normal(0, 1, n).round(3),
                         "y": t + r.normal(0, 2, n), "id": r.permutation(n) + 1000,
                         "region": r.choice(["n", "s", "e", "w"], n)})


def confounded_table(seed=4, n=3000):
    r = np.random.default_rng(seed)
    x1, x2 = r.normal(0, 1, n), r.normal(0, 1, n)
    t = (r.uniform(size=n) < sig(1.2 * x1 + 0.8 * x2)).astype(int)
    return pd.DataFrame({"t": t, "x1": x1, "x2": x2, "y": x1 + 0.7 * x2 + r.normal(0, 1, n),
                         "id": r.permutation(n) + 1000})


def encouragement_table(seed=5, n=4000):
    r = np.random.default_rng(seed)
    u, z, x = r.normal(0, 1, n), r.integers(0, 2, n), r.normal(0, 1, n)
    t = (0.9 * z + u + r.normal(0, 0.8, n) > 0.45).astype(int)
    y = t - 1.3 * u + 0.5 * x + r.normal(0, 1, n)
    return pd.DataFrame({"z": z, "t": t, "x": x, "y": y, "id": r.permutation(n) + 1000,
                         "desk": r.normal(50, 10, n)})


def mediator_table(seed=6, n=3000):
    r = np.random.default_rng(seed)
    t = r.integers(0, 2, n)
    m = 0.8 * t + r.normal(0, 1, n)
    prior = r.normal(0, 1, n)
    return pd.DataFrame({"t": t, "age": r.normal(40, 10, n), "prior": prior, "m": m,
                         "y": 1.2 * m + 0.4 * prior + r.normal(0, 1, n)})


def _run(df, t="t", y="y"):
    lines, numbers = layout.run(df, {"treatment": t, "outcome": y})
    assert lines and all(isinstance(s, str) and s for s in lines)
    for s in lines:
        assert not BANNED.search(s), s
    json.dumps(numbers)  # plain JSON
    for k in ("thresholds", "panels", "treatment_predictability", "instrument_candidates", "associated_columns"):
        assert k in numbers
    return lines, numbers


def test_tool_spec_shape():
    s = layout.TOOL_SPEC
    assert s["name"] == "layout_facts" and s["strict"] is True and s["description"]
    sch = s["input_schema"]
    assert sch["additionalProperties"] is False
    assert set(sch["required"]) == set(sch["properties"]) == {"treatment", "outcome"}


@pytest.mark.parametrize("args", [{}, {"treatment": "t"}, {"treatment": "nope", "outcome": "y"},
                                  {"treatment": "t", "outcome": "t"}, {"treatment": 3, "outcome": "y"}])
def test_bad_args_raise_value_error(args):
    with pytest.raises(ValueError):
        layout.run(randomized_table(n=200), args)


@pytest.mark.parametrize("names", ["plain", "anon"])
def test_sharp_cutoff_found(names):
    df = cutoff_table()
    t, y = "t", "y"
    if names == "anon":
        cols = list(df.columns)
        df = anon(df)
        t, y = f"c{cols.index('t') + 1:02d}", f"c{cols.index('y') + 1:02d}"
        score = f"c{cols.index('score') + 1:02d}"
    else:
        score = "score"
    lines, nums = _run(df, t, y)
    th = nums["thresholds"]
    assert th and th[0]["column"] == score
    assert abs(th[0]["cutoff"] - 50) < 0.5 and th[0]["share_consistent"] > 0.99
    assert th[0]["n_below_near"] > 50 and th[0]["n_above_near"] > 50 and th[0]["window"] > 0
    assert any("switches at a single value of '" + score + "'" in s for s in lines)


def test_staggered_panel_found():
    lines, nums = _run(panel_table())
    assert nums["panels"], lines
    p = nums["panels"][0]
    assert (p["unit"], p["time"]) == ("store", "year")
    assert p["n_units"] == 100 and p["n_periods"] == 8
    assert p["share_units_switching_on"] == pytest.approx(0.5) and p["share_units_switching_off"] == 0
    assert p["n_never_treated"] == 50 and set(p["start_times"]) == {2014, 2016}
    assert nums["treatment_predictability"]["n_columns"] == 1  # row_id, store, year left out
    assert not nums["thresholds"]


def test_randomized_low_predictability_no_threshold():
    lines, nums = _run(randomized_table())
    tp = nums["treatment_predictability"]
    assert tp["metric"] == "auc" and abs(tp["value"] - 0.5) < 0.06
    assert not nums["thresholds"] and not nums["panels"]
    assert not nums["instrument_candidates"] and not nums["associated_columns"]
    assert {r["column"] for r in tp["per_column"]} == {"age", "prior", "region"}  # id left out


def test_confounded_predictable_and_associated():
    lines, nums = _run(confounded_table())
    assert nums["treatment_predictability"]["value"] > 0.7
    assoc = [r["column"] for r in nums["associated_columns"]]
    assert assoc[:2] == ["x1", "x2"]
    assert any("cause of treatment or a consequence" in s and "question for the user" in s for s in lines)


def test_random_encouragement_is_instrument_candidate():
    lines, nums = _run(encouragement_table())
    ic = nums["instrument_candidates"]
    assert ic and ic[0]["column"] == "z"
    z = ic[0]
    assert z["first_stage_f"] > 100 and z["predictability_from_others"] < 0.6
    assert z["n_values"] == 2 and len(z["treatment_by_value"]) == 2
    assert {"assoc_with_outcome", "p_value_outcome"} <= set(z)
    line = next(s for s in lines if s.startswith("'z'"))
    assert "only through treatment cannot be tested" in line


def test_mediator_reported_as_associated_with_limit_stated():
    lines, nums = _run(mediator_table())
    assert [r["column"] for r in nums["associated_columns"]] == ["m"]
    assert not nums["thresholds"]
    assert any("consequence" in s and "'m'" in s for s in lines)


def test_text_treatment_and_non_binary():
    df = randomized_table(n=500)
    df["arm"] = np.where(df["t"] == 1, "new", "old")
    lines, nums = _run(df.drop(columns="t"), "arm", "y")
    assert "read as 0 and 1" in lines[0]
    df2 = confounded_table(n=800)
    df2["dose"] = df2["x1"] * 2 + np.random.default_rng(0).normal(0, 1, 800)
    lines, nums = _run(df2.drop(columns="t"), "dose", "y")
    assert nums["treatment_predictability"]["metric"] == "r2" and nums["treatment_predictability"]["value"] > 0.5
    assert nums["thresholds"] == []


def test_speed_10k_by_30():
    r = np.random.default_rng(9)
    n = 10_000
    cols = {f"x{i}": r.normal(0, 1, n) for i in range(24)}
    cols["z"] = r.integers(0, 2, n)
    cols["score"] = r.normal(50, 10, n).round(1)
    cols["grp"] = r.choice(list("abcdef"), n)
    cols["id"] = r.permutation(n)
    t = ((cols["score"] >= 50) | ((cols["z"] == 1) & (r.uniform(size=n) < 0.3))).astype(int)
    df = pd.DataFrame(cols).assign(t=t, y=r.normal(0, 1, n) + t)
    assert df.shape == (n, 30)
    t0 = time.time()
    _run(df)
    assert time.time() - t0 < 6.0
