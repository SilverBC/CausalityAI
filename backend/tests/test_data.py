import numpy as np
import pandas as pd
import pytest

from causality import data, sim
from causality.data import InputError
from causality.schema import Roles, Status


def test_types_bool_onehot_impute_and_duplicates():
    df = pd.DataFrame({
        "flag": ["yes", "no"] * 15,
        "grp": ["a", "b", "c"] * 10,
        "x": [1.0, np.nan] + list(np.arange(28.0)),
        "y": np.arange(30.0),
    })
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    p = data.prepare(df, Roles(treatment="flag", outcome="y"))
    kinds = {c.name: c.kind for c in p.columns}
    assert kinds["flag"] == "binary" and set(p.df["flag"].unique()) == {0.0, 1.0}
    assert "grp=b" in p.df and "grp=c" in p.df and "grp" not in p.adjustment_set
    assert "x__missing" in p.df and p.df["x"].isna().sum() == 0
    assert p.n_duplicates == 1 and len(p.df) == len(df)  # kept, not deleted
    assert any("duplicate" in n for n in p.notes)
    assert "flag" not in p.adjustment_set and "y" not in p.adjustment_set


def test_outcome_and_treatment_not_imputed():
    df, _ = sim.rct(attrition=0.2)
    p = data.prepare(df, Roles(treatment="treated", outcome="score"))
    assert p.df["score"].isna().sum() > 0
    assert "score__missing" not in p.df


def test_post_treatment_never_in_adjustment_set():
    df, _ = sim.rct_with_post_treatment()
    p = data.prepare(df, Roles(treatment="treated", outcome="score", post_treatment=["engagement"]))
    assert "engagement" not in p.adjustment_set
    p2 = data.prepare(df, Roles(treatment="treated", outcome="score"))
    data.mark_post_treatment(p2, ["engagement"], "test")
    assert "engagement" not in p2.adjustment_set


def test_missing_role_column_is_plain_error():
    df, _ = sim.rct()
    with pytest.raises(InputError, match="not in the table"):
        data.prepare(df, Roles(treatment="nope", outcome="score"))


@pytest.mark.parametrize("name,expected", [
    ("rct", "observational"),            # never an experiment unless the user says so
    ("observational", "observational"),
    ("did", "did"), ("iv", "iv"), ("rd", "rd"),
    ("rct_post", "observational"),       # a post-treatment column is not a running variable
])
def test_design_inference(name, expected):
    df, tr = sim.ALL[name]()
    p = data.prepare(df, Roles(treatment=tr["treatment"], outcome=tr["outcome"]))
    assert data.resolve_design(p, None).kind == expected


def test_randomized_flag_gives_experiment():
    df, _ = sim.rct()
    p = data.prepare(df, Roles(treatment="treated", outcome="score", randomized=True))
    assert data.resolve_design(p, None).kind == "experiment"


def test_declared_conflicts_are_failed_assumptions():
    df, _ = sim.rd()
    p = data.prepare(df, Roles(treatment="scholarship", outcome="gpa_later", running="exam_score", cutoff=90))
    d = data.resolve_design(p, "rd")
    assert any(a.status == Status.FAIL for a in d.conflicts)
    df, _ = sim.observational()
    p = data.prepare(df, Roles(treatment="program", outcome="earnings"))
    d = data.resolve_design(p, "did")
    assert any(a.status == Status.FAIL and a.key.startswith("decl_") for a in d.conflicts)


def test_whole_word_hints():
    assert data.hint("lottery_win", "instrument") == 1.0
    assert data.hint("lotteryish", "instrument") == 0.0
    assert data.words("examScore2020") == ["exam", "score2020"]
