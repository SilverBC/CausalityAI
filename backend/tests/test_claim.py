import pytest

from causality import data, sim
from causality.claim import parse_claim
from causality.schema import Claim


@pytest.fixture(scope="module")
def P():
    names = ["rct", "observational", "observational_hidden", "iv", "rd", "dose", "did"]
    return {k: data.prepare(sim.ALL[k]()[0]) for k in names}


def test_percent_among_women(P):
    c = parse_claim("Treatment increases score by 15% among women", P["rct"])
    assert (c.treatment_col, c.outcome_col) == ("treated", "score")
    assert c.direction == "increase" and c.size == 15 and c.size_unit == "percent"
    assert c.population_filter == {"column": "female", "op": "==", "value": 1}
    assert not c.needs_confirmation and c.testable
    assert c.estimand.contrast == "relative"
    assert "among women" in c.estimand.text


def test_at_least_points(P):
    c = parse_claim("The program raises earnings by at least 2 points", P["observational"])
    assert (c.treatment_col, c.outcome_col) == ("program", "earnings")
    assert c.treatment_score == 1.0 and c.outcome_score == 1.0
    assert (c.size, c.size_unit, c.bound) == (2.0, "absolute", "at_least")
    assert c.estimand.contrast == "difference" and c.estimand.target == "ATE"


def test_negated_decrease(P):
    c = parse_claim("Coffee does not reduce health", P["observational_hidden"])
    assert (c.treatment_col, c.outcome_col) == ("coffee", "health")
    assert c.direction == "decrease" and c.negated
    assert any("negated" in n for n in c.notes)


def test_no_effect(P):
    c = parse_claim("Coffee has no effect on health", P["observational_hidden"])
    assert c.direction == "no_effect" and not c.negated
    assert (c.treatment_col, c.outcome_col) == ("coffee", "health")


def test_does_not_affect(P):
    c = parse_claim("Coffee doesn't affect health", P["observational_hidden"])
    assert c.direction == "no_effect" and not c.negated


def test_sd_comparative(P):
    c = parse_claim("Enrolled students score 0.2 SD higher on test_score", P["iv"])
    assert (c.treatment_col, c.outcome_col) == ("enrolled", "test_score")
    assert c.direction == "increase" and c.size == 0.2 and c.size_unit == "sd"
    assert c.treatment_score >= 0.8 and not c.needs_confirmation


def test_doubles(P):
    c = parse_claim("Scholarship doubles gpa_later", P["rd"])
    assert (c.treatment_col, c.outcome_col) == ("scholarship", "gpa_later")
    assert (c.direction, c.size, c.size_unit) == ("increase", 2.0, "times")
    assert c.estimand.contrast == "relative"


def test_diminishing_dose(P):
    c = parse_claim("Study hours improve grade, with diminishing returns", P["dose"])
    assert (c.treatment_col, c.outcome_col) == ("study_hours", "grade")
    assert c.direction == "increase" and c.shape == "diminishing"
    assert c.estimand.contrast == "dose_response"


def test_mediation_untestable(P):
    c = parse_claim("The lottery affects test score through enrollment", P["iv"])
    assert not c.testable and "mediation" in c.untestable_reason
    assert (c.treatment_col, c.outcome_col) == ("lottery", "test_score")
    assert c.direction == "nonzero"


def test_counterfactual_untestable(P):
    c = parse_claim("If they had not been treated their score would have been lower", P["rct"])
    assert not c.testable and "counterfactual" in c.untestable_reason
    assert (c.treatment_col, c.outcome_col) == ("treated", "score")
    assert c.direction == "increase"


def test_comparison_untestable(P):
    c = parse_claim("Drug A lowers pressure more than drug B", P["rct"])
    assert not c.testable and "another treatment" in c.untestable_reason
    assert c.direction == "decrease"
    assert c.needs_confirmation


def test_control_comparison_is_testable(P):
    c = parse_claim("Treated people have higher scores than untreated people", P["rct"])
    assert c.testable and c.direction == "increase"
    assert (c.treatment_col, c.outcome_col) == ("treated", "score")


def test_misspelling_needs_confirmation(P):
    c = parse_claim("Treatment increases scroe", P["rct"])
    assert c.outcome_col == "score" and c.outcome_score < 0.8 and c.needs_confirmation
    assert c.candidates["outcome"][0][0] == "score"
    assert len(c.candidates["treatment"]) <= 3


def test_synonym_needs_confirmation(P):
    c = parse_claim("The program raises income", P["observational"])
    assert c.outcome_col == "earnings" and 0 < c.outcome_score < 0.8
    assert c.needs_confirmation


def test_age_population(P):
    c = parse_claim("Treatment raises score among people over 40", P["rct"])
    assert c.population_filter == {"column": "age", "op": ">", "value": 40}
    assert c.population_score >= 0.8
    assert c.outcome_col == "score"


def test_aged_plus_and_pp(P):
    c = parse_claim("Treatment increases score by 5 percentage points for people aged 40+", P["rct"])
    assert c.size_unit == "pp" and c.size == 5
    assert c.population_filter == {"column": "age", "op": ">=", "value": 40}


def test_onehot_level(P):
    c = parse_claim("Treatment raises score in the north", P["rct"])
    assert c.population_filter == {"column": "region=north", "op": "==", "value": 1}


def test_reference_level_uses_raw(P):
    c = parse_claim("Treatment raises score in the east", P["rct"])
    assert c.population_filter == {"column": "region", "op": "==", "value": "east", "raw": True}


def test_men_hedge_about(P):
    c = parse_claim("Among men, treatment may increase score by about 1.5 points", P["rct"])
    assert c.population_filter == {"column": "female", "op": "==", "value": 0}
    assert "may" in c.hedges and c.bound == "about" and c.size == 1.5
    assert c.treatment_col == "treated"


def test_no_more_than(P):
    c = parse_claim("Treatment raises score by no more than 3 points", P["rct"])
    assert c.bound == "at_most" and c.size == 3


def test_more_the_more_linear(P):
    c = parse_claim("The more study hours, the higher the grade", P["dose"])
    assert c.shape == "linear" and c.direction == "increase"
    assert (c.treatment_col, c.outcome_col) == ("study_hours", "grade")


def test_passive(P):
    c = parse_claim("Earnings are raised by the program", P["observational"])
    assert (c.treatment_col, c.outcome_col) == ("program", "earnings")
    assert c.direction == "increase"


def test_binary_population_and_lead_to(P):
    c = parse_claim("The program leads to higher earnings for members", P["observational"])
    assert c.direction == "increase"
    assert c.population_filter == {"column": "member", "op": "==", "value": 1}
    assert c.outcome_col == "earnings"


def test_threshold_and_nonmonotone(P):
    c = parse_claim("The scholarship increases gpa_later only above a threshold", P["rd"])
    assert c.shape == "threshold" and c.outcome_col == "gpa_later"
    c = parse_claim("Study hours have a U-shaped effect on grade", P["dose"])
    assert c.shape == "nonmonotone" and c.outcome_col == "grade"


def test_hedges_associated_and_times(P):
    c = parse_claim("Coffee is associated with lower health", P["observational_hidden"])
    assert c.direction == "decrease" and "associated" in c.hedges
    c = parse_claim("Treatment tends to raise score 2x", P["rct"])
    assert (c.size, c.size_unit) == (2.0, "times") and "tends to" in c.hedges


def test_empty_and_garbage_never_raise(P):
    c = parse_claim("", P["rct"])
    assert isinstance(c, Claim) and c.raw == "" and not c.testable
    assert c.untestable_reason == "no claim given"
    for t in ["   ", "???", "42", "by by by 5%", None]:
        c = parse_claim(t, P["rct"])
        assert isinstance(c, Claim)


def test_columns_differ_and_did(P):
    c = parse_claim("The policy increased sales by 2 units", P["did"])
    assert (c.treatment_col, c.outcome_col) == ("policy", "sales")
    assert c.treatment_col != c.outcome_col
    assert c.size_unit == "absolute"


# ----------------------------------------------------------------- instrument + design hint
def test_instrument_collected_and_matched(P):
    c = parse_claim("Enrolling raises test score, using the lottery as an instrument", P["iv"])
    assert c.instrument_phrase == "lottery"
    assert c.instrument_col == "lottery" and c.instrument_score == 1.0
    assert c.candidates["instrument"][0] == ("lottery", 1.0) and len(c.candidates["instrument"]) <= 3
    assert (c.treatment_col, c.outcome_col) == ("enrolled", "test_score")
    assert (c.treatment_phrase, c.outcome_phrase) == ("enrolling", "test score")
    assert c.direction == "increase" and c.design_hint == "iv"
    assert c.design_hint_phrase == "using the lottery as an instrument"
    assert not c.needs_confirmation and c.testable
    assert any(n.startswith("The claim names an instrument") for n in c.notes)
    assert any("hint only" in n for n in c.notes)


@pytest.mark.parametrize("text", [
    "Using the lottery as an instrument, enrolling raises test score",
    "Enrolling raises test score, with the lottery as an instrumental variable",
    "Enrolling raises test score, with the lottery as an instrumental variable for enrollment",
    "Enrollment, instrumented by the lottery, raises test score",
    "Enrollment instrumented with the lottery raises test score",
    "Enrolling raises test score, with enrollment instrumented by the lottery",
    "Enrolling raises test score (lottery as an IV)",
    "IV: lottery. Enrolling raises test score",
    "Enrolling raises test score using the lottery instrument",
    "Instrumenting enrollment with the lottery, enrolling raises test score",
])
def test_instrument_wordings(P, text):
    c = parse_claim(text, P["iv"])
    assert c.instrument_phrase == "lottery" and c.instrument_col == "lottery"
    assert (c.treatment_col, c.outcome_col) == ("enrolled", "test_score")
    assert c.outcome_phrase == "test score" and c.population_phrase is None
    assert c.direction == "increase" and c.design_hint == "iv" and c.testable


def test_instrument_never_treatment_or_outcome(P):
    c = parse_claim("The lottery raises test score, using the lottery as an instrument", P["iv"])
    assert c.instrument_col == "lottery"
    assert c.treatment_col != "lottery" and c.outcome_col == "test_score"
    assert all(col != "lottery" for col, _s in c.candidates["treatment"] + c.candidates["outcome"])
    c = parse_claim("Enrolling affects the lottery, instrumented by the lottery", P["iv"])
    assert c.instrument_col == "lottery" and c.outcome_col != "lottery"


def test_instrument_without_column_does_not_ask(P):
    c = parse_claim("Enrolling raises test score, using rainfall as an instrument", P["iv"])
    assert c.instrument_phrase == "rainfall" and c.instrument_col is None and c.instrument_score == 0.0
    assert c.candidates["instrument"] == []
    assert (c.treatment_col, c.outcome_col) == ("enrolled", "test_score")
    assert not c.needs_confirmation and c.design_hint == "iv"
    assert any("no column matches" in n for n in c.notes)


def test_iv_hint_without_named_instrument(P):
    for text, phrase in [("Enrolling raises test score by 2SLS", "2SLS"),
                         ("IV estimates show that enrolling raises test score", "IV"),
                         ("Enrolling raises test score. The instrument is weak", "instrument"),
                         ("Enrolling raises test score using two-stage least squares", "two-stage least squares")]:
        c = parse_claim(text, P["iv"])
        assert c.design_hint == "iv" and c.design_hint_phrase == phrase
        assert c.instrument_phrase is None and c.instrument_col is None
        assert (c.treatment_col, c.outcome_col) == ("enrolled", "test_score")
        assert c.size is None            # the 2 in "2SLS" is not a size


@pytest.mark.parametrize("text", [
    "In a randomized trial, treatment increases score",
    "Treatment increases score in this RCT",
    "The randomised treatment raises score",
    "Treatment was randomly assigned and raises score",
    "Treatment raises score in an A/B test",
    "Treatment raises score in a randomized controlled trial",
    "In an experiment, treatment raises score",
])
def test_experiment_hint(P, text):
    c = parse_claim(text, P["rct"])
    assert c.design_hint == "experiment" and c.design_hint_phrase
    assert (c.treatment_col, c.outcome_col) == ("treated", "score")
    assert (c.treatment_phrase, c.outcome_phrase) == ("treatment", "score")
    assert not c.needs_confirmation
    assert any("randomized experiment design" in n and "hint only" in n for n in c.notes)


def test_design_clause_keeps_size_and_population(P):
    c = parse_claim("In a randomized trial, treatment increases score by 15% among women", P["rct"])
    assert c.design_hint == "experiment" and c.design_hint_phrase == "randomized trial"
    assert c.size == 15 and c.size_unit == "percent"
    assert c.population_filter == {"column": "female", "op": "==", "value": 1}
    c = parse_claim("Treatment raises score in the north in a randomised controlled trial", P["rct"])
    assert c.population_filter == {"column": "region=north", "op": "==", "value": 1}
    assert c.outcome_phrase == "score"


@pytest.mark.parametrize("text", [
    "The policy increased sales by 2 units (difference-in-differences)",
    "Using a diff-in-diff, the policy increased sales",
    "The policy (DiD) increased sales",
    "The policy increased sales in an event study",
    "With a staggered rollout, the policy increased sales",
    "With a staggered roll-out, the policy increased sales",
])
def test_did_hint(P, text):
    c = parse_claim(text, P["did"])
    assert c.design_hint == "did"
    assert (c.treatment_col, c.outcome_col) == ("policy", "sales")
    assert (c.treatment_phrase, c.outcome_phrase) == ("policy", "sales")


def test_did_before_after_compared_and_lowercase_did(P):
    c = parse_claim("Sales increased before and after the policy compared with stores without it", P["did"])
    assert c.design_hint == "did" and c.testable      # the comparison group is not another treatment
    c = parse_claim("What did the policy do: the policy increased sales", P["did"])
    assert c.design_hint is None and (c.treatment_col, c.outcome_col) == ("policy", "sales")


@pytest.mark.parametrize("text", [
    "The scholarship increases gpa_later, using a regression discontinuity design",
    "Scholarship increases gpa_later (RDD)",
    "In an RD design, the scholarship increases gpa_later",
    "The scholarship increases gpa_later at the cutoff",
    "The scholarship increases gpa_later for students just above the cutoff",
    "The scholarship increases gpa_later for those just below the cutoff",
    "Because of a threshold rule, the scholarship increases gpa_later",
])
def test_rd_hint(P, text):
    c = parse_claim(text, P["rd"])
    assert c.design_hint == "rd" and c.shape is None
    assert (c.treatment_col, c.outcome_col) == ("scholarship", "gpa_later")
    assert c.outcome_phrase == "gpa_later" and c.population_phrase is None


def test_threshold_shape_is_not_rd_hint(P):
    c = parse_claim("The scholarship increases gpa_later only above a threshold", P["rd"])
    assert c.shape == "threshold" and c.design_hint is None and c.design_hint_phrase is None
    c = parse_claim("Study hours improve grade, with a threshold effect", P["dose"])
    assert c.shape == "threshold" and c.design_hint is None


@pytest.mark.parametrize("text,phrase", [
    ("In observational data, the program raises earnings", "observational"),
    ("The program raises earnings in survey data", "survey data"),
    ("The program raises earnings without randomization", "without randomization"),
    ("The program raises earnings, although the program was not randomized", "not randomized"),
    ("In this non-randomized study, the program raises earnings", "non-randomized"),
    ("The nonrandomized program raises earnings", "nonrandomized"),
    ("The program wasn't randomised but raises earnings", "not randomised"),
])
def test_observational_hint_and_negation_wins(P, text, phrase):
    c = parse_claim(text, P["observational"])
    assert c.design_hint == "observational" and c.design_hint_phrase == phrase
    assert (c.treatment_col, c.outcome_col) == ("program", "earnings")
    assert c.direction == "increase" and not c.negated
    assert not any("randomized experiment design" in n for n in c.notes)


def test_no_hint_when_in_doubt(P):
    c = parse_claim("Treatment raises score in a quasi-experiment", P["rct"])
    assert c.design_hint is None and c.outcome_phrase == "score"
    c = parse_claim("The experiment raises score", P["rct"])      # the word is the subject: kept in the text
    assert c.design_hint == "experiment" and c.treatment_phrase == "experiment"


def test_plain_claims_get_no_instrument_or_hint(P):
    for key, text in [("rct", "Treatment increases score by 15% among women"),
                      ("observational", "The program raises earnings by at least 2 points"),
                      ("iv", "The lottery affects test score through enrollment"),
                      ("did", "The policy increased sales by 2 units"),
                      ("dose", "Study hours improve grade, with diminishing returns")]:
        c = parse_claim(text, P[key])
        assert c.instrument_phrase is None and c.instrument_col is None and c.instrument_score == 0.0
        assert c.design_hint is None and c.design_hint_phrase is None
        assert "instrument" not in c.candidates
        assert not any("instrument" in n or "hint" in n for n in c.notes)
    c = parse_claim("The lottery affects test score through enrollment", P["iv"])
    assert (c.treatment_col, c.outcome_col) == ("lottery", "test_score")     # unchanged without an instrument clause


def test_instrument_and_design_never_raise(P):
    for t in ["as an instrument", "IV:", "instrumented by", "using the as an instrument", "(RCT)", "DiD",
              "not randomized", "instrumenting with", "at the cutoff", "IV IV IV", "randomized"]:
        c = parse_claim(t, P["iv"])
        assert isinstance(c, Claim) and not any("could not be fully read" in n for n in c.notes)
