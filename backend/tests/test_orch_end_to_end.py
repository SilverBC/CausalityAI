"""End to end: the REAL evidence.py and floor.py driven by loop.py with a scripted FakeClient.

No stubs, no key, no network. Each test runs a complete audit on a simulation from
causality.sim where the truth is known, and checks the report the way the page and the
API read it. One test goes through the HTTP endpoints with app.state.audit_client set.
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import app as app_module
from causality import sim
from causality.orchestrator import contract, evidence, floor, loop

ROLE_NULLS = {k: None for k in contract.ROLE_KEYS} | {"cutoff": None}


# ------------------------------------------------------------------------------ script helpers

def roles(**kw) -> dict:
    """roles as the strict submit_verdict schema sends them: every key present, unused ones null."""
    return {**ROLE_NULLS, **kw}


def submit(treatment, outcome, design, label, direction, est_id, judgments=(), assumed=(), covariates=None,
           excluded=(), reasons=None, sentence=None, **role_kw):
    return loop.tool_use(contract.SUBMIT_VERDICT, {
        "reading": {"treatment": treatment, "outcome": outcome, "design": design,
                    "design_reason": "Set by the scripted model for this test.",
                    "roles": roles(treatment=treatment, outcome=outcome, **role_kw),
                    "covariates": covariates, "excluded": list(excluded)},
        "verdict": {"label": label,
                    "sentence": sentence or f"The estimate [{est_id}] backs the claim.",
                    "estimate_evidence_id": est_id, "claimed_direction": direction,
                    "reasons": reasons if reasons is not None else [f"The 95% interval in [{est_id}] is on one side of zero."],
                    "assumed": list(assumed), "caveats": [],
                    "judgments": [{"evidence_id": i, "reading": r, "why": "scripted"} for i, r in judgments]},
    })


def ev(report, eid) -> dict:
    return next(e for e in report["evidence"] if e["id"] == eid)


def untestable(report) -> list[str]:
    out = []
    for e in report["evidence"]:
        for s in (e["numbers"] or {}).get("untestable") or []:
            if s not in out:
                out.append(s)
    return out


def assert_plain_report(report):
    """Contract shape, and plain JSON all the way down (what the API sends)."""
    assert set(contract.empty_report()) <= set(report)
    json.dumps(report, allow_nan=False)
    for i, e in enumerate(report["evidence"], 1):
        assert e["id"] == f"e{i}"
        assert set(contract.evidence_item("x", "y", {})) <= set(e)


def assert_done(report):
    assert report["status"] == "done", report.get("error")
    assert_plain_report(report)
    v = report["verdict"]
    assert v["label"] in contract.LABELS
    # rule 1 always holds after the floor: every untestable assumption is listed as assumed
    for s in untestable(report):
        assert s in v["assumed"]
    for j in v["judgments"]:
        if ev(report, j["evidence_id"])["tool"] == "sensitivity":
            assert j["reading"] != "supports"
    assert report["floor"]["changed"] == bool(report["floor"]["violations"])


def rules(report) -> set[str]:
    return {x["rule"] for x in report["floor"]["violations"]}


def ci95(item) -> list:
    return item["numbers"]["ci"]["95"]


# ------------------------------------------------------------------------------ the audits

def test_randomized_trial_supported_survives_the_floor():
    df, truth = sim.rct()
    cov = ["age", "female", "income"]
    r = roles(treatment="treated", outcome="score")
    fc = loop.FakeClient([
        [loop.thinking("Randomized per the description."), loop.text("Checking the design."),
         loop.tool_use("design_tests", {"design": "experiment", "roles": r, "covariates": cov}),
         loop.tool_use("general_checks", {"treatment": "treated", "outcome": "score", "covariates": cov})],
        [loop.tool_use("estimate_effect", {"design": "experiment", "roles": r, "covariates": cov})],
        [loop.tool_use("sensitivity", {"design": "experiment", "roles": r, "covariates": cov})],
        [loop.text("Submitting."),
         submit("treated", "score", "experiment", contract.SUPPORTED, "increase", "e4", covariates=cov,
                excluded=[{"column": "region", "reason": "kept out to keep the example small"}],
                judgments=[("e2", "supports"), ("e3", "supports"), ("e4", "supports"), ("e5", "concern")])],
    ])
    rep = loop.run_audit(df, "Treatment raised the score.", "Participants were assigned by coin flip.",
                         client=fc)
    assert_done(rep)
    assert [e["tool"] for e in rep["evidence"]] == ["table_facts", "design_tests", "general_checks",
                                                     "estimate_effect", "sensitivity"]
    assert all(e["ok"] for e in rep["evidence"]), [e["error"] for e in rep["evidence"]]
    est = ev(rep, "e4")
    lo, hi = ci95(est)
    assert lo < truth["effect"] < hi and lo > 0
    assert rep["verdict"]["label"] == contract.SUPPORTED
    assert rep["draft_verdict"]["label"] == contract.SUPPORTED
    assert rep["verdict"]["sentence"] == rep["draft_verdict"]["sentence"]
    # the label survives; only rule 1 may have added the untestable assumptions
    assert rules(rep) <= {floor.RULE_ASSUMED}
    assert rep["reading"]["roles"] == {"treatment": "treated", "outcome": "score"}
    assert rep["reading"]["excluded"][0]["column"] == "region"
    # the tool results the model saw start with the evidence id
    sent = fc.calls[1]["messages"][-1]["content"]
    assert [b["content"].split(" ")[0] for b in sent] == ["[e2]", "[e3]"]
    assert rep["usage"]["turns"] == 4 and rep["state"] is not None      # kept so a follow-up can continue


def test_null_trial_wrongly_supported_is_moved_to_inconclusive():
    df, truth = sim.null_rct()
    r = roles(treatment="treated", outcome="score")
    fc = loop.FakeClient([
        [loop.tool_use("estimate_effect", {"design": "experiment", "roles": r, "covariates": None})],
        [submit("treated", "score", "experiment", contract.SUPPORTED, "increase", "e2",
                sentence="Treatment clearly raised the score [e2].", judgments=[("e2", "supports")])],
    ])
    rep = loop.run_audit(df, "Treatment raised the score.", "A randomized trial.", client=fc)
    assert_done(rep)
    lo, hi = ci95(ev(rep, "e2"))
    assert lo < 0 < hi
    assert rep["draft_verdict"]["label"] == contract.SUPPORTED
    assert rep["draft_verdict"]["sentence"] == "Treatment clearly raised the score [e2]."
    assert rep["verdict"]["label"] == contract.INCONCLUSIVE
    assert rep["verdict"]["sentence"] != rep["draft_verdict"]["sentence"]
    assert floor.RULE_NO_MORE in rules(rep) and rep["floor"]["changed"]


def test_staggered_difference_in_differences():
    df, truth = sim.did_staggered()
    r = roles(treatment="policy", outcome="sales", unit="state", time="year")
    fc = loop.FakeClient([
        [loop.tool_use("structure_facts", {"treatment": "policy", "outcome": "sales"})],
        [loop.tool_use("design_tests", {"design": "did", "roles": r, "covariates": []})],
        [loop.tool_use("estimate_effect", {"design": "did", "roles": r, "covariates": []})],
        [submit("policy", "sales", "did", contract.SUPPORTED, "increase", "e4", covariates=[],
                excluded=[{"column": "noise_cov", "reason": "not related to adoption"}],
                judgments=[("e3", "supports"), ("e4", "supports")], unit="state", time="year")],
    ])
    rep = loop.run_audit(df, "The policy increased sales.",
                         "States adopted the policy in different years; some never did.", client=fc)
    assert_done(rep)
    assert all(e["ok"] for e in rep["evidence"]), [e["error"] for e in rep["evidence"]]
    tests = {t["key"]: t for t in ev(rep, "e3")["numbers"]["tests"]}
    assert tests["design_pretrends"]["p_value"] is not None
    lo, hi = ci95(ev(rep, "e4"))
    assert lo < truth["effect"] < hi
    assert rep["verdict"]["label"] == contract.SUPPORTED
    assert rep["reading"]["roles"] == {"treatment": "policy", "outcome": "sales", "unit": "state", "time": "year"}


def test_instrument():
    df, truth = sim.iv()
    r = roles(treatment="enrolled", outcome="test_score", instrument="lottery")
    fc = loop.FakeClient([
        [loop.tool_use("structure_facts", {"treatment": "enrolled", "outcome": "test_score",
                                            "instruments": ["lottery"], "covariates": ["x"]})],
        [loop.tool_use("design_tests", {"design": "iv", "roles": r, "covariates": ["x"]}),
         loop.tool_use("estimate_effect", {"design": "iv", "roles": r, "covariates": ["x"]})],
        [loop.tool_use("sensitivity", {"design": "iv", "roles": r, "covariates": ["x"]})],
        [submit("enrolled", "test_score", "iv", contract.SUPPORTED, "increase", "e4", covariates=["x"],
                judgments=[("e3", "supports"), ("e4", "supports"), ("e5", "supports")],
                instrument="lottery")],
    ])
    rep = loop.run_audit(df, "Enrolling raised test scores.",
                         "Places were offered by lottery; people chose whether to enroll.", client=fc)
    assert_done(rep)
    first_stage = {t["key"]: t for t in ev(rep, "e3")["numbers"]["tests"]}["design_first_stage"]
    assert first_stage["statistic"] > 10
    lo, hi = ci95(ev(rep, "e4"))
    assert lo < truth["effect"] < hi
    expected = contract.SUPPORTED if lo > 0 else contract.INCONCLUSIVE
    assert rep["verdict"]["label"] == expected
    # the sensitivity item read as "supports" is changed by rule 1, and the exclusion restriction is assumed
    j5 = next(j for j in rep["verdict"]["judgments"] if j["evidence_id"] == "e5")
    assert j5["reading"] == "not informative"
    assert floor.RULE_ASSUMED in rules(rep)
    assert len(rep["verdict"]["assumed"]) >= 1


def test_cutoff_design():
    df, truth = sim.rd()
    r = roles(treatment="scholarship", outcome="gpa_later", running="exam_score", cutoff=50.0)
    fc = loop.FakeClient([
        [loop.tool_use("structure_facts", {"treatment": "scholarship", "outcome": "gpa_later"})],
        [loop.tool_use("design_tests", {"design": "rd", "roles": r, "covariates": ["female"]}),
         loop.tool_use("estimate_effect", {"design": "rd", "roles": r, "covariates": ["female"]})],
        [submit("scholarship", "gpa_later", "rd", contract.SUPPORTED, "increase", "e4", covariates=["female"],
                judgments=[("e3", "supports"), ("e4", "supports")], running="exam_score", cutoff=50.0)],
    ])
    rep = loop.run_audit(df, "The scholarship raised later GPA.",
                         "Scholarships went to everyone scoring 50 or more on the exam.", client=fc)
    assert_done(rep)
    est = ev(rep, "e4")
    assert est["ok"] and abs(est["numbers"]["cutoff_used"] - truth["cutoff"]) < 1e-9
    lo, hi = ci95(est)
    assert lo < truth["effect"] < hi
    assert rep["verdict"]["label"] == contract.SUPPORTED
    assert rep["reading"]["roles"]["cutoff"] == 50.0


def test_observational_asks_how_treatment_was_assigned_then_resumes():
    df, truth = sim.observational()
    cov = ["x1", "x2", "member"]
    r = roles(treatment="program", outcome="earnings")
    fc = loop.FakeClient([
        [loop.text("The description does not say how people got the program."),
         loop.tool_use("structure_facts", {"treatment": "program", "outcome": "earnings"}),
         loop.tool_use(contract.ASK_USER, {"questions": [
             {"id": "assign", "question": "How did people end up in the program?",
              "options": ["By lottery", "They chose it, or staff chose them", "Not sure"]},
             {"id": "timing", "question": "Were x1 and x2 measured before the program?",
              "options": ["Yes", "No", "Not sure"]}]})],
        [loop.tool_use("estimate_effect", {"design": "observational", "roles": r, "covariates": cov}),
         loop.tool_use("sensitivity", {"design": "observational", "roles": r, "covariates": cov})],
        [submit("program", "earnings", "observational", contract.SUPPORTED, "increase", "e3", covariates=cov,
                judgments=[("e3", "supports"), ("e4", "concern")])],
    ])
    first = loop.run_audit(df, "The program raised earnings.", "Records from a job program.", client=fc)
    assert first["status"] == "needs_input", first.get("error")
    assert_plain_report(first)
    assert [q["id"] for q in first["questions"]] == ["assign", "timing"]
    assert [e["tool"] for e in first["evidence"]] == ["table_facts", "structure_facts"]
    # the state survives a trip through JSON (the API keeps it server side; the CLI may print it)
    state = json.loads(json.dumps(first["state"]))

    rep = loop.resume(df, state, {"assign": "They chose it, or staff chose them"}, client=fc)
    assert_done(rep)
    # the held structure_facts result and both answers went back in one user turn
    sent = fc.calls[1]["messages"][-1]["content"]
    assert sent[0]["content"].startswith("[e2]")
    answers_text = " ".join(b["content"] for b in sent[1:])
    assert "They chose it, or staff chose them" in answers_text and "(the user did not answer)" in answers_text
    lo, hi = ci95(ev(rep, "e3"))
    assert lo < truth["effect"] < hi and lo > 0
    assert rep["verdict"]["label"] == contract.SUPPORTED
    # observational: no hidden confounding is assumed, never tested
    assert untestable(rep) and all(s in rep["verdict"]["assumed"] for s in untestable(rep))
    assert rep["usage"]["turns"] == 3


def test_estimate_for_another_design_is_not_a_real_interval():
    """The model estimates as an experiment but reads the study as observational."""
    df, _ = sim.rct()
    r = roles(treatment="treated", outcome="score")
    fc = loop.FakeClient([
        [loop.tool_use("estimate_effect", {"design": "experiment", "roles": r})],
        [submit("treated", "score", "observational", contract.SUPPORTED, "increase", "e2")],
    ])
    rep = loop.run_audit(df, "Treatment raised the score.", "", client=fc)
    assert_done(rep)
    assert rep["verdict"]["label"] == contract.INCONCLUSIVE
    assert floor.RULE_INTERVAL in rules(rep)


def test_bad_tool_arguments_come_back_as_an_error_the_model_can_fix():
    df, _ = sim.rct()
    fc = loop.FakeClient([
        [loop.tool_use("estimate_effect", {"design": "rct", "roles": {"treatment": "Treated", "outcome": "score"}})],
        [loop.tool_use("estimate_effect", {"design": "experiment",
                                           "roles": roles(treatment="treated", outcome="score")})],
        [submit("treated", "score", "experiment", contract.SUPPORTED, "increase", "e3")],
    ])
    rep = loop.run_audit(df, "Treatment raised the score.", "Randomized.", client=fc)
    assert_done(rep)
    bad = ev(rep, "e2")
    assert bad["ok"] is False and bad["error"]
    sent = fc.calls[1]["messages"][-1]["content"][0]
    assert sent.get("is_error") is True and sent["content"].startswith("[e2]")
    assert rep["verdict"]["label"] == contract.SUPPORTED


# ------------------------------------------------------------------------------ through HTTP

@pytest.fixture
def http(monkeypatch):
    with app_module._audit_lock:
        app_module._audit_sessions.clear()
    yield TestClient(app_module.app)
    with app_module._audit_lock:
        app_module._audit_sessions.clear()


def test_http_audit_with_a_question_then_answer(http, monkeypatch):
    df, truth = sim.rct()
    r = roles(treatment="treated", outcome="score")
    fc = loop.FakeClient([
        [loop.tool_use(contract.ASK_USER, {"questions": [
            {"id": "q1", "question": "How was treatment assigned?", "options": ["At random", "By choice"]}]})],
        [loop.tool_use("estimate_effect", {"design": "experiment", "roles": r,
                                           "covariates": ["age", "female", "income"]})],
        [submit("treated", "score", "experiment", contract.SUPPORTED, "increase", "e2",
                covariates=["age", "female", "income"], judgments=[("e2", "supports")])],
    ])
    monkeypatch.setattr(app_module.app.state, "audit_client", fc, raising=False)
    body = df.to_csv(index=False).encode()
    resp = http.post("/api/audit", files={"file": ("trial.csv", body, "text/csv")},
                     data={"claim": "Treatment raised the score.", "context": ""})
    assert resp.status_code == 200, resp.text
    rep = resp.json()
    assert rep["status"] == "needs_input" and rep["session_id"]
    assert "state" not in rep
    assert rep["questions"][0]["id"] == "q1"

    resp = http.post(f"/api/audit/{rep['session_id']}/answer", json={"answers": {"q1": "At random"}})
    assert resp.status_code == 200, resp.text
    rep2 = resp.json()
    assert rep2["status"] == "done", rep2.get("error")
    assert rep2["session_id"] == rep["session_id"] and "state" not in rep2
    assert rep2["verdict"]["label"] == contract.SUPPORTED
    lo, hi = rep2["evidence"][1]["numbers"]["ci"]["95"]
    assert lo < truth["effect"] < hi
    # the session is no longer waiting for answers
    again = http.post(f"/api/audit/{rep['session_id']}/answer", json={"answers": {"q1": "At random"}})
    assert again.status_code == 409


def test_http_audit_without_a_key_is_a_clean_error_report(http, monkeypatch):
    """No injected client, no API key and no Claude Code to fall back on: a status "error" report, not a 500."""
    from causality.orchestrator import headless
    monkeypatch.delattr(app_module.app.state, "audit_client", raising=False)
    for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_BIN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(headless, "find_claude", lambda: None)
    df, _ = sim.rct(n=300)
    resp = http.post("/api/audit", files={"file": ("trial.csv", df.to_csv(index=False).encode(), "text/csv")},
                     data={"claim": "Treatment raised the score.", "context": ""})
    assert resp.status_code == 200, resp.text
    rep = resp.json()
    assert rep["status"] == "error" and rep["error"]
    assert rep["evidence"] and rep["evidence"][0]["tool"] == "table_facts"
