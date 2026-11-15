"""The model-led audit inside the web app's analysis job (services/model_audit.py, services/jobs.py).

A scripted FakeClient stands in for the model: no key, no network."""
from __future__ import annotations

import pytest

from causality import sim
from causality.orchestrator import contract, loop
from services import jobs, model_audit, store

ROLE_NULLS = {k: None for k in contract.ROLE_KEYS} | {"cutoff": None}


def roles(**kw):
    return {**ROLE_NULLS, **kw}


def script(label="Supported", excluded=()):
    r = roles(treatment="treated", outcome="score")
    return [
        [loop.tool_use("estimate_effect", {"design": "experiment", "roles": r, "covariates": None})],
        [loop.tool_use(contract.SUBMIT_VERDICT, {
            "reading": {"treatment": "treated", "outcome": "score", "design": "experiment",
                        "design_reason": "The description says assignment was by coin flip.", "roles": r,
                        "covariates": None, "excluded": list(excluded)},
            "verdict": {"label": label, "sentence": "Treatment raised the score [e2].",
                        "estimate_evidence_id": "e2", "claimed_direction": "increase",
                        "reasons": ["The 95% interval in [e2] is above zero."], "assumed": [],
                        "caveats": ["Applies to the people in this trial."], "judgments": []}})],
    ]


def LOOP(turns):
    return {"engine": "loop", "model": "fake", "client": loop.FakeClient(turns), "writer": None}


class NoSteps:
    deleted = False

    def __init__(self):
        self.seen = []

    def start(self, key):
        self.seen.append(key)

    def finish(self, error=None):
        pass


@pytest.fixture
def job(monkeypatch):
    df, truth = sim.rct(n=600)
    monkeypatch.setenv("CAUSAL_LLM", "fake")
    monkeypatch.setattr(store, "load_upload", lambda _id: df.to_csv(index=False).encode())
    monkeypatch.setattr(store, "save_frame", lambda *a, **k: None)
    a = {"id": "t1", "mode": "audit", "claim": "Treatment raised the score.", "filename": "trial.csv",
         "params": {"design": None, "roles": {}, "confirm": {}, "online_literature": False}}
    return a, df, truth


def test_run_and_declarations():
    df, _ = sim.rct(n=600)
    ex = [{"column": "region", "reason": "not a cause"}, {"column": "nope", "reason": "not in the table"}]
    audit = model_audit.run(df, "Treatment raised the score.", picked=LOOP(script(excluded=ex)))
    assert model_audit.usable(audit) and audit["verdict"]["label"] == contract.SUPPORTED and "state" not in audit
    design, r = model_audit.declarations(audit, df.columns)
    assert design == "experiment" and r["treatment"] == "treated" and r["outcome"] == "score" and r["randomized"] is True
    assert r["post_treatment"] == (["region"] if "region" in df.columns else [])


def test_job_uses_the_models_reading_and_verdict(job, monkeypatch):
    a, df, truth = job
    monkeypatch.setattr(model_audit, "pick_client", lambda: LOOP(script()))
    steps = NoSteps()
    result, status = jobs._pipeline(a, steps)
    assert status == "done" and steps.seen[:2] == ["read", "model"]
    rep = result["report"]
    assert rep["verdict_source"] == "model" and rep["verdict"]["label"] == "Supported"
    assert rep["verdict"]["sentence"].startswith("Treatment raised the score")
    assert any(x.startswith("Caveat:") for x in rep["verdict"]["reasons"])
    assert rep["design"]["kind"] == "experiment"                       # computed under the model's design
    lo, hi = rep["estimate"]["ci95"]
    assert lo < truth["effect"] < hi
    m = result["model_audit"]
    assert m["status"] == "done" and m["engine"] == "loop" and m["reading"]["treatment"] == "treated"
    assert m["evidence"][0]["lines"]
    assert "state" not in m and "numbers" not in m["evidence"][0]


def test_floor_downgrade_reaches_the_page(job, monkeypatch):
    a, _, _ = job
    monkeypatch.setattr(model_audit, "pick_client", lambda: LOOP(script(label="Not supported")))
    result, _ = jobs._pipeline(a, NoSteps())
    v = result["report"]["verdict"]
    assert v["label"] == "Inconclusive"                                # the interval is above zero, so code refused it
    assert any(x.startswith("Checked by code:") for x in v["reasons"])


def test_model_failure_falls_back_to_the_rule_based_analysis(job, monkeypatch):
    a, _, _ = job
    monkeypatch.setattr(model_audit, "pick_client", lambda: LOOP([RuntimeError("quota")]))
    result, status = jobs._pipeline(a, NoSteps())
    assert status in ("done", "needs_input")
    assert result["model_audit"]["status"] == "error" and "verdict_source" not in result["report"]


def test_no_model_or_user_declared_roles_means_no_model_step(job, monkeypatch):
    a, _, _ = job
    steps = NoSteps()
    result, _ = jobs._pipeline(a, steps)                               # conftest: CAUSAL_MODEL_AUDIT=off
    assert result["model_audit"] is None and "model" not in steps.seen
    called = []
    monkeypatch.setattr(model_audit, "pick_client", lambda: called.append(1) or LOOP(script()))
    a["params"]["roles"] = {"treatment": "treated", "outcome": "score"}
    steps = NoSteps()
    result, _ = jobs._pipeline(a, steps)
    assert result["model_audit"] is None and "model" not in steps.seen


def test_pick_client_respects_the_switches(monkeypatch):
    monkeypatch.setenv("CAUSAL_MODEL_AUDIT", "off")
    assert model_audit.pick_client() is None
    monkeypatch.setenv("CAUSAL_MODEL_AUDIT", "on")
    monkeypatch.setenv("CAUSAL_LLM", "fake")
    assert model_audit.pick_client() is None
    monkeypatch.delenv("CAUSAL_LLM")
    monkeypatch.setenv("CAUSAL_MODEL_AUDIT", "auto")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert model_audit.pick_client() is None                           # no key: auto never starts headless Claude
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    picked = model_audit.pick_client()
    assert picked["engine"] == "free" and picked["model"].startswith("gemini")       # Gemini runs the inverted design


# ----------------------------------------------------------------------------- the inverted design on Gemini
import json  # noqa: E402


def g_reply(calls):
    body = {"text": "", "tool_calls": [{"name": n, "input_json": json.dumps(a)} for n, a in calls]}
    return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
                 "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}


class Transport:
    def __init__(self, turns):
        self.turns, self.bodies = list(turns), []

    def __call__(self, url, body, key, timeout):
        self.bodies.append(body)
        return self.turns.pop(0)


def free_answer(verdict="supported", **over):
    a = {"verdict": verdict, "claimed_direction": "increase", "sentence": "Treatment raised the score.",
         "treatment": "treated", "outcome": "score", "design": "experiment",
         "design_reason": "No column predicts who was treated.", "unit": None, "time": None, "instrument": None,
         "running": None, "cutoff": None, "left_out": [{"column": "region", "reason": "not needed"}],
         "estimate": 2.0, "ci95_low": 1.5, "ci95_high": 2.5, "reasons": ["The difference is clear."],
         "assumed": ["Assignment was random."], "caveats": []}
    return {**a, **over}


def FREE(answer):
    t = Transport([g_reply([("layout_facts", {"treatment": "treated", "outcome": "score"})]),
                   g_reply([("write_answer", {"answer_json": json.dumps(answer)})])])
    return {"engine": "free", "model": "gemini-test", "transport": t, "sleep": lambda s: None}, t


@pytest.fixture
def gem(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("GEMINI_RPM", "6000")


def test_free_engine_drives_the_job(job, gem, monkeypatch):
    a, df, truth = job
    picked, t = FREE(free_answer())
    monkeypatch.setattr(model_audit, "pick_client", lambda: picked)
    result, status = jobs._pipeline(a, NoSteps())
    rep, m = result["report"], result["model_audit"]
    assert status == "done" and rep["verdict_source"] == "model" and rep["verdict"]["label"] == "Supported"
    assert rep["design"]["kind"] == "experiment" and rep["verdict"]["caveat"] == "assignment was random"
    assert m["engine"] == "free" and m["evidence"][0]["tool"] == "layout_facts" and m["evidence"][0]["lines"]
    assert m["reading"]["excluded"][0]["column"] == "region" and not m["floor"]["changed"]
    system = t.bodies[0]["systemInstruction"]["parts"][0]["text"]
    assert "Treatment raised the score." in system and "layout_facts" in system
    assert "Facts about data.csv" in t.bodies[0]["contents"][0]["parts"][0]["text"]


def test_free_engine_cannot_claim_more_than_the_interval(job, gem, monkeypatch):
    a, _, _ = job
    picked, _ = FREE(free_answer(verdict="contradicted", sentence="Treatment lowered the score."))
    monkeypatch.setattr(model_audit, "pick_client", lambda: picked)
    result, _ = jobs._pipeline(a, NoSteps())
    v, m = result["report"]["verdict"], result["model_audit"]
    assert v["label"] == "Inconclusive" and m["floor"]["changed"]
    assert m["draft_verdict"]["label"] == contract.NOT_SUPPORTED          # what the model said is kept
    assert any("Checked by code" in x for x in v["reasons"])


def test_free_engine_bad_columns_fall_back(job, gem, monkeypatch):
    a, _, _ = job
    picked, _ = FREE(free_answer(treatment="no_such_column"))
    monkeypatch.setattr(model_audit, "pick_client", lambda: picked)
    result, status = jobs._pipeline(a, NoSteps())
    assert "verdict_source" not in result["report"] and result["model_audit"]["status"] == "done"
    assert status in ("done", "needs_input")
