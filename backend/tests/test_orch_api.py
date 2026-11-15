"""Part D: the audit API and the command line, with loop.py replaced by a stub (no key, no network)."""
from __future__ import annotations

import sys
import types

import pytest
from fastapi.testclient import TestClient

import app as app_module
import causality.orchestrator as orch_pkg
from causality import sim
from causality.orchestrator import contract


class StubLoop(types.ModuleType):
    """Stands in for causality.orchestrator.loop: first call asks how treatment was assigned,
    resume finishes with a contract-shaped report."""

    def __init__(self, ask=True):
        super().__init__("causality.orchestrator.loop")
        self.ask = ask
        self.calls = []

    def _done(self, claim, context):
        r = contract.empty_report(claim, context)
        r["reading"].update({"treatment": "treated", "outcome": "y", "design": "experiment",
                             "design_reason": "The user said assignment was by lottery."})
        r["evidence"] = [contract.evidence_item("e1", "estimate_effect", {"design": "experiment"},
                                                lines=["The difference in means is 2.0 (95% interval 1.5 to 2.5)."],
                                                numbers={"estimate": 2.0, "ci": {"95": [1.5, 2.5]}})]
        v = contract.empty_verdict()
        v.update({"label": contract.SUPPORTED, "sentence": "The interval is above zero.",
                  "estimate_evidence_id": "e1", "claimed_direction": "increase"})
        r["draft_verdict"] = dict(v)
        r["verdict"] = v
        r["usage"]["turns"] = 3
        r["state"] = {"done": True, "claim": claim, "context": context}      # kept for follow-ups
        return r

    def run_audit(self, df, claim, context="", client=None, **kw):
        self.calls.append(("run_audit", len(df), claim, context, client))
        if not self.ask:
            return self._done(claim, context)
        r = contract.empty_report(claim, context)
        r["status"] = "needs_input"
        r["questions"] = [{"id": "q1", "question": "How was treatment assigned?",
                           "options": ["By lottery", "People chose it", "Not sure"]}]
        r["state"] = {"messages": ["opaque"], "claim": claim, "context": context}
        return r

    def resume(self, df, state, answers, client=None, **kw):
        self.calls.append(("resume", len(df), state, dict(answers), client))
        assert state and state["messages"] == ["opaque"]
        return self._done(state["claim"], state["context"])


@pytest.fixture
def stub(monkeypatch):
    s = StubLoop()
    monkeypatch.setitem(sys.modules, "causality.orchestrator.loop", s)
    monkeypatch.setattr(orch_pkg, "loop", s, raising=False)
    monkeypatch.setattr(app_module.app.state, "audit_client", "fake-client", raising=False)
    with app_module._audit_lock:
        app_module._audit_sessions.clear()
    yield s
    with app_module._audit_lock:
        app_module._audit_sessions.clear()


@pytest.fixture(scope="module")
def csv_bytes():
    df, _ = sim.ALL["rct"]()
    return df.to_csv(index=False).encode()


@pytest.fixture
def client():
    return TestClient(app_module.app)


def _post(client, csv_bytes, claim="treatment raises y", context="a trial", name="data.csv"):
    return client.post("/api/audit", files={"file": (name, csv_bytes, "text/csv")},
                       data={"claim": claim, "context": context})


def test_health_still_works(client):
    r = client.get("/api/health")
    assert r.status_code == 200 and isinstance(r.json(), dict)


def test_sample_route_still_works(client):
    names = client.get("/api/samples")
    assert names.status_code == 200 and names.json()


def test_needs_input_then_answer_round_trip(client, stub, csv_bytes):
    r = _post(client, csv_bytes)
    assert r.status_code == 200
    rep = r.json()
    assert rep["status"] == "needs_input"
    assert rep["questions"][0]["id"] == "q1"
    assert "state" not in rep
    sid = rep["session_id"]
    assert isinstance(sid, str) and sid
    assert stub.calls[0][0] == "run_audit"
    assert stub.calls[0][2] == "treatment raises y" and stub.calls[0][3] == "a trial"
    assert stub.calls[0][4] == "fake-client"

    r2 = client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "By lottery"}})
    assert r2.status_code == 200
    rep2 = r2.json()
    assert rep2["status"] == "done"
    assert rep2["session_id"] == sid
    assert "state" not in rep2
    assert rep2["verdict"]["label"] == contract.SUPPORTED
    assert rep2["evidence"][0]["id"] == "e1"
    name, n_rows, state, answers, cl = stub.calls[1]
    assert name == "resume" and answers == {"q1": "By lottery"} and cl == "fake-client"
    assert n_rows == stub.calls[0][1]

    # The audit is finished: answering again is refused in plain words.
    r3 = client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "By lottery"}})
    assert r3.status_code == 409 and "not waiting" in r3.json()["error"]


def test_done_on_first_call_keeps_shape(client, stub, csv_bytes):
    stub.ask = False
    rep = _post(client, csv_bytes).json()
    assert rep["status"] == "done" and rep["session_id"]
    assert set(contract.empty_report()) - {"state"} <= set(rep)
    assert "state" not in rep


def test_unknown_session_is_404(client, stub):
    r = client.post("/api/audit/nope/answer", json={"answers": {"q1": "x"}})
    assert r.status_code == 404
    assert "not found" in r.json()["error"]


def test_bad_answers_body(client, stub, csv_bytes):
    sid = _post(client, csv_bytes).json()["session_id"]
    r = client.post(f"/api/audit/{sid}/answer", json={"answers": ["q1"]})
    assert r.status_code == 400 and "answers" in r.json()["error"]


def test_bad_file(client, stub):
    r = client.post("/api/audit", files={"file": ("data.csv", b"", "text/csv")}, data={"claim": "x raises y"})
    assert r.status_code == 400 and "empty" in r.json()["error"]
    r = client.post("/api/audit", files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
                    data={"claim": "x raises y"})
    assert r.status_code == 400 and r.json()["error"]
    assert stub.calls == []


def test_missing_claim(client, stub, csv_bytes):
    r = _post(client, csv_bytes, claim="  ")
    assert r.status_code == 400 and "claim" in r.json()["error"]


def test_loop_exception_is_plain_500(client, stub, csv_bytes, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("model went away")
    monkeypatch.setattr(stub, "run_audit", boom)
    r = _post(client, csv_bytes)
    assert r.status_code == 500
    assert "Nothing was concluded" in r.json()["error"]


def test_session_cap_drops_oldest(client, stub, csv_bytes, monkeypatch):
    monkeypatch.setattr(app_module, "AUDIT_MAX_SESSIONS", 3)
    ids = [_post(client, csv_bytes).json()["session_id"] for _ in range(4)]
    assert len(set(ids)) == 4
    assert len(app_module._audit_sessions) == 3
    assert client.post(f"/api/audit/{ids[0]}/answer", json={"answers": {"q1": "x"}}).status_code == 404
    assert client.post(f"/api/audit/{ids[3]}/answer", json={"answers": {"q1": "x"}}).status_code == 200


def test_default_cap_is_50():
    assert app_module.AUDIT_MAX_SESSIONS == 50


def test_client_none_when_not_injected_and_a_key_is_set(client, stub, csv_bytes, monkeypatch):
    monkeypatch.delattr(app_module.app.state, "audit_client", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    _post(client, csv_bytes)
    assert stub.calls[0][4] is None


def test_headless_client_when_no_key(client, stub, csv_bytes, monkeypatch):
    from causality.orchestrator import headless
    monkeypatch.delattr(app_module.app.state, "audit_client", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _post(client, csv_bytes)
    assert isinstance(stub.calls[0][4], headless.HeadlessClient)


# ------------------------------------------------------------------ follow-up

def _followup_stub(stub):
    def followup(df, state, question, client=None, **kw):
        stub.calls.append(("followup", len(df), state, question, client))
        r = stub._done(state["claim"], state["context"])
        r["followups"] = list(state.get("followups", [])) + [{"question": question}]
        r["state"] = {**state, "followups": r["followups"]}
        return r
    stub.followup = followup


def test_followup_after_verdict(client, stub, csv_bytes):
    _followup_stub(stub)
    sid = _post(client, csv_bytes).json()["session_id"]
    # Not finished yet: a follow-up is refused until the questions are answered.
    r0 = client.post(f"/api/audit/{sid}/followup", json={"question": "Why?"})
    assert r0.status_code == 409 and "Answer its questions first" in r0.json()["error"]
    assert client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "By lottery"}}).json()["status"] == "done"
    r = client.post(f"/api/audit/{sid}/followup", json={"question": "  What if people chose?  "})
    assert r.status_code == 200
    rep = r.json()
    assert rep["status"] == "done" and rep["session_id"] == sid and "state" not in rep
    assert rep["followups"] == [{"question": "What if people chose?"}]
    name, _, state, question, cl = stub.calls[-1]
    assert name == "followup" and question == "What if people chose?" and cl == "fake-client"
    assert state["done"] is True
    # a second follow-up continues from the state the first one returned
    rep2 = client.post(f"/api/audit/{sid}/followup", json={"question": "And by region?"}).json()
    assert [f["question"] for f in rep2["followups"]] == ["What if people chose?", "And by region?"]
    # answering is refused once the audit is done
    assert client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "x"}}).status_code == 409


def test_followup_bad_body_and_unknown_session(client, stub, csv_bytes):
    _followup_stub(stub)
    stub.ask = False
    sid = _post(client, csv_bytes).json()["session_id"]
    assert client.post(f"/api/audit/{sid}/followup", json={"question": "  "}).status_code == 400
    assert client.post(f"/api/audit/{sid}/followup", json={"q": "x"}).status_code == 400
    assert client.post("/api/audit/nope/followup", json={"question": "x"}).status_code == 404


def test_failed_followup_keeps_the_finished_state(client, stub, csv_bytes, monkeypatch):
    _followup_stub(stub)
    real = stub.followup
    calls = {"n": 0}

    def flaky(df, state, question, client=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            r = contract.empty_report(state["claim"], state["context"])
            r["status"], r["error"] = "error", "The call to the Claude API timed out. Try again."
            return r
        return real(df, state, question, client=client, **kw)

    monkeypatch.setattr(stub, "followup", flaky)
    stub.ask = False
    sid = _post(client, csv_bytes).json()["session_id"]
    assert client.post(f"/api/audit/{sid}/followup", json={"question": "Why?"}).json()["status"] == "error"
    r = client.post(f"/api/audit/{sid}/followup", json={"question": "Why?"})
    assert r.status_code == 200 and r.json()["status"] == "done"


def test_error_on_resume_keeps_the_paused_state(client, stub, csv_bytes, monkeypatch):
    """A temporary API error on resume must leave the session answerable, so the same answers can be resent."""
    real = stub.resume
    calls = {"n": 0}

    def flaky(df, state, answers, client=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            r = contract.empty_report(state["claim"], state["context"])
            r["status"] = "error"
            r["error"] = "The Claude API is rate limiting this key. Wait a minute and try again."
            return r                                   # like loop._error: no 'state' key
        return real(df, state, answers, client=client, **kw)

    monkeypatch.setattr(stub, "resume", flaky)
    sid = _post(client, csv_bytes).json()["session_id"]
    r1 = client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "By lottery"}})
    assert r1.status_code == 200 and r1.json()["status"] == "error"
    r2 = client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "By lottery"}})
    assert r2.status_code == 200 and r2.json()["status"] == "done"
    r3 = client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "By lottery"}})
    assert r3.status_code == 409


def test_exception_on_resume_keeps_the_paused_state(client, stub, csv_bytes, monkeypatch):
    real = stub.resume
    calls = {"n": 0}

    def boom_once(df, state, answers, client=None, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("connection reset")
        return real(df, state, answers, client=client, **kw)

    monkeypatch.setattr(stub, "resume", boom_once)
    sid = _post(client, csv_bytes).json()["session_id"]
    assert client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "x"}}).status_code == 500
    assert client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "x"}}).json()["status"] == "done"


def test_second_answer_while_first_runs_is_refused(client, stub, csv_bytes, monkeypatch):
    """Two answers for one session at once: the second must not start a second model run."""
    real = stub.resume
    inner = {}

    def slow(df, state, answers, client=None, **kw):
        # A second request arrives while this resume is still running.
        inner["resp"] = TestClient(app_module.app).post(f"/api/audit/{inner['sid']}/answer",
                                                        json={"answers": {"q1": "again"}})
        return real(df, state, answers, client=client, **kw)

    monkeypatch.setattr(stub, "resume", slow)
    sid = _post(client, csv_bytes).json()["session_id"]
    inner["sid"] = sid
    r = client.post(f"/api/audit/{sid}/answer", json={"answers": {"q1": "By lottery"}})
    assert r.status_code == 200 and r.json()["status"] == "done"
    assert inner["resp"].status_code == 409 and "already" in inner["resp"].json()["error"]
    assert [c[0] for c in stub.calls].count("resume") == 1


def test_table_is_parsed_off_the_event_loop(client, stub, csv_bytes, monkeypatch):
    import asyncio
    seen = {}
    real = app_module.data.load_table

    def spy(content, name):
        try:
            asyncio.get_running_loop()
            seen["on_loop"] = True
        except RuntimeError:
            seen["on_loop"] = False
        return real(content, name)

    monkeypatch.setattr(app_module.data, "load_table", spy)
    r = _post(client, csv_bytes)
    assert r.status_code == 200
    assert seen == {"on_loop": False}


# ------------------------------------------------------------------ command line

@pytest.fixture
def csv_path(tmp_path, csv_bytes):
    p = tmp_path / "data.csv"
    p.write_bytes(csv_bytes)
    return str(p)


def test_cli_prints_questions_and_exits_3(stub, csv_path, capsys, monkeypatch):
    import audit_cli
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    code = audit_cli.main([csv_path, "--claim", "treatment raises y"])
    out = capsys.readouterr().out
    assert code == 3
    assert "[q1] How was treatment assigned?" in out and "--answer" in out


def test_cli_with_answer_finishes(stub, csv_path, capsys, monkeypatch, tmp_path):
    import audit_cli
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    ctx = tmp_path / "ctx.txt"
    ctx.write_text("a lottery decided who was treated", encoding="utf-8")
    code = audit_cli.main([csv_path, "--claim", "treatment raises y", "--context-file", str(ctx),
                           "--answer", "q1=By lottery"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Verdict: Supported" in out and "e1 estimate_effect:" in out
    assert stub.calls[0][3] == "a lottery decided who was treated"
    assert stub.calls[1][3] == {"q1": "By lottery"}


def test_cli_asks_in_a_terminal(stub, csv_path, capsys, monkeypatch):
    import audit_cli
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt="": "1")
    code = audit_cli.main([csv_path, "--claim", "treatment raises y"])
    assert code == 0
    assert stub.calls[1][3] == {"q1": "By lottery"}


def test_cli_full_json(stub, csv_path, capsys, monkeypatch):
    import json
    import audit_cli
    stub.ask = False
    code = audit_cli.main([csv_path, "--claim", "treatment raises y", "--full"])
    rep = json.loads(capsys.readouterr().out)
    assert code == 0 and rep["verdict"]["label"] == contract.SUPPORTED


def test_cli_auto_answer_followup_json_out(stub, csv_path, capsys, monkeypatch, tmp_path):
    import json
    import audit_cli
    from causality.orchestrator import headless
    _followup_stub(stub)
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(headless, "HeadlessClient", lambda *a, **k: "headless-client")
    out_file = tmp_path / "report.json"
    code = audit_cli.main([csv_path, "--claim", "treatment raises y", "--headless", "--auto-answer", "Not sure",
                           "--followup", "Why?", "--followup", "And for women?", "--json-out", str(out_file)])
    out = capsys.readouterr().out
    assert code == 0
    assert stub.calls[1][0] == "resume" and stub.calls[1][3] == {"q1": "Not sure"}
    assert [c[0] for c in stub.calls] == ["run_audit", "resume", "followup", "followup"]
    assert stub.calls[0][4] == "headless-client" and stub.calls[3][4] == "headless-client"
    assert stub.calls[3][3] == "And for women?"
    assert "Follow-up questions asked:" in out and "And for women?" in out
    assert "Total time:" in out
    saved = json.loads(out_file.read_text(encoding="utf-8"))
    assert [f["question"] for f in saved["followups"]] == ["Why?", "And for women?"]


def test_cli_auto_answer_stops_after_two_rounds(stub, csv_path, capsys, monkeypatch):
    import audit_cli
    monkeypatch.setattr(sys, "stdin", types.SimpleNamespace(isatty=lambda: False))

    def always_ask(df, state, answers, client=None, **kw):
        stub.calls.append(("resume", len(df), state, dict(answers), client))
        return stub.run_audit(df, state["claim"], state["context"], client=client)
    monkeypatch.setattr(stub, "resume", always_ask)
    code = audit_cli.main([csv_path, "--claim", "treatment raises y", "--auto-answer", "Not sure"])
    assert code == 3
    assert [c[0] for c in stub.calls].count("resume") == 2


def test_cli_bad_answer_format(stub, csv_path, capsys):
    import audit_cli
    assert audit_cli.main([csv_path, "--claim", "c", "--answer", "noequals"]) == 2
