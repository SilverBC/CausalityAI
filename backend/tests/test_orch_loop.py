"""Part C: the model loop. Every test uses FakeClient; evidence and floor are small stand-ins
that follow the contract, so these tests do not depend on parts A and B."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import causality.orchestrator as pkg
from causality.orchestrator import contract, loop, prompt
from causality.orchestrator.loop import FakeClient, text, thinking, tool_use


# --------------------------------------------------------------------------- stand-ins

def _make_evidence():
    ev = types.ModuleType("causality.orchestrator.evidence")
    ev.TOOL_SPECS = [{"name": n, "description": f"stand-in {n}",
                      "input_schema": {"type": "object", "properties": {}}} for n in contract.EVIDENCE_TOOLS]
    ev.seen = []

    def call(name, args, df, id):
        ev.seen.append((name, args, id))
        if name == "table_facts":
            return contract.evidence_item(id, name, args, lines=[f"The table has {len(df)} rows.",
                                                                 "Columns: " + ", ".join(df.columns) + "."])
        if name == "explore" and args.get("break"):
            return contract.evidence_item(id, name, args, ok=False, error="The table could not be explored.")
        if name == "estimate_effect":
            roles = args.get("roles", {})
            return contract.evidence_item(id, name, args, lines=["The estimate is 2.0 (95% interval 1.0 to 3.0)."],
                                          numbers={"design": args.get("design"), "roles": roles, "estimate": 2.0,
                                                   "ci": {"95": [1.0, 3.0]}, "refused": False, "outcome_sd": 4.0})
        if name == "sensitivity":
            return contract.evidence_item(id, name, args, lines=["A confounder would need a partial R2 of 0.2."],
                                          numbers={"robustness_value": 0.2,
                                                   "untestable": ["No hidden confounder affects both."]})
        return contract.evidence_item(id, name, args, lines=[f"{name} computed something."])

    ev.call = call
    return ev


def _make_floor():
    fl = types.ModuleType("causality.orchestrator.floor")
    fl.calls = []

    def apply(draft, reading, evidence):
        fl.calls.append((draft, reading, evidence))
        verdict = json.loads(json.dumps(draft))
        violations = []
        for item in evidence:
            for a in item.get("numbers", {}).get("untestable", []):
                if a not in verdict["assumed"]:
                    verdict["assumed"].append(a)
                    violations.append({"rule": "assumed_is_not_tested", "detail": a, "action": "added"})
        return verdict, {"changed": bool(violations), "violations": violations}

    fl.apply = apply
    return fl


@pytest.fixture(autouse=True)
def stand_ins(monkeypatch):
    ev, fl = _make_evidence(), _make_floor()
    monkeypatch.setitem(sys.modules, "causality.orchestrator.evidence", ev)
    monkeypatch.setitem(sys.modules, "causality.orchestrator.floor", fl)
    monkeypatch.setattr(pkg, "evidence", ev, raising=False)
    monkeypatch.setattr(pkg, "floor", fl, raising=False)
    # The optional parts (layout, helpers, screen) are not loaded here: a part that does not import
    # is not offered. tests/test_orch_followup.py covers them with stand-ins.
    for mod in ("layout", "helpers", "screen"):
        monkeypatch.setitem(sys.modules, f"causality.orchestrator.{mod}", None)
    return ev, fl


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    age = rng.normal(40, 10, n)
    t = rng.integers(0, 2, n)
    return pd.DataFrame({"age": age, "training": t, "earnings": 2 * t + 0.1 * age + rng.normal(0, 4, n),
                         "job_after": t})


ROLES = {"treatment": "training", "outcome": "earnings"}
CLAIM = "The training program raised earnings."
CONTEXT = "People signed up for training if they wanted to."


def _submit(est="e4", label=contract.SUPPORTED, judgments=None, **reading_extra):
    reading = {"treatment": "training", "outcome": "earnings", "design": "observational",
               "design_reason": "People chose training themselves, per the description.",
               "roles": {**{k: None for k in contract.ROLE_KEYS}, "cutoff": None,
                         "treatment": "training", "outcome": "earnings"},
               "covariates": ["age"],
               "excluded": [{"column": "job_after", "reason": "It is measured after training and training can change it."}]}
    reading.update(reading_extra)
    verdict = {"label": label, "sentence": "Earnings were higher after training [e4].",
               "estimate_evidence_id": est, "claimed_direction": "increase",
               "reasons": ["The 95% interval is above zero [e4]."], "assumed": [],
               "caveats": ["Sensitivity is moderate [e5]."],
               "judgments": judgments if judgments is not None else [
                   {"evidence_id": "e3", "reading": "supports", "why": "Groups overlap."},
                   {"evidence_id": "e4", "reading": "supports", "why": "Interval above zero."},
                   {"evidence_id": "e5", "reading": "not informative", "why": "Describes fragility only."}]}
    return tool_use("submit_verdict", {"reading": reading, "verdict": verdict})


def _full_script():
    return [
        [thinking("look at structure"), text("I will look at how treatment is laid out."),
         tool_use("structure_facts", ROLES), tool_use("general_checks", {**ROLES, "covariates": ["age"]})],
        [tool_use("estimate_effect", {"design": "observational", "roles": ROLES, "covariates": ["age"]}),
         tool_use("sensitivity", {"design": "observational", "roles": ROLES, "covariates": ["age"]})],
        [text("Submitting."), _submit()],
    ]


# --------------------------------------------------------------------------- tests

def test_full_run_ends_in_a_verdict(df, stand_ins):
    ev, fl = stand_ins
    client = FakeClient(_full_script())
    report = loop.run_audit(df, CLAIM, CONTEXT, client=client)
    assert report["status"] == "done", report["error"]
    assert report["error"] is None
    # the state is kept after the verdict so a follow-up can continue; it never holds the table
    assert report["state"]["finished"] is True and report["state"]["asks"] == []
    assert json.loads(json.dumps(report["state"])) == report["state"]
    assert [e["id"] for e in report["evidence"]] == ["e1", "e2", "e3", "e4", "e5"]
    assert [e["tool"] for e in report["evidence"]] == ["table_facts", "structure_facts", "general_checks",
                                                       "estimate_effect", "sensitivity"]
    assert report["reading"]["design"] == "observational"
    assert report["reading"]["roles"] == {"treatment": "training", "outcome": "earnings"}
    assert report["reading"]["excluded"][0]["column"] == "job_after"
    assert report["draft_verdict"]["label"] == contract.SUPPORTED
    assert report["draft_verdict"]["assumed"] == []
    # the floor ran and its output fills verdict and floor
    assert len(fl.calls) == 1
    assert report["verdict"]["assumed"] == ["No hidden confounder affects both."]
    assert report["floor"]["changed"] is True
    assert set(report) == set(contract.empty_report())
    json.dumps(report)

    first = client.calls[0]
    assert first["model"] == contract.MODEL
    assert first["thinking"] == {"type": "adaptive"}
    assert first["system"] == prompt.SYSTEM_PROMPT
    names = [t["name"] for t in first["tools"]]
    assert names == list(contract.EVIDENCE_TOOLS) + ["ask_user", "submit_verdict"]
    assert all(t.get("strict") for t in first["tools"] if t["name"] in ("ask_user", "submit_verdict"))
    opening = first["messages"][0]["content"]
    assert CLAIM in opening and CONTEXT in opening
    assert "[e1]" in opening and "The table has 200 rows." in opening

    # assistant blocks, thinking included, go back unmodified; tool results are "[id] " + lines
    second = client.calls[1]["messages"]
    assert second[1]["role"] == "assistant"
    assert second[1]["content"][1] == {"type": "text", "text": "I will look at how treatment is laid out."}
    assert [b.get("name") for b in second[1]["content"][2:]] == ["structure_facts", "general_checks"]
    assert second[1]["content"][0] == {"type": "thinking", "thinking": "look at structure",
                                       "signature": "fake-signature"}
    results = second[2]["content"]
    assert [r["type"] for r in results] == ["tool_result", "tool_result"]
    assert results[0]["content"] == "[e2] structure_facts computed something."
    assert results[1]["content"].startswith("[e3] ")
    assert "is_error" not in results[0]
    assert report["transcript"][0]["role"] == "user"
    assert {"role": "assistant", "text": "Submitting."} in report["transcript"]


def test_ask_user_then_resume(df):
    client = FakeClient([
        [tool_use("structure_facts", ROLES),
         tool_use("ask_user", {"questions": [{"id": "assign", "question": "How was training assigned?",
                                              "options": ["Randomized", "People chose"]}]})],
    ])
    report = loop.run_audit(df, CLAIM, "", client=client)
    assert report["status"] == "needs_input"
    assert report["questions"] == [{"id": "assign", "question": "How was training assigned?",
                                    "options": ["Randomized", "People chose"]}]
    assert [e["id"] for e in report["evidence"]] == ["e1", "e2"]
    state = json.loads(json.dumps(report["state"]))       # JSON round-trip, table not inside

    client2 = FakeClient([
        [tool_use("general_checks", ROLES)],
        [tool_use("estimate_effect", {"design": "observational", "roles": ROLES}),
         tool_use("sensitivity", {"design": "observational", "roles": ROLES})],
        [_submit(est="e4", judgments=[{"evidence_id": "e4", "reading": "supports", "why": "ok"}])],
    ])
    done = loop.resume(df, state, {"assign": "People chose"}, client=client2)
    assert done["status"] == "done", done["error"]
    assert [e["id"] for e in done["evidence"]] == ["e1", "e2", "e3", "e4", "e5"]
    sent = client2.calls[0]["messages"][-1]
    assert sent["role"] == "user"
    by_type = [r for r in sent["content"]]
    assert by_type[0]["content"].startswith("[e2] ")              # the evidence run before pausing
    assert "How was training assigned?" in by_type[1]["content"]
    assert "Answer: People chose" in by_type[1]["content"]
    ask_id = client2.calls[0]["messages"][-2]["content"][1]["id"]
    assert by_type[1]["tool_use_id"] == ask_id
    assert done["usage"]["turns"] == 4
    assert any("People chose" in t["text"] for t in done["transcript"] if t["role"] == "user")


def test_unanswered_question_is_said_plainly(df):
    client = FakeClient([[tool_use("ask_user", {"questions": [{"id": "q1", "question": "Randomized?",
                                                               "options": []}]})]])
    report = loop.run_audit(df, CLAIM, "", client=client)
    sub =_submit(est=None, label=contract.INCONCLUSIVE, judgments=[])
    sub["input"]["verdict"].update({"sentence": "Nothing was estimated.", "reasons": [], "caveats": []})
    client2 = FakeClient([[sub]])
    done = loop.resume(df, report["state"], {}, client=client2)
    assert done["status"] == "done", done["error"]
    assert "the user did not answer" in client2.calls[0]["messages"][-1]["content"][0]["content"]


def test_invalid_submit_is_corrected_on_the_second_try(df):
    bad = _submit(est="e2", label="Probably", judgments=[{"evidence_id": "e99", "reading": "passed", "why": "x"}],
                  treatment="salary")
    client = FakeClient(_full_script()[:2] + [[bad], [_submit()]])
    report = loop.run_audit(df, CLAIM, CONTEXT, client=client)
    assert report["status"] == "done", report["error"]
    err = client.calls[3]["messages"][-1]["content"][0]
    assert err["is_error"] is True and err["tool_use_id"] == bad["id"]
    for bit in ("label", "e99", "salary", "not an estimate_effect item", "supports"):
        assert bit in err["content"]
    assert report["draft_verdict"]["label"] == contract.SUPPORTED


def test_two_invalid_submits_give_up(df):
    bad = _submit(label="Probably")
    client = FakeClient(_full_script()[:2] + [[bad], [_submit(label="Still wrong")]])
    report = loop.run_audit(df, CLAIM, CONTEXT, client=client)
    assert report["status"] == "error"
    assert "not valid" in report["error"]
    assert report["verdict"] is None


def test_model_stops_without_submitting(df):
    client = FakeClient([[text("I think it is fine.")]])
    report = loop.run_audit(df, CLAIM, "", client=client)
    assert report["status"] == "error"
    assert "without submitting" in report["error"]
    assert "I think it is fine." in report["error"]


def test_turn_limit(df):
    client = FakeClient([[tool_use("explore")] for _ in range(5)])
    report = loop.run_audit(df, CLAIM, "", client=client, max_turns=2)
    assert report["status"] == "error"
    assert "2 turns" in report["error"]
    assert len(client.calls) == 2
    assert [e["id"] for e in report["evidence"]] == ["e1", "e2", "e3"]


def test_api_error_never_raises(df):
    report = loop.run_audit(df, CLAIM, "", client=FakeClient([RuntimeError("socket closed")]))
    assert report["status"] == "error"
    assert "socket closed" in report["error"]
    assert report["evidence"][0]["id"] == "e1"


def test_sdk_api_error_gets_a_plain_message(df):
    anthropic = pytest.importorskip("anthropic")
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    exc = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    report = loop.run_audit(df, CLAIM, "", client=FakeClient([exc]))
    assert report["status"] == "error"
    assert report["error"] == "Could not reach the Claude API. Check the network connection."


def test_refusal_and_max_tokens(df):
    r = loop.run_audit(df, CLAIM, "", client=FakeClient([{"content": [], "stop_reason": "refusal",
                                                          "stop_details": {"category": "cyber"}}]))
    assert r["status"] == "error" and "declined" in r["error"] and "cyber" in r["error"]
    cut = {"content": [text("partial"), tool_use("explore")], "stop_reason": "max_tokens"}
    fake = FakeClient([cut])
    r = loop.run_audit(df, CLAIM, "", client=fake)
    assert r["status"] == "error" and "cut off" in r["error"]
    assert [e["tool"] for e in r["evidence"]] == ["table_facts"]          # truncated call never ran


def test_state_round_trips_through_json(df):
    client = FakeClient([[thinking("hmm"), tool_use("ask_user", {"questions": [
        {"id": "q1", "question": "Was it randomized?", "options": ["Yes", "No"]},
        {"id": "q1", "question": "Who chose?", "options": []}]})]])
    report = loop.run_audit(df, CLAIM, "", client=client)
    assert [q["id"] for q in report["questions"]] == ["q1", "q1_2"]
    dumped = json.dumps(report)
    back = json.loads(dumped)
    assert back["state"] == report["state"]
    assert back["state"]["messages"][1]["content"][0]["type"] == "thinking"


def test_usage_is_counted(df):
    script = _full_script()
    script[0] = {"content": script[0], "usage": {"input_tokens": 1000, "output_tokens": 300,
                                                 "cache_read_input_tokens": 50, "cache_creation_input_tokens": 7}}
    report = loop.run_audit(df, CLAIM, "", client=FakeClient(script))
    assert report["usage"] == {"model": contract.MODEL, "turns": 3,
                               "input_tokens": 1057 + 200, "output_tokens": 340}


def test_evidence_failure_goes_back_as_tool_error(df):
    client = FakeClient([[tool_use("explore", {"break": True})], [text("done")]])
    report = loop.run_audit(df, CLAIM, "", client=client)
    res = client.calls[1]["messages"][-1]["content"][0]
    assert res["is_error"] is True
    assert res["content"] == "[e2] The table could not be explored."
    assert report["evidence"][1]["ok"] is False


def test_unknown_tool_and_bad_resume_state(df):
    client = FakeClient([[tool_use("make_coffee")], [text("stop")]])
    loop.run_audit(df, CLAIM, "", client=client)
    assert "no tool named make_coffee" in client.calls[1]["messages"][-1]["content"][0]["content"]
    r = loop.resume(df, {"nonsense": True}, {"q1": "x"}, client=FakeClient([]))
    assert r["status"] == "error" and "no paused audit" in r["error"]


def test_fallback_blocks_are_echoed_by_the_rules():
    blocks = [{"type": "thinking", "thinking": "", "signature": "a"}, {"type": "text", "text": "part"},
              {"type": "tool_use", "id": "t1", "name": "explore", "input": {}},
              {"type": "fallback", "from": {"model": "x"}, "to": {"model": "y"}},
              {"type": "thinking", "thinking": "", "signature": "b"},
              {"type": "tool_use", "id": "t2", "name": "explore", "input": {}}]
    out = loop._echo_blocks(blocks)
    assert [b.get("signature") or b.get("id") or b["type"] for b in out] == ["text", "fallback", "b", "t2"]
    assert loop._echo_blocks(blocks[:3]) == blocks[:3]


def test_prompt_states_the_rules():
    s = " ".join(prompt.SYSTEM_PROMPT.split())
    for bit in ("never claim more than the data can show", "ask_user", "randomized", "consequence of treatment",
                "Assumed is not tested", "real interval", "No more than the interval"):
        assert bit in s
    for bit in ("layout_facts", "custom_checks", "show_chart", "cause of treatment or a consequence",
                "only the user can", "follow-up", "Never call a design an experiment unless the user or the study "
                "description says assignment was randomized"):
        assert bit in s
