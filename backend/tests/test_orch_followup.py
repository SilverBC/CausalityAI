"""Part W: the optional tools (layout_facts, custom_checks, show_chart) wired into the loop, and
follow-up questions after the verdict. FakeClient only; evidence, floor, layout, helpers and screen
are small stand-ins that follow the contract, so nothing here depends on the real parts."""
import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import causality.orchestrator as pkg
from causality.orchestrator import contract, loop, prompt
from causality.orchestrator.loop import FakeClient, text, tool_use


# --------------------------------------------------------------------------- stand-ins

def _spec(name, props):
    return {"name": name, "description": f"stand-in {name}", "strict": True,
            "input_schema": {"type": "object", "properties": props, "required": list(props),
                             "additionalProperties": False}}


def _make_evidence():
    ev = types.ModuleType("causality.orchestrator.evidence")
    ev.TOOL_SPECS = [_spec(n, {}) for n in contract.EVIDENCE_TOOLS]

    def call(name, args, df, id):
        if name == "estimate_effect":
            return contract.evidence_item(id, name, args, lines=["The estimate is 2.0 (95% interval 1.0 to 3.0)."],
                                          numbers={"design": args.get("design"), "roles": args.get("roles", {}),
                                                   "estimate": 2.0, "ci": {"95": [1.0, 3.0]}, "refused": False,
                                                   "outcome_sd": 4.0, "comparisons": []})
        return contract.evidence_item(id, name, args, lines=[f"{name} computed something."],
                                      numbers={"untestable": ["No hidden confounder."]} if name == "sensitivity" else {})
    ev.call = call
    return ev


def _make_floor():
    fl = types.ModuleType("causality.orchestrator.floor")
    fl.calls = []

    def apply(draft, reading, evidence):
        fl.calls.append([e["id"] for e in evidence])
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


def _make_layout():
    m = types.ModuleType("causality.orchestrator.layout")
    m.TOOL_SPEC = _spec("layout_facts", {"treatment": {"type": "string"}, "outcome": {"type": "string"}})

    def run(df, args):
        if args.get("treatment") not in df.columns:
            raise ValueError(f"'{args.get('treatment')}' is not a column in the table.")
        return (["Treatment switches at score = 50 for 98% of rows."],
                {"thresholds": [{"column": "score", "cutoff": 50.0}], "panels": [],
                 "treatment_predictability": {"value": 0.9, "metric": "auc"},
                 "instrument_candidates": [], "associated_columns": []})
    m.run = run
    return m


def _make_helpers():
    m = types.ModuleType("causality.orchestrator.helpers")
    m.TOOL_SPEC = _spec("custom_checks", {"questions": {"type": "array"}})
    m.calls = []

    def run(df, args, writer=None, **kw):
        m.calls.append((args, writer))
        out = []
        for q in args["questions"]:
            if "fail" in q["question"]:
                out.append({"id": q["id"], "question": q["question"], "ok": False, "lines": [], "numbers": {},
                            "error": "The script did not run after one repair.", "code": "raise SystemExit(1)",
                            "seconds": 1.5})
            else:
                out.append({"id": q["id"], "question": q["question"], "ok": True,
                            "lines": [f"Difference in means: 1.2 (95% interval 0.4 to 2.0), n = 200."],
                            "numbers": {"estimate": 1.2, "ci95_low": 0.4, "ci95_high": 2.0, "n": 200},
                            "error": None, "code": "print('hi')", "seconds": 2.0})
        return out
    m.run = run
    return m


def _make_screen():
    m = types.ModuleType("causality.orchestrator.screen")
    m.TOOL_SPEC = _spec("show_chart", {k: {"anyOf": [{"type": "string"}, {"type": "null"}]}
                                       for k in ("evidence_id", "kind", "x", "y", "group", "title", "note")})

    def charts_for(item, df):
        if item.get("tool") != "estimate_effect":
            return []
        return [{"kind": "forest", "title": "Estimate and alternatives", "data": [{"label": "main", "estimate": 2.0,
                                                                                  "lo": 1.0, "hi": 3.0}],
                 "x_label": "effect", "y_label": "", "note": ""},
                {"kind": "scatter", "title": "Outcome by score", "data": {"x": [1, 2], "y": [3, 4]},
                 "x_label": "score", "y_label": "y", "note": ""}]

    def draw(df, args):
        if args.get("x") not in df.columns:
            raise ValueError(f"'{args.get('x')}' is not a column in the table.")
        return {"kind": args.get("kind") or "hist", "title": args.get("title") or "", "data": {"bins": [0, 1],
                "counts": [5]}, "x_label": args["x"], "y_label": "count", "note": ""}
    m.charts_for, m.draw = charts_for, draw
    return m


@pytest.fixture(autouse=True)
def parts(monkeypatch):
    mods = {"evidence": _make_evidence(), "floor": _make_floor(), "layout": _make_layout(),
            "helpers": _make_helpers(), "screen": _make_screen()}
    for name, mod in mods.items():
        monkeypatch.setitem(sys.modules, f"causality.orchestrator.{name}", mod)
        monkeypatch.setattr(pkg, name, mod, raising=False)
    return types.SimpleNamespace(**mods)


@pytest.fixture
def df():
    rng = np.random.default_rng(1)
    n = 200
    score = rng.uniform(0, 100, n)
    t = (score >= 50).astype(int)
    return pd.DataFrame({"score": score, "training": t, "earnings": 2 * t + rng.normal(0, 4, n)})


ROLES = {"treatment": "training", "outcome": "earnings"}
CLAIM = "The training program raised earnings."


def _submit(est, label=contract.SUPPORTED, cite="e3", sentence=None):
    reading = {"treatment": "training", "outcome": "earnings", "design": "observational",
               "design_reason": "People chose training themselves.",
               "roles": {**{k: None for k in contract.ROLE_KEYS}, "cutoff": None, **ROLES},
               "covariates": None, "excluded": []}
    verdict = {"label": label, "sentence": sentence or f"Earnings were higher after training [{est}].",
               "estimate_evidence_id": est, "claimed_direction": "increase",
               "reasons": [f"The interval is above zero [{est}]."], "assumed": [], "caveats": [],
               "judgments": [{"evidence_id": cite, "reading": "supports", "why": "ok"}]}
    return tool_use("submit_verdict", {"reading": reading, "verdict": verdict})


def _est():
    return tool_use("estimate_effect", {"design": "observational", "roles": ROLES})


# --------------------------------------------------------------------------- the three tools

def test_new_tools_are_offered_next_to_the_evidence_tools(df):
    client = FakeClient([[_est()], [_submit("e2", cite="e2")]])
    loop.run_audit(df, CLAIM, "", client=client)
    names = [t["name"] for t in client.calls[0]["tools"]]
    assert names == list(contract.EVIDENCE_TOOLS) + ["layout_facts", "custom_checks", "show_chart",
                                                     "ask_user", "submit_verdict"]


def test_a_part_that_does_not_import_is_not_offered(df, monkeypatch):
    monkeypatch.setitem(sys.modules, "causality.orchestrator.screen", None)
    client = FakeClient([[tool_use("show_chart", {"x": "score"})], [text("stop")]])
    report = loop.run_audit(df, CLAIM, "", client=client)
    assert "show_chart" not in [t["name"] for t in client.calls[0]["tools"]]
    assert "no tool named show_chart" in client.calls[1]["messages"][-1]["content"][0]["content"]
    assert report["screen"] == []


def test_layout_facts_is_one_evidence_item(df):
    client = FakeClient([[tool_use("layout_facts", ROLES), tool_use("layout_facts", {"treatment": "nope",
                                                                                     "outcome": "earnings"})],
                         [text("stop")]])
    report = loop.run_audit(df, CLAIM, "", client=client)
    e2, e3 = report["evidence"][1:3]
    assert (e2["id"], e2["tool"], e2["ok"]) == ("e2", "layout_facts", True)
    assert e2["numbers"]["thresholds"][0]["cutoff"] == 50.0
    assert (e3["id"], e3["ok"]) == ("e3", False) and "not a column" in e3["error"]
    res = client.calls[1]["messages"][-1]["content"]
    assert res[0]["content"] == "[e2] Treatment switches at score = 50 for 98% of rows."
    assert res[1]["is_error"] is True and res[1]["content"].startswith("[e3] ")


def test_custom_checks_give_one_item_per_question(df, parts):
    def writer(system, prompt_text):
        return "print('never called by the stand-in')"
    qs = [{"id": "a", "question": "Compare with the never-treated region only."},
          {"id": "b", "question": "This one will fail."}]
    client = FakeClient([[tool_use("custom_checks", {"questions": qs}), _est()], [text("stop")]])
    report = loop.run_audit(df, CLAIM, "", client=client, helper_writer=writer)
    ev = report["evidence"]
    assert [(e["id"], e["tool"]) for e in ev] == [("e1", "table_facts"), ("e2", "custom_check"),
                                                  ("e3", "custom_check"), ("e4", "estimate_effect")]
    assert ev[1]["args"] == {"question": qs[0]["question"]} and ev[1]["ok"] is True
    assert ev[1]["numbers"]["estimate"] == 1.2
    assert ev[1]["numbers"]["code"] == "print('hi')" and ev[1]["numbers"]["seconds"] == 2.0
    assert ev[2]["ok"] is False and "one repair" in ev[2]["error"] and ev[2]["numbers"]["code"]
    assert parts.helpers.calls[0][1] is writer
    body = client.calls[1]["messages"][-1]["content"][0]
    assert "is_error" not in body                         # one of the two ran
    assert "[e2] Difference in means" in body["content"] and "[e3] The script did not run" in body["content"]
    assert "print('hi')" not in body["content"]           # the code stays out of the conversation
    json.dumps(report)


def test_custom_checks_bad_args_run_nothing(df, parts):
    client = FakeClient([[tool_use("custom_checks", {"questions": [{"id": str(i), "question": "x"}
                                                                   for i in range(5)]})], [text("stop")]])
    report = loop.run_audit(df, CLAIM, "", client=client)
    assert parts.helpers.calls == [] and len(report["evidence"]) == 1
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_show_chart_fills_the_screen(df):
    client = FakeClient([
        [_est()],
        [tool_use("show_chart", {"evidence_id": "e2", "kind": None, "x": None, "y": None, "group": None,
                                 "title": "Effect", "note": "The bar right of zero is the effect."}),
         tool_use("show_chart", {"evidence_id": None, "kind": "hist", "x": "score", "y": None, "group": None,
                                 "title": "Scores", "note": "Most scores are mid-range."}),
         tool_use("show_chart", {"evidence_id": None, "kind": "hist", "x": "nope", "y": None, "group": None,
                                 "title": "", "note": ""}),
         tool_use("show_chart", {"evidence_id": "e9", "kind": None, "x": None, "y": None, "group": None,
                                 "title": "", "note": ""}),
         tool_use("show_chart", {"evidence_id": "e1", "kind": None, "x": None, "y": None, "group": None,
                                 "title": "", "note": ""})],
        [_submit("e2", cite="e2")],
    ])
    report = loop.run_audit(df, CLAIM, "", client=client)
    assert report["status"] == "done", report["error"]
    screen = report["screen"]
    assert [(s["evidence_id"], s["chart"]["kind"]) for s in screen] == [("e2", "forest"), ("e2", "scatter"),
                                                                       (None, "hist")]
    assert screen[0]["note"] == "The bar right of zero is the effect."
    assert screen[2]["chart"]["title"] == "Scores"
    res = client.calls[2]["messages"][-1]["content"]
    assert res[0]["content"].startswith("Shown on the page: chart 1 (forest") and "chart 2" in res[0]["content"]
    assert res[1]["content"] == "Shown on the page: chart 3 (hist: Scores)."
    assert res[2]["is_error"] and "not a column" in res[2]["content"]
    assert res[3]["is_error"] and "no evidence item e9" in res[3]["content"]
    assert res[4]["is_error"] and "no standard chart" in res[4]["content"]
    assert len(report["evidence"]) == 2                    # charts are not evidence
    assert report["state"]["screen"] == screen


# --------------------------------------------------------------------------- follow-ups

def _finished(df):
    client = FakeClient([[_est(), tool_use("sensitivity", {"design": "observational", "roles": ROLES})],
                         [tool_use("show_chart", {"evidence_id": "e2", "kind": None, "x": None, "y": None,
                                                  "group": None, "title": "", "note": "n"}),
                          _submit("e2", cite="e2")]])
    report = loop.run_audit(df, CLAIM, "", client=client)
    assert report["status"] == "done", report["error"]
    return report, client


def test_followup_continues_the_conversation(df, parts):
    report, first_client = _finished(df)
    state = json.loads(json.dumps(report["state"]))         # JSON round-trip; the table is not inside
    assert "score" not in json.dumps(state["evidence"][0]["args"])
    client = FakeClient([
        [text("Checking another comparison."),
         tool_use("custom_checks", {"questions": [{"id": "q1", "question": "Only units near the cutoff?"}]})],
        [_submit("e2", cite="e4", sentence="The effect holds near the cutoff too [e4] [e2].")],
    ])
    done = loop.followup(df, state, "Does it hold near the cutoff?", client=client)
    assert done["status"] == "done", done["error"]
    assert [e["id"] for e in done["evidence"]] == ["e1", "e2", "e3", "e4"]
    assert done["evidence"][3]["tool"] == "custom_check"
    assert done["followups"] == [{"question": "Does it hold near the cutoff?"}]
    assert len(done["screen"]) == 2                          # earlier screen entries kept
    assert done["verdict"]["sentence"].startswith("The effect holds near the cutoff")
    assert done["verdict"]["assumed"] == ["No hidden confounder."]     # the floor ran again
    assert parts.floor.calls == [["e1", "e2", "e3"], ["e1", "e2", "e3", "e4"]]
    assert done["usage"]["turns"] == 4
    # the follow-up message answers every tool call of the verdict turn, then asks the question
    msgs = client.calls[0]["messages"]
    verdict_turn, asked = msgs[-2], msgs[-1]
    owed = {b["id"] for b in verdict_turn["content"] if b["type"] == "tool_use"}
    results = [b for b in asked["content"] if b["type"] == "tool_result"]
    assert {r["tool_use_id"] for r in results} == owed
    assert any("The verdict was accepted" in r["content"] and "Supported" in r["content"] for r in results)
    assert asked["content"][-1]["type"] == "text"
    assert "Does it hold near the cutoff?" in asked["content"][-1]["text"]
    assert asked["content"][-1]["text"] == prompt.followup_message("Does it hold near the cutoff?")
    assert client.calls[0]["system"] == prompt.SYSTEM_PROMPT
    assert {"role": "user", "text": "Does it hold near the cutoff?"} in done["transcript"]
    # a second follow-up keeps going from the new state
    client2 = FakeClient([[_submit("e2", cite="e2")]])
    again = loop.followup(df, done["state"], "Thanks. Anything else?", client=client2)
    assert again["status"] == "done", again["error"]
    assert [f["question"] for f in again["followups"]] == ["Does it hold near the cutoff?", "Thanks. Anything else?"]
    assert len(again["evidence"]) == 4


def test_followup_submit_is_checked_and_corrected(df):
    report, _ = _finished(df)
    bad = _submit("e99", cite="e2")
    client = FakeClient([[bad], [_submit("e2", cite="e2")]])
    done = loop.followup(df, report["state"], "Why Supported?", client=client)
    assert done["status"] == "done", done["error"]
    err = client.calls[1]["messages"][-1]["content"][0]
    assert err["is_error"] is True and "e99" in err["content"]


def test_followup_two_bad_submits_keep_the_earlier_verdict(df):
    report, _ = _finished(df)
    client = FakeClient([[_submit("e2", label="Probably")], [_submit("e2", label="Still no")]])
    out = loop.followup(df, report["state"], "Why?", client=client)
    assert out["status"] == "error" and "not valid" in out["error"]
    assert out["state"] is None
    assert out["verdict"] == report["verdict"]                # the earlier verdict still stands


def test_followup_can_ask_the_user(df):
    report, _ = _finished(df)
    client = FakeClient([[tool_use("ask_user", {"questions": [{"id": "q1", "question": "Was it a lottery?",
                                                               "options": ["Yes", "No"]}]})]])
    paused = loop.followup(df, report["state"], "What if it was a lottery?", client=client)
    assert paused["status"] == "needs_input" and paused["questions"][0]["id"] == "q1"
    assert loop.followup(df, paused["state"], "Another?", client=FakeClient([]))["status"] == "error"
    done = loop.resume(df, paused["state"], {"q1": "No"}, client=FakeClient([[_submit("e2", cite="e2")]]))
    assert done["status"] == "done", done["error"]
    assert done["followups"] == [{"question": "What if it was a lottery?"}]
    assert done["state"]["finished"] is True


def test_followup_refuses_without_a_finished_audit(df):
    r = loop.followup(df, {"nonsense": True}, "Why?", client=FakeClient([]))
    assert r["status"] == "error" and "no finished audit" in r["error"]
    report, _ = _finished(df)
    r = loop.followup(df, report["state"], "   ", client=FakeClient([]))
    assert r["status"] == "error" and "follow-up question" in r["error"]
    assert loop.resume(df, report["state"], {"q1": "x"}, client=FakeClient([]))["status"] == "error"


def test_followup_api_error_never_raises(df):
    report, _ = _finished(df)
    out = loop.followup(df, report["state"], "Why?", client=FakeClient([RuntimeError("socket closed")]))
    assert out["status"] == "error" and "socket closed" in out["error"]
    assert out["verdict"] == report["verdict"]
    assert report["state"]["followups"] == []                 # the caller's state is not changed
