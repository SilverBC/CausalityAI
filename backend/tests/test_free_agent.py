"""The inverted design on Gemini (free_agent.py), with a scripted transport: no key, no network."""
import json
import os

import pandas as pd
import pytest

from causality.orchestrator import free_agent, gemini


def reply(text="", calls=()):
    body = {"text": text, "tool_calls": [{"name": n, "input_json": json.dumps(a)} for n, a in calls]}
    return 200, {"candidates": [{"content": {"parts": [{"text": json.dumps(body)}]}}],
                 "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}


class Script:
    def __init__(self, turns):
        self.turns, self.bodies = list(turns), []

    def __call__(self, url, body, key, timeout):
        self.bodies.append(body)
        return self.turns.pop(0)


@pytest.fixture
def folder(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_RPM", "6000")
    pd.DataFrame({"t": [0, 1, 0, 1] * 10, "y": [1.0, 2.0, 1.5, 2.5] * 10}).to_csv(tmp_path / "data.csv", index=False)
    return str(tmp_path)


def test_runs_code_then_writes_the_answer(folder):
    code = "import pandas as pd\nd = pd.read_csv('data.csv')\nprint(d.groupby('t').y.mean().diff().iloc[-1])"
    s = Script([reply("look", [("run_python", {"code": code})]),
                reply("done", [("write_answer", {"answer_json": json.dumps({"verdict": "supported", "estimate": 1.0})})])])
    out = free_agent.run(folder, "Audit the claim.", transport=s, sleep=lambda x: None)
    assert out["status"] == "done" and out["turns"] == 2 and out["python_runs"] == 1
    assert json.load(open(os.path.join(folder, "answer.json")))["estimate"] == 1.0
    # the second request carried the script's output back to the model
    assert "1.0" in s.bodies[1]["contents"][0]["parts"][0]["text"]
    assert out["input_tokens"] == 20 and out["output_tokens"] == 10


def test_helper_is_only_offered_when_asked(folder):
    s = Script([reply("", [("layout_facts", {"treatment": "t", "outcome": "y"})]),
                reply("", [("write_answer", {"answer_json": "{}"})])])
    out = free_agent.run(folder, "x", helpers=False, transport=s, sleep=lambda x: None)
    assert out["helper_calls"] == [] and "no such tool" in s.bodies[1]["contents"][0]["parts"][0]["text"]
    assert "layout_facts" not in s.bodies[0]["systemInstruction"]["parts"][0]["text"]


def test_helper_runs_the_calculator(folder):
    s = Script([reply("", [("layout_facts", {"treatment": "t", "outcome": "y"})]),
                reply("", [("write_answer", {"answer_json": "{\"verdict\": \"inconclusive\"}"})])])
    out = free_agent.run(folder, "x", helpers=True, transport=s, sleep=lambda x: None)
    assert out["status"] == "done" and out["helper_calls"] == ["layout"]
    assert "Treatment 't'" in s.bodies[1]["contents"][0]["parts"][0]["text"]


def test_bad_helper_command_and_bad_answer_are_told_to_the_model(folder):
    s = Script([reply("", [("estimate_effect", {"design": "rd", "treatment": "t", "outcome": "nope"})]),
                reply("", [("write_answer", {"answer_json": "not json"})]),
                reply("", [("write_answer", {"answer_json": "{\"verdict\": \"supported\"}"})])])
    out = free_agent.run(folder, "x", helpers=True, transport=s, sleep=lambda x: None)
    assert out["status"] == "done" and out["turns"] == 3
    assert "Could not run" in s.bodies[1]["contents"][0]["parts"][0]["text"]
    assert "not valid JSON" in s.bodies[2]["contents"][0]["parts"][0]["text"]


def test_answer_object_passed_directly_is_accepted(folder):
    s = Script([reply("", [("write_answer", {"verdict": "inconclusive", "estimate": 0.02})])])
    out = free_agent.run(folder, "x", transport=s, sleep=lambda x: None)
    assert out["status"] == "done"
    assert json.load(open(os.path.join(folder, "answer.json")))["verdict"] == "inconclusive"


def test_stops_without_an_answer(folder):
    s = Script([reply("thinking"), reply("still thinking")])
    out = free_agent.run(folder, "x", transport=s, sleep=lambda x: None)
    assert out["status"] == "error" and "without writing an answer" in out["error"]
    assert not os.path.exists(os.path.join(folder, "answer.json"))


def test_turn_limit_and_api_failure_are_plain_errors(folder):
    s = Script([reply("", [("run_python", {"code": "print(1)"})])] * 3)
    out = free_agent.run(folder, "x", max_turns=3, transport=s, sleep=lambda x: None)
    assert out["status"] == "error" and "3 turns" in out["error"]
    out = free_agent.run(folder, "x", transport=lambda *a: (401, {"error": {"message": "bad key"}}), sleep=lambda x: None)
    assert out["status"] == "error" and "not accepted" in out["error"]


def test_model_code_cannot_see_keys(folder, monkeypatch):
    monkeypatch.setenv("SOME_SERVICE_TOKEN", "abc")
    env = free_agent.child_env()
    assert "GEMINI_API_KEY" not in env and "SOME_SERVICE_TOKEN" not in env
    code = "import os\nprint(sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k))"
    s = Script([reply("", [("run_python", {"code": code})]), reply("", [("write_answer", {"answer_json": "{}"})])])
    free_agent.run(folder, "x", transport=s, sleep=lambda x: None)
    assert "[]" in s.bodies[1]["contents"][0]["parts"][0]["text"]


def test_script_error_and_timeout_come_back_as_text(folder, monkeypatch):
    monkeypatch.setattr(free_agent, "SCRIPT_TIMEOUT", 2)
    s = Script([reply("", [("run_python", {"code": "raise ValueError('boom')"})]),
                reply("", [("run_python", {"code": "import time\ntime.sleep(30)"})]),
                reply("", [("write_answer", {"answer_json": "{}"})])])
    out = free_agent.run(folder, "x", transport=s, sleep=lambda x: None)
    assert out["status"] == "done"
    assert "boom" in s.bodies[1]["contents"][0]["parts"][0]["text"]
    assert "stopped after 2 seconds" in s.bodies[2]["contents"][0]["parts"][0]["text"]
