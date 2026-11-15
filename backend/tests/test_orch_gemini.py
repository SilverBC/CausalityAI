"""The Gemini client: reply parsing, retries, pacing, and a whole audit through the real loop.
No network: every test passes its own transport. One live test runs only with CAUSALITYAI_LIVE=1."""
import json
import os

import pytest

from causality import sim
from causality.orchestrator import contract, gemini, loop


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setattr(gemini, "_env_loaded", True)          # never read a real .env in tests
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_RPM", "6000")
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.delenv("GEMINI_HELPER_MODEL", raising=False)
    gemini._next_slot.clear()


def ok(reply, usage=None):
    text = reply if isinstance(reply, str) else json.dumps(reply)
    return 200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
                 "usageMetadata": usage or {"promptTokenCount": 100, "candidatesTokenCount": 20}}


class Transport:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, body, key, timeout):
        self.calls.append({"url": url, "body": body, "key": key})
        return self.replies.pop(0)


def turn(client, **kw):
    return client.beta.messages.stream(model=kw.pop("model", "claude-opus-5-5"), system="SYS", tools=[{"name": "t"}],
                                       messages=[{"role": "user", "content": "hello"}], **kw).get_final_message()


def test_reply_becomes_tool_use_blocks_and_usage():
    t = Transport([ok({"text": "Looking.", "tool_calls": [
        {"name": "table_facts", "input_json": "{}"},
        {"name": "layout_facts", "input_json": json.dumps({"treatment": "t", "outcome": "y"})}]},
        {"promptTokenCount": 900, "candidatesTokenCount": 50, "thoughtsTokenCount": 30})])
    msg = turn(gemini.GeminiClient(transport=t))
    assert [b["type"] for b in msg.content] == ["text", "tool_use", "tool_use"]
    assert msg.content[2]["input"] == {"treatment": "t", "outcome": "y"}
    assert msg.content[1]["id"] != msg.content[2]["id"]
    assert msg.stop_reason == "tool_use"
    assert (msg.usage.input_tokens, msg.usage.output_tokens) == (900, 80)
    call = t.calls[0]
    assert call["key"] == "test-key" and "gemini-3.5-flash-lite:generateContent" in call["url"]
    assert "test-key" not in call["url"]                      # the key travels in a header, never in the URL
    sys_text = call["body"]["systemInstruction"]["parts"][0]["text"]
    assert sys_text.startswith("SYS") and '"name": "t"' in sys_text and "input_json" in sys_text
    assert call["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert "hello" in call["body"]["contents"][0]["parts"][0]["text"]


def test_model_setting_and_claude_names_are_not_sent_to_gemini(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-4-hydrogen")
    t = Transport([ok({"text": "", "tool_calls": []}), ok({"text": "", "tool_calls": []})])
    c = gemini.GeminiClient(transport=t)
    turn(c)
    turn(c, model="gemini-3.1-flash-lite")
    assert "gemini-4-hydrogen:" in t.calls[0]["url"] and "claude" not in t.calls[0]["url"]
    assert "gemini-3.1-flash-lite:" in t.calls[1]["url"]


@pytest.mark.parametrize("text", [
    '```json\n{"text": "hi", "tool_calls": []}\n```',
    'Sure. {"text": "hi", "tool_calls": []} Done.',
])
def test_fenced_or_wrapped_json_is_still_read(text):
    msg = turn(gemini.GeminiClient(transport=Transport([ok(text)])))
    assert msg.content == [{"type": "text", "text": "hi"}] and msg.stop_reason == "end_turn"


def test_bad_arguments_become_an_empty_call_not_a_crash():
    t = Transport([ok({"text": "", "tool_calls": [{"name": "table_facts", "input_json": "{not json"},
                                                    {"name": "x", "input": {"a": 1}}]})])
    msg = turn(gemini.GeminiClient(transport=t))
    assert [b["input"] for b in msg.content] == [{}, {"a": 1}]


def test_retries_after_a_per_minute_limit_then_succeeds():
    slept = []
    limit = 429, {"error": {"message": "slow down", "details": [
        {"retryDelay": "3s"}, {"violations": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}]}]}}
    t = Transport([limit, (503, {"error": {"message": "overloaded"}}), ok({"text": "ok", "tool_calls": []})])
    msg = turn(gemini.GeminiClient(transport=t, sleep=slept.append))
    assert msg.content[0]["text"] == "ok" and len(t.calls) == 3
    assert slept and slept[0] >= 3


def test_daily_quota_stops_at_once_with_a_plain_message():
    day = 429, {"error": {"message": "quota", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    t = Transport([day])
    with pytest.raises(gemini.GeminiError, match="quota .* is used up"):
        turn(gemini.GeminiClient(transport=t, sleep=lambda s: None))
    assert len(t.calls) == 1


@pytest.mark.parametrize("status, words", [(401, "not accepted"), (404, "not available"), (400, "rejected")])
def test_key_and_model_errors_are_not_retried(status, words):
    t = Transport([(status, {"error": {"message": "nope"}})])
    with pytest.raises(gemini.GeminiError, match=words):
        turn(gemini.GeminiClient(transport=t, sleep=lambda s: None))
    assert len(t.calls) == 1


def test_gives_up_after_repeated_failures_and_never_leaks_the_key():
    t = Transport([(0, "URLError: offline")] * gemini.MAX_TRIES)
    with pytest.raises(gemini.GeminiError) as e:
        turn(gemini.GeminiClient(transport=t, sleep=lambda s: None))
    assert "test-key" not in str(e.value) and len(t.calls) == gemini.MAX_TRIES


def test_no_key_is_a_plain_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert not gemini.available()
    with pytest.raises(gemini.GeminiError, match="GEMINI_API_KEY"):
        turn(gemini.GeminiClient(transport=Transport([])))


def test_empty_or_blocked_answers_are_plain_errors():
    with pytest.raises(gemini.GeminiError, match="blocked"):
        turn(gemini.GeminiClient(transport=Transport([(200, {"promptFeedback": {"blockReason": "SAFETY"}})])))
    with pytest.raises(gemini.GeminiError, match="empty"):
        turn(gemini.GeminiClient(transport=Transport([(200, {"candidates": [{"content": {"parts": []}}]})])))


def test_requests_to_one_model_are_spaced(monkeypatch):
    monkeypatch.setenv("GEMINI_RPM", "6")                     # one every 10 seconds
    now, slept = [100.0], []

    def sleep(s):
        slept.append(s)
        now[0] += s
    for _ in range(3):
        gemini._wait_for_slot("m", sleep=sleep, clock=lambda: now[0])
    assert [round(s) for s in slept] == [10, 10]              # the first goes at once


def test_env_file_is_read_without_overriding_real_variables(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text('# comment\nGEMINI_API_KEY="from-file"\nexport GEMINI_MODEL=gemini-x\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GEMINI_API_KEY")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-real")
    monkeypatch.setattr(gemini, "_env_loaded", False)
    assert gemini.api_key() == "from-file" and gemini.model_name() == "gemini-real"


def test_writer_returns_the_script_text():
    t = Transport([ok("```python\nprint(1)\n```")])
    out = gemini.GeminiWriter(transport=t)("be brief", "write it")
    assert "print(1)" in out
    assert "generationConfig" not in t.calls[0]["body"]       # plain text, not the JSON reply mode


def test_a_whole_audit_runs_through_the_real_loop_and_floor():
    df, truth = sim.rct(n=600)
    roles = {**{k: None for k in contract.ROLE_KEYS}, "treatment": "treated", "outcome": "score", "cutoff": None}
    spec = {"design": "experiment", "roles": roles, "covariates": None}
    submit = {"reading": {"treatment": "treated", "outcome": "score", "design": "experiment",
                          "design_reason": "The description says assignment was by coin flip.", "roles": roles,
                          "covariates": None, "excluded": []},
              "verdict": {"label": "Supported", "sentence": "Treatment raised the score [e2].",
                          "estimate_evidence_id": "e2", "claimed_direction": "increase",
                          "reasons": ["The interval is above zero [e2]."], "assumed": [], "caveats": [],
                          "judgments": [{"evidence_id": "e2", "reading": "supports", "why": "Interval above zero."}]}}
    t = Transport([
        ok({"text": "", "tool_calls": [{"name": "estimate_effect", "input_json": json.dumps(spec)}]}),   # e1 = table facts, run by the loop itself
        ok({"text": "Done.", "tool_calls": [{"name": "submit_verdict", "input_json": json.dumps(submit)}]}),
    ])
    rep = loop.run_audit(df, "Treatment raised the score.", "Participants were assigned by coin flip.",
                         client=gemini.GeminiClient(transport=t), model=gemini.model_name())
    assert rep["status"] == "done", rep["error"]
    assert [e["tool"] for e in rep["evidence"]] == ["table_facts", "estimate_effect"]
    assert rep["verdict"]["label"] == "Supported"
    assert rep["usage"]["model"] == "gemini-3.5-flash-lite" and rep["usage"]["turns"] == 2
    assert "<tool_result" in t.calls[1]["body"]["contents"][0]["parts"][0]["text"]   # results replayed as text


def test_a_gemini_failure_is_an_error_report_not_a_crash():
    df, _ = sim.rct(n=200)
    t = Transport([(401, {"error": {"message": "API key not valid"}})])
    rep = loop.run_audit(df, "Treatment raised the score.", "", client=gemini.GeminiClient(transport=t))
    assert rep["status"] == "error" and rep["error"]


@pytest.mark.skipif(not os.environ.get("CAUSALITYAI_LIVE"), reason="set CAUSALITYAI_LIVE=1 and a real key to run")
def test_live_one_turn(monkeypatch):
    monkeypatch.undo()
    gemini._env_loaded = False
    msg = turn(gemini.GeminiClient())
    assert msg.content
