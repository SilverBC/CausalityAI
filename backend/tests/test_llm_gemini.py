"""services/llm.py on Gemini: provider choice, request shape, structured and web answers, errors. No network:
a fake transport (url, body, key, timeout) -> (status, payload) stands in for Google."""
import json

import pytest

from causality.orchestrator import gemini

KEY = "test-key-not-real-123"
llm = None


@pytest.fixture(autouse=True)
def _current_modules():
    """Other test files re-import services.*; use the llm module the current chat module actually calls."""
    global llm
    from services import chat
    llm = chat.llm


def ok(text):
    return 200, {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}],
                 "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5}}


class Transport:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def __call__(self, url, body, key, timeout):
        self.calls.append({"url": url, "body": body, "key": key, "timeout": timeout})
        return self.replies.pop(0)


@pytest.fixture
def gem(monkeypatch, _current_modules):
    """A fake Gemini key, no pacing waits, the daily budget off, and a way to plug in a transport."""
    monkeypatch.delenv("CAUSAL_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", KEY)
    monkeypatch.setenv("GEMINI_MODEL", "gemini-test")
    monkeypatch.setenv("GEMINI_RPM", "6000")
    monkeypatch.setattr(llm, "_gemini_sleep", lambda s: None)
    monkeypatch.setattr(gemini, "_next_slot", {})
    import importlib
    monkeypatch.setattr(importlib.import_module("services.limits"), "spend_llm_call", lambda: True)

    def plug(replies):
        t = Transport(replies)
        monkeypatch.setattr(llm, "_gemini_transport", t)
        return t
    return plug


# --------------------------------------------------------------------------------------------- choice
def test_gemini_is_chosen_when_its_key_is_set_even_with_an_anthropic_key(gem, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    assert llm.backend() == "gemini" and llm.available()
    assert llm.describe() == "Google Gemini (gemini-test)"


def test_no_key_means_off_and_a_plain_reason(monkeypatch):
    monkeypatch.delenv("CAUSAL_LLM", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm.backend() == "off" and not llm.available()
    assert llm.run("hi") is None and "GEMINI_API_KEY" in llm.last_error()


def test_anthropic_key_alone_still_picks_the_api(monkeypatch):
    monkeypatch.delenv("CAUSAL_LLM", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    assert llm.backend() == "api"


def test_the_removed_cli_setting_falls_back_to_auto(gem, monkeypatch):
    monkeypatch.setenv("CAUSAL_LLM", "cli")
    assert llm.backend() == "gemini"
    assert not hasattr(llm, "_cli") and not hasattr(llm, "claude_bin")


def test_fake_and_off_still_win(gem, monkeypatch):
    monkeypatch.setenv("CAUSAL_LLM", "fake")
    assert llm.backend() == "fake" and llm.run("x", kind="summary").startswith("Fake executive summary")
    monkeypatch.setenv("CAUSAL_LLM", "off")
    assert llm.backend() == "off" and llm.run("x") is None


# --------------------------------------------------------------------------------------------- answers
def test_text_answer_and_request_shape(gem):
    t = gem([ok("  The effect is about 1,800 dollars.  ")])
    assert llm.run("Summarise", system="Be brief.", kind="summary") == "The effect is about 1,800 dollars."
    call = t.calls[0]
    assert call["url"].endswith("/models/gemini-test:generateContent") and call["key"] == KEY
    b = call["body"]
    assert b["systemInstruction"]["parts"][0]["text"] == "Be brief."
    assert b["contents"][0]["parts"][0]["text"] == "Summarise"
    assert "generationConfig" not in b and "tools" not in b
    assert llm.last_error() is None


def test_structured_answer_uses_json_mode_with_a_gemini_schema(gem):
    from services import chat
    reply = {"reply": "The estimate is 1794.", "cards": [{"kind": "estimate", "ref": None}], "investigate": None}
    t = gem([ok(json.dumps(reply))])
    assert llm.run("q", schema=chat.SCHEMA, system=chat.SYSTEM, kind="chat") == reply
    gc = t.calls[0]["body"]["generationConfig"]
    assert gc["responseMimeType"] == "application/json"
    s = gc["responseSchema"]
    assert s["type"] == "OBJECT" and s["required"] == ["reply", "cards", "investigate"]
    assert s["properties"]["investigate"] == {"nullable": True, "type": "STRING"}
    card = s["properties"]["cards"]["items"]
    assert card["properties"]["kind"]["enum"] == ["estimate", "check", "graph", "investigation"]
    assert card["properties"]["ref"]["nullable"] is True and "additionalProperties" not in json.dumps(s)


def test_web_search_uses_grounding_and_reads_json_from_text(gem):
    from services import investigate
    found = {"summary": "s", "factors": []}
    t = gem([ok("```json\n" + json.dumps(found) + "\n```")])
    assert llm.run("Investigate", schema=investigate.SCHEMA, web=True, kind="investigate") == found
    b = t.calls[0]["body"]
    assert b["tools"] == [{"google_search": {}}] and "generationConfig" not in b
    assert "JSON schema" in b["systemInstruction"]["parts"][0]["text"]
    assert t.calls[0]["url"].endswith("/models/gemini-2.5-flash-lite:generateContent")   # the web model, not GEMINI_MODEL


def test_web_model_can_be_set(gem, monkeypatch):
    monkeypatch.setenv("GEMINI_WEB_MODEL", "gemini-web-test")
    t = gem([ok("found"), ok("plain")])
    assert llm.run("Search", web=True) == "found" and llm.run("No search") == "plain"
    assert "/models/gemini-web-test:" in t.calls[0]["url"] and "/models/gemini-test:" in t.calls[1]["url"]


def test_search_refused_on_the_free_tier_fails_at_once(gem):
    """A 429 with no quota id and no retry delay on a search call is a refusal, not a rate limit: one try only."""
    refused = (429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "You exceeded your current quota",
                               "details": [{"@type": "type.googleapis.com/google.rpc.Help", "links": []}]}})
    t = gem([refused] * 5)
    assert llm.run("Search", web=True) is None and len(t.calls) == 1
    assert "Google Search is not available" in llm.last_error() and KEY not in llm.last_error()
    t = gem([refused, ok("fine")])                       # the same body without search is still retried
    assert llm.run("No search") == "fine" and len(t.calls) == 2


def test_unreadable_structured_answer_fails_plainly(gem):
    gem([ok("not json at all")])
    assert llm.run("q", schema={"type": "object", "properties": {}}) is None
    assert "not valid JSON" in llm.last_error()


# --------------------------------------------------------------------------------------------- errors
def test_daily_quota_429_is_a_plain_error_without_the_key(gem):
    body = {"error": {"code": 429, "message": "Quota exceeded", "details": [
        {"violations": [{"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}}
    gem([(429, body)])
    assert llm.run("hi") is None
    err = llm.last_error()
    assert "free Gemini quota" in err and KEY not in err


def test_bad_key_is_reported_without_the_key(gem):
    gem([(400, {"error": {"code": 400, "message": "API key not valid. Please pass a valid API key."}})])
    assert llm.run("hi") is None
    assert "Gemini rejected the request" in llm.last_error() and KEY not in llm.last_error()


def test_transient_errors_are_retried(gem):
    t = gem([(503, {"error": {"message": "overloaded"}}), (0, "TimeoutError: timed out"), ok("fine")])
    assert llm.run("hi") == "fine" and len(t.calls) == 3


def test_retries_stop_at_the_callers_timeout(gem, monkeypatch):
    clock = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    monkeypatch.setattr(llm.time, "monotonic", lambda: next(clock, 100.0))
    gem([(503, {"error": {"message": "overloaded"}})] * 5)
    assert llm.run("hi", timeout=30) is None
    assert llm.last_error() == "Gemini did not answer within 30 s."


def test_chat_reply_through_gemini(gem, tmp_path, monkeypatch):
    """chat.reply end to end on a minimal analysis: the Gemini answer becomes the reply, invented cards are dropped."""
    from services import chat, investigate
    monkeypatch.setattr(investigate, "public", lambda aid: [])
    a = {"id": "a1", "title": "toy", "mode": "audit", "claim": "x raises y",
         "result": {"report": {"estimate": {"estimate": 2.5, "ci95": [1.0, 4.0]},
                               "sections": [{"key": "balance", "title": "Balance", "status": "pass"}]}}}
    gem([ok(json.dumps({"reply": "The estimate is 2.5 (95% CI 1.0 to 4.0).",
                        "cards": [{"kind": "estimate", "ref": None}, {"kind": "check", "ref": "nope"}],
                        "investigate": None}))])
    out = chat.reply(a, [], "Does the data support the claim?")
    assert out == {"reply": "The estimate is 2.5 (95% CI 1.0 to 4.0).", "cards": [{"kind": "estimate", "ref": None}]}


def test_chat_failure_raises_chat_unavailable(gem, monkeypatch):
    from services import chat, investigate
    monkeypatch.setattr(investigate, "public", lambda aid: [])
    a = {"id": "a1", "title": "toy", "mode": "audit", "claim": None, "result": {}}
    gem([(429, {"error": {"message": "q", "details": [{"violations": [{"quotaId": "PerDayX"}]}]}})])
    with pytest.raises(chat.ChatUnavailable, match="free Gemini quota"):
        chat.reply(a, [], "hi")
