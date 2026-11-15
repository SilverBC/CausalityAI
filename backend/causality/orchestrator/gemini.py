"""Run the audit on Google's Gemini API instead of Claude.

    from causality.orchestrator import loop, gemini
    if gemini.available():
        report = loop.run_audit(df, claim, context, client=gemini.GeminiClient(), model=gemini.model_name())

Settings come from the environment or from a `.env` file (repository root, `backend/`, or the
current folder; real environment variables win):

    GEMINI_API_KEY        required (GOOGLE_API_KEY also works)
    GEMINI_MODEL          the orchestrator model, default gemini-3.5-flash-lite
    GEMINI_WEB_MODEL      the model for calls that search the web, default gemini-2.5-flash-lite
    GEMINI_HELPER_MODEL   the model that writes helper scripts, default = GEMINI_MODEL
    GEMINI_RPM            requests per minute to stay under, per model, default 5 (the free tier is tight)

`GeminiClient` has the surface the loop uses on the Anthropic client
(`client.beta.messages.stream(**kwargs)` -> `.get_final_message()`), so the loop, the checks on
the submitted verdict and the floor rules are unchanged.

How a turn works (the same idea as headless.py): the conversation so far and the tool definitions
go in as text, and the model must answer in a fixed JSON shape,
{"text": ..., "tool_calls": [{"name", "input_json"}]}, enforced with Gemini's JSON response mode.
Gemini's own function calling is not used: our tool schemas use forms it may reject, and its
thinking models need opaque signatures replayed on every turn, which a paused and resumed audit
cannot carry. Arguments are still checked by the loop and by each tool, as with any model.

Nothing is sent anywhere but generativelanguage.googleapis.com. On the free tier Google may use
what is sent (column names, sample rows, summary numbers) to improve its products.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

from . import headless

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_WEB_MODEL = "gemini-2.5-flash-lite"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
MAX_TRIES = 5

REPLY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "text": {"type": "STRING"},
        "tool_calls": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {"name": {"type": "STRING"}, "input_json": {"type": "STRING"}},
            "required": ["name", "input_json"]}},
    },
    "required": ["text", "tool_calls"],
}

PROTOCOL = """

# How to call tools in this session

You cannot call tools natively here. Reply only with one JSON object that has two fields:
- `text`: what you want to say (may be an empty string).
- `tool_calls`: a list of calls, each {"name": <tool name>, "input_json": <the arguments object for
  that tool, written as a JSON string>}. Follow each tool's input_schema exactly: every required
  key present, null where a value does not apply, column names spelled as in the table.

You may put several calls in one reply; their results come back together in the next message,
each marked with the call's id. A reply with an empty `tool_calls` list ends the audit, so
always finish by calling `submit_verdict` (or `ask_user`). Never write a tool result yourself.

The tools:

"""


class GeminiError(RuntimeError):
    """A plain-words failure talking to Gemini."""


# --------------------------------------------------------------------------- settings

_env_loaded = False


def load_env() -> None:
    """Read KEY=VALUE lines from the first .env files found. Never overrides a variable already set."""
    global _env_loaded
    if _env_loaded:
        return
    _env_loaded = True
    here = os.path.dirname(os.path.abspath(__file__))
    backend = os.path.dirname(os.path.dirname(here))
    for folder in (os.getcwd(), backend, os.path.dirname(backend)):
        path = os.path.join(folder, ".env")
        if not os.path.isfile(path):
            continue
        try:
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.replace("export ", "", 1).strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except OSError:
            continue


def api_key() -> str | None:
    load_env()
    return (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip() or None


def available() -> bool:
    return api_key() is not None


def model_name() -> str:
    load_env()
    return (os.environ.get("GEMINI_MODEL") or "").strip() or DEFAULT_MODEL


def web_model_name() -> str:
    """The model for calls that search the web. Google's free tier has no search allowance on the Gemini 3 models,
    so these calls default to a model that has one."""
    load_env()
    return (os.environ.get("GEMINI_WEB_MODEL") or "").strip() or DEFAULT_WEB_MODEL


def helper_model_name() -> str:
    load_env()
    return (os.environ.get("GEMINI_HELPER_MODEL") or "").strip() or model_name()


def _rpm() -> float:
    load_env()
    try:
        return max(0.1, float(os.environ.get("GEMINI_RPM") or 5))
    except ValueError:
        return 5.0


# --------------------------------------------------------------------------- pacing and transport

_pace_lock = threading.Lock()
_next_slot: dict[str, float] = {}


def _wait_for_slot(model: str, sleep=time.sleep, clock=time.monotonic) -> None:
    """Space requests to one model so a burst (several helper scripts at once) stays under the limit."""
    gap = 60.0 / _rpm()
    with _pace_lock:
        now = clock()
        at = max(now, _next_slot.get(model, 0.0))
        _next_slot[model] = at + gap
    if at > now:
        sleep(at - now)


def _http_post(url: str, body: dict, key: str, timeout: float):
    """(status, parsed JSON or text). The default transport; tests pass their own."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "x-goog-api-key": key})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            return e.code, json.loads(raw)
        except Exception:  # noqa: BLE001
            return e.code, raw
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}"


def _error_bits(payload):
    """(message, retry_seconds, quota_ids) from a Gemini error body."""
    if not isinstance(payload, dict):
        return str(payload)[:300], None, []
    err = payload.get("error") or {}
    msg, retry, quotas = str(err.get("message") or "")[:300], None, []
    for d in err.get("details") or []:
        if not isinstance(d, dict):
            continue
        m = re.match(r"^\s*([\d.]+)s\s*$", str(d.get("retryDelay") or ""))
        if m:
            retry = float(m.group(1))
        for v in d.get("violations") or []:
            if isinstance(v, dict) and v.get("quotaId"):
                quotas.append(str(v["quotaId"]))
    return msg, retry, quotas


def generate(model: str, body: dict, timeout: float = 300, transport=None, sleep=time.sleep) -> dict:
    """One generateContent call with pacing and retries. Raises GeminiError with a plain message."""
    key = api_key()
    if not key:
        raise GeminiError("No Gemini API key was found. Put GEMINI_API_KEY=<your key> in the .env file.")
    post = transport or _http_post
    url = ENDPOINT.format(model=model)
    last = "no reply"
    for attempt in range(MAX_TRIES):
        _wait_for_slot(model, sleep=sleep)
        status, payload = post(url, body, key, timeout)
        if status == 200 and isinstance(payload, dict):
            return payload
        msg, retry, quotas = _error_bits(payload)
        last = f"HTTP {status}: {msg}" if status else msg
        if status == 429 and any("PerDay" in q for q in quotas):
            raise GeminiError(f"Today's free Gemini quota for {model} is used up. Try again tomorrow, set "
                              "GEMINI_MODEL to another model, or turn on billing for the key.")
        searching = any(isinstance(t, dict) and "google_search" in t for t in body.get("tools") or [])
        if status == 429 and searching and not quotas and retry is None:
            # No quota id and no retry delay: the request was refused, not throttled. Google gives the free tier
            # no search allowance on some models, so trying again cannot help.
            raise GeminiError(f"Google Search is not available to this key on {model} (the free tier has no search "
                              "allowance for it). Set GEMINI_WEB_MODEL to a model that has one, or turn on billing.")
        if status in (400, 401, 403, 404):
            hint = {400: "Gemini rejected the request", 401: "The Gemini API key was not accepted",
                    403: "The Gemini API key is not allowed to use this model",
                    404: f"The model '{model}' is not available to this key (set GEMINI_MODEL to another one)"}[status]
            raise GeminiError(f"{hint}. {msg}".strip())
        if attempt < MAX_TRIES - 1:
            sleep(min(60.0, (retry or 0) + 2.0 * (2 ** attempt)))
    raise GeminiError(f"Gemini did not answer after {MAX_TRIES} tries ({last}).")


def _text_of(payload: dict) -> str:
    cands = payload.get("candidates") or []
    if not cands:
        why = (payload.get("promptFeedback") or {}).get("blockReason")
        raise GeminiError("Gemini returned no answer" + (f" (blocked: {why})." if why else "."))
    parts = ((cands[0].get("content") or {}).get("parts")) or []
    text = "".join(str(p.get("text") or "") for p in parts if isinstance(p, dict) and not p.get("thought"))
    if not text.strip():
        raise GeminiError(f"Gemini returned an empty answer (finish reason: {cands[0].get('finishReason')}).")
    return text


def _parse_reply(text: str) -> dict:
    """The reply object, tolerating a code fence or stray text around it."""
    s = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        out = json.loads(s)
    except Exception:  # noqa: BLE001
        a, b = s.find("{"), s.rfind("}")
        try:
            out = json.loads(s[a:b + 1]) if 0 <= a < b else None
        except Exception:  # noqa: BLE001
            out = None
    return out if isinstance(out, dict) else {"text": text, "tool_calls": []}


def _call_input(call: dict):
    """(arguments dict, problem or None) from one tool call in the reply."""
    if isinstance(call.get("input"), dict):
        return call["input"], None
    raw = call.get("input_json")
    if isinstance(raw, dict):
        return raw, None
    if raw is None or str(raw).strip() == "":
        return {}, None
    try:
        val = json.loads(raw)
    except Exception as e:  # noqa: BLE001
        return {}, f"input_json was not valid JSON ({e})"
    return (val, None) if isinstance(val, dict) else ({}, "input_json must be a JSON object")


# --------------------------------------------------------------------------- the client

class GeminiClient:
    def __init__(self, model: str | None = None, timeout: float = 300, transport=None, sleep=time.sleep):
        self.model = model or model_name()
        self.timeout, self._transport, self._sleep = timeout, transport, sleep
        self._n = 0
        self.messages = _Messages(self)
        self.beta = SimpleNamespace(messages=self.messages)

    def _turn(self, kwargs: dict):
        asked = str(kwargs.get("model") or "")
        model = asked if asked.startswith("gemini") else self.model
        system = (kwargs.get("system") or "") + PROTOCOL + json.dumps(kwargs.get("tools") or [], indent=1)
        body = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user",
                          "parts": [{"text": headless.render_messages(kwargs.get("messages") or [])}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseSchema": REPLY_SCHEMA},
        }
        payload = generate(model, body, self.timeout, self._transport, self._sleep)
        reply = _parse_reply(_text_of(payload))
        content = []
        if str(reply.get("text") or "").strip():
            content.append({"type": "text", "text": str(reply["text"])})
        for c in reply.get("tool_calls") or []:
            if not (isinstance(c, dict) and c.get("name")):
                continue
            args, _problem = _call_input(c)      # unreadable arguments become {}; the tool then says what is missing
            self._n += 1
            content.append({"type": "tool_use", "id": f"toolu_gemini_{self._n}", "name": str(c["name"]),
                            "input": args})
        u = payload.get("usageMetadata") or {}
        usage = SimpleNamespace(input_tokens=int(u.get("promptTokenCount") or 0),
                                cache_creation_input_tokens=0,
                                cache_read_input_tokens=0,
                                output_tokens=int(u.get("candidatesTokenCount") or 0)
                                + int(u.get("thoughtsTokenCount") or 0))
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(id=f"msg_gemini_{self._n}", type="message", role="assistant", model=model,
                               content=content, stop_reason=stop, stop_details=None, usage=usage)


class _Messages:
    def __init__(self, owner: GeminiClient):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._turn(kwargs)

    def stream(self, **kwargs):
        return _Stream(self._owner, kwargs)


class _Stream:
    def __init__(self, owner: GeminiClient, kwargs: dict):
        self._owner, self._kwargs, self._msg = owner, kwargs, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if self._msg is None:
            self._msg = self._owner._turn(self._kwargs)
        return self._msg


# --------------------------------------------------------------------------- the helper-script writer

class GeminiWriter:
    """`writer(system, prompt) -> text` for helpers.py: asks Gemini for one Python script."""

    def __init__(self, model: str | None = None, timeout: float = 300, transport=None, sleep=time.sleep):
        self.model = model or helper_model_name()
        self.timeout, self._transport, self._sleep = timeout, transport, sleep

    def __call__(self, system: str, prompt: str) -> str:
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": prompt}]}]}
        return _text_of(generate(self.model, body, self.timeout, self._transport, self._sleep))
