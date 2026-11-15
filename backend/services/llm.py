"""One interface for every language-model call, with interchangeable backends.

    run(prompt, schema=None, web=False, system=None, timeout=180, kind=None) -> dict | str | None
    available() -> bool        backend() -> "gemini" | "api" | "fake" | "off"        last_error() -> str | None

CAUSAL_LLM selects the backend:
    gemini  Google's Gemini API (free tier works): GEMINI_API_KEY or GOOGLE_API_KEY, in the environment or in .env.
            GEMINI_MODEL picks the model (default gemini-3.5-flash-lite), GEMINI_RPM paces requests per minute.
            Uses causality/orchestrator/gemini.py; `web` turns on Google Search grounding.
    api     Anthropic SDK (ANTHROPIC_API_KEY or an `ant auth login` profile). Kept as an optional fallback.
            CAUSAL_API_MODEL picks the model (default claude-opus-5-5).
    fake    deterministic canned answers for tests: no network, no quota.
    off     no language model; features that need one say so.
Unset (or any other value): `gemini` when a Gemini key is found, else `api` when ANTHROPIC_API_KEY is set, else `off`.
Headless Claude Code is never started from here.

The core analysis never calls a language model. Failures are never silent: run() returns None and last_error()
keeps the reason, which the API reports (v1 hid a usage-limit outage behind "Sorry…"). Error text never contains
an API key: keys travel in request headers only, and messages are built from status text.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from typing import Any, Optional

from causality.orchestrator import gemini

_state: dict = {"error": None, "at": None}
_lock = threading.Lock()

# Gemini transport and sleep, replaceable by tests (a callable (url, body, key, timeout) -> (status, payload)).
_gemini_transport = None
_gemini_sleep = time.sleep

NO_MODEL = "No language model is configured (set GEMINI_API_KEY in .env, or set CAUSAL_LLM)."


# --------------------------------------------------------------------------------------------- selection
def backend() -> str:
    gemini.load_env()   # so CAUSAL_LLM and the keys may also come from .env
    chosen = (os.environ.get("CAUSAL_LLM") or "").strip().lower()
    if chosen in ("gemini", "api", "fake", "off"):
        return chosen
    if gemini.available():
        return "gemini"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    return "off"


def available() -> bool:
    b = backend()
    if b == "gemini":
        return gemini.available()
    if b == "api":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True
    return b == "fake"


def api_model() -> str:
    return os.environ.get("CAUSAL_API_MODEL", "claude-opus-5-5")


def describe() -> str:
    return {"gemini": f"Google Gemini ({gemini.model_name()})", "api": f"Anthropic API ({api_model()})",
            "fake": "Fake language model (test mode)", "off": "No language model"}[backend()]


def last_error() -> Optional[str]:
    return _state["error"]


def _fail(msg: str) -> None:
    with _lock:
        _state["error"], _state["at"] = msg[:500], time.time()
    print(f"[llm] {msg[:500]}", flush=True)  # never includes keys: messages are built from status text only


def _ok() -> None:
    with _lock:
        _state["error"] = None


# --------------------------------------------------------------------------------------------- entry point
def run(prompt: str, schema: Optional[dict] = None, web: bool = False, system: Optional[str] = None,
        timeout: int = 180, kind: Optional[str] = None) -> Any:
    """Structured output (dict) when `schema` is given, else text. None on failure (see last_error()).
    `web` allows web search; `kind` only steers the fake backend."""
    if not available():
        _fail(NO_MODEL)
        return None
    from . import limits  # late: limits imports store, which must not import llm
    if not limits.spend_llm_call():
        _fail(limits.BUDGET_MESSAGE)
        return None
    try:
        b = backend()
        out = _fake(prompt, schema, kind) if b == "fake" else \
            _gemini(prompt, schema, web, system, timeout) if b == "gemini" else _api(prompt, schema, web, system, timeout)
    except gemini.GeminiError as e:
        _fail(str(e))
        return None
    except Exception as e:  # noqa: BLE001
        _fail(f"{type(e).__name__}: {e}")
        return None
    if out is not None:
        _ok()
    return out


# --------------------------------------------------------------------------------------------- gemini
def gemini_schema(schema: Any) -> Any:
    """A JSON schema in Gemini's responseSchema dialect: upper-case types, ["x", "null"] -> nullable,
    keys Gemini does not know (additionalProperties, ...) dropped."""
    if isinstance(schema, list):
        return [gemini_schema(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    t = schema.get("type")
    if isinstance(t, list):
        rest = [x for x in t if x != "null"]
        if "null" in t:
            out["nullable"] = True
        t = rest[0] if rest else "string"
    if isinstance(t, str):
        out["type"] = t.upper()
    for k in ("description", "enum", "format"):
        if k in schema:
            out[k] = schema[k]
    if isinstance(schema.get("properties"), dict):
        out["properties"] = {k: gemini_schema(v) for k, v in schema["properties"].items()}
        out["propertyOrdering"] = list(schema["properties"])
    if "items" in schema:
        out["items"] = gemini_schema(schema["items"])
    if "required" in schema:
        out["required"] = list(schema["required"])
    return out


def _json_from_text(text: str) -> Any:
    """The JSON object in a reply, tolerating a code fence or stray text around it. None if there is none."""
    s = text.strip()
    m = re.match(r"^```[a-zA-Z]*\s*(.*?)\s*```$", s, re.S)
    if m:
        s = m.group(1)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if 0 <= a < b:
            try:
                return json.loads(s[a:b + 1])
            except json.JSONDecodeError:
                return None
    return None


def _deadline_transport(timeout: float):
    """The Gemini transport, bounded so retries cannot outlast `timeout` seconds in total."""
    post = _gemini_transport or gemini._http_post
    end = time.monotonic() + float(timeout)

    def call(url, body, key, _per_call_timeout):
        left = end - time.monotonic()
        if left <= 1:
            raise gemini.GeminiError(f"Gemini did not answer within {int(timeout)} s.")
        return post(url, body, key, left)
    return call


def _gemini(prompt, schema, web, system, timeout):
    sys_text = system or ""
    gen: dict = {}
    body: dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    if web:
        # Google Search grounding. JSON response mode is not combined with search (not every model accepts both),
        # so the schema goes in the instructions and the JSON is read from the text.
        body["tools"] = [{"google_search": {}}]
        if schema:
            sys_text += ("\n\nReply with only one JSON object (no code fence, no other text) that follows this JSON "
                         "schema exactly, every required key present:\n" + json.dumps(schema))
    elif schema:
        gen = {"responseMimeType": "application/json", "responseSchema": gemini_schema(schema)}
    if sys_text.strip():
        body["systemInstruction"] = {"parts": [{"text": sys_text.strip()}]}
    if gen:
        body["generationConfig"] = gen
    payload = gemini.generate(gemini.web_model_name() if web else gemini.model_name(), body, timeout=timeout, transport=_deadline_transport(timeout),
                              sleep=_gemini_sleep)
    text = gemini._text_of(payload)
    if not schema:
        return text.strip()
    out = _json_from_text(text)
    if not isinstance(out, dict):
        _fail("Gemini's structured answer was not valid JSON.")
        return None
    return out


# --------------------------------------------------------------------------------------------- api
def _strict(schema: Any) -> Any:
    """Structured outputs need every object closed (additionalProperties: false) with all properties required."""
    if isinstance(schema, list):
        return [_strict(x) for x in schema]
    if not isinstance(schema, dict):
        return schema
    s = {k: (_strict(v) if k != "properties" else {pk: _strict(pv) for pk, pv in v.items()}) for k, v in schema.items()}
    t = s.get("type")
    if t == "object" or (isinstance(t, list) and "object" in t):
        s.setdefault("additionalProperties", False)
        if "properties" in s:
            s["required"] = list(s["properties"])
    return s


def _api(prompt, schema, web, system, timeout):
    import anthropic

    client = anthropic.Anthropic(timeout=float(timeout), max_retries=2)
    kwargs: dict = {"model": api_model(), "max_tokens": 16000, "messages": [{"role": "user", "content": prompt}],
                    "output_config": {"effort": "medium"},
                    # a safety refusal is retried on the default fallback model instead of failing the request
                    "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
    if system:
        kwargs["system"] = system
    if schema:
        kwargs["output_config"]["format"] = {"type": "json_schema", "schema": _strict(schema)}
    if web:
        kwargs["tools"] = [{"type": "web_search_20260209", "name": "web_search", "max_uses": 4}]
    try:
        resp = client.beta.messages.create(**kwargs)
        for _ in range(5):  # the server-side tool loop hit its iteration limit: let it resume
            if resp.stop_reason != "pause_turn":
                break
            kwargs["messages"] = [kwargs["messages"][0], {"role": "assistant", "content": resp.content}]
            resp = client.beta.messages.create(**kwargs)
    except anthropic.AuthenticationError:
        _fail("The Anthropic API rejected the credentials (check ANTHROPIC_API_KEY).")
        return None
    except anthropic.RateLimitError:
        _fail("The Anthropic API rate limit was reached. Try again in a minute.")
        return None
    except anthropic.APIStatusError as e:
        _fail(f"Anthropic API error {e.status_code}: {getattr(e, 'message', e)}")
        return None
    except anthropic.APIConnectionError:
        _fail("Could not reach the Anthropic API (network error).")
        return None
    if resp.stop_reason == "refusal":
        cat = getattr(getattr(resp, "stop_details", None), "category", None)
        _fail(f"Claude declined this request{f' ({cat})' if cat else ''}.")
        return None
    texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
    if not texts:
        _fail(f"The model returned no text (stop reason: {resp.stop_reason}).")
        return None
    if not schema:
        return "\n".join(texts).strip()
    for candidate in (texts[-1], "".join(texts)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    _fail("The model's structured answer was not valid JSON.")
    return None


# --------------------------------------------------------------------------------------------- fake
def _fake(prompt: str, schema: Optional[dict], kind: Optional[str]):
    """Deterministic answers keyed by `kind`."""
    if kind == "investigate":
        import re
        time.sleep(float(os.environ.get("CAUSAL_FAKE_DELAY", "4")))  # a search takes a while: lets tests see the queue
        m = re.search(r"Investigate the variable '([^']+)'", prompt)
        target = m.group(1) if m else "the target"
        return {"summary": f"Fake search: published research links a few factors to {target}.",
                "factors": [
                    {"name": f"Motivation ({target})"[:40], "affects": ["target", "outcome"],
                     "explanation": "Unmeasured motivation raises both take-up and later outcomes.", "dataset_column": None,
                     "sources": [{"title": "Using Randomization in Development Economics Research: A Toolkit", "year": 2007,
                                  "url": "https://doi.org/10.1016/S1573-4471(07)04061-2",
                                  "doi": "10.1016/S1573-4471(07)04061-2"}]},
                    {"name": "Pre-program earnings dip", "affects": ["target", "treatment"],
                     "explanation": "People often enrol after a temporary drop in earnings (Ashenfelter's dip).",
                     "dataset_column": None,
                     "sources": [{"title": "Estimating the Effect of Training Programs on Earnings", "year": 1978,
                                  "url": "https://www.jstor.org/stable/1924332", "doi": None},
                                 {"title": "A working paper on program entry", "year": 2001,
                                  "url": "https://www.nber.org/papers/w8000", "doi": None}]},
                    {"name": "Age", "affects": ["target"], "explanation": "Older people enrol less often.",
                     "dataset_column": "age",
                     "sources": [{"title": "A labour economics web page", "year": None, "url": "https://example.com/age",
                                  "doi": None}]},
                    {"name": "Unsourced hunch", "affects": ["outcome"], "explanation": "No source: the server drops it.",
                     "dataset_column": None, "sources": [{"title": "x", "year": None, "url": "not a url", "doi": None}]},
                ]}
    if kind == "chat":
        import re
        q = prompt.rsplit("USER:", 1)[-1].split("\n\nRespond with", 1)[0].lower()
        ids = re.search(r"Valid node ids: \[([^\]]*)\]", prompt)
        nodes = [x.strip().strip("'\"") for x in ids.group(1).split(",")] if ids and ids.group(1).strip() else []
        target = next((n for n in nodes if n in ("treat", "treated", "program")), nodes[0] if nodes else None)
        if "investigat" in q and target:
            return {"reply": f"I've started a literature search on what drives '{target}'. It runs in the background "
                             "(30 s to 2 min); its findings will be hypothetical leads from published research, not "
                             "results from your data.", "cards": [], "investigate": target}
        if "support" in q or "estimate" in q:
            m = re.search(r'"estimate": (-?[0-9.]+)', prompt)
            return {"reply": f"Fake Investigator: the estimate is {m.group(1) if m else 'not available'}.",
                    "cards": [{"kind": "estimate", "ref": None}, {"kind": "check", "ref": "no_such_check"}],
                    "investigate": None}
        if "problem" in q:
            return {"reply": "Fake Investigator: look at the balance check.", "cards": [{"kind": "check", "ref": "balance"}],
                    "investigate": None}
        return {"reply": "Fake Investigator: ask about the estimate, a check, or start an investigation.",
                "cards": [{"kind": "graph", "ref": None}], "investigate": None}
    if kind == "summary":
        return "Fake executive summary: the analysis ran end to end in test mode."
    if kind == "probe":
        return "ok"
    if kind == "claim_only":
        return {"treatment": "job training", "outcome": "earnings", "population": None,
                "summary": "Fake: randomized studies find modest positive effects; observational ones disagree.",
                "evidence": [{"finding": "A randomized trial found earnings gains.", "direction": "supports",
                              "sources": [{"title": "Evaluating the Econometric Evaluations of Training Programs",
                                           "year": 1986, "url": "https://www.jstor.org/stable/1806062", "doi": None}]},
                             {"finding": "An unsourced claim (dropped by the server).", "direction": "mixed",
                              "sources": [{"title": "x", "year": None, "url": "not-a-url", "doi": None}]}],
                "confounders": [{"name": "Motivation", "why": "Drives both enrolment and earnings."}],
                "designs": [{"design": "Randomized experiment", "why": "Removes selection on motivation."}],
                "data_needed": ["Treatment indicator", "Earnings after the program", "Earnings before it"]}
    return {} if schema else "ok"
