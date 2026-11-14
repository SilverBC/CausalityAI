"""One interface for every language-model call, with interchangeable backends.

    run(prompt, schema=None, web=False, system=None, timeout=180, kind=None) -> dict | str | None
    available() -> bool        backend() -> "cli" | "api" | "fake" | "off"        last_error() -> str | None

CAUSAL_LLM selects the backend:
    cli   headless Claude Code (`claude -p`) on the signed-in subscription: no API key, but it shares the developer's
          usage limits. Binary from CLAUDE_BIN, PATH, ~/.local/bin or the VS Code / Cursor extension bundle.
          CAUSAL_CLAUDE_MODEL picks the model.
    api   Anthropic SDK (ANTHROPIC_API_KEY or an `ant auth login` profile). Needed for any public deploy.
          CAUSAL_API_MODEL picks the model (default claude-opus-5-5).
    fake  deterministic canned answers for tests: no network, no quota.
    off   no language model; features that need one say so.
Unset: `cli` when a Claude Code binary is found, else `api` when ANTHROPIC_API_KEY is set, else `off`.

The core analysis never calls a language model. Failures are never silent: run() returns None and last_error()
keeps the reason, which the API reports (v1 hid a usage-limit outage behind "Sorry…").
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

_state: dict = {"error": None, "at": None}
_lock = threading.Lock()


# --------------------------------------------------------------------------------------------- selection
def claude_bin() -> Optional[str]:
    exe = "claude.exe" if os.name == "nt" else "claude"
    bundled = sorted(
        (p for ext in (".vscode", ".cursor") for p in
         (Path.home() / ext / "extensions").glob(f"anthropic.claude-code-*/resources/native-binary/{exe}")),
        key=lambda p: p.stat().st_mtime, reverse=True)
    for c in [os.environ.get("CLAUDE_BIN"), shutil.which("claude"), str(Path.home() / ".local" / "bin" / exe),
              *map(str, bundled)]:
        if c and Path(c).exists():
            return c
    return None


def backend() -> str:
    chosen = (os.environ.get("CAUSAL_LLM") or "").strip().lower()
    if chosen in ("cli", "api", "fake", "off"):
        return chosen
    if claude_bin():
        return "cli"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "api"
    return "off"


def available() -> bool:
    b = backend()
    if b == "cli":
        return claude_bin() is not None
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
    return {"cli": "Claude Code (headless claude -p)", "api": f"Anthropic API ({api_model()})",
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
        _fail("No language model is configured (install Claude Code, set ANTHROPIC_API_KEY, or set CAUSAL_LLM).")
        return None
    try:
        b = backend()
        out = _fake(prompt, schema, kind) if b == "fake" else \
            _cli(prompt, schema, web, system, timeout) if b == "cli" else _api(prompt, schema, web, system, timeout)
    except subprocess.TimeoutExpired:
        _fail(f"Claude did not answer within {timeout} s.")
        return None
    except Exception as e:  # noqa: BLE001
        _fail(f"{type(e).__name__}: {e}")
        return None
    if out is not None:
        _ok()
    return out


# --------------------------------------------------------------------------------------------- cli
_DROP_ENV = {"CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT", "CLAUDE_AGENT_SDK_VERSION", "CLAUDE_AGENT_SDK_MCP_NO_PREFIX",
             "CLAUDE_PREVIEW_CLASSIFIER_FLOOR", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"}


def _clean_env() -> dict:
    """Drop API keys and parent-session variables so the child CLI uses the logged-in subscription.
    CAUSAL_KEEP_API_KEY=1 passes ANTHROPIC_API_KEY through instead."""
    env = dict(os.environ)
    for k in list(env):
        if k.startswith("CLAUDE_CODE_") or k in _DROP_ENV:
            env.pop(k)
        elif k == "ANTHROPIC_API_KEY" and not env.get("CAUSAL_KEEP_API_KEY"):
            env.pop(k)
    return env


def _cli(prompt, schema, web, system, timeout):
    tools = ["WebSearch", "WebFetch"] if web else []
    cmd = [claude_bin(), "-p", prompt, "--output-format", "json", "--no-session-persistence", "--tools", ",".join(tools)]
    if tools:
        cmd += ["--allowedTools", ",".join(tools)]
    if schema:
        cmd += ["--json-schema", json.dumps(schema)]
    if system:
        cmd += ["--append-system-prompt", system]
    if os.environ.get("CAUSAL_CLAUDE_MODEL"):
        cmd += ["--model", os.environ["CAUSAL_CLAUDE_MODEL"]]
    # run outside the project so no project CLAUDE.md or settings are picked up
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
                          cwd=tempfile.gettempdir(), stdin=subprocess.DEVNULL, env=_clean_env())
    data: dict = {}
    lines = proc.stdout.strip().splitlines()
    if lines:
        try:
            data = json.loads(lines[-1])
        except json.JSONDecodeError:
            data = {}
    if proc.returncode != 0 and not data:
        _fail(f"Claude Code exited with {proc.returncode}: {(proc.stderr or proc.stdout)[-400:]}")
        return None
    if data.get("is_error"):
        _fail(str(data.get("result"))[:400])
        return None
    if not schema:
        return data.get("result")
    out = data.get("structured_output")
    if out is None and data.get("result"):
        try:
            out = json.loads(data["result"])
        except json.JSONDecodeError:
            out = None
    if out is None:
        _fail("Claude Code returned no structured answer.")
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
    """Deterministic answers keyed by `kind`. Later prompts add the chat case."""
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
