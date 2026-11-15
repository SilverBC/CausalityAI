"""A model client that runs each turn through headless Claude Code (`claude -p`), so the audit
loop can run on a Claude subscription with no API key.

    from causality.orchestrator import loop, headless
    report = loop.run_audit(df, claim, context, client=headless.HeadlessClient())

It has the same surface the loop uses on the Anthropic SDK client
(`client.beta.messages.stream(**kwargs)` -> `.get_final_message()`), so nothing else changes:
the same system prompt, the same tools, the same checks on the submitted verdict, the same
floor rules.

How a turn works: Claude Code is started with every built-in tool switched off. The
conversation so far and the tool definitions are sent as text, and the model replies through a
fixed JSON shape: {"text": ..., "tool_calls": [{"name", "input"}]}. The loop then runs the
calls and sends the results in the next turn. Each turn is a fresh process, so the model's
private reasoning is not carried between turns (on the SDK client it is).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "input": {"type": "object"}},
            "required": ["name", "input"]}},
    },
    "required": ["text", "tool_calls"],
}

PROTOCOL = """

# How to call tools in this session

You cannot call tools natively here. Reply only through the structured output, which has two fields:
- `text`: what you want to say (may be an empty string).
- `tool_calls`: a list of calls, each {"name": <tool name>, "input": <arguments object>}.

You may put several calls in one reply; their results come back together in the next message,
each marked with the call's id. A reply with an empty `tool_calls` list ends the audit, so
always finish by calling `submit_verdict` (or `ask_user`). Never write a tool result yourself.

The tools:

"""


def find_claude() -> str | None:
    """The Claude Code executable: $CLAUDE_CODE_BIN, else the newest copy shipped with the desktop
    app, else `claude` on PATH."""
    env = os.environ.get("CLAUDE_CODE_BIN")
    if env and os.path.exists(env):
        return env
    appdata = os.environ.get("APPDATA")
    if appdata:
        found = glob.glob(os.path.join(appdata, "Claude", "claude-code", "*", "*", "claude.exe"))

        def version(p):
            v = os.path.basename(os.path.dirname(os.path.dirname(p)))
            return tuple(int(x) if x.isdigit() else 0 for x in v.split("."))
        if found:
            return max(found, key=lambda p: (version(p), os.path.getmtime(p)))
    return shutil.which("claude")


def _render_block(b: dict) -> str:
    t = b.get("type")
    if t == "text":
        return b.get("text") or ""
    if t == "tool_use":
        return f'<tool_call id="{b.get("id")}" name="{b.get("name")}">{json.dumps(b.get("input") or {}, ensure_ascii=False)}</tool_call>'
    if t == "tool_result":
        c = b.get("content")
        if isinstance(c, list):
            c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
        err = ' error="true"' if b.get("is_error") else ""
        return f'<tool_result id="{b.get("tool_use_id")}"{err}>\n{c or ""}\n</tool_result>'
    return ""          # thinking and anything else is not replayed


def render_messages(messages: list) -> str:
    parts = []
    for m in messages:
        c = m.get("content")
        body = c if isinstance(c, str) else "\n".join(x for x in (_render_block(b) for b in c or []) if x)
        parts.append(f"=== {m.get('role', 'user').upper()} ===\n{body}")
    parts.append("=== YOUR NEXT REPLY (structured output only) ===")
    return "\n\n".join(parts)


class HeadlessClient:
    def __init__(self, claude_bin: str | None = None, timeout: int = 900):
        self.claude_bin = claude_bin or find_claude()
        self.timeout = timeout
        self.cost_usd = 0.0
        self._n = 0
        self.messages = _Messages(self)
        self.beta = SimpleNamespace(messages=self.messages)

    def _turn(self, kwargs: dict):
        if not self.claude_bin:
            raise RuntimeError("Claude Code was not found. Install it, or set CLAUDE_CODE_BIN to its path.")
        system = (kwargs.get("system") or "") + PROTOCOL + json.dumps(kwargs.get("tools") or [], indent=1)
        work = tempfile.mkdtemp(prefix="causalityai_headless_")
        try:
            sys_path = os.path.join(work, "system.txt")
            with open(sys_path, "w", encoding="utf-8") as f:
                f.write(system)
            cmd = [self.claude_bin, "-p", "--model", kwargs.get("model") or "claude-opus-5-5", "--tools", "",
                   "--strict-mcp-config", "--system-prompt-file", sys_path, "--output-format", "json",
                   "--json-schema", json.dumps(REPLY_SCHEMA)]
            env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
            p = subprocess.run(cmd, input=render_messages(kwargs.get("messages") or []), capture_output=True,
                               text=True, encoding="utf-8", errors="replace", cwd=work, timeout=self.timeout, env=env)
        finally:
            shutil.rmtree(work, ignore_errors=True)
        try:
            out = json.loads(p.stdout)
        except Exception:  # noqa: BLE001
            raise RuntimeError(f"Claude Code gave no readable reply (exit {p.returncode}): "
                               f"{(p.stderr or p.stdout or '')[:300]}") from None
        if out.get("is_error"):
            raise RuntimeError(f"Claude Code reported an error: {str(out.get('result'))[:300]}")
        reply = out.get("structured_output")
        if not isinstance(reply, dict):
            try:
                reply = json.loads(out.get("result") or "")
            except Exception:  # noqa: BLE001
                reply = {"text": str(out.get("result") or ""), "tool_calls": []}
        self.cost_usd += float(out.get("total_cost_usd") or 0)
        content = []
        if (reply.get("text") or "").strip():
            content.append({"type": "text", "text": reply["text"]})
        for c in reply.get("tool_calls") or []:
            if isinstance(c, dict) and c.get("name"):
                self._n += 1
                content.append({"type": "tool_use", "id": f"toolu_headless_{self._n}", "name": str(c["name"]),
                                "input": c.get("input") if isinstance(c.get("input"), dict) else {}})
        u = out.get("usage") or {}
        usage = SimpleNamespace(input_tokens=int(u.get("input_tokens") or 0),
                                cache_creation_input_tokens=int(u.get("cache_creation_input_tokens") or 0),
                                cache_read_input_tokens=int(u.get("cache_read_input_tokens") or 0),
                                output_tokens=int(u.get("output_tokens") or 0))
        stop = "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn"
        return SimpleNamespace(id=f"msg_headless_{self._n}", type="message", role="assistant",
                               model=kwargs.get("model"), content=content, stop_reason=stop, stop_details=None,
                               usage=usage)


class _Messages:
    def __init__(self, owner: HeadlessClient):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._turn(kwargs)

    def stream(self, **kwargs):
        return _Stream(self._owner, kwargs)


class _Stream:
    def __init__(self, owner: HeadlessClient, kwargs: dict):
        self._owner, self._kwargs, self._msg = owner, kwargs, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if self._msg is None:
            self._msg = self._owner._turn(self._kwargs)
        return self._msg
