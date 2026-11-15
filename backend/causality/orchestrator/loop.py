"""The model loop: Opus reads the claim, calls the evidence tools, may ask the user, and submits a
verdict that code then holds to the floor rules. See ORCHESTRATOR.md, part C.

  run_audit(df, claim, context="", client=None, model=contract.MODEL, max_turns=24, helper_writer=None)
  resume(df, state, answers, client=None, max_turns=24, helper_writer=None)
  followup(df, state, question, client=None, max_turns=16, helper_writer=None)
  FakeClient(script)   replays scripted assistant turns; no key or network needed

Nothing here raises: failures come back as a report with status "error" and a plain message.
evidence.py and floor.py are imported lazily inside functions so this module loads on its own.
layout.py, helpers.py and screen.py (ORCHESTRATOR_MERGED.md, F G H) are optional: a part that does
not import is simply not offered to the model.
"""
from __future__ import annotations

import copy
import itertools
import json
import re
from types import SimpleNamespace

from . import contract, prompt

MAX_TOKENS = 64000                                  # streamed, so a large cap is safe
EFFORT = "high"                                     # Opus 5.5 defaults to medium; this work needs more
FALLBACK_BETA = "server-side-fallback-2026-07-01"   # re-runs a policy decline on another model
STATE_VERSION = 1
DIRECTIONS = ("increase", "decrease", "none")

_ID_RE = re.compile(r"\be\d+\b")


# --------------------------------------------------------------------------- control tool specs

def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


_ROLES_SCHEMA = {
    "type": "object",
    "description": "Column names for each role this design uses; null for roles that do not apply. "
                   "cutoff is the threshold on the running column for rd, else null.",
    "properties": {**{k: _nullable({"type": "string"}) for k in contract.ROLE_KEYS},
                   "cutoff": _nullable({"type": "number"})},
    "required": [*contract.ROLE_KEYS, "cutoff"],
    "additionalProperties": False,
}

ASK_USER_SPEC = {
    "name": contract.ASK_USER,
    "description": (
        "Ask the user what the data cannot show, above all how treatment was assigned. The audit "
        "pauses until the user answers; the answers come back as this tool's result. Ask a few short "
        "questions, each with concrete options the user can pick from (they may also answer freely)."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Short id, for example q1."},
                        "question": {"type": "string", "description": "One plain question."},
                        "options": {"type": "array", "items": {"type": "string"},
                                    "description": "Suggested answers."},
                    },
                    "required": ["id", "question", "options"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    },
}

SUBMIT_VERDICT_SPEC = {
    "name": contract.SUBMIT_VERDICT,
    "description": (
        "Submit your reading of the claim and your verdict. This ends the audit. Code then applies "
        "the three floor rules to the label. Cite evidence ids like [e4] in reasons and caveats; "
        "every evidence id you cite must exist."),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "reading": {
                "type": "object",
                "properties": {
                    "treatment": {"type": "string", "description": "Treatment column."},
                    "outcome": {"type": "string", "description": "Outcome column."},
                    "design": {"type": "string", "enum": list(contract.DESIGNS)},
                    "design_reason": {"type": "string",
                                      "description": "Why this design, in plain words, citing the "
                                                     "description, the user's answers or evidence ids."},
                    "roles": _ROLES_SCHEMA,
                    "covariates": _nullable({"type": "array", "items": {"type": "string"}}),
                    "excluded": {
                        "type": "array",
                        "description": "Columns left out of the adjustment on purpose, with the reason.",
                        "items": {
                            "type": "object",
                            "properties": {"column": {"type": "string"}, "reason": {"type": "string"}},
                            "required": ["column", "reason"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["treatment", "outcome", "design", "design_reason", "roles", "covariates",
                             "excluded"],
                "additionalProperties": False,
            },
            "verdict": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": list(contract.LABELS)},
                    "sentence": {"type": "string",
                                 "description": "One plain sentence a non-statistician can read."},
                    "estimate_evidence_id": _nullable({"type": "string"}),
                    "claimed_direction": {"type": "string", "enum": list(DIRECTIONS)},
                    "reasons": {"type": "array", "items": {"type": "string"}},
                    "assumed": {"type": "array", "items": {"type": "string"},
                                "description": "What the answer leans on that no data can test."},
                    "caveats": {"type": "array", "items": {"type": "string"}},
                    "judgments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "evidence_id": {"type": "string"},
                                "reading": {"type": "string", "enum": list(contract.READINGS)},
                                "why": {"type": "string"},
                            },
                            "required": ["evidence_id", "reading", "why"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["label", "sentence", "estimate_evidence_id", "claimed_direction", "reasons",
                             "assumed", "caveats", "judgments"],
                "additionalProperties": False,
            },
        },
        "required": ["reading", "verdict"],
        "additionalProperties": False,
    },
}


# Optional parts: tool name -> module name. A part that cannot be imported is simply not offered.
_OPTIONAL = ((contract.LAYOUT_FACTS, "layout"), (contract.CUSTOM_CHECKS, "helpers"), (contract.SHOW_CHART, "screen"))


def _optional_parts() -> dict:
    """{tool name: module} for the optional parts that import and expose a TOOL_SPEC dict."""
    import importlib
    parts = {}
    for name, mod in _OPTIONAL:
        try:
            m = importlib.import_module(f"{__package__}.{mod}")
        except Exception:
            continue
        if isinstance(getattr(m, "TOOL_SPEC", None), dict):
            parts[name] = m
    return parts


def _tools(evidence, parts: dict | None = None) -> list[dict]:
    specs = copy.deepcopy(list(getattr(evidence, "TOOL_SPECS", []) or []))
    for name, _ in _OPTIONAL:
        if parts and name in parts:
            spec = copy.deepcopy(parts[name].TOOL_SPEC)
            spec["name"] = name
            specs.append(spec)
    return specs + [copy.deepcopy(ASK_USER_SPEC), copy.deepcopy(SUBMIT_VERDICT_SPEC)]


# --------------------------------------------------------------------------- small helpers

def _get(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _plain(value):
    """Force a value into plain JSON types."""
    return json.loads(json.dumps(value, default=str))


def _block_dict(block) -> dict:
    """An SDK content block (or a dict) as a plain dict, with the wire field names, unmodified."""
    if isinstance(block, dict):
        return _plain(block)
    to_dict = getattr(block, "to_dict", None)
    if callable(to_dict):
        try:
            return _plain(to_dict(mode="json"))
        except TypeError:
            return _plain(to_dict())
    dump = getattr(block, "model_dump", None)
    if callable(dump):
        return _plain(dump(mode="json", by_alias=True, exclude_unset=True))
    return _plain(vars(block))


def _echo_blocks(blocks: list[dict]) -> list[dict]:
    """Blocks to send back on the next request. Unchanged, except after a server-side fallback:
    model-internal blocks from the declined attempt (before the last `fallback` block) are dropped,
    and the `fallback` block stays where it was."""
    last = max((i for i, b in enumerate(blocks) if b.get("type") == "fallback"), default=-1)
    if last < 0:
        return blocks
    keep_types = {"text", "fallback"}
    server_results = {b.get("tool_use_id") for b in blocks[:last] if b.get("type", "").endswith("_tool_result")}
    out = []
    for i, b in enumerate(blocks):
        t = b.get("type", "")
        if i < last and t not in keep_types:
            paired_use = t == "server_tool_use" and b.get("id") in server_results
            paired_result = t.endswith("_tool_result") and t != "tool_result"
            if not (paired_use or paired_result):
                continue
        out.append(b)
    return out


def _tool_result(tool_use_id: str, text: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": tool_use_id, "content": text}
    if is_error:
        block["is_error"] = True
    return block


def _evidence_text(item: dict) -> tuple[str, bool]:
    if item.get("ok"):
        lines = [str(x) for x in item.get("lines") or []]
        return f"[{item['id']}] " + ("\n".join(lines) if lines else "(the tool returned no lines)"), False
    return f"[{item['id']}] " + (item.get("error") or "The tool failed without saying why."), True


def _api_error_text(exc: Exception) -> str:
    try:
        import anthropic
    except Exception:  # pragma: no cover - the package is optional at import time
        anthropic = None
    if anthropic is not None:
        if isinstance(exc, anthropic.AuthenticationError):
            return "The Claude API did not accept the API key. Check ANTHROPIC_API_KEY."
        if isinstance(exc, anthropic.PermissionDeniedError):
            return "The API key does not have access to this model."
        if isinstance(exc, anthropic.NotFoundError):
            return "The Claude API did not find the model. Check the model name."
        if isinstance(exc, anthropic.RateLimitError):
            return "The Claude API is rate limiting this key. Wait a minute and try again."
        if isinstance(exc, anthropic.BadRequestError):
            return f"The Claude API rejected the request: {getattr(exc, 'message', exc)}"
        if isinstance(exc, anthropic.APIStatusError):
            code = getattr(exc, "status_code", None)
            if code and code >= 500:
                return f"The Claude API had a server problem (status {code}). Try again later."
            return f"The Claude API returned an error (status {code}): {getattr(exc, 'message', exc)}"
        if isinstance(exc, anthropic.APITimeoutError):
            return "The call to the Claude API timed out. Try again."
        if isinstance(exc, anthropic.APIConnectionError):
            return "Could not reach the Claude API. Check the network connection."
    text = str(exc) or type(exc).__name__
    if any(w in text.lower() for w in ("api_key", "api key", "auth", "credential")):
        return f"No usable Claude API credentials were found ({text}). Set ANTHROPIC_API_KEY."
    return f"The model call failed: {type(exc).__name__}: {text}"


def _make_client():
    """(client, error message)."""
    try:
        import anthropic
    except Exception:
        return None, "The anthropic package is not installed. Run: python -m pip install anthropic"
    try:
        return anthropic.Anthropic(), None
    except Exception as exc:
        return None, _api_error_text(exc)


def _add_usage(usage: dict, msg) -> None:
    usage["turns"] += 1
    u = _get(msg, "usage")
    if u is None:
        return
    parts = _get(u, "iterations") or [u]      # with a fallback, iterations hold every attempt

    def n(obj, key):
        v = _get(obj, key, 0)
        return int(v) if isinstance(v, (int, float)) else 0

    for p in parts:
        usage["input_tokens"] += (n(p, "input_tokens") + n(p, "cache_creation_input_tokens")
                                  + n(p, "cache_read_input_tokens"))
        usage["output_tokens"] += n(p, "output_tokens")


def _call_model(client, state: dict, tools: list[dict]):
    kwargs = dict(
        model=state["model"],
        max_tokens=MAX_TOKENS,
        system=prompt.SYSTEM_PROMPT,
        tools=tools,
        messages=state["messages"],
        thinking={"type": "adaptive"},
        output_config={"effort": EFFORT},
        cache_control={"type": "ephemeral"},
        betas=[FALLBACK_BETA],
        fallbacks="default",
    )
    with client.beta.messages.stream(**kwargs) as stream:
        return stream.get_final_message()


# --------------------------------------------------------------------------- reports and state

def _report_from(state: dict) -> dict:
    report = contract.empty_report(state.get("claim", ""), state.get("context", ""))
    report["evidence"] = state["evidence"]
    report["transcript"] = state["transcript"]
    report["usage"] = state["usage"]
    report["screen"] = state.get("screen") or []
    report["followups"] = state.get("followups") or []
    # After a verdict, the last accepted verdict stays in every later report (a follow-up that fails
    # leaves it where it was).
    for key in ("reading", "draft_verdict", "verdict", "floor"):
        if state.get(key) is not None:
            report[key] = state[key]
    return report


def _error(state: dict, message: str) -> dict:
    report = _report_from(state)
    report["status"] = "error"
    report["error"] = message
    return _plain(report)


def _new_state(claim: str, context: str, model: str) -> dict:
    return {
        "version": STATE_VERSION, "claim": claim or "", "context": context or "", "model": model,
        "messages": [], "evidence": [], "transcript": [],
        "usage": {"model": model, "turns": 0, "input_tokens": 0, "output_tokens": 0},
        "next_id": 1, "invalid_submits": 0,
        "pending_results": [], "asks": [],
        "screen": [], "followups": [], "finished": False,
    }


def _next_id(state: dict) -> str:
    eid = f"e{state['next_id']}"
    state["next_id"] += 1
    return eid


def _run_evidence(evidence, name: str, args, df, state: dict) -> dict:
    eid = _next_id(state)
    args = args if isinstance(args, dict) else {}
    try:
        item = evidence.call(name, args, df, eid)
        if not isinstance(item, dict):
            raise TypeError("evidence.call did not return a dict")
        item = _plain(item)
        item["id"] = eid
    except Exception as exc:
        item = contract.evidence_item(eid, name, _plain(args), ok=False,
                                      error=f"The {name} tool failed: {type(exc).__name__}: {exc}")
    state["evidence"].append(item)
    return item


# --------------------------------------------------------------------------- optional tools

def _run_layout(layout, args, df, state: dict) -> dict:
    """layout_facts -> one evidence item."""
    eid = _next_id(state)
    args = args if isinstance(args, dict) else {}
    try:
        lines, numbers = layout.run(df, dict(args))
        item = contract.evidence_item(eid, contract.LAYOUT_FACTS, _plain(args), lines=[str(x) for x in lines or []],
                                      numbers=_plain(numbers or {}))
    except ValueError as exc:
        item = contract.evidence_item(eid, contract.LAYOUT_FACTS, _plain(args), ok=False, error=str(exc))
    except Exception as exc:
        item = contract.evidence_item(eid, contract.LAYOUT_FACTS, _plain(args), ok=False,
                                      error=f"The layout_facts tool failed: {type(exc).__name__}: {exc}")
    state["evidence"].append(item)
    return item


def _run_custom_checks(helpers, args, df, state: dict, writer) -> tuple[str, bool]:
    """custom_checks -> one evidence item per question (tool "custom_check"). Returns the tool result text."""
    raw = _get(args, "questions") if isinstance(args, dict) else None
    questions = []
    for i, q in enumerate(raw if isinstance(raw, list) else [], 1):
        if isinstance(q, dict) and isinstance(q.get("question"), str) and q["question"].strip():
            qid = q.get("id") if isinstance(q.get("id"), str) and q.get("id") else f"q{i}"
            questions.append({"id": qid, "question": q["question"].strip()})
    if not 1 <= len(questions) <= 4:
        return "Give between 1 and 4 questions, each with an id and a question. Nothing was run.", True
    try:
        results = helpers.run(df, {"questions": questions}, writer=writer)
        results = list(results) if isinstance(results, (list, tuple)) else []
    except Exception as exc:
        results = [{"ok": False, "error": f"The custom checks failed: {type(exc).__name__}: {exc}"}] * len(questions)
    out, all_failed = [], True
    for q, r in itertools.zip_longest(questions, results[:len(questions)]):
        r = r if isinstance(r, dict) else {"ok": False, "error": "No result came back for this question."}
        eid = _next_id(state)
        numbers = dict(r.get("numbers") or {}) if isinstance(r.get("numbers"), dict) else {}
        numbers["code"] = r.get("code") if isinstance(r.get("code"), str) else ""
        numbers["seconds"] = r.get("seconds") if isinstance(r.get("seconds"), (int, float)) else None
        ok = bool(r.get("ok"))
        error = None if ok else (r.get("error") or "The check did not produce a result.")
        item = contract.evidence_item(eid, "custom_check", {"question": q["question"]}, ok=ok,
                                      lines=[str(x) for x in r.get("lines") or []] if ok else [],
                                      numbers=_plain(numbers), error=error)
        state["evidence"].append(item)
        body, is_err = _evidence_text(item)
        all_failed = all_failed and is_err
        out.append(f"({q['id']}: {q['question']})\n{body}")
    return "\n\n".join(out), all_failed


def _run_show_chart(screen, args, df, state: dict) -> tuple[str, bool]:
    """show_chart -> entries in state["screen"]. Returns a short confirmation or the plain error."""
    args = args if isinstance(args, dict) else {}
    note = args.get("note") if isinstance(args.get("note"), str) else ""
    title = args.get("title") if isinstance(args.get("title"), str) else ""
    eid = args.get("evidence_id")
    try:
        if eid:
            item = next((e for e in state["evidence"] if e.get("id") == eid), None)
            if item is None:
                return f"There is no evidence item {eid}. Nothing was shown.", True
            charts = [c for c in (screen.charts_for(copy.deepcopy(item), df) or []) if isinstance(c, dict)]
            if not charts:
                return f"There is no standard chart for [{eid}] ({item.get('tool')}). Nothing was shown.", True
            if title and len(charts) == 1:
                charts[0]["title"] = title
        else:
            chart = screen.draw(df, dict(args))
            if not isinstance(chart, dict):
                raise TypeError("draw did not return a chart")
            charts = [chart]
    except ValueError as exc:
        return f"The chart was not drawn: {exc}", True
    except Exception as exc:
        return f"The chart was not drawn: {type(exc).__name__}: {exc}", True
    shown = []
    for c in charts:
        state.setdefault("screen", []).append({"evidence_id": eid or None, "chart": _plain(c), "note": note})
        shown.append(f"chart {len(state['screen'])} ({c.get('kind')}: {c.get('title') or 'untitled'})")
    return "Shown on the page: " + "; ".join(shown) + ".", False


# --------------------------------------------------------------------------- submit_verdict

def _str_list(value, field: str, problems: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        problems.append(f"{field} must be a list of strings.")
        return []
    return list(value)


def _check_submission(inp, evidence: list[dict], columns: set[str] | None):
    """(reading, draft_verdict, problems). Problems are plain sentences sent back to the model."""
    problems: list[str] = []
    if not isinstance(inp, dict):
        return None, None, ["The input must be an object with reading and verdict."]
    r, v = inp.get("reading"), inp.get("verdict")
    if not isinstance(r, dict):
        problems.append("reading is missing or not an object.")
        r = {}
    if not isinstance(v, dict):
        problems.append("verdict is missing or not an object.")
        v = {}
    ids = {e.get("id") for e in evidence}

    def col(value, field):
        if not isinstance(value, str) or not value:
            problems.append(f"{field} must be a column name.")
        elif columns is not None and value not in columns:
            problems.append(f"{field} '{value}' is not a column in the table.")

    col(r.get("treatment"), "reading.treatment")
    col(r.get("outcome"), "reading.outcome")
    design = r.get("design")
    if design not in contract.DESIGNS:
        problems.append(f"reading.design must be one of {', '.join(contract.DESIGNS)}.")
    roles_in = r.get("roles") if isinstance(r.get("roles"), dict) else {}
    if not isinstance(r.get("roles"), dict):
        problems.append("reading.roles must be an object.")
    roles = {}
    for k, val in roles_in.items():
        if val is None:
            continue
        if k == "cutoff":
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                problems.append("reading.roles.cutoff must be a number or null.")
            else:
                roles[k] = val
        elif k in contract.ROLE_KEYS:
            col(val, f"reading.roles.{k}")
            roles[k] = val
        else:
            problems.append(f"reading.roles has an unknown key '{k}'.")
    for k in ("treatment", "outcome"):
        if isinstance(r.get(k), str) and r.get(k):
            if k in roles and roles[k] != r[k]:
                problems.append(f"reading.roles.{k} ('{roles[k]}') differs from reading.{k} ('{r[k]}').")
            roles.setdefault(k, r[k])
    covariates = r.get("covariates")
    if covariates is not None:
        covariates = _str_list(covariates, "reading.covariates", problems)
        missing = [c for c in covariates if columns is not None and c not in columns]
        if missing:
            problems.append("reading.covariates names columns that are not in the table: " + ", ".join(missing) + ".")
    excluded = []
    for x in r.get("excluded") or []:
        if not isinstance(x, dict) or not isinstance(x.get("column"), str) or not isinstance(x.get("reason"), str):
            problems.append("each reading.excluded entry needs a column and a reason.")
            continue
        excluded.append({"column": x["column"], "reason": x["reason"]})
    reading = {"treatment": r.get("treatment"), "outcome": r.get("outcome"), "design": design,
               "design_reason": r.get("design_reason") if isinstance(r.get("design_reason"), str) else "",
               "roles": roles, "covariates": covariates, "excluded": excluded}

    draft = contract.empty_verdict()
    label = v.get("label")
    if label not in contract.LABELS:
        problems.append(f"verdict.label must be one of {', '.join(contract.LABELS)}.")
    sentence = v.get("sentence")
    if not isinstance(sentence, str) or not sentence.strip():
        problems.append("verdict.sentence must be a plain sentence.")
    est = v.get("estimate_evidence_id")
    if est is not None:
        if est not in ids:
            problems.append(f"verdict.estimate_evidence_id '{est}' is not an evidence id from this audit.")
        else:
            item = next(e for e in evidence if e.get("id") == est)
            if item.get("tool") != "estimate_effect":
                problems.append(f"verdict.estimate_evidence_id '{est}' is a {item.get('tool')} item, "
                                "not an estimate_effect item.")
    direction = v.get("claimed_direction")
    if direction not in DIRECTIONS:
        problems.append("verdict.claimed_direction must be increase, decrease or none.")
    reasons = _str_list(v.get("reasons"), "verdict.reasons", problems)
    assumed = _str_list(v.get("assumed"), "verdict.assumed", problems)
    caveats = _str_list(v.get("caveats"), "verdict.caveats", problems)
    judgments = []
    for j in v.get("judgments") or []:
        if not isinstance(j, dict):
            problems.append("each judgment must be an object.")
            continue
        jid, jr, why = j.get("evidence_id"), j.get("reading"), j.get("why")
        if jid not in ids:
            problems.append(f"judgment cites '{jid}', which is not an evidence id from this audit.")
        if jr not in contract.READINGS:
            problems.append(f"judgment for '{jid}' must read as one of {', '.join(contract.READINGS)}.")
        if not isinstance(why, str):
            problems.append(f"judgment for '{jid}' needs a reason.")
        judgments.append({"evidence_id": jid, "reading": jr, "why": why if isinstance(why, str) else ""})
    cited = set()
    for text in [sentence if isinstance(sentence, str) else "", *reasons, *caveats]:
        cited |= set(_ID_RE.findall(text))
    unknown = sorted(cited - ids, key=lambda s: int(s[1:]))
    if unknown:
        problems.append("the verdict cites evidence ids that do not exist: " + ", ".join(unknown) + ".")
    draft.update({"label": label, "sentence": sentence if isinstance(sentence, str) else "",
                  "estimate_evidence_id": est, "claimed_direction": direction, "reasons": reasons,
                  "assumed": assumed, "caveats": caveats, "judgments": judgments})
    return reading, draft, problems


def _accepted_text(verdict: dict, floor_out: dict) -> str:
    """The tool result for an accepted submit_verdict; it is sent only if the user asks a follow-up."""
    out = f"The verdict was accepted. After the floor rules the label is {verdict.get('label')}."
    rules = sorted({str(v.get("rule")) for v in (floor_out or {}).get("violations") or [] if isinstance(v, dict)})
    if rules:
        out += " The floor rules applied: " + ", ".join(rules) + "."
    return out


def _finish(state: dict, reading: dict, draft: dict, submit_id: str | None = None,
            results: list | None = None) -> dict:
    try:
        from . import floor
    except Exception as exc:
        return _error(state, f"The floor rules could not be loaded: {exc}")
    try:
        verdict, floor_out = floor.apply(copy.deepcopy(draft), copy.deepcopy(reading),
                                         copy.deepcopy(state["evidence"]))
    except Exception as exc:
        return _error(state, f"The floor rules failed: {type(exc).__name__}: {exc}")
    verdict, floor_out = _plain(verdict), _plain(floor_out)
    # Keep what a follow-up needs: the results still owed to the model for this turn (the accepted
    # submit_verdict included), and the verdict itself.
    owed = list(results or [])
    if submit_id:
        owed.append(_tool_result(submit_id, _accepted_text(verdict, floor_out)))
    state["pending_results"], state["asks"] = owed, []
    state["invalid_submits"] = 0
    state["finished"] = True
    state["reading"], state["draft_verdict"] = _plain(reading), _plain(draft)
    state["verdict"], state["floor"] = verdict, floor_out
    report = _report_from(state)
    report["state"] = state
    return _plain(report)


# --------------------------------------------------------------------------- the loop

def _questions_from(asks: list[dict]) -> list[dict]:
    return [q for ask in asks for q in ask["questions"]]


def _clean_questions(raw, taken: set[str]) -> list[dict]:
    out = []
    for i, q in enumerate(raw if isinstance(raw, list) else [], 1):
        if not isinstance(q, dict) or not isinstance(q.get("question"), str) or not q["question"].strip():
            continue
        qid = q.get("id") if isinstance(q.get("id"), str) and q.get("id") else f"q{i}"
        base, n = qid, 2
        while qid in taken:
            qid, n = f"{base}_{n}", n + 1
        taken.add(qid)
        opts = [str(o) for o in q.get("options") or [] if isinstance(o, (str, int, float))]
        out.append({"id": qid, "question": q["question"].strip(), "options": opts})
    return out


def _loop(df, state: dict, client, max_turns: int, helper_writer=None) -> dict:
    try:
        from . import evidence
    except Exception as exc:
        return _error(state, f"The evidence tools could not be loaded: {exc}")
    parts = _optional_parts()
    tools = _tools(evidence, parts)
    state.setdefault("screen", [])
    state.setdefault("followups", [])
    state["finished"] = False
    columns = {str(c) for c in getattr(df, "columns", [])} if df is not None else None

    for _ in range(max(0, int(max_turns))):
        try:
            msg = _call_model(client, state, tools)
        except Exception as exc:
            return _error(state, _api_error_text(exc))
        _add_usage(state["usage"], msg)
        try:
            blocks = _echo_blocks([_block_dict(b) for b in (_get(msg, "content") or [])])
        except Exception as exc:
            return _error(state, f"The model's reply could not be read: {type(exc).__name__}: {exc}")
        stop = _get(msg, "stop_reason")
        text = "\n\n".join(b.get("text", "") for b in blocks if b.get("type") == "text" and b.get("text"))
        if text:
            state["transcript"].append({"role": "assistant", "text": text})
        if blocks:
            state["messages"].append({"role": "assistant", "content": blocks})

        if stop == "refusal":
            cat = _get(_get(msg, "stop_details"), "category")
            why = f" (category: {cat})" if cat else ""
            return _error(state, f"The model declined to continue this audit{why}.")
        if stop == "max_tokens":
            return _error(state, "The model's reply was cut off by the length limit before it finished.")
        if stop == "pause_turn":
            continue

        tool_uses = [b for b in blocks if b.get("type") == "tool_use"]
        if not tool_uses:
            return _error(state, "The model stopped without submitting a verdict."
                                 + (f" Its last message: {text[:600]}" if text else ""))

        results, asks, submits = [], [], []
        for b in tool_uses:
            name = b.get("name")
            if name in contract.EVIDENCE_TOOLS:
                item = _run_evidence(evidence, name, b.get("input"), df, state)
                body, is_err = _evidence_text(item)
                results.append(_tool_result(b["id"], body, is_err))
            elif name == contract.LAYOUT_FACTS and name in parts:
                item = _run_layout(parts[name], b.get("input"), df, state)
                body, is_err = _evidence_text(item)
                results.append(_tool_result(b["id"], body, is_err))
            elif name == contract.CUSTOM_CHECKS and name in parts:
                body, is_err = _run_custom_checks(parts[name], b.get("input"), df, state, helper_writer)
                results.append(_tool_result(b["id"], body, is_err))
            elif name == contract.SHOW_CHART and name in parts:
                body, is_err = _run_show_chart(parts[name], b.get("input"), df, state)
                results.append(_tool_result(b["id"], body, is_err))
            elif name == contract.ASK_USER:
                asks.append(b)
            elif name == contract.SUBMIT_VERDICT:
                submits.append(b)
            else:
                results.append(_tool_result(b["id"], f"There is no tool named {name}.", True))

        if asks:
            taken: set[str] = set()
            pending = []
            for b in asks:
                qs = _clean_questions(_get(b.get("input"), "questions"), taken)
                if qs:
                    pending.append({"tool_use_id": b["id"], "questions": qs})
                else:
                    results.append(_tool_result(b["id"], "No question was given; nothing was asked.", True))
            if pending:
                for b in submits:
                    results.append(_tool_result(
                        b["id"], "Not submitted: you asked the user questions in the same turn. "
                                 "Submit again after reading the answers.", True))
                state["pending_results"] = results
                state["asks"] = pending
                report = _report_from(state)
                report["status"] = "needs_input"
                report["questions"] = _questions_from(pending)
                report["state"] = state
                return _plain(report)

        if submits:
            for b in submits[:-1]:
                results.append(_tool_result(b["id"], "Ignored: only the last submit_verdict in a turn counts.",
                                            True))
            b = submits[-1]
            reading, draft, problems = _check_submission(b.get("input"), state["evidence"], columns)
            if not problems:
                return _finish(state, reading, draft, b["id"], results)
            state["invalid_submits"] += 1
            if state["invalid_submits"] >= 2:
                return _error(state, "The model's verdict was not valid after one correction: "
                                     + " ".join(problems))
            results.append(_tool_result(
                b["id"], "The verdict was not accepted. Fix these and call submit_verdict again:\n- "
                         + "\n- ".join(problems), True))

        state["messages"].append({"role": "user", "content": results})

    return _error(state, f"The model used all {max_turns} turns without submitting a verdict.")


# --------------------------------------------------------------------------- public surface

def run_audit(df, claim: str, context: str = "", client=None, model: str = contract.MODEL,
              max_turns: int = 24, helper_writer=None) -> dict:
    """Run an audit from the start. Returns a report in the contract.empty_report() shape."""
    state = _new_state(claim, context, model)
    try:
        if client is None:
            client, err = _make_client()
            if err:
                return _error(state, err)
        try:
            from . import evidence
        except Exception as exc:
            return _error(state, f"The evidence tools could not be loaded: {exc}")
        facts = _run_evidence(evidence, "table_facts", {}, df, state)
        first = prompt.first_message(claim, context, facts["id"], facts.get("lines") or [],
                                     None if facts.get("ok") else facts.get("error"))
        state["messages"].append({"role": "user", "content": first})
        state["transcript"].append({"role": "user", "text": first})
        return _loop(df, state, client, max_turns, helper_writer)
    except Exception as exc:  # last resort: never raise
        return _error(state, f"The audit stopped unexpectedly: {type(exc).__name__}: {exc}")


def resume(df, state: dict, answers: dict, client=None, max_turns: int = 24, helper_writer=None) -> dict:
    """Continue an audit that stopped with status "needs_input", given {question id: answer}."""
    if not isinstance(state, dict) or not isinstance(state.get("messages"), list) or not state.get("asks"):
        return _error(_new_state("", "", contract.MODEL), "There is no paused audit to continue.")
    state = copy.deepcopy(state)
    try:
        answers = {str(k): v for k, v in (answers or {}).items()}
        if client is None:
            client, err = _make_client()
            if err:
                return _error(state, err)
        results = list(state.get("pending_results") or [])
        shown = []
        for ask in state["asks"]:
            text = prompt.answers_message(ask["questions"], answers)
            results.append(_tool_result(ask["tool_use_id"], text))
            shown.append(text)
        state["messages"].append({"role": "user", "content": results})
        state["transcript"].append({"role": "user", "text": "\n\n".join(shown)})
        state["pending_results"], state["asks"] = [], []
        return _loop(df, state, client, max_turns, helper_writer)
    except Exception as exc:
        return _error(state, f"The audit stopped unexpectedly: {type(exc).__name__}: {exc}")


def followup(df, state: dict, question: str, client=None, max_turns: int = 16, helper_writer=None) -> dict:
    """Ask a follow-up question after a finished audit (status "done", its `state` kept). The model may
    call tools again and must end with submit_verdict; the floor rules run again. Earlier evidence,
    screen entries and ids are kept."""
    if (not isinstance(state, dict) or not isinstance(state.get("messages"), list) or not state.get("finished")
            or state.get("asks")):
        return _error(_new_state("", "", contract.MODEL), "There is no finished audit to ask about.")
    question = (question or "").strip() if isinstance(question, str) else ""
    if not question:
        return _error(copy.deepcopy(state), "Write the follow-up question.")
    state = copy.deepcopy(state)
    try:
        if client is None:
            client, err = _make_client()
            if err:
                return _error(state, err)
        state.setdefault("followups", []).append({"question": question})
        msg = prompt.followup_message(question)
        content = list(state.get("pending_results") or []) + [{"type": "text", "text": msg}]
        state["messages"].append({"role": "user", "content": content})
        state["transcript"].append({"role": "user", "text": question})
        state["pending_results"], state["asks"] = [], []
        state["invalid_submits"] = 0
        return _loop(df, state, client, max_turns, helper_writer)
    except Exception as exc:
        return _error(state, f"The follow-up stopped unexpectedly: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- FakeClient

_fake_ids = itertools.count(1)


def text(s: str) -> dict:
    """A scripted text block."""
    return {"type": "text", "text": s}


def thinking(s: str = "", signature: str = "fake-signature") -> dict:
    """A scripted thinking block."""
    return {"type": "thinking", "thinking": s, "signature": signature}


def tool_use(name: str, input: dict | None = None, id: str | None = None) -> dict:
    """A scripted tool call."""
    return {"type": "tool_use", "id": id or f"toolu_fake_{next(_fake_ids)}", "name": name, "input": input or {}}


class FakeClient:
    """Replays a scripted list of assistant turns in place of anthropic.Anthropic().

    Each script item is one of:
      * a list of content blocks (dicts; see text(), thinking(), tool_use()); stop_reason is
        "tool_use" when a tool_use block is present, else "end_turn";
      * a dict {"content": [...], "stop_reason": ..., "usage": {...}, "stop_details": {...}};
      * an Exception instance, which is raised by the call (to test API errors).
    Every request's keyword arguments are deep-copied into `calls`. Each turn reports
    DEFAULT_USAGE unless the item gives its own usage.
    """

    DEFAULT_USAGE = {"input_tokens": 100, "output_tokens": 20}

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self.messages = _FakeMessages(self)
        self.beta = SimpleNamespace(messages=self.messages)

    def _next(self, kwargs: dict):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.script:
            raise RuntimeError("FakeClient script ran out of turns")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        if isinstance(item, list):
            item = {"content": item}
        content = copy.deepcopy(item.get("content") or [])
        stop = item.get("stop_reason") or (
            "tool_use" if any(b.get("type") == "tool_use" for b in content) else "end_turn")
        usage = SimpleNamespace(**{"input_tokens": 0, "output_tokens": 0,
                                   **(item.get("usage") or self.DEFAULT_USAGE)})
        return SimpleNamespace(id=f"msg_fake_{len(self.calls)}", type="message", role="assistant",
                               model=kwargs.get("model", contract.MODEL), content=content,
                               stop_reason=stop, stop_details=item.get("stop_details"), usage=usage)


class _FakeMessages:
    def __init__(self, owner: FakeClient):
        self._owner = owner

    def create(self, **kwargs):
        return self._owner._next(kwargs)

    def stream(self, **kwargs):
        return _FakeStream(self._owner, kwargs)


class _FakeStream:
    def __init__(self, owner: FakeClient, kwargs: dict):
        self._owner, self._kwargs, self._msg = owner, kwargs, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        if self._msg is None:
            self._msg = self._owner._next(self._kwargs)
        return self._msg
