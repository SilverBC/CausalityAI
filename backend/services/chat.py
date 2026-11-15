"""The Investigator: a chat assistant that talks about one analysis using its real numbers.

Each reply is one language-model call with a persona, a JSON digest of the analysis (≤ 24k characters), the last 12
messages and the new one. Structured output: {reply, cards: [{kind, ref}], investigate: node id | null}. The server
keeps only cards that point at something real, and when `investigate` names a real node it starts (or queues) that
literature investigation and adds its card. The model is whatever services/llm.py picks (Gemini when a
GEMINI_API_KEY is set). When the model fails, nothing is stored and the reason is returned: v1
answered "Sorry…" and hid the cause, which made a usage-limit outage look like a bug.
"""
from __future__ import annotations

import json
from typing import Optional

from . import investigate, llm, store

HISTORY = 12
DIGEST_CHARS = 24000
MAX_MESSAGE = 4000

SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "cards": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string", "enum": ["estimate", "check", "graph", "investigation"]},
            "ref": {"type": ["string", "null"]},
        }, "required": ["kind", "ref"]}},
        "investigate": {"type": ["string", "null"]},
    },
    "required": ["reply", "cards", "investigate"],
}

SYSTEM = (
    "You are the Investigator, a careful econometrician chatting with someone about the causal analysis of their "
    "dataset. Answer from the analysis digest; quote its real numbers. If the digest does not contain something, say "
    "so instead of guessing. Keep replies short (2-6 sentences, plain text, no headings or tables). Explain jargon "
    "briefly. Distinguish what the data show from hypothetical leads found in the literature. Anything with status "
    "'assumed' was not tested: never present it as a pass."
)


class ChatUnavailable(RuntimeError):
    pass


def _short(x, n: int = 400) -> str:
    s = x if isinstance(x, str) else json.dumps(x, default=str)
    return s if len(s) <= n else s[: n - 1] + "…"


def _scalars(d: Optional[dict], limit: int = 12) -> dict:
    """A check's simple metrics (numbers, short text, flags); tables and charts are left out."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (int, float, bool)) or v is None or isinstance(v, str):
            out[k] = _short(v, 200) if isinstance(v, str) else v
            if len(out) >= limit:
                break
    return out


def digest(a: dict) -> dict:
    """Everything the Investigator may talk about, compact and JSON-able."""
    res = a.get("result") or {}
    rep = res.get("report") or {}
    design, est, verdict = rep.get("design") or {}, rep.get("estimate") or {}, rep.get("verdict") or {}
    data = rep.get("data") or {}
    d: dict = {
        "dataset": a["title"], "mode": res.get("mode") or a["mode"], "claim": a.get("claim") or None,
        "summary": (res.get("summary") or {}).get("text"),
        "design": {k: design.get(k) for k in ("kind", "declared", "grade", "grade_reasons")} if design else None,
        "verdict": {k: verdict.get(k) for k in ("label", "sentence", "caveat", "reasons", "sub")} if verdict else None,
        "estimand": (rep.get("estimand") or {}).get("text"),
        "estimate": {k: est.get(k) for k in ("method", "target", "scope", "estimate", "se", "ci90", "ci95", "p_value", "n",
                                             "baseline_mean", "outcome_sd", "refused", "refusal_reason", "comparisons")
                     if est.get(k) not in (None, [])} if est else None,
        "variables": {"roles": {k: v for k, v in (data.get("roles") or {}).items() if v not in (None, [], False)},
                      "adjusted_for": data.get("adjustment_set"), "rows": data.get("n_rows")},
        "checks": {s["key"]: {"title": s.get("title"), "status": s.get("status"), "summary": _short(s.get("summary", ""), 500),
                              "warnings": [_short(w, 220) for w in (s.get("warnings") or [])[:4]],
                              "assumptions": [{"name": x.get("name"), "status": x.get("status")} for x in (s.get("assumptions") or [])[:6]],
                              "metrics": _scalars(s.get("metrics"))}
                   for s in rep.get("sections") or []},
        "assumed_not_tested": [x.get("name") for x in rep.get("assumed_panel") or []],
        "next_study": _short((rep.get("suggestion") or {}).get("summary", ""), 600) or None,
        "errors": rep.get("errors") or None,
    }
    if rep.get("explore"):
        d["exploration"] = {"summary": rep["explore"].get("summary"),
                            "warnings": [_short(w, 200) for w in rep["explore"].get("warnings") or []][:5]}
    if res.get("claim_only"):
        d["claim_only"] = res["claim_only"]
    inf = res.get("influence") or {}
    if inf.get("nodes"):
        d["graph_nodes"] = [{"id": n["id"], "role": n.get("role"), "influence": n.get("influence"),
                             "observed": n.get("observed", True)} for n in inf["nodes"][:60]]
    invs = []
    for it in investigate.public(a["id"]):
        e = {"node": it["node"], "label": it["label"], "status": it["status"]}
        if it["status"] == "done" and it.get("result"):
            e["summary"] = it["result"].get("summary")
            e["factors"] = [{"id": f["id"], "label": f["label"], "in_data": f["in_data"], "affects": f["affects"],
                             "reliability": f.get("reliability"), "explanation": _short(f.get("explanation", ""), 200)}
                            for f in it["result"]["factors"]]
        invs.append(e)
    if invs:
        d["investigations"] = invs
    return d


def prompt(a: dict, history: list[dict], text: str) -> str:
    dg = digest(a)
    node_ids = [n["id"] for n in dg.get("graph_nodes", [])]
    for it in dg.get("investigations", []):
        node_ids += [f["id"] for f in it.get("factors", [])]
    convo = "\n".join(f"{m['role'].upper()}: {_short(m['content'], 1200)}" for m in history[-HISTORY:])
    return (
        "ANALYSIS DIGEST (JSON):\n" + json.dumps(dg, default=str)[:DIGEST_CHARS] + "\n\n"
        + (f"CONVERSATION SO FAR:\n{convo}\n\n" if convo else "")
        + f"USER: {text}\n\n"
        "Respond with:\n"
        "- reply: your answer.\n"
        "- cards: 0-2 visual cards that support the answer. kind 'estimate' (ref null) shows the effect estimate; "
        f"kind 'check' with ref one of {list(dg['checks'])} shows that check; kind 'graph' (ref null) offers the causal "
        "graph; kind 'investigation' with ref = a node id shows that node's literature investigation. Only add cards "
        "that help.\n"
        "- investigate: a node id to start a literature investigation of what causes that variable, ONLY when the user "
        "asks to investigate / look into the research / find other causes or confounders of something. Otherwise null. "
        f"Valid node ids: {list(dict.fromkeys(node_ids))[:80]}. If you start one, say in the reply that it runs in the "
        "background (30 s to 2 min) and its findings will be hypothetical leads from published research, not from their data."
    )


def _node_info(a: dict, node: str) -> Optional[tuple[str, bool]]:
    for n in ((a.get("result") or {}).get("influence") or {}).get("nodes") or []:
        if n["id"] == node:
            return n.get("label", node), bool(n.get("observed", True))
    for it in investigate.public(a["id"]):
        for f in (it.get("result") or {}).get("factors", []):
            if f["id"] == node:
                return f["label"], bool(f["in_data"])
    return None


def reply(a: dict, history: list[dict], text: str) -> dict:
    """{'reply', 'cards'} for the user's message. Raises ChatUnavailable with the reason when Claude can't answer."""
    if not llm.available():
        raise ChatUnavailable("No language model is configured on this server (set GEMINI_API_KEY in .env, "
                              "or set CAUSAL_LLM).")
    out = llm.run(prompt(a, history, text), schema=SCHEMA, system=SYSTEM, timeout=150, kind="chat")
    if not isinstance(out, dict) or not str(out.get("reply") or "").strip():
        raise ChatUnavailable(llm.last_error() or "no answer came back")

    res = a.get("result") or {}
    rep = res.get("report") or {}
    checks = {s["key"] for s in rep.get("sections") or []}
    investigated = {i["node"] for i in investigate.public(a["id"])}
    cards: list[dict] = []
    for c in (out.get("cards") or [])[:3]:
        kind, ref = c.get("kind"), c.get("ref")
        real = ((kind == "estimate" and rep.get("estimate")) or (kind == "check" and ref in checks)
                or (kind == "graph" and (res.get("influence") or {}).get("nodes")) or (kind == "investigation" and ref in investigated))
        card = {"kind": kind, "ref": ref if kind in ("check", "investigation") else None}
        if real and card not in cards:
            cards.append(card)
    text_out = str(out["reply"]).strip()
    node = out.get("investigate")
    if node and (info := _node_info(a, node)) is not None:
        try:
            investigate.start(a, node, info[0], info[1])
            cards = [c for c in cards if c != {"kind": "investigation", "ref": node}] + [{"kind": "investigation", "ref": node}]
        except investigate.QueueFull as e:
            text_out += f"\n\n(I couldn't start the investigation: {e})"
    return {"reply": text_out, "cards": cards}


def send(chat: dict, a: dict, text: str) -> dict:
    """Get the reply first; store the user's message and the reply together only when it worked."""
    history = chat.get("messages") or []
    out = reply(a, history, text)
    user, assistant = store.add_messages(chat["id"], [("user", text, []), ("assistant", out["reply"], out["cards"])])
    if not history:
        store.rename_chat(chat["id"], " ".join(text.split())[:60])  # a chat is named after its first message
    return {"user": user, "assistant": assistant, "chat": store.get_chat(chat["id"], with_messages=False)}
