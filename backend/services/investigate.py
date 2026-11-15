"""Literature investigations ("sleuth mode"): what does published research say causes a variable?

A dataset can never contain the confounders nobody measured; published research can name them. One investigation
= one language-model call with web search (at most 4 searches, 300 s) under a strict JSON schema, then OpenAlex
lookups for each source's Reliability Score. Factors without a real http(s) source are dropped. A factor maps to a
dataset column only on an exact column-name match; everything else becomes a hypothetical node ("hyp:<slug>").
No strengths are attached: these are leads from outside the data, not measured effects.

Queue, per analysis, kept in SQLite: FIFO, at most MAX_RUNNING at once (protects the Claude quota) and MAX_PENDING
running + queued (QueueFull beyond, HTTP 429). `start` is idempotent per node: a queued, running or finished
investigation of that node is returned as is; a failed one is replaced. The list of variables already on the graph
is computed when a search actually starts, so later searches see earlier findings.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Optional

from . import llm, store
from .reliability import score_factors

MAX_RUNNING = 3
MAX_PENDING = 40
TIMEOUT_S = 300
_lock = threading.RLock()

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "factors": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"},
            "affects": {"type": "array", "items": {"type": "string", "enum": ["target", "treatment", "outcome"]}},
            "explanation": {"type": "string"},
            "dataset_column": {"type": ["string", "null"]},
            "sources": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "year": {"type": ["integer", "null"]}, "url": {"type": "string"},
                "doi": {"type": ["string", "null"]}},
                "required": ["title", "year", "url", "doi"]}},
        }, "required": ["name", "affects", "explanation", "dataset_column", "sources"]}},
    },
    "required": ["summary", "factors"],
}


class QueueFull(Exception):
    pass


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:48] or "factor"


# --------------------------------------------------------------------------------------------- the search
def context(analysis: dict) -> dict:
    res = analysis.get("result") or {}
    rep = res.get("report") or {}
    data = rep.get("data") or {}
    roles = data.get("roles") or {}
    inf = res.get("influence") or {}
    return {"treatment": roles.get("treatment") or inf.get("treatment"),
            "outcome": roles.get("outcome") or inf.get("outcome"),
            "columns": [c["name"] for c in data.get("columns") or []],
            "graph": [n["id"] for n in inf.get("nodes") or []],
            "claim": analysis.get("claim") or ""}


def known_labels(analysis: dict) -> list[str]:
    """Variables already on the graph, including earlier findings."""
    labels = list(context(analysis)["graph"])
    for it in store.list_investigations(analysis["id"]):
        labels += [f["label"] for f in (it.get("result") or {}).get("factors", [])]
    return list(dict.fromkeys(labels))


def prompt(ctx: dict, target: str, in_data: bool, known: list[str]) -> str:
    t, y = ctx["treatment"] or "(no treatment given)", ctx["outcome"] or "(no outcome given)"
    return (
        "You are assisting a causal-inference audit of a dataset.\n"
        f"The analysis studies the effect of {t!r} (treatment) on {y!r} (outcome).\n"
        + (f"The claim under audit: {ctx['claim']!r}.\n" if ctx["claim"] else "")
        + f"Dataset columns: {ctx['columns'][:150]}.\n"
        f"Variables already in the causal graph: {known[:120]}.\n\n"
        f"Investigate the variable {target!r}"
        + (" (a column in the dataset)." if in_data else
           " (a factor suggested by earlier research; it is NOT in the dataset).")
        + "\nUse web search (at most 4 searches; be quick) to find published research on what causally affects it. "
        "Report 3-6 factors that the literature documents as causes of it, prioritising factors that ALSO plausibly "
        f"affect the outcome {y!r}, because those would confound the {t!r} -> {y!r} relationship.\n"
        "For each factor give:\n"
        "- name: a short variable name (1-4 words)\n"
        f"- affects: which of 'target' ({target}), 'treatment' ({t}), 'outcome' ({y}) the research says it affects\n"
        "- explanation: 1-2 sentences on the mechanism, as reported in the research\n"
        "- dataset_column: the exact column name from the list above ONLY if that column measures essentially the "
        "same concept; otherwise null\n"
        "- sources: at least one real publication you actually found: its exact title only (no authors, series "
        "numbers or notes), year, URL, and DOI if it has one (else null). Prefer peer-reviewed papers over web pages\n"
        f"Do not return {t!r}, {y!r} or variables already in the graph unless the research specifically links them "
        f"to {target!r}. Do not invent studies or URLs; drop a factor rather than guess a source. "
        "Finish with a 1-2 sentence summary."
    )


def _match_column(ctx: dict, name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    want = name.strip().lower()
    return next((c for c in ctx["columns"] if c.lower() == want), None)


def search(analysis: dict, target: str, label: str, in_data: bool) -> dict:
    """Run one literature search. Returns {summary, factors, edges}; raises with a plain reason on failure."""
    ctx = context(analysis)
    out = llm.run(prompt(ctx, label, in_data, known_labels(analysis)), schema=SCHEMA, web=True, timeout=TIMEOUT_S,
                  kind="investigate")
    if not isinstance(out, dict):
        raise RuntimeError(f"The literature search failed: {llm.last_error() or 'no answer came back'}")
    T, Y = ctx["treatment"], ctx["outcome"]
    factors, edges, seen = [], [], set()
    for f in out.get("factors") or []:
        sources = [dict(s) for s in f.get("sources") or [] if str(s.get("url", "")).startswith(("http://", "https://"))]
        if not sources or not f.get("name"):
            continue  # no verifiable source: drop it
        col = _match_column(ctx, f.get("dataset_column"))
        node_id = col or f"hyp:{slug(f['name'])}"
        if node_id in seen or node_id == target:
            continue
        seen.add(node_id)
        affects = sorted(set(f.get("affects") or ["target"]))
        factors.append({"id": node_id, "label": col or f["name"].strip(), "in_data": bool(col), "column": col,
                        "explanation": f.get("explanation", ""), "sources": sources[:4], "affects": affects})
        targets = {target} | ({Y} if "outcome" in affects and Y else set()) | ({T} if "treatment" in affects and T else set())
        edges += [{"source": node_id, "target": t, "explanation": f.get("explanation", "")}
                  for t in sorted(targets) if t != node_id]
    score_factors(factors)  # Reliability Score per factor, from OpenAlex citations, venue and corroboration
    return {"summary": out.get("summary", ""), "factors": factors, "edges": edges}


# --------------------------------------------------------------------------------------------- the queue
def public(analysis_id: str) -> list[dict]:
    """Investigations with elapsed seconds and, for queued ones, their position in line."""
    now = time.time()
    items = store.list_investigations(analysis_id)
    pos = {i["id"]: n + 1 for n, i in enumerate(x for x in items if x["status"] == "queued")}
    for it in items:
        if it["status"] == "queued":
            it["elapsed"], it["position"] = 0.0, pos[it["id"]]
        else:
            it["elapsed"] = round((it["finished_at"] or now) - (it["started_at"] or it["queued_at"]), 1)
    return items


def counts(analysis_id: str) -> dict:
    st = [i["status"] for i in store.list_investigations(analysis_id)]
    return {"running": st.count("running"), "queued": st.count("queued")}


def start(analysis: dict, node: str, label: str, in_data: bool) -> dict:
    """Start or queue an investigation of one node. Idempotent per node; raises QueueFull at the limit."""
    aid = analysis["id"]
    with _lock:
        items = store.list_investigations(aid)
        same = next((i for i in items if i["node"] == node and i["status"] in ("queued", "running", "done")), None)
        if same:
            return same
        if sum(i["status"] in ("queued", "running") for i in items) >= MAX_PENDING:
            raise QueueFull(f"At most {MAX_PENDING} investigations can be running or queued at once. "
                            "Wait for some to finish, or cancel queued ones.")
        store.delete_investigations(aid, node=node)  # replaces an earlier failed attempt
        inv = store.insert_investigation(analysis_id=aid, node=node, label=label or node, in_data=in_data)
    pump(aid)
    return store.get_investigation(inv["id"]) or inv


def pump(analysis_id: str) -> None:
    """Start queued investigations, oldest first, while fewer than MAX_RUNNING run."""
    while True:
        with _lock:
            items = store.list_investigations(analysis_id)
            if sum(i["status"] == "running" for i in items) >= MAX_RUNNING:
                return
            nxt = next((i for i in items if i["status"] == "queued"), None)
            if not nxt:
                return
            store.update_investigation(nxt["id"], status="running", started_at=time.time())
        threading.Thread(target=_work, args=(nxt,), daemon=True, name=f"investigate-{nxt['id']}").start()


def _work(inv: dict) -> None:
    try:
        analysis = store.get_analysis(inv["analysis_id"])
        if analysis is None:
            return
        res = search(analysis, inv["node"], inv["label"], inv["in_data"])
        # if the row was deleted (cancelled) meanwhile, the result goes nowhere
        store.update_investigation(inv["id"], status="done", result=res, error=None, finished_at=time.time())
    except Exception as e:  # noqa: BLE001
        store.update_investigation(inv["id"], status="error", error=str(e)[:400], finished_at=time.time())
    finally:
        pump(inv["analysis_id"])


def remove(analysis_id: str, node: str) -> bool:
    """Delete (or cancel) one node's investigation. A search already running finishes into the void."""
    n = store.delete_investigations(analysis_id, node=node)
    pump(analysis_id)
    return n > 0


def cancel_queued(analysis_id: str) -> int:
    return store.delete_investigations(analysis_id, status="queued")
