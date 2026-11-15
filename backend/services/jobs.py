"""Background analyses: run the core pipeline in a worker thread and keep the analysis row up to date.

Modes:   audit (claim + data) · explore (data only) · claim-only (claim only: literature, no verdict)
Status:  running -> done | needs_input | error
         needs_input = the core stopped to ask (which column the claim means, which design); answers re-run the
         same analysis in place.
Steps are written to the row as the pipeline reports progress, so polling survives a page reload.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
from typing import Optional

from causality import data, pipeline, units
from causality.data import InputError
from causality.schema import Roles, to_jsonable

from . import graph, llm, mailer, store

STEP_LABELS = {
    "read": "Read the file",
    "prepare": "Prepare the data (types, missing values, duplicates)",
    "claim": "Read the claim",
    "design": "Work out the study design",
    "balance": "Balance between groups",
    "checks": "Overlap, missing outcomes, spillovers, reading list, design tests",
    "estimate": "Estimate the effect",
    "sensitivity": "Sensitivity to hidden confounding",
    "verdict": "Grade, verdict and next study",
    "explore": "Causal discovery (FCI, PC, LiNGAM)",
    "graph": "Influence graph",
    "summary": "Executive summary",
    "literature": "Search the literature",
}
PLAN = {
    "audit": ["read", "prepare", "claim", "design", "balance", "checks", "estimate", "sensitivity", "verdict", "graph",
              "summary"],
    "explore": ["read", "prepare", "explore", "graph", "summary"],
    "claim-only": ["literature"],
}


class Steps:
    """The pipeline's stages for the UI: pending -> running -> done | error | skipped, with seconds."""

    def __init__(self, analysis_id: str, mode: str):
        self.id = analysis_id
        self.items = [self._new(k) for k in PLAN.get(mode, [])]
        self.t: Optional[float] = None
        self.deleted = False

    @staticmethod
    def _new(key: str) -> dict:
        return {"key": key, "label": STEP_LABELS.get(key, key), "status": "pending", "seconds": None}

    def _close_running(self, status: str, note: Optional[str] = None) -> None:
        now = time.time()
        for s in self.items:
            if s["status"] == "running":
                s["status"], s["seconds"] = status, round(now - (self.t or now), 1)
                if note:
                    s["note"] = note[:200]

    def start(self, key: str) -> None:
        self._close_running("done")
        s = next((x for x in self.items if x["key"] == key), None)
        if s is None:  # a stage the plan did not expect (e.g. explore after a claim that names no columns)
            s = self._new(key)
            self.items.append(s)
        s["status"] = "running"
        self.t = time.time()
        self.save()

    def finish(self, error: Optional[str] = None) -> None:
        self._close_running("error" if error else "done", error)
        for s in self.items:
            if s["status"] == "pending":
                s["status"] = "skipped"
        self.save()

    def save(self) -> None:
        if not store.update_analysis(self.id, steps=self.items):
            self.deleted = True


def title_for(filename: str, claim: str) -> str:
    if filename:
        return filename.rsplit("/", 1)[-1]
    c = " ".join((claim or "").split())
    return (c[:57] + "…") if len(c) > 58 else (c or "Untitled analysis")


def start(analysis_id: str) -> threading.Thread:
    t = threading.Thread(target=run, args=(analysis_id,), daemon=True, name=f"analysis-{analysis_id}")
    t.start()
    return t


def run(analysis_id: str) -> None:
    a = store.get_analysis(analysis_id)
    if not a:
        return
    steps = Steps(analysis_id, a["mode"])
    try:
        if a["mode"] == "claim-only":
            result, status = _claim_only(a, steps), "done"
        else:
            result, status = _pipeline(a, steps)
    except InputError as e:  # bad data, in plain words
        steps.finish(str(e))
        _finish(analysis_id, steps, status="error", error=str(e))
        return
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        msg = f"Something went wrong during the analysis ({type(e).__name__}: {e}). Nothing was concluded."
        steps.finish(msg)
        _finish(analysis_id, steps, status="error", error=msg)
        return
    steps.finish()
    _finish(analysis_id, steps, status=status, error=None, result=result)


def _finish(analysis_id: str, steps: Steps, **fields) -> None:
    """Write the outcome, unless the analysis was deleted while it ran (it must not come back)."""
    if steps.deleted or not store.update_analysis(analysis_id, **fields):
        return
    if fields["status"] in ("done", "error"):  # not while it waits for the user's answer
        mailer.notify(analysis_id)


# --------------------------------------------------------------------------------------------- data modes
def _pipeline(a: dict, steps: Steps) -> tuple[dict, str]:
    p = a["params"]
    steps.start("read")
    content = store.load_upload(a["id"])
    if content is None:
        raise InputError("The uploaded file is no longer on the server. Upload it again.")
    raw = data.load_table(content, a["filename"] or "data.csv")
    rep = pipeline.run(raw, claim_text=a["claim"], design_decl=p.get("design") or None, roles=p.get("roles") or {},
                       confirm=p.get("confirm") or {}, online_literature=bool(p.get("online_literature")),
                       progress=steps.start)
    report = to_jsonable(rep)
    result = {"mode": report["mode"], "report": report, "influence": None, "base_graph": None, "frame": None,
              "summary": None, "column_stats": column_stats(raw), "claimed_effect": None}
    try:  # the claim's size in outcome units (drawn on the Treatment Effect forest plot)
        result["claimed_effect"] = to_jsonable(units.claimed_effect(rep.claim, rep.estimate))
    except Exception:  # noqa: BLE001
        pass
    if report.get("needs_input"):
        return result, "needs_input"

    steps.start("graph")
    try:
        roles = (report.get("data") or {}).get("roles") or {}
        prep = data.prepare(raw, Roles(**{k: v for k, v in roles.items() if k in Roles.__dataclass_fields__}))
        numeric = prep.df.select_dtypes(include="number")
        store.save_frame(a["id"], numeric)
        meta, base = graph.frame_meta(report), graph.base_graph(report)
        result["frame"], result["base_graph"] = meta, base
        if base:
            result["influence"] = graph.influence_at_depth(graph.Frame(numeric, meta["encoding"], meta["kinds"],
                                                                       meta["skip"]), base)
    except Exception as e:  # noqa: BLE001  (the report stands without the graph)
        traceback.print_exc()
        result["influence"] = {"error": f"{type(e).__name__}: {e}", "nodes": [], "edges": []}

    steps.start("summary")
    result["summary"] = summarize(a["claim"], report)
    return result, "done"


def column_stats(raw) -> dict:
    """Per original column: % missing, and the mean of numeric columns (for the Data tab)."""
    import pandas as pd
    out = {}
    n = max(1, len(raw))
    for c in raw.columns:
        col = raw[c]
        mean = None
        if pd.api.types.is_numeric_dtype(col) and not pd.api.types.is_bool_dtype(col):
            m = col.mean()
            mean = float(m) if pd.notna(m) else None
        out[str(c)] = {"pct_missing": round(100 * float(col.isna().sum()) / n, 2), "mean": mean}
    return out


def template_summary(report: dict) -> str:
    v, d, e = report.get("verdict") or {}, report.get("design") or {}, report.get("estimate") or {}
    parts = []
    if v:
        parts.append(f"{v.get('label')}: {v.get('sentence')}")
    if d.get("grade"):
        parts.append(f"Design grade {d['grade']} ({d.get('kind')}).")
    if e.get("refused"):
        parts.append(f"No number was given: {e.get('refusal_reason')}")
    elif e.get("estimate") is not None and e.get("ci95"):
        lo, hi = e["ci95"]
        parts.append(f"{e.get('method')}: {e['estimate']:.4g} (95% interval {lo:.4g} to {hi:.4g}), n = {e.get('n')}.")
    if report.get("mode") == "explore":
        parts.append((report.get("explore") or {}).get("summary")
                     or "No claim was given, so the analysis looked for what may cause what.")
    return " ".join(x for x in parts if x)


SUMMARY_SYSTEM = ("You write the executive summary of a causal analysis for a non-specialist. Use only the facts in "
                  "the report JSON and never claim more than it shows: anything marked 'assumed' was not tested. "
                  "Plain text, 3-5 sentences, no headings or lists.")


def summarize(claim: str, report: dict) -> dict:
    """Claude writes the summary when available; the template is the fallback, and the reason is kept."""
    fallback = template_summary(report)
    if not llm.available():
        return {"text": fallback, "source": "template"}
    est = report.get("estimate") or {}
    slim = {"mode": report.get("mode"), "verdict": report.get("verdict"), "estimand": report.get("estimand"),
            "assumed_not_tested": [x.get("name") for x in report.get("assumed_panel") or []],
            "design": {k: (report.get("design") or {}).get(k) for k in ("kind", "grade", "grade_reasons", "declared")},
            "estimate": {k: est.get(k) for k in ("method", "target", "scope", "estimate", "ci95", "p_value", "n",
                                                  "refused", "refusal_reason")},
            "checks": [{"title": s.get("title"), "status": s.get("status"), "summary": s.get("summary")}
                       for s in report.get("sections") or []],
            "exploration": (report.get("explore") or {}).get("summary")}
    text = llm.run(f"Claim: {claim or '(none: exploration)'}\nReport JSON:\n{json.dumps(slim, default=str)[:20000]}"
                   "\n\nWrite the executive summary.", system=SUMMARY_SYSTEM, timeout=120, kind="summary")
    if isinstance(text, str) and text.strip():
        return {"text": text.strip(), "source": "llm"}
    return {"text": fallback, "source": "template", "llm_error": llm.last_error()}


# --------------------------------------------------------------------------------------------- claim only
SOURCE = {"type": "object", "properties": {"title": {"type": "string"}, "year": {"type": ["integer", "null"]},
                                           "url": {"type": "string"}, "doi": {"type": ["string", "null"]}},
          "required": ["title", "year", "url", "doi"]}
CLAIM_ONLY_SCHEMA = {
    "type": "object",
    "properties": {
        "treatment": {"type": ["string", "null"]}, "outcome": {"type": ["string", "null"]},
        "population": {"type": ["string", "null"]}, "summary": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "object", "properties": {
            "finding": {"type": "string"}, "direction": {"type": "string", "enum": ["supports", "contradicts", "mixed"]},
            "sources": {"type": "array", "items": SOURCE}}, "required": ["finding", "direction", "sources"]}},
        "confounders": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "why": {"type": "string"}}, "required": ["name", "why"]}},
        "designs": {"type": "array", "items": {"type": "object", "properties": {
            "design": {"type": "string"}, "why": {"type": "string"}}, "required": ["design", "why"]}},
        "data_needed": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["treatment", "outcome", "population", "summary", "evidence", "confounders", "designs", "data_needed"],
}


def _claim_only(a: dict, steps: Steps) -> dict:
    """No data, so nothing is estimated and there is no verdict: what research says, and what data would test it."""
    steps.start("literature")
    out = llm.run(
        f"A user wants to know whether this causal claim holds: {a['claim']!r}. They have no dataset.\n"
        "Use web search (at most 4 searches) to find published research on it. Report:\n"
        "- treatment, outcome, population: as stated in the claim (null if not stated)\n"
        "- summary: 2-3 sentences on what the research says, without overstating it\n"
        "- evidence: 2-5 findings, each with direction (supports / contradicts / mixed) and at least one real source you "
        "actually found (exact title, year, URL, DOI or null). Never invent studies or URLs\n"
        "- confounders: 2-5 things that would bias a naive comparison, with why\n"
        "- designs: 1-3 study designs that could test the claim credibly, with why\n"
        "- data_needed: the columns a dataset would need to test it",
        schema=CLAIM_ONLY_SCHEMA, web=True, timeout=300, kind="claim_only")
    if not isinstance(out, dict):
        raise RuntimeError(f"the literature search failed: {llm.last_error() or 'no answer'}")
    out["evidence"] = [e for e in out.get("evidence", [])
                       if any(str(s.get("url", "")).startswith("http") for s in e.get("sources", []))]
    return {"mode": "claim-only", "report": None, "influence": None, "claim_only": out,
            "summary": {"text": out.get("summary", ""), "source": "llm"}}
