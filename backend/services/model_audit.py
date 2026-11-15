"""The model-led audit, wired into the web app's analysis.

When a claim comes with data and the user declared neither a design nor column roles, a language model reads
the claim and audits it. The core pipeline then computes the whole report under the model's reading (treatment,
outcome, design, role columns, columns to leave out), so every tab of the web app works as before, and the
model's verdict replaces the rule-based one after a check in code that it claims no more than the computed
interval shows. If no model is configured, or the model fails, the analysis runs exactly as it did before.

Two engines:
    free   (Gemini)  the inverted design: the model analyses the table itself, with its own code, and may call
                     CausalityAI's tools (orchestrator/free_agent.py).
    loop   (Claude)  the model as orchestrator of the fixed tools (orchestrator/loop.py), used when the model is
                     Claude, because the free agent runs on Gemini only.

    CAUSAL_MODEL_AUDIT   auto (default) | on | off
        auto  Gemini when GEMINI_API_KEY is set (environment or .env), else Claude when ANTHROPIC_API_KEY is set
        on    as auto, and fall back to headless Claude Code (the signed-in subscription) when neither key is set
        off   never; the analysis is the rule-based pipeline only
    The model is never used when CAUSAL_LLM is "fake" or "off".
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import Optional

from causality.orchestrator import contract

NOBODY = "Nobody is available to answer right now. Decide from the claim and the data, and say what you assumed."
# orchestrator label -> the web app's verdict label
LABELS = {contract.SUPPORTED: "Supported", contract.NOT_SUPPORTED: "Contradicted", contract.NO_EFFECT: "No evidence",
          contract.INCONCLUSIVE: "Inconclusive"}
_FREE_LABELS = {"supported": contract.SUPPORTED, "contradicted": contract.NOT_SUPPORTED,
                "inconclusive": contract.INCONCLUSIVE}
_ROLE_KEYS = ("treatment", "outcome", "unit", "time", "instrument", "running", "cluster", "strata")

FREE_INSTRUCTIONS = """data.csv is a table a user uploaded. Nobody is available to answer questions.

Claim to audit: "{claim}"

Decide whether the data support the claim. Analyse the data as you see fit, by writing and running your own code.

Facts about the table, computed by code, are given in the first message. This software also has its own analysis
tools, listed below with exactly what each one computes and returns: layout_facts, design_tests, estimate_effect,
general_checks and sensitivity. They are tested implementations, each runs in seconds, and you can call any of them
at any point and as often as you like, alongside your own code.

Your answer is a JSON object with exactly these keys, given by calling write_answer:
- verdict: "supported", "contradicted" or "inconclusive" (inconclusive when the data cannot settle the claim)
- claimed_direction: "increase", "decrease" or "none", the direction of effect the claim asserts
- sentence: one plain sentence a non-statistician can read
- treatment, outcome: the column names the claim is about
- design: the study design you relied on, one of "experiment", "did", "iv", "rd", "observational"
- design_reason: one or two sentences on what in the data made you choose that design
- unit, time, instrument, running: column names for the roles this design uses, null for the rest
- cutoff: the threshold on the running column (rd only), else null
- left_out: list of {{"column", "reason"}} for columns that should not be adjusted for
- estimate, ci95_low, ci95_high: your best estimate of the effect in outcome units and its 95% interval, or null
- reasons: list of short sentences giving the evidence for the verdict, with numbers
- assumed: list of short sentences naming what the answer relies on that the data cannot test
- caveats: list of short sentences, may be empty
"""


def mode() -> str:
    m = (os.environ.get("CAUSAL_MODEL_AUDIT") or "auto").strip().lower()
    return m if m in ("auto", "on", "off") else "auto"


def pick_client() -> Optional[dict]:
    """{"engine", "model", ...} for the audit, or None when no model should be used."""
    if mode() == "off" or (os.environ.get("CAUSAL_LLM") or "").strip().lower() in ("fake", "off"):
        return None
    from causality.orchestrator import gemini
    if gemini.available():
        return {"engine": "free", "model": gemini.model_name()}
    if os.environ.get("ANTHROPIC_API_KEY"):
        return {"engine": "loop", "model": contract.MODEL, "client": None, "writer": None}   # the loop builds the SDK client
    if mode() == "on":
        from causality.orchestrator import headless
        if headless.find_claude():
            return {"engine": "loop", "model": contract.MODEL, "client": headless.HeadlessClient(), "writer": None}
    return None


def available() -> bool:
    return pick_client() is not None


# --------------------------------------------------------------------------------------------- the two engines
def _col(value, cols):
    return value if isinstance(value, str) and value in cols else None


def _strings(x) -> list:
    return [str(v) for v in x if isinstance(v, (str, int, float)) and str(v).strip()] if isinstance(x, list) else []


def _run_free(raw, claim: str, picked: dict) -> dict:
    """The inverted design. Returns an audit dict in the same shape the loop gives."""
    from causality import agent
    from causality.orchestrator import free_agent
    folder = tempfile.mkdtemp(prefix="causalityai_audit_")
    try:
        raw.to_csv(os.path.join(folder, "data.csv"), index=False)
        try:
            facts = "# Facts about data.csv (computed by code, no conclusions)\n\n" + agent.facts_card(raw)
        except Exception:  # noqa: BLE001
            facts = None
        res = free_agent.run(folder, FREE_INSTRUCTIONS.format(claim=claim.replace('"', "'")), context=facts,
                             helpers=True, model=picked.get("model"), transport=picked.get("transport"),
                             sleep=picked.get("sleep") or __import__("time").sleep)
        ans = {}
        if res["status"] == "done":
            import json
            with open(os.path.join(folder, "answer.json"), encoding="utf-8") as f:
                ans = json.load(f)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    usage = {"model": res.get("model"), "turns": res.get("turns"), "input_tokens": res.get("input_tokens"),
             "output_tokens": res.get("output_tokens"), "seconds": res.get("seconds")}
    evidence = [{"id": f"a{i}", "tool": c.get("tool"), "ok": True, "error": None,
                 "args": {k: v for k, v in (c.get("args") or {}).items() if k != "answer_json"},
                 "lines": [ln for ln in str(c.get("result") or "").splitlines() if ln.strip()][:40]}
                for i, c in enumerate(res.get("calls") or [], 1) if c.get("tool") != "write_answer"]
    if res["status"] != "done":
        return {"status": "error", "error": res.get("error") or "The model gave no answer.", "engine": "free",
                "usage": usage, "evidence": evidence, "asked": []}
    cols = {str(c) for c in raw.columns}
    cut = ans.get("cutoff")
    roles = {k: _col(ans.get(k), cols) for k in _ROLE_KEYS}
    roles["cutoff"] = float(cut) if isinstance(cut, (int, float)) and not isinstance(cut, bool) else None
    direction = ans.get("claimed_direction") if ans.get("claimed_direction") in ("increase", "decrease", "none") else None
    draft = {"label": _FREE_LABELS.get(str(ans.get("verdict", "")).strip().lower(), contract.INCONCLUSIVE),
             "sentence": str(ans.get("sentence") or ""), "estimate_evidence_id": None, "claimed_direction": direction,
             "reasons": _strings(ans.get("reasons")), "assumed": _strings(ans.get("assumed")),
             "caveats": _strings(ans.get("caveats")), "judgments": [],
             "own_estimate": {"estimate": ans.get("estimate"), "ci95": [ans.get("ci95_low"), ans.get("ci95_high")]}}
    reading = {"treatment": roles["treatment"], "outcome": roles["outcome"],
               "design": ans.get("design") if ans.get("design") in contract.DESIGNS else None,
               "design_reason": str(ans.get("design_reason") or ""), "roles": roles, "covariates": None,
               "excluded": [{"column": str(x.get("column")), "reason": str(x.get("reason") or "")}
                            for x in (ans.get("left_out") or []) if isinstance(x, dict) and x.get("column")]}
    return {"status": "done", "error": None, "engine": "free", "reading": reading, "draft_verdict": draft,
            "verdict": dict(draft), "floor": {"changed": False, "violations": []}, "evidence": evidence,
            "usage": usage, "asked": [], "screen": []}


def _run_loop(raw, claim: str, context: str, picked: dict) -> dict:
    from causality.orchestrator import loop
    client, writer = picked.get("client"), picked.get("writer")
    rep = loop.run_audit(raw, claim, context, client=client, model=picked.get("model") or contract.MODEL,
                         helper_writer=writer)
    asked = []
    if rep.get("status") == "needs_input":
        asked = [q.get("question") for q in rep.get("questions") or []]
        answers = {q["id"]: NOBODY for q in rep.get("questions") or [] if q.get("id")}
        rep = loop.resume(raw, rep["state"], answers, client=client, helper_writer=writer)
    if rep.get("status") == "needs_input":
        rep = dict(rep, status="error", error="The model asked a second round of questions and nobody could answer.")
    rep = {k: v for k, v in rep.items() if k != "state"}
    rep["asked"], rep["engine"] = asked, "loop"
    return rep


def run(raw, claim: str, context: str = "", picked: Optional[dict] = None) -> Optional[dict]:
    """Run the audit. Returns the audit (status "done" or "error"), or None when no model is configured.
    Never raises."""
    picked = picked or pick_client()
    if picked is None:
        return None
    try:
        return _run_free(raw, claim, picked) if picked.get("engine") == "free" else _run_loop(raw, claim, context, picked)
    except Exception as e:  # noqa: BLE001  (the analysis must go on without the model)
        return {"status": "error", "error": f"{type(e).__name__}: {e}", "asked": [], "engine": picked.get("engine")}


# --------------------------------------------------------------------------------------------- into the report
def usable(audit: Optional[dict]) -> bool:
    """A finished audit whose reading names a treatment and an outcome."""
    r = (audit or {}).get("reading") or {}
    return bool(audit) and audit.get("status") == "done" and bool(r.get("treatment")) and bool(r.get("outcome")) \
        and bool(audit.get("verdict"))


def declarations(audit: dict, columns) -> tuple[Optional[str], dict]:
    """(design, roles) for pipeline.run from the model's reading. Unknown columns are dropped, never guessed."""
    reading = (audit or {}).get("reading") or {}
    cols = {str(c) for c in columns}
    src = {**(reading.get("roles") or {}), "treatment": reading.get("treatment"), "outcome": reading.get("outcome")}
    roles: dict = {k: src[k] for k in _ROLE_KEYS if src.get(k) in cols}
    cut = (reading.get("roles") or {}).get("cutoff")
    if isinstance(cut, (int, float)) and not isinstance(cut, bool):
        roles["cutoff"] = float(cut)
    design = reading.get("design") if reading.get("design") in contract.DESIGNS else None
    if design == "experiment":
        roles["randomized"] = True
    taken = set(roles.values())
    left_out = [x.get("column") for x in reading.get("excluded") or [] if isinstance(x, dict)]
    roles["post_treatment"] = [c for c in left_out if c in cols and c not in taken]
    return design, roles


def _fmt(x) -> str:
    return f"{x:.4g}" if isinstance(x, (int, float)) else "?"


def check_against_report(audit: dict, report: dict) -> None:
    """Free engine only: hold the model's label to the interval the pipeline computed under its own reading.
    The one rule: a verdict cannot claim more than the 95% interval shows. Changes the audit in place."""
    if audit.get("engine") != "free":
        return                                   # the loop already applied the floor rules to its own evidence
    v, est = audit["verdict"], report.get("estimate") or {}
    label, direction = v.get("label"), v.get("claimed_direction")
    if label not in (contract.SUPPORTED, contract.NOT_SUPPORTED) or direction not in ("increase", "decrease"):
        return
    ci = est.get("ci95") or [None, None]
    lo, hi = (ci + [None, None])[:2]
    why = None
    if est.get("refused") or not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
        why = "the tested estimator gave no interval for this reading" + (
            f" ({est.get('refusal_reason')})" if est.get("refusal_reason") else "")
    else:
        sign = 1 if direction == "increase" else -1
        backs = (lo > 0) if sign == 1 else (hi < 0)
        against = (hi < 0) if sign == 1 else (lo > 0)
        if label == contract.SUPPORTED and not backs:
            why = f"the 95% interval from the tested estimator, {_fmt(lo)} to {_fmt(hi)}, is not entirely on the claimed side of zero"
        if label == contract.NOT_SUPPORTED and not against:
            why = f"the 95% interval from the tested estimator, {_fmt(lo)} to {_fmt(hi)}, is not entirely on the opposite side of zero"
    if why:
        audit["floor"] = {"changed": True, "violations": [{
            "rule": "no_more_than_the_interval", "detail": f"The model said '{LABELS[label]}', but {why}.",
            "action": "The verdict was changed to Inconclusive."}]}
        v["label"] = contract.INCONCLUSIVE
        v["sentence"] = ("The data cannot settle this claim: " + why + ".")


def apply_verdict(report: dict, audit: dict) -> None:
    """Replace the rule-based verdict in a pipeline report (already JSON) with the model's, in place."""
    v = audit.get("verdict") or {}
    label = LABELS.get(v.get("label"), "Inconclusive")
    reasons = [str(x) for x in (v.get("reasons") or [])] + [f"Caveat: {x}" for x in (v.get("caveats") or [])]
    for x in (audit.get("floor") or {}).get("violations") or []:
        if x.get("detail"):
            reasons.append(f"Checked by code: {x['detail']} {x.get('action') or ''}".strip())
    assumed = [str(x) for x in (v.get("assumed") or [])]
    caveat = (assumed[0][0].lower() + assumed[0][1:]).rstrip(".") if assumed and label == "Supported" else None
    report["verdict"] = {"label": label, "sentence": str(v.get("sentence") or ""), "sub": [], "reasons": reasons,
                         "caveat": caveat}
    report["verdict_source"] = "model"


def public(audit: dict) -> dict:
    """What the web app stores next to the report: the reading, the verdict, what the model did and saw, and what
    the code check changed. Large fields the page does not need are dropped."""
    keep = ("status", "error", "engine", "reading", "verdict", "draft_verdict", "floor", "usage", "asked", "screen")
    out = {k: audit.get(k) for k in keep}
    out["evidence"] = [{"id": e.get("id"), "tool": e.get("tool"), "args": e.get("args"), "ok": e.get("ok"),
                        "lines": e.get("lines"), "error": e.get("error")} for e in audit.get("evidence") or []]
    return out
