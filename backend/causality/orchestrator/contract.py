"""Shared shapes for the orchestrator. Every part imports from here; nobody edits this file
without saying so in their report. All values are plain JSON types (dict, list, str, number, bool, None).
"""
from __future__ import annotations

MODEL = "claude-opus-5-5"
DESIGNS = ("experiment", "did", "iv", "rd", "observational")
ROLE_KEYS = ("treatment", "outcome", "unit", "time", "instrument", "running", "cluster", "strata")

# Verdict labels the model may choose from. The floor rules may only move a label to INCONCLUSIVE.
SUPPORTED = "Supported"                      # the interval backs the claimed direction
NOT_SUPPORTED = "Not supported"              # the interval backs the opposite of the claim
NO_EFFECT = "No effect"                      # the interval is tight around zero
INCONCLUSIVE = "Inconclusive"                # the data cannot settle it
LABELS = (SUPPORTED, NOT_SUPPORTED, NO_EFFECT, INCONCLUSIVE)

# How the model may read one piece of evidence.
READINGS = ("supports", "concern", "fails", "not informative")

# Evidence tool names (evidence.py implements them; loop.py offers them to the model).
EVIDENCE_TOOLS = ("table_facts", "general_checks", "structure_facts", "design_tests", "estimate_effect",
                  "sensitivity", "explore")
# Control tools handled by loop.py itself.
ASK_USER, SUBMIT_VERDICT = "ask_user", "submit_verdict"
# Optional tools from layout.py, helpers.py and screen.py (offered only when the part loads).
LAYOUT_FACTS = "layout_facts"
CUSTOM_CHECKS = "custom_checks"              # one evidence item per question, tool "custom_check"
SHOW_CHART = "show_chart"


def evidence_item(id: str, tool: str, args: dict, ok: bool = True, lines: list | None = None,
                  numbers: dict | None = None, error: str | None = None) -> dict:
    """One tool result. `lines` is what the model reads (plain sentences with numbers, NO pass/fail
    words). `numbers` is what the floor rules and the page read. See ORCHESTRATOR.md for the keys."""
    return {"id": id, "tool": tool, "args": args, "ok": ok, "lines": lines or [], "numbers": numbers or {},
            "error": error}


def empty_verdict() -> dict:
    return {"label": INCONCLUSIVE, "sentence": "", "estimate_evidence_id": None, "claimed_direction": None,
            "reasons": [], "assumed": [], "caveats": [], "judgments": []}


def empty_report(claim: str = "", context: str = "") -> dict:
    return {
        "status": "done",              # "done" | "needs_input" | "error"
        "session_id": None,            # set by the API layer
        "claim": claim, "context": context,
        "questions": [],               # [{"id", "question", "options": [str]}] when status == "needs_input"
        "reading": {"treatment": None, "outcome": None, "design": None, "design_reason": "", "roles": {},
                    "covariates": None, "excluded": []},     # excluded: [{"column", "reason"}]
        "draft_verdict": None,         # what the model submitted, untouched
        "verdict": None,               # after the floor rules
        "floor": {"changed": False, "violations": []},       # violations: [{"rule", "detail", "action"}]
        "evidence": [],                # [evidence_item, ...] in call order
        "transcript": [],              # [{"role": "assistant"|"user", "text"}] visible text only
        "screen": [],                  # [{"evidence_id": str|None, "chart": {...}, "note": str}] in order shown
        "followups": [],               # [{"question": str}] asked after the verdict, in order
        "usage": {"model": MODEL, "turns": 0, "input_tokens": 0, "output_tokens": 0},
        "state": None,                 # opaque state: resume ("needs_input") or follow-up ("done")
        "error": None,
    }
