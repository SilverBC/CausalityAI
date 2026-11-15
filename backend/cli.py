"""Command-line entry to CausalityAI, for scripts and for coding agents that call it as a tool.

    python cli.py data.csv --claim "X increases Y" [--context "study description" | --context-file task.md]
                  [--design experiment|did|iv|rd|observational]
                  [--treatment COL] [--outcome COL] [--unit COL] [--time COL] [--instrument COL]
                  [--running COL] [--cutoff NUMBER] [--cluster COL] [--strata COL]
                  [--post-treatment COL1,COL2] [--randomized] [--full]

Prints a compact JSON summary: the design and why, the grade, each check with its status,
the estimate with intervals, the verdict, what is assumed and not tested, and any question the
tool needs answered (answer it by declaring the role or design and running again).
--full prints the whole report instead.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore")


def summarize(rep) -> dict:
    from causality.schema import to_jsonable
    r = to_jsonable(rep)
    out: dict = {"mode": r["mode"]}
    if r.get("needs_input"):
        out["questions"] = [{"about": q.get("role") or q.get("kind"), "question": q.get("question"),
                             "options": (q.get("options") or [])[:12]} for q in r["needs_input"]]
    if r.get("errors"):
        out["errors"] = r["errors"]
    c = r.get("claim")
    if c:
        out["claim_read_as"] = {k: c.get(k) for k in ("treatment_col", "outcome_col", "direction", "size", "size_unit",
                                                       "population_phrase", "instrument_col", "design_hint", "testable",
                                                       "untestable_reason") if c.get(k) not in (None, "", [])}
    d = r.get("design")
    roles = (r.get("data") or {}).get("roles") or {}
    if d:
        out["design"] = {"kind": d["kind"], "declared_by_user": d["declared"], "grade": d.get("grade"),
                         "grade_reasons": (d.get("grade_reasons") or [])[:3], "why": (d.get("reasons") or [])[:4],
                         "fit_scores": {k: round(v, 2) for k, v in (d.get("scores") or {}).items() if v},
                         "conflicts_with_declaration": [a["detail"] for a in d.get("conflicts") or []]}
    out["roles"] = {k: v for k, v in roles.items() if v not in (None, [], False)}
    out["adjustment_set"] = ((r.get("data") or {}).get("adjustment_set") or [])[:40]
    out["checks"] = [{"check": s["key"], "status": s.get("status") or "information",
                      "affects_grade": s.get("affects_grade", True),
                      "summary": (s.get("error") or s.get("summary") or "")[:400]} for s in r.get("sections") or []]
    e = r.get("estimate")
    if e:
        out["estimate"] = {"method": e.get("method"), "target": e.get("target"), "scope": e.get("scope"),
                           "estimate": e.get("estimate"), "se": e.get("se"), "ci95": e.get("ci95"),
                           "p_value": e.get("p_value"), "n": e.get("n"), "refused": e.get("refused"),
                           "refusal_reason": e.get("refusal_reason"),
                           "other_methods": [{"method": x.get("method"), "estimate": x.get("estimate"), "ci95": x.get("ci95")}
                                             for x in (e.get("comparisons") or [])[:5]]}
    v = r.get("verdict")
    if v:
        out["verdict"] = {"label": v["label"], "sentence": v["sentence"]}
    out["assumed_not_tested"] = [a["name"] for a in r.get("assumed_panel") or []]
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Audit a causal claim against a data table.")
    p.add_argument("file")
    p.add_argument("--claim", default="")
    p.add_argument("--context", default="")
    p.add_argument("--context-file")
    p.add_argument("--design", default=None)
    for role in ("treatment", "outcome", "unit", "time", "instrument", "running", "cluster", "strata", "saturation"):
        p.add_argument(f"--{role}")
    p.add_argument("--cutoff", type=float)
    p.add_argument("--post-treatment", default="")
    p.add_argument("--randomized", action="store_true")
    p.add_argument("--full", action="store_true")
    a = p.parse_args(argv)

    from causality import data, pipeline
    from causality.data import InputError
    from causality.schema import to_jsonable
    try:
        df = data.load_table(open(a.file, "rb").read(), a.file)
        context = a.context
        if a.context_file:
            context = (context + "\n" + open(a.context_file, encoding="utf-8").read()).strip()
        roles = {k: getattr(a, k) for k in ("treatment", "outcome", "unit", "time", "instrument", "running",
                                             "cluster", "strata", "saturation") if getattr(a, k)}
        if a.cutoff is not None:
            roles["cutoff"] = a.cutoff
        roles["post_treatment"] = [c.strip() for c in a.post_treatment.split(",") if c.strip()]
        roles["randomized"] = a.randomized
        claim = a.claim
        if not claim and roles.get("treatment") and roles.get("outcome"):
            claim = f"{roles['treatment']} affects {roles['outcome']}"
        rep = pipeline.run(df, claim_text=claim, design_decl=a.design, roles=roles,
                           confirm={"randomized": "yes"} if a.randomized else {}, context=context)
        out = to_jsonable(rep) if a.full else summarize(rep)
    except InputError as e:
        out = {"error": str(e)}
    except Exception as e:  # noqa: BLE001
        out = {"error": f"The analysis could not run ({type(e).__name__}: {e})."}
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
