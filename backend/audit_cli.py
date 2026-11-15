"""Audit a claim with Opus as orchestrator, from the command line.

  python audit_cli.py data.csv --claim "X increases Y" [--context TEXT | --context-file FILE]
                      [--answer id=text ...] [--auto-answer TEXT] [--followup TEXT ...]
                      [--headless] [--json-out FILE] [--full]

The model reads the claim, maps it to columns, picks the design and judges the evidence. Code
computes every number and applies three floor rules. Needs ANTHROPIC_API_KEY in the environment,
or --headless to run each turn through Claude Code on a Claude subscription.

If the model needs to ask something the data cannot show (for example how treatment was assigned):
  * answers given with --answer are used first, then --auto-answer (at most 2 rounds);
  * in a terminal, the questions are asked here and the audit continues with your answers;
  * otherwise the questions are printed and the exit code is 3. Run again with --answer id=text.
After the verdict, each --followup question is asked in turn.

Exit codes: 0 done, 1 the audit failed, 2 bad input, 3 answers needed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")


def _parse_answers(items) -> dict:
    answers = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"'{item}' is not in the form id=text.")
        key, text = item.split("=", 1)
        if not key.strip():
            raise ValueError(f"'{item}' has no question id before '='.")
        answers[key.strip()] = text.strip()
    return answers


def _first_line(item: dict) -> str:
    if not item.get("ok", True):
        return f"could not run: {item.get('error') or 'no reason given'}"
    lines = item.get("lines") or []
    return lines[0] if lines else "(no lines)"


def summary(report: dict) -> str:
    """A compact plain-text view of a report."""
    out = []
    status = report.get("status")
    if status == "error":
        out.append(f"The audit did not finish: {report.get('error') or 'no reason given'}")
    reading = report.get("reading") or {}
    if any(reading.get(k) for k in ("treatment", "outcome", "design")):
        out.append("Reading")
        out.append(f"  treatment: {reading.get('treatment')}   outcome: {reading.get('outcome')}")
        out.append(f"  design: {reading.get('design')}" +
                   (f" ({reading.get('design_reason')})" if reading.get("design_reason") else ""))
        roles = {k: v for k, v in (reading.get("roles") or {}).items()
                 if v is not None and k not in ("treatment", "outcome")}
        if roles:
            out.append("  roles: " + ", ".join(f"{k}={v}" for k, v in roles.items()))
        cov = reading.get("covariates")
        out.append("  adjusted for: " + ("every other column" if cov is None else (", ".join(cov) or "nothing")))
        for ex in reading.get("excluded") or []:
            out.append(f"  left out {ex.get('column')}: {ex.get('reason')}")
    verdict = report.get("verdict")
    if verdict:
        out.append("")
        out.append(f"Verdict: {verdict.get('label')}")
        if verdict.get("sentence"):
            out.append(f"  {verdict['sentence']}")
        for r in verdict.get("reasons") or []:
            out.append(f"  - {r}")
        if verdict.get("assumed"):
            out.append("  Assumed, not tested:")
            for a in verdict["assumed"]:
                out.append(f"    - {a}")
        for c in verdict.get("caveats") or []:
            out.append(f"  caveat: {c}")
    floor = report.get("floor") or {}
    if floor.get("violations"):
        out.append("")
        out.append("Floor rules changed the model's verdict:" if floor.get("changed")
                   else "Floor rules added to the model's verdict:")
        for v in floor["violations"]:
            out.append(f"  {v.get('rule')}: {v.get('detail')} -> {v.get('action')}")
        draft = report.get("draft_verdict") or {}
        if floor.get("changed") and draft.get("label"):
            out.append(f"  The model had said: {draft.get('label')}. {draft.get('sentence') or ''}".rstrip())
    evidence = report.get("evidence") or []
    if evidence:
        out.append("")
        out.append("Evidence")
        for item in evidence:
            out.append(f"  {item.get('id')} {item.get('tool')}: {_first_line(item)}")
    if status == "needs_input":
        out.append("")
        out.append("The model needs answers the data cannot give:")
        for q in report.get("questions") or []:
            opts = q.get("options") or []
            out.append(f"  [{q.get('id')}] {q.get('question')}" + (f"  (options: {' | '.join(opts)})" if opts else ""))
    screen = report.get("screen") or []
    if screen:
        out.append("")
        out.append(f"Charts for the page: {len(screen)}")
        for s in screen:
            c = s.get("chart") or {}
            out.append(f"  {c.get('kind')}: {c.get('title') or 'untitled'}"
                       + (f" [{s.get('evidence_id')}]" if s.get("evidence_id") else ""))
    followups = report.get("followups") or []
    if followups:
        out.append("")
        out.append("Follow-up questions asked:")
        for f in followups:
            out.append(f"  - {f.get('question')}")
        last = next((t.get("text") for t in reversed(report.get("transcript") or [])
                     if t.get("role") == "assistant" and t.get("text")), None)
        if last:
            out.append(f"  The model's last message: {last}")
    usage = report.get("usage") or {}
    if usage.get("turns"):
        out.append("")
        out.append(f"{usage.get('model')}: {usage.get('turns')} turns, {usage.get('input_tokens')} tokens in, "
                   f"{usage.get('output_tokens')} tokens out")
    return "\n".join(out).strip()


def _ask(questions: list) -> dict:
    answers = {}
    print("The model needs answers the data cannot give.")
    for q in questions:
        print()
        print(q.get("question"))
        opts = q.get("options") or []
        for i, o in enumerate(opts, 1):
            print(f"  {i}. {o}")
        raw = input("Your answer" + (" (a number or your own words)" if opts else "") + ": ").strip()
        if opts and raw.isdigit() and 1 <= int(raw) <= len(opts):
            raw = opts[int(raw) - 1]
        answers[str(q.get("id"))] = raw
    return answers


def main(argv=None, client=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", help="the table (csv, tsv, xlsx, parquet or json)")
    p.add_argument("--claim", required=True, help="the claim to audit, in plain words")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--context", default="", help="what you know about how the data was collected")
    g.add_argument("--context-file", help="a text file with that description")
    p.add_argument("--answer", action="append", default=[], metavar="ID=TEXT",
                   help="an answer to a question the model asked (repeat for several)")
    p.add_argument("--auto-answer", metavar="TEXT",
                   help="answer every question the model asks with TEXT and continue (at most 2 rounds)")
    p.add_argument("--followup", action="append", default=[], metavar="TEXT",
                   help="a follow-up question asked after the verdict (repeat for several)")
    p.add_argument("--headless", action="store_true",
                   help="run the model through headless Claude Code (no API key needed)")
    p.add_argument("--json-out", metavar="FILE", help="write the full report as JSON to FILE")
    p.add_argument("--full", action="store_true", help="print the whole report as JSON")
    a = p.parse_args(argv)
    started = time.perf_counter()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    from causality import agent
    from causality.data import InputError
    from causality.schema import to_jsonable
    try:
        df = agent.load(a.data)
        context = a.context or ""
        if a.context_file:
            with open(a.context_file, encoding="utf-8") as f:
                context = f.read()
        given = _parse_answers(a.answer)
    except (InputError, ValueError, OSError) as e:
        print(f"Cannot run: {e}")
        return 2

    from causality.orchestrator import loop
    if client is None and a.headless:
        from causality.orchestrator import headless
        client = headless.HeadlessClient()
    auto_rounds = {"n": 0}

    def settle(report):
        """Answer questions while the model asks them: --answer first, then --auto-answer, then the terminal."""
        while report.get("status") == "needs_input":
            questions = report.get("questions") or []
            ids = [str(q.get("id")) for q in questions]
            if given and any(i in given for i in ids):
                answers = {i: given[i] for i in ids if i in given}
                for i in answers:
                    given.pop(i)
            elif a.auto_answer is not None and auto_rounds["n"] < 2:
                auto_rounds["n"] += 1
                answers = {i: a.auto_answer for i in ids}
            elif sys.stdin is not None and sys.stdin.isatty():
                answers = _ask(questions)
            else:
                break
            report = loop.resume(df, report.get("state"), answers, client=client)
        return report

    try:
        report = settle(loop.run_audit(df, a.claim, context, client=client))
        for question in a.followup:
            if report.get("status") != "done":
                break
            report = settle(loop.followup(df, report.get("state"), question, client=client))
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Nothing was concluded.")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"The audit failed ({type(e).__name__}: {e}). Nothing was concluded.")
        return 1

    full = to_jsonable(report)
    if a.json_out:
        try:
            with open(a.json_out, "w", encoding="utf-8") as f:
                json.dump(full, f, indent=2, ensure_ascii=False)
        except OSError as e:
            print(f"Could not write {a.json_out}: {e}", file=sys.stderr)
    if a.full:
        print(json.dumps(full, indent=2, ensure_ascii=False))
    else:
        print(summary(report))
    if report.get("status") == "needs_input" and not a.full:
        print("\nRun again with --answer <id>=<text> for each question.")
    seconds = f"Total time: {time.perf_counter() - started:.1f} s"
    print(seconds, file=sys.stderr if a.full else sys.stdout)
    if report.get("status") == "needs_input":
        return 3
    return 1 if report.get("status") == "error" else 0


if __name__ == "__main__":
    sys.exit(main())
