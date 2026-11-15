# Orchestrator contract (branch `opus-orchestrator`)

The model (Opus 5.5) reads the claim, maps it to columns, picks the design, chooses the
adjustment columns, decides which calculators to run, and judges what the numbers mean.
Code computes numbers and enforces three floor rules. Nothing in code guesses a design, parses
the claim, or attaches pass/fail labels to evidence.

Everything lives in `backend/causality/orchestrator/`. Shared shapes: `contract.py` (read it
first). The old pipeline (`pipeline.py`, `claim.py`, `verdict.py`, `/api/analyze`) stays as is.
Do not edit: `contract.py`, `schema.py`, `data.py`, `sim.py`, `agent.py`, `checks/*`,
`design_assess.py`, `estimate.py`, `explore.py`. If one of them has a bug, report it.

## Parts and owners (one owner per file; never edit another owner's file)

| Part | Files | Public surface |
|---|---|---|
| A evidence | `orchestrator/evidence.py`, `tests/test_orch_evidence.py` | `TOOL_SPECS`, `call(name, args, df, id) -> evidence_item` |
| B floor | `orchestrator/floor.py`, `tests/test_orch_floor.py` | `apply(draft_verdict, reading, evidence) -> (verdict, floor)` |
| C loop | `orchestrator/loop.py`, `orchestrator/prompt.py`, `tests/test_orch_loop.py` | `run_audit(...)`, `resume(...)`, `FakeClient` |
| D api | `backend/app.py` (add only), `backend/audit_cli.py`, `requirements.txt`, `README.md`, `tests/test_orch_api.py` | `POST /api/audit`, `POST /api/audit/{session_id}/answer` |
| E page | `frontend/audit.js`, `frontend/index.html`, `frontend/style.css`, `frontend/sample_audit.json` | "Audit with Opus" mode |

## A. evidence.py

`TOOL_SPECS: list[dict]` holds Anthropic tool definitions (`name`, `description`, `input_schema`)
for the seven names in `contract.EVIDENCE_TOOLS`. Descriptions say what is computed and when it
applies, in plain words.

`call(name: str, args: dict, df: pandas.DataFrame, id: str) -> dict` never raises; it returns
`contract.evidence_item(...)`. Bad arguments or failures give `ok=False` and `error` in plain words.

Arguments (all column names are names in the uploaded table):

| tool | args | builds on |
|---|---|---|
| `table_facts` | none | `agent.facts_card` |
| `general_checks` | `treatment, outcome, covariates?` | `agent.general_checks` |
| `structure_facts` | `treatment, outcome, instruments?, covariates?` | `agent.structure_facts` |
| `design_tests` | `design, roles, covariates?` | `agent.design_tests` |
| `estimate_effect` | `design, roles, covariates?` | `estimate.estimate` |
| `sensitivity` | `design, roles, covariates?` | `checks/sensitivity.run` |
| `explore` | none | `explore.run` |

`roles` is an object with any of `contract.ROLE_KEYS` plus `cutoff` (number).
`covariates` omitted or null = every non-role column; a list = adjust only for these.

`lines` are sentences with the numbers, no pass/fail/concern words. `numbers` keys the others rely on:

* `estimate_effect`: `design`, `roles`, `method`, `target`, `scope`, `n`, `estimate`, `se`,
  `p_value`, `ci` = `{"90": [lo, hi], "95": [...], "98": [...], "99": [...]}`, `refused` (bool),
  `refusal_reason`, `outcome_sd` (SD of the outcome in the rows used), `comparisons`
  (`[{"method", "estimate", "ci95"}]`).
* `design_tests`: `design`, `roles`, `tests` = `[{"key", "title", "statistic", "p_value", "metrics"}]`,
  `conflicts` (`[str]`), `untestable` (`[str]`, the assumptions no data can test).
* `sensitivity`: `robustness_value`, `robustness_value_alpha`, `e_value`, `benchmarks`, `untestable` (`[str]`).
* `general_checks`: `checks` = `[{"key", "title", "metrics"}]`, `untestable` (`[str]`).
* others: whatever is useful, small (no arrays over 400 items).

## B. floor.py

`apply(draft: dict, reading: dict, evidence: list[dict]) -> tuple[dict, dict]` returns
`(verdict, floor)`; `verdict` is a copy of the draft with only these changes, `floor` is
`{"changed": bool, "violations": [{"rule", "detail", "action"}]}`. Pure function, never raises.

The only rules (no other thresholds or grades; everything else is the model's call):

1. `assumed_is_not_tested`: every string in any evidence item's `numbers["untestable"]` must
   appear in `verdict["assumed"]`; missing ones are added. A judgment that reads a
   `sensitivity` item as `"supports"` is changed to `"not informative"`.
2. `needs_a_real_interval`: a label other than Inconclusive needs `estimate_evidence_id`
   pointing at an `estimate_effect` item with `ok`, not `refused`, finite `ci["95"]` of width > 0,
   computed with the same `design`, treatment and outcome as `reading`. Otherwise the label
   becomes Inconclusive.
3. `no_more_than_the_interval`: Supported needs the 95% interval strictly on the claimed side
   of zero (`claimed_direction` "increase" means lo > 0; "decrease" means hi < 0); Not supported
   needs it strictly on the other side; No effect needs it inside plus or minus 0.1 x `outcome_sd`.
   Otherwise the label becomes Inconclusive. `claimed_direction` is
   `"increase" | "decrease" | "none"` (none = the claim is "no effect").

When a rule changes the label, the sentence is replaced by a plain sentence saying what the
data can and cannot show; the model's own sentence stays in `draft_verdict`.

## C. loop.py

```
run_audit(df, claim, context="", client=None, model=contract.MODEL, max_turns=24) -> report
resume(df, state, answers: dict[str, str], client=None, max_turns=24) -> report
```

* `client` is an `anthropic.Anthropic()` by default (created lazily; a missing key or package
  gives a report with `status="error"` and a plain message). Tests pass `FakeClient(script)`,
  which replays a scripted list of assistant turns, so no test needs a key or the network.
* Manual tool loop: evidence tool calls go to `evidence.call` with ids `e1, e2, ...` and are
  appended to `report["evidence"]`; the tool result sent back to the model is the id plus the
  `lines` (or the error).
* `ask_user(questions=[{"id","question","options"}])` stops the loop: `status="needs_input"`,
  `questions` filled, `state` holds what `resume` needs (JSON-serializable; the table is not in it).
* `submit_verdict(reading=..., verdict=...)` ends the loop: shapes as in
  `contract.empty_report()["reading"]` and `contract.empty_verdict()`; `judgments` =
  `[{"evidence_id", "reading" in contract.READINGS, "why"}]`. Then `floor.apply` runs and fills
  `verdict` and `floor`.
* The system prompt (`prompt.py`) tells the model: ask the user only what the data cannot show
  (above all, how treatment was assigned); never call a design an experiment unless the user or
  the description says assignment was randomized; leave out covariates that are consequences of
  treatment and say why; every claim in the verdict must cite an evidence id; untested is never "passed".
* Running out of turns, or the model ending without `submit_verdict`, gives `status="error"`.

## D. api

* `POST /api/audit` (multipart: `file`, `claim`, `context`) returns a report. Keeps the table and
  state in an in-memory session store keyed by `session_id` (cap 50 sessions, oldest dropped).
* `POST /api/audit/{session_id}/answer` (JSON `{"answers": {"<question id>": "<text>"}}`) returns a report.
* The model call must not block the event loop (run in a thread).
* `audit_cli.py data.csv --claim "..." [--context ... | --context-file f] [--answer id=text ...]`.
* Tests inject a `FakeClient` (for example through `app.state.audit_client`), never the network.

## E. page

A second mode on the existing page, "Audit with Opus", in a new `audit.js` (do not rewrite
`app.js`; add one small hook if needed). Shows: questions with an answer box, the reading
(columns, design and why, covariates left out and why), the verdict and what the floor rules
changed, the evidence list (each tool call with its lines, expandable), what is assumed and
not tested, and the model's judgments next to the evidence they cite. `sample_audit.json` is a
full example report in the contract shape so the page can be developed without the server.

## Tests

From `backend/`: `python -m pytest tests/test_orch_<part>.py -q -p no:cacheprovider`. Use
`causality.sim` where the truth is known. Run only your own test file while others are working.
