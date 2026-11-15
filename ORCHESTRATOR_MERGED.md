# Merged additions to the orchestrator (parts F, G, H, W, P)

Read `ORCHESTRATOR.md` first; this file adds to it. Same rules: one owner per file, never edit
another owner's file, numbers only (no pass/fail/concern words in anything the model reads),
nothing raises out of a public function unless stated. Parts F, G, H are new files with no imports
from each other. W wires them in. Work in the existing working tree (branch `opus-orchestrator`);
do not commit, do not switch branch, do not use the web, do not run any `chatoverflow` command.

| Part | Files | Public surface |
|---|---|---|
| F layout | `orchestrator/layout.py`, `tests/test_orch_layout.py` | `TOOL_SPEC`, `run(df, args) -> (lines, numbers)` |
| G helpers | `orchestrator/helpers.py`, `tests/test_orch_helpers.py` | `TOOL_SPEC`, `run(df, args, writer=None, ...) -> list[dict]` |
| H screen | `orchestrator/screen.py`, `tests/test_orch_screen.py` | `TOOL_SPEC`, `charts_for(item, df)`, `draw(df, args)` |
| W wiring | `orchestrator/loop.py`, `prompt.py`, `contract.py`, `backend/app.py`, `backend/audit_cli.py`, `tests/test_orch_loop.py`, `tests/test_orch_api.py`, `tests/test_orch_followup.py` | tools offered to the model, `followup(...)`, API route, CLI flags |
| P page | `frontend/audit.js`, `frontend/index.html`, `frontend/style.css`, `frontend/sample_audit.json` | charts on the audit page, follow-up box |

Each `TOOL_SPEC` is one Anthropic tool definition (`name`, `description`, `input_schema`,
`"strict": True`, `additionalProperties: False`, every property in `required`, nullable via `anyOf`
with null). Look at `ASK_USER_SPEC` in `loop.py` for the style.

Tests: from `backend/`, `python -m pytest tests/test_orch_<part>.py -q -p no:cacheprovider`. Run
only your own test file; others are editing the tree at the same time. Set `LOKY_MAX_CPU_COUNT=4`
to silence a joblib warning on Windows.

## F. layout.py : tool `layout_facts`

Design-free facts about HOW TREATMENT IS LAID OUT in the table, so the model can form design
hypotheses when the description is thin. Args: `{"treatment": str, "outcome": str}`.
`run(df, args) -> (lines: list[str], numbers: dict)`; raises `ValueError` with a plain message on
bad args. Must finish in under about 5 s on 10,000 rows x 30 columns. Never names a design as the
answer and never says which columns to adjust for. `numbers` keys (lists may be empty):

* `thresholds`: `[{"column", "cutoff", "share_consistent", "n_below_near", "n_above_near", "window"}]`
  numeric columns on which a 0/1 treatment switches at a single value (reuse `data.find_cutoff`,
  `data._dense_at`, `data._sharp_at`; see `agent.structure_facts`).
* `panels`: `[{"unit", "time", "n_units", "n_periods", "share_units_switching_on",
  "share_units_switching_off", "n_never_treated", "start_times": [..]}]` column pairs that identify a
  row and within which treatment changes over time.
* `treatment_predictability`: `{"value", "metric": "auc"|"r2", "n_columns", "per_column": [{"column",
  "std_diff_or_corr", "p_value"}]}` cross-validated, from all columns except treatment and outcome
  (and except the unit and time columns of a detected panel, and row-id columns).
* `instrument_candidates`: `[{"column", "first_stage_f", "predictability_from_others",
  "assoc_with_outcome", "p_value_outcome"}]` columns that move treatment and are themselves NOT
  predictable from the remaining columns. The line must say that whether such a column affects the
  outcome only through treatment cannot be tested from data.
* `associated_columns`: `[{"column", "std_diff_or_corr", "p_value"}]` columns strongly associated with
  treatment. The line must say that numbers cannot tell a cause of treatment from a consequence of
  it, and that this is a question for the user.

Tests: build small simulated tables (sharp cutoff; staggered panel; randomized; confounded; random
encouragement column plus hidden confounder; a mediator) and assert each fact is found, and that
a randomized table reports low predictability and no threshold. The generators in
`../../benchmarks/harness/exp6/build6.py` show the shapes (read it, do not import from it).

## G. helpers.py : tool `custom_checks`

For a question no calculator covers. Args: `{"questions": [{"id": str, "question": str}]}` (1 to 4).
`run(df, args, writer=None, model="claude-haiku-5-5", timeout=150, max_workers=4) -> list[dict]`,
one dict per question, in order, never raises:
`{"id", "question", "ok", "lines": [str], "numbers": dict, "error": str|None, "code": str, "seconds": float}`.

How one question runs:

1. `writer(system: str, prompt: str) -> str` returns Python source (inside one python code fence, or
   bare). The default writer calls headless Claude Code with every tool off: see
   `headless.find_claude()` and how `HeadlessClient._turn` builds its command (`-p --model <model>
   --tools "" --strict-mcp-config --system-prompt-file <file> --output-format json`, prompt on
   stdin, `ANTHROPIC_API_KEY` removed from the environment, cwd a temp folder); the reply text is
   `json.loads(stdout)["result"]`. The system prompt is short. The prompt holds the question, the
   column list with types, and 5 sample rows. The script reads `data.csv` from the current folder,
   may use pandas, numpy, scipy and statsmodels, and must itself write `result.json` with the keys
   listed in `../.claude/agents/check-writer.md` (read that file: question, method, statistic,
   estimate, std_error, ci95_low, ci95_high, p_value, n, detectable, alternatives, limitations,
   error). It reports evidence only: no judgement words, no significance cutoffs, every
   specification tried is reported.
2. OUR code writes the script to a fresh temp folder next to a copy of the table and runs it with
   `sys.executable -I -X utf8`, cwd = that folder, with the timeout.
3. If the run fails or `result.json` is missing or invalid, ONE repair round: the writer gets the
   script and the tail of the error.
4. `numbers` = the parsed result.json; `lines` = a few plain sentences built from it (method,
   statistic, estimate with interval, n, detectable, alternatives, limitations).

Questions run in parallel threads. Temp folders are removed afterwards.

Tests use a fake `writer` (no network): a good script; a script that fails and is then repaired;
one that never works; a timeout; three questions in parallel keeping order. Add one test marked
`@pytest.mark.skipif(not os.environ.get("CAUSALITYAI_LIVE"))` that uses the real default writer.

## H. screen.py : charts the model can put on the page

Chart dicts follow `schema.Chart` as a plain dict (`kind` in bar|hist|scatter|line|forest, `title`,
`data`, `x_label`, `y_label`, `note`). Read the docstring of `Chart` in `schema.py` for each kind's
`data` shape and `frontend/charts.js` for what the page draws. Keep every array at or under 400
points (bin or sample).

* `charts_for(item: dict, df) -> list[dict]`: the standard charts for one evidence item (never
  raises, `[]` when none). `estimate_effect` -> a forest of the estimate and its `comparisons`, plus
  for `rd` a binned scatter of the outcome against the running column with the cutoff marked, for
  `did` the mean outcome per period for ever-treated and never-treated units (line), for `iv` a bar
  of treated share and mean outcome by instrument value. `design_tests` with event-study
  coefficients in a test's `metrics` -> a line with its interval. `general_checks` with per-column
  standardized differences -> a bar. Read `evidence.py` for the real `numbers` shapes (call the
  tools on a `causality.sim` table to see them); do not edit it.
* `TOOL_SPEC` for tool `show_chart`, args `{"evidence_id": str|null, "kind": str|null, "x": str|null,
  "y": str|null, "group": str|null, "title": str, "note": str}`: either an evidence id (the standard
  charts for that item) or a custom chart from columns.
* `draw(df, args) -> dict`: the custom chart computed from the table (hist of x, optionally by
  group; scatter of y on x, binned means when large; line of mean y by x per group; bar of mean y by
  x category). Raises `ValueError` with a plain message on bad args.

## W. wiring

* `contract.py`: add `LAYOUT_FACTS = "layout_facts"`, `CUSTOM_CHECKS = "custom_checks"`,
  `SHOW_CHART = "show_chart"`, and to `empty_report()`: `"screen": []` (list of
  `{"evidence_id": str|None, "chart": {...}, "note": str}` in the order shown) and `"followups": []`
  (list of `{"question": str}`).
* `loop.py`: offer the three new tools next to the evidence tools. Import each part lazily; if a
  part cannot be imported its tool is simply not offered. `layout_facts` -> one evidence item (tool
  `layout_facts`). `custom_checks` -> ONE EVIDENCE ITEM PER QUESTION (tool `custom_check`, args
  `{"question"}`, `numbers` = the result plus `"code"` and `"seconds"`), ids continue e1, e2, ...; the
  tool result sent to the model lists each id with its lines. `show_chart` -> append to
  `report["screen"]` (with an evidence id: every chart from `screen.charts_for`; otherwise
  `screen.draw`); the result to the model is a short confirmation or the plain error.
  `run_audit(..., helper_writer=None)` and `resume(..., helper_writer=None)` pass the writer through.
* Keep `state` in the report when `status == "done"` so a follow-up can continue.
  `followup(df, state, question, client=None, max_turns=16, helper_writer=None) -> report` appends the
  user's question to the conversation; the model may call tools again and must end with
  `submit_verdict` (the floor rules run again); earlier evidence, screen entries and ids are kept.
* `prompt.py`: call `layout_facts` early; if neither the description nor the layout settles how
  treatment was assigned, `ask_user`; a column strongly tied to treatment may be a cause or a
  consequence of it, and only the user can say which; use `custom_checks` when no tool covers the
  question (for example another comparison group after a design test looks bad); use `show_chart`
  for the two to four pictures that would help a non-statistician follow the verdict.
* `app.py`: `POST /api/audit/{session_id}/followup` (JSON `{"question": str}`). Check which client the
  API uses today when `ANTHROPIC_API_KEY` is not set; it must fall back to
  `headless.HeadlessClient()`.
* `audit_cli.py`: `--headless` (use `HeadlessClient`), `--auto-answer TEXT` (answer every question
  the model asks with TEXT and continue, at most 2 rounds), `--followup TEXT` (repeatable, asked
  after the verdict), `--json-out FILE` (write the full report as JSON). Print total seconds at the end.
* Tests with `FakeClient` and a fake writer only. The existing tests in your files must keep passing.

## P. page

* Render `report.screen` as a "What the data look like" section with the existing drawing code in
  `charts.js` (see how `app.js` calls it; do not edit `charts.js`, and add at most one small hook to
  `app.js`).
* A follow-up box under the verdict: posts to `/api/audit/{session_id}/followup`, then re-renders;
  earlier follow-up questions are listed.
* `custom_check` evidence items show their lines, and their code in an expandable block.
* Update `sample_audit.json` with a `screen` (two charts), one `custom_check` item and one follow-up,
  so the page can be checked without the server.
