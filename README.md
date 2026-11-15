# CausalityAI

Upload a table and, optionally, a sentence like *"X increases Y by 15% among Z"*.
With a sentence, CausalityAI **audits** it. Without one, it **explores** what may cause what.

The one rule: **never claim more than the data can show.** Every assumption gets one of four
statuses — *tested: OK*, *tested: concern*, *failed*, or *assumed (not tested)*. Things data can
never show — no hidden confounders, the exclusion restriction, continuity at a cutoff, no
interference between units, outcomes missing for reasons tied to the outcome — are always
*assumed*, and *assumed* is never displayed as a pass.

## Run

```bash
pip install -r requirements.txt
```

```bash
python -m uvicorn app:app --app-dir backend --port 8000
```

Open http://localhost:8000. To try it without your own data, download a simulated dataset with a
known truth from `http://localhost:8000/api/sample/<name>` (`rct`, `observational`, `did`, `iv`,
`rd`, `dose`, `cluster_saturation`, `chain`, …).

Tests (from `backend/`):

```bash
python -m pytest tests -q
```

## Audit with Opus

A second way to audit a claim, with Opus 5.5 (`claude-opus-5-5`) as the orchestrator. The contract
between the parts is `ORCHESTRATOR.md`; the code is in `backend/causality/orchestrator/`.

**The model decides:** which columns the claim is about, the design (experiment,
difference-in-differences, instrument, cutoff, or observational) and why, which columns to adjust
for and which to leave out (with a reason), which calculations to run, what to ask you, and what
each piece of evidence means. It asks only what the data cannot show, above all how treatment
was assigned, and it never calls a design an experiment unless you say assignment was randomized.

**Code computes:** every number (table facts, balance and overlap, design tests, the estimate with
its 90/95/98/99% intervals, sensitivity to hidden confounding, exploration). Each calculation is
kept as an evidence item with an id (`e1`, `e2`, ...), and every claim in the verdict must cite one.
Code never labels evidence as passed or failed; the model judges it.

**Three floor rules** are then applied to the model's verdict. They can only move it to
*Inconclusive* or add to what is assumed; the model's own verdict stays in the report:

1. *Assumed is not tested*: anything no data can test (for example no hidden confounders) is listed
   as assumed, and a sensitivity result is never read as support.
2. *Needs a real interval*: any answer other than Inconclusive must point at an estimate with a
   finite 95% interval of non-zero width, made with the same design, treatment and outcome.
3. *No more than the interval*: Supported needs the 95% interval wholly on the claimed side of zero;
   Not supported needs it wholly on the other side; No effect needs it within plus or minus 0.1
   standard deviations of the outcome.

Set your key first (the model calls go to the Anthropic API):

```bash
export ANTHROPIC_API_KEY=sk-ant-...        # PowerShell: $env:ANTHROPIC_API_KEY = "sk-ant-..."
```

Command line (from `backend/`):

```bash
python audit_cli.py data.csv --claim "the program raises earnings" --context "who got the program and how"
python audit_cli.py data.csv --claim "..." --context-file notes.txt --answer q1="assigned by lottery"
python audit_cli.py data.csv --claim "..." --full     # the whole report as JSON
```

In a terminal it asks the model's questions and continues; otherwise it prints them and exits
with code 3, so you can run it again with `--answer id=text`.

Page: start the server as above, open http://localhost:8000 and choose "Audit with Opus". The API
behind it is `POST /api/audit` (form fields `file`, `claim`, `context`) and
`POST /api/audit/{session_id}/answer` (JSON `{"answers": {"<question id>": "<text>"}}`). Both
return the report; uploads are kept in server memory only (the last 50 audits).

## What it does

| Stage | Module |
|---|---|
| Read the table: column types, true/false → 0/1, median imputation with `__missing` flags (never for treatment, outcome or other role columns), one-hot for ≤ 10 levels, duplicates counted and kept | `data.py` |
| Declared design and roles checked against the data (conflicts become **failed** assumptions); otherwise the design is inferred from the data first, column names only as whole-word hints; ties are asked; never "experiment" unless you say treatment was randomized | `data.py` |
| Read the claim with rules (no language model): treatment, outcome, direction, size and unit, bound, shape, population, hedges, negation, a named instrument, and design wording as a hint (randomization wording is confirmed with you first); column matches scored, confirmation asked below 0.8; mediation, counterfactual and treatment-vs-treatment claims flagged as not testable | `claim.py` |
| Balance (one joint permutation test decides; per-column differences vs chance; under randomization a column treatment strongly predicts is treated as post-treatment) | `checks/balance.py` |
| Overlap (average of two cross-fitted propensity models, percentile common support, trimming bound set by a fixed rule from sample size and number of covariates, where it breaks down) | `checks/overlap.py` |
| Hidden confounding: robustness value, benchmark bounds, E-value — never a pass | `checks/sensitivity.py` |
| Missing outcomes: overall/differential attrition vs WWC boundaries, Lee and Manski bounds; any concern caps the verdict | `checks/attrition.py` |
| Interference: spillover tested only when the treated share per cluster was randomized; otherwise assumed | `checks/interference.py` |
| Reading list (never changes the grade; online search opt-in, sends only treatment/outcome words, and is skipped when headers are placeholders like X1, X2) | `checks/literature.py` |
| Design assumptions and tests (pre-trends and placebo-in-time; bunching + covariate jumps at a cutoff; first-stage strength and a Durbin–Wu–Hausman note; Ramsey RESET to degree 3 on panel data; Boehmer–Musumeci–Poulsen and Corrado rank tests for one-off events), grade A–F by the worst problem (observational tops out at C), assumed causal diagram with confounder and possible-collider groups | `design_assess.py` |
| Estimation: Lin-adjusted comparison (cluster-aware) · Callaway–Sant'Anna with TWFE comparison · 2SLS with Anderson–Rubin set · local linear RD with robust bias-corrected CI · cross-fitted AIPW · partially linear DML + dose-response curve. Zero-width intervals are refused | `estimate.py` |
| Verdict: identification first, then direction on the 95% interval (90%, 98% and 99% are shown too); the claimed size is an unscored note; no-effect claims need the 95% interval within ±0.1 SD; scope caps; "if no unmeasured confounding" | `verdict.py` |
| Next study, powered on the **claimed** effect, never the observed one | `suggest.py` |
| Explore: FCI main graph, PC and LiNGAM second views, subsampling stability, per-pair sanity checks; uncertain edges never drawn as arrows | `explore.py` |
| Orchestration and API | `pipeline.py`, `app.py` |
| Page with hand-written SVG charts, JSON and standalone-HTML export | `frontend/` |

The shared result format is `backend/causality/schema.py`; the module contract is `CONTRACT.md`.
Known-truth simulations are in `backend/causality/sim.py`, and `tests/test_acceptance.py` runs
each through the whole pipeline.

## Limits to know

* The claimed size is not scored: "X raises Y by 60%" reads "Supported" when only the direction holds. The
  sentence and the size note say whether the claimed value is inside the 95% interval.
* RESET, the Durbin–Wu–Hausman test and the event-study tests are diagnostics. They never raise the grade and
  never count as a test of an untestable assumption.
* A one-off event (treatment on for one period) is estimated as the average abnormal outcome on the event
  date; the difference-in-differences estimate is kept only as a comparison, because it assumes treatment stays on.
* A placebo *group* test is not run: it needs you to name a group known to be unaffected.

* It reproduces a design, not a reference implementation; numbers will not match other tools case for case.
* The rules were written to fit a claim benchmark; on outside test suites they did worse than a plain analyst.
* The WWC attrition boundary tables were entered from memory (believed within ~0.3 points);
  check them against the published WWC Standards Handbook before relying on exact cut-offs.
* A confounded observational dataset with no measured confounder (see `sim.observational_hidden`)
  will still read "Partly supported … if no unmeasured confounding": the data cannot reveal it.
  The sensitivity card and the assumed panel are what make that visible.
