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

## What it does

| Stage | Module |
|---|---|
| Read the table: column types, true/false → 0/1, median imputation with `__missing` flags (never for treatment, outcome or other role columns), one-hot for ≤ 10 levels, duplicates counted and kept | `data.py` |
| Declared design and roles checked against the data (conflicts become **failed** assumptions); otherwise the design is inferred from the data first, column names only as whole-word hints; ties are asked; never "experiment" unless you say treatment was randomized | `data.py` |
| Read the claim with rules (no language model): treatment, outcome, direction, size and unit, bound, shape, population, hedges, negation; column matches scored, confirmation asked below 0.8; mediation, counterfactual and treatment-vs-treatment claims flagged as not testable | `claim.py` |
| Balance (one joint permutation test decides; per-column differences vs chance; under randomization a column treatment strongly predicts is treated as post-treatment) | `checks/balance.py` |
| Overlap (two propensity models, percentile common support, trimming, where it breaks down) | `checks/overlap.py` |
| Hidden confounding: robustness value, benchmark bounds, E-value — never a pass | `checks/sensitivity.py` |
| Missing outcomes: overall/differential attrition vs WWC boundaries, Lee and Manski bounds; any concern caps the verdict | `checks/attrition.py` |
| Interference: spillover tested only when the treated share per cluster was randomized; otherwise assumed | `checks/interference.py` |
| Reading list (never changes the grade; online search opt-in, sends only treatment/outcome words) | `checks/literature.py` |
| Design assumptions and tests (pre-trends, bunching + covariate jumps at a cutoff, first-stage strength), grade A–F by the worst problem (observational tops out at C), assumed causal diagram | `design_assess.py` |
| Estimation: Lin-adjusted comparison (cluster-aware) · Callaway–Sant'Anna with TWFE comparison · 2SLS with Anderson–Rubin set · local linear RD with robust bias-corrected CI · cross-fitted AIPW · partially linear DML + dose-response curve. Zero-width intervals are refused | `estimate.py` |
| Verdict: identification first, then direction, size (90% CI within ±½ claimed / claim outside 95% CI), equivalence for no-effect claims (±0.1 SD), scope caps, "if no unmeasured confounding" | `verdict.py` |
| Next study, powered on the **claimed** effect, never the observed one | `suggest.py` |
| Explore: FCI main graph, PC and LiNGAM second views, subsampling stability, per-pair sanity checks; uncertain edges never drawn as arrows | `explore.py` |
| Orchestration and API | `pipeline.py`, `app.py` |
| Page with hand-written SVG charts, JSON and standalone-HTML export | `frontend/` |

The shared result format is `backend/causality/schema.py`; the module contract is `CONTRACT.md`.
Known-truth simulations are in `backend/causality/sim.py`, and `tests/test_acceptance.py` runs
each through the whole pipeline.

## Limits to know

* It reproduces a design, not a reference implementation; numbers will not match other tools case for case.
* The rules were written to fit a claim benchmark; on outside test suites they did worse than a plain analyst.
* The WWC attrition boundary tables were entered from memory (believed within ~0.3 points);
  check them against the published WWC Standards Handbook before relying on exact cut-offs.
* A confounded observational dataset with no measured confounder (see `sim.observational_hidden`)
  will still read "Partly supported … if no unmeasured confounding": the data cannot reveal it.
  The sensitivity card and the assumed panel are what make that visible.
