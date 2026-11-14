# Module contract

All parts talk through the dataclasses in `backend/causality/schema.py`. Do **not** edit
`schema.py`, `data.py`, `sim.py` or `units.py`; put extra fields in `Section.metrics`,
`Estimate.extra`, etc. If you find a bug in them, say so in your report.

The one rule: **never claim more than the data can show.** Every assumption has one of four
statuses (`Status.PASS`, `CONCERN`, `FAIL`, `ASSUMED`). `ASSUMED` is never a pass.
Things that can never be tested from data are always `ASSUMED` (`testable=False`): no hidden
confounders, the exclusion restriction, continuity of potential outcomes at a cutoff, no
interference between units (unless the saturation was randomized), outcomes missing for
reasons tied to the outcome.

Modules never raise on bad data: catch, and return a `Section` with `error="plain words"`.
Use `prep.adjustment_set` as the one adjustment set — never build another one, never add
post-treatment columns. Use `prep.complete()` for rows with observed treatment & outcome.

## Inputs every module gets

* `prep: Prepared` — from `data.prepare(...)`; `prep.roles` has resolved role column names
  (df column names). `prep.df` is numeric. `prep.treatment_is_binary()`.
* `design: Design` — `design.kind in {experiment, did, iv, rd, observational}`;
  `prep.roles.randomized` is True only if the user said so (experiment ⇔ randomized).
* `claim: Claim | None` — None in explore mode.

## Functions (file → signature → returns)

| file | signature | returns |
|---|---|---|
| `claim.py` | `parse_claim(text: str, prep: Prepared) -> Claim` | parsed claim, column matches with scores, estimand |
| `checks/balance.py` | `run(prep, design, claim) -> Section` | key `"balance"`; `metrics["suspected_post_treatment"]: list[str]` (df columns) |
| `checks/overlap.py` | `run(prep, design, claim) -> Section`; `propensity(prep) -> np.ndarray`; `trim_mask(prep) -> pd.Series[bool]` | key `"overlap"` |
| `checks/sensitivity.py` | `run(prep, design, claim, estimate: Estimate) -> Section` | key `"hidden_confounding"`; status is never PASS |
| `checks/attrition.py` | `run(prep, design, claim) -> Section` | key `"missing_outcomes"`; `caps_verdict=True` on any concern |
| `checks/interference.py` | `run(prep, design, claim) -> Section` | key `"interference"` |
| `checks/literature.py` | `run(prep, design, claim, online: bool=False) -> Section` | key `"literature"`, `affects_grade=False` |
| `design_assess.py` | `assess(prep, design, claim) -> tuple[list[Assumption], list[Section]]` | design assumptions + design-specific test cards |
| `design_assess.py` | `grade(design, sections: list[Section]) -> tuple[str, list[str]]` | grade A–F and reasons |
| `design_assess.py` | `dag(prep, design, claim) -> Chart` | assumed causal diagram (kind `"dag"`) |
| `estimate.py` | `estimate(prep, design, claim) -> Estimate` | the estimate for the design |
| `verdict.py` | `make_verdict(claim, design, estimate, sections) -> Verdict` | |
| `suggest.py` | `suggest_study(prep, design, claim, estimate) -> Section` | key `"next_study"` |
| `explore.py` | `run(prep) -> Section` | key `"explore"`, has a `dag` chart |

`units.claimed_effect(claim, estimate)` converts the claimed size to signed outcome units.

## Order the orchestrator (`pipeline.py`) runs things

1. `data.prepare` → `claim.parse_claim` (sets treatment/outcome roles) → apply population filter
   → `data.resolve_design`.
2. `balance.run` first; any `suspected_post_treatment` → `data.mark_post_treatment` (adjustment
   set recomputed once, then frozen).
3. `overlap`, `attrition`, `interference`, `literature`, `design_assess.assess`.
4. `estimate.estimate` → `sensitivity.run(..., estimate)`.
5. `design.grade` via `design_assess.grade`, `verdict.make_verdict`, `suggest.suggest_study`,
   `design_assess.dag`.
6. Assumed panel = every assumption with status ASSUMED across design + sections.

Charts use the `Chart` kinds documented in `schema.py` (bar, hist, scatter, line, forest, dag).
Keep chart data small (≤ 400 points; bin or subsample).

## Tests

`backend/tests/test_<module>.py`, pytest, using `causality.sim` generators where the truth is
known. Run from `backend/`: `python -m pytest tests/test_<module>.py -q`. Keep each test file
under ~60 s.
