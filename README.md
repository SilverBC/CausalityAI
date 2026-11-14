# CausalityAI

Upload a table and, optionally, a sentence like *"X increases Y by 15% among Z"*.
With a sentence, CausalityAI **audits** it. Without one, it **explores** what may cause what.

The one rule: **never claim more than the data can show.** Every assumption gets one of four
statuses — *tested: OK*, *tested: concern*, *failed*, or *assumed (not tested)*. Things data can
never show — no hidden confounders, the exclusion restriction, continuity at a cutoff, no
interference between units, outcomes missing for reasons tied to the outcome — are always
*assumed*, and *assumed* is never displayed as a pass.

## Run

Python 3.12 is recommended (some scientific packages have no wheels for newer versions yet).

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
(cd frontend && npm install && npm run build)       # Node 18+; builds the web app into frontend/dist
python -m uvicorn app:app --app-dir backend --port 8000
```

Open http://localhost:8000. While working on the web app, run `npm run dev` in `frontend/` as well and open
http://localhost:5173: it reloads on every change and forwards `/api` to the server on port 8000. To try it
without your own data, click a sample on the page (three real datasets with known benchmarks, plus simulations
with a known truth); shift-click loads the data without its claim.

Tests (from `backend/`). They use a throwaway data folder and the fake language model, so they touch neither
your data nor your Claude quota:

```bash
python -m pytest tests -q
```

Browser tests (from the project root, after `npm run build`). They drive your installed Chrome through Playwright and
save screenshots to `frontend/e2e/screens/`:

```bash
pip install -r requirements-dev.txt
python -m pytest frontend/e2e -q
```

### Configuration

Settings come from environment variables or a `.env` file in the project root (gitignored; never commit it).

| Variable | Default | What it does |
|---|---|---|
| `CAUSAL_LLM` | `cli` if Claude Code is found, else `api` if `ANTHROPIC_API_KEY` is set, else `off` | Which language model backend: `cli`, `api`, `fake` (tests) or `off` |
| `CLAUDE_BIN` | found on `PATH`, `~/.local/bin` or the VS Code / Cursor extension | Path to the Claude Code binary for `cli` |
| `CAUSAL_CLAUDE_MODEL` | Claude Code's default | Model for `cli` |
| `ANTHROPIC_API_KEY` | none | Key for `api` (or sign in with `ant auth login`) |
| `CAUSAL_API_MODEL` | `claude-opus-5-5` | Model for `api` |
| `CAUSAL_DATA_DIR` | `data/` | Where the database (`causal.db`) and per-analysis files (`blobs/`) live |
| `RESEND_API_KEY`, `MAIL_FROM` | none | "Analysis ready" email through Resend. Without a verified domain Resend only delivers to the Resend account's own address |
| `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` | none | Email fallback over SMTP (e.g. Gmail with an app password) |

The analysis itself never uses a language model. Claude writes the executive summary (with a plain template when
it is unavailable), runs the literature investigations and the Investigator chat, and handles claim-only mode (a
claim with no data). `GET /api/health` shows
which backend is active and its last error; `?probe=true` makes one tiny call to check that it works.

### Local only by default

uvicorn listens on `127.0.0.1`, so only this computer can reach the app. `--host 0.0.0.0` makes it reachable by
anyone on your network. With the `cli` backend every summary or literature search they trigger then runs on your
own Claude subscription and uses up your limits. There are no rate limits yet. Any shared or public deployment
needs the `api` backend, accounts and rate limits.

### Investigations: beyond the data

A dataset can never contain the confounders nobody measured, but published research can name them. On the
Investigation page (`#/investigate/<id>`) you pick a variable and Claude searches the literature (at most 4 web
searches) for what causes it. Each factor needs a real source; it joins the graph as a **hypothetical** variable
(dashed, no edge strengths), or as the matching column when one measures the same concept. Each factor gets a
0–100 **Reliability Score** from its sources' OpenAlex citation counts and venue type, plus a small bonus when several
sources agree. The score says how established the research is, not whether the factor confounds your data.

Searches run in a per-analysis queue: 3 at a time, up to 40 waiting. They survive page reloads; a server restart marks
running and queued ones as interrupted, so they can be retried. Renaming a variable is display-only: the name stays in
your browser and is never sent to the server or to Claude. `CAUSAL_OPENALEX=off` skips the OpenAlex lookups (the tests
do this), and `CAUSAL_FAKE_DELAY` sets how long a fake search takes.

### Accounts and guests

You can use the app as a guest: your browser keeps a random id and your analyses belong to it, so they last as long
as this browser keeps its data. Creating an account (email and password, stored as an scrypt hash) moves this
browser's work to the account, so it survives across devices. Everyone sees only their own analyses; anything else
answers 404. The first account created on a server also adopts analyses made before accounts existed.

Signed-in users can ask for an email when an analysis finishes. If the email can't be sent, the analysis says why
instead of failing silently. Deleting an analysis removes its investigations and files too; every delete asks first.

### The Investigator and the phone app

The Investigator is a chat assistant (the alligator in a deerstalker, bottom right on a finished analysis and on the
Investigation page) that answers questions about one analysis using its real numbers. Each reply is one Claude call
with a compact digest of the analysis and the last 12 messages; it can show cards (the estimate, a check, the graph,
an investigation) and start a literature investigation when you ask it to. When Claude can't answer, the chat says why
and keeps your message so you can send it again.

The phone app (`#/m`, where phones land by default) is built around those chats: your datasets with their chats, an
upload screen, the chat itself, and the causal graph full screen with a bottom sheet for investigating variables.

### How the web app runs an analysis

`POST /api/analyze` stores the upload and returns `{job_id, mode}` at once. The analysis runs in a background thread
and `GET /api/jobs/<id>` reports its live steps and, at the end, the result. If the analysis needs an answer
first (which column the claim means, or which design), its status is `needs_input` and
`POST /api/jobs/<id>/answer` re-runs it with the answers. `GET /api/jobs/<id>/influence?depth=N` returns the
weighted causal graph at a search depth (see `services/graph.py`). Analyses are SQLite rows of plain JSON plus
files under `data/blobs/`. Nothing is pickled, so they survive restarts and code changes. Analyses still
running when the server stops are marked as interrupted when it starts again.

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
| Web app services: storage, background jobs, language-model adapter, influence graph and depth search, literature investigations and Reliability Score, the Investigator chat, accounts and ownership, email | `backend/services/` |
| Web app (React + Vite): analysis page, result tabs, influence graph, Investigation page, the Investigator chat, phone app, accounts, light/dark theme. The report tabs use the original hand-written SVG charts and renderer; JSON and standalone-HTML export | `frontend/src/` |

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
