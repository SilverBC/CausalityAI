"""FastAPI server: the Causal Audit API plus the web page.

Run from the project root:
    python -m uvicorn app:app --app-dir backend --port 8000
uvicorn binds to 127.0.0.1 by default. Binding to 0.0.0.0 lets anyone on your network use the app, and every
literature search or summary they trigger spends your Claude quota. See the README before doing that.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from causality import data, sim
from causality.data import InputError
from services import graph, investigate, jobs, llm, store
from services.config import FRONTEND_DIST, MAX_UPLOAD_BYTES, SAMPLES_DIR
from services.ownership import owned_analysis, owner_of

warnings.filterwarnings("ignore")

app = FastAPI(title="Causal Audit", version="2.0")
store.init()
_interrupted = store.interrupt_running()
if _interrupted:
    print(f"[startup] {_interrupted} analysis run(s) were interrupted by the restart.", flush=True)


# --------------------------------------------------------------------------------------------- health
@app.get("/api/health")
def health(probe: bool = False):
    """Language-model availability and its last error. probe=true makes one tiny call to check it really works."""
    out = {"ok": True, "llm": llm.available(), "llm_backend": llm.backend(), "llm_label": llm.describe(),
           "llm_last_error": llm.last_error()}
    if probe and out["llm"]:
        reply = llm.run("Reply with the single word: ok", timeout=60, kind="probe")
        out["probe_ok"] = bool(reply)
        out["llm_last_error"] = llm.last_error()
    return out


# --------------------------------------------------------------------------------------------- samples
REAL_SAMPLES = {
    "lalonde_experimental": {
        "file": "real/lalonde_experimental.csv", "title": "Real: LaLonde NSW experiment",
        "description": "1970s randomized job training (Dehejia–Wahba subsample). Benchmark effect about +$1,800.",
        "claim": "Job training increases 1978 earnings by $1,800",
        "roles": {"treatment": "treat", "outcome": "earnings_1978", "randomized": True}, "design": "experiment"},
    "lalonde_observational": {
        "file": "real/lalonde_observational.csv", "title": "Real: LaLonde vs PSID",
        "description": "The same trainees against a survey comparison group: famously misleading.",
        "claim": "Job training increases 1978 earnings by $1,800",
        "roles": {"treatment": "treat", "outcome": "earnings_1978"}, "design": "observational"},
    "card_krueger": {
        "file": "real/card_krueger_minwage.csv", "title": "Real: Card & Krueger minimum wage",
        "description": "New Jersey vs Pennsylvania fast-food stores, 1992. The paper's DiD is +2.76 FTE.",
        "claim": "Raising the minimum wage reduced fast-food employment",
        "roles": {"treatment": "nj_after_raise", "outcome": "fte_employment", "unit": "store_id", "time": "post"},
        "design": "did"},
}
SIM_SAMPLES = {
    "rct": ("Simulated: randomized trial", "Treatment randomized; true effect +1.5."),
    "observational": ("Simulated: observational", "Selection on measured covariates; true effect +2."),
    "observational_hidden": ("Simulated: hidden confounder", "A confounder that is not in the data; true effect 0."),
    "did": ("Simulated: policy panel (DiD)", "States adopt a policy; true effect on the treated +2."),
    "iv": ("Simulated: lottery (IV)", "A lottery shifts enrolment; true local effect +1."),
    "rd": ("Simulated: scholarship cutoff (RD)", "Scholarship at exam score 50; true effect at the cutoff +3."),
    "dose": ("Simulated: study hours (dose)", "Continuous treatment; true slope +0.5 per hour."),
    "cluster_saturation": ("Simulated: village spillovers", "Treated share randomized per village; spillovers."),
    "chain": ("Simulated: causal chain (explore)", "A → B → C ← D, E independent. No claim: exploration."),
}


def _sim_info(name: str) -> dict:
    """Roles, design and a claim built from the simulation's known truth."""
    _, truth = sim.ALL[name]()
    roles = {k: truth[k] for k in ("treatment", "outcome", "unit", "time", "instrument", "running", "cutoff",
                                   "cluster", "saturation") if truth.get(k) is not None}
    design = truth.get("design")
    if design == "experiment":
        roles["randomized"] = True
    claim = ""
    if truth.get("treatment") and truth.get("outcome"):
        eff = truth.get("effect")
        claim = (f"{truth['treatment']} has no effect on {truth['outcome']}" if eff == 0
                 else f"{truth['treatment']} increases {truth['outcome']} by {eff:g}")
    title, desc = SIM_SAMPLES[name]
    return {"id": name, "kind": "simulated", "filename": f"{name}.csv", "title": title, "description": desc,
            "claim": claim, "roles": roles,
            "design": design if design in ("experiment", "did", "iv", "rd", "observational") else None}


@app.get("/api/samples")
def samples():
    """Whitelisted demo datasets: real ones from samples/real, and simulations with a known truth."""
    real = [{"id": k, "kind": "real", "filename": Path(v["file"]).name,
             **{x: v[x] for x in ("title", "description", "claim", "roles", "design")}}
            for k, v in REAL_SAMPLES.items() if (SAMPLES_DIR / v["file"]).exists()]
    return real + [_sim_info(n) for n in SIM_SAMPLES if n in sim.ALL]


@app.get("/api/samples/{name}")
def sample_file(name: str):
    if name in REAL_SAMPLES:  # whitelist only: never a path from the request
        p = SAMPLES_DIR / REAL_SAMPLES[name]["file"]
        if p.exists():
            return FileResponse(p, media_type="text/csv", filename=p.name)
    if name in sim.ALL:
        df, _ = sim.ALL[name]()
        return PlainTextResponse(df.to_csv(index=False), media_type="text/csv",
                                 headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
    raise HTTPException(404, "Unknown sample.")


# --------------------------------------------------------------------------------------------- uploads
async def _read_upload(file: Optional[UploadFile]) -> tuple[bytes, str]:
    if file is None:
        return b"", ""
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "The file is larger than 50 MB.")
    return content, file.filename or "data.csv"


def _load(content: bytes, filename: str):
    if not content:
        raise HTTPException(400, "The uploaded file is empty.")
    try:
        return data.load_table(content, filename)
    except InputError as e:
        raise HTTPException(400, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"The file could not be read ({type(e).__name__}).")


def _json_field(raw: Optional[str], name: str) -> dict:
    if not raw:
        return {}
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(400, f"The '{name}' field is not valid JSON.")
    if not isinstance(v, dict):
        raise HTTPException(400, f"The '{name}' field must be a JSON object.")
    return v


@app.post("/api/columns")
async def columns(file: UploadFile = File(...)):
    df = _load(*(await _read_upload(file)))
    prep = data.prepare(df)
    return {"n_rows": int(len(df)),
            "columns": [{"name": c.name, "kind": c.kind, "n_missing": c.n_missing, "n_unique": c.n_unique}
                        for c in prep.columns]}


# --------------------------------------------------------------------------------------------- analyses
@app.post("/api/analyze")
async def analyze(request: Request, file: Optional[UploadFile] = File(None), claim: str = Form(""),
                  design: str = Form(""), roles: str = Form(""), confirm: str = Form(""),
                  online_literature: str = Form("false"), notify: str = Form("0")):
    """Start an analysis in the background. Mode: audit (claim + data), explore (data), claim-only (claim)."""
    content, filename = await _read_upload(file)
    claim = (claim or "").strip()
    if not content and not claim:
        raise HTTPException(400, "Provide a dataset, a causal claim, or both.")
    if content:
        _load(content, filename)  # fail fast on unreadable files, before creating a row
    mode = ("audit" if claim else "explore") if content else "claim-only"
    if mode == "claim-only" and not llm.available():
        raise HTTPException(503, "A claim without data needs Claude for the literature search, and no language model "
                                 "is configured. Upload a dataset to analyse the claim from data.")
    params = {"design": design or None, "roles": _json_field(roles, "roles"), "confirm": _json_field(confirm, "confirm"),
              "online_literature": online_literature.lower() == "true", "notify": notify in ("1", "true", "on")}
    a = store.create_analysis(owner=owner_of(request), mode=mode, title=jobs.title_for(filename, claim),
                              filename=filename, claim=claim, params=params, link_base=str(request.base_url))
    if content:
        store.save_upload(a["id"], content)
    jobs.start(a["id"])
    return {"job_id": a["id"], "mode": mode}


def _public(a: dict) -> dict:
    out = {k: a[k] for k in ("id", "mode", "title", "filename", "claim", "status", "error", "steps", "params",
                             "created", "updated")}
    out["elapsed"] = round((time.time() if a["status"] == "running" else a["updated"]) - a["created"], 1)
    if a["status"] in ("done", "needs_input"):
        out["result"] = a["result"]
    return out


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str, request: Request):
    """Status, steps and elapsed time; the result once it is done (or once it asks a question)."""
    return _public(owned_analysis(request, job_id))


@app.get("/api/datasets")
def list_datasets(request: Request):
    """The caller's analyses, newest activity first, with what the sidebar shows."""
    out = []
    for a in store.list_analyses(owner_of(request)):
        res = a["result"] or {}
        rep = res.get("report") or {}
        out.append({"id": a["id"], "title": a["title"], "mode": a["mode"], "status": a["status"], "error": a["error"],
                    "filename": a["filename"], "claim": a["claim"], "created": a["created"], "updated": a["updated"],
                    "grade": (rep.get("design") or {}).get("grade"), "verdict": (rep.get("verdict") or {}).get("label"),
                    "has_graph": bool((res.get("influence") or {}).get("nodes")),
                    "investigations": [{"node": i["node"], "label": i["label"], "status": i["status"],
                                        "found": len((i.get("result") or {}).get("factors", []))}
                                       for i in store.list_investigations(a["id"])]})
    return out


class Answers(BaseModel):
    confirm: dict = {}
    design: Optional[str] = None


@app.post("/api/jobs/{job_id}/answer")
def answer(job_id: str, body: Answers, request: Request):
    """Answers to the questions the analysis asked. Re-runs the same analysis in place."""
    a = owned_analysis(request, job_id)
    if a["status"] == "running":
        raise HTTPException(409, "The analysis is still running.")
    if a["mode"] == "claim-only":
        raise HTTPException(409, "A claim-only analysis has nothing to answer.")
    params = dict(a["params"])
    params["confirm"] = {**(params.get("confirm") or {}), **{k: v for k, v in body.confirm.items() if v}}
    if body.design:
        params["design"] = body.design
    store.update_analysis(job_id, params=params, status="running", error=None, result=None, steps=[])
    jobs.start(job_id)
    return {"job_id": job_id}


_depth_cache: dict[tuple[str, int, float], dict] = {}


@app.get("/api/jobs/{job_id}/influence")
def job_influence(job_id: str, depth: int, request: Request):
    """The influence graph at a data-search depth: pruned below the default, searched upstream above it."""
    a = owned_analysis(request, job_id)
    if a["status"] != "done":
        raise HTTPException(409, "The analysis is not finished.")
    res = a["result"] or {}
    if not res.get("base_graph"):
        raise HTTPException(404, "No graph for this analysis.")
    key = (job_id, int(depth), a["updated"])
    if key not in _depth_cache:
        df = store.load_frame(job_id)
        if df is None:
            raise HTTPException(404, "The prepared data for this analysis is no longer on the server.")
        meta = res["frame"]
        fr = graph.Frame(df, meta["encoding"], meta["kinds"], meta["skip"])
        if len(_depth_cache) > 200:
            _depth_cache.clear()
        _depth_cache[key] = graph.influence_at_depth(fr, res["base_graph"], depth)
    return _depth_cache[key]


# --------------------------------------------------------------------------------------------- investigations
class InvestigateRequest(BaseModel):
    node: str
    label: Optional[str] = None
    in_data: bool = True


class BatchRequest(BaseModel):
    nodes: list[InvestigateRequest]


def _investigable(request: Request, job_id: str) -> dict:
    a = owned_analysis(request, job_id)
    if a["status"] != "done" or not ((a["result"] or {}).get("influence") or {}).get("nodes"):
        raise HTTPException(409, "This analysis has no graph to investigate yet.")
    if not llm.available():
        raise HTTPException(503, "Investigations search the literature with Claude, and no language model is "
                                 "configured (install Claude Code, set ANTHROPIC_API_KEY, or set CAUSAL_LLM).")
    return a


def _in_data(a: dict, req: InvestigateRequest) -> bool:
    cols = {c["name"] for c in ((a["result"].get("report") or {}).get("data") or {}).get("columns") or []}
    return bool(req.in_data and req.node in cols)


@app.get("/api/jobs/{job_id}/investigations")
def list_investigations(job_id: str, request: Request):
    """Every investigation of the analysis, with elapsed seconds and queue positions."""
    owned_analysis(request, job_id)
    return investigate.public(job_id)


@app.post("/api/jobs/{job_id}/investigate")
def start_investigation(job_id: str, req: InvestigateRequest, request: Request):
    """Search the literature for what causes one variable (runs in the background; queued if 3 already run)."""
    a = _investigable(request, job_id)
    try:
        return investigate.start(a, req.node, req.label or req.node, _in_data(a, req))
    except investigate.QueueFull as e:
        raise HTTPException(429, str(e))


@app.post("/api/jobs/{job_id}/investigate-batch")
def start_batch(job_id: str, req: BatchRequest, request: Request):
    """Queue investigations of several variables (at most 100 per request). Stops at the queue limit."""
    a = _investigable(request, job_id)
    queued, skipped = [], []
    for n in req.nodes[:100]:
        try:
            queued.append(investigate.start(a, n.node, n.label or n.node, _in_data(a, n)))
        except investigate.QueueFull:
            skipped.append(n.node)
    n = investigate.counts(job_id)
    # `queued` lists what this request started or queued; the counts describe the whole queue afterwards
    return {"queued": queued, "skipped": skipped, "running": n["running"], "queued_count": n["queued"]}


@app.post("/api/jobs/{job_id}/investigations/cancel-queued")
def cancel_queued(job_id: str, request: Request):
    owned_analysis(request, job_id)
    return {"cancelled": investigate.cancel_queued(job_id)}


@app.delete("/api/jobs/{job_id}/investigations/{node:path}")
def delete_investigation(job_id: str, node: str, request: Request):
    """Delete one variable's investigation, or cancel it if it is still queued."""
    owned_analysis(request, job_id)
    if not investigate.remove(job_id, node):
        raise HTTPException(404, "There is no investigation of that variable.")
    return {"ok": True}


# --------------------------------------------------------------------------------------------- page
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    @app.get("/", response_class=HTMLResponse)
    def frontend_not_built():
        return ("<!doctype html><meta charset='utf-8'><title>Causal Audit</title>"
                "<body style='font:16px/1.5 system-ui,sans-serif;max-width:640px;margin:60px auto;padding:0 16px'>"
                "<h2>The Causal Audit API is running, but the web app is not built yet.</h2>"
                "<p>Build it once with <code>cd frontend &amp;&amp; npm install &amp;&amp; npm run build</code>, then reload "
                "this page. While developing, run <code>npm run dev</code> in <code>frontend/</code> and open "
                "<a href='http://localhost:5173'>http://localhost:5173</a>.</p>"
                "<p>The API itself works: try <a href='/api/health'>/api/health</a>.</p></body>")
