"""FastAPI server: the API plus the static page.

Run from backend/:  python -m uvicorn app:app --port 8000
"""
from __future__ import annotations

import json
import os
import threading
import traceback
import uuid
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles

from causality import data, pipeline, sim
from causality.data import InputError
from causality.schema import to_jsonable

warnings.filterwarnings("ignore")

FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
MAX_BYTES = 50 * 1024 * 1024

app = FastAPI(title="CausalityAI", version="1.0")


def _error(msg: str, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": msg}, status_code=code)


def _json_field(raw: Optional[str], name: str) -> dict:
    if not raw:
        return {}
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        raise InputError(f"The '{name}' field is not valid JSON.")
    if not isinstance(v, dict):
        raise InputError(f"The '{name}' field must be a JSON object.")
    return v


async def _read(file: UploadFile):
    content = await file.read()
    if not content:
        raise InputError("The uploaded file is empty.")
    if len(content) > MAX_BYTES:
        raise InputError("The file is larger than 50 MB.")
    return data.load_table(content, file.filename or "data.csv")


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/columns")
async def columns(file: UploadFile = File(...)):
    try:
        df = await _read(file)
        prep = data.prepare(df)
        return {"n_rows": int(len(df)),
                "columns": [{"name": c.name, "kind": c.kind, "n_missing": c.n_missing, "n_unique": c.n_unique}
                            for c in prep.columns]}
    except InputError as e:
        return _error(str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return _error(f"The file could not be read ({type(e).__name__}).", 500)


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...), claim: str = Form(""), design: str = Form(""),
                  roles: str = Form(""), confirm: str = Form(""), online_literature: str = Form("false"),
                  context: str = Form("")):
    try:
        df = await _read(file)
        report = pipeline.run(df, claim_text=claim, design_decl=design or None,
                              roles=_json_field(roles, "roles"), confirm=_json_field(confirm, "confirm"),
                              online_literature=online_literature.lower() == "true", context=context)
        return {"report": to_jsonable(report)}
    except InputError as e:
        return _error(str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return _error(f"Something went wrong during the analysis ({type(e).__name__}: {e}). "
                      "Nothing was concluded from the data.", 500)


@app.get("/api/sample/{name}")
def sample(name: str):
    """Simulated datasets with a known truth, to try the tool."""
    if name not in sim.ALL:
        return _error(f"Unknown sample '{name}'. Available: {', '.join(sim.ALL)}.", 404)
    df, _ = sim.ALL[name]()
    return PlainTextResponse(df.to_csv(index=False), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})


# ---------------------------------------------------------------------------------------------
# Audit with Opus (see ORCHESTRATOR.md, part D). The model reads the claim, picks the design and
# judges; code computes the numbers and applies the floor rules (loop.py, floor.py).
# Sessions live in memory only: the table, the claim, the context and the last resume state.
# ---------------------------------------------------------------------------------------------

AUDIT_MAX_SESSIONS = 50
_audit_sessions: "OrderedDict[str, dict]" = OrderedDict()
_audit_lock = threading.Lock()


def _audit_client():
    """A client injected by tests (app.state.audit_client); else None when ANTHROPIC_API_KEY is set, so
    the loop makes the SDK client; else headless Claude Code (runs on a Claude subscription, no key)."""
    injected = getattr(app.state, "audit_client", None)
    if injected is not None:
        return injected
    from causality.orchestrator import gemini
    if gemini.available():                     # GEMINI_API_KEY in the environment or in .env
        return gemini.GeminiClient()
    if os.environ.get("ANTHROPIC_API_KEY"):
        return None
    from causality.orchestrator import headless
    return headless.HeadlessClient()


def _audit_model() -> str:
    """The model name shown in the report: Gemini's when it is the one answering."""
    from causality.orchestrator import contract, gemini
    if getattr(app.state, "audit_client", None) is None and gemini.available():
        return gemini.model_name()
    return contract.MODEL


def _audit_store(session_id: str, entry: dict) -> None:
    with _audit_lock:
        _audit_sessions[session_id] = entry
        _audit_sessions.move_to_end(session_id)
        while len(_audit_sessions) > AUDIT_MAX_SESSIONS:
            _audit_sessions.popitem(last=False)


def _audit_get(session_id: str) -> Optional[dict]:
    with _audit_lock:
        entry = _audit_sessions.get(session_id)
        if entry is not None:
            _audit_sessions.move_to_end(session_id)
        return entry


def _audit_public(report: dict, session_id: str) -> dict:
    """The report as sent to the browser: session_id set, resume state kept on the server."""
    out = {k: v for k, v in (report or {}).items() if k != "state"}
    out["session_id"] = session_id
    return to_jsonable(out)


def _audit_failed(e: Exception) -> JSONResponse:
    traceback.print_exc()
    return _error(f"Something went wrong during the audit ({type(e).__name__}: {e}). "
                  "Nothing was concluded from the data.", 500)


async def _read_off_loop(file: UploadFile):
    """Like _read, but the table is parsed in a worker thread so a large upload does not block the event loop."""
    content = await file.read()
    if not content:
        raise InputError("The uploaded file is empty.")
    if len(content) > MAX_BYTES:
        raise InputError("The file is larger than 50 MB.")
    return await run_in_threadpool(data.load_table, content, file.filename or "data.csv")


@app.post("/api/audit")
async def audit(file: UploadFile = File(...), claim: str = Form(""), context: str = Form("")):
    try:
        df = await _read_off_loop(file)
    except InputError as e:
        return _error(str(e))
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return _error(f"The file could not be read ({type(e).__name__}).", 500)
    if not (claim or "").strip():
        return _error("Write the claim you want to audit, for example 'X increases Y'.")
    try:
        from causality.orchestrator import loop
        report = await run_in_threadpool(loop.run_audit, df, claim, context or "", client=_audit_client(),
                                         model=_audit_model())
    except Exception as e:  # noqa: BLE001
        return _audit_failed(e)
    session_id = uuid.uuid4().hex
    report = report or {}
    _audit_store(session_id, {"df": df, "claim": claim, "context": context or "",
                              "status": report.get("status"), "state": report.get("state")})
    return _audit_public(report, session_id)


async def _audit_continue(session_id: str, entry: dict, want: str, refusal: str, call, *args):
    """Run loop.resume or loop.followup on a stored session. The session is claimed before the model
    runs, so two requests at once cannot both continue the same state. A run that fails without new
    state (rate limit, timeout, lost connection, ...) leaves the session as it was, so the same
    request can be sent again."""
    with _audit_lock:
        if entry.get("state") is None or entry.get("status", "needs_input") != want:
            return _error(refusal, 409)
        if entry.get("busy"):
            return _error("This audit is already working on a request. Wait for it to finish.", 409)
        entry["busy"] = True
        before = (entry.get("status", "needs_input"), entry["state"])
    try:
        try:
            from causality.orchestrator import loop
            report = await run_in_threadpool(getattr(loop, call), entry["df"], before[1], *args,
                                             client=_audit_client())
        except Exception as e:  # noqa: BLE001
            return _audit_failed(e)
        with _audit_lock:
            report = report or {}
            if report.get("status") == "error" and report.get("state") is None:
                entry["status"], entry["state"] = before
            else:
                entry["status"], entry["state"] = report.get("status"), report.get("state")
        return _audit_public(report, session_id)
    finally:
        with _audit_lock:
            entry["busy"] = False


_NO_SESSION = "This audit session was not found. It may have expired; upload the table again."


@app.post("/api/audit/{session_id}/answer")
async def audit_answer(session_id: str, payload: dict = Body(...)):
    entry = _audit_get(session_id)
    if entry is None:
        return _error(_NO_SESSION, 404)
    answers = payload.get("answers") if isinstance(payload, dict) else None
    if not isinstance(answers, dict):
        return _error('Send the answers as {"answers": {"<question id>": "<your answer>"}}.')
    answers = {str(k): "" if v is None else str(v) for k, v in answers.items()}
    return await _audit_continue(
        session_id, entry, "needs_input",
        "This audit is not waiting for answers. Start a new audit to change anything.", "resume", answers)


@app.post("/api/audit/{session_id}/followup")
async def audit_followup(session_id: str, payload: dict = Body(...)):
    entry = _audit_get(session_id)
    if entry is None:
        return _error(_NO_SESSION, 404)
    question = payload.get("question") if isinstance(payload, dict) else None
    if not isinstance(question, str) or not question.strip():
        return _error('Send the follow-up as {"question": "<your question>"}.')
    return await _audit_continue(
        session_id, entry, "done",
        "This audit has no finished verdict to ask about yet. Answer its questions first, or start a new audit.",
        "followup", question.strip())


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND)), name="static")
