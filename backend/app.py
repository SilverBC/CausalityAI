"""FastAPI server: the API plus the static page.

Run from backend/:  python -m uvicorn app:app --port 8000
"""
from __future__ import annotations

import json
import traceback
import warnings
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
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
                  roles: str = Form(""), confirm: str = Form(""), online_literature: str = Form("false")):
    try:
        df = await _read(file)
        report = pipeline.run(df, claim_text=claim, design_decl=design or None,
                              roles=_json_field(roles, "roles"), confirm=_json_field(confirm, "confirm"),
                              online_literature=online_literature.lower() == "true")
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


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND)), name="static")
