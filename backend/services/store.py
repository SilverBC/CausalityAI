"""Persistence: one SQLite file (standard library, no ORM) plus per-analysis blobs.

    data/causal.db              analyses (later prompts add users, sessions, investigations, chats, messages)
    data/blobs/<id>.upload      the uploaded file, byte for byte (a re-run reads it again)
    data/blobs/<id>.parquet     the prepared numeric frame (the depth search reloads it on demand)

Rows hold plain JSON, never pickled Python objects: renaming or moving a class can never lose stored work
(v1 pickled whole jobs and lost them silently whenever a class changed).
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .config import data_dir

_lock = threading.RLock()

# Tables are created if missing; MIGRATIONS only ever add columns. Later prompts append to both.
TABLES = {
    "analyses": """
        CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY, owner TEXT, mode TEXT NOT NULL, title TEXT NOT NULL, filename TEXT, claim TEXT,
            status TEXT NOT NULL, error TEXT, steps TEXT NOT NULL DEFAULT '[]', result TEXT,
            params TEXT NOT NULL DEFAULT '{}', notify_email TEXT, link_base TEXT,
            created REAL NOT NULL, updated REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS analyses_by_owner ON analyses(owner, updated);
    """,
}
MIGRATIONS: list[tuple[str, str, str]] = []   # (table, column, declaration), e.g. ("analyses", "mail_status", "TEXT")
JSON_COLUMNS = {"steps", "result", "params"}


def db_path() -> Path:
    return data_dir() / "causal.db"


def blob_dir() -> Path:
    d = data_dir() / "blobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(db_path(), timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    return con


def migrate(con: sqlite3.Connection, tables: dict[str, str], migrations: list[tuple[str, str, str]]) -> None:
    """CREATE TABLE IF NOT EXISTS, then additive ALTERs for columns that are missing."""
    for ddl in tables.values():
        con.executescript(ddl)
    for table, col, decl in migrations:
        have = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
        if col not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def init() -> None:
    with _lock, connect() as con:
        migrate(con, TABLES, MIGRATIONS)


def new_id(n: int = 12) -> str:
    return uuid.uuid4().hex[:n]


def _loads(s: Optional[str], default: Any) -> Any:
    if not s:
        return default
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return default


# ------------------------------------------------------------------------------------------- analyses
def _row(r: sqlite3.Row) -> dict:
    out = dict(r)
    out["steps"] = _loads(out.get("steps"), [])
    out["result"] = _loads(out.get("result"), None)
    out["params"] = _loads(out.get("params"), {})
    out["filename"] = out.get("filename") or ""
    out["claim"] = out.get("claim") or ""
    return out


def create_analysis(*, owner: str, mode: str, title: str, filename: str, claim: str, params: dict,
                    notify_email: Optional[str] = None, link_base: Optional[str] = None) -> dict:
    now = time.time()
    row = {"id": new_id(), "owner": owner, "mode": mode, "title": title, "filename": filename, "claim": claim,
           "status": "running", "steps": "[]", "params": json.dumps(params), "notify_email": notify_email,
           "link_base": link_base, "created": now, "updated": now}
    with _lock, connect() as con:
        con.execute(f"INSERT INTO analyses ({', '.join(row)}) VALUES ({', '.join(':' + k for k in row)})", row)
    return get_analysis(row["id"])


def get_analysis(analysis_id: str) -> Optional[dict]:
    with connect() as con:
        r = con.execute("SELECT * FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    return _row(r) if r else None


def list_analyses(owner: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM analyses WHERE owner=? ORDER BY updated DESC", (owner,)).fetchall()
    return [_row(r) for r in rows]


def update_analysis(analysis_id: str, **fields) -> bool:
    """Update columns. Returns False if the analysis no longer exists (e.g. deleted while running)."""
    fields["updated"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = [json.dumps(v) if k in JSON_COLUMNS and v is not None else v for k, v in fields.items()]
    with _lock, connect() as con:
        return con.execute(f"UPDATE analyses SET {cols} WHERE id=?", (*vals, analysis_id)).rowcount > 0


def delete_analysis(analysis_id: str) -> None:
    with _lock, connect() as con:
        con.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
    for suffix in (".upload", ".parquet"):
        (blob_dir() / f"{analysis_id}{suffix}").unlink(missing_ok=True)
    with _lock:
        _frames.pop(analysis_id, None)


def interrupt_running() -> int:
    """On server start: analyses that were running when the server stopped can never finish."""
    with _lock, connect() as con:
        return con.execute("UPDATE analyses SET status='error', error='Interrupted by a server restart. Run it again.', "
                           "updated=? WHERE status='running'", (time.time(),)).rowcount


# ------------------------------------------------------------------------------------------- blobs
def save_upload(analysis_id: str, content: bytes) -> None:
    (blob_dir() / f"{analysis_id}.upload").write_bytes(content)


def load_upload(analysis_id: str) -> Optional[bytes]:
    p = blob_dir() / f"{analysis_id}.upload"
    return p.read_bytes() if p.exists() else None


_frames: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
FRAME_CACHE_SIZE = 8


def save_frame(analysis_id: str, df: pd.DataFrame) -> None:
    df.to_parquet(blob_dir() / f"{analysis_id}.parquet", index=False)
    with _lock:
        _frames.pop(analysis_id, None)


def load_frame(analysis_id: str) -> Optional[pd.DataFrame]:
    """The prepared numeric frame, from a small LRU cache or from disk."""
    with _lock:
        if analysis_id in _frames:
            _frames.move_to_end(analysis_id)
            return _frames[analysis_id]
    p = blob_dir() / f"{analysis_id}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    with _lock:
        _frames[analysis_id] = df
        while len(_frames) > FRAME_CACHE_SIZE:
            _frames.popitem(last=False)
    return df
