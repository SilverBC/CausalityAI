"""Persistence: one SQLite file (standard library, no ORM) plus per-analysis blobs.

    data/causal.db              analyses, investigations, users, sessions, chats, messages
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
    "investigations": """
        CREATE TABLE IF NOT EXISTS investigations (
            id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, node TEXT NOT NULL, label TEXT NOT NULL,
            in_data INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL, result TEXT, error TEXT,
            queued_at REAL NOT NULL, started_at REAL, finished_at REAL);
        CREATE INDEX IF NOT EXISTS investigations_by_analysis ON investigations(analysis_id, queued_at);
    """,
    "users": """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, pw_hash TEXT NOT NULL, created REAL NOT NULL);
    """,
    "sessions": """
        CREATE TABLE IF NOT EXISTS sessions (token TEXT PRIMARY KEY, user_id TEXT NOT NULL, created REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS sessions_by_user ON sessions(user_id);
    """,
    "chats": """
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, analysis_id TEXT NOT NULL, title TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS chats_by_analysis ON chats(analysis_id, updated);
    """,
    "messages": """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY, chat_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL,
            cards TEXT NOT NULL DEFAULT '[]', created REAL NOT NULL);
        CREATE INDEX IF NOT EXISTS messages_by_chat ON messages(chat_id, created);
    """,
    "llm_usage": """
        CREATE TABLE IF NOT EXISTS llm_usage (day TEXT PRIMARY KEY, calls INTEGER NOT NULL);
    """,
}
MIGRATIONS: list[tuple[str, str, str]] = [   # (table, column, declaration)
    ("analyses", "mail_status", "TEXT"),        # stage 04: outcome of the "analysis ready" email
]
JSON_COLUMNS = {"steps", "result", "params", "mail_status"}


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
    out["mail_status"] = _loads(out.get("mail_status"), None)
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


def move_owner(old: str, new: str) -> int:
    """Signing up or in moves a guest's analyses (and so their investigations) to the account."""
    with _lock, connect() as con:
        return con.execute("UPDATE analyses SET owner=? WHERE owner=?", (new, old)).rowcount


def adopt_unowned(new: str) -> int:
    """Migration only: analyses made before accounts existed (no owner, or the stage-01 placeholder 'anon')."""
    with _lock, connect() as con:
        return con.execute("UPDATE analyses SET owner=? WHERE owner IS NULL OR owner IN ('', 'anon')", (new,)).rowcount


def delete_analysis(analysis_id: str) -> None:
    with _lock, connect() as con:
        con.execute("DELETE FROM messages WHERE chat_id IN (SELECT id FROM chats WHERE analysis_id=?)", (analysis_id,))
        con.execute("DELETE FROM chats WHERE analysis_id=?", (analysis_id,))
        con.execute("DELETE FROM investigations WHERE analysis_id=?", (analysis_id,))
        con.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
    for suffix in (".upload", ".parquet"):
        (blob_dir() / f"{analysis_id}{suffix}").unlink(missing_ok=True)
    with _lock:
        _frames.pop(analysis_id, None)


def interrupt_running() -> int:
    """On server start: work that was running or queued when the server stopped can never finish.
    Returns the number of interrupted analyses; interrupted investigations can be retried."""
    now = time.time()
    with _lock, connect() as con:
        con.execute("UPDATE investigations SET status='error', error='Interrupted by a server restart. Retry it.', "
                    "finished_at=? WHERE status IN ('running', 'queued')", (now,))
        return con.execute("UPDATE analyses SET status='error', error='Interrupted by a server restart. Run it again.', "
                           "updated=? WHERE status='running'", (now,)).rowcount


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


# ------------------------------------------------------------------------------------------- investigations
def _inv(r: sqlite3.Row) -> dict:
    out = dict(r)
    out["in_data"] = bool(out["in_data"])
    out["result"] = _loads(out.get("result"), None)
    return out


def list_investigations(analysis_id: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM investigations WHERE analysis_id=? ORDER BY queued_at, rowid",
                           (analysis_id,)).fetchall()
    return [_inv(r) for r in rows]


def get_investigation(inv_id: str) -> Optional[dict]:
    with connect() as con:
        r = con.execute("SELECT * FROM investigations WHERE id=?", (inv_id,)).fetchone()
    return _inv(r) if r else None


def insert_investigation(*, analysis_id: str, node: str, label: str, in_data: bool) -> dict:
    row = {"id": new_id(10), "analysis_id": analysis_id, "node": node, "label": label, "in_data": int(in_data),
           "status": "queued", "queued_at": time.time()}
    with _lock, connect() as con:
        con.execute(f"INSERT INTO investigations ({', '.join(row)}) VALUES ({', '.join(':' + k for k in row)})", row)
    return get_investigation(row["id"])


def update_investigation(inv_id: str, **fields) -> bool:
    """Returns False if the row is gone (deleted or cancelled meanwhile)."""
    cols = ", ".join(f"{k}=?" for k in fields)
    vals = [json.dumps(v) if k == "result" and v is not None else v for k, v in fields.items()]
    with _lock, connect() as con:
        return con.execute(f"UPDATE investigations SET {cols} WHERE id=?", (*vals, inv_id)).rowcount > 0


def delete_investigations(analysis_id: str, node: Optional[str] = None, status: Optional[str] = None) -> int:
    q, args = "DELETE FROM investigations WHERE analysis_id=?", [analysis_id]
    if node is not None:
        q, args = q + " AND node=?", args + [node]
    if status is not None:
        q, args = q + " AND status=?", args + [status]
    with _lock, connect() as con:
        return con.execute(q, args).rowcount


# ------------------------------------------------------------------------------------------- chats
def _chat(r: sqlite3.Row) -> dict:
    out = dict(r)
    out["dataset_id"] = out["analysis_id"]  # the phone app calls analyses "datasets"
    return out


def _message(r: sqlite3.Row) -> dict:
    out = dict(r)
    out["cards"] = _loads(out.get("cards"), [])
    return out


def create_chat(analysis_id: str, title: str = "New chat") -> dict:
    now = time.time()
    row = {"id": new_id(), "analysis_id": analysis_id, "title": title, "created": now, "updated": now}
    with _lock, connect() as con:
        con.execute("INSERT INTO chats (id, analysis_id, title, created, updated) VALUES (:id, :analysis_id, :title, :created, :updated)", row)
    return get_chat(row["id"], with_messages=False)


def list_chats(analysis_id: str) -> list[dict]:
    with connect() as con:
        rows = con.execute("SELECT * FROM chats WHERE analysis_id=? ORDER BY updated DESC", (analysis_id,)).fetchall()
    return [_chat(r) for r in rows]


def get_chat(chat_id: str, with_messages: bool = True) -> Optional[dict]:
    with connect() as con:
        r = con.execute("SELECT * FROM chats WHERE id=?", (chat_id,)).fetchone()
        if not r:
            return None
        out = _chat(r)
        if with_messages:
            rows = con.execute("SELECT * FROM messages WHERE chat_id=? ORDER BY created, rowid", (chat_id,)).fetchall()
            out["messages"] = [_message(m) for m in rows]
    return out


def add_messages(chat_id: str, items: list[tuple[str, str, list]]) -> list[dict]:
    """Store (role, content, cards) messages together, and bump the chat's `updated`."""
    now = time.time()
    rows = [{"id": new_id(), "chat_id": chat_id, "role": role, "content": content, "cards": json.dumps(cards or []),
             "created": now + i * 1e-3} for i, (role, content, cards) in enumerate(items)]
    with _lock, connect() as con:
        con.executemany("INSERT INTO messages (id, chat_id, role, content, cards, created) "
                        "VALUES (:id, :chat_id, :role, :content, :cards, :created)", rows)
        con.execute("UPDATE chats SET updated=? WHERE id=?", (now, chat_id))
    return [{**r, "cards": json.loads(r["cards"])} for r in rows]


def rename_chat(chat_id: str, title: str) -> None:
    with _lock, connect() as con:
        con.execute("UPDATE chats SET title=? WHERE id=?", (title, chat_id))


def delete_chat(chat_id: str) -> None:
    with _lock, connect() as con:
        con.execute("DELETE FROM messages WHERE chat_id=?", (chat_id,))
        con.execute("DELETE FROM chats WHERE id=?", (chat_id,))
