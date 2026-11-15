"""Accounts, sessions, guests and ownership: who is asking, and may they see this?

- Accounts: email + password. Passwords are hashed with scrypt (stdlib; n=2^14, r=8, p=1, random salt) and
  compared in constant time. Sessions are random tokens in an HttpOnly, SameSite=Lax cookie (Secure on https)
  that last 30 days.
- Guests: the browser keeps a random id in localStorage and sends it as X-Guest-Id ([A-Za-z0-9-]{8,64}). Guest work
  is owned by "guest:<id>" and lasts as long as that browser keeps the id; signing up or in moves it to the account.
- Ownership: every analysis has an owner, "user:<id>" or "guest:<id>". Investigations and chats follow
  their analysis. `owned_analysis` is the one gate: anything the caller does not own answers 404, never 403, so ids
  can't be probed.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import time
import uuid
from typing import Optional

from fastapi import HTTPException, Request

from . import store

COOKIE = "ca_session"
SESSION_DAYS = 30
MIN_PASSWORD = 8
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
GUEST_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")
NOT_FOUND = "Unknown analysis (it may belong to another account, or it was deleted)."


class AuthError(ValueError):
    pass


# --------------------------------------------------------------------------------------------- passwords
def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or os.urandom(16)
    h = hashlib.scrypt(password.encode(), salt=salt, n=2 ** 14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${h.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        return hmac.compare_digest(hash_password(password, bytes.fromhex(salt)).split("$")[2], h)
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------------------------- accounts
def signup(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    if not EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    if len(password or "") < MIN_PASSWORD:
        raise AuthError(f"Use a password of at least {MIN_PASSWORD} characters.")
    user = {"id": uuid.uuid4().hex[:12], "email": email, "pw_hash": hash_password(password), "created": time.time()}
    with store._lock, store.connect() as con:
        if con.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise AuthError("An account with this email already exists. Sign in instead.")
        con.execute("INSERT INTO users (id, email, pw_hash, created) VALUES (:id, :email, :pw_hash, :created)", user)
    return {"id": user["id"], "email": email}


def login(email: str, password: str) -> dict:
    email = (email or "").strip().lower()
    with store.connect() as con:
        r = con.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    # verify against a dummy hash when the email is unknown, so timing does not reveal which emails exist
    ok = verify_password(password or "", r["pw_hash"] if r else _DUMMY_HASH)
    if not r or not ok:
        raise AuthError("Wrong email or password.")
    return {"id": r["id"], "email": r["email"]}


_DUMMY_HASH = hash_password(secrets.token_hex(8))


def user_count() -> int:
    with store.connect() as con:
        return con.execute("SELECT COUNT(*) FROM users").fetchone()[0]


def new_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with store._lock, store.connect() as con:
        con.execute("INSERT INTO sessions (token, user_id, created) VALUES (?, ?, ?)", (token, user_id, time.time()))
    return token


def end_session(token: Optional[str]) -> None:
    if token:
        with store._lock, store.connect() as con:
            con.execute("DELETE FROM sessions WHERE token=?", (token,))


def current_user(request: Request) -> Optional[dict]:
    token = request.cookies.get(COOKIE)
    if not token:
        return None
    with store.connect() as con:
        r = con.execute("SELECT u.id, u.email, s.created FROM sessions s JOIN users u ON u.id = s.user_id "
                        "WHERE s.token=?", (token,)).fetchone()
    if not r or time.time() - r["created"] > SESSION_DAYS * 86400:
        return None
    return {"id": r["id"], "email": r["email"]}


def guest_id(request: Request) -> Optional[str]:
    g = request.headers.get("x-guest-id", "")
    return g if GUEST_RE.match(g) else None


# --------------------------------------------------------------------------------------------- ownership
def owner_of(request: Request) -> Optional[str]:
    """'user:<id>' when signed in, else 'guest:<id>', else None (no session and no valid guest id)."""
    u = current_user(request)
    if u:
        return f"user:{u['id']}"
    g = guest_id(request)
    return f"guest:{g}" if g else None


def require_owner(request: Request) -> str:
    who = owner_of(request)
    if not who:
        raise HTTPException(401, "Reload the page and try again (no guest id or session was sent).")
    return who


def owned_analysis(request: Request, analysis_id: str) -> dict:
    """The analysis if the caller owns it; 404 otherwise, also for other people's analyses."""
    who = owner_of(request)
    a = store.get_analysis(analysis_id) if who else None
    if not a or a["owner"] != who:
        raise HTTPException(404, NOT_FOUND)
    return a


def owned_chat(request: Request, chat_id: str) -> tuple[dict, dict]:
    """(chat with messages, its analysis) if the caller owns the analysis; 404 otherwise."""
    c = store.get_chat(chat_id)
    who = owner_of(request)
    a = store.get_analysis(c["analysis_id"]) if c and who else None
    if not a or a["owner"] != who:
        raise HTTPException(404, "Unknown chat (it may belong to another account, or it was deleted).")
    return c, a
