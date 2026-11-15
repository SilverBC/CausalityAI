"""Rate limits for a shared demo: who may do how much, and a daily cap on Claude calls.

Four layers:
1. A daily budget of Claude calls for the whole server (stored in SQLite, so it survives restarts). Past it, every
   Claude feature says so and analyses fall back to the template summary. This bounds the cost whatever else happens.
2. Per-person limits on the expensive actions, checked twice: per owner (account or guest id) and per client IP. The IP
   limit is a multiple of the per-person one, because a venue's Wi-Fi puts many people behind one address.
3. At most CAUSAL_MAX_ANALYSES analyses compute at once (the others wait their turn; see jobs.run).
4. Upload size and row caps.

Requests from this computer itself (a loopback peer with no proxy headers) are exempt from 2 and 4: that is you, your
tests and the Vite dev server. Behind a tunnel (cloudflared, ngrok) every request also arrives from loopback, but with
a forwarding header, so it is limited and keyed by the visitor's address. Forwarding headers are only believed from a
loopback peer (the tunnel runs on this machine); from anyone else they are ignored, so they cannot be spoofed.

Everything is configured by environment variables (see SETTINGS); CAUSAL_LIMITS=off turns layers 1, 2 and 4 off.
"""
from __future__ import annotations

import datetime as dt
import os
import threading
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Request

from . import store

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def enabled() -> bool:
    return os.environ.get("CAUSAL_LIMITS", "on").lower() not in ("off", "0", "false", "no")


# action -> (env name, default count, window seconds, what the person sees)
ACTIONS = {
    "analyze": ("CAUSAL_LIMIT_ANALYSES", 10, 3600, "analyses"),
    "claim": ("CAUSAL_LIMIT_CLAIMS", 5, 3600, "claim-only searches"),
    "investigate": ("CAUSAL_LIMIT_INVESTIGATIONS", 15, 3600, "investigations"),
    "chat": ("CAUSAL_LIMIT_CHAT", 30, 3600, "chat messages"),
    "signup": ("CAUSAL_LIMIT_SIGNUPS", 10, 3600, "sign-ups from this network"),
    "login_fail": ("CAUSAL_LIMIT_LOGIN_FAILS", 10, 900, "failed log-ins from this network"),
}
IP_ONLY = {"signup", "login_fail"}          # no owner yet; keyed by address alone, at the plain limit
CHAT_GAP_S = 3                              # at least this long between two chat messages from one person


def limit(action: str) -> tuple[int, int]:
    env, default, window, _ = ACTIONS[action]
    return _int(env, default), window


def daily_budget() -> int:
    return _int("CAUSAL_LLM_DAILY", 300)


def ip_factor() -> int:
    return max(1, _int("CAUSAL_LIMIT_IP_FACTOR", 4))


def max_upload_bytes() -> int:
    return _int("CAUSAL_MAX_UPLOAD_MB", 20) * 1024 * 1024


def max_rows() -> int:
    return _int("CAUSAL_MAX_ROWS", 200_000)


def max_analyses() -> int:
    return max(1, _int("CAUSAL_MAX_ANALYSES", 2))


class RateLimited(Exception):
    """HTTP 429 with a plain-words message and a Retry-After in seconds."""

    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = max(1, int(retry_after))


# --------------------------------------------------------------------------------------------- who is asking
def _peer(request: Request) -> str:
    return request.client.host if request.client else ""


def _forwarded(request: Request) -> Optional[str]:
    """The visitor's address from a local tunnel's headers, or None when there are none."""
    h = request.headers
    if h.get("cf-connecting-ip"):
        return h["cf-connecting-ip"].strip()
    xff = h.get("x-forwarded-for")
    if xff:
        return xff.split(",")[-1].strip()   # the entry the proxy itself added; earlier ones are client-supplied
    if h.get("x-real-ip"):
        return h["x-real-ip"].strip()
    return None


def client_ip(request: Request) -> str:
    peer = _peer(request)
    if peer in LOOPBACK:
        return _forwarded(request) or peer
    return peer or "unknown"


def is_local(request: Request) -> bool:
    """This computer itself: a loopback peer that is not relaying someone else."""
    return _peer(request) in LOOPBACK and _forwarded(request) is None


def applies(request: Request) -> bool:
    return enabled() and not is_local(request)


# --------------------------------------------------------------------------------------------- sliding windows
_hits: dict[str, deque] = defaultdict(deque)
_lock = threading.Lock()


def _prune(q: deque, window: int, now: float) -> None:
    while q and q[0] <= now - window:
        q.popleft()


def reset() -> None:
    """Forget every window (tests)."""
    with _lock:
        _hits.clear()


def _wait_text(seconds: float) -> str:
    m = int(seconds // 60) + (1 if seconds % 60 else 0)
    return "a few seconds" if seconds < 60 else f"{m} minute{'s' if m != 1 else ''}"


def check(request: Request, action: str, owner: Optional[str] = None, record: bool = True) -> None:
    """Raise RateLimited if this person (or their network) is over the limit for `action`; otherwise count it."""
    if not applies(request):
        return
    n, window = limit(action)
    noun = ACTIONS[action][3]
    now = time.time()
    keys = [(f"ip:{client_ip(request)}:{action}", n if action in IP_ONLY else n * ip_factor(), "network")]
    if owner and action not in IP_ONLY:
        keys.insert(0, (f"{owner}:{action}", n, "you"))
    with _lock:
        if action == "chat" and owner:
            last = _hits[f"{owner}:chat"]
            if last and now - last[-1] < CHAT_GAP_S:
                raise RateLimited("Slow down a little: one message every few seconds.", CHAT_GAP_S - (now - last[-1]))
        for key, cap, who in keys:
            q = _hits[key]
            _prune(q, window, now)
            if len(q) >= cap:
                wait = q[0] + window - now
                span = "hour" if window == 3600 else f"{window // 60} minutes"
                whose = "You've" if who == "you" else "This network has"
                raise RateLimited(f"{whose} reached the demo limit of {cap} {noun} per {span}. "
                                  f"Try again in {_wait_text(wait)}.", wait)
        if record:
            for key, _, _ in keys:
                _hits[key].append(now)


def record(request: Request, action: str) -> None:
    """Count an event without checking (failed log-ins are counted after they fail)."""
    if not applies(request):
        return
    with _lock:
        _hits[f"ip:{client_ip(request)}:{action}"].append(time.time())


# --------------------------------------------------------------------------------------------- daily Claude budget
_budget_lock = threading.Lock()


def _today() -> str:
    return dt.date.today().isoformat()


def _seconds_to_midnight() -> int:
    now = dt.datetime.now()
    return int((dt.datetime.combine(now.date() + dt.timedelta(days=1), dt.time()) - now).total_seconds())


def calls_today() -> int:
    with store.connect() as con:
        r = con.execute("SELECT calls FROM llm_usage WHERE day=?", (_today(),)).fetchone()
    return int(r["calls"]) if r else 0


BUDGET_MESSAGE = ("The demo's Claude budget for today is used up. Analyses from data still work (with the automatic "
                  "summary); chat, investigations and claim-only searches come back tomorrow.")


def budget_left() -> bool:
    return not enabled() or calls_today() < daily_budget()


def require_budget() -> None:
    """For endpoints whose whole point is a Claude call: refuse up front instead of failing later."""
    if not budget_left():
        raise RateLimited(BUDGET_MESSAGE, _seconds_to_midnight())


def spend_llm_call() -> bool:
    """Count one Claude call. False (and nothing counted) when today's budget is used up. Called by llm.run."""
    if not enabled():
        return True
    with _budget_lock, store.connect() as con:
        day = _today()
        r = con.execute("SELECT calls FROM llm_usage WHERE day=?", (day,)).fetchone()
        used = int(r["calls"]) if r else 0
        if used >= daily_budget():
            return False
        con.execute("INSERT INTO llm_usage(day, calls) VALUES(?, 1) ON CONFLICT(day) DO UPDATE SET calls = calls + 1", (day,))
    return True


def status() -> dict:
    return {"enabled": enabled(), "llm_calls_today": calls_today() if enabled() else None,
            "llm_daily_budget": daily_budget() if enabled() else None, "max_concurrent_analyses": max_analyses()}


# --------------------------------------------------------------------------------------------- concurrent analyses
_slots: Optional[threading.BoundedSemaphore] = None
_slots_n = 0
_slots_lock = threading.Lock()


def analysis_slots() -> threading.BoundedSemaphore:
    """At most max_analyses() analyses compute at once, for everyone (protects the machine's CPU)."""
    global _slots, _slots_n
    with _slots_lock:
        if _slots is None or _slots_n != max_analyses():
            _slots, _slots_n = threading.BoundedSemaphore(max_analyses()), max_analyses()
        return _slots
