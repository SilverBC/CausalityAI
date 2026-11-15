"""Do the links a model cites open?

A model that searches the web still sometimes writes a link that is wrong or made up. Before a source is shown,
its link is requested once. A source is dropped only when the link is known to be dead: the host does not exist, or
the page answers 404 or 410. Anything else (a site that blocks robots, a timeout, a server error) is kept, because
many publishers refuse automated requests for pages that are real.

Only public web addresses are requested: a link to this machine or a private network is treated as dead without
being fetched, at every redirect.

CAUSAL_LINK_CHECK=off skips the check."""
from __future__ import annotations

import ipaddress
import os
import socket
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

TIMEOUT_S = 8
MAX_LINKS = 40
DEAD_STATUS = (404, 410)
AGENT = "Mozilla/5.0 (compatible; link-check)"


def enabled() -> bool:
    return os.environ.get("CAUSAL_LINK_CHECK", "").lower() != "off"


class _Private(Exception):
    pass


def _public_host(url: str) -> Optional[bool]:
    """True: a public address. False: not one, or the host does not exist. None: could not tell (lookup failed)."""
    p = urllib.parse.urlsplit(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return False
    try:
        infos = socket.getaddrinfo(p.hostname, p.port or (443 if p.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        return False if e.errno in (socket.EAI_NONAME, getattr(socket, "EAI_NODATA", socket.EAI_NONAME), 11001) else None
    except OSError:
        return None
    return bool(infos) and all(ipaddress.ip_address(i[4][0]).is_global for i in infos)


class _Redirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if _public_host(newurl) is False:
            raise _Private(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch(url: str, timeout: float) -> int:
    """The HTTP status of one GET (redirects followed). 0 when no answer came back."""
    opener = urllib.request.build_opener(_Redirects)
    req = urllib.request.Request(url, headers={"User-Agent": AGENT, "Accept": "text/html,application/pdf,*/*"})
    try:
        with opener.open(req, timeout=timeout) as r:
            r.read(1)
            return r.status
    except urllib.error.HTTPError as e:
        return e.code
    except _Private:
        return 404
    except Exception:  # noqa: BLE001  (timeout, reset, TLS trouble: unknown, not dead)
        return 0


def is_dead(url: str, timeout: float = TIMEOUT_S) -> bool:
    """True only when the link is known not to open."""
    public = _public_host(url)
    if public is False:
        return True
    if public is None:
        return False
    return _fetch(url, timeout) in DEAD_STATUS


def dead(urls, timeout: float = TIMEOUT_S, check=None) -> set[str]:
    """The links in `urls` that are known not to open. Checked at the same time; at most MAX_LINKS are checked."""
    todo = list(dict.fromkeys(u for u in urls if isinstance(u, str) and u))[:MAX_LINKS]
    if not todo or not enabled():
        return set()
    check = check or is_dead
    with ThreadPoolExecutor(max_workers=min(8, len(todo))) as pool:
        flags = list(pool.map(lambda u: _safe(check, u, timeout), todo))
    return {u for u, bad in zip(todo, flags) if bad}


def _safe(check, url, timeout) -> bool:
    try:
        return bool(check(url, timeout))
    except Exception:  # noqa: BLE001
        return False


def drop_dead_sources(items: list[dict], check=None) -> int:
    """Remove sources with dead links from each item's "sources" list, in place. Returns how many were removed."""
    bad = dead([s.get("url") for it in items for s in it.get("sources") or [] if isinstance(s, dict)], check=check)
    if not bad:
        return 0
    n = 0
    for it in items:
        kept = [s for s in it.get("sources") or [] if not (isinstance(s, dict) and s.get("url") in bad)]
        n += len(it.get("sources") or []) - len(kept)
        it["sources"] = kept
    return n
