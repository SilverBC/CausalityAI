"""Reliability score for literature-suggested (hypothetical) factors, from their sources' bibliometrics.

Each source is looked up in OpenAlex (by DOI when the URL has one, else by title). Per source:
    citation score = min(1, log10(citations + 1) / 3)            # 1,000+ citations -> 1.0
    venue score    = journal article 1.0 · book / chapter / conference 0.8 · working paper / preprint 0.6
                     · institutional report 0.5 · other web page 0.25   (by OpenAlex type, else by URL domain)
    source score   = 0.6 * citation score + 0.4 * venue score    (sources OpenAlex can't find get no citation credit)
Per factor:
    reliability = 100 * min(1, 0.85 * best source score + 0.075 * (min(#sources, 3) - 1))
i.e. mostly the strongest source, with a small bonus for independent corroboration. It measures how well-established
the cited research is, not whether the factor actually confounds this dataset.

Ported from v1. CAUSAL_OPENALEX=off skips the network (tests); sources then get no citation credit.
"""
from __future__ import annotations

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from urllib.parse import unquote, urlparse

import httpx

OPENALEX = "https://api.openalex.org/works"
HEADERS = {"User-Agent": "causal-audit/2.0"}
TIMEOUT = 8
_CACHE: dict[str, dict | None] = {}

DOI_RE = re.compile(r"10\.\d{4,9}/[^\s?#]+", re.I)
WORKING_PAPER_DOMAINS = ("nber.org", "ssrn.com", "repec.org", "iza.org", "arxiv.org", "econpapers", "ideas.repec", "papers.")
INSTITUTION_DOMAINS = (".gov", "mdrc.org", "worldbank.org", "oecd.org", "who.int", "un.org", "brookings.edu", "rand.org",
                       "urban.org", "irp.wisc.edu", "imf.org", "europa.eu")
PUBLISHER_DOMAINS = ("doi.org", "jstor.org", "sciencedirect.com", "springer", "wiley.com", "tandfonline.com", "oup.com",
                     "cambridge.org", "sagepub.com", "aeaweb.org", "journals.", "nature.com", "science.org", "pnas.org",
                     "nih.gov/pmc", "ncbi.nlm.nih.gov", "uchicago.edu/doi")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).strip()


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def _same_title(a: str, b: str) -> bool:
    """Close match, or one title is the other plus/minus a subtitle."""
    na, nb = _norm(a), _norm(b)
    short, long_ = sorted((na, nb), key=len)
    return _similar(na, nb) >= 0.85 or (len(short) >= 25 and long_.startswith(short))


def _clean_title(t: str) -> str:
    """Strip notes the model sometimes adds: '(Author, NBER w6983; ...)', 'RePEc entry: ...', quotes."""
    t = re.sub(r"\s*[\(\[][^)\]]*[\)\]]\s*$", "", t or "").strip().strip('"“”')
    t = re.sub(r"^[\w .&-]{0,40}\b(entry|page|record|summary|abstract|pdf)\s*:\s*", "", t, flags=re.I)
    # leading "Author, " / "Author & Author, " / "Author et al., "
    t = re.sub(r"^[A-Z][\w'\-]+(?:\s+(?:&|and)\s+[A-Z][\w'\-]+)?(?:\s+et al\.?)?,\s+(?=[A-Z])", "", t)
    return t.strip()


def _doi_from(url: str, doi: str | None) -> str | None:
    for text in (doi, unquote(url or "")):
        m = DOI_RE.search(text or "")
        if m:
            return m.group(0).rstrip(".")
    m = re.search(r"nber\.org/papers/(w\d+)", url or "", re.I)  # NBER working papers have DOI 10.3386/wNNNN
    return f"10.3386/{m.group(1).lower()}" if m else None


def _summarize(work: dict) -> dict:
    src = (work.get("primary_location") or {}).get("source") or {}
    return {"openalex_id": work.get("id"), "title": work.get("display_name"), "year": work.get("publication_year"),
            "citations": work.get("cited_by_count") or 0, "work_type": work.get("type"),
            "venue": src.get("display_name"), "venue_type": src.get("type")}


def lookup(title: str, url: str, doi: str | None = None) -> dict | None:
    """Find a source in OpenAlex. Returns a summary dict or None (not found / offline)."""
    key = f"{url}|{title}|{doi}".lower()
    if key in _CACHE:
        return _CACHE[key]
    if os.environ.get("CAUSAL_OPENALEX", "").lower() == "off":
        return None
    found = None
    try:
        d = _doi_from(url, doi)
        if d:
            r = httpx.get(f"{OPENALEX}/doi:{d}", timeout=TIMEOUT, headers=HEADERS)
            if r.status_code == 200:
                found = _summarize(r.json())
        clean = _clean_title(title)
        queries = [clean] + ([clean.split(":")[0]] if ":" in clean and len(clean.split(":")[0]) >= 25 else [])
        for q in queries if not found and clean else []:
            r = httpx.get(OPENALEX, params={"search": q[:250], "per-page": 8}, timeout=TIMEOUT, headers=HEADERS)
            if r.status_code != 200:
                continue
            matches = [w for w in r.json().get("results", []) if _same_title(clean, w.get("display_name", ""))]
            if matches:  # the same paper often exists as working paper + journal article; take the best-cited version
                found = _summarize(max(matches, key=lambda w: w.get("cited_by_count") or 0))
                break
    except Exception:  # noqa: BLE001  (offline or OpenAlex down: no citation credit)
        found = None
    _CACHE[key] = found
    return found


def _venue_score(oa: dict | None, url: str) -> tuple[float, str]:
    if oa:
        wt, vt = oa.get("work_type"), oa.get("venue_type")
        if wt in ("article", "review") and vt == "journal":
            return 1.0, "peer-reviewed journal"
        if wt in ("book", "book-chapter") or vt in ("book series", "ebook platform", "conference"):
            return 0.8, "book / conference"
        if wt in ("preprint", "report") or vt == "repository":
            return 0.6, "working paper / preprint"
        if wt in ("article", "review"):
            return 0.8, "article"
    host = (urlparse(url or "").netloc or "").lower() + (urlparse(url or "").path or "").lower()
    if any(d in host for d in PUBLISHER_DOMAINS):
        return 0.8, "publisher page"
    if any(d in host for d in WORKING_PAPER_DOMAINS):
        return 0.6, "working paper / preprint"
    if any(d in host for d in INSTITUTION_DOMAINS):
        return 0.5, "institutional report"
    return 0.25, "web page"


def score_source(src: dict) -> dict:
    oa = lookup(src.get("title", ""), src.get("url", ""), src.get("doi"))
    citations = oa["citations"] if oa else None
    cite_score = min(1.0, math.log10(citations + 1) / 3) if citations is not None else 0.0
    venue_score, venue_kind = _venue_score(oa, src.get("url", ""))
    return {**src, "verified": bool(oa), "citations": citations, "venue": (oa or {}).get("venue"), "venue_kind": venue_kind,
            "matched_title": (oa or {}).get("title"), "openalex_id": (oa or {}).get("openalex_id"),
            "cite_score": round(cite_score, 3), "venue_score": venue_score,
            "source_score": round(0.6 * cite_score + 0.4 * venue_score, 3)}


def score_factors(factors: list[dict]) -> list[dict]:
    """Annotate each factor's sources with bibliometrics and add a 0-100 `reliability` (in place; returns factors)."""
    pairs = [(f, i, s) for f in factors for i, s in enumerate(f.get("sources", []))]
    with ThreadPoolExecutor(max_workers=6) as pool:
        scored = list(pool.map(lambda p: score_source(p[2]), pairs))
    for (f, i, _), s in zip(pairs, scored):
        f["sources"][i] = s
    for f in factors:
        srcs = f.get("sources", [])
        if not srcs:
            f["reliability"] = 0
            continue
        best = max(s["source_score"] for s in srcs)
        bonus = 0.075 * (min(len(srcs), 3) - 1)
        raw = 0.85 * best + bonus
        f["reliability"] = int(100 * min(1.0, raw) + 0.5)  # round half up, matching the breakdown shown to users
        f["reliability_breakdown"] = {"best_source_score": best, "best_weighted": round(0.85 * best, 3), "n_sources": len(srcs),
                                      "corroboration_bonus": round(bonus, 3), "raw": round(raw, 3), "capped": raw > 1}
        f["sources"] = sorted(srcs, key=lambda s: -s["source_score"])
    return factors
