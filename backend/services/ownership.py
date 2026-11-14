"""Who owns what. Every analysis lookup goes through `owned_analysis`, so access rules live in one place.

For now there are no accounts: every caller is "anon" and sees every analysis. Prompt 04 replaces `owner_of`
with users ("user:<id>") and guests ("guest:<id>"), and `owned_analysis` then answers 404 for other people's
analyses (never 403, so ids can't be probed).
"""
from __future__ import annotations

from fastapi import HTTPException, Request

from . import store

ANON = "anon"
NOT_FOUND = "Unknown analysis (it may have been deleted)."


def owner_of(request: Request) -> str:
    return ANON


def owned_analysis(request: Request, analysis_id: str) -> dict:
    a = store.get_analysis(analysis_id)
    if not a or a["owner"] != owner_of(request):
        raise HTTPException(404, NOT_FOUND)
    return a
