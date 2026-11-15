"""services/links.py: sources whose links are known not to open are dropped. No network: the check is faked."""
import socket

import pytest

from services import links


@pytest.fixture
def on(monkeypatch):
    monkeypatch.setenv("CAUSAL_LINK_CHECK", "on")


def src(url):
    return {"title": "t", "year": 2020, "url": url, "doi": None}


def test_dead_sources_are_dropped_and_the_rest_kept(on):
    items = [{"name": "a", "sources": [src("https://ok.example/1"), src("https://gone.example/2")]},
             {"name": "b", "sources": [src("https://gone.example/2")]},
             {"name": "c"}]
    seen = []
    n = links.drop_dead_sources(items, check=lambda u, t: seen.append(u) or "gone" in u)
    assert n == 2 and sorted(seen) == ["https://gone.example/2", "https://ok.example/1"]      # each link asked once
    assert [s["url"] for s in items[0]["sources"]] == ["https://ok.example/1"] and items[1]["sources"] == []


def test_switched_off_checks_nothing(monkeypatch):
    monkeypatch.setenv("CAUSAL_LINK_CHECK", "off")
    items = [{"sources": [src("https://gone.example/2")]}]
    assert links.drop_dead_sources(items, check=lambda u, t: True) == 0 and len(items[0]["sources"]) == 1


def test_a_failing_check_keeps_the_source(on):
    def boom(u, t):
        raise RuntimeError("no network")
    assert links.dead(["https://x.example/"], check=boom) == set()


def test_only_known_dead_answers_count(on, monkeypatch):
    monkeypatch.setattr(links, "_public_host", lambda u: True)
    for status, dead in ((200, False), (404, True), (410, True), (403, False), (429, False), (500, False), (0, False)):
        monkeypatch.setattr(links, "_fetch", lambda u, t, s=status: s)
        assert links.is_dead("https://x.example/p") is dead, status


def test_missing_host_is_dead_and_private_addresses_are_never_fetched(on, monkeypatch):
    called = []
    monkeypatch.setattr(links, "_fetch", lambda u, t: called.append(u) or 200)

    def lookup(host, port, **kw):
        if host == "nowhere.example":
            raise socket.gaierror(socket.EAI_NONAME, "not known")
        ip = {"inside.example": "10.0.0.5", "localhost": "127.0.0.1"}.get(host, "93.184.216.34")
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    assert links.is_dead("https://nowhere.example/a") and links.is_dead("http://inside.example/a")
    assert links.is_dead("http://localhost:8770/api/health") and links.is_dead("ftp://files.example/a")
    assert called == []
    assert not links.is_dead("https://paper.example/a") and called == ["https://paper.example/a"]


def test_investigation_drops_a_factor_left_with_no_live_source(on, monkeypatch):
    from services import investigate
    out = {"summary": "s", "factors": [
        {"name": "Parent income", "affects": ["target"], "explanation": "e", "dataset_column": None,
         "sources": [src("https://gone.example/1")]},
        {"name": "School quality", "affects": ["target"], "explanation": "e", "dataset_column": None,
         "sources": [src("https://ok.example/2"), src("https://gone.example/3")]}]}
    monkeypatch.setattr(investigate.llm, "run", lambda *a, **k: out)
    monkeypatch.setattr(investigate.llm, "backend", lambda: "gemini")
    monkeypatch.setattr(investigate.links, "is_dead", lambda u, t: "gone" in u)
    monkeypatch.setattr(investigate, "known_labels", lambda a: [])
    monkeypatch.setattr(investigate, "score_factors", lambda f: f)
    a = {"id": "x", "claim": "", "result": {"report": {"data": {"roles": {"treatment": "t", "outcome": "y"},
                                                                "columns": [{"name": "t"}, {"name": "y"}]}}}}
    r = investigate.search(a, "education", "education", False)
    assert [f["label"] for f in r["factors"]] == ["School quality"]
    assert [s["url"] for s in r["factors"][0]["sources"]] == ["https://ok.example/2"]
