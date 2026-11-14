"""Investigations: the queue, result shaping, restarts and the Reliability Score.

Throwaway data dir, fake language model (short fake searches) and OpenAlex switched off.
"""
import json
import sys
import time

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUSAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CAUSAL_LLM", "fake")
    monkeypatch.setenv("CAUSAL_FAKE_DELAY", "0.6")
    monkeypatch.setenv("CAUSAL_OPENALEX", "off")
    yield tmp_path


def boot():
    from fastapi.testclient import TestClient
    for m in [m for m in sys.modules if m == "app" or m == "services" or m.startswith("services.")]:
        del sys.modules[m]
    import app as app_mod
    return TestClient(app_mod.app)


def finished_analysis(c, sample="lalonde_experimental"):
    s = next(x for x in c.get("/api/samples").json() if x["id"] == sample)
    csv = c.get(f"/api/samples/{s['id']}").content
    jid = c.post("/api/analyze", files={"file": (s["filename"], csv, "text/csv")},
                 data={"claim": s["claim"], "design": s["design"] or "", "roles": json.dumps(s["roles"])}).json()["job_id"]
    for _ in range(600):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            assert j["status"] == "done", j.get("error")
            return j
        time.sleep(0.1)
    raise AssertionError("analysis did not finish")


def wait_all(c, jid, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        items = c.get(f"/api/jobs/{jid}/investigations").json()
        if items and all(i["status"] in ("done", "error") for i in items):
            return items
        time.sleep(0.1)
    raise AssertionError("investigations did not finish")


def test_one_investigation_shapes_its_result(env):
    c = boot()
    j = finished_analysis(c)
    r = c.post(f"/api/jobs/{j['id']}/investigate", json={"node": "treat", "label": "treat"})
    assert r.status_code == 200 and r.json()["status"] in ("queued", "running")
    [inv] = wait_all(c, j["id"])
    assert inv["status"] == "done" and inv["in_data"] is True
    res = inv["result"]
    ids = [f["id"] for f in res["factors"]]
    assert "age" in ids                                     # exact column match -> a dataset node
    assert all(i == "age" or i.startswith("hyp:") for i in ids)
    assert not any("unsourced" in i for i in ids)           # a factor without a real URL is dropped
    age = next(f for f in res["factors"] if f["id"] == "age")
    assert age["in_data"] is True and age["label"] == "age"
    for f in res["factors"]:
        assert 0 <= f["reliability"] <= 100 and f["reliability_breakdown"]["n_sources"] == len(f["sources"])
    # edges: factor -> target, and -> outcome / treatment as the research says
    motivation = next(f["id"] for f in res["factors"] if f["id"].startswith("hyp:motivation"))
    assert {"source": motivation, "target": "earnings_1978"} in [{k: e[k] for k in ("source", "target")} for e in res["edges"]]
    # idempotent: asking again returns the finished one, no duplicate
    again = c.post(f"/api/jobs/{j['id']}/investigate", json={"node": "treat"}).json()
    assert again["id"] == inv["id"] and len(c.get(f"/api/jobs/{j['id']}/investigations").json()) == 1
    # the sidebar summary counts it
    d = next(x for x in c.get("/api/datasets").json() if x["id"] == j["id"])
    assert d["investigations"] == [{"node": "treat", "label": "treat", "status": "done", "found": len(ids)}]


def test_queue_runs_three_and_numbers_the_rest(env):
    c = boot()
    j = finished_analysis(c)
    nodes = [n["id"] for n in j["result"]["influence"]["nodes"] if n["observed"]][:7]
    r = c.post(f"/api/jobs/{j['id']}/investigate-batch", json={"nodes": [{"node": n} for n in nodes]}).json()
    assert len(r["queued"]) == 7 and r["skipped"] == []
    assert r["running"] == 3 and r["queued_count"] == 4
    items = c.get(f"/api/jobs/{j['id']}/investigations").json()
    assert sum(i["status"] == "running" for i in items) == 3
    queued = [i for i in items if i["status"] == "queued"]
    assert [i["position"] for i in queued] == [1, 2, 3, 4]       # FIFO, numbered in line
    # cancel one queued, then all queued
    assert c.delete(f"/api/jobs/{j['id']}/investigations/{queued[0]['node']}").status_code == 200
    assert c.post(f"/api/jobs/{j['id']}/investigations/cancel-queued").json()["cancelled"] == 3
    done = wait_all(c, j["id"])
    assert len(done) == 3 and all(i["status"] == "done" for i in done)
    # delete a finished one
    assert c.delete(f"/api/jobs/{j['id']}/investigations/{done[0]['node']}").status_code == 200
    assert c.delete(f"/api/jobs/{j['id']}/investigations/{done[0]['node']}").status_code == 404
    assert len(c.get(f"/api/jobs/{j['id']}/investigations").json()) == 2


def test_queue_limit_answers_429(env, monkeypatch):
    c = boot()
    from services import investigate
    monkeypatch.setattr(investigate, "MAX_PENDING", 4)
    j = finished_analysis(c)
    nodes = [{"node": f"hyp:n{i}", "in_data": False} for i in range(6)]
    r = c.post(f"/api/jobs/{j['id']}/investigate-batch", json={"nodes": nodes}).json()
    assert len(r["queued"]) == 4 and len(r["skipped"]) == 2
    assert c.post(f"/api/jobs/{j['id']}/investigate", json={"node": "hyp:other", "in_data": False}).status_code == 429
    c.post(f"/api/jobs/{j['id']}/investigations/cancel-queued")


def test_a_failed_search_is_reported_and_can_be_retried(env, monkeypatch):
    c = boot()
    j = finished_analysis(c)
    from services import investigate, llm
    monkeypatch.setattr(llm, "run", lambda *a, **k: (llm._fail("usage limit reached"), None)[1])
    c.post(f"/api/jobs/{j['id']}/investigate", json={"node": "age"})
    [inv] = wait_all(c, j["id"])
    assert inv["status"] == "error" and "usage limit reached" in inv["error"]
    monkeypatch.undo()
    monkeypatch.setenv("CAUSAL_DATA_DIR", str(env)); monkeypatch.setenv("CAUSAL_LLM", "fake")
    monkeypatch.setenv("CAUSAL_FAKE_DELAY", "0.3"); monkeypatch.setenv("CAUSAL_OPENALEX", "off")
    retry = c.post(f"/api/jobs/{j['id']}/investigate", json={"node": "age"}).json()
    assert retry["id"] != inv["id"]                          # a failed one is replaced
    [inv2] = wait_all(c, j["id"])
    assert inv2["status"] == "done"


def test_restart_marks_running_and_queued_as_interrupted(env):
    boot()
    from services import store
    a = store.create_analysis(owner="anon", mode="audit", title="t", filename="t.csv", claim="", params={})
    q = store.insert_investigation(analysis_id=a["id"], node="x", label="x", in_data=True)
    r = store.insert_investigation(analysis_id=a["id"], node="y", label="y", in_data=True)
    store.update_investigation(r["id"], status="running", started_at=time.time())
    boot()  # server start
    from services import store as s2
    for inv_id in (q["id"], r["id"]):
        got = s2.get_investigation(inv_id)
        assert got["status"] == "error" and "restart" in got["error"]


def test_needs_a_graph_and_a_language_model(env, monkeypatch):
    c = boot()
    assert c.post("/api/jobs/nope/investigate", json={"node": "x"}).status_code == 404
    j = finished_analysis(c)
    monkeypatch.setenv("CAUSAL_LLM", "off")
    assert c.post(f"/api/jobs/{j['id']}/investigate", json={"node": "treat"}).status_code == 503


@pytest.mark.parametrize("sources, expected", [
    # unverified (no OpenAlex): citation score 0; publisher page venue 0.8 -> source 0.32
    ([("https://www.jstor.org/stable/1", None)], round(100 * 0.85 * 0.32)),
    # two sources: best 0.32 (publisher page) + corroboration 0.075
    ([("https://www.jstor.org/stable/1", None), ("https://example.com/x", None)], int(100 * (0.85 * 0.32 + 0.075) + 0.5)),
    # three sources: corroboration capped at +0.15
    ([("https://www.nber.org/papers/w1", None)] * 2 + [("https://example.com/x", None)] * 2, int(100 * (0.85 * 0.24 + 0.15) + 0.5)),
])
def test_reliability_formula_offline(monkeypatch, sources, expected):
    monkeypatch.setenv("CAUSAL_OPENALEX", "off")
    from services.reliability import score_factors
    f = [{"sources": [{"title": f"t{i}", "url": u, "doi": d} for i, (u, d) in enumerate(sources)]}]
    score_factors(f)
    assert f[0]["reliability"] == expected


def test_reliability_formula_with_citations(monkeypatch):
    """A journal article with 999 citations scores 0.6*1 + 0.4*1 = 1.0: 85 alone, 0.85 + 0.15 = 100 with two more."""
    from services import reliability
    monkeypatch.setattr(reliability, "lookup", lambda title, url, doi=None: {
        "citations": 999, "work_type": "article", "venue_type": "journal", "venue": "AER", "title": title,
        "openalex_id": "W1"})
    one = [{"sources": [{"title": "a", "url": "https://doi.org/10.1/a", "doi": None}]}]
    three = [{"sources": [{"title": c, "url": "https://doi.org/10.1/" + c, "doi": None} for c in "abc"]}]
    reliability.score_factors(one)
    reliability.score_factors(three)
    assert one[0]["reliability"] == 85
    assert three[0]["reliability"] == 100 and three[0]["reliability_breakdown"]["raw"] == pytest.approx(1.0)
