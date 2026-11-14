"""Acceptance tests for the web app's foundation (storage, jobs, samples, graph, LLM adapter).

Each test runs against a throwaway data dir (CAUSAL_DATA_DIR) with the fake language model, so no real data or
quota is touched. "Restart" = a fresh import of the app against the same data dir.
"""
import json
import os
import sqlite3
import sys
import time

import pytest

FAST = {"CAUSAL_LLM": "fake"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUSAL_DATA_DIR", str(tmp_path))
    for k, v in FAST.items():
        monkeypatch.setenv(k, v)
    yield tmp_path


def boot():
    """Import the app fresh (as a server start would)."""
    from fastapi.testclient import TestClient
    for m in [m for m in sys.modules if m == "app" or m == "services" or m.startswith("services.")]:
        del sys.modules[m]
    import app as app_mod
    return TestClient(app_mod.app, headers={"X-Guest-Id": "test-guest-0001"})  # like the browser does


def wait(c, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            return j
        time.sleep(0.2)
    raise AssertionError(f"analysis {jid} still running after {timeout}s")


def run_sample(c, s, claim=None):
    csv = c.get(f"/api/samples/{s['id']}").content
    r = c.post("/api/analyze", files={"file": (s["filename"], csv, "text/csv")},
               data={"claim": s["claim"] if claim is None else claim, "design": s["design"] or "",
                     "roles": json.dumps(s["roles"])})
    assert r.status_code == 200, r.text
    return wait(c, r.json()["job_id"])


def test_health_and_samples(env):
    c = boot()
    h = c.get("/api/health").json()
    assert h["llm_backend"] == "fake" and h["llm"] is True and h["llm_last_error"] is None
    assert c.get("/api/health?probe=true").json()["probe_ok"] is True
    ids = [s["id"] for s in c.get("/api/samples").json()]
    assert {"lalonde_experimental", "lalonde_observational", "card_krueger", "rct", "chain"} <= set(ids)
    assert c.get("/api/samples/card_krueger").status_code == 200
    for bad in ("../app.py", "..%2Fapp.py", "real/lalonde_experimental.csv", "nope"):
        assert c.get(f"/api/samples/{bad}").status_code == 404


def test_every_sample_runs_end_to_end(env):
    c = boot()
    for s in c.get("/api/samples").json():
        j = run_sample(c, s)
        assert j["status"] == "done", (s["id"], j["status"], j.get("error"), (j.get("result") or {}).get("report", {}).get("needs_input"))
        res = j["result"]
        assert res["summary"]["text"], s["id"]
        assert res["influence"] and res["influence"].get("nodes"), s["id"]
        assert all(st["status"] in ("done", "skipped") for st in j["steps"]), (s["id"], j["steps"])


def test_lalonde_experimental_numbers(env):
    """This repo's core (not v1's): Lin-adjusted comparison, about +$1,583 with 95% CI about 254 to 2,912."""
    c = boot()
    s = next(x for x in c.get("/api/samples").json() if x["id"] == "lalonde_experimental")
    j = run_sample(c, s)
    rep = j["result"]["report"]
    assert rep["verdict"]["label"] == "Supported"
    assert rep["design"]["kind"] == "experiment"
    est = rep["estimate"]
    assert est["estimate"] == pytest.approx(1583, abs=60)
    assert est["ci95"][0] == pytest.approx(254, abs=80) and est["ci95"][1] == pytest.approx(2912, abs=80)
    assert [st["key"] for st in j["steps"] if st["status"] == "done"][:3] == ["read", "prepare", "claim"]
    inf = j["result"]["influence"]
    assert inf["outcome"] == "earnings_1978" and inf["treatment"] == "treat"
    assert any(e["weight"] is not None for e in inf["edges"])


def test_needs_input_then_answer_reruns_in_place(env):
    c = boot()
    s = next(x for x in c.get("/api/samples").json() if x["id"] == "lalonde_experimental")
    s = {**s, "roles": {"randomized": True}}  # roles not declared: the claim's words don't match the columns
    j = run_sample(c, s)
    assert j["status"] == "needs_input"
    assert {q["role"] for q in j["result"]["report"]["needs_input"]} >= {"treatment"}
    r = c.post(f"/api/jobs/{j['id']}/answer", json={"confirm": {"treatment": "treat", "outcome": "earnings_1978"}})
    assert r.status_code == 200
    j2 = wait(c, j["id"])
    assert j2["status"] == "done" and j2["result"]["report"]["verdict"]["label"] == "Supported"


def test_restart_keeps_finished_work_and_depth_reloads_parquet(env):
    c = boot()
    s = next(x for x in c.get("/api/samples").json() if x["id"] == "lalonde_observational")
    j = run_sample(c, s)
    inf = j["result"]["influence"]
    assert inf["max_depth"] > inf["default_depth"]

    c2 = boot()  # restart: new process state, same data dir, empty frame cache
    j2 = c2.get(f"/api/jobs/{j['id']}").json()
    assert j2["status"] == "done" and j2["result"]["report"] == j["result"]["report"]
    deep = c2.get(f"/api/jobs/{j['id']}/influence?depth={inf['max_depth']}").json()
    assert deep["depth"] == inf["max_depth"] and len(deep["nodes"]) >= len(inf["nodes"])
    shallow = c2.get(f"/api/jobs/{j['id']}/influence?depth=1").json()
    assert all(n["depth"] is not None and n["depth"] <= 1 for n in shallow["nodes"])


def test_restart_marks_running_as_interrupted(env):
    boot()
    from services import store
    a = store.create_analysis(owner="anon", mode="explore", title="t", filename="t.csv", claim="", params={})
    boot()
    from services import store as s2
    got = s2.get_analysis(a["id"])
    assert got["status"] == "error" and "restart" in got["error"]


def test_renaming_a_class_does_not_lose_stored_analyses(env, monkeypatch):
    """v1 pickled jobs and lost them when a class changed. Rows here are JSON: no class is needed to read them."""
    c = boot()
    s = next(x for x in c.get("/api/samples").json() if x["id"] == "rct")
    j = run_sample(c, s)
    import causality.schema as schema
    monkeypatch.delattr(schema, "Report")       # "rename" the classes the result was built from
    monkeypatch.delattr(schema, "Estimate")
    c2 = boot()
    got = c2.get(f"/api/jobs/{j['id']}").json()
    assert got["status"] == "done" and got["result"]["report"]["estimate"]["estimate"] is not None


def test_migration_helper_adds_columns_and_keeps_rows(env):
    boot()
    from services import store
    con = sqlite3.connect(env / "m.db")
    con.row_factory = sqlite3.Row
    con.execute("CREATE TABLE t (id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO t VALUES ('a')")
    store.migrate(con, {"t": "CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY);"}, [("t", "note", "TEXT")])
    store.migrate(con, {"t": "CREATE TABLE IF NOT EXISTS t (id TEXT PRIMARY KEY);"}, [("t", "note", "TEXT")])  # idempotent
    assert [tuple(r) for r in con.execute("SELECT id, note FROM t")] == [("a", None)]


def test_claim_only_and_input_errors(env):
    c = boot()
    r = c.post("/api/analyze", data={"claim": "Job training raises earnings"})
    j = wait(c, r.json()["job_id"])
    assert r.json()["mode"] == "claim-only" and j["status"] == "done"
    assert len(j["result"]["claim_only"]["evidence"]) == 1   # the source without a real URL was dropped
    assert c.post("/api/analyze", data={"claim": ""}).status_code == 400
    bad = c.post("/api/analyze", files={"file": ("x.csv", b"", "text/csv")}, data={"claim": ""})
    assert bad.status_code == 400
    assert c.get("/api/jobs/doesnotexist").status_code == 404


def test_llm_off_is_reported_not_silent(env, monkeypatch):
    monkeypatch.setenv("CAUSAL_LLM", "off")
    c = boot()
    assert c.get("/api/health").json()["llm"] is False
    assert c.post("/api/analyze", data={"claim": "x causes y"}).status_code == 503
    s = next(x for x in c.get("/api/samples").json() if x["id"] == "rct")
    j = run_sample(c, s)
    assert j["status"] == "done" and j["result"]["summary"]["source"] == "template"


def test_api_schema_is_strict_for_structured_outputs():
    from services.llm import _strict
    s = _strict({"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}}},
                                                  "type": {"type": "string"}}})
    assert s["additionalProperties"] is False and s["required"] == ["a", "type"]
    assert s["properties"]["a"]["additionalProperties"] is False and s["properties"]["type"] == {"type": "string"}
