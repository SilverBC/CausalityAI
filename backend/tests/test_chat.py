"""The Investigator chat: replies, cards, investigations from chat, failures, ownership.

Throwaway data dir, fake language model, OpenAlex off.
"""
import json
import sys
import time

import pytest

A = {"X-Guest-Id": "guest-aaaaaaaa"}
B = {"X-Guest-Id": "guest-bbbbbbbb"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUSAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CAUSAL_LLM", "fake")
    monkeypatch.setenv("CAUSAL_FAKE_DELAY", "0.3")
    monkeypatch.setenv("CAUSAL_OPENALEX", "off")
    yield tmp_path


def boot():
    from fastapi.testclient import TestClient
    for m in [m for m in sys.modules if m == "app" or m == "services" or m.startswith("services.")]:
        del sys.modules[m]
    import app as app_mod
    return TestClient(app_mod.app, headers=A)


def lalonde(c):
    s = next(x for x in c.get("/api/samples").json() if x["id"] == "lalonde_experimental")
    csv = c.get("/api/samples/lalonde_experimental").content
    jid = c.post("/api/analyze", files={"file": (s["filename"], csv, "text/csv")},
                 data={"claim": s["claim"], "design": s["design"], "roles": json.dumps(s["roles"])}).json()["job_id"]
    for _ in range(600):
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] != "running":
            assert j["status"] == "done"
            return j
        time.sleep(0.1)


def test_reply_with_real_numbers_and_valid_cards(env):
    c = boot()
    j = lalonde(c)
    chat = c.post(f"/api/datasets/{j['id']}/chats").json()
    assert chat["title"] == "New chat" and chat["dataset_id"] == j["id"]
    r = c.post(f"/api/chats/{chat['id']}/messages", json={"text": "Does the data support the claim?"})
    assert r.status_code == 200, r.text
    out = r.json()
    est = j["result"]["report"]["estimate"]["estimate"]
    assert str(round(est))[:4] in out["assistant"]["content"]          # the digest carried the real estimate
    assert out["assistant"]["cards"] == [{"kind": "estimate", "ref": None}]   # the invented check card was dropped
    assert out["chat"]["title"] == "Does the data support the claim?"   # named after the first message
    r = c.post(f"/api/chats/{chat['id']}/messages", json={"text": "What's the biggest problem with this data?"}).json()
    assert r["assistant"]["cards"] == [{"kind": "check", "ref": "balance"}]
    full = c.get(f"/api/chats/{chat['id']}").json()
    assert [m["role"] for m in full["messages"]] == ["user", "assistant", "user", "assistant"]
    assert full["title"] == "Does the data support the claim?"          # the title doesn't change later


def test_asking_to_investigate_starts_one(env):
    c = boot()
    j = lalonde(c)
    chat = c.post(f"/api/datasets/{j['id']}/chats").json()
    out = c.post(f"/api/chats/{chat['id']}/messages", json={"text": "Investigate what else could drive the treatment"}).json()
    assert out["assistant"]["cards"] == [{"kind": "investigation", "ref": "treat"}]
    assert "hypothetical" in out["assistant"]["content"]
    invs = c.get(f"/api/jobs/{j['id']}/investigations").json()
    assert [i["node"] for i in invs] == ["treat"] and invs[0]["status"] in ("queued", "running", "done")


def test_failure_is_reported_and_nothing_is_stored(env, monkeypatch):
    c = boot()
    j = lalonde(c)
    chat = c.post(f"/api/datasets/{j['id']}/chats").json()
    from services import llm
    monkeypatch.setattr(llm, "run", lambda *a, **k: (llm._fail("You've hit your usage limit · resets 5pm"), None)[1])
    r = c.post(f"/api/chats/{chat['id']}/messages", json={"text": "Hello?"})
    assert r.status_code == 503 and r.json()["detail"] == "Claude is unavailable: You've hit your usage limit · resets 5pm"
    assert c.get(f"/api/chats/{chat['id']}").json()["messages"] == []    # the user can simply send it again
    assert c.get(f"/api/chats/{chat['id']}").json()["title"] == "New chat"


def test_limits_and_ownership(env):
    c = boot()
    j = lalonde(c)
    chat = c.post(f"/api/datasets/{j['id']}/chats").json()
    assert c.post(f"/api/chats/{chat['id']}/messages", json={"text": "   "}).status_code == 400
    assert c.post(f"/api/chats/{chat['id']}/messages", json={"text": "x" * 4001}).status_code == 400
    for method, path, body in [("get", f"/api/chats/{chat['id']}", None), ("delete", f"/api/chats/{chat['id']}", None),
                               ("post", f"/api/chats/{chat['id']}/messages", {"text": "hi"}),
                               ("post", f"/api/datasets/{j['id']}/chats", None), ("get", f"/api/datasets/{j['id']}", None)]:
        kw = {"json": body} if body else {}
        assert getattr(c, method)(path, headers=B, **kw).status_code == 404, path
    assert c.get(f"/api/chats/{chat['id']}").status_code == 200


def test_datasets_list_chats_newest_activity_first_and_delete_cascades(env):
    c = boot()
    j1 = lalonde(c)
    j2 = lalonde(c)
    assert [d["id"] for d in c.get("/api/datasets").json()] == [j2["id"], j1["id"]]
    chat = c.post(f"/api/datasets/{j1['id']}/chats").json()
    c.post(f"/api/chats/{chat['id']}/messages", json={"text": "Does the data support the claim?"})
    ds = c.get("/api/datasets").json()
    assert [d["id"] for d in ds] == [j1["id"], j2["id"]]                 # the chat made j1 the most recent
    assert ds[0]["chats"][0]["title"] == "Does the data support the claim?"
    assert c.delete(f"/api/jobs/{j1['id']}").status_code == 200
    assert c.get(f"/api/chats/{chat['id']}").status_code == 404
    from services import store
    with store.connect() as con:
        assert con.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0
    # deleting one chat
    chat2 = c.post(f"/api/datasets/{j2['id']}/chats").json()
    assert c.delete(f"/api/chats/{chat2['id']}").status_code == 200
    assert c.get("/api/datasets").json()[0]["chats"] == []


def test_digest_is_compact_and_honest(env):
    c = boot()
    j = lalonde(c)
    from services import chat, store
    d = chat.digest(store.get_analysis(j["id"]))
    assert len(json.dumps(d, default=str)) <= chat.DIGEST_CHARS
    assert d["estimate"]["ci95"] == j["result"]["report"]["estimate"]["ci95"]
    assert d["verdict"]["label"] == "Supported" and d["assumed_not_tested"]
    assert {"balance", "hidden_confounding"} <= set(d["checks"])
    p = chat.prompt(store.get_analysis(j["id"]), [{"role": "user", "content": f"m{i}"} for i in range(20)], "next")
    assert "USER: m19" in p and "USER: m7" not in p                       # only the last 12 messages
