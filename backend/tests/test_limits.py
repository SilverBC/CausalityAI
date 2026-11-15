"""Demo rate limits (services/limits.py): per person and per network, the daily Claude budget, upload caps, the
loopback exemption, tunnel headers, and at most N analyses computing at once."""
import json
import time

import pytest

from test_accounts import boot, wait

A = {"X-Guest-Id": "guest-aaaaaaaa"}
B = {"X-Guest-Id": "guest-bbbbbbbb"}
VISITOR = ("203.0.113.7", 50000)        # a device on the network
TUNNEL = ("127.0.0.1", 50000)           # cloudflared / ngrok run on this machine


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUSAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CAUSAL_LLM", "fake")
    monkeypatch.setenv("CAUSAL_FAKE_DELAY", "0")
    monkeypatch.setenv("CAUSAL_OPENALEX", "off")
    monkeypatch.setenv("CAUSAL_LIMITS", "on")
    for k in ("RESEND_API_KEY", "SMTP_HOST", "MAIL_FROM"):
        monkeypatch.delenv(k, raising=False)
    yield monkeypatch


def client(peer=VISITOR):
    app, TC = boot()
    from services import limits
    limits.reset()
    return TC(app, client=peer)


def claim_only(c, headers):
    return c.post("/api/analyze", headers=headers, data={"claim": "Job training raises earnings"})


def test_per_person_limit_429_with_retry_after_and_plain_message(env):
    env.setenv("CAUSAL_LIMIT_CLAIMS", "2")
    c = client()
    assert claim_only(c, A).status_code == 200
    assert claim_only(c, A).status_code == 200
    r = claim_only(c, A)
    assert r.status_code == 429
    assert "You've reached the demo limit of 2 claim-only searches per hour" in r.json()["detail"]
    assert 0 < int(r.headers["Retry-After"]) <= 3600
    assert claim_only(c, B).status_code == 200          # another person on the same network still can


def test_network_limit_is_a_multiple_of_the_personal_one(env):
    env.setenv("CAUSAL_LIMIT_CLAIMS", "1")
    env.setenv("CAUSAL_LIMIT_IP_FACTOR", "2")
    c = client()
    assert claim_only(c, {"X-Guest-Id": "guest-11111111"}).status_code == 200
    assert claim_only(c, {"X-Guest-Id": "guest-22222222"}).status_code == 200
    r = claim_only(c, {"X-Guest-Id": "guest-33333333"})   # a third new guest id on the same address
    assert r.status_code == 429 and "This network has reached" in r.json()["detail"]


def test_this_computer_is_exempt_but_a_tunnel_is_not(env):
    env.setenv("CAUSAL_LIMIT_CLAIMS", "1")
    local = client(TUNNEL)
    for _ in range(3):                                   # loopback, no proxy headers: you, your tests
        assert claim_only(local, A).status_code == 200
    # through a tunnel: limited, keyed by the visitor's address from the tunnel's header
    cf = {**A, "CF-Connecting-IP": "198.51.100.1"}
    assert claim_only(local, cf).status_code == 200
    assert claim_only(local, cf).status_code == 429
    other = {**B, "CF-Connecting-IP": "198.51.100.2"}
    assert claim_only(local, other).status_code == 200


def test_forwarding_headers_from_a_non_local_peer_are_ignored(env):
    env.setenv("CAUSAL_LIMIT_CLAIMS", "1")
    env.setenv("CAUSAL_LIMIT_IP_FACTOR", "1")
    c = client()
    assert claim_only(c, {"X-Guest-Id": "guest-11111111", "CF-Connecting-IP": "1.1.1.1"}).status_code == 200
    # a fresh guest id and a made-up address do not get round the network limit
    r = claim_only(c, {"X-Guest-Id": "guest-22222222", "CF-Connecting-IP": "2.2.2.2"})
    assert r.status_code == 429


def test_daily_claude_budget_stops_claude_features_but_not_analyses(env):
    env.setenv("CAUSAL_LLM_DAILY", "1")
    c = client()
    j = wait(c, claim_only(c, A).json()["job_id"], A)             # spends the one call
    assert j["status"] == "done"
    r = claim_only(c, A)
    assert r.status_code == 429 and "Claude budget for today is used up" in r.json()["detail"]
    # an analysis from data still runs; its summary falls back to the template
    csv = c.get("/api/samples/rct").content
    r = c.post("/api/analyze", headers=A, files={"file": ("rct.csv", csv, "text/csv")},
               data={"claim": "treated increases score by 1.5", "roles": json.dumps({"randomized": True})})
    assert r.status_code == 200
    done = wait(c, r.json()["job_id"], A)
    assert done["status"] == "done" and done["result"]["summary"]["source"] == "template"
    h = c.get("/api/health").json()["limits"]
    assert h["llm_calls_today"] == 1 and h["llm_daily_budget"] == 1
    # chat refuses up front
    chat = c.post(f"/api/datasets/{done['id']}/chats", headers=A).json()
    r = c.post(f"/api/chats/{chat['id']}/messages", headers=A, json={"text": "What drives this?"})
    assert r.status_code == 429


def test_budget_survives_a_restart(env):
    env.setenv("CAUSAL_LLM_DAILY", "1")
    c = client()
    wait(c, claim_only(c, A).json()["job_id"], A)
    c = client()                                                  # a fresh app on the same data folder
    assert claim_only(c, A).status_code == 429


def test_chat_spacing_and_failed_logins(env):
    env.setenv("CAUSAL_LIMIT_LOGIN_FAILS", "2")
    c = client()
    creds = {"email": "a@example.com", "password": "correct horse battery"}
    assert c.post("/api/auth/signup", json=creds).status_code == 200
    c.post("/api/auth/logout")
    bad = {**creds, "password": "wrong password!"}
    assert c.post("/api/auth/login", json=bad).status_code == 401
    assert c.post("/api/auth/login", json=bad).status_code == 401
    r = c.post("/api/auth/login", json=creds)                     # even the right password waits now
    assert r.status_code == 429 and "failed log-ins" in r.json()["detail"]

    c2 = client()
    csv = c2.get("/api/samples/rct").content
    jid = c2.post("/api/analyze", headers=A, files={"file": ("rct.csv", csv, "text/csv")},
                  data={"claim": "treated increases score by 1.5", "roles": json.dumps({"randomized": True})}).json()["job_id"]
    wait(c2, jid, A)
    chat = c2.post(f"/api/datasets/{jid}/chats", headers=A).json()
    assert c2.post(f"/api/chats/{chat['id']}/messages", headers=A, json={"text": "hi"}).status_code == 200
    r = c2.post(f"/api/chats/{chat['id']}/messages", headers=A, json={"text": "again"})
    assert r.status_code == 429 and "one message every few seconds" in r.json()["detail"]


def test_upload_caps_for_visitors_only(env):
    env.setenv("CAUSAL_MAX_ROWS", "100")
    rows = "x,y\n" + "\n".join(f"{i},{i % 2}" for i in range(150))
    c = client()
    r = c.post("/api/analyze", headers=A, files={"file": ("big.csv", rows, "text/csv")})
    assert r.status_code == 413 and "at most 100 rows" in r.json()["detail"]
    local = client(TUNNEL)
    assert local.post("/api/analyze", headers=A, files={"file": ("big.csv", rows, "text/csv")}).status_code == 200


def test_off_switch(env):
    env.setenv("CAUSAL_LIMITS", "off")
    env.setenv("CAUSAL_LIMIT_CLAIMS", "1")
    c = client()
    for _ in range(3):
        assert claim_only(c, A).status_code == 200
    assert c.get("/api/health").json()["limits"]["enabled"] is False


def test_at_most_n_analyses_compute_at_once(env):
    env.setenv("CAUSAL_MAX_ANALYSES", "1")
    c = client()
    from services import limits
    slot = limits.analysis_slots()
    assert slot.acquire(blocking=False)                          # someone else's analysis holds the only slot
    try:
        jid = claim_only(c, A).json()["job_id"]
        time.sleep(0.5)
        waiting = c.get(f"/api/jobs/{jid}", headers=A).json()
        assert waiting["status"] == "running"
        assert waiting["steps"][0]["key"] == "wait" and "runs 1 at a time" in waiting["steps"][0]["label"]
    finally:
        slot.release()
    assert wait(c, jid, A)["status"] == "done"                   # it starts as soon as the slot frees
