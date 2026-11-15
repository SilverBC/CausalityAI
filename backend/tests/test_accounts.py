"""Accounts, guests, ownership, deletion and the 'analysis ready' email.

Throwaway data dir, fake language model, OpenAlex off, no real email sent.
"""
import io
import json
import sys
import time
import urllib.error

import pytest

A = {"X-Guest-Id": "guest-aaaaaaaa"}
B = {"X-Guest-Id": "guest-bbbbbbbb"}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("CAUSAL_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("CAUSAL_LLM", "fake")
    monkeypatch.setenv("CAUSAL_FAKE_DELAY", "0.3")
    monkeypatch.setenv("CAUSAL_OPENALEX", "off")
    for k in ("RESEND_API_KEY", "SMTP_HOST", "MAIL_FROM"):
        monkeypatch.delenv(k, raising=False)
    yield tmp_path


def boot():
    """A fresh app (as a server start would). Each TestClient keeps its own cookies, like a separate browser."""
    from fastapi.testclient import TestClient
    for m in [m for m in sys.modules if m == "app" or m == "services" or m.startswith("services.")]:
        del sys.modules[m]
    import app as app_mod
    return app_mod.app, TestClient


def wait(c, jid, headers=None, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}", headers=headers or {}).json()
        if j.get("status") != "running":
            return j
        time.sleep(0.1)
    raise AssertionError("still running")


def run_rct(c, headers=None, notify=False):
    csv = c.get("/api/samples/rct").content
    r = c.post("/api/analyze", headers=headers or {}, files={"file": ("rct.csv", csv, "text/csv")},
               data={"claim": "treated increases score by 1.5", "roles": json.dumps({"randomized": True}),
                     "notify": "1" if notify else "0"})
    assert r.status_code == 200, r.text
    return wait(c, r.json()["job_id"], headers)


def test_guests_only_see_their_own_work(env):
    app, TC = boot()
    c = TC(app)
    assert c.post("/api/analyze", data={"claim": "x causes y"}).status_code == 401          # no id at all
    assert c.post("/api/analyze", data={"claim": "x"}, headers={"X-Guest-Id": "bad id!"}).status_code == 401
    j = run_rct(c, A)
    jid = j["id"]
    c.post(f"/api/jobs/{jid}/investigate", headers=A, json={"node": "treated"})
    for method, path in [("get", f"/api/jobs/{jid}"), ("get", f"/api/jobs/{jid}/investigations"),
                         ("get", f"/api/jobs/{jid}/influence?depth=1"), ("delete", f"/api/jobs/{jid}"),
                         ("post", f"/api/jobs/{jid}/investigate"), ("post", f"/api/jobs/{jid}/answer"),
                         ("delete", f"/api/jobs/{jid}/investigations/treated")]:
        kw = {"json": {"node": "treated"}} if method == "post" else {}
        r = getattr(c, method)(path, headers=B, **kw)
        assert r.status_code == 404, (method, path, r.status_code)   # never 403: ids can't be probed
    assert c.get("/api/datasets", headers=B).json() == []
    assert [d["id"] for d in c.get("/api/datasets", headers=A).json()] == [jid]
    assert c.get("/api/datasets").json() == []                       # nobody


def test_signup_login_logout(env):
    app, TC = boot()
    c = TC(app)
    j = run_rct(c, A)
    bad = [({"email": "not-an-email", "password": "password123"}, "valid email"),
           ({"email": "a@example.org", "password": "short"}, "at least 8")]
    for body, msg in bad:
        r = c.post("/api/auth/signup", headers=A, json=body)
        assert r.status_code == 400 and msg in r.json()["detail"]
    r = c.post("/api/auth/signup", headers=A, json={"email": "Jane.Doe@Example.org", "password": "password123"})
    assert r.status_code == 200 and r.json()["moved"] == 1 and r.json()["user"]["email"] == "jane.doe@example.org"
    cookie = r.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=lax" in cookie and "max-age=2592000" in cookie
    # signed in: the work follows the account, whatever guest id this browser sends
    assert c.get(f"/api/jobs/{j['id']}", headers=B).status_code == 200
    assert c.get("/api/auth/me").json()["user"]["email"] == "jane.doe@example.org"
    # duplicate (any case) rejected
    other = TC(app)
    r = other.post("/api/auth/signup", json={"email": "JANE.DOE@example.org", "password": "password123"})
    assert r.status_code == 400 and "already exists" in r.json()["detail"]
    # logout hides it again (the guest id no longer owns it)
    c.post("/api/auth/logout")
    assert c.get("/api/auth/me").json()["user"] is None
    assert c.get(f"/api/jobs/{j['id']}", headers=A).status_code == 404
    # wrong password; then the right one
    r = c.post("/api/auth/login", json={"email": "jane.doe@example.org", "password": "wrong-password"})
    assert r.status_code == 401 and r.json()["detail"] == "Wrong email or password."
    assert c.post("/api/auth/login", json={"email": "nobody@example.org", "password": "password123"}).status_code == 401
    r = c.post("/api/auth/login", json={"email": "jane.doe@example.org", "password": "password123"})
    assert r.status_code == 200 and r.json()["moved"] == 0
    assert c.get(f"/api/jobs/{j['id']}").status_code == 200


def test_first_account_adopts_work_made_before_accounts(env):
    app, TC = boot()
    from services import store
    old = store.create_analysis(owner="anon", mode="explore", title="old.csv", filename="old.csv", claim="", params={})
    c = TC(app)
    c.post("/api/auth/signup", json={"email": "first@example.org", "password": "password123"})
    assert c.get(f"/api/jobs/{old['id']}").status_code == 200
    c2 = TC(app)
    c2.post("/api/auth/signup", json={"email": "second@example.org", "password": "password123"})
    assert c2.get("/api/datasets").json() == []


def test_delete_removes_everything(env):
    app, TC = boot()
    c = TC(app)
    j = run_rct(c, A)
    jid = j["id"]
    c.post(f"/api/jobs/{jid}/investigate", headers=A, json={"node": "treated"})
    blobs = env / "blobs"
    assert {p.suffix for p in blobs.glob(f"{jid}.*")} == {".upload", ".parquet"}
    assert c.delete(f"/api/jobs/{jid}", headers=A).status_code == 200
    assert c.get(f"/api/jobs/{jid}", headers=A).status_code == 404
    assert c.get(f"/api/jobs/{jid}/investigations", headers=A).status_code == 404
    assert not list(blobs.glob(f"{jid}.*"))
    from services import store
    assert store.list_investigations(jid) == []
    time.sleep(0.6)  # the investigation that was running finishes into the void
    assert store.list_investigations(jid) == [] and store.get_analysis(jid) is None


def test_running_analysis_cannot_be_deleted_and_never_comes_back(env, monkeypatch):
    app, TC = boot()
    c = TC(app)
    from services import jobs, store
    real = jobs._pipeline

    def slow(a, steps):
        time.sleep(1.0)
        return real(a, steps)

    monkeypatch.setattr(jobs, "_pipeline", slow)
    csv = c.get("/api/samples/rct").content
    jid = c.post("/api/analyze", headers=A, files={"file": ("rct.csv", csv, "text/csv")}, data={"claim": ""}).json()["job_id"]
    r = c.delete(f"/api/jobs/{jid}", headers=A)
    assert r.status_code == 409 and "still running" in r.json()["detail"]
    store.delete_analysis(jid)  # e.g. deleted by another process while the thread runs
    time.sleep(2.5)
    assert store.get_analysis(jid) is None


class FakeHTTPError(urllib.error.HTTPError):
    def __init__(self, code, body):
        super().__init__("https://api.resend.com/emails", code, "Forbidden", {}, io.BytesIO(body.encode()))


def test_email_only_on_request_and_failures_are_kept(env, monkeypatch):
    monkeypatch.setenv("RESEND_API_KEY", "re_test_dummy")
    app, TC = boot()
    from services import mailer
    sent = []

    def fake_urlopen(req, timeout=0):
        payload = json.loads(req.data)
        sent.append(payload)
        assert req.headers["Authorization"] == "Bearer re_test_dummy"
        if payload["to"] == ["delivered@resend.dev"]:
            class R:
                status = 200
                def __enter__(self): return self
                def __exit__(self, *a): return False
            return R()
        raise FakeHTTPError(403, json.dumps({"message": "You can only send testing emails to your own email address"}))

    monkeypatch.setattr(mailer.urllib.request, "urlopen", fake_urlopen)
    c = TC(app)
    assert c.get("/api/auth/me").json()["email_notifications"] is True

    # a guest can't ask for email (no address)
    j = run_rct(c, A, notify=True)
    time.sleep(0.3)
    assert j.get("notify_email") is None and sent == []

    # signed in and asked: success to Resend's test inbox
    c.post("/api/auth/signup", headers=A, json={"email": "delivered@resend.dev", "password": "password123"})
    j = run_rct(c, notify=True)
    for _ in range(50):
        status = c.get(f"/api/jobs/{j['id']}").json()["mail_status"]
        if status:
            break
        time.sleep(0.1)
    assert status == {"ok": True, "to": "delivered@resend.dev", "error": None}
    mail = sent[-1]
    assert mail["subject"] == "Your analysis of rct.csv is ready"
    assert f"#/?job={j['id']}" in mail["html"] and "Verdict: Supported" in mail["text"]

    # not asked: nothing sent
    n = len(sent)
    run_rct(c, notify=False)
    time.sleep(0.3)
    assert len(sent) == n

    # Resend's sandbox refuses other addresses: the reason is stored for the UI, not swallowed
    c2 = TC(app)
    c2.post("/api/auth/signup", json={"email": "someone.else@example.org", "password": "password123"})
    j = run_rct(c2, notify=True)
    for _ in range(50):
        status = c2.get(f"/api/jobs/{j['id']}").json()["mail_status"]
        if status:
            break
        time.sleep(0.1)
    assert status["ok"] is False and "403" in status["error"] and "own email address" in status["error"]


def test_password_hashes():
    from services.auth import hash_password, verify_password
    h = hash_password("correct horse")
    assert h.startswith("scrypt$") and verify_password("correct horse", h)
    assert not verify_password("correct hors", h) and hash_password("correct horse") != h   # random salt
