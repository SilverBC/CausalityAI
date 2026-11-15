"""'Your analysis is ready' email, for signed-in users who ask for it.

Resend's HTTP API (RESEND_API_KEY, MAIL_FROM) and, when SMTP_HOST is set, an SMTP fallback (SMTP_PORT, SMTP_USER,
SMTP_PASSWORD, SMTP_FROM; e.g. Gmail with an app password). Settings come from the gitignored .env and are never
logged. Known trap: without a verified domain, Resend only delivers mail from onboarding@resend.dev to the address
that owns the Resend account (HTTP 403 "You can only send testing emails to your own email address"). The outcome
of every send is stored on the analysis (`mail_status`) and shown in the app, never swallowed.
"""
from __future__ import annotations

import html
import json
import os
import smtplib
import threading
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Optional

from . import store

RESEND_URL = "https://api.resend.com/emails"


def available() -> bool:
    return bool(os.environ.get("RESEND_API_KEY") or os.environ.get("SMTP_HOST"))


def _sender() -> str:
    return os.environ.get("MAIL_FROM") or "Bluff-or-Buff <onboarding@resend.dev>"


def send_resend(to: str, subject: str, html_body: str, text_body: str) -> Optional[str]:
    """None on success, else the reason in plain words."""
    payload = {"from": _sender(), "to": [to], "subject": subject, "html": html_body, "text": text_body}
    req = urllib.request.Request(RESEND_URL, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {os.environ['RESEND_API_KEY']}",
                                          "Content-Type": "application/json", "User-Agent": "causal-audit/2.0"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return None if 200 <= r.status < 300 else f"Resend answered HTTP {r.status}"
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:300]
        try:
            body = json.loads(body).get("message", body)
        except (json.JSONDecodeError, AttributeError):
            pass
        return f"Resend answered HTTP {e.code}: {body}"
    except Exception as e:  # noqa: BLE001
        return f"Could not reach Resend ({type(e).__name__})."


def send_smtp(to: str, subject: str, html_body: str, text_body: str) -> Optional[str]:
    msg = EmailMessage()
    msg["From"] = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER", "")
    msg["To"], msg["Subject"] = to, subject
    msg.set_content(text_body)
    msg.add_alternative(html_body, subtype="html")
    try:
        with smtplib.SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587")), timeout=20) as s:
            s.starttls()
            if os.environ.get("SMTP_USER"):
                s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            s.send_message(msg)
        return None
    except Exception as e:  # noqa: BLE001  (the message never includes the password)
        return f"SMTP: {type(e).__name__}: {str(e)[:200]}"


def send(to: str, subject: str, html_body: str, text_body: str) -> Optional[str]:
    """Resend first, then SMTP. None on success, else every reason."""
    errors = []
    if os.environ.get("RESEND_API_KEY"):
        err = send_resend(to, subject, html_body, text_body)
        if err is None:
            return None
        errors.append(err)
    if os.environ.get("SMTP_HOST"):
        err = send_smtp(to, subject, html_body, text_body)
        if err is None:
            return None
        errors.append(err)
    return "; ".join(errors) or "Email is not set up on this server."


def compose(a: dict) -> tuple[str, str, str]:
    """(subject, html, text) for a finished or failed analysis."""
    link = f"{(a.get('link_base') or 'http://localhost:8000/').rstrip('/')}/#/?job={a['id']}"
    res = a.get("result") or {}
    rep = res.get("report") or {}
    if a["status"] == "error":
        subject = f"Your analysis of {a['title']} failed"
        lines = [f"The analysis of {a['title']} did not finish: {a.get('error')}"]
    else:
        subject = f"Your analysis of {a['title']} is ready"
        lines = []
        d, v = rep.get("design") or {}, rep.get("verdict") or {}
        if d.get("grade"):
            lines.append(f"Design: {d.get('kind')} · grade {d['grade']}")
        if v.get("label"):
            lines.append(f"Verdict: {v['label']}. {v.get('sentence', '')}")
        summary = (res.get("summary") or {}).get("text")
        if summary:
            lines.append(summary)
    text = "\n\n".join(lines + [f"Open it: {link}"])
    body = ("<div style='font-family:system-ui,sans-serif;max-width:560px;line-height:1.5;color:#0f172a'>"
            f"<h2 style='margin:0 0 12px'>{html.escape(subject)}</h2>"
            + "".join(f"<p>{html.escape(x)}</p>" for x in lines)
            + f"<p><a href='{html.escape(link)}' style='display:inline-block;background:#4f46e5;color:#fff;"
              "padding:10px 16px;border-radius:8px;text-decoration:none;font-weight:600'>Open the analysis</a></p>"
              "<p style='color:#64748b;font-size:13px'>You asked Bluff-or-Buff to email you when this analysis "
              "finished.</p></div>")
    return subject, body, text


def notify(analysis_id: str) -> Optional[threading.Thread]:
    """If the owner asked for an email, send it in the background and store the outcome on the analysis."""
    a = store.get_analysis(analysis_id)
    if not a or not a.get("notify_email"):
        return None

    def work():
        subject, body, text = compose(a)
        err = send(a["notify_email"], subject, body, text)
        store.update_analysis(analysis_id, mail_status={"ok": err is None, "to": a["notify_email"], "error": err})
        if err:
            print(f"[mail] analysis {analysis_id}: {err}", flush=True)

    t = threading.Thread(target=work, daemon=True, name=f"mail-{analysis_id}")
    t.start()
    return t
