"""Shared fixtures for the browser tests: one real server (fake language model, throwaway data dir, OpenAlex off)
and one Chrome per test session."""
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pw = pytest.importorskip("playwright.sync_api")

ROOT = Path(__file__).resolve().parents[2]
SHOTS = Path(__file__).parent / "screens"
FAKE_SEARCH_SECONDS = "2.5"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def base_url(tmp_path_factory):
    port = _free_port()
    env = {**os.environ, "CAUSAL_DATA_DIR": str(tmp_path_factory.mktemp("data")), "CAUSAL_LLM": "fake",
           "CAUSAL_FAKE_DELAY": FAKE_SEARCH_SECONDS, "CAUSAL_OPENALEX": "off"}
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--app-dir", "backend", "--port", str(port)],
                           cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    for _ in range(80):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.25)
    yield url
    srv.terminate()
    srv.wait(10)


@pytest.fixture(scope="session")
def browser():
    with pw.sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        yield b
        b.close()


def new_page(browser, width, scheme="light"):
    ctx = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme, accept_downloads=True)
    page = ctx.new_page()
    page.errors = []
    page.on("console", lambda m: m.type == "error" and page.errors.append(m.text))
    page.on("pageerror", lambda e: page.errors.append(str(e)))
    return page


def assert_layout(page, label):
    """No horizontal page scroll; the header is one 58 px row."""
    over = page.evaluate("document.scrollingElement.scrollWidth - window.innerWidth")
    assert over <= 0, f"{label}: page scrolls sideways by {over}px"
    h = page.evaluate("document.querySelector('.topbar').getBoundingClientRect().height")
    assert abs(h - 58) < 1.5, f"{label}: header is {h}px tall"
    tops = page.evaluate("[...document.querySelectorAll('.topbar-left > *, .topbar-right > *')]"
                         ".filter(e => e.offsetParent).map(e => Math.round(e.getBoundingClientRect().top + e.getBoundingClientRect().height/2))")
    assert max(tops) - min(tops) <= 6, f"{label}: header items not on one row {tops}"


def run_lalonde(page, base_url):
    page.goto(base_url + "/")
    page.get_by_role("button", name="Real: LaLonde NSW experiment").click()
    page.wait_for_function("document.querySelector('#claim').value.length > 0")
    page.get_by_role("button", name="Run analysis").click()
    page.wait_for_url("**/#/?job=*")
    page.get_by_role("tab", name="Overview", exact=True).wait_for(timeout=120_000)
    return page.url.split("job=")[1]


