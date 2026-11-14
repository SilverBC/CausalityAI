"""Browser tests for the Investigation page ("sleuth mode"), with the fake language model (2.5 s searches)."""
import json
import re

import pytest

from conftest import ROOT, SHOTS, assert_layout, confirm_dialog, new_page, run_lalonde

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(),
                                reason="frontend not built (npm run build)")


def api(page, path):
    """GET through the page, so the request carries this browser's guest id (the API answers 404 to anyone else)."""
    return page.evaluate(f"fetch({json.dumps(path)}).then(r => r.json())")


def open_investigation(page, base_url):
    job = run_lalonde(page, base_url)
    page.get_by_role("link", name="Investigate further →").click()
    page.wait_for_url(f"**/#/investigate/{job}")
    page.get_by_role("heading", name=re.compile("^Investigation")).wait_for()
    page.locator(".graph-canvas canvas").wait_for()
    return job


def several_row(page, name):
    return page.locator(".multi-list li").filter(has=page.locator(".multi-name", has_text=re.compile(rf"^{re.escape(name)} ·")))


def test_investigate_one_variable(browser, base_url):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, 1440)
    job = open_investigation(page, base_url)
    assert "not from the dataset you provided" in page.locator(".disclaimer").inner_text()
    switch = page.get_by_role("switch")
    assert switch.is_disabled() and "None yet" in page.locator(".graph-toolbar").inner_text()

    page.goto(f"{base_url}/#/investigate/{job}?node=treat")  # a deep link selects the node
    page.get_by_role("button", name="Investigate causes of “treat”").click()
    page.get_by_text("Searching the literature").wait_for()
    page.get_by_text("Investigated in").wait_for(timeout=30_000)

    panel = page.locator("#inv-panel")
    factors = panel.locator(".factor")
    assert factors.count() == 3                                   # the unsourced one was dropped
    assert panel.locator(".factor .pill.hypo").count() == 2 and panel.locator(".factor .pill.in-data").count() == 1
    assert panel.locator(".reliability").count() == 3
    # the reliability breakdown dialog
    panel.locator(".rel-info").first.click()
    dlg = page.locator("dialog.rel-dialog[open]")
    assert "source score = 0.6 × citation score" in dlg.inner_text()
    page.keyboard.press("Escape")
    assert page.locator("dialog.rel-dialog[open]").count() == 0

    # hypothetical variables are on the graph; the switch now has a count and hides/shows them
    assert "2 from the literature, not in your data" in page.locator(".graph-toolbar").inner_text()
    page.wait_for_timeout(2500)
    page.locator(".inv-layout").screenshot(path=str(SHOTS / "investigation-1440-light.png"))
    page.get_by_text("Show hypothetical variables").click()      # what a user clicks: the label
    assert not switch.is_checked() and "2 hidden" in page.locator(".graph-toolbar").inner_text()
    page.get_by_text("Show hypothetical variables").click()
    assert switch.is_checked()

    # selecting a hypothetical factor from the list opens its panel with its reliability and source notes
    panel.locator(".factor .link-btn", has_text="Pre-program earnings dip").click()
    assert "hypothetical · not in your data" in panel.inner_text() and "Suggested while investigating" in panel.inner_text()

    # rename is display only: shown in the panel, kept in this browser, never sent to the server
    page.get_by_role("button", name="Rename Pre-program earnings dip").click()
    page.get_by_label("Display name for Pre-program earnings dip").fill("Renamed-by-test 7f3a")
    page.get_by_role("button", name="Save").click()
    assert "Original name: Pre-program earnings dip" in panel.inner_text()
    stored = page.evaluate(f"localStorage.getItem('causal-audit:names:{job}')")
    assert "Renamed-by-test 7f3a" in stored
    invs = api(page, f"/api/jobs/{job}/investigations")
    assert "Renamed-by-test" not in json.dumps(invs)

    # the sidebar lists the investigated variable as a deep link
    side = page.locator(".side-invs a", has_text="treat")
    assert side.count() == 1
    # delete it (temporary native confirm: accept)
    page.goto(f"{base_url}/#/investigate/{job}?node=treat")
    page.get_by_text("Investigated in").wait_for()
    page.get_by_role("button", name="Delete investigation").click()
    confirm_dialog(page).get_by_role("button", name="Delete investigation").click()
    page.get_by_role("button", name="Investigate causes of “treat”").wait_for()
    assert api(page, f"/api/jobs/{job}/investigations") == []
    assert page.errors == [], page.errors


def test_investigate_several_and_the_queue(browser, base_url):
    page = new_page(browser, 1440)
    job = open_investigation(page, base_url)

    # select two -> both run at once
    several_row(page, "treat").locator("input").check()
    several_row(page, "age").locator("input").check()
    page.get_by_role("button", name="Investigate selected (2)").click()
    page.locator(".multi-status").wait_for()
    assert "2 running" in page.locator(".multi-status").inner_text()

    # investigate all -> the heavy-use warning first (warn tone); cancelling it queues nothing
    page.get_by_role("button", name=re.compile(r"^Investigate all \(\d+\)$")).click()
    dlg = confirm_dialog(page)
    text = dlg.inner_text()
    assert "uses a lot of your Claude usage" in text and re.search(r"roughly \d+–\d+ minutes", text)
    assert "warn" in dlg.get_attribute("class") and dlg.locator(".btn.primary").count() == 1
    dlg.get_by_role("button", name="Cancel").click()
    assert len(api(page, f"/api/jobs/{job}/investigations")) == 2

    page.get_by_role("button", name=re.compile(r"^Investigate all \(\d+\)$")).click()
    confirm_dialog(page).get_by_role("button", name=re.compile("^Queue ")).click()
    page.wait_for_function("document.querySelector('.multi-status')?.textContent.includes('queued')")
    items = api(page, f"/api/jobs/{job}/investigations")
    assert sum(i["status"] == "running" for i in items) == 3
    queued = [i for i in items if i["status"] == "queued"]
    assert len(queued) >= 3 and [i["position"] for i in queued] == list(range(1, len(queued) + 1))
    assert "#1 in queue" in page.locator(".multi-list").inner_text()

    # cancel one queued, then all queued
    first = queued[0]["node"]
    several_row(page, first).get_by_role("button", name=re.compile("^Cancel the queued")).click()
    page.wait_for_function(f"!document.querySelector('.multi-list').textContent.includes('#{len(queued)} in queue')")
    page.get_by_role("button", name="Cancel queued").click()
    page.wait_for_function("!document.querySelector('.multi-list').textContent.includes('in queue')")
    assert all(i["status"] != "queued" for i in api(page, f"/api/jobs/{job}/investigations"))

    # findings appear as each finishes; the sidebar shows the counts
    page.wait_for_function("document.querySelectorAll('.multi-list .link-btn.ok').length >= 3", timeout=30_000)
    page.wait_for_function("!document.querySelector('.multi-status')", timeout=30_000)
    page.wait_for_function("document.querySelector('.side-list li.active .side-sub')?.textContent.includes('investigated')",
                           timeout=10_000)  # the sidebar refreshes every few seconds
    assert page.errors == [], page.errors


def test_depth_slider_and_restart_safe_reload(browser, base_url):
    page = new_page(browser, 1440)
    job = open_investigation(page, base_url)
    slider = page.locator("#depth")
    default = int(slider.input_value())
    top = int(slider.get_attribute("max"))
    assert top > default
    slider.fill(str(top))                      # data search upstream
    page.wait_for_function("document.querySelector('#depth-status').textContent.match(/Found|No further|stops at/)", timeout=30_000)
    slider.fill(str(default))                  # back to the analysis's own graph
    page.wait_for_function("document.querySelector('#depth-status').textContent.includes('The graph from the analysis')",
                           timeout=30_000)
    # reloading the page keeps everything (it comes from the server, not from memory)
    page.reload()
    page.locator(".graph-canvas canvas").wait_for()
    assert page.locator("#depth").input_value() == str(default)
    assert page.errors == [], page.errors


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_investigation_layout(browser, base_url, width, scheme):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, width, scheme)
    job = open_investigation(page, base_url)
    page.goto(f"{base_url}/#/investigate/{job}?node=treat")
    page.get_by_role("button", name="Investigate causes of “treat”").click()
    page.get_by_text("Investigated in").wait_for(timeout=30_000)
    page.wait_for_timeout(1500)
    assert_layout(page, f"investigation {width} {scheme}")
    page.screenshot(path=str(SHOTS / f"investigation-{width}-{scheme}.png"), full_page=True)
    assert page.errors == [], page.errors
