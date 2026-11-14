"""Browser acceptance tests for the desktop web app (Playwright, driving the installed Chrome).

Needs a built frontend (cd frontend && npm run build) and `pip install -r requirements-dev.txt`. Runs the real
server on a spare port with a throwaway data dir and the fake language model. From the project root:
    python -m pytest frontend/e2e -q
Screenshots land in frontend/e2e/screens/ (gitignored) for a visual check.
"""
from pathlib import Path

import pytest

from conftest import ROOT, SHOTS, assert_layout, new_page, pw, run_lalonde  # noqa: F401

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(),
                                reason="frontend not built (npm run build)")


def test_run_lalonde_every_tab_sidebar_and_reload(browser, base_url):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, 1440)
    job = run_lalonde(page, base_url)

    # the verdict and estimate from this repo's core
    assert page.locator(".rh .verdict-label").first.inner_text().strip() == "Supported"
    # every tab renders, none shows the error card
    tabs = page.get_by_role("tab")
    names = [tabs.nth(i).inner_text() for i in range(tabs.count())]
    assert names[0] == "Overview" and names[-1] == "Pipeline"
    for want in ("Data Usability Report", "Causal DAG", "Treatment Effect", "Future Design", "Python Code", "Data"):
        assert want in names
    for n in names:
        page.get_by_role("tab", name=n, exact=True).click()
        pane = page.locator("#result-pane")
        assert pane.inner_text().strip(), f"tab {n} is empty"
        assert pane.locator(".card.err").count() == 0, f"tab {n} failed to render"
    page.get_by_role("tab", name="Treatment Effect", exact=True).click()
    assert page.locator("#result-pane svg.chart").count() >= 1

    # the graph is drawn and settles; screenshot it for a visual check of the arrowheads
    canvas = page.locator(".graph-canvas canvas").first
    canvas.wait_for()
    page.wait_for_timeout(3500)
    page.locator(".graph-card").screenshot(path=str(SHOTS / "graph-light.png"))
    painted = page.evaluate("""() => { const c = document.querySelector('.graph-canvas canvas');
        const d = c.getContext('2d').getImageData(0, 0, c.width, c.height).data; let n = 0;
        for (let i = 3; i < d.length; i += 4) if (d[i] > 0) n++; return n; }""")
    assert painted > 2000, "the graph canvas is empty"
    assert page.locator(".rank-row").count() >= 3

    # sidebar lists the run, and a reload restores it from the URL
    item = page.locator(f".side-item[href='#/?job={job}']")
    assert item.count() == 1 and "lalonde_experimental.csv" in item.inner_text()
    page.reload()
    page.get_by_role("tab", name="Overview", exact=True).wait_for(timeout=30_000)
    assert page.locator(".rh .verdict-label").first.inner_text().strip() == "Supported"
    assert page.errors == [], page.errors

    # the standalone report export renders on its own
    page.get_by_role("tab", name="Overview", exact=True).click()
    with page.expect_download() as dl:
        page.get_by_role("button", name="Download report").click()
    path = SHOTS / dl.value.suggested_filename
    assert path.suffix == ".html"
    dl.value.save_as(path)
    offline = browser.new_page()
    offline.goto(path.as_uri())
    assert "Supported" in offline.locator("#results").inner_text()
    offline.close()


@pytest.mark.parametrize("width", [1440, 390])
@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_layout_screens(browser, base_url, width, scheme):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, width, scheme)
    run_lalonde(page, base_url)
    page.wait_for_timeout(1500)
    assert_layout(page, f"{width} {scheme} results")
    page.screenshot(path=str(SHOTS / f"analysis-{width}-{scheme}.png"), full_page=False)
    # after Run, the page scrolled down to the output
    assert page.evaluate("window.scrollY") > 200, "the page did not scroll to the results"
    page.locator(".results").screenshot(path=str(SHOTS / f"overview-{width}-{scheme}.png"))
    for tab in ("Data Usability Report", "Treatment Effect", "Data"):
        page.get_by_role("tab", name=tab, exact=True).click()
        assert_layout(page, f"{width} {scheme} {tab}")
        if tab == "Treatment Effect":
            page.locator(".results").screenshot(path=str(SHOTS / f"effect-{width}-{scheme}.png"))
    if width < 1100:  # the sidebar overlays on narrow screens
        page.get_by_role("button", name="Show your analyses").click()
        assert page.locator(".sidebar").is_visible()
        assert_layout(page, f"{width} {scheme} sidebar open")
        page.screenshot(path=str(SHOTS / f"sidebar-{width}-{scheme}.png"))
    assert page.errors == [], page.errors


def test_explore_mode_and_needs_input(browser, base_url):
    page = new_page(browser, 1440)
    page.goto(base_url + "/")
    page.get_by_role("button", name="Simulated: causal chain (explore)").click()
    page.wait_for_timeout(300)
    page.get_by_role("button", name="Run analysis").click()
    page.get_by_role("tab", name="Causal Discovery", exact=True).wait_for(timeout=120_000)
    page.get_by_role("tab", name="Causal Discovery", exact=True).click()
    assert page.locator("#result-pane svg.chart.dag").count() == 1

    # a claim whose words match no column: the analysis asks instead of guessing
    page.get_by_role("button", name="Real: LaLonde NSW experiment").click(modifiers=["Shift"])  # data only
    page.wait_for_timeout(300)
    page.fill("#claim", "Job training increases 1978 earnings by $1,800")
    page.get_by_role("button", name="Run analysis").click()
    page.get_by_text("We need your answer before we can go on").wait_for(timeout=60_000)
    selects = page.locator(".needs-input select")
    for i in range(selects.count()):
        q = page.locator(".needs-input .ni-text").nth(i).inner_text()
        if "treatment" in q:
            selects.nth(i).select_option("treat")
        elif "outcome" in q:
            selects.nth(i).select_option("earnings_1978")
    page.get_by_role("button", name="Use these answers and analyze again").click()
    page.get_by_role("tab", name="Treatment Effect", exact=True).wait_for(timeout=120_000)
    assert page.errors == [], page.errors


def test_theme_toggle_circle_reveal_and_persistence(browser, base_url):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, 1440, "light")
    page.goto(base_url + "/")
    page.wait_for_timeout(500)
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(246, 247, 251)"
    page.get_by_role("button", name="Switch to dark mode").click()
    page.wait_for_timeout(220)  # mid-transition: the dark circle is growing from the toggle
    page.screenshot(path=str(SHOTS / "theme-mid-transition.png"))
    page.wait_for_timeout(700)
    assert page.evaluate("document.documentElement.dataset.theme") == "dark"
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(11, 16, 32)"
    page.reload()  # the pick is stored and applied before the first paint
    assert page.evaluate("document.documentElement.dataset.theme") == "dark"
    page.get_by_role("button", name="Switch to light mode").wait_for()
    assert page.errors == [], page.errors
