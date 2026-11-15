"""Acceptance tests for the results tabs (rebuild-plan/06-results-tabs-ux.md §8), on this repo's engine.

The numbers are this engine's (LaLonde experimental: +1,583.5, 95% CI 254.5 to 2,912.4, grade C), not v1's.
"""
import re

import pytest

from conftest import ROOT, new_page, run_lalonde

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(),
                                reason="frontend not built (npm run build)")

AUDIT_TABS = ["Overview", "Data Usability Report", "Causal DAG", "Treatment Effect", "Literature", "Future Design",
              "Python Code", "Data", "Pipeline"]


def tab_names(page):
    tabs = page.get_by_role("tab")
    return [tabs.nth(i).inner_text() for i in range(tabs.count())]


def run_sample(page, base_url, sample, shift=False, roles=None):
    page.goto(base_url + "/#/")
    page.get_by_role("button", name=sample).click(modifiers=["Shift"] if shift else [])
    page.wait_for_timeout(400)
    if roles:
        page.get_by_text("Study design and roles").click()
        for label, value in roles.items():
            page.get_by_label(label).select_option(value)
    page.get_by_role("button", name="Run analysis").click()
    page.get_by_role("tab", name="Overview", exact=True).wait_for(timeout=120_000)


def visit_every_tab(page, label):
    for n in tab_names(page):
        page.get_by_role("tab", name=n, exact=True).click()
        pane = page.locator("#result-pane")
        assert pane.inner_text().strip(), f"{label}: tab {n} is empty"
        assert pane.locator(".card.err").count() == 0, f"{label}: tab {n} failed to render"
        over = page.evaluate("document.scrollingElement.scrollWidth - window.innerWidth")
        assert over <= 0, f"{label}: tab {n} scrolls the page sideways by {over}px"


def test_audit_tabs_overview_headline_and_every_tab(browser, base_url):
    page = new_page(browser, 1440)
    run_lalonde(page, base_url)
    assert tab_names(page) == AUDIT_TABS
    pane = page.locator("#result-pane")
    # the answer at a glance, without opening another tab
    hero = pane.locator(".hero-estimate")
    assert "+1,583.5" in hero.inner_text() and "95% CI [254.5, 2,912.4]" in hero.inner_text()
    assert pane.locator(".hero-usability .grade-letter").inner_text() == "C"
    assert pane.locator(".mini-pills .status").count() >= 4
    assert pane.locator(".mini-pills .status.pass").first.inner_text() == "PASS"      # v1 pills: PASS / WARN / FAIL
    assert pane.locator(".mini-pills .status.assumed").count() >= 1                 # assumed is its own (never green) pill
    assert "Claim supported" in hero.inner_text()                                    # the verdict sits under the number
    assert pane.get_by_role("heading", name="Assumed, not tested").is_visible()      # kept on Overview, compact
    assert pane.get_by_role("heading", name="Parsed claim (rule-based)").is_visible()
    assert pane.locator("h2").count() == 0                      # card titles are h3 inside panes
    visit_every_tab(page, "audit 1440")
    # Usability Report: checks two per row, "Numbers" collapsed, no repeated "Assumed" panel
    page.get_by_role("tab", name="Data Usability Report", exact=True).click()
    assert pane.locator(".checks-grid .check-card").count() >= 5
    assert pane.locator("details.metrics[open]").count() == 0
    assert pane.locator("svg[aria-label^='Standardized differences']").count() == 1   # Love plot (balance)
    assert pane.locator("svg[aria-label^='Propensity score by group']").count() == 1  # mirrored histogram (overlap)
    # Causal DAG: pill nodes, no edge labels
    page.get_by_role("tab", name="Causal DAG", exact=True).click()
    assert pane.locator("svg[aria-label='Assumed causal diagram'] .dag-node").count() == 12
    # Treatment Effect: the forest plot with square markers
    page.get_by_role("tab", name="Treatment Effect", exact=True).click()
    assert pane.locator("svg[aria-label='Estimates with 95% intervals'] rect").count() == 2
    # Future Design: v1 recommendation rows with a priority
    page.get_by_role("tab", name="Future Design", exact=True).click()
    assert pane.locator(".rec .prio").count() >= 2
    assert pane.locator(".assumed-panel").count() == 0
    # Literature: every linked study opens in a new tab with an http(s) URL
    page.get_by_role("tab", name="Literature", exact=True).click()
    links = pane.locator("a.study-title")
    assert links.count() >= 3
    for i in range(links.count()):
        assert re.match(r"https?://", links.nth(i).get_attribute("href"))
        assert links.nth(i).get_attribute("target") == "_blank"
    # Data: the column table is visible (not collapsed) with % missing
    page.get_by_role("tab", name="Data", exact=True).click()
    assert pane.locator("table.tbl").is_visible() and "0.0%" in pane.locator("table.tbl").inner_text()
    assert pane.get_by_role("heading", name="445 rows × 12 columns").is_visible()
    # Python Code: one card per step, each with Copy; Download .py for all of them
    page.get_by_role("tab", name="Python Code", exact=True).click()
    for step in ("load", "claim", "preprocess", "design", "balance", "checks", "estimate", "sensitivity", "verdict"):
        assert pane.get_by_role("heading", name=step, exact=True).is_visible()
    assert pane.get_by_role("button", name="Copy").count() >= 9 and pane.get_by_role("button", name="Download .py").is_visible()
    # Pipeline: the checks sit in one dashed box, each with its pill
    page.get_by_role("tab", name="Pipeline", exact=True).click()
    assert pane.locator(".par .par-grid .status").count() >= 3
    assert page.errors == [], page.errors


def test_keyboard_tabs_wrap(browser, base_url):
    page = new_page(browser, 1440)
    run_lalonde(page, base_url)
    page.get_by_role("tab", name="Overview", exact=True).focus()
    for _ in range(len(AUDIT_TABS) - 1):
        page.keyboard.press("ArrowRight")
    assert page.evaluate("document.activeElement.textContent") == "Pipeline"
    page.keyboard.press("ArrowRight")                            # wraps
    assert page.evaluate("document.activeElement.textContent") == "Overview"
    page.keyboard.press("End")
    assert page.get_by_role("tab", name="Pipeline").get_attribute("aria-selected") == "true"
    page.keyboard.press("Home")
    assert page.get_by_role("tab", name="Overview", exact=True).get_attribute("aria-selected") == "true"


def test_explore_and_claim_only_tab_sets(browser, base_url):
    page = new_page(browser, 1440)
    run_sample(page, base_url, "Simulated: causal chain (explore)")
    assert tab_names(page) == ["Overview", "Causal Discovery", "Python Code", "Data", "Pipeline"]
    visit_every_tab(page, "explore")
    page.get_by_role("tab", name="Causal Discovery", exact=True).click()
    pane = page.locator("#result-pane")
    assert pane.locator("svg[aria-label='Discovered causal graph'] .dag-node").count() == 5
    assert pane.get_by_role("heading", name="Edges").is_visible()
    assert pane.get_by_role("heading", name="Sanity checks on discovered pairs").is_visible()

    page.goto(base_url + "/#/")
    page.get_by_role("button", name="Clear").click()
    page.fill("#claim", "Job training raises earnings")
    page.get_by_role("button", name="Run analysis").click()
    page.get_by_role("tab", name="Overview", exact=True).wait_for(timeout=60_000)
    assert tab_names(page) == ["Overview", "Literature", "Future Design", "Pipeline"]
    visit_every_tab(page, "claim only")
    assert page.errors == [], page.errors


def test_refused_estimate_keeps_the_treatment_effect_tab(browser, base_url):
    page = new_page(browser, 1440)
    # Card & Krueger with the state indicator as the treatment: this engine refuses a DiD number
    run_sample(page, base_url, "Real: Card & Krueger minimum wage", roles={"Treatment (the cause)": "new_jersey"})
    assert "Treatment Effect" in tab_names(page)
    pane = page.locator("#result-pane")
    assert "No credible estimate" in pane.inner_text()             # Overview says so too
    page.get_by_role("tab", name="Treatment Effect", exact=True).click()
    assert pane.get_by_role("heading", name="Not estimated").is_visible()
    assert "We refused to give a number." in pane.inner_text()
    assert "No treated cohort has a usable comparison group" in pane.inner_text()
    assert pane.get_by_role("heading", name="Critique of the data collection").is_visible()
    assert page.errors == [], page.errors


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_every_tab_fits_a_phone(browser, base_url, scheme):
    page = new_page(browser, 390, scheme)
    run_lalonde(page, base_url)
    visit_every_tab(page, f"390 {scheme}")
    # the last tab, selected, sits fully inside the scrolling strip; the page itself did not scroll sideways
    inside = page.evaluate("""() => { const l = document.querySelector('.tabs').getBoundingClientRect();
        const t = document.querySelector('.tab[aria-selected="true"]').getBoundingClientRect();
        return t.left >= l.left - 1 && t.right <= l.right + 1; }""")
    assert inside, "the selected tab is half hidden in the tab strip"
    assert page.errors == [], page.errors


def test_new_result_resets_to_overview_and_scrolls_into_view(browser, base_url):
    page = new_page(browser, 1440)
    first = run_lalonde(page, base_url)
    # the run finished in this browser tab: the results came into view
    page.wait_for_timeout(800)
    top = page.evaluate("document.querySelector('.results').getBoundingClientRect().top")
    assert 0 <= top < 200, f"results not scrolled into view (top {top})"
    page.get_by_role("tab", name="Data", exact=True).click()
    second = run_lalonde(page, base_url)
    assert page.get_by_role("tab", name="Overview", exact=True).get_attribute("aria-selected") == "true"
    page.get_by_role("tab", name="Data", exact=True).click()
    # opening another saved analysis from the sidebar: Overview again, and no scroll
    page.wait_for_timeout(1200)                                   # let the finished run's smooth scroll end
    page.evaluate("window.scrollTo(0, 0)")
    page.locator(f".side-item[href='#/?job={first}']").click()
    page.get_by_role("tab", name="Overview", exact=True).wait_for()
    assert page.get_by_role("tab", name="Overview", exact=True).get_attribute("aria-selected") == "true"
    page.wait_for_timeout(800)
    assert page.evaluate("window.scrollY") == 0
    assert second != first
    assert page.errors == [], page.errors
