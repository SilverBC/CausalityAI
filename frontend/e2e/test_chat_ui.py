"""Browser tests for the Investigator (desktop bubble + chat) and the phone app, with the fake language model."""
import re

import pytest

from conftest import ROOT, SHOTS, assert_layout, confirm_dialog, new_page, run_lalonde

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(),
                                reason="frontend not built (npm run build)")


def no_sideways_scroll(page, label):
    over = page.evaluate("document.scrollingElement.scrollWidth - window.innerWidth")
    assert over <= 0, f"{label}: page scrolls sideways by {over}px"


def test_bubble_teaser_timing_and_session_rules(browser, base_url):
    page = new_page(browser, 1440)
    run_lalonde(page, base_url)
    bubble = page.get_by_role("button", name="Ask the Investigator about this analysis")
    box = bubble.bounding_box()
    assert abs(box["width"] - 66) < 1 and abs(box["height"] - 66) < 1
    assert bubble.get_attribute("title") == "Ask the Investigator"
    teaser = page.locator(".ab-teaser")
    teaser.wait_for(timeout=4000)                                   # first line after 2.5 s
    first = teaser.locator(".ab-teaser-text").inner_text()
    teaser.wait_for(state="detached", timeout=7000)                 # visible for 5.5 s
    teaser.wait_for(timeout=13_000)                                 # back after an 11 s gap, with the next line
    assert teaser.locator(".ab-teaser-text").inner_text() != first
    page.get_by_role("button", name="Dismiss").click()              # dismissing turns it off for the session
    page.reload()
    page.get_by_role("tab", name="Overview", exact=True).wait_for(timeout=30_000)
    page.wait_for_timeout(4000)
    assert page.locator(".ab-teaser").count() == 0
    # a different saved analysis: one "welcome back" line even though the teaser is off
    other = run_lalonde(page, base_url)
    page.locator(".ab-teaser").wait_for(timeout=5000)
    assert page.locator(".ab-teaser-text").inner_text() in (
        "Back on the case?", "Picking up where we left off?", "Fresh eyes on this one?", "Shall we reopen the case?")
    # moving to the same analysis's Investigation page: no new line
    page.locator(".ab-teaser").wait_for(state="detached", timeout=7000)
    page.goto(f"{base_url}/#/investigate/{other}")
    page.wait_for_timeout(3500)
    assert page.locator(".ab-teaser").count() == 0
    assert page.get_by_role("button", name="Ask the Investigator about this analysis").is_visible()
    assert page.errors == [], page.errors


def test_chat_quotes_real_numbers_and_starts_an_investigation(browser, base_url):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, 1440)
    job = run_lalonde(page, base_url)
    page.get_by_role("button", name="Ask the Investigator about this analysis").click()
    panel = page.get_by_role("dialog", name="Chat with the Investigator")
    panel.locator(".ab-title", has_text="lalonde_experimental.csv").wait_for()
    assert "Answers use the real numbers from your data" in panel.inner_text()
    panel.get_by_role("button", name="Does the data support the claim?").click()
    card = panel.locator(".m-card", has_text="Estimated effect")
    card.wait_for(timeout=15_000)
    assert "1583" in panel.locator(".m-msg.assistant .m-bubble").last.inner_text()
    assert re.search(r"95% confidence interval 254\.?\d* to 2,?912", card.inner_text())
    assert "claim: supported" in card.inner_text().lower()          # the chip is styled in capitals
    # the chat is now saved and named after its first message
    assert panel.locator(".picker-value").inner_text() == "Does the data support the claim?"

    # asking to investigate starts a literature search; its card updates live and links into the graph
    panel.get_by_label("Message").fill("Investigate what else could drive the treatment")
    panel.get_by_label("Message").press("Enter")
    inv = panel.locator(".m-card", has_text="Investigation · treat")
    inv.wait_for(timeout=15_000)
    inv.locator("text=/Found \\d+ possible causes/").wait_for(timeout=30_000)
    assert inv.get_attribute("href") == f"#/investigate/{job}?node=treat"
    page.screenshot(path=str(SHOTS / "investigator-1440.png"))

    # chat picker, keyboard only: opens, moves, picks "New chat"; Esc closes only the list, then the panel
    picker = panel.locator(".picker-btn")
    picker.focus()
    page.keyboard.press("ArrowDown")
    assert panel.get_by_role("listbox").is_visible()
    page.keyboard.press("Home")
    page.keyboard.press("Escape")
    assert panel.get_by_role("listbox").count() == 0 and panel.is_visible()
    page.keyboard.press("ArrowDown")
    page.keyboard.press("Home")
    page.keyboard.press("Enter")
    assert panel.locator(".picker-value").inner_text() == "New chat"
    page.keyboard.press("Escape")
    assert page.get_by_role("dialog", name="Chat with the Investigator").count() == 0
    # the last chat of this analysis is remembered
    page.get_by_role("button", name="Ask the Investigator about this analysis").click()
    assert page.locator(".picker-value").inner_text() in ("New chat", "Does the data support the claim?")
    # the card's link opens the node on the Investigation page
    page.goto(f"{base_url}/#/investigate/{job}?node=treat")
    page.get_by_text("Investigated in").wait_for(timeout=30_000)
    assert page.errors == [], page.errors


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_phone_app(browser, base_url, scheme):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, 390, scheme)
    page.goto(base_url + "/")
    page.wait_for_url("**/#/m")                                     # phones land on the phone app
    page.get_by_role("button", name="Upload a dataset").wait_for()
    assert "guest mode" in page.locator(".guest-nudge").inner_text()
    no_sideways_scroll(page, f"phone home {scheme}")

    # upload a sample; when the analysis is done a chat opens
    page.get_by_role("button", name="Upload a dataset").click()
    page.get_by_role("button", name="Real: LaLonde NSW experiment").click()
    page.wait_for_function("document.querySelector('#m-claim').value.length > 0")
    page.get_by_role("button", name="Audit the claim").click()
    page.wait_for_url(re.compile(r".*#/m/c/\w+$"), timeout=120_000)
    page.get_by_role("button", name="Explain the estimate in plain words").click()
    page.locator(".m-card", has_text="Estimated effect").wait_for(timeout=15_000)
    no_sideways_scroll(page, f"phone chat {scheme}")
    page.screenshot(path=str(SHOTS / f"phone-chat-{scheme}.png"))

    # the graph: full screen, a bottom sheet for "Several…", the canvas shrinks above it
    page.get_by_role("link", name="Investigate the causal graph").click()
    page.locator(".m-graph-canvas canvas").wait_for()
    full = page.locator(".m-graph-canvas").bounding_box()["height"]
    page.get_by_role("button", name="Several…").click()
    page.get_by_role("region", name="Investigate several variables").wait_for()
    assert page.locator(".m-graph-canvas").bounding_box()["height"] < full
    no_sideways_scroll(page, f"phone graph {scheme}")
    page.screenshot(path=str(SHOTS / f"phone-graph-{scheme}.png"))

    # home lists the dataset with its chat; deleting the chat asks first
    page.goto(base_url + "/#/m")
    ds = page.locator(".m-ds").first
    ds.wait_for()
    assert ds.locator(".m-chat-row").count() == 1 and ds.locator(".m-grade").inner_text() == "C"
    ds.get_by_role("button", name=re.compile("^Delete the chat")).click()
    confirm_dialog(page).get_by_role("button", name="Delete chat").click()
    page.wait_for_function("document.querySelectorAll('.m-chat-row').length === 0")
    page.screenshot(path=str(SHOTS / f"phone-home-{scheme}.png"))
    assert page.errors == [], page.errors


def test_desktop_link_to_the_phone_app_and_bubble_on_narrow_screens(browser, base_url):
    page = new_page(browser, 390)
    run_lalonde(page, base_url)                                      # the desktop page, opened directly on a phone
    assert_layout(page, "390 desktop page with the bubble")
    box = page.get_by_role("button", name="Ask the Investigator about this analysis").bounding_box()
    assert abs(box["width"] - 60) < 1                                # 60 px on phones
    page.get_by_role("link", name="Phone app").click()
    page.wait_for_url("**/#/m")
    assert page.errors == [], page.errors
