"""Browser tests for accounts, guest mode, the confirmation dialog, deletion and the email notice."""
import re
import uuid

import pytest

from conftest import ROOT, SHOTS, assert_layout, confirm_dialog, new_page, run_lalonde

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(),
                                reason="frontend not built (npm run build)")


def email():
    return f"jane.doe@{uuid.uuid4().hex[:6]}.example.org"   # unique per test, initials JD


def sign_up(page, address, password="password123"):
    page.get_by_role("button", name="Create account").click()
    dlg = page.locator("dialog.auth-dialog[open]")
    dlg.get_by_label("Email").fill(address)
    dlg.get_by_label("Password").fill(password)
    dlg.get_by_role("button", name="Create account").click()
    page.locator(".acct-avatar").wait_for()


def test_guest_signup_signout_signin(browser, base_url):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, 1440)
    job = run_lalonde(page, base_url)
    nudge = page.locator(".guest-nudge")
    assert "guest mode" in nudge.inner_text()
    assert page.locator("text=Email me at").count() == 0            # guests can't ask for email

    # create an account: the guest's analysis moves to it
    address = email()
    sign_up(page, address)
    assert page.locator(".acct-avatar").inner_text() == "JD"         # jane.doe.xxxx -> J + D
    assert page.locator(".acct-email").inner_text() == address
    assert page.locator(".guest-nudge").count() == 0
    assert page.locator(f".side-item[href='#/?job={job}']").count() == 1

    # sign out: the open analysis is cleared and the account's work is hidden
    page.locator(".acct-btn").click()
    page.get_by_role("menuitem", name="Sign out").click()
    page.get_by_role("button", name="Sign in", exact=True).wait_for()
    assert page.evaluate("location.hash") in ("#/", "")
    page.wait_for_function(f"!document.querySelector(\".side-item[href='#/?job={job}']\")")

    # wrong password: an error line in the dialog; then the right one brings the analysis back
    page.get_by_role("button", name="Sign in", exact=True).click()
    dlg = page.locator("dialog.auth-dialog[open]")
    dlg.get_by_label("Email").fill(address)
    dlg.get_by_label("Password").fill("not-the-password")
    dlg.get_by_role("button", name="Sign in").click()
    dlg.get_by_role("alert").wait_for()
    assert dlg.get_by_role("alert").inner_text() == "Wrong email or password."
    # Chrome logs the expected 401 for the wrong password; that one is fine
    page.errors[:] = [e for e in page.errors if "status of 401" not in e]
    dlg.get_by_label("Password").fill("password123")
    dlg.get_by_role("button", name="Sign in").click()
    page.locator(f".side-item[href='#/?job={job}']").wait_for()
    assert page.errors == [], page.errors


def test_email_checkbox_and_send_failure_is_shown(browser, base_url):
    page = new_page(browser, 1440)
    page.goto(base_url + "/")
    sign_up(page, email())
    page.get_by_role("button", name="Real: LaLonde NSW experiment").click()
    box = page.get_by_label(re.compile(r"^Email me at .+ when it's done$"))
    assert box.is_checked()                                           # on by default for signed-in users
    page.get_by_role("button", name="Run analysis").click()
    page.get_by_role("tab", name="Overview", exact=True).wait_for(timeout=120_000)
    # the test server's mail goes to a closed port, so the failure must be shown, not swallowed
    alert = page.locator(".banner-warn", has_text="Couldn't send the email notification")
    alert.wait_for(timeout=30_000)
    assert "SMTP" in alert.inner_text()
    assert page.errors == [], page.errors


def test_confirm_dialog_and_deleting_the_open_analysis(browser, base_url):
    page = new_page(browser, 1440)
    job = run_lalonde(page, base_url)
    item = page.locator(".side-list > li", has=page.locator(f".side-item[href='#/?job={job}']"))
    item.hover()
    item.get_by_role("button", name="Delete lalonde_experimental.csv").click()
    dlg = confirm_dialog(page)
    assert dlg.locator("h2").inner_text() == "Delete “lalonde_experimental.csv”?"
    assert "This can't be undone." in dlg.inner_text()
    assert page.evaluate("document.activeElement.textContent") == "Cancel"   # Enter never deletes
    page.keyboard.press("Escape")                                             # Esc cancels
    assert page.locator("dialog.confirm-dialog[open]").count() == 0
    assert item.count() == 1

    item.hover()
    item.get_by_role("button", name="Delete lalonde_experimental.csv").click()
    dlg = confirm_dialog(page)
    dlg.get_by_role("button", name="Delete analysis").click()
    page.wait_for_function(f"!document.querySelector(\".side-item[href='#/?job={job}']\")")
    assert page.evaluate("location.hash") in ("#/", "")                      # back to a blank analysis page
    assert page.locator(".results").count() == 0
    assert page.errors == [], page.errors
    # the server agrees (asked with this browser's guest id; this request logs its own 404 to the console)
    assert page.evaluate(f"fetch('/api/jobs/{job}').then(r => r.status)") == 404


def test_phone_header_and_bottom_sheet(browser, base_url):
    SHOTS.mkdir(exist_ok=True)
    page = new_page(browser, 390)
    job = run_lalonde(page, base_url)
    assert_layout(page, "390 guest header")
    assert page.get_by_role("button", name="Create account").is_hidden()      # just "Sign in" below 560 px
    page.get_by_role("button", name="Show your analyses").click()
    page.get_by_role("button", name="Delete lalonde_experimental.csv").first.click()
    dlg = confirm_dialog(page)
    page.wait_for_timeout(300)
    box = dlg.bounding_box()
    assert abs(box["width"] - 390) < 2 and abs(box["y"] + box["height"] - 900) < 2   # a bottom sheet
    buttons = dlg.locator(".confirm-actions .btn")
    assert buttons.nth(0).bounding_box()["y"] > buttons.nth(1).bounding_box()["y"]  # Cancel last (lowest)
    page.screenshot(path=str(SHOTS / "confirm-390.png"))
    dlg.get_by_role("button", name="Cancel").click()
    assert page.errors == [], page.errors
