"""Graph gestures (GraphCanvas). Where nodes open details (Investigation page): a click always selects, even with a
little wobble or a slow press; a node moves only after it is held still for a moment, without the graph flashing; a
quick drag pans instead; right-click unpins. On the Analysis page nodes have no details, so they drag at once."""
import pytest

from conftest import ROOT, new_page, run_lalonde
from test_investigate_ui import open_investigation

pytestmark = pytest.mark.skipif(not (ROOT / "frontend" / "dist" / "index.html").exists(),
                                reason="frontend not built (npm run build)")

NODE = "treat"


def pos(page, node=NODE):
    return page.evaluate(f"document.querySelector('.graph-canvas').__graph.screenOf({node!r})")


def selected(page):
    return page.evaluate("decodeURIComponent((location.hash.match(/node=([^&]+)/) || [])[1] || '')")


def settle(page):
    page.wait_for_timeout(4500)        # the layout cools down (150 ticks), so nodes stop drifting


def press(page, p, hold_ms=0, path=(), step_ms=10):
    m = page.mouse
    m.move(p["x"], p["y"])
    m.down()
    if hold_ms:
        page.wait_for_timeout(hold_ms)
    for dx, dy in path:
        m.move(p["x"] + dx, p["y"] + dy)
        page.wait_for_timeout(step_ms)
    m.up()
    page.wait_for_timeout(300)


@pytest.fixture()
def graph(browser, base_url):
    page = new_page(browser, 1440)
    job = open_investigation(page, base_url)
    page.locator(".graph-canvas canvas").scroll_into_view_if_needed()
    settle(page)
    assert pos(page) is not None
    return page, job


def test_click_with_wobble_and_slow_click_select_without_moving(graph):
    page, _ = graph
    p = pos(page)
    press(page, p, path=[(2, 1), (3, 2)])                    # a shaky click
    assert selected(page) == NODE
    assert not pos(page)["pinned"]
    page.mouse.click(5, 5)                                     # outside the canvas: nothing changes
    other = pos(page, "age")
    press(page, other, hold_ms=600)                            # a slow click: held, but never moved
    assert selected(page) == "age"
    assert not pos(page, "age")["pinned"]
    assert page.errors == [], page.errors


def test_quick_drag_pans_instead_of_moving_the_node(graph):
    page, _ = graph
    before = pos(page)
    press(page, before, path=[(20 * i, 10 * i) for i in range(1, 6)])   # moves at once: a pan
    after = pos(page)
    assert not after["pinned"]
    assert selected(page) == ""                                # and not a click either
    # the whole canvas moved with the pointer (the node kept its place within it)
    assert abs((after["x"] - before["x"]) - 100) < 25 and abs((after["y"] - before["y"]) - 50) < 25
    assert page.errors == [], page.errors


def test_hold_then_drag_moves_and_pins_the_node_and_right_click_unpins(graph):
    page, _ = graph
    start = pos(page)
    age0 = pos(page, "age")
    press(page, start, hold_ms=500, path=[(-15 * i, 8 * i) for i in range(1, 9)], step_ms=30)
    moved = pos(page)
    assert moved["pinned"]
    assert abs(moved["x"] - (start["x"] - 120)) < 15 and abs(moved["y"] - (start["y"] + 64)) < 15
    assert selected(page) == ""                                # moving a node does not open it
    # the canvas did not pan along with the node
    age1 = pos(page, "age")
    assert abs(age1["x"] - age0["x"]) < 60 and abs(age1["y"] - age0["y"]) < 60
    page.mouse.move(moved["x"], moved["y"])
    page.wait_for_timeout(200)
    page.mouse.click(moved["x"], moved["y"], button="right")
    page.wait_for_timeout(300)
    assert not pos(page)["pinned"]
    assert page.errors == [], page.errors


def test_a_pinned_node_shows_a_pin_and_clicking_the_pin_releases_it(graph):
    page, _ = graph
    start = pos(page)
    press(page, start, hold_ms=500, path=[(-12 * i, 6 * i) for i in range(1, 9)], step_ms=30)
    p = pos(page)
    assert p["pinned"] and p["pin"]["x"] > p["x"] and p["pin"]["y"] < p["y"]    # upper right of the node
    page.mouse.move(p["x"] - 150, p["y"] + 150)                # no tooltip over it
    page.wait_for_timeout(250)
    page.screenshot(path=str(ROOT / "frontend" / "e2e" / "screens" / "graph-pinned.png"),
                    clip={"x": p["x"] - 60, "y": p["y"] - 60, "width": 200, "height": 160})
    press(page, p["pin"])                                      # a click on the pin
    assert not pos(page)["pinned"]
    assert selected(page) == ""                                # releasing is not opening
    assert page.errors == [], page.errors


def test_dragging_does_not_flash_the_graph(graph):
    page, _ = graph
    # the highlighted node dims everything else; it must stay the dragged node, frame after frame
    page.evaluate("""() => { const g = document.querySelector('.graph-canvas').__graph; window.__h = [];
        const tick = () => { window.__h.push(g.hoverId()); if (window.__h.length < 600) requestAnimationFrame(tick); }; tick(); }""")
    start = pos(page)
    press(page, start, hold_ms=500, path=[(-6 * i, 4 * i) for i in range(1, 31)], step_ms=16)
    h = page.evaluate("window.__h")
    changes = sum(1 for a, b in zip(h, h[1:]) if a != b)
    assert changes <= 2, f"the highlight flipped {changes} times while dragging"
    assert page.errors == [], page.errors


def test_a_node_cannot_be_dragged_off_the_canvas(graph):
    page, _ = graph
    start = pos(page)
    box = page.locator(".graph-canvas").bounding_box()
    up = start["y"] - box["y"] + 200                            # 200 px past the top edge
    press(page, start, hold_ms=500, path=[(0, -up * i / 8) for i in range(1, 9)], step_ms=30)
    p = pos(page)
    assert p["pinned"] and box["y"] <= p["y"] <= box["y"] + 20
    assert page.errors == [], page.errors


def test_background_click_clears_the_selection(graph):
    page, job = graph
    press(page, pos(page))
    assert selected(page) == NODE
    box = page.locator(".graph-canvas canvas").bounding_box()
    page.mouse.click(box["x"] + 8, box["y"] + 8)               # an empty corner
    page.wait_for_timeout(300)
    assert selected(page) == ""


def test_analysis_page_graph_drags_at_once(browser, base_url):
    page = new_page(browser, 1440)
    run_lalonde(page, base_url)
    settle(page)                                               # also lets the finished run's scroll to the results end
    page.locator(".graph-canvas canvas").first.scroll_into_view_if_needed()
    page.wait_for_timeout(300)
    start = pos(page)
    press(page, start, path=[(-8 * i, 5 * i) for i in range(1, 9)], step_ms=30)   # no hold needed
    moved = pos(page)
    assert moved["pinned"] and abs(moved["x"] - (start["x"] - 64)) < 15 and abs(moved["y"] - (start["y"] + 40)) < 15, (start, moved, page.evaluate("window.scrollY"))
    assert page.errors == [], page.errors
