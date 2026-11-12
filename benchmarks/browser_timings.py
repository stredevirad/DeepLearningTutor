"""
Real-browser interaction timings for a running Deep Learning Tutor server
========================================================================
AppTest (used by run_benchmarks.py) runs the script in-process, so it cannot see
work the real Streamlit server does around each run (file watching, sending
messages over the websocket). This script drives headless Chrome with
Playwright and measures, from the user's point of view:

  * how long a new browser session takes until the page is usable, and
  * how long the page takes to settle after an interaction.

"Settled" = no DOM change for QUIET seconds and Streamlit is not running; the
reported time is click -> last DOM change.

    pip install playwright            # uses the installed Chrome, no browser download
    streamlit run app.py --server.headless true --server.port 8599
    python benchmarks/browser_timings.py http://localhost:8599 --out browser.json
"""

import argparse
import json
import statistics
import time

from playwright.sync_api import sync_playwright

QUIET = 1.2          # seconds without DOM changes that count as "settled"
OBSERVER_JS = """() => {
  window.__lastMut = performance.now();
  if (!window.__obs) {
    window.__obs = new MutationObserver(() => { window.__lastMut = performance.now(); });
    window.__obs.observe(document.body, {subtree: true, childList: true, attributes: true, characterData: true});
  }
}"""


def running(page):
    """True while Streamlit runs the script (the app root carries data-test-script-state)."""
    return page.evaluate("""() => { const app = document.querySelector('[data-testid="stApp"]');
        if (app && app.getAttribute('data-test-script-state')) return app.getAttribute('data-test-script-state') === 'running';
        const s = document.querySelector('[data-testid="stStatusWidget"]');
        return !!(s && /stop|running/i.test(s.innerText)); }""")


def wait_settled(page, start_ms, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(0.1)
        now, last = page.evaluate("() => [performance.now(), window.__lastMut]")
        if now - last >= QUIET * 1000 and not running(page):
            return last - start_ms
    raise TimeoutError("page did not settle")


def interact(page, action):
    page.evaluate(OBSERVER_JS)
    start = page.evaluate("() => performance.now()")
    action()
    return wait_settled(page, start)


def stats(values):
    return {"median": round(statistics.median(values), 1), "min": round(min(values), 1),
            "max": round(max(values), 1), "n": len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--train", action="store_true", help="also train Iris and time interactions with a model")
    args = ap.parse_args()
    results = {}

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome", headless=True)
        # Warm the server process (first session imports TensorFlow etc.)
        warm = browser.new_page()
        warm.goto(args.url)
        warm.wait_for_selector('[data-testid="stSelectbox"]', timeout=180_000)
        warm.close()

        loads = []
        for _ in range(3):
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            t0 = time.perf_counter()
            page.goto(args.url)
            page.wait_for_selector('[data-testid="stSelectbox"]', timeout=180_000)
            page.evaluate(OBSERVER_JS)
            wait_settled(page, 0)
            loads.append((time.perf_counter() - t0 - QUIET) * 1000)
            page.close()
        results["new_session_until_settled_ms"] = stats(loads)

        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(args.url)
        page.wait_for_selector('[data-testid="stSelectbox"]', timeout=180_000)
        page.evaluate(OBSERVER_JS)
        wait_settled(page, 0)

        def checkbox():
            return page.locator('[data-testid="stCheckbox"]').first

        results["toggle_checkbox_ms"] = stats(
            [interact(page, lambda: checkbox().click()) for _ in range(args.repeats)])

        def lang(index):   # the first radio group in the sidebar is the language switch
            group = page.locator('[data-testid="stSidebar"] [role="radiogroup"]').first
            return lambda: group.locator("label").nth(index).click()
        results["language_switch_ms"] = stats([interact(page, lang(1 if i % 2 == 0 else 0)) for i in range(4)])

        tabs = page.get_by_role("tab")
        results["tab_switch_ms"] = stats([interact(page, lambda i=i: tabs.nth(1 + i % 3).click()) for i in range(4)])
        interact(page, lambda: tabs.nth(0).click())

        if args.train:
            train = page.get_by_role("button", name="Train").first
            results["train_iris_default_click_ms"] = interact(page, lambda: train.click())
            results["toggle_checkbox_trained_ms"] = stats(
                [interact(page, lambda: checkbox().click()) for _ in range(args.repeats)])
            results["tab_switch_trained_ms"] = stats(
                [interact(page, lambda i=i: tabs.nth(1 + i % 3).click()) for i in range(4)])
        browser.close()

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
