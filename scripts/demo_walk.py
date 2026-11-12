"""
A real-browser walk through the whole app, with screenshots
===========================================================
Drives a RUNNING Deep Learning Tutor server with headless Chromium (Playwright) the way a
person would in a demo, checks after every step that no Streamlit error is on the page,
and saves the screenshots used in the README.

  1. Iris: train, result card, charts + confusion matrix + per-class table,
     predict a random test example, show the exported code.
  2. MNIST, 3 epochs: train, the bundled handwritten 7, Grad-CAM.
  3. One question to the AI tutor (only with --tutor; needs a Gemini key in .env).
  4. The result screen again in Malayalam.

Playwright is not in requirements.txt (the app does not need it):

    pip install playwright
    python -m playwright install chromium
    streamlit run app.py --server.headless true --server.port 8599      # in another terminal
    python scripts/demo_walk.py http://localhost:8599 --shots docs/images --tutor

Exit code 0 = every step ran without a Streamlit error.
"""

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.browser_timings import OBSERVER_JS, interact, wait_settled  # noqa: E402

TUTOR_QUESTION = "In three sentences: what does the confusion matrix of my trained model tell me?"
problems = []


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def check_no_errors(page, step):
    """Fail the step if Streamlit shows an exception box or a red error message."""
    exceptions = page.locator('[data-testid="stException"]').count()
    errors = page.locator('[data-testid="stAlertContentError"]').count()
    if exceptions or errors:
        text = page.locator('[data-testid="stException"], [data-testid="stAlertContentError"]').first.inner_text()
        problems.append(f"{step}: {text[:200]}")
        log(f"  PROBLEM in '{step}': {text[:200]}")
    else:
        log(f"  ok: {step}")


def shot(page, folder, name, height=None):
    if folder is None:
        return
    if height:
        page.set_viewport_size({"width": 1440, "height": height})
        time.sleep(1.0)                                   # let the charts re-layout
    path = Path(folder) / name
    page.screenshot(path=str(path))
    log(f"  screenshot: {path}")
    if height:
        page.set_viewport_size({"width": 1440, "height": 900})


def open_tab(page, index):
    interact(page, lambda: page.get_by_role("tab").nth(index).click())


def click_button(page, name):
    interact(page, lambda: page.get_by_role("button", name=name, exact=True).first.click())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("--shots", help="folder for the screenshots (omit to take none)")
    ap.add_argument("--tutor", action="store_true", help="also ask the AI tutor one question (one Gemini call)")
    args = ap.parse_args()
    if args.shots:
        Path(args.shots).mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(args.url)
        page.wait_for_selector('[data-testid="stSelectbox"]', timeout=180_000)     # TensorFlow has loaded
        page.evaluate(OBSERVER_JS)
        wait_settled(page, 0)
        sidebar = page.locator('[data-testid="stSidebar"]')
        check_no_errors(page, "page loads")

        # ── 1. Iris ──────────────────────────────────────────────────────────
        log("Iris: train with the default settings")
        click_button(page, "Train model")
        page.get_by_text("Test accuracy").first.wait_for(timeout=120_000)
        check_no_errors(page, "Iris trained, result card shown")
        shot(page, args.shots, "result_card.png", height=1150)

        open_tab(page, 1)
        page.get_by_text("Precision").first.wait_for(timeout=60_000)
        check_no_errors(page, "Iris charts, confusion matrix and per-class table")
        shot(page, args.shots, "charts.png", height=1500)

        open_tab(page, 3)
        click_button(page, "Fill in a random test example")
        click_button(page, "Predict")
        page.get_by_text("Prediction:").first.wait_for(timeout=60_000)
        interact(page, lambda: page.get_by_text("Show the equivalent Python code").first.click())
        page.get_by_text("load_iris").first.wait_for(timeout=30_000)
        check_no_errors(page, "Iris prediction and exported code")

        # ── 2. MNIST, 3 epochs ──────────────────────────────────────────────
        log("MNIST: switch to the CNN, 3 epochs")
        interact(page, lambda: sidebar.locator('[data-testid="stSelectbox"]').first.click())
        interact(page, lambda: page.get_by_role("option", name="CNN (images)").click())
        slider = sidebar.locator('[data-testid="stSlider"] input[type="range"]').first    # Epochs (default 5)
        for _ in range(2):                                 # arrow keys move it one epoch at a time: 5 -> 3
            slider.focus()
            interact(page, lambda: page.keyboard.press("ArrowLeft"))
        if slider.input_value() != "3":
            problems.append(f"epochs slider is at {slider.input_value()}, expected 3")
        open_tab(page, 0)
        click_button(page, "Train model")
        page.get_by_text("3 / 3").first.wait_for(timeout=300_000)
        wait_settled(page, 0)
        check_no_errors(page, "MNIST trained (3 epochs)")

        open_tab(page, 3)
        click_button(page, "Handwritten 7")
        page.get_by_text("Prediction:").first.wait_for(timeout=60_000)
        click_button(page, "Explain with Grad-CAM")
        page.get_by_text("red = strong influence").first.wait_for(timeout=60_000)
        check_no_errors(page, "sample image and Grad-CAM")
        shot(page, args.shots, "gradcam.png", height=1250)

        # ── 3. One tutor question ───────────────────────────────────────────
        if args.tutor:
            log("AI tutor: one question")
            open_tab(page, 2)
            box = page.locator('[data-testid="stChatInput"] textarea')
            if box.count() == 0:
                problems.append("tutor: no chat input (is a Gemini key configured?)")
            else:
                box.fill(TUTOR_QUESTION)
                interact(page, lambda: page.keyboard.press("Enter"))
                answers = page.locator('[data-testid="stChatMessage"]')
                if answers.count() >= 2 and page.locator('[data-testid="stAlertContentError"]').count() == 0:
                    check_no_errors(page, "tutor answered")
                    shot(page, args.shots, "tutor.png")
                else:
                    check_no_errors(page, "tutor answer")
                    problems.append("tutor: no answer was shown (screenshot skipped)")

        # ── 4. Malayalam ────────────────────────────────────────────────────
        log("Malayalam interface")
        interact(page, lambda: sidebar.locator('[role="radiogroup"]').first.locator("label").nth(1).click())
        for index in (2, 3, 1, 0):                         # every tab renders in Malayalam; end on the result card
            open_tab(page, index)
            check_no_errors(page, f"Malayalam, tab {index + 1}")
        page.get_by_text("ടെസ്റ്റ് കൃത്യത").first.wait_for(timeout=30_000)
        shot(page, args.shots, "malayalam.png")
        browser.close()

    if problems:
        print("\nPROBLEMS:")
        for line in problems:
            print(" -", line)
        sys.exit(1)
    print("\nDemo walk finished: no Streamlit error on any screen.")


if __name__ == "__main__":
    main()
