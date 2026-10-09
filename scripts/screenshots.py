"""Regenerate the README screenshots in docs/screenshots.

Starts the application in-process with a temporary recording library and
drives it in a headless Chromium browser through example designs and
synthetic signals only (no recording from your own library is used).

    python -m pip install -e '.[ui-test]'
    python scripts/screenshots.py [output folder]

Set VECTORRF_BROWSER_PATH to use an installed Chromium-based browser
(Microsoft Edge or Google Chrome) instead of Playwright's.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "ui"))
import ui_smoke as harness  # noqa: E402  (points VECTORRF_CAPTURES_DIR at a temporary folder)
from playwright.sync_api import sync_playwright  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "screenshots"
DESIGNS = ROOT / "examples" / "designs"
MARGIN_LEFT, MARGIN_RIGHT = 58, 12   # the analyzer plot's margins (plots/analyzer.js)


def design(name: str, **changes) -> dict:
    tx = json.loads((DESIGNS / f"{name}.json").read_text(encoding="utf-8"))["tx"]
    return {**tx, **changes}


def apply(t: harness.Session, tx: dict, noise: bool = True):
    """Make ``tx`` the transmitter, as the app restores the last design on load."""
    t.page.evaluate("([tx, noise]) => localStorage.setItem('vectorrf.design.v1', JSON.stringify({tx, noise}))", [tx, noise])
    t.page.evaluate("localStorage.setItem('vectorrf.mode', 'tx')")
    t.page.reload()
    t.page.wait_for_function("window.vectorrf && window.vectorrf.state.ready", timeout=20000)
    t.page.wait_for_timeout(1200)


def shot(page, name: str, settle_ms: int = 900):
    page.wait_for_timeout(settle_ms)
    page.evaluate("document.querySelectorAll('.toast').forEach(t => t.remove())")
    page.mouse.move(2, 2)
    page.screenshot(path=str(OUT / f"{name}.png"))
    print("saved", OUT / f"{name}.png")


def x_at(box: dict, hz: float, center: float, span: float) -> float:
    width = box["width"] - MARGIN_LEFT - MARGIN_RIGHT
    return box["x"] + MARGIN_LEFT + (hz - (center - span / 2)) / span * width


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    base = harness.start_server()
    with sync_playwright() as p:
        launch = {"headless": True}
        if os.environ.get("VECTORRF_BROWSER_PATH"):
            launch["executable_path"] = os.environ["VECTORRF_BROWSER_PATH"]
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(base + "/")
        page.wait_for_function("window.vectorrf && window.vectorrf.state.ready", timeout=20000)
        t = harness.Session(page, base)

        # A CW tone at 0 Hz and a 10 MHz QPSK carrier at +20 MHz, each at its own power.
        apply(t, design("tone-plus-qpsk"))
        t.wait_for("s.txUi.plan && s.txUi.plan.components.length === 2")
        page.click(".component[data-index='1'] .c-name")
        shot(page, "transmitter-composite", 2500)

        # The analyzer on it: a peak marker on the tone, a delta marker and a channel window on the QPSK.
        t.mode("spectrum")
        page.wait_for_timeout(2000)
        box = page.locator(".analyzer-plot canvas").bounding_box()
        fs, y = t.state("s.tx.sample_rate"), box["y"] + box["height"] * 0.45
        page.mouse.click(x_at(box, 0, 0, fs), y)
        page.keyboard.press("p")
        page.keyboard.down("Control")
        page.mouse.click(x_at(box, 20e6, 0, fs), y)
        page.keyboard.up("Control")
        page.click("[data-action='marker-delta']")
        page.mouse.move(x_at(box, 14.8e6, 0, fs), y)
        page.mouse.down()
        page.mouse.move(x_at(box, 25.2e6, 0, fs), y, steps=8)
        page.mouse.up()
        page.click("[data-selection-action='channel']")
        shot(page, "spectrum-analyzer", 2500)

        # QPSK hopping 900 MHz – 1 GHz in 100 kHz steps around a 950 MHz RF centre.
        apply(t, design("fhss-qpsk-900m"))
        t.wait_for("s.txUi.plan && s.txUi.plan.agility && s.txUi.plan.agility.segments === 1001")
        page.click("#txMain [data-txpage='agility']")
        shot(page, "transmitter-hopping", 2500)

        # 60 ms of it exported into the library and opened as a recording.
        saved = page.evaluate("""async () => {
            const {txRequest} = await import('/assets/js/core/source.js');
            const r = await fetch('/api/files/export', {method: 'POST', headers: {'Content-Type': 'application/json'},
              body: JSON.stringify({tx: txRequest(), duration_s: 0.06, format: 'ci16_le', container: 'sigmf',
                                    destination: 'library', name: 'fhss_qpsk_900M_1G'})});
            return (await r.json()).saved;
        }""")
        t.mode("files")
        page.click(f"[data-name='{saved}']")
        page.wait_for_selector("#ovSpec", timeout=30000)
        # The library folder is a temporary local path: keep it out of the image.
        page.evaluate("[...document.querySelectorAll('p.note')].filter(n => n.textContent.startsWith('Folder:')).forEach(n => n.style.visibility = 'hidden')")
        shot(page, "recordings-overview", 2500)

        # The default QPSK link carrying text, demodulated.
        apply(t, design("qpsk-link", data_source="text", message="VECTORRF LAB "))
        t.mode("rx")
        page.click("#rxMain [data-rxpage='acquire']")
        page.wait_for_function("/%/.test(document.getElementById('rEvm')?.textContent || '')", timeout=15000)
        shot(page, "receiver-demodulation", 2500)

        # An echo two symbols late: the channel estimate and the converged CMA equalizer.
        page.evaluate("""() => { const s = window.vectorrf.state; Object.assign(s.channel, {echo_gain: 0.3, echo_delay_symbols: 2});
                                 Object.assign(s.rx, {equalizer: 'cma', equalizer_gain: 0.005}); }""")
        page.click("#rxMain [data-rxpage='equalizer']")
        page.wait_for_function("/channel/.test(document.getElementById('rxChanValue')?.textContent || '')", timeout=15000)
        shot(page, "receiver-equalizer", 5000)
        page.evaluate("""() => { const s = window.vectorrf.state; Object.assign(s.channel, {echo_gain: 0});
                                 Object.assign(s.rx, {equalizer: 'off'}); }""")

        page.click("#rxMain [data-rxpage='pipeline']")
        page.select_option("[data-change-action='block-kind']", "lowpass")
        page.click("[data-action='block-add']")
        t.fill("pipeline.blocks.0.cutoff_hz", "40k")
        shot(page, "receiver-processing", 2000)
        page.click("[data-action='block-remove']")

        t.mode("blind")
        page.wait_for_function("document.getElementById('cClass') && document.getElementById('cClass').textContent !== '—'", timeout=20000)
        page.click("#clMain [data-clpage='decision']")
        shot(page, "classifier-decision", 1500)

        # Modulation quality of a 16-QAM PN15 carrier: the record of the most recent symbols.
        apply(t, design("qpsk-link", waveform="16qam"))
        t.mode("spectrum")
        page.click("#specMain [data-page='quality']")
        page.wait_for_function("/2,048 symbols/.test(document.getElementById('qAll')?.textContent || '')", timeout=30000)
        shot(page, "modulation-quality", 1500)

        # Radar Studio.
        t.mode("radar")
        page.wait_for_function("/#\\d+/.test(document.getElementById('rdCapture')?.textContent || '')", timeout=20000)
        page.click("#radarMain [data-rpage='waveform']")
        shot(page, "radar-waveform", 2000)
        page.click("#radarMain [data-rpage='scene']")
        shot(page, "radar-scene", 2500)
        page.click("#radarMain [data-rpage='live']")
        page.select_option("[data-bind='studio.processing.detector']", "os-cfar")
        page.wait_for_function("/OS-CFAR/.test(document.getElementById('rResults')?.textContent || '')", timeout=15000)
        shot(page, "radar-detection", 2500)
        browser.close()


if __name__ == "__main__":
    main()
