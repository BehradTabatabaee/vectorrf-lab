"""End-to-end browser smoke test of the VectorRF Lab UI.

Starts the real application on a free local port with a temporary recording
library, then drives every workspace in Chromium through the same controls a
person uses.  Run directly:

    python tests/ui/ui_smoke.py

Set VECTORRF_BROWSER_PATH to use an installed Chromium-based browser (Chrome,
Edge) instead of Playwright's bundled Chromium; VECTORRF_SCREENSHOTS to keep
screenshots of each workspace.
"""
from __future__ import annotations

import atexit
import os
import random
import shutil
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
LIBRARY = Path(tempfile.mkdtemp(prefix="vectorrf-ui-"))


@atexit.register
def _remove_library():
    from vectorrf import recordings
    recordings.close_readers()  # Windows cannot delete a memory-mapped file
    shutil.rmtree(LIBRARY, ignore_errors=True)


os.environ["VECTORRF_CAPTURES_DIR"] = str(LIBRARY)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import uvicorn  # noqa: E402
from playwright.sync_api import Page, sync_playwright  # noqa: E402

from vectorrf import iqfile  # noqa: E402
from vectorrf.app import app  # noqa: E402

SHOTS = os.environ.get("VECTORRF_SCREENSHOTS")


def make_recording() -> str:
    """A 2 MS/s SigMF recording: a tone and a QPSK-like noise band."""
    fs, n = 2e6, 400_000
    k = np.arange(n)
    rng = np.random.default_rng(3)
    x = 0.4 * np.exp(2j * np.pi * 250e3 * k / fs) + 0.05 * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
    fmt = iqfile.SampleFormat.parse("ci16_le")
    data, _ = iqfile.encode_recording(x.astype(np.complex64), fmt, fs, "raw", 0.5)
    (LIBRARY / "ui_test.sigmf-data").write_bytes(data)
    import json
    meta = iqfile.sigmf_meta(fmt, fs, 433.92e6, "UI smoke recording", {})
    (LIBRARY / "ui_test.sigmf-meta").write_text(json.dumps(meta), encoding="utf-8")
    return "ui_test.sigmf-meta"


def free_port() -> int:
    """A free local port in 20000–59999.  Browsers refuse a list of lower
    "unsafe" ports (e.g. 3659), and some systems hand out ephemeral ports there."""
    rng = random.Random()
    for _ in range(200):
        port = rng.randrange(20000, 60000)
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
            except OSError:
                continue
        return port
    raise RuntimeError("no free port in 20000–59999")


def start_server() -> str:
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("server did not start")
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}"


class Session:
    def __init__(self, page: Page, base: str):
        self.page, self.base = page, base

    def state(self, expr: str):
        return self.page.evaluate(f"(() => {{ const s = window.vectorrf.state; return {expr}; }})()")

    def mode(self, name: str):
        self.page.click(f".workspace-tab[data-mode='{name}']")
        self.page.wait_for_timeout(400)

    def fill(self, bind: str, text: str):
        field = self.page.locator(f"[data-bind='{bind}']").first
        field.fill(text)
        field.press("Enter")
        self.page.wait_for_timeout(250)

    def wait_for(self, expr: str, timeout: float = 15000):
        self.page.wait_for_function(f"(() => {{ const s = window.vectorrf.state; return {expr}; }})()", timeout=timeout)

    def capture(self, element_id: str) -> int:
        """Capture number shown by a readout such as ``#123``."""
        return int(self.page.inner_text(f"#{element_id}").lstrip("#"))

    def advances(self, element_id: str, at_least: int = 2, timeout: float = 15000):
        """Wait until a capture readout has moved on by ``at_least`` captures."""
        self.page.wait_for_function(f"/#\\d+/.test(document.getElementById('{element_id}')?.textContent || '')", timeout=timeout)
        first = self.capture(element_id)
        self.page.wait_for_function(
            f"Number((document.getElementById('{element_id}')?.textContent || '').replace('#', '')) >= {first + at_least}",
            timeout=timeout)

    def quantity(self, element_id: str, unit: str) -> float:
        """Parse a readout with the UI's own engineering-notation parser."""
        return self.page.evaluate(
            "async ([id, unit]) => (await import('/assets/js/core/format.js')).parseQuantity(document.getElementById(id).textContent, unit)",
            [element_id, unit])

    def shot(self, name: str):
        if SHOTS:
            Path(SHOTS).mkdir(parents=True, exist_ok=True)
            self.page.screenshot(path=str(Path(SHOTS) / f"{name}.png"))


PEAKS = """async (floor) => { const live = (await import('/assets/js/core/live.js')).live; const f = live.freq, p = live.psd, out = [];
  for (let i = 2; i < p.length - 2; i++) if (p[i] > floor && p[i] >= p[i - 1] && p[i] > p[i + 1] && p[i] >= p[i - 2] && p[i] > p[i + 2]) out.push(f[i]);
  return out; }"""


def live_peaks(t: Session, floor: float = -30.0) -> list[float]:
    """Frequencies of the live spectrum's local maxima above ``floor`` (dBm per RBW)."""
    return t.page.evaluate(PEAKS, floor)


def reset_design(t: Session):
    """Back to a fresh page with the default design (nothing remembered)."""
    t.page.evaluate("localStorage.clear()")
    t.page.reload()
    t.page.wait_for_function("window.vectorrf && window.vectorrf.state.ready", timeout=20000)
    t.mode("tx")


def check_transmitter(t: Session):
    t.mode("tx")
    # The top-bar logo is the same mark as the browser tab's icon.
    assert t.page.locator(".brand svg.brand-mark path").count() == 1
    # The default signal: QPSK carrying a full-period PN15 sequence.
    assert t.state("[s.tx.waveform, s.tx.data_source, s.tx.pn_order, s.tx.pn_bits].join()") == "qpsk,pn,15,32767"
    # Engineering notation in a number field; in rate fields "m" is mega.
    t.fill("tx.sample_rate", "2M")
    assert t.state("s.tx.sample_rate") == 2e6
    assert t.page.input_value("[data-bind='tx.sample_rate']") == "2 MS/s"
    t.fill("tx.sample_rate", "1msps")
    assert t.state("s.tx.sample_rate") == 1e6
    # Composite: the current signal becomes component 1, add a second one.
    t.page.click("[data-set-path='txUi.family'][data-set-value='\"composite\"']")
    t.page.click("[data-action='add-component']")
    t.wait_for("s.txUi.plan && s.txUi.plan.components.length === 2")
    total = t.state("s.txUi.plan.total_power_dbm")
    assert abs(total - 10 * np.log10(1 + 0.1)) < 0.02, total
    # A duplicate lands where it overlaps nothing, never on its original.
    t.page.click("[data-action='duplicate-component'][data-index='1']")
    t.wait_for("s.txUi.plan && s.txUi.plan.components.length === 3 && s.txApplied.components.length === 3")
    tone, copy = t.state("[s.tx.components[1].center_hz, s.tx.components[2].center_hz]")
    assert abs(copy - tone) >= 25e3, (tone, copy)
    # Select the copy and move it: the live spectrum follows (the reported "it never moves").
    t.page.click(".component[data-index='2'] .c-name")
    t.fill("tx.components.2.center_hz", "300k")
    t.wait_for("s.txApplied.components[2].center_hz === 300e3")
    t.page.wait_for_function(f"({PEAKS})(-30).then(p => p.some(f => Math.abs(f - 300e3) < 2e3))", timeout=10000)
    # Lowercase prefixes: 10 mhz is 10 MHz (beyond Nyquist at 1 MS/s: refused, with the rate that would fit).
    t.fill("tx.components.2.center_hz", "10mhz")
    assert t.state("s.tx.components[2].center_hz") == 300e3
    note = t.page.inner_text("[data-field='tx.components.2.center_hz'] .field-error")
    assert "10 MHz" in note and "20 MS/s" in note, note
    assert t.page.locator(".toast.bad").count() >= 1
    # Steppers move a frequency visibly from 0 Hz (a tenth of the sample rate's decade).
    t.fill("tx.components.2.center_hz", "0")
    t.page.click("[data-field='tx.components.2.center_hz'] .up")
    assert t.state("s.tx.components[2].center_hz") == 10e3
    t.shot("transmitter-composite")
    # Hopping 900 MHz – 1 GHz in 100 kHz steps around a 950 MHz RF centre, entered as absolute frequencies.
    reset_design(t)
    t.fill("tx.sample_rate", "125M")
    t.fill("tx.rf_center_hz", "950M")
    t.fill("tx.symbol_rate", "80k")
    t.page.click("#txMain [data-txpage='agility']")
    t.page.check("[data-bind='tx.agility.enabled']")
    t.fill("tx.agility.start_hz", "900M")
    # Text being typed survives a re-render answering an earlier edit: commit
    # another field while "1G" is typed but not yet entered.
    stop = t.page.locator("[data-bind='tx.agility.stop_hz']")
    stop.fill("1G")
    t.page.evaluate("""() => { const start = document.querySelector("[data-bind='tx.agility.start_hz']");
                               start.value = '901M'; start.dispatchEvent(new Event('change', {bubbles: true})); }""")
    t.wait_for("s.txApplied.agility.start_hz === -49e6")
    assert stop.input_value() == "1G", stop.input_value()
    stop.press("Enter")
    t.fill("tx.agility.start_hz", "900M")
    t.fill("tx.agility.step_hz", "100k")
    t.wait_for("s.txUi.plan && s.txUi.plan.agility && s.txUi.plan.agility.segments === 1001", timeout=20000)
    t.fill("tx.agility.start_hz", "920M")
    assert t.state("s.tx.agility.start_hz") == -30e6
    t.shot("transmitter-hopping")
    reset_design(t)


def check_spectrum(t: Session):
    t.mode("spectrum")
    frames = t.page.evaluate("new Promise(r => { let n = 0; const id = setInterval(() => { n = document.getElementById('statusRate').textContent; }, 100); setTimeout(() => { clearInterval(id); r(n); }, 2500); })")
    assert frames not in ("—", ""), frames
    # The waterfall fills the label gutters with its edge colours; they used
    # to remain unused black strips despite the canvas occupying the pane.
    gutter = t.page.evaluate("""() => {
      const c = document.querySelector('#specMain .analyzer-waterfall canvas');
      const x = c.getContext('2d');
      const dpr = c.width / c.getBoundingClientRect().width;
      const pixel = px => [...x.getImageData(Math.round(px * dpr), Math.round(3 * dpr), 1, 1).data].slice(0, 3);
      return [pixel(2), pixel(55), pixel(59)];
    }""")
    assert gutter[0] == gutter[1] == gutter[2], gutter
    canvas = t.page.locator(".analyzer-plot canvas")
    box = canvas.bounding_box()
    # Click places a marker; drag selects a band; channel power measures it.
    t.page.mouse.click(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.4)
    assert t.state("s.specUi.markers.length") == 1
    t.page.mouse.move(box["x"] + box["width"] * 0.62, box["y"] + 80)
    t.page.mouse.down()
    t.page.mouse.move(box["x"] + box["width"] * 0.75, box["y"] + 80, steps=6)
    t.page.mouse.up()
    assert t.state("s.specUi.selection !== null")
    t.page.click("[data-selection-action='channel']")
    t.wait_for("s.spectrum.channels.length === 1")
    t.page.wait_for_function("document.querySelector('.channel-chip b') && /dBm/.test(document.querySelector('.channel-chip b').textContent)", timeout=10000)
    # Wheel over the trace zooms and the analyzer down-converts.
    t.page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    for _ in range(4):
        t.page.mouse.wheel(0, -300)
        t.page.wait_for_timeout(80)
    t.wait_for("s.spectrum.span_hz !== null && s.spectrum.span_hz < 250e3")
    t.page.wait_for_function("/DDC/.test(document.getElementById('sZoom').textContent)", timeout=10000)
    t.shot("spectrum-instrument")
    t.page.dblclick(".analyzer-plot canvas")
    t.wait_for("s.spectrum.span_hz === null")
    # Zero span follows consecutive captures.
    t.page.click("#specMain [data-page='zero']")
    t.advances("zCapture")
    previous = t.state("s.measurement.offset_s")
    t.fill("measurement.offset_s", "garbage")
    assert t.state("s.measurement.offset_s") == previous
    t.page.check("[data-bind='measurement.tuned']")
    center_before = t.page.input_value("[data-bind='measurement.center_hz']")
    t.fill("measurement.center_hz", "700k")
    t.page.wait_for_function("document.getElementById('zCapture')?.textContent === 'Unavailable'", timeout=15000)
    t.fill("measurement.center_hz", center_before)
    t.advances("zCapture")
    # Modulation quality tracks the stream and accumulates a record of symbols.
    t.page.click("#specMain [data-page='quality']")
    t.page.wait_for_function("/symbols/.test(document.getElementById('qAll')?.textContent || '')", timeout=15000)
    evm = t.quantity("qEvm", "%")
    assert evm < 3, evm
    t.page.click("#specMain [data-page='sweep']")


def check_receiver(t: Session):
    # A text payload, so the decoded output can be read back.
    t.mode("tx")
    t.page.click("#txMain [data-txpage='data']")
    t.page.select_option("[data-bind='tx.data_source']", "text")
    t.wait_for("s.txApplied.data_source === 'text'")
    t.mode("rx")
    t.page.click("#rxMain [data-rxpage='acquire']")
    t.page.wait_for_function("/%/.test(document.getElementById('rEvm')?.textContent || '')", timeout=15000)
    t.page.wait_for_timeout(1500)
    evm = float(t.page.inner_text("#rEvm").split()[0])
    text = t.page.inner_text("#rxDecoded pre")
    assert evm < 5 and "VECTORRF LAB" in text, (evm, text[:80])
    t.shot("receiver-demodulation")
    assert "FOLLOWING" in t.page.inner_text("#rxFollowBadge").upper()
    t.fill("rx.symbol_rate", "40k")
    assert "MANUAL" in t.page.inner_text("#rxFollowBadge").upper()
    t.page.click("[data-action='refollow']")
    assert t.state("s.rxUi.follow") is True
    # The equalizer page shows the channel estimated against the known symbols.
    t.page.click("#rxMain [data-rxpage='equalizer']")
    t.page.wait_for_function("/channel/.test(document.getElementById('rxChanValue')?.textContent || '')", timeout=15000)
    # The monitor consumes consecutive captures.
    t.page.click("#rxMain [data-rxpage='monitor']")
    t.advances("mCapture", at_least=3)
    # A processing block runs in the live chain.
    t.page.click("#rxMain [data-rxpage='pipeline']")
    t.page.select_option("[data-change-action='block-kind']", "lowpass")
    t.page.click("[data-action='block-add']")
    t.fill("pipeline.blocks.0.cutoff_hz", "180k")
    assert t.state("s.pipeline.blocks.length === 1 && s.pipeline.blocks[0].cutoff_hz === 180e3")
    t.advances("pCapture")
    t.shot("receiver-processing")
    t.page.click("[data-action='block-remove']")
    t.wait_for("s.pipeline.blocks.length === 0")


def check_ofdm(t: Session):
    t.mode("tx")
    t.page.click("#txMain [data-txpage='signal']")
    t.page.select_option("[data-change-action='pick-family'][data-path='tx']", "multicarrier")
    t.wait_for("s.tx.waveform === 'ofdm'")
    t.mode("rx")
    t.wait_for("s.rx.modulation === 'ofdm'")  # following the transmitter
    t.page.click("#rxMain [data-rxpage='acquire']")
    t.page.wait_for_function("/Hz/.test(document.getElementById('rCoarse')?.textContent || '')", timeout=15000)
    t.page.wait_for_timeout(800)
    cfo = t.quantity("rCoarse", "Hz")
    assert abs(cfo - t.state("s.channel.freq_offset_hz")) < 120, cfo
    t.shot("receiver-ofdm")
    reset_design(t)


def check_classifier(t: Session):
    t.mode("blind")
    t.page.wait_for_function("document.getElementById('cClass') && document.getElementById('cClass').textContent !== '—'", timeout=20000)
    t.page.click("#clMain [data-clpage='decision']")
    stages = t.page.locator(".trace-stage").count()
    accepted = t.page.locator(".trace-stage.accepted").count()
    assert stages >= 2 and accepted == 1, (stages, accepted)
    t.shot("classifier-decision")


def check_recordings(t: Session, name: str):
    t.mode("files")
    t.page.click(f"[data-name='{name}']")
    t.page.wait_for_selector("#ovSpec", timeout=15000)
    t.page.click("[data-action='toggle-live']")
    t.wait_for("s.source.kind === 'file'")
    assert "FILE" in t.page.inner_text(".source-pill")
    t.shot("recordings-overview")
    t.mode("spectrum")
    t.page.wait_for_function("/433\\.\\d+ MHz/.test(document.getElementById('sCenter').textContent)", timeout=10000)
    t.mode("files")
    t.page.click("[data-action='toggle-live']")
    t.wait_for("s.source.kind === 'transmitter'")


def check_radar(t: Session):
    t.mode("radar")
    t.page.wait_for_function("document.getElementById('rdCapture') && /#\\d+/.test(document.getElementById('rdCapture').textContent)", timeout=20000)
    t.advances("rdCapture")
    t.page.click("#radarMain [data-rpage='scene']")
    t.page.wait_for_selector("#rScene")
    t.shot("radar-scene")
    if t.state("s.studio.running"):
        t.page.click("[data-action='run']")
    t.page.click(".target-list [data-action='select-target'][data-index='0']")
    def scene_state():
        return t.page.evaluate("""() => {
          const scene = window.vectorrf.scene, item = scene.items[0], rect = scene.canvas.getBoundingClientRect();
          return {position: [...scene.states[0].position], velocity: [...scene.states[0].velocity],
                  body: [rect.left + item.bx, rect.top + item.by], tip: [rect.left + item.tx, rect.top + item.ty]};
        }""")
    before = scene_state()
    t.page.mouse.move(*before["body"])
    t.page.mouse.down()
    t.page.mouse.move(before["body"][0] + 18, before["body"][1] - 13, steps=5)
    t.page.mouse.up()
    moved = scene_state()
    assert np.linalg.norm(np.array(moved["position"]) - before["position"]) > 1
    np.testing.assert_allclose(moved["velocity"], before["velocity"], atol=1e-6)
    t.page.mouse.move(*moved["tip"])
    t.page.mouse.down()
    t.page.mouse.move(moved["tip"][0] + 16, moved["tip"][1] - 12, steps=5)
    t.page.mouse.up()
    aimed = scene_state()
    np.testing.assert_allclose(aimed["position"], moved["position"], atol=1e-6)
    assert np.linalg.norm(np.array(aimed["velocity"]) - moved["velocity"]) > 1
    field = t.page.locator('[data-change-action="target-vector"][data-key="velocity"][data-axis="0"]')
    field.fill("12 m/s")
    field.press("Enter")
    edited = scene_state()
    np.testing.assert_allclose(edited["position"], aimed["position"], atol=1e-6)
    assert edited["velocity"][0] == pytest.approx(12, abs=1e-6)
    t.page.click("[data-action='run']")
    # Every detector mode reports grouped detections with their velocities.
    t.page.click("#radarMain [data-rpage='live']")
    for detector in ("go-cfar", "so-cfar", "os-cfar", "ca-cfar"):
        t.page.select_option("[data-bind='studio.processing.detector']", detector)
        t.page.wait_for_function(f"/{detector.upper()}/.test(document.getElementById('rResults')?.textContent || '')", timeout=15000)
    # Another processor on the same scene.
    t.page.click("#radarMain [data-rpage='processing']")
    t.page.select_option("[data-bind='studio.processing.method']", "pulse-doppler")
    t.page.wait_for_timeout(300)
    t.advances("rdCapture")
    assert t.page.locator("#radarError").is_hidden(), t.page.inner_text("#radarError")
    # Pause holds the capture (once a frame already requested has arrived); Step advances exactly one.
    t.page.click("[data-action='run']")
    paused = t.capture("rdCapture")
    for _ in range(20):
        t.page.wait_for_timeout(500)
        if t.capture("rdCapture") == paused:
            break
        paused = t.capture("rdCapture")
    t.page.wait_for_timeout(600)
    assert t.capture("rdCapture") == paused
    t.page.click("[data-action='step']")
    t.page.wait_for_function(f"document.getElementById('rdCapture').textContent === '#{paused + 1}'", timeout=10000)
    t.page.click("#radarMain [data-rpage='scene']")
    while t.state("s.studio.targets.length") < 12:
        t.page.click("[data-action='add-target']")
    t.page.click("[data-action='add-target']")
    assert t.state("s.studio.targets.length") == 12
    t.page.click("[data-action='remove-target']")
    assert t.state("s.studio.targets.length") == 11


def set_hidden(page: Page, hidden: bool):
    """Simulate the window being covered or minimised (Alt+Tab): the page becomes hidden."""
    page.evaluate("""hidden => {
        window.__hidden = hidden;
        if (!window.__hiddenPatched) {
            Object.defineProperty(document, 'hidden', {configurable: true, get: () => window.__hidden});
            Object.defineProperty(document, 'visibilityState', {configurable: true, get: () => window.__hidden ? 'hidden' : 'visible'});
            window.__hiddenPatched = true;
        }
        document.dispatchEvent(new Event('visibilitychange'));
    }""", hidden)


def check_resume_after_hidden(t: Session):
    """Every live display keeps updating after the page was hidden and shown twice."""
    for mode, select, canvas in (("spectrum", None, "#specMain .analyzer-plot canvas"),
                                 ("tx", None, "#txMain .analyzer-plot canvas"),
                                 ("blind", "#clMain [data-clpage='overview']", "#clMain .analyzer-plot canvas"),
                                 ("rx", "#rxMain [data-rxpage='monitor']", "canvas#monIQ")):
        t.mode(mode)
        if select:
            t.page.click(select)
        t.page.wait_for_timeout(800)
        for _ in range(2):
            set_hidden(t.page, True)
            t.page.wait_for_timeout(300)
            set_hidden(t.page, False)
            t.page.wait_for_timeout(300)
        t.page.wait_for_timeout(1200)
        first = t.page.locator(canvas).screenshot()
        for _ in range(12):
            t.page.wait_for_timeout(500)
            if t.page.locator(canvas).screenshot() != first:
                break
        else:
            raise AssertionError(f"{mode} display froze after the page was hidden")


def check_compact_layout(t: Session):
    """Below 900 px the dock stacks above the displays and the page scrolls."""
    t.page.set_viewport_size({"width": 760, "height": 900})
    t.mode("tx")
    t.page.mouse.move(380, 200)  # over the dock (the editor below is a scroll area of its own)
    t.page.mouse.wheel(0, 1200)
    t.page.wait_for_timeout(300)
    assert t.page.evaluate("window.scrollY + document.scrollingElement.scrollTop") > 0, "compact layout does not scroll"
    t.shot("compact")
    t.page.set_viewport_size({"width": 1440, "height": 900})


def check_viewports(t: Session):
    for width in (1920, 1280, 900, 760):
        t.page.set_viewport_size({"width": width, "height": 900})
        for mode in ("tx", "spectrum", "rx", "blind", "radar", "files"):
            t.mode(mode)
            overflow = t.page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
            assert overflow <= 2, (width, mode, overflow)
    t.page.set_viewport_size({"width": 1440, "height": 900})


def check_rapid_view_changes(t: Session):
    for _ in range(4):
        t.mode("spectrum")
        t.page.click("#specMain [data-page='quality']")
        t.mode("rx")
        t.page.click("#rxMain [data-rxpage='monitor']")
        t.mode("spectrum")
        t.page.click("#specMain [data-page='zero']")
    t.advances("zCapture", at_least=2)
    assert t.page.inner_text("#zRms") not in ("—", "NaN", "undefined")


def check_help(page: Page, base: str):
    page.goto(base + "/help")
    page.mouse.wheel(0, 1600)
    page.wait_for_timeout(300)
    assert page.evaluate("window.scrollY") > 0, "help page does not scroll"


def main():
    name = make_recording()
    base = start_server()
    with sync_playwright() as p:
        launch = {"headless": True}
        if os.environ.get("VECTORRF_BROWSER_PATH"):
            launch["executable_path"] = os.environ["VECTORRF_BROWSER_PATH"]
        browser = p.chromium.launch(**launch)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        errors, failures = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("response", lambda r: failures.append((r.url, r.status)) if r.status >= 400 else None)
        page.goto(base + "/")
        page.wait_for_function("window.vectorrf && window.vectorrf.state.ready", timeout=20000)
        t = Session(page, base)
        assert page.locator(".workspace-tab").count() == 6
        for check in (check_transmitter, check_spectrum, check_receiver, check_ofdm, check_classifier):
            check(t)
        check_recordings(t, name)
        check_radar(t)
        check_resume_after_hidden(t)
        check_compact_layout(t)
        check_viewports(t)
        check_rapid_view_changes(t)
        check_help(page, base)
        browser.close()
    assert not failures, failures
    assert not errors, errors
    print({"passed": True})


if __name__ == "__main__":
    main()
