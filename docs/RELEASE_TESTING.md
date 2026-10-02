# Release testing — 1.2.0

The release gate covers DSP correctness, API validation/recovery, packaging and browser behavior.

## Automated backend suite

Run:

```bash
python -m pytest
```

The current 189-test regression suite covers every `WaveformKind`, representative blind-classifier families, spectrum and dBm calibration, receiver synchronization/demodulation, OFDM, radar processing, programmable expressions/extensions, generic TX/RX framing, WebSocket recovery and release-specific span/zero-span regressions.

## Browser UI smoke

Run:

```bash
python -m pip install -e '.[ui-test]'
python -m playwright install chromium
python tests/ui/ui_smoke.py
```

This deterministic smoke test loads the real HTML/CSS/JavaScript in Chromium with mocked API/WebSocket transport. It verifies that a typed 50 MHz span clamps to 10 MHz, the client view becomes −5 MHz…+5 MHz even with a 1 MS/s source, the wide-span renderer accepts narrower source data without page errors, the Help button is present and zero-span polling runs at the release cadence.

`tests/ui/live_smoke.cjs` remains the live-server Playwright test. Run `npm install`, start VectorRF Lab, then run `npm run test:ui:live`.

## Packaging gate

Build a wheel with:

```bash
python -m pip wheel . --no-deps -w dist
```

Install the wheel into an isolated target and verify `vectorrf.app` can serve `/`, `/help`, `/assets/app.js`, `/api/health` and `/api/catalog`. This specifically catches missing packaged web assets.

## Performance sanity

`examples/benchmark.py` exercises broader DSP paths. For release 1.2, the live spectrum transport is capped at 2,048 display points and zero span at 900 points; these are display reductions only and do not lower the sample rate used for DSP measurements.
