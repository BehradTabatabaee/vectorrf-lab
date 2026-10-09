# Testing

VectorRF Lab is tested at three levels: a Python suite that checks the DSP against analytic references and exercises the API and live engine, unit tests of the browser's pure modules, and an end-to-end browser test that drives the real application. CI runs all three on every push (see `.github/workflows/ci.yml`).

## Python suite

```bash
python -m pip install -e '.[dev]'
python -m pytest
```

About 1,180 tests run in a few minutes on a desktop. They are grouped by subject:

| File | What it checks |
| --- | --- |
| `test_math_audit.py` | Radar code definitions and sidelobes (Barker, Frank, P1–P4, Zadoff–Chu, Costas), Gray mappings and unit-power constellations, RRC unit energy and Nyquist property, window ENBW, transmit-power calibration with noise and bursts, spectrum levels for short captures and between FFT bins, noise PSD and integrated-noise calibration, interpolated bandwidths, exact long-time phase, capture continuity, FM/SSB closed forms, CPFSK alignment, frame-block alignment, CRC-16/CCITT-FALSE, unbiased EVM |
| `test_stream_model.py` | The transmitter as one stream: consecutive captures join exactly for every waveform, capture length never changes the stream, noise statistics, long-run and on-power calibration, pulses cut by capture edges, jittered PRIs, continuous-phase chirps, text decoding from captures that start mid-stream |
| `test_agility_composite.py` | Hop grids, lists and seeded permutations; DDS phase exactness and continuity for continuous, coherent and swept schedules; gating and ramps; composite powers (independent and coherent components), seeds, scale and Nyquist validation; design plans; every example design in `examples/designs` |
| `test_receiver_robustness.py` | Every linear format locks on every live capture with ±5 kHz carrier offsets at 50 kBd, with PRBS and with a repeating text payload; dense constellations acquire cold; FSK/MSK/GMSK decode captures starting mid-symbol with carrier offsets; pilotless OFDM; sub-symbol captures give valid JSON; 40,000 samples per symbol; modulation-quality accumulation and payload size; reference-aided channel estimate and a converging CMA equalizer |
| `test_receiver_matrix.py` | Linear formats at 1, 10 and 100 MS/s with carrier and phase offsets; tracking across skipped captures; zero span tuning, clipping and Nyquist limits |
| `test_instrument_upgrades.py`, `test_instruments.py`, `test_spectrum_rx.py`, `test_receiver_workflows.py` | Zoom FFT, channel power, loop gains and settling, de-hopping, tuned zero span, classifier traces and band isolation, radar design figures, SigMF design embedding, display rounding, OFDM pilot interpolation and timing |
| `test_radar_physics.py` | Kinematics, Doppler signs, range and velocity accuracy, resolved targets, every CFAR detector's false-alarm rate (Monte Carlo) and detections with velocities, angle estimates, the radar equation and thermal noise, leakage and clutter, Swerling I, band-limited delayed records |
| `test_files.py` | Every sample format, WAV/RF64/SigMF readers and writers, raw-format sniffing, the library API, overview, calibration, export round trips |
| `test_wideband.py` | Scale invariance at 2–5 GS/s, exact phase far into simulated time, GHz mixers and resamplers, a 1 GHz FMCW radar, wideband export |
| `test_capture_identity.py`, `test_live_engine.py` | Capture identity across observers, real-time and every-sample pacing, dropped work and its reporting, ordered products, product fault recovery |
| `test_api.py`, `test_api_fuzz.py` | Endpoints, defaults, the WebSocket origin check, and hundreds of malformed requests that must return readable 422s rather than errors |
| `test_blind.py` | Classifier regression matrices |
| `test_programmable.py`, `test_processing_and_streaming.py` | Expressions, extensions, processing chains |
| `test_packaging.py`, `test_docs.py`, `test_js_units.py` | Every browser module is packaged and its imports resolve; documentation examples validate; the browser unit tests (below) |

Every behaviour change in the DSP comes with a test that reproduces the defect against an analytic or noise-free reference.

## Browser unit tests

`tests/js/units.test.mjs` tests the browser's pure modules (engineering-notation parsing and formatting, request construction) with Node's built-in test runner. `test_js_units.py` runs it when Node is installed:

```bash
node --test tests/js/units.test.mjs
```

## End-to-end browser test

```bash
python -m pip install -e '.[ui-test]'
python -m playwright install chromium
python tests/ui/ui_smoke.py
```

Set `VECTORRF_BROWSER_PATH` to use an installed Chromium-based browser (Microsoft Edge or Google Chrome) instead of Playwright's, and `VECTORRF_SCREENSHOTS` to a folder to keep a screenshot of each step.

The test starts the real application on a free port with a temporary recording library (holding a generated SigMF recording), then drives the interface as a person would and fails on any page error, console error or HTTP error:

- **Transmitter**: the logo and the default PN15 QPSK; engineering-notation entry; a composite with an added component (total power checked); a duplicated component placed off its original and moved by typing, followed in the live spectrum; lowercase `mhz` and the out-of-band refusal; stepper size from 0 Hz; a 900 MHz – 1 GHz hopping plan entered as absolute frequencies.
- **Spectrum**: live frames, markers, band selection, a channel-power window, wheel zoom into a down-converted span, zero span following consecutive captures and refusing an out-of-band centre, modulation quality building its record.
- **Receiver**: demodulation below 5 % EVM with the decoded text, the follow badge, the equalizer page's channel estimate, the live monitor and a processing chain, OFDM coarse CFO.
- **Classifier**: a verdict with exactly one accepted stage in its decision path.
- **Recordings**: a recording as the live source, its RF centre in Spectrum.
- **Radar Studio**: captures advancing, target and velocity drags, field edits, every detector mode, pulse-Doppler processing, Pause and Step, the 12-target limit.
- **Resume and layout**: displays keep updating after the page is hidden and shown; no horizontal overflow at 1920, 1280, 900 and 760 px; the compact layout and help page scroll.

## Packaging

```bash
python -m pip install build
python -m build
```

CI installs the wheel into a clean environment, starts the server from outside the repository and checks that it serves the interface, the help page, the stylesheet and every JavaScript module (with a JavaScript content type and `Cache-Control: no-cache`), `/api/health`, `/api/catalog` and `/api/defaults`.

## Performance

`examples/benchmark.py` times generation, spectrum, receiver and classifier at 1 MS/s, 5 MS/s and 2 GS/s, spectrum frames up to 1 M points, recording decoding, hop and sweep oscillators, a composite and the radar simulator. [BENCHMARK.md](BENCHMARK.md) lists the results and the live update rates measured in a browser.
