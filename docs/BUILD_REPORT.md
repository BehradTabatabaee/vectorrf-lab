# Build report — 1.2

Version 1.2 is the publication-hardening release. It retains the 1.1 DSP/framing feature set and closes the remaining spectrum/zero-span performance and usability issues while making the installable package self-contained.

## Spectrum and zero span

- Communications spectrum remains fixed at baseband DC with a 10 MHz maximum requested span.
- The requested span now owns the displayed horizontal axis. A 10 MHz span displays −5 MHz…+5 MHz even if the active source is sampled more narrowly. Regions outside the source Nyquist interval are shaded and labeled unavailable; no spectrum values are fabricated there.
- Typed out-of-range spectrum spans use product-level clamping. For example, 50,000,000 Hz becomes 10,000,000 Hz in the UI and the live stream remains valid. Direct API clients still receive a normal validation error for unsupported values.
- Live spectrum frames are peak-preserving reduced to at most 2,048 display points. Only the live write trace is enabled by default; hold/average traces remain available on demand.
- Zero span keeps its phase/frequency calculations at acquisition rate, but the browser payload is reduced to 900 points. The zero-span page targets 12 updates/s while modulation-quality polling remains at 4 updates/s.
- Spectrum marker, delta, zoom/pan, memory, absolute-dBm calibration and record-boundary fixes from 1.1 remain intact.

## Help and product packaging

- The application header now includes a Help button that opens a dedicated web user guide covering spectrum semantics, absolute power, frame building/acquisition, receiver workflows, radar, performance and troubleshooting.
- `vectorrf-lab` is an installable command-line entry point.
- Web assets are packaged inside the Python package and are the single source of truth, so an installed wheel can serve the UI without a duplicate repository-level web bundle.
- Release notes, security guidance and a release-testing document are included.

## Validation

- The cleaned repository regression suite passes with 189 tests (the original 187 release checks plus two packaging/static-asset checks), including 1.2 checks for 10 MHz full-span data, narrower-source Nyquist behavior, 2,048-point spectrum transport, 900-point zero-span output, instantaneous-frequency accuracy at 10 MS/s, Help publication and recovery after an invalid 50 MHz direct-WebSocket edit.
- `app.js`, `studio.js` and `workbench.js` pass Node syntax checks.
- A Chromium UI smoke test loads the actual product HTML/CSS/JavaScript with deterministic mocked transport, verifies 50 MHz → 10 MHz clamping, checks −5 MHz…+5 MHz client bounds with a 1 MS/s source, renders the Nyquist-limited wide-span spectrum without page errors and confirms the 12 Hz zero-span scheduler.
- The live-server Node Playwright test remains available for environments with Node Playwright installed.
