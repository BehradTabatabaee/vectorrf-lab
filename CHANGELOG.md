# Changelog

## Unreleased — repository cleanup

- Reorganized tests into a conventional top-level `tests/` tree and renamed revision-number test files by behavior.
- Removed the duplicate root web bundle; the packaged `backend/vectorrf/web/` assets are now the single source of truth.
- Added packaging/static-asset regression tests, GitHub Actions CI, contribution templates, repository hygiene files, and open-source contributor/support documentation.
- Expanded package metadata and separated optional browser-test dependencies.
- Reworked the README around setup, supported features, explicit non-features, architecture, validation, security, and inspiration.
- Runtime DSP/application source files were intentionally left unchanged.

## 1.2.0

- Made requested Spectrum span control the displayed start/stop frequencies up to 10 MHz. When the source sample rate is narrower, unavailable regions outside source Nyquist are shown explicitly rather than silently shrinking the axis.
- Oversized typed spectrum spans now clamp safely to 10 MHz in the UI; direct API clients continue to receive validation errors for unsupported values.
- Reduced live spectrum display payloads to a peak-preserving maximum of 2,048 points and disabled unused hold traces by default.
- Reduced zero-span display payloads to 900 points and increased the zero-span refresh target to 12 Hz; modulation-quality polling remains intentionally slower.
- Added an in-app Help button and standalone web Help page covering spectrum semantics, markers/zoom, zero span, absolute dBm power, TX/RX framing, receiver workflows, radar, performance and troubleshooting.
- Added release regression tests for 10 MHz wide-span behavior, source-Nyquist limits, zero-span accuracy at 10 MS/s, browser payload bounds, 50 MHz validation recovery and Help-page publication.
- Added installable `vectorrf-lab` CLI entry point and packaged web assets so wheels remain self-contained.

## 1.1.0

- Added 10 MHz maximum spectrum span, baseband-only analyzer operation, absolute dBm transmitter calibration, spectrum markers/zoom and generic mixed-modulation TX/RX framing.
- Fixed record-boundary spectrum discontinuities and modulation-quality handling for basic signals.
