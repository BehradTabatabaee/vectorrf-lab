# Contributing to VectorRF Lab

Thanks for considering a contribution. VectorRF Lab is a numerical RF/DSP laboratory, so correctness, reproducibility, and clear model boundaries matter as much as UI polish.

## Development setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

For the end-to-end browser test (it starts the application itself):

```bash
python -m pip install -e '.[ui-test]'
python -m playwright install chromium
python tests/ui/ui_smoke.py
```

Set `VECTORRF_BROWSER_PATH` to an installed Chromium-based browser (Chrome, Edge) to skip the Playwright browser download. [docs/TESTING.md](docs/TESTING.md) describes the suites.

## Before opening a pull request

- Keep changes focused and explain the user-visible or numerical reason for them.
- Preserve existing behavior unless the pull request intentionally changes it.
- Add or update regression tests for DSP, API, protocol, or UI behavior changes.
- Document numerical assumptions, units, estimator limitations, and performance tradeoffs where relevant.
- Run `python -m pytest` locally.
- If browser behavior changed, run `python tests/ui/ui_smoke.py` as well.
- Avoid committing generated caches, virtual environments, build artifacts, or local screenshots.

## DSP and numerical changes

For a DSP change, include enough information for reviewers to reproduce the result: signal configuration, sample rate, expected metric/tolerance, and why the tolerance is appropriate. Prefer tests that pin an analytic property (a code's sidelobe level, a calibration identity, scale invariance across sample rates) over snapshots of current output. Do not silently normalize, clamp, or reinterpret user inputs unless the behavior is explicitly documented and tested.

Limits are expressed in samples, not seconds or hertz, and estimators should be invariant to the sample rate (`tests/test_wideband.py` checks this). Use `dsp.phasor` for carrier phase over absolute time.

## Browser code

The UI in `backend/vectorrf/web/js/` is plain ES modules with no build step or bundler; see the browser section of `docs/ARCHITECTURE.md`. In short:

- one module per concern: `core/` (state, controls, formatting, transport), `plots/`, `editors/` (reusable editors), `workspaces/` (one per tab);
- workspaces do not import each other (shared pieces live in `workspaces/common.js`); they communicate through `core/state.js` and the event bus in `core/events.js`;
- declare controls with `numberField`, `selectField` … from `core/controls.js` (`data-bind` paths) rather than wiring listeners by hand, and format numbers with `core/format.js`;
- defaults belong to the Python models and reach the browser through `/api/defaults`; do not duplicate them in JavaScript.

`tests/test_packaging.py` resolves every relative import, so a new module needs no registration. Extend `tests/ui/ui_smoke.py` when a workflow changes.

## Documentation and screenshots

User-facing behaviour is described in the in-app guide (`backend/vectorrf/web/help.html`); models and estimators in `docs/DSP_NOTES.md`; APIs in `docs/PROGRAMMING.md`. `tests/test_docs.py` validates the guide's JSON examples and every relative link in the Markdown files.

The README screenshots are generated, not taken by hand: `python scripts/screenshots.py` (with the `ui-test` extra) starts the application with a temporary recording library and synthetic signals and rewrites `docs/screenshots/`. Never capture your own recordings or folders in screenshots.

## Extensions

Example extensions belong in `extensions/`. Treat extension code as trusted executable Python. Do not add network download/execution behavior to the extension loader without a separate security review.

## Issues

A useful bug report includes the VectorRF Lab version, Python version, operating system, exact configuration or steps, expected behavior, actual behavior, and a minimal reproduction when possible. A short SigMF export (**Transmitter → Export I/Q file**) embeds the transmitter design and reproduces a signal exactly; so does a design JSON like those in `examples/designs/`. For security-sensitive reports, follow `SECURITY.md` instead of posting exploit details publicly.
