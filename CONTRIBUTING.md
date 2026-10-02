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

For the deterministic browser smoke test:

```bash
python -m pip install -e '.[ui-test]'
python -m playwright install chromium
python tests/ui/ui_smoke.py
```

## Before opening a pull request

- Keep changes focused and explain the user-visible or numerical reason for them.
- Preserve existing behavior unless the pull request intentionally changes it.
- Add or update regression tests for DSP, API, protocol, or UI behavior changes.
- Document numerical assumptions, units, estimator limitations, and performance tradeoffs where relevant.
- Run `python -m pytest` locally.
- If browser behavior changed, run `python tests/ui/ui_smoke.py` as well.
- Avoid committing generated caches, virtual environments, build artifacts, or local screenshots.

## DSP and numerical changes

For a DSP change, include enough information for reviewers to reproduce the result: signal configuration, sample rate, expected metric/tolerance, and why the tolerance is appropriate. Do not silently normalize, clamp, or reinterpret user inputs unless the behavior is explicitly documented and tested.

## Extensions

Example extensions belong in `extensions/`. Treat extension code as trusted executable Python. Do not add network download/execution behavior to the extension loader without a separate security review.

## Issues

A useful bug report includes the VectorRF Lab version, Python version, operating system, exact configuration or steps, expected behavior, actual behavior, and a minimal reproduction when possible. For security-sensitive reports, follow `SECURITY.md` instead of posting exploit details publicly.
