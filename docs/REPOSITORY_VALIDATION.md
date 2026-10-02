# Repository cleanup validation

This document records the validation performed while preparing VectorRF Lab 1.2.0 for public GitHub use. The cleanup was intentionally limited to repository structure, packaging metadata, documentation, contribution/CI scaffolding, and tests. Runtime DSP/application behavior was not intentionally changed.

## No-runtime-change check

Before the cleanup, SHA-256 hashes were recorded for every non-cache file under `backend/vectorrf/`, including the packaged HTML/CSS/JavaScript assets. The same set was hashed after the cleanup; the manifests were identical.

The removed repository-level `web/` directory was a byte-for-byte duplicate of `backend/vectorrf/web/` for all six shipped assets (`app.js`, `studio.js`, `workbench.js`, `styles.css`, `index.html`, and `help.html`). Removing that duplicate leaves the packaged assets as the single source of truth without changing the bytes served by the application.

## Automated regression results

- Before cleanup: **187 tests passed**.
- After cleanup: **189 tests passed**.
- The two added tests only verify that browser assets are present in the Python package and served by the FastAPI application.
- Deterministic Chromium UI smoke test: **passed**, including spectrum-span clamping, requested view bounds, zero-span cadence, and no page errors.
- JavaScript syntax checks for `app.js`, `studio.js`, `workbench.js`, and the live Playwright harness: **passed**.

## Packaging validation

A wheel was built from the cleaned source tree and installed into an isolated target directory. From outside the repository, the installed wheel successfully served:

- `/`
- `/help`
- `/assets/app.js`
- `/assets/styles.css`
- `/api/health`
- `/api/catalog`

The installed package reported version `1.2.0`.

## What changed structurally

- Tests moved from `backend/tests/` and `web/tests/` into top-level `tests/` and `tests/ui/`.
- Revision-number test filenames were renamed by behavior/domain.
- Duplicate root web assets were removed.
- Package metadata, optional test dependencies, GitHub Actions CI, issue/PR templates, ignore/line-ending/editor files, contributing/support/security documentation, and packaging tests were added or improved.
- README and selected docs were reorganized/clarified.

None of these changes require a user to re-validate DSP behavior solely because of the cleanup; the existing automated suite plus byte-identical runtime-source check covers that concern.
