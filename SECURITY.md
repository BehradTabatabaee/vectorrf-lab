# Security policy

## Supported version

Security fixes are expected to target the latest published release and the default branch.

## Reporting a vulnerability

Do not publish secrets, exploit code, or detailed reproduction steps for an unpatched vulnerability in a public issue.

If GitHub Private Vulnerability Reporting is enabled for the repository, use it. Otherwise, open a minimal public issue asking for a private maintainer contact path without including sensitive details.

Please include the affected version, impact, prerequisites, and a minimal private reproduction where possible.

## Deployment model

VectorRF Lab is designed as a local laboratory application. The default launch scripts and `vectorrf-lab` command bind to `127.0.0.1`. The application does not provide authentication or TLS; do not expose it directly to an untrusted network without adding your own access controls and transport security.

The waveform/processing extension hooks execute local Python modules. Only install or load extensions you trust. Treat repository files and extension modules as executable code.
