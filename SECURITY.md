# Security policy

## Supported version

Security fixes are expected to target the latest published release and the default branch.

## Reporting a vulnerability

Do not publish secrets, exploit code, or detailed reproduction steps for an unpatched vulnerability in a public issue.

If GitHub Private Vulnerability Reporting is enabled for the repository, use it. Otherwise, open a minimal public issue asking for a private maintainer contact path without including sensitive details.

Please include the affected version, impact, prerequisites, and a minimal private reproduction where possible.

## Deployment model

VectorRF Lab is designed as a local laboratory application. The default launch scripts and `vectorrf-lab` command bind to `127.0.0.1`. The application does not provide authentication or TLS; do not expose it directly to an untrusted network without adding your own access controls and transport security.

The live WebSocket (`/ws/live`) refuses connections whose `Origin` is not the server's own, so a web page from another site cannot drive a local server through the browser. Waveform expressions are parsed into a syntax tree and evaluated with arithmetic, comparisons and a fixed list of NumPy functions only, with bounded size, exponents and values; they never reach Python's `eval`.

The waveform/processing extension hooks execute local Python modules. Only install or load extensions you trust. Treat repository files and extension modules as executable code.

The recording API reads, writes and deletes files only inside the capture library folder (`VECTORRF_CAPTURES_DIR`) and selects them by plain file name; names with directories, hidden files or unsupported extensions are rejected. Uploads are size-limited (`VECTORRF_MAX_UPLOAD_GB`). Anyone who can reach the server can upload into and delete from that folder, which is another reason not to expose it to untrusted networks.
