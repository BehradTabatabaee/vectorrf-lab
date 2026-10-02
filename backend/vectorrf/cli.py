from __future__ import annotations

import os


def main() -> None:
    """Launch VectorRF Lab from an installed package."""
    import uvicorn

    host = os.environ.get("VECTORRF_HOST", "127.0.0.1")
    port = int(os.environ.get("VECTORRF_PORT", "8765"))
    uvicorn.run("vectorrf.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
