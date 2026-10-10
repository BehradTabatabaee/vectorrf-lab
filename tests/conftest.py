"""Shared pytest hooks."""
import os


def _escape(text: str, prop: bool = False) -> str:
    """Escape text for a GitHub Actions workflow command (a message, or a property value)."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if prop else text


def pytest_runtest_logreport(report):
    """On GitHub Actions, report each failure as an annotation on the run.

    Annotations appear on the run's summary page, next to the line that
    failed, without opening the job log.
    """
    if os.environ.get("GITHUB_ACTIONS") != "true" or not report.failed:
        return
    path, line, _ = report.location
    lines = (report.longreprtext or "").strip().splitlines()
    message = "\n".join(lines[-15:])[:2000]
    title = _escape(report.nodeid, prop=True)
    print(f"\n::error file={_escape(path, prop=True)},line={(line or 0) + 1},title={title}::{_escape(message)}", flush=True)
