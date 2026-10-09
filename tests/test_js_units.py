"""Runs the Node unit tests of the browser's pure modules (tests/js/*.test.mjs)."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB_JS = ROOT / "backend" / "vectorrf" / "web" / "js"
SUITES = sorted((ROOT / "tests" / "js").glob("*.test.mjs"))


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("suite", SUITES, ids=[s.name for s in SUITES])
def test_javascript_units(suite):
    result = subprocess.run(["node", "--no-warnings", str(suite), str(WEB_JS)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "passed" in result.stdout
