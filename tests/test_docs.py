"""Documented examples must stay valid as the models evolve."""
import json
import re
from pathlib import Path

import pytest

from vectorrf.models import TxConfig

DOCS = Path(__file__).resolve().parents[1] / "docs"
JSON_BLOCK = re.compile(r"```json\n(.*?)```", re.S)


def _design_examples():
    text = (DOCS / "PROGRAMMING.md").read_text(encoding="utf-8")
    blocks = [json.loads(block) for block in JSON_BLOCK.findall(text) if '"..."' not in block]
    designs = [b for b in blocks if isinstance(b, dict) and "waveform" in b]
    designs += [b["tx"] for b in blocks if isinstance(b, dict) and isinstance(b.get("tx"), dict)]
    return designs


def test_programming_guide_has_design_examples():
    assert len(_design_examples()) >= 3


@pytest.mark.parametrize("design", _design_examples(), ids=lambda d: d.get("waveform", "?"))
def test_documented_designs_validate(design):
    TxConfig.model_validate(design)

