"""Engineering notation for messages, matching the UI's number fields."""
from __future__ import annotations

import math

_PREFIXES = [(1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k"), (1.0, ""), (1e-3, "m"), (1e-6, "µ"), (1e-9, "n"), (1e-12, "p")]


def eng(value: float, unit: str = "Hz", digits: int = 6) -> str:
    """``value`` with an SI prefix and at most ``digits`` significant digits.

    >>> eng(5e6), eng(-12_500.0), eng(1e7, 'S/s'), eng(2.5e-5, 's')
    ('5 MHz', '-12.5 kHz', '10 MS/s', '25 µs')
    """
    value = float(value)
    if not math.isfinite(value):
        return f"{value} {unit}".strip()
    if value == 0:
        return f"0 {unit}".strip()
    value = float(f"{value:.{digits}g}")  # round first so 999.9999 kHz prints as 1 MHz
    magnitude = abs(value)
    scale, prefix = next(((s, p) for s, p in _PREFIXES if magnitude >= s * (1 - 5e-13)), _PREFIXES[-1])
    text = f"{value / scale:.{digits}g}"
    if "e" in text:  # beyond the prefix table
        return f"{value:.{digits}g} {unit}".strip()
    return f"{text} {prefix}{unit}".strip()
