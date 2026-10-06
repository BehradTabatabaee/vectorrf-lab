"""Readable validation messages.

Pydantic's own text ("2 validation errors for TxConfig components.1 Value
error, … [type=value_error, input_value={…}]") names internal paths and dumps
the input.  ``describe`` turns a validation error into one sentence per
problem that names the control and the signal it belongs to, e.g.
"Signal 2 · Power: must be at least -120".
"""
from __future__ import annotations

import re
from typing import Any, Iterable

from pydantic import ValidationError

from .units import eng

# Field names as the UI labels them.
LABELS = {
    "sample_rate": "Sample rate", "duration": "Capture length", "power_dbm": "Power", "center_hz": "Frequency offset",
    "snr_db": "SNR", "seed": "Seed", "rf_center_hz": "RF centre", "agility": "Hopping / sweep", "components": "Components",
    "label": "Name", "message": "Message", "custom_bits": "Bits", "pn_order": "PRBS order", "pn_bits": "Repeat length",
    "frame_blocks": "Frame blocks", "preamble_bits": "Preamble bits", "payload_bits": "Payload bits",
    "gaussian_bt": "Gaussian BT", "apsk_ratio2": "Ring 2 / ring 1", "apsk_ratio3": "Ring 3 / ring 1",
    "burst_period_s": "Burst period", "burst_duty": "Burst duty", "noise_bandwidth_hz": "Noise bandwidth",
    "iq_gain_db": "I/Q gain imbalance", "iq_phase_deg": "Quadrature error", "dc_i": "DC I", "dc_q": "DC Q",
    "symbol_rate": "Symbol rate", "rolloff": "Roll-off", "filter_span_symbols": "Filter span",
    "fsk_deviation_hz": "Peak deviation", "phase_offset_deg": "Phase", "message_freq_hz": "Message tone",
    "modulation_index": "AM index", "fm_deviation_hz": "FM deviation", "pm_deviation_rad": "PM deviation",
    "tones_hz": "Tone frequencies", "chirp_start_hz": "Chirp start", "chirp_stop_hz": "Chirp stop",
    "ofdm_fft": "FFT size", "ofdm_cp": "Cyclic prefix", "ofdm_occupied": "Occupied subcarriers", "ofdm_design": "OFDM design",
    "pulse_width_s": "Pulse width", "pri_s": "PRI", "sweep_bw_hz": "Sweep bandwidth", "start_hz": "Start", "stop_hz": "Stop",
    "step_hz": "Step", "frequencies_hz": "Hop frequencies", "dwell_s": "Dwell", "off_s": "Off-time", "ramp_s": "Edge ramp",
    "offset_s": "Start delay", "span_hz": "Span", "rbw_hz": "RBW", "averaging": "Averaging", "max_fft": "FFT limit",
    "channels": "Channel bands", "width_hz": "Width", "timing_gain": "Timing loop gain", "timing_phase": "Timing phase",
    "tuner_hz": "Tuner", "max_symbols": "Symbols analysed", "equalizer_gain": "Equalizer step", "equalizer_taps": "Equalizer taps",
    "freq_offset_hz": "Carrier offset", "phase_offset_deg_channel": "Phase offset", "timing_offset_samples": "Timing offset",
    "echo_gain": "Echo gain", "echo_delay_symbols": "Echo delay", "capture_samples": "Window", "step_samples": "Step",
    "start_sample": "Start sample", "header_bytes": "Header bytes", "power_offset_db": "Power offset",
    "pacing": "Pacing", "fps": "Update target", "duration_s": "Duration", "headroom_db": "Headroom",
    "modulation": "Modulation", "waveform": "Waveform", "costas": "Carrier loop", "alpha": "α", "beta": "β",
    "dehop": "De-hopping", "frame_enabled": "Frame builder", "bit_count": "Bit count", "repeat": "Repeat",
}

_PYDANTIC = [
    (r"^Value error, ", ""),
    (r"^Assertion failed, ", ""),
    (r"^Input should be greater than or equal to (.+)$", r"must be at least \1"),
    (r"^Input should be less than or equal to (.+)$", r"must be at most \1"),
    (r"^Input should be greater than (.+)$", r"must be greater than \1"),
    (r"^Input should be less than (.+)$", r"must be less than \1"),
    (r"^Input should be a finite number$", "must be a finite number"),
    (r"^Input should be a valid (number|integer|boolean|string|list|dictionary).*$", r"must be a valid \1"),
    (r"^Input should be (.+)$", r"must be \1"),
    (r"^Field required$", "is required"),
    (r"^List should have at most (\d+) items? after validation.*$", r"accepts at most \1 entries"),
    (r"^List should have at least (\d+) items? after validation.*$", r"needs at least \1 entries"),
    (r"^String should have at most (\d+) characters?$", r"accepts at most \1 characters"),
    (r"^String should have at least (\d+) characters?$", r"needs at least \1 characters"),
    (r"^Extra inputs are not permitted$", "is not a known setting"),
]


def unit_of(field: str | None) -> str:
    """Display unit of a model field, from its name."""
    if not field:
        return ""
    special = {"sample_rate": "S/s", "symbol_rate": "Bd", "pmcw_chip_rate": "chip/s"}
    if field in special:
        return special[field]
    for suffix, unit in (("_hz", "Hz"), ("_s", "s"), ("_db", "dB"), ("_dbm", "dBm"), ("_deg", "°")):
        if field.endswith(suffix):
            return unit
    return ""


def message(text: str, field: str | None = None) -> str:
    """A pydantic message as a readable clause ("must be at least 0"); bounds of
    ``field`` are shown in its unit with an SI prefix (5 GHz, not 5000000000)."""
    for pattern, replacement in _PYDANTIC:
        text = re.sub(pattern, replacement, text)
    unit = unit_of(field)
    if unit:
        text = re.sub(r"^(must be (?:at least|at most|greater than|less than)) (-?[0-9.]+(?:e[+-]?\d+)?)$",
                      lambda m: f"{m.group(1)} {eng(float(m.group(2)), unit)}", text)
    return text


def _where(loc: Iterable[Any], data: Any, last_input: Any = None) -> str:
    """Human path: list items of ``components`` are named after their label."""
    parts, node, previous = [], data, None
    loc = list(loc)
    for position, key in enumerate(loc):
        if key in ("body", "query", "path") and not parts and previous is None:
            continue
        if isinstance(key, int):
            item = node[key] if isinstance(node, list) and 0 <= key < len(node) else None
            if item is None and position == len(loc) - 1:
                item = last_input
            if previous == "components":
                parts[-1:] = [(item.get("label") if isinstance(item, dict) else None) or f"Signal {key + 1}"]
            elif previous in ("tones_hz", "frequencies_hz"):
                parts.append(f"#{key + 1}")
            else:
                parts.append(f"{key + 1}")
            node = item
        elif isinstance(key, str):
            if key.startswith("function-") or key in ("function-after", "function-before", "function-wrap"):
                continue
            parts.append(LABELS.get(key, key.replace("_", " ")))
            node = node.get(key) if isinstance(node, dict) else None
        previous = key
    return " · ".join(parts)


def errors_of(exc: ValidationError | Exception) -> list[dict]:
    if isinstance(exc, ValidationError):
        return exc.errors(include_url=False)
    errors = getattr(exc, "errors", None)
    return errors() if callable(errors) else []


def describe(exc: Exception, data: Any = None, prefix: str = "") -> str:
    """One readable sentence per problem in ``exc`` (a pydantic or plain error)."""
    details = errors_of(exc)
    if not details:
        text = str(exc)
        return f"{prefix}: {text}" if prefix else text
    lines = []
    for e in details:
        where = _where(e.get("loc", ()), data, e.get("input"))
        where = " · ".join(p for p in (prefix, where) if p)
        field = next((k for k in reversed(e.get("loc", ())) if isinstance(k, str)), None)
        text = message(str(e.get("msg", "")), field)
        line = f"{where}: {text}" if where else text
        if line not in lines:
            lines.append(line)
    return "; ".join(lines)
