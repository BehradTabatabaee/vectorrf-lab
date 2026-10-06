"""Nominal occupied bands of transmitter designs.

These are textbook estimates used to draw component bands, plan hopping and
warn about aliasing.  They are not measurements: Spectrum reports measured
99 % and channel bandwidths.
"""
from __future__ import annotations

from .agility import frequency_extent
from .models import RADAR_WAVEFORMS, TxConfig, WaveformKind
from .units import eng

_LINEAR = {"bpsk", "qpsk", "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk", "8psk", "16qam", "64qam", "16apsk", "32apsk"}


def _symbol_rate(cfg: TxConfig, rate: float | None = None) -> float:
    """Symbol rate actually generated (integer samples per symbol)."""
    rate = float(rate or cfg.symbol_rate)
    return cfg.sample_rate / max(2, round(cfg.sample_rate / rate))


def content_band(cfg: TxConfig) -> tuple[float, float] | None:
    """Occupied band of the modulated content around its carrier, or ``None`` if unknown."""
    w = cfg.waveform.value
    fs = cfg.sample_rate
    if w == "composite":
        bands = [nominal_band(c) for c in cfg.components if c.enabled]
        if any(b is None for b in bands) or not bands:
            return None
        return min(b[0] for b in bands), max(b[1] for b in bands)
    if cfg.frame_enabled and cfg.frame_mode == "blocks" and cfg.frame_blocks:
        rs = max(_symbol_rate(cfg, b.symbol_rate) for b in cfg.frame_blocks)
        half = rs * (1 + cfg.rolloff) / 2 if cfg.pulse_shape == "rrc" else rs
        return -half, half
    if w in {"tone", "radar-cw"}:
        return 0.0, 0.0
    if w == "multitone":
        return (min(cfg.tones_hz), max(cfg.tones_hz)) if cfg.tones_hz else (0.0, 0.0)
    if w == "awgn":
        half = min(cfg.noise_bandwidth_hz or fs, fs) / 2
        return -half, half
    if w == "chirp":
        return min(cfg.chirp_start_hz, cfg.chirp_stop_hz), max(cfg.chirp_start_hz, cfg.chirp_stop_hz)
    if w in _LINEAR:
        rs = _symbol_rate(cfg)
        # RRC occupies Rs(1+β); a rectangular pulse is reported by its main lobe.
        half = rs * (1 + cfg.rolloff) / 2 if cfg.pulse_shape == "rrc" else rs
        return -half, half
    if w in {"2fsk", "4fsk"}:
        half = cfg.fsk_deviation_hz + _symbol_rate(cfg)  # Carson's rule
        return -half, half
    if w in {"msk", "gmsk"}:
        half = (0.59 if w == "msk" else 0.46) * _symbol_rate(cfg)  # 99 % power bandwidth
        return -half, half
    if w == "ofdm":
        from .ofdm import layout
        _, active, _, _ = layout(cfg.ofdm_fft, cfg.ofdm_cp, cfg.ofdm_occupied, cfg.ofdm_design)
        spacing = fs / cfg.ofdm_fft
        return (min(active) - 0.5) * spacing, (max(active) + 0.5) * spacing
    fm = cfg.message_freq_hz
    if w in {"am", "dsb-sc"}:
        return -fm, fm
    if w == "ssb":
        return (0.0, fm) if cfg.sideband == "usb" else (-fm, 0.0)
    if w == "fm":
        half = cfg.fm_deviation_hz + fm
        return -half, half
    if w == "pm":
        half = (cfg.pm_deviation_rad + 1) * fm
        return -half, half
    if cfg.waveform in RADAR_WAVEFORMS:
        return _radar_band(cfg)
    return None


def _radar_band(cfg: TxConfig) -> tuple[float, float] | None:
    w = cfg.waveform
    if w in {WaveformKind.LFM, WaveformKind.NLFM, WaveformKind.FMCW}:
        return -cfg.sweep_bw_hz / 2, cfg.sweep_bw_hz / 2
    if w == WaveformKind.COSTAS:
        return -cfg.sweep_bw_hz / 2, cfg.sweep_bw_hz / 2
    if w == WaveformKind.STEPPED_FREQ:
        half = (cfg.stepped_steps - 1) * cfg.stepped_spacing_hz / 2 + cfg.stepped_steps / cfg.pulse_width_s
        return -half, half
    if w == WaveformKind.PULSE:
        return -1 / cfg.pulse_width_s, 1 / cfg.pulse_width_s
    if w == WaveformKind.PMCW:
        return -cfg.pmcw_chip_rate, cfg.pmcw_chip_rate
    if w == WaveformKind.ARBITRARY_RADAR:
        f = cfg.arbitrary_start_freq_hz + cfg.arbitrary_stop_freq_hz
        return (min(f), max(f)) if f else (0.0, 0.0)
    chips = {WaveformKind.BARKER: cfg.barker_length, WaveformKind.FRANK: cfg.polyphase_order ** 2,
             WaveformKind.P1: cfg.polyphase_order ** 2, WaveformKind.P2: cfg.polyphase_order ** 2,
             WaveformKind.CUSTOM_PHASE: max(2, len(cfg.custom_phase_deg))}.get(w, cfg.phase_code_chips)
    half = chips / cfg.pulse_width_s
    return -half, half


def nominal_band(cfg: TxConfig) -> tuple[float, float] | None:
    """Occupied band in absolute baseband Hz, including offset and agility."""
    band = content_band(cfg)
    if band is None:
        return None
    a, b = frequency_extent(cfg.agility)
    return cfg.center_hz + band[0] + a, cfg.center_hz + band[1] + b


def design_plan(cfg: TxConfig, max_segments: int = 1024) -> dict:
    """Everything the transmitter UI draws for a design, without generating samples.

    ``config`` is the validated, defaults-filled design (what a saved design
    file is normalized to on import).
    """
    import numpy as np
    from .agility import summary
    from .models import carrier_extent
    from .signal import capture_samples
    composite = cfg.waveform == WaveformKind.COMPOSITE
    components = []
    for i, c in enumerate(cfg.components if composite else []):
        band = nominal_band(c)
        components.append({
            "index": i, "label": c.label or f"Component {i + 1}", "enabled": c.enabled, "waveform": c.waveform.value,
            "power_dbm": c.power_dbm, "center_hz": cfg.center_hz + c.center_hz,
            "band_hz": None if band is None else [band[0] + cfg.center_hz, band[1] + cfg.center_hz],
            "agility": summary(c.agility, cfg.sample_rate, max_segments) if c.agility.enabled else None,
        })
    if composite:
        # Power sum of independent components; coherent copies add as amplitudes.
        from .signal import composite_power
        reference, relative = composite_power(cfg)
        total = float(reference + 10 * np.log10(relative))
    else:
        total = float(cfg.power_dbm)
    band = nominal_band(cfg)
    return {
        "config": cfg.model_dump(mode="json"),
        "sample_rate": cfg.sample_rate, "capture_samples": capture_samples(cfg), "rf_center_hz": cfg.rf_center_hz,
        "band_hz": None if band is None else list(band), "carrier_extent_hz": list(carrier_extent(cfg)),
        "total_power_dbm": total, "components": components,
        "agility": summary(cfg.agility, cfg.sample_rate, max_segments) if cfg.agility.enabled else None,
        "warnings": band_warnings(cfg),
    }


def component_problems(design: dict) -> list[str | None] | None:
    """Why each component of a (possibly invalid) composite design is rejected.

    Each component is validated on its own with the composite's clock and its
    offset folded in, so a component pushed beyond Nyquist by the composite
    offset is named too.  Returns ``None`` for a design that is not a composite.
    """
    from pydantic import ValidationError
    from .errors import describe
    if not isinstance(design, dict) or design.get("waveform") != "composite" or not isinstance(design.get("components"), list):
        return None
    shared = {k: design[k] for k in ("sample_rate", "duration", "rf_center_hz") if k in design}
    try:
        parent_offset = float(design.get("center_hz") or 0.0)
    except (TypeError, ValueError):
        parent_offset = 0.0
    out: list[str | None] = []
    for item in design["components"][:64]:
        if not isinstance(item, dict):
            out.append("not a signal design")
            continue
        candidate = {**item, **shared, "components": []}
        try:
            candidate["center_hz"] = float(item.get("center_hz") or 0.0) + parent_offset
        except (TypeError, ValueError):
            pass
        try:
            TxConfig.model_validate(candidate)
            out.append(None)
        except ValidationError as exc:
            out.append(describe(exc, candidate))
        except ValueError as exc:  # pragma: no cover - defensive
            out.append(str(exc))
    return out


def band_warnings(cfg: TxConfig) -> list[str]:
    """Aliasing warnings for the design and each enabled component."""
    nyq = cfg.sample_rate / 2
    out = []
    items = [(c.label or f"Component {i + 1}", c, cfg.center_hz) for i, c in enumerate(cfg.components) if c.enabled] \
        if cfg.waveform == WaveformKind.COMPOSITE else [("The signal", cfg, 0.0)]
    a, b = frequency_extent(cfg.agility) if cfg.waveform == WaveformKind.COMPOSITE else (0.0, 0.0)
    for name, c, shift in items:
        band = nominal_band(c)
        if band is None:
            continue
        lo, hi = band[0] + shift + a, band[1] + shift + b
        if lo < -nyq or hi > nyq:
            out.append(f"{name} occupies about {eng(lo)} … {eng(hi)}, beyond the sampled band (±{eng(nyq)} at "
                       f"{eng(cfg.sample_rate, 'S/s')}); that part aliases.")
    return out
