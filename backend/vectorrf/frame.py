"""Generic mixed-modulation frame synthesis and acquisition helpers.

The frame model is intentionally protocol-neutral.  It is useful for laboratory
experiments where a preamble, header and payload may use different modulations
or rates without pretending to implement a named standard.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from .dsp import rrc_taps
from .models import FrameBlock
from .modem import bits_per_symbol, map_bits

_LINEAR = {
    "bpsk", "qpsk", "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk", "8psk",
    "16qam", "64qam", "16apsk", "32apsk",
}
_FSK = {"2fsk", "4fsk", "msk", "gmsk"}

def _ofdm_sample_count(block: FrameBlock, cfg) -> int:
    from .ofdm import layout
    _, active, pilots, cpv = layout(cfg.ofdm_fft, cfg.ofdm_cp, cfg.ofdm_occupied, cfg.ofdm_design)
    modulation = cfg.ofdm_design.modulation if cfg.ofdm_design.enabled else "qpsk"
    bits_per_ofdm_symbol = max(1, (len(active)-len(pilots))*bits_per_symbol(modulation))
    symbols = max(1, int(math.ceil(block.bit_count/bits_per_ofdm_symbol)))
    return int(sum(cfg.ofdm_fft + cpv[i % len(cpv)] for i in range(symbols)))


def _block_rate(block: FrameBlock, default_rate: float) -> float:
    return float(block.symbol_rate or default_rate)


def block_bits(block: FrameBlock, payload_provider, rng: np.random.Generator, offset: int = 0) -> np.ndarray:
    """Return exactly ``bit_count`` bits for one logical block repetition."""
    count = int(block.bit_count)
    if block.role in {"preamble", "bits"}:
        pattern = np.fromiter((int(b) for b in block.bits), dtype=np.uint8)
        return np.resize(pattern, count).astype(np.uint8)
    if block.role == "pn":
        return rng.integers(0, 2, count, dtype=np.uint8)
    return np.asarray(payload_provider(count, offset), dtype=np.uint8)[:count]


def block_sample_count(block: FrameBlock, fs: float, default_rate: float, ofdm_cfg=None) -> int:
    if block.modulation == "ofdm":
        if ofdm_cfg is None:
            raise ValueError("OFDM frame block needs the parent frame configuration")
        return _ofdm_sample_count(block, ofdm_cfg) * int(block.repeat)
    rate = _block_rate(block, default_rate)
    sps = max(2, int(round(fs / rate)))
    if block.modulation in _LINEAR:
        base = {"dbpsk":"bpsk", "dqpsk":"qpsk", "pi4-dqpsk":"qpsk", "oqpsk":"qpsk"}.get(block.modulation, block.modulation)
        k = bits_per_symbol(base)
    elif block.modulation in {"4fsk"}:
        k = 2
    else:
        k = 1
    symbols = int(math.ceil(block.bit_count / k))
    return max(sps, symbols * sps) * int(block.repeat)


def synthesize_block(
    block: FrameBlock,
    bits: np.ndarray,
    fs: float,
    default_rate: float,
    *,
    pulse_shape: str = "rrc",
    rolloff: float = .35,
    filter_span_symbols: int = 10,
    fsk_deviation_hz: float = 25_000.,
    gaussian_bt: float = .3,
    apsk_ratio2: float = 2.85,
    apsk_ratio3: float = 5.27,
    ofdm_cfg=None,
) -> tuple[np.ndarray, dict]:
    """Synthesize one block repetition with an exact symbol-duration length."""
    mod = block.modulation
    rate = _block_rate(block, default_rate)
    sps = max(2, int(round(fs / rate)))
    actual_rate = fs / sps
    bits = np.asarray(bits, dtype=np.uint8)

    if mod == "ofdm":
        if ofdm_cfg is None:
            raise ValueError("OFDM frame block needs the parent frame configuration")
        from .ofdm import synthesize
        target = _ofdm_sample_count(block, ofdm_cfg)
        def source(count, offset=0):
            if not len(bits):
                return np.zeros(count, dtype=np.uint8)
            return np.resize(bits, max(count + offset, 1))[offset:offset+count].astype(np.uint8)
        y, _, ometa = synthesize(ofdm_cfg, target, source, 0.0)
        symbols = None
        sps = 1
        actual_rate = float(ofdm_cfg.sample_rate/ofdm_cfg.ofdm_fft)
    elif mod in _LINEAR:
        base = {"dbpsk":"bpsk", "dqpsk":"qpsk", "pi4-dqpsk":"qpsk", "oqpsk":"qpsk"}.get(mod, mod)
        k = bits_per_symbol(base)
        symbols = map_bits(bits, base, apsk_ratio2, apsk_ratio3)
        if mod in {"dbpsk", "dqpsk", "pi4-dqpsk"}:
            if mod == "dqpsk":
                symbols = symbols * np.exp(-1j*np.pi/4)
            symbols = np.cumprod(symbols.astype(np.complex128)).astype(np.complex64)
        target = max(sps, len(symbols) * sps)
        if pulse_shape == "rect":
            y = np.repeat(symbols, sps)
        else:
            taps = rrc_taps(sps, rolloff, filter_span_symbols)
            impulses = np.zeros(target, dtype=np.complex64)
            impulses[::sps][:len(symbols)] = symbols
            # Centered convolution avoids adding a filter-tail duration to the
            # block.  Adjacent blocks therefore have deterministic boundaries.
            y = signal.fftconvolve(impulses, taps, mode="same")
        y = np.asarray(y[:target], dtype=np.complex64)
        if mod == "oqpsk":
            q = np.interp(np.arange(len(y))-sps/2, np.arange(len(y)), y.imag, left=0.0)
            y = (y.real + 1j*q).astype(np.complex64)
    elif mod in _FSK:
        levels = 4 if mod == "4fsk" else 2
        k = 2 if levels == 4 else 1
        pad = (-len(bits)) % k
        bb = np.pad(bits, (0, pad)) if pad else bits
        groups = bb.reshape(-1, k)
        idx = groups @ (1 << np.arange(k-1, -1, -1))
        if mod in {"msk", "gmsk"}:
            freqs = np.where(idx > 0, actual_rate/4.0, -actual_rate/4.0)
        else:
            centered = idx.astype(float) - (levels - 1)/2.0
            spacing = 2.0*fsk_deviation_hz/max(1, levels-1)
            freqs = centered*spacing
        inst = np.repeat(freqs, sps).astype(float)
        if mod == "gmsk" and len(inst):
            sigma = np.sqrt(np.log(2))*sps/(2*np.pi*gaussian_bt)
            from scipy.ndimage import gaussian_filter1d
            inst = gaussian_filter1d(inst, sigma=sigma, mode="nearest")
        phase = 2*np.pi*np.cumsum(inst)/fs
        y = np.exp(1j*phase).astype(np.complex64)
        symbols = None
    else:  # guarded by FrameBlock schema
        raise ValueError(f"unsupported frame-block modulation: {mod}")

    gain = 10.0 ** (float(block.power_offset_db) / 20.0)
    y = (y * gain).astype(np.complex64)
    return y, {
        "name": block.name,
        "role": block.role,
        "modulation": mod,
        "bit_count": int(block.bit_count),
        "symbol_rate": float(actual_rate),
        "samples_per_symbol": int(sps),
        "sample_count": int(len(y)),
        "power_offset_db": float(block.power_offset_db),
        **({"ofdm_fft": int(ofdm_cfg.ofdm_fft), "ofdm_cp": int(ofdm_cfg.ofdm_cp), "subcarrier_spacing_hz": float(ofdm_cfg.sample_rate/ofdm_cfg.ofdm_fft)} if mod == "ofdm" and ofdm_cfg is not None else {}),
    }


def synthesize_frame(cfg, n: int, rng: np.random.Generator, payload_provider) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build and repeat a mixed-modulation frame to fill ``n`` samples."""
    frame, bits, meta = frame_period(cfg, rng, payload_provider)
    reps = int(math.ceil(n/len(frame)))
    return np.tile(frame, reps)[:n].astype(np.complex64), bits, meta


def frame_period(cfg, rng: np.random.Generator, payload_provider) -> tuple[np.ndarray, np.ndarray, dict]:
    """One period of a mixed-modulation frame: its samples, bits and block layout."""
    segments: list[np.ndarray] = []
    meta: list[dict] = []
    all_bits: list[np.ndarray] = []
    payload_offset = 0
    cursor = 0
    for block_index, block in enumerate(cfg.frame_blocks):
        for rep in range(block.repeat):
            bits = block_bits(block, payload_provider, rng, payload_offset)
            if block.role == "payload":
                payload_offset += int(block.bit_count)
            y, info = synthesize_block(
                block, bits, cfg.sample_rate, cfg.symbol_rate,
                pulse_shape=cfg.pulse_shape, rolloff=cfg.rolloff,
                filter_span_symbols=cfg.filter_span_symbols,
                fsk_deviation_hz=cfg.fsk_deviation_hz,
                gaussian_bt=cfg.gaussian_bt,
                apsk_ratio2=cfg.apsk_ratio2, apsk_ratio3=cfg.apsk_ratio3,
                ofdm_cfg=cfg,
            )
            info.update(block_index=block_index, repeat_index=rep, start_sample=cursor, end_sample=cursor+len(y))
            cursor += len(y)
            segments.append(y)
            meta.append(info)
            all_bits.append(bits)
    if not segments:
        raise ValueError("Block frame mode requires at least one frame block")
    frame = np.concatenate(segments).astype(np.complex64)
    if not len(frame):
        raise ValueError("Frame contains no samples")
    bits = np.concatenate(all_bits).astype(np.uint8) if all_bits else np.array([], dtype=np.uint8)
    return frame, bits, {"frame_mode":"blocks", "frame_samples":int(len(frame)), "frame_duration_s":float(len(frame)/cfg.sample_rate), "frame_blocks":meta}


def frame_acquisition(samples: np.ndarray, cfg, *, power_dbm: float = 0.0, reference_rms: float = 1.0) -> dict:
    """Energy detection and normalized preamble correlation for an RX frame plan."""
    x = np.asarray(samples, dtype=np.complex64)
    if not len(x):
        return {"detected": False, "start_sample": None, "energy": {}, "correlation": {}, "blocks": []}
    default_sps = max(2, int(round(cfg.sample_rate/max(cfg.symbol_rate, 1.0))))
    win = max(1, int(round(default_sps*cfg.energy_window_symbols)))
    energy_lin = uniform_filter1d(np.abs(x).astype(np.float64)**2, size=win, mode="nearest")
    ref2 = max(float(reference_rms)**2, 1e-30)
    energy_dbm = float(power_dbm) + 10*np.log10(np.maximum(energy_lin/ref2, 1e-30))
    energy_cross = np.flatnonzero(energy_dbm >= float(cfg.energy_threshold_dbm)) if cfg.energy_detection else np.array([], dtype=int)
    energy_start = int(energy_cross[0]) if energy_cross.size else None

    corr_score = np.array([], dtype=np.float64)
    corr_start = None
    corr_peak = None
    preamble = next((b for b in cfg.frame_blocks if b.role == "preamble"), None)
    if cfg.preamble_correlation and preamble is not None:
        pattern = np.fromiter((int(b) for b in preamble.bits), dtype=np.uint8)
        bits = np.resize(pattern, preamble.bit_count).astype(np.uint8)
        ref, _ = synthesize_block(
            preamble, bits, cfg.sample_rate, cfg.symbol_rate,
            pulse_shape=cfg.pulse_shape, rolloff=cfg.rolloff,
            filter_span_symbols=cfg.filter_span_symbols,
            fsk_deviation_hz=cfg.fsk_deviation_hz,
            apsk_ratio2=cfg.apsk_ratio2, apsk_ratio3=cfg.apsk_ratio3,
            ofdm_cfg=cfg,
        )
        if len(ref) > 131072:
            ref = ref[:131072]
        if 4 <= len(ref) <= len(x):
            corr = signal.fftconvolve(x, np.conj(ref[::-1]), mode="valid")
            local = signal.fftconvolve(np.abs(x)**2, np.ones(len(ref)), mode="valid")
            denom = np.sqrt(np.maximum(local*np.sum(np.abs(ref)**2), 1e-30))
            corr_score = np.abs(corr)/denom
            k = int(np.argmax(corr_score))
            corr_peak = float(corr_score[k])
            if corr_peak >= cfg.preamble_threshold:
                corr_start = k

    start = corr_start if corr_start is not None else energy_start
    blocks = []
    if start is not None:
        cursor = int(start)
        for i, block in enumerate(cfg.frame_blocks):
            count = block_sample_count(block, cfg.sample_rate, cfg.symbol_rate, cfg)
            blocks.append({
                "index": i, "name": block.name, "role": block.role,
                "modulation": block.modulation, "start_sample": cursor,
                "end_sample": min(len(x), cursor+count), "expected_samples": count,
            })
            cursor += count

    def reduced(values, maximum=1600):
        if len(values) <= maximum:
            return np.asarray(values, dtype=float).tolist()
        idx = np.linspace(0, len(values)-1, maximum, dtype=int)
        return np.asarray(values)[idx].astype(float).tolist()

    return {
        "detected": start is not None,
        "start_sample": start,
        "start_time_s": None if start is None else float(start/cfg.sample_rate),
        "energy": {
            "enabled": bool(cfg.energy_detection), "threshold_dbm": float(cfg.energy_threshold_dbm),
            "window_samples": int(win), "first_crossing_sample": energy_start,
            "dbm": reduced(energy_dbm),
        },
        "correlation": {
            "enabled": bool(cfg.preamble_correlation), "threshold": float(cfg.preamble_threshold),
            "peak": corr_peak, "start_sample": corr_start, "score": reduced(corr_score),
        },
        "blocks": blocks,
    }
