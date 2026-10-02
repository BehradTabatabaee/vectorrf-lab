from __future__ import annotations

import math
import numpy as np
from scipy import signal

from .dsp import gray_to_binary, rrc_taps


SUPPORTED_DIGITAL = {"bpsk", "qpsk", "8psk", "16qam", "64qam"}


def bits_per_symbol(modulation: str) -> int:
    return {"bpsk": 1, "qpsk": 2, "8psk": 3, "16qam": 4, "64qam": 6, "16apsk": 4, "32apsk": 5}[modulation]


def _bit_groups(bits: np.ndarray, k: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8)
    pad = (-bits.size) % k
    if pad:
        bits = np.pad(bits, (0, pad))
    return bits.reshape(-1, k)


def _bits_to_int(groups: np.ndarray) -> np.ndarray:
    weights = 1 << np.arange(groups.shape[1] - 1, -1, -1)
    return groups @ weights


def _int_to_bits(values: np.ndarray, k: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64).reshape(-1, 1)
    shifts = np.arange(k - 1, -1, -1)
    return ((values >> shifts) & 1).astype(np.uint8).reshape(-1)


def binary_to_gray(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.int64)
    return v ^ (v >> 1)


def map_bits(bits: np.ndarray, modulation: str, ratio2: float = 2.85, ratio3: float = 5.27) -> np.ndarray:
    modulation = modulation.lower()
    k = bits_per_symbol(modulation)
    g = _bit_groups(bits, k)
    if modulation in {"16apsk", "32apsk"}:
        counts = [4, 12] if modulation == "16apsk" else [4, 12, 16]
        radii = [1, ratio2, ratio3]
        points = np.concatenate([radii[j] * np.exp(1j*(2*np.pi*np.arange(c)/c + np.pi/c)) for j,c in enumerate(counts)])
        points /= np.sqrt(np.mean(np.abs(points)**2))
        return points[_bits_to_int(g)].astype(np.complex64)
    if modulation == "bpsk":
        return (1.0 - 2.0 * g[:, 0]).astype(np.complex64)
    if modulation == "qpsk":
        i = 1.0 - 2.0 * g[:, 0]
        q = 1.0 - 2.0 * g[:, 1]
        return ((i + 1j * q) / np.sqrt(2.0)).astype(np.complex64)
    if modulation == "8psk":
        idx = _bits_to_int(g)
        # Gray-coded phase order around the circle.
        phase_index = gray_to_binary(idx)
        return np.exp(1j * 2 * np.pi * phase_index / 8.0).astype(np.complex64)
    if modulation in {"16qam", "64qam"}:
        side = 4 if modulation == "16qam" else 8
        axis_bits = k // 2
        gi = _bits_to_int(g[:, :axis_bits])
        gq = _bits_to_int(g[:, axis_bits:])
        bi = gray_to_binary(gi)
        bq = gray_to_binary(gq)
        levels_i = 2.0 * bi - (side - 1)
        levels_q = 2.0 * bq - (side - 1)
        z = levels_i + 1j * levels_q
        norm = np.sqrt((2.0 / 3.0) * (side * side - 1.0))
        return (z / norm).astype(np.complex64)
    raise ValueError(f"unsupported modulation {modulation}")


def reference_constellation(modulation: str, ratio2: float = 2.85, ratio3: float = 5.27) -> np.ndarray:
    k = bits_per_symbol(modulation)
    all_bits = _int_to_bits(np.arange(1 << k), k)
    return map_bits(all_bits, modulation, ratio2, ratio3)


def slice_symbols(symbols: np.ndarray, modulation: str, ratio2: float = 2.85, ratio3: float = 5.27, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    refs = reference_constellation(modulation, ratio2, ratio3)
    z = np.asarray(symbols, dtype=np.complex64)
    # Remove a scalar complex gain before slicing. This is useful after matched filtering.
    p = np.sqrt(np.mean(np.abs(z) ** 2) + 1e-15)
    if normalize and p > 0:
        z = z / p
    dist = np.abs(z[:, None] - refs[None, :]) ** 2
    idx = np.argmin(dist, axis=1)
    decisions = refs[idx]
    k = bits_per_symbol(modulation)
    if modulation in {"16apsk", "32apsk"}:
        bits = _int_to_bits(idx, k)
    elif modulation == "bpsk":
        bits = (decisions.real < 0).astype(np.uint8)
    elif modulation == "qpsk":
        bits = np.column_stack([decisions.real < 0, decisions.imag < 0]).astype(np.uint8).reshape(-1)
    elif modulation == "8psk":
        phase_idx = np.mod(np.rint(np.angle(decisions) * 8 / (2 * np.pi)), 8).astype(int)
        gray = binary_to_gray(phase_idx)
        bits = _int_to_bits(gray, 3)
    else:
        side = 4 if modulation == "16qam" else 8
        axis_bits = k // 2
        refs_scale = np.sqrt(np.mean(np.abs((2*np.arange(side)-(side-1))[:,None] + 1j*(2*np.arange(side)-(side-1))[None,:])**2))
        # Work from the selected reference point rather than raw symbol.
        ri = decisions.real * refs_scale
        rq = decisions.imag * refs_scale
        bi = np.clip(np.rint((ri + side - 1) / 2), 0, side - 1).astype(int)
        bq = np.clip(np.rint((rq + side - 1) / 2), 0, side - 1).astype(int)
        gi = binary_to_gray(bi)
        gq = binary_to_gray(bq)
        bits_i = _int_to_bits(gi, axis_bits).reshape(-1, axis_bits)
        bits_q = _int_to_bits(gq, axis_bits).reshape(-1, axis_bits)
        bits = np.hstack([bits_i, bits_q]).reshape(-1)
    return decisions.astype(np.complex64), bits


def pulse_shape(symbols: np.ndarray, sps: int, rolloff: float, span_symbols: int) -> tuple[np.ndarray, np.ndarray]:
    taps = rrc_taps(sps, rolloff, span_symbols)
    y = signal.upfirdn(taps, symbols, up=sps)
    delay = (len(taps) - 1) // 2
    # Keep group delay; RX and eye plots can explicitly show it. Pad to avoid clipping the tail.
    return y.astype(np.complex64), taps


def evm_rms(measured: np.ndarray, modulation: str, ratio2: float = 2.85, ratio3: float = 5.27) -> dict[str, float]:
    measured = np.asarray(measured, dtype=np.complex128)
    refs = reference_constellation(modulation, ratio2, ratio3).astype(np.complex128)
    if measured.size == 0:
        return {"evm_rms_percent": float("nan"), "evm_peak_percent": float("nan"), "mer_db": float("nan")}
    # Normalize RMS and remove best residual common phase.
    measured = measured / np.sqrt(np.mean(np.abs(measured) ** 2) + 1e-15)
    dist = np.abs(measured[:, None] - refs[None, :]) ** 2
    nearest = refs[np.argmin(dist, axis=1)]
    phase = np.angle(np.vdot(measured, nearest))
    measured = measured * np.exp(1j * phase)
    dist = np.abs(measured[:, None] - refs[None, :]) ** 2
    nearest = refs[np.argmin(dist, axis=1)]
    err = measured - nearest
    ref_rms = np.sqrt(np.mean(np.abs(nearest) ** 2) + 1e-15)
    evm = np.abs(err) / ref_rms
    rms_pct = 100.0 * np.sqrt(np.mean(evm ** 2))
    peak_pct = 100.0 * np.max(evm)
    mer_db = -20.0 * np.log10(max(rms_pct / 100.0, 1e-15))
    return {"evm_rms_percent": float(rms_pct), "evm_peak_percent": float(peak_pct), "mer_db": float(mer_db)}
