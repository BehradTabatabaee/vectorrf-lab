from __future__ import annotations

import math
import numpy as np
from scipy import signal

_EPS = 1e-15


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(x) ** 2) + _EPS))


def normalize_rms(x: np.ndarray, target: float = 1.0) -> np.ndarray:
    r = rms(x)
    return (x * (target / r)).astype(np.complex64, copy=False) if r > 0 else x.astype(np.complex64)


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    p = np.mean(np.abs(x) ** 2) + _EPS
    noise_p = p / (10.0 ** (snr_db / 10.0))
    n = (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size)) * math.sqrt(noise_p / 2.0)
    return (x + n).astype(np.complex64)


def rrc_taps(sps: int, rolloff: float, span_symbols: int) -> np.ndarray:
    """Root-raised-cosine FIR sampled at integer samples/symbol."""
    sps = max(2, int(sps))
    beta = float(rolloff)
    n = span_symbols * sps
    if n % 2:
        n += 1
    t = np.arange(-n // 2, n // 2 + 1, dtype=np.float64) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif beta > 0 and abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-10:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            den = np.pi * ti * (1 - (4 * beta * ti) ** 2)
            h[i] = num / den
    h /= np.sqrt(np.sum(h * h) + _EPS)
    return h.astype(np.float32)


def fractional_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    if abs(delay_samples) < 1e-8:
        return x.astype(np.complex64, copy=True)
    # Windowed-sinc fractional delay. Short enough for interactive use, stable for |delay| < 1 sample.
    ntaps = 41
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.sinc(n - delay_samples) * np.hamming(ntaps)
    h /= np.sum(h)
    y = signal.lfilter(h, [1.0], x)
    gd = (ntaps - 1) // 2
    y = np.roll(y, -gd)
    y[-gd:] = 0
    return y.astype(np.complex64)


def next_pow2(n: float) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(1.0, n)))))


def bytes_to_bits(message: str) -> np.ndarray:
    b = message.encode("utf-8")
    if not b:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    n = (bits.size // 8) * 8
    if n == 0:
        return b""
    return np.packbits(bits[:n]).tobytes()


def gray_to_binary(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.int64)
    b = g.copy()
    shift = 1
    while np.max(g, initial=0) >> shift:
        b ^= g >> shift
        shift += 1
    return b
