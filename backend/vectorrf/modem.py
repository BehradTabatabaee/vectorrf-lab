from __future__ import annotations

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


def square_grid(refs: np.ndarray):
    """``(levels, table)`` when ``refs`` is a square grid (square QAM): the
    common I/Q levels and the index of each (I level, Q level) point;
    otherwise None."""
    refs = np.asarray(refs)
    levels = np.unique(np.round(refs.real, 9))
    side = len(levels)
    if side < 2 or side * side != refs.size or not np.allclose(np.unique(np.round(refs.imag, 9)), levels):
        return None
    if not np.allclose(np.diff(levels), levels[1] - levels[0]):
        return None
    step = levels[1] - levels[0]
    i = np.rint((refs.real - levels[0]) / step).astype(int)
    q = np.rint((refs.imag - levels[0]) / step).astype(int)
    table = np.empty((side, side), dtype=np.int64)
    table[i, q] = np.arange(refs.size)
    return levels, table


def _nearest(z: np.ndarray, refs: np.ndarray) -> np.ndarray:
    """Index of the nearest reference point, evaluated in bounded memory.

    A square grid is sliced by rounding I and Q; otherwise the distances come
    from one matrix product, |z - r|² = |z|² - 2 Re(z r*) + |r|² (|z|² does
    not change the nearest point)."""
    z = np.asarray(z)
    refs = np.asarray(refs)
    grid = square_grid(refs)
    if grid is not None:
        levels, table = grid
        step, side = levels[1] - levels[0], len(levels)
        i = np.clip(np.rint((z.real - levels[0]) / step), 0, side - 1).astype(np.int64)
        q = np.clip(np.rint((z.imag - levels[0]) / step), 0, side - 1).astype(np.int64)
        return table[i, q].reshape(z.shape).ravel()
    flat = z.ravel()
    points = np.stack([refs.real, refs.imag]).astype(np.float64)       # 2 × M
    power = np.sum(points ** 2, axis=0)
    out = np.empty(flat.size, dtype=np.int64)
    chunk = max(1, 4_000_000 // max(1, refs.size))
    for start in range(0, flat.size, chunk):
        part = flat[start:start + chunk]
        dot = np.stack([part.real, part.imag], axis=1).astype(np.float64) @ points
        out[start:start + chunk] = np.argmin(power[None, :] - 2.0 * dot, axis=1)
    return out


def radius_scales(measured: np.ndarray, refs: np.ndarray, count: int = 3) -> list[float]:
    """Candidate gains ``g`` (``measured / g`` on the constellation) from magnitudes alone.

    The symbol magnitudes are matched to the constellation's distinct radii
    over gains from 0.5 to 2 times the RMS normalization; the ``count`` best
    local minima of the mean squared radial error are returned, best first.
    Magnitudes do not depend on the carrier phase, and unlike the RMS they do
    not depend on how a payload uses the constellation: a repeating text can
    use a dozen of 64-QAM's points, with an RMS 16 % off the constellation's.
    """
    a = np.abs(np.asarray(measured, dtype=np.complex128))[:1024]
    radii = np.unique(np.round(np.abs(np.asarray(refs, dtype=np.complex128)), 9))
    rms = float(np.sqrt(np.mean(a * a) + 1e-30))
    if a.size < 8 or radii.size < 2:
        return [rms]
    edges = 0.5 * (radii[1:] + radii[:-1])

    def cost(gains):
        scaled = a[None, :] / np.asarray(gains)[:, None]
        nearest = radii[np.searchsorted(edges, scaled)]
        return np.mean((scaled - nearest) ** 2, axis=1)

    gains = rms * np.geomspace(0.5, 2.0, 97)
    costs = cost(gains)
    minima = [k for k in range(1, len(costs) - 1) if costs[k] <= costs[k - 1] and costs[k] <= costs[k + 1]]
    minima.sort(key=lambda k: costs[k])
    out = []
    for k in minima[:count]:
        fine = np.geomspace(gains[k - 1], gains[k + 1], 41)
        out.append(float(fine[int(np.argmin(cost(fine)))]))
    return out or [rms]


def _dd_fit(z: np.ndarray, refs: np.ndarray, complex_gain: bool, iterations: int) -> tuple[np.ndarray, np.ndarray]:
    idx = _nearest(z, refs)
    for _ in range(max(1, iterations)):
        d = refs[idx]
        denom = float(np.vdot(d, d).real)
        if denom <= 0:
            break
        g = np.vdot(d, z) / denom
        if not complex_gain:
            # Magnitude of the complex LS gain: exact amplitude scale even when
            # a residual rotation is present, without correcting that rotation.
            g = abs(g)
        if not np.isfinite(g) or abs(g) < 1e-12:
            break
        z = z / g
        new = _nearest(z, refs)
        if np.array_equal(new, idx):
            break
        idx = new
    return z, idx


def fit_constellation(measured: np.ndarray, refs: np.ndarray, complex_gain: bool = True, iterations: int = 3,
                      robust: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Scale (and optionally rotate) symbols onto a unit-power constellation.

    Iterates decision-directed least squares, ``g = <d, z> / <d, d>`` with
    ``d`` the current nearest reference points.  Unlike RMS normalization
    alone, the LS gain is not inflated by additive noise power and is exact
    for noise-free symbols of any finite data mix (an RMS fit of e.g. 64 random
    16-QAM symbols is not).  The iteration starts from the RMS normalization
    and, for constellations with several radii, also from the gains that match
    the symbol magnitudes to the radii (``radius_scales``) when the first fit
    is poor and ``robust``; the start that ends with the smallest error wins.

    Returns the fitted symbols and the indices of their nearest references.
    """
    z = np.asarray(measured, dtype=np.complex128)
    if z.size == 0:
        return z, np.zeros(0, dtype=np.int64)
    refs = np.asarray(refs, dtype=np.complex128)
    fitted, idx = _dd_fit(z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-30), refs, complex_gain, iterations)
    error = float(np.mean(np.abs(fitted - refs[idx]) ** 2))
    # Decisions within a quarter of the half-spacing (in power) are settled;
    # otherwise the RMS start may be off a payload-dependent scale.
    spacing = float(np.min(np.abs(refs[:, None] - refs[None, :]) + np.eye(len(refs)) * 1e9)) if len(refs) > 1 else 1.0
    if robust and error > 0.25 * (spacing / 2) ** 2 and np.unique(np.round(np.abs(refs), 9)).size > 1 and z.size >= 8:
        for gain in radius_scales(z, refs):
            candidate, cidx = _dd_fit(z / gain, refs, complex_gain, iterations)
            cerror = float(np.mean(np.abs(candidate - refs[cidx]) ** 2))
            if cerror < error - 1e-15:
                fitted, idx, error = candidate, cidx, cerror
    return fitted, idx


def slice_symbols(symbols: np.ndarray, modulation: str, ratio2: float = 2.85, ratio3: float = 5.27, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    refs = reference_constellation(modulation, ratio2, ratio3)
    z = np.asarray(symbols, dtype=np.complex64)
    if normalize and z.size:
        # Real (scale-only) LS fit: carrier phase stays the job of the carrier
        # loop / manual phase control, so a residual rotation still shows up as
        # decision errors instead of being silently corrected here.
        _, idx = fit_constellation(z, refs, complex_gain=False)
    else:
        idx = _nearest(z, refs)
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
    # Keep group delay; RX and eye plots can explicitly show it. Pad to avoid clipping the tail.
    return y.astype(np.complex64), taps


def evm_rms(measured: np.ndarray, modulation: str, ratio2: float = 2.85, ratio3: float = 5.27, robust: bool = True) -> dict[str, float | None]:
    return evm_with_decisions(measured, modulation, ratio2, ratio3, robust)[0]


def evm_with_decisions(measured: np.ndarray, modulation: str, ratio2: float = 2.85, ratio3: float = 5.27,
                       robust: bool = True) -> tuple[dict[str, float | None], np.ndarray]:
    """EVM metrics plus the index of each symbol's decided reference point."""
    measured = np.asarray(measured, dtype=np.complex128)
    refs = reference_constellation(modulation, ratio2, ratio3).astype(np.complex128)
    if measured.size == 0:
        return {"evm_rms_percent": None, "evm_peak_percent": None, "mer_db": None}, np.zeros(0, np.int64)
    # Decision-directed LS complex gain: removes scale and common phase without
    # letting noise power bias the normalization.
    measured, idx = fit_constellation(measured, refs, complex_gain=True, robust=robust)
    nearest = refs[idx]
    err = measured - nearest
    ref_rms = np.sqrt(np.mean(np.abs(nearest) ** 2) + 1e-15)
    evm = np.abs(err) / ref_rms
    rms_pct = 100.0 * np.sqrt(np.mean(evm ** 2))
    peak_pct = 100.0 * np.max(evm)
    mer_db = -20.0 * np.log10(max(rms_pct / 100.0, 1e-15))
    return {"evm_rms_percent": float(rms_pct), "evm_peak_percent": float(peak_pct), "mer_db": float(mer_db)}, idx
