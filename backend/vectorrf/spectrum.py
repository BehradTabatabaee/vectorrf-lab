from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import numpy as np
from scipy import fft as sfft
from scipy import signal

from .dsp import next_pow2, phasor
from .models import ChannelBand, SpectrumConfig

MIN_SEGMENT = 16
# Spans up to this fraction of the sample rate use the digital down-converter.
ZOOM_SPAN_FRACTION = 0.25
# Decimated rate relative to the span: leaves a guard band for the filters.
ZOOM_OVERSAMPLING = 1.25
# Alias rejection of every decimation stage (dB).
ZOOM_REJECTION_DB = 110.0
# The FFT is computed this many times longer than the window (zero-padded):
# bins then lie at a quarter of the window's resolution, so a tone shows the
# same RBW shape and level wherever it falls between bins (Hann scalloping
# drops from 1.42 dB to under 0.1 dB).  Very long windows are padded less.
FFT_OVERSAMPLING = 4
MAX_PADDED_FFT = 1 << 18


def oversampling(nfft: int) -> int:
    """Zero-padding factor for a window of ``nfft`` samples."""
    return max(1, min(FFT_OVERSAMPLING, MAX_PADDED_FFT // max(1, int(nfft))))


@dataclass
class SpectrumResult:
    freq_hz: np.ndarray
    psd_db: np.ndarray
    nfft: int                  # window (segment) length
    rbw_actual_hz: float
    bandwidth_3db_hz: float
    bandwidth_99_hz: float
    peak_hz: float
    peak_db: float
    unit: str = "dBFS/Hz"
    rbw_limited_by_capture: bool = False
    segments: int = 1
    center_hz: float = 0.0
    span_hz: float = 0.0
    decimation: int = 1
    analysis_rate: float = 0.0
    channels: list[dict] = field(default_factory=list)
    fft_size: int = 0          # zero-padded transform length (bin spacing = analysis_rate / fft_size)


@dataclass(frozen=True)
class ZoomPlan:
    """How a requested centre/span is analysed.

    ``stages`` are the decimation factors of a multistage digital
    down-converter (empty for a full-band FFT).  Each stage is a linear-phase
    Kaiser low-pass whose stopband starts where its aliases would fold into
    the analysed span, so the span is protected by ``ZOOM_REJECTION_DB``.
    """
    stages: tuple[int, ...]
    fs: float
    center_hz: float
    span_hz: float

    @property
    def decimation(self) -> int:
        return math.prod(self.stages)

    @property
    def rate(self) -> float:
        return self.fs / self.decimation


def zoom_plan(fs: float, cfg: SpectrumConfig, min_rate: float = 0.0) -> ZoomPlan:
    """Decimation stages for ``cfg``; the output rate never drops below ``min_rate``."""
    span = min(float(cfg.span_hz or fs), fs)
    center = float(cfg.center_hz)
    stages: list[int] = []
    if span <= ZOOM_SPAN_FRACTION * fs and abs(center) + span / 2 <= fs / 2:
        rate, minimum = float(fs), max(ZOOM_OVERSAMPLING * span, float(min_rate))
        while rate / 2 >= minimum:
            d = next(d for d in (8, 4, 2) if rate / d >= minimum)
            stages.append(d)
            rate /= d
    return ZoomPlan(tuple(stages), float(fs), center, span)


@lru_cache(maxsize=64)
def _stage_filter(fs_in: float, decimation: int, span: float) -> np.ndarray:
    """Kaiser low-pass for one decimation stage (unity DC gain)."""
    fs_out = fs_in / decimation
    passband, stopband = span / 2, fs_out - span / 2
    numtaps, beta = signal.kaiserord(ZOOM_REJECTION_DB, (stopband - passband) / (fs_in / 2))
    h = signal.firwin(numtaps | 1, (passband + stopband) / 2, window=("kaiser", beta), fs=fs_in)
    h.flags.writeable = False
    return h


def _stage_filters(p: ZoomPlan) -> list[np.ndarray]:
    filters, rate = [], p.fs
    for d in p.stages:
        filters.append(_stage_filter(rate, d, p.span_hz))
        rate /= d
    return filters


def required_samples(fs: float, cfg: SpectrumConfig) -> int:
    """Input samples one analyzer update needs: the averaged FFT segments at
    the analysis rate plus the down-converter's filter transients."""
    p = zoom_plan(fs, cfg)
    nfft, _ = choose_nfft(p.rate, cfg.rbw_hz, cfg.window, cfg.max_fft)
    margin, scale = 0, 1
    for h, d in zip(_stage_filters(p), p.stages):
        margin += (len(h) - 1) * scale
        scale *= d
    return max(4096, nfft * max(1, cfg.averaging) * p.decimation + margin)


def down_convert(x: np.ndarray, p: ZoomPlan) -> np.ndarray:
    """Mix the plan's centre to DC and decimate, keeping only outputs whose
    filters lie entirely inside the input (no start-up transients)."""
    y = np.asarray(x, dtype=np.complex64)
    if p.center_hz:
        y = (y * phasor(-p.center_hz, p.fs, len(y)).astype(np.complex64)).astype(np.complex64)
    for h, d in zip(_stage_filters(p), p.stages):
        full = signal.upfirdn(h.astype(np.float32), y, up=1, down=d)
        first, last = -(-(len(h) - 1) // d), (len(y) - 1) // d
        y = full[first:last + 1].astype(np.complex64)
    return y


@lru_cache(maxsize=64)
def _window(name: str, n: int) -> np.ndarray:
    if name == "hann":
        w = signal.windows.hann(n, sym=False)
    elif name == "blackmanharris":
        w = signal.windows.blackmanharris(n, sym=False)
    elif name == "flattop":
        w = signal.windows.flattop(n, sym=False)
    elif name == "kaiser":
        w = signal.windows.kaiser(n, beta=8.6, sym=False)
    else:
        w = np.ones(n)
    w.flags.writeable = False
    return w


def _enbw_bins(w: np.ndarray) -> float:
    return float(len(w) * np.sum(w*w) / (np.sum(w)**2 + 1e-30))


@lru_cache(maxsize=64)
def _window_sums(name: str, n: int) -> tuple[float, float, float]:
    """(ENBW in bins, Σw, Σw²) for a cached window."""
    w = _window(name, n)
    s1, s2 = float(np.sum(w)), float(np.sum(w * w))
    return n * s2 / (s1 * s1 + 1e-30), s1, s2


def choose_nfft(fs: float, rbw_hz: float, window_name: str, max_fft: int) -> tuple[int, float]:
    """FFT length implied by a requested RBW, ignoring how many samples exist.

    ENBW (in bins) depends only weakly on length for these windows, so one
    probe length is sufficient.  ``analyze`` additionally limits the segment to
    the samples actually available; see ``segment_length``.
    """
    enbw = _window_sums(window_name, 4096)[0]
    nfft = min(max_fft, max(256, next_pow2(enbw * fs / rbw_hz)))
    actual = _window_sums(window_name, nfft)[0] * fs / nfft
    return int(nfft), float(actual)


def segment_length(requested_nfft: int, available: int) -> int:
    """Segment length that never extends past the captured samples.

    Zero-padding a short capture inside a full-length window would place the
    data under the window's leading edge, attenuating it by tens of dB and
    reporting an RBW the data cannot support.  Instead the segment shrinks to
    the largest power of two that fits, and RBW is reported for that length.
    """
    if available >= requested_nfft:
        return int(requested_nfft)
    if available < MIN_SEGMENT:
        return MIN_SEGMENT
    return 1 << int(np.floor(np.log2(available)))


def _crossing(f0: float, f1: float, p0: float, p1: float, level: float) -> float:
    """Linear-power interpolation of the frequency where power crosses ``level``."""
    if p1 == p0:
        return 0.5 * (f0 + f1)
    u = float(np.clip((level - p0) / (p1 - p0), 0.0, 1.0))
    return f0 + u * (f1 - f0)


def bandwidth_metrics(freq_hz: np.ndarray, power: np.ndarray) -> tuple[float, float, float, float]:
    """-3 dB width around the peak, 99 % occupied bandwidth, peak frequency and level.

    Both widths interpolate the crossing between bins instead of counting whole
    bins, so they converge to the analytic width as resolution increases.
    """
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    if power.size == 0 or not np.any(np.isfinite(power)):
        return 0.0, 0.0, 0.0, -300.0
    k = int(np.argmax(power))
    peak = power[k]
    threshold = peak / (10 ** (3/10))
    # Contiguous run of bins at or above -3 dB around the peak (vectorized: a
    # wideband signal can span hundreds of thousands of bins).
    below = power < threshold
    before = np.flatnonzero(below[:k])
    after = np.flatnonzero(below[k + 1:])
    left = int(before[-1]) + 1 if before.size else 0
    right = k + int(after[0]) if after.size else power.size - 1
    lo = freq_hz[left] if left == 0 else _crossing(freq_hz[left - 1], freq_hz[left], power[left - 1], power[left], threshold)
    hi = freq_hz[right] if right == power.size - 1 else _crossing(freq_hz[right], freq_hz[right + 1], power[right], power[right + 1], threshold)
    bw3 = abs(float(hi - lo))

    total = float(np.sum(power))
    if total <= 0 or power.size < 2:
        bw99 = 0.0
    else:
        cdf = np.cumsum(power) / total
        # Treat each bin's power as uniformly spread over its width: the CDF
        # reaches cdf[i] at the bin's upper edge.
        df = float(np.median(np.diff(freq_hz)))
        edges = np.r_[freq_hz[0] - df / 2, freq_hz + df / 2]
        levels = np.r_[0.0, cdf]
        f_lo = float(np.interp(0.005, levels, edges))
        f_hi = float(np.interp(0.995, levels, edges))
        bw99 = abs(f_hi - f_lo)
    return bw3, bw99, float(freq_hz[k]), float(10*np.log10(peak + 1e-30))


def analyze(samples: np.ndarray, fs: float, cfg: SpectrumConfig, reference_power_dbm: float | None = None,
            reference_rms: float | None = None, unit: str | None = None) -> SpectrumResult:
    """Analyze a baseband record over ``cfg.center_hz ± cfg.span_hz / 2``.

    Without a calibration reference the result is a PSD in dBFS/Hz.  With
    ``reference_power_dbm`` it reports power per resolution bandwidth, scaled
    so that a coherent tone whose RMS equals ``reference_rms`` reads
    ``reference_power_dbm`` at its peak (within the window's scalloping
    between the oversampled bins, under 0.1 dB for Hann).  ``unit`` only relabels that
    calibrated scale (for example ``"dBFS"`` with a 0 dB reference at full
    scale).  Narrow spans are digitally down-converted first; the decimation
    filters have unity passband gain, so both scales are unchanged by zooming.
    """
    x = np.asarray(samples, dtype=np.complex64)
    if reference_power_dbm is not None and reference_rms is None:
        xr = x.view(np.float32).astype(np.float64)
        reference_rms = float(np.sqrt(2.0 * np.mean(xr * xr))) if x.size else 0.0
    plan = zoom_plan(fs, cfg)
    if plan.stages:
        x = down_convert(x, plan)
    rate, offset = plan.rate, (plan.center_hz if plan.stages else 0.0)
    requested, _ = choose_nfft(rate, cfg.rbw_hz, cfg.window, cfg.max_fft)
    nfft = segment_length(requested, x.size)
    limited = nfft < requested
    if x.size < nfft:
        # Only reachable for captures shorter than MIN_SEGMENT samples.
        x = np.pad(x, (0, nfft - x.size))
    w64 = _window(cfg.window, nfft)
    enbw_bins, sum_w, sum_w2 = _window_sums(cfg.window, nfft)
    rbw_actual = enbw_bins * rate / nfft
    w = w64.astype(np.float32)
    pad = oversampling(nfft)
    size = nfft * pad
    step = max(1, nfft // 2)
    starts = np.arange(0, x.size - nfft + 1, step, dtype=np.int64)
    if starts.size > max(1, cfg.averaging):
        starts = starts[-max(1, cfg.averaging):]
    if reference_power_dbm is None:
        scale = sum_w2 * rate + 1e-30
        calibration = 1.0
        db_offset = 0.0
        out_unit = "dBFS/Hz"
    else:
        # Bin-power normalization makes a bin-centred complex CW read its total
        # calibrated power.  The absolute power stays a dB-domain offset rather
        # than exponentiating dBm, which avoids artificial overflow limits for
        # very large/small finite simulator power settings.
        scale = sum_w ** 2 + 1e-30
        calibration = 1.0 / max(float(reference_rms) ** 2, 1e-30)
        db_offset = float(reference_power_dbm)
        out_unit = unit or "dBm"
    # Process segments in bounded batches: one 2-D FFT call per batch lets
    # scipy.fft thread across segments while keeping peak memory modest.
    batch = max(1, (1 << 22) // size)
    accumulated = None
    for b0 in range(0, starts.size, batch):
        idx = starts[b0:b0 + batch]
        blocks = np.stack([x[s:s + nfft] for s in idx]) * w[None, :]
        X = sfft.fft(blocks, size, axis=1, workers=-1)
        P = (np.abs(X) ** 2).astype(np.float64) * (calibration / scale)
        if cfg.detector == "peak":
            part = P.max(axis=0)
            accumulated = part if accumulated is None else np.maximum(accumulated, part)
        elif cfg.detector == "sample":
            accumulated = P[-1]
        else:
            part = P.sum(axis=0)
            accumulated = part if accumulated is None else accumulated + part
    p = accumulated / starts.size if cfg.detector == "average" else accumulated
    # Finite power keeps every dB product finite (JSON cannot carry inf/NaN),
    # even for pathological input such as a recording read with the wrong
    # format, without clamping the calibrated range itself.
    p = np.nan_to_num(sfft.fftshift(p), nan=0.0, posinf=1e300)
    f = sfft.fftshift(sfft.fftfreq(size, d=1.0 / rate)).astype(np.float64) + offset

    lo, hi = plan.center_hz - plan.span_hz / 2, plan.center_hz + plan.span_hz / 2
    mask = (f >= lo) & (f <= hi)
    f2 = f[mask]
    p2 = p[mask]
    db = (10*np.log10(p2 + 1e-30) + db_offset).astype(np.float32)
    bw3, bw99, peak_hz, peak_db = bandwidth_metrics(f2, p2)
    peak_db += db_offset
    # Zero padding keeps the power per bin and multiplies the bins per RBW.
    channels = [channel_measurement(f2, p2, rate / size, enbw_bins * pad, band, db_offset, reference_power_dbm is not None, out_unit)
                for band in cfg.channels]
    return SpectrumResult(f2, db, nfft, float(rbw_actual), bw3, bw99, peak_hz, peak_db, out_unit,
                          rbw_limited_by_capture=bool(limited), segments=int(starts.size),
                          center_hz=plan.center_hz, span_hz=plan.span_hz, decimation=plan.decimation,
                          analysis_rate=rate, channels=channels, fft_size=int(size))


def channel_measurement(f: np.ndarray, p: np.ndarray, bin_hz: float, enbw_bins: float, band: ChannelBand,
                        db_offset: float, calibrated: bool, unit: str) -> dict:
    """Integrated power, density, 99 % occupied bandwidth and peak inside ``band``.

    Each FFT bin covers ``bin_hz``; bins straddling a band edge contribute
    their overlapping fraction.  A calibrated spectrum holds power per bin
    scaled so a bin-centred tone reads its power, so integrated power is the
    bin sum divided by the window's equivalent noise bandwidth in bins.  An
    uncalibrated PSD (per Hz) integrates as ``Σ PSD · bin_hz``, in dBFS.
    """
    lo, hi = band.center_hz - band.width_hz / 2, band.center_hz + band.width_hz / 2
    weight = np.clip((np.minimum(f + bin_hz / 2, hi) - np.maximum(f - bin_hz / 2, lo)) / bin_hz, 0.0, 1.0)
    inside = weight > 0
    out = {"label": band.label, "center_hz": band.center_hz, "width_hz": band.width_hz,
           "unit": unit if calibrated else "dBFS", "covered": False}
    if not np.any(inside):
        return out
    covered_lo, covered_hi = float(f[inside][0] - bin_hz / 2), float(f[inside][-1] + bin_hz / 2)
    power_bins = p * weight
    total = float(np.sum(power_bins))
    power = total / enbw_bins if calibrated else total * bin_hz
    power_db = 10 * math.log10(max(power, 1e-300)) + db_offset
    k = int(np.argmax(np.where(inside, p, -1.0)))
    obw = 0.0
    if total > 0:
        cdf = np.cumsum(power_bins[inside]) / total
        edges = np.r_[covered_lo, f[inside] + bin_hz / 2]
        levels = np.r_[0.0, cdf]
        obw = float(np.interp(0.995, levels, edges) - np.interp(0.005, levels, edges))
    return {**out, "covered": covered_lo <= lo + bin_hz and covered_hi >= hi - bin_hz,
            "power_db": power_db, "density_db_hz": power_db - 10 * math.log10(band.width_hz),
            "obw99_hz": obw, "peak_hz": float(f[k]), "peak_db": 10 * math.log10(max(float(p[k]), 1e-300)) + db_offset}


def stft_ridge(samples: np.ndarray, fs: float, nperseg: int = 512) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(samples)
    nperseg = min(max(64, nperseg), max(64, x.size))
    f, t, Z = signal.stft(x, fs=fs, window="hann", nperseg=nperseg, noverlap=3*nperseg//4,
                          return_onesided=False, boundary=None, padded=False)
    if Z.size == 0:
        return np.array([]), np.array([]), np.array([])
    f = np.fft.fftshift(f)
    Z = np.fft.fftshift(Z, axes=0)
    ridge = f[np.argmax(np.abs(Z), axis=0)]
    return t, ridge, np.abs(Z)
