"""Time-domain measurements on bounded captures with truthful analysis bandwidth."""
import math

import numpy as np
from scipy import fft as sfft

from .dsp import phasor

# Display buckets of a zero-span trace.  A bucket holding several analysis
# samples shows their minimum and maximum, like an oscilloscope's peak detect:
# picking one sample per bucket aliased everything faster than the display
# grid (a 5 MHz beat, noise) into a pattern that changed every sweep.
ZERO_SPAN_POINTS = 600

# Gaussian RBW filter: amplitude response exp(-f²/2σ²) with |H|² = ½ at ±RBW/2
# (an analyzer's -3 dB resolution bandwidth); its noise bandwidth is 1.064 RBW.
_SIGMA_PER_RBW = 0.5 / math.sqrt(math.log(2.0))
_KEEP_RBW = 3.0            # band kept around the centre, ±3 RBW (H < 4·10⁻⁶ beyond)
_SETTLE_RBW = 1.2          # impulse-response half-length, in 1/RBW (±4.5 σ in time)
ENBW_PER_RBW = math.sqrt(math.pi) * _SIGMA_PER_RBW


def auto_bandwidth(duration_s: float, points: int = ZERO_SPAN_POINTS) -> float:
    """The RBW a sweep of ``duration_s`` resolves on ``points`` display points:
    one resolution cell (1/RBW) per two points."""
    return points / (2.0 * duration_s)


def settling_samples(fs, duration_s, bandwidth_hz=None, full_band=False, points=ZERO_SPAN_POINTS) -> int:
    """Samples beyond each end of a sweep that its RBW filter reads."""
    if full_band:
        return 0
    rbw = auto_bandwidth(duration_s, points) if bandwidth_hz is None else bandwidth_hz
    return int(math.ceil(_SETTLE_RBW * fs / max(rbw, 4.0 / duration_s)))


def rbw_filter(samples, fs, a, b, center_hz, rbw_hz, start_sample=0, around=None):
    """Samples ``a … b`` of ``samples`` mixed down by ``center_hz`` and filtered
    by a Gaussian RBW filter, at the lowest convenient rate.

    One FFT covers the interval plus the filter's settling time on each side
    (from the record, or from ``around(lo, hi)`` -- the source's samples at
    record positions ``lo … hi`` even beyond the record -- when given; zeros
    otherwise) and is padded for linear convolution; the band kept around the centre is inverse
    transformed alone, so the cost follows the interval, not the record, and
    the output rate follows the RBW.  The mixing phase refers to absolute
    sample ``start_sample`` (the record's first), so the phase of consecutive
    sweeps is continuous.

    Returns ``(y, positions, rate)``: the filtered samples, their positions in
    record samples (fractional when the rate is not an integer division) and
    the output rate.
    """
    margin = int(math.ceil(_SETTLE_RBW * fs / rbw_hz))
    if around is not None:
        lo, hi = a - margin, b + margin
        segment = around(lo, hi)
    else:
        lo, hi = max(0, a - margin), min(len(samples), b + margin)
        segment = samples[lo:hi]
    n = sfft.next_fast_len(hi - lo + margin)
    nd = min(n, sfft.next_fast_len(int(math.ceil(2 * _KEEP_RBW * rbw_hz / fs * n)) + 1))
    X = sfft.fft(np.asarray(segment, dtype=np.complex64), n, workers=-1)
    kc = int(round(center_hz / fs * n))
    j = np.arange(-(nd // 2), nd - nd // 2)
    f = (kc + j) * (fs / n) - center_hz
    H = np.exp(-0.5 * (f / (_SIGMA_PER_RBW * rbw_hz)) ** 2).astype(np.float32)
    Y = np.zeros(nd, dtype=np.complex64)
    Y[j % nd] = X[(kc + j) % n] * H
    y = sfft.ifft(Y, workers=-1) * np.float32(nd / n)
    step = n / nd                                  # record samples per output sample
    m0, m1 = int(math.ceil((a - lo) / step)), int(math.ceil((b - lo) / step))
    m = np.arange(m0, max(m0 + 1, m1))
    positions = lo + m * step
    # Bins were shifted by kc; the rest of the tuning and the absolute phase are a phasor.
    residual = center_hz - kc * fs / n
    start = phasor(-center_hz, fs, 1, int(start_sample) + lo)[0]
    tune = start * np.exp(-2j * np.pi * residual * (m * step) / fs)
    return (y[m] * tune).astype(np.complex64), positions, fs / step


def buckets(values, points=ZERO_SPAN_POINTS):
    """``(low, high)`` per display bucket (``high`` is None when every sample is
    its own point).  NaN marks undefined samples and is skipped."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) <= points:
        return v, None
    edges = np.linspace(0, len(v), points + 1).astype(np.int64)[:-1]
    return np.fmin.reduceat(v, edges), np.fmax.reduceat(v, edges)


def zero_span(samples, fs, offset=0., duration=.002, center_hz=0., bandwidth_hz=None, full_band=False,
              points=ZERO_SPAN_POINTS, start_sample=0, around=None):
    """Zero-span sweep of ``[offset, offset + duration)`` of the record.

    Like an analyzer in zero span, the record is tuned to ``center_hz`` and
    filtered by the resolution bandwidth before the AM, PM and FM detectors:
    ``bandwidth_hz`` (None: the RBW the display resolves, see
    ``auto_bandwidth``), or with ``full_band`` the whole sampled band.
    ``around`` supplies samples beyond the record for the filter's settling
    (see ``rbw_filter``); without it the sweep's edges show the filter rising.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Zero span needs a finite, positive sample rate")
    if not np.isfinite(center_hz) or abs(center_hz) >= fs / 2:
        raise ValueError("Zero-span centre must be inside the source Nyquist interval")
    if bandwidth_hz is not None and (not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0):
        raise ValueError("Zero-span bandwidth must be positive")
    if len(samples) < 2:
        raise ValueError("Zero span needs at least two captured samples")
    a = max(0, int(round(offset * fs)))
    if a >= len(samples) - 1:
        raise ValueError(
            f"Zero-span offset {offset:g} s is outside this {len(samples) / fs:g} s capture; "
            "reduce the offset or capture more samples"
        )
    b = min(len(samples), max(a + 2, a + int(round(duration * fs))))
    span_s = (b - a) / fs
    if full_band:
        x = np.asarray(samples[a:b])
        if center_hz:
            x = x * phasor(-center_hz, fs, len(x), int(start_sample) + a).astype(x.dtype if x.dtype == np.complex128 else np.complex64)
        positions, rate, rbw = np.arange(a, b, dtype=np.float64), float(fs), None
    else:
        room = fs - 2 * abs(center_hz)       # the widest band around the centre inside Nyquist
        if bandwidth_hz is not None and bandwidth_hz > room:
            raise ValueError("Zero-span bandwidth must fit inside the source Nyquist interval")
        requested = auto_bandwidth(span_s, points) if bandwidth_hz is None else float(bandwidth_hz)
        # A filter longer than the sweep shows only its own settling: resolve at
        # least four cells.
        rbw = float(min(room, max(requested, 4.0 / span_s)))
        x, positions, rate = rbw_filter(samples, fs, a, b, center_hz, rbw, start_sample, around)
    amplitude = np.abs(x).astype(np.float64)
    # Phase and frequency of near-zero samples are undefined: gaps, not inventions.
    valid = amplitude > max(float(amplitude.max(initial=0.0)) * .01, 1e-12)
    # Phase steps are the wrapped differences of the sample phases, so the
    # phase after a null resumes at the signal's own phase.
    angle = np.angle(x).astype(np.float64)
    step = (np.diff(angle) + np.pi) % (2 * np.pi) - np.pi
    phase = angle[0] + np.r_[0.0, np.cumsum(step)]
    fm = np.r_[np.nan, step * (rate / (2 * np.pi))]
    frequency_valid = valid.copy()
    frequency_valid[0] = False
    frequency_valid[1:] &= valid[:-1]
    fm[~frequency_valid] = np.nan
    phase[~valid] = np.nan
    t_low, t_high = buckets(positions / fs, points)
    out = {'time_s': t_low if t_high is None else (t_low + t_high) / 2, 'bucketed': t_high is not None}
    for key, values in (('i', x.real), ('q', x.imag), ('amplitude', amplitude),
                        ('phase_deg', np.rad2deg(phase)), ('frequency_hz', fm)):
        low, high = buckets(values, points)
        out[key] = low
        if high is not None:
            out[f'{key}_max'] = high
    measured = fm[frequency_valid]
    out.update({
        'sample_rate': fs, 'analysis_rate': rate, 'analysis_samples': int(len(x)),
        'capture_start_s': a / fs, 'capture_duration_s': span_s,
        'center_hz': float(center_hz), 'full_band': bool(full_band),
        'bandwidth_hz': None if full_band or bandwidth_hz is None else float(bandwidth_hz),
        'bandwidth_actual_hz': rbw, 'bandwidth_auto': not full_band and bandwidth_hz is None,
        'noise_bandwidth_hz': None if rbw is None else rbw * ENBW_PER_RBW,
        'rms': float(np.sqrt(np.mean(amplitude ** 2))), 'peak': float(amplitude.max(initial=0.0)),
        'mean_frequency_hz': float(np.mean(measured)) if len(measured) else None,
    })
    return out
