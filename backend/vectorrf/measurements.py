"""Time-domain measurements on bounded captures with truthful analysis bandwidth."""
import math
import numpy as np
from scipy import signal

from .dsp import display_indices, filter_same, phasor, resample


def zero_span(samples, fs, offset=0., duration=.002, center_hz=0., bandwidth_hz=None,
              max_points=900, start_sample=0):
    """Zero-span view of ``[offset, offset + duration)``.

    Like an analyzer in zero span, the record is tuned to ``center_hz`` and
    limited to ``bandwidth_hz`` before AM/PM/FM detection.  Narrow bands are
    anti-aliased and decimated first; the cost then follows the captured sample
    count rather than the enormous FIR a GS/s clock would otherwise require.
    A band finer than the capture can resolve is reported at the achieved
    bandwidth, never silently represented by a truncated FIR.
    """
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("Zero span needs a finite, positive sample rate")
    if not np.isfinite(center_hz) or abs(center_hz) >= fs / 2:
        raise ValueError("Zero-span centre must be inside the source Nyquist interval")
    if bandwidth_hz is not None and (not np.isfinite(bandwidth_hz) or bandwidth_hz <= 0 or
                                    bandwidth_hz > fs - 2 * abs(center_hz)):
        raise ValueError("Zero-span bandwidth must fit inside the source Nyquist interval")
    if len(samples) < 2:
        raise ValueError("Zero span needs at least two captured samples")
    start = max(0, int(offset*fs))
    if start >= len(samples) - 1:
        raise ValueError(
            f"Zero-span offset {offset:g} s is outside this {len(samples) / fs:g} s capture; "
            "reduce the offset or capture more samples"
        )
    end = min(len(samples),max(start+2,start+int(duration*fs)))
    analysis_rate = fs
    decimation = 1
    achieved_bandwidth = None
    if bandwidth_hz is not None:
        # Four cycles across the record is a conservative usable resolution.
        if fs - 2 * abs(center_hz) < 4 * fs / len(samples):
            raise ValueError("Capture is too short to resolve a zero-span band inside Nyquist; move the centre or capture more samples")
        achieved_bandwidth = float(min(fs - 2 * abs(center_hz), max(bandwidth_hz, 4 * fs / len(samples))))
        decimation = max(1, min(int(fs / (8 * achieved_bandwidth)), len(samples) // 32,
                                 max(1, (end - start) // 16)))
        analysis_rate = fs / decimation
    if center_hz or bandwidth_hz is not None:
        y = np.asarray(samples, dtype=np.complex64)
        if center_hz:
            y = y * phasor(-center_hz, fs, len(y), int(start_sample))
        if decimation > 1:
            y = resample(y, 1, decimation)
        if achieved_bandwidth is not None and achieved_bandwidth < 0.98 * analysis_rate:
            taps = int(np.clip(8 * analysis_rate / achieved_bandwidth, 31, 4095)) | 1
            h = signal.firwin(taps, achieved_bandwidth / 2, fs=analysis_rate)
            y = filter_same(y, h)
        first = math.ceil(start / decimation)
        last = min(len(y), math.ceil(end / decimation))
        x = y[first:last]
        sample_positions = decimation * np.arange(first, last, dtype=np.int64)
    else:
        x = np.asarray(samples[start:end])
        sample_positions = np.arange(start, end, dtype=np.int64)
    # The unwrapped phase is the running sum of the per-sample phase steps the
    # FM detector measures (identical to np.unwrap, at half the cost).
    step=np.angle(x[1:]*np.conj(x[:-1]))
    phase=float(np.angle(x[0]))+np.r_[0.,np.cumsum(step,dtype=np.float64)] if len(x) else np.zeros(0)
    fm=np.r_[0.,step*(analysis_rate/(2*np.pi))]
    amplitude=np.abs(x)
    # Phase/frequency of near-zero samples is undefined; mask rather than invent frequency.
    valid=amplitude>max(float(amplitude.max())*.01,1e-12)
    valid_frequency = valid.copy()
    valid_frequency[0] = False
    valid_frequency[1:] &= valid[:-1]
    fm[~valid_frequency]=0
    ix=display_indices(len(x),max_points)
    return {'time_s':(sample_positions[ix]/fs).tolist(),'i':x.real[ix].tolist(),'q':x.imag[ix].tolist(),
        'amplitude':amplitude[ix].tolist(),'phase_deg':np.rad2deg(phase[ix]).tolist(),
        'frequency_hz':fm[ix].tolist(),'valid':valid[ix].tolist(),
        'frequency_valid':valid_frequency[ix].tolist(),
        'sample_rate':fs,'analysis_rate':analysis_rate,'decimation':decimation,
        'capture_start_s':start/fs,'capture_duration_s':(end-start)/fs,
        'center_hz':float(center_hz),'bandwidth_hz':None if bandwidth_hz is None else float(bandwidth_hz),
        'bandwidth_actual_hz':achieved_bandwidth,
        'rms':float(np.sqrt(np.mean(amplitude**2))), 'peak':float(amplitude.max())}
