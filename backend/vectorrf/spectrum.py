from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy import signal

from .dsp import next_pow2
from .models import SpectrumConfig


@dataclass
class SpectrumResult:
    freq_hz: np.ndarray
    psd_db: np.ndarray
    nfft: int
    rbw_actual_hz: float
    bandwidth_3db_hz: float
    bandwidth_99_hz: float
    peak_hz: float
    peak_db: float
    unit: str = "dBFS/Hz"


@lru_cache(maxsize=64)
def _window(name: str, n: int) -> np.ndarray:
    if name == "hann":
        return signal.windows.hann(n, sym=False)
    if name == "blackmanharris":
        return signal.windows.blackmanharris(n, sym=False)
    if name == "flattop":
        return signal.windows.flattop(n, sym=False)
    if name == "kaiser":
        return signal.windows.kaiser(n, beta=8.6, sym=False)
    return np.ones(n)


def _enbw_bins(w: np.ndarray) -> float:
    return float(len(w) * np.sum(w*w) / (np.sum(w)**2 + 1e-30))


def choose_nfft(fs: float, rbw_hz: float, window_name: str, max_fft: int) -> tuple[int, float]:
    # Iterate once because ENBW depends only weakly on n for these windows.
    probe = _window(window_name, 4096)
    enbw = _enbw_bins(probe)
    nfft = min(max_fft, max(256, next_pow2(enbw * fs / rbw_hz)))
    w = _window(window_name, nfft)
    actual = _enbw_bins(w) * fs / nfft
    return int(nfft), float(actual)


def bandwidth_metrics(freq_hz: np.ndarray, power: np.ndarray) -> tuple[float, float, float, float]:
    power = np.asarray(power, dtype=np.float64)
    if power.size == 0 or not np.any(np.isfinite(power)):
        return 0.0, 0.0, 0.0, -300.0
    k = int(np.argmax(power))
    peak = power[k]
    threshold = peak / (10 ** (3/10))
    left = k
    while left > 0 and power[left] >= threshold:
        left -= 1
    right = k
    while right < power.size-1 and power[right] >= threshold:
        right += 1
    bw3 = abs(float(freq_hz[right] - freq_hz[left])) if right > left else 0.0

    total = np.sum(power)
    if total <= 0:
        bw99 = 0.0
    else:
        cdf = np.cumsum(power) / total
        lo = int(np.searchsorted(cdf, 0.005))
        hi = int(np.searchsorted(cdf, 0.995))
        hi = min(hi, len(freq_hz)-1)
        bw99 = abs(float(freq_hz[hi] - freq_hz[lo]))
    return bw3, bw99, float(freq_hz[k]), float(10*np.log10(peak + 1e-30))


def analyze(samples: np.ndarray, fs: float, cfg: SpectrumConfig, reference_power_dbm: float | None = None, reference_rms: float | None = None) -> SpectrumResult:
    """Analyze a baseband record.

    Without a calibration reference the result retains the historical dBFS/Hz
    PSD convention.  With ``reference_power_dbm`` it reports dBm per FFT bin; a
    coherent tone whose average sample power equals ``reference_rms`` therefore
    reads the configured transmit power at its peak marker.
    """
    x = np.asarray(samples, dtype=np.complex64)
    input_rms = float(np.sqrt(np.mean(np.abs(x.astype(np.complex128))**2))) if x.size else 0.0
    nfft, rbw_actual = choose_nfft(fs, cfg.rbw_hz, cfg.window, cfg.max_fft)
    if x.size < nfft:
        x = np.pad(x, (0, nfft-x.size))
    w = _window(cfg.window, nfft).astype(np.float32)
    step = max(1, nfft // 2)
    max_segments = max(1, cfg.averaging)
    starts = np.arange(0, max(1, x.size-nfft+1), step, dtype=int)
    if starts.size > max_segments:
        starts = starts[-max_segments:]
    spectra = []
    if reference_power_dbm is None:
        scale = (np.sum(w*w) * fs) + 1e-30
        calibration = 1.0
        db_offset = 0.0
        unit = "dBFS/Hz"
    else:
        # Bin-power normalization makes a bin-centered complex CW read its total
        # calibrated power.  Keep the absolute power as a dB-domain offset rather
        # than exponentiating dBm; this avoids artificial overflow limits for very
        # large/small finite simulator power settings.
        scale = (np.sum(w)**2) + 1e-30
        rr = float(reference_rms) if reference_rms is not None else input_rms
        calibration = 1.0 / max(rr*rr, 1e-30)
        db_offset = float(reference_power_dbm)
        unit = "dBm"
    for s in starts:
        block = x[s:s+nfft]
        if block.size < nfft:
            block = np.pad(block, (0, nfft-block.size))
        X = np.fft.fftshift(np.fft.fft(block*w, n=nfft))
        spectra.append(((np.abs(X)**2)/scale)*calibration)
    P = np.stack(spectra)
    if cfg.detector == "peak":
        p = np.max(P, axis=0)
    elif cfg.detector == "sample":
        p = P[-1]
    else:
        p = np.mean(P, axis=0)
    f = np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs))

    span = cfg.span_hz if cfg.span_hz is not None else fs
    span = min(float(span), fs)
    # Communications spectrum is baseband-only: DC is always the acquisition center.
    mask = (f >= -span/2) & (f <= span/2)
    f2 = f[mask].astype(np.float32)
    p2 = p[mask]
    db = (10*np.log10(p2 + 1e-30) + db_offset).astype(np.float32)
    bw3, bw99, peak_hz, peak_db = bandwidth_metrics(f2, p2)
    peak_db += db_offset
    return SpectrumResult(f2, db, nfft, rbw_actual, bw3, bw99, peak_hz, peak_db, unit)


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
