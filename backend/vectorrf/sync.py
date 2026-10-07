from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import cmath
import math
import numpy as np
from scipy import fft as sfft
from scipy import signal

from .dsp import add_awgn, fractional_delay, phasor, rrc_taps
from .models import ChannelConfig, CostasConfig


@dataclass
class CostasResult:
    output: np.ndarray
    error: np.ndarray
    phase: np.ndarray
    freq_hz: np.ndarray
    # Loop state after the last symbol (phase in rad, frequency in rad/symbol),
    # so tracking can continue in the next block.
    final_phase: float = 0.0
    final_freq: float = 0.0


@dataclass
class TimingResult:
    output: np.ndarray
    error: np.ndarray
    mu: np.ndarray
    # Input position (samples) of the next symbol instant after the last output.
    next_position: float = 0.0


def apply_channel(samples: np.ndarray, fs: float, cfg: ChannelConfig, seed: int = 123, start_s: float = 0,
                  start_sample: int | None = None) -> np.ndarray:
    """Receiver-side channel experiment.  The CFO phase uses absolute time.

    ``start_sample`` (absolute index of the first sample) takes precedence over
    ``start_s`` and keeps the CFO phase exact at any rate or simulated time.
    """
    # complex64 throughout: 24-bit mantissas are far below every modelled impairment.
    x = np.array(samples, dtype=np.complex64)
    if cfg.timing_offset_samples:
        x = fractional_delay(x, cfg.timing_offset_samples)
    if cfg.freq_offset_hz or cfg.phase_offset_deg:
        start = int(round(start_s * fs)) if start_sample is None else int(start_sample)
        x *= phasor(cfg.freq_offset_hz, fs, x.size, start, np.deg2rad(cfg.phase_offset_deg)).astype(np.complex64)
    if cfg.iq_gain_db or cfg.iq_phase_deg:
        # Educational IQ imbalance model: unequal I/Q gains plus quadrature phase error.
        g = 10**(cfg.iq_gain_db/20)
        phi = np.deg2rad(cfg.iq_phase_deg)
        i = x.real * np.float32(g)
        q = x.imag / np.float32(max(g, 1e-12))
        x = (i + 1j*(q*np.float32(np.cos(phi)) + i*np.float32(np.sin(phi)))).astype(np.complex64)
    if cfg.dc_i or cfg.dc_q:
        x += np.complex64(cfg.dc_i + 1j*cfg.dc_q)
    if cfg.snr_db is not None:
        x = add_awgn(x, cfg.snr_db, np.random.default_rng(seed))
    return x.astype(np.complex64, copy=False)


def costas_loop(samples: np.ndarray, fs: float, cfg: CostasConfig, phase0: float = 0.0, freq0: float = 0.0) -> CostasResult:
    x = np.asarray(samples, dtype=np.complex64)
    out, err_log, phase_log, freq_log = [], [], [], []
    phase, freq = float(phase0), float(freq0)
    alpha, beta, bpsk = cfg.alpha, cfg.beta, cfg.order == 2
    for s in x.astype(np.complex128).tolist():
        z = s * cmath.exp(-1j * phase)
        if bpsk:
            err = z.real * z.imag
        else:
            err = (1.0 if z.real >= 0 else -1.0) * z.imag - (1.0 if z.imag >= 0 else -1.0) * z.real
        freq += beta * err
        phase = (phase + freq + alpha * err + math.pi) % (2 * math.pi) - math.pi
        out.append(z); err_log.append(err); phase_log.append(phase); freq_log.append(freq)
    return CostasResult(np.asarray(out, np.complex64), np.asarray(err_log, np.float32), np.asarray(phase_log, np.float32),
                        (np.asarray(freq_log) * (fs / (2 * np.pi))).astype(np.float32), phase, freq)


def mueller_muller(samples: np.ndarray, sps: int, gain: float = 0.015, position0: float = 0.0) -> TimingResult:
    """Educational Mueller & Müller symbol timing recovery.

    The implementation follows the decision-directed form commonly used in SDR
    teaching material.  It deliberately logs the timing error and fractional
    phase accumulator so the UI can show acquisition rather than only the final
    sampled constellation.  ``gain`` is in symbols per unit detector output
    (see :func:`timing_settling_symbols`).
    """
    x = np.asarray(samples, dtype=np.complex64)
    sps = max(2, int(sps))
    max_out = x.size // sps + 16
    out = np.zeros(max_out, dtype=np.complex64)
    rail = np.zeros(max_out, dtype=np.complex64)
    err_log = np.zeros(max_out, dtype=np.float32)
    mu_log = np.zeros(max_out, dtype=np.float32)
    i_in = int(position0)
    i_out = 2
    mu = float(position0) - i_in
    while i_out < max_out and i_in + 2 < x.size:
        out[i_out] = x[i_in]
        rail[i_out] = (1.0 if out[i_out].real >= 0 else -1.0) + 1j * (1.0 if out[i_out].imag >= 0 else -1.0)
        a = (rail[i_out] - rail[i_out - 2]) * np.conj(out[i_out - 1])
        b = (out[i_out] - out[i_out - 2]) * np.conj(rail[i_out - 1])
        err = float(np.real(b - a))
        mu += sps * (1.0 + float(np.clip(gain * err, -0.25, 0.25)))
        advance = max(1, int(np.floor(mu)))
        i_in += advance
        mu -= advance
        err_log[i_out] = err
        mu_log[i_out] = mu
        i_out += 1
    return TimingResult(out[2:i_out], err_log[2:i_out], mu_log[2:i_out], float(i_in + mu))


@dataclass
class JointResult:
    timing: TimingResult
    carrier: CostasResult


def _nearest_ref(z: complex, refs: list[complex]) -> complex:
    return min(refs, key=lambda r: abs(z - r))


def mueller_muller_carrier(samples: np.ndarray, sps: float, timing_gain: float, carrier: CostasConfig,
                           symbol_rate: float, order: int = 4, refs: np.ndarray | None = None,
                           position0: float = 0.0, phase0: float = 0.0, freq0: float = 0.0) -> JointResult:
    """Joint Mueller & Müller timing and carrier tracking.

    M&M timing is decision-directed, so its decisions must be taken on
    carrier-corrected symbols: with a residual frequency offset the raw
    constellation rotates through the decision boundaries and biases the
    timing detector (a 50 Hz offset at 50 kBd used to give ~30 % EVM).  Each
    symbol is interpolated at the current timing estimate, derotated by the
    current carrier phase, sliced, and both loops are updated from that one
    decision.  ``refs`` selects a decision-directed phase detector (any
    constellation); otherwise a Costas detector of ``order`` 2 or 4 is used.
    Input should be normalized to unit-RMS symbols so the loop gains are
    level-independent; ``timing_gain`` is in symbols per unit detector output,
    so the timing loop behaves the same at any oversampling.
    """
    x = np.asarray(samples, dtype=np.complex128).tolist()
    n_max = int(len(x) / sps) + 2
    ref_list = None if refs is None else np.asarray(refs, np.complex128).tolist()
    root_half = math.sqrt(0.5)
    alpha, beta = carrier.alpha, carrier.beta
    out, t_err, mus, c_err, phases, freqs = [], [], [], [], [], []
    phase, freq = float(phase0), float(freq0)
    position = float(position0)
    z1 = d1 = 0j
    k = 0
    while k < n_max and position + 1 < len(x):
        i = int(position)
        frac = position - i
        z = (x[i] * (1.0 - frac) + x[i + 1] * frac) * cmath.exp(-1j * phase)
        # Decision and carrier phase error from the same carrier-corrected symbol.
        if ref_list is not None:
            d = _nearest_ref(z, ref_list)
            ce = cmath.phase(z * d.conjugate())
        elif order == 2:
            d = 1.0 if z.real >= 0 else -1.0
            ce = z.real * z.imag
        else:
            si, sq = (1.0 if z.real >= 0 else -1.0), (1.0 if z.imag >= 0 else -1.0)
            d = complex(si, sq) * root_half
            ce = si * z.imag - sq * z.real
        e = (d1.conjugate() * z - d.conjugate() * z1).real if k else 0.0
        freq += beta * ce
        phase = (phase + freq + alpha * ce + math.pi) % (2 * math.pi) - math.pi
        out.append(z); t_err.append(e); mus.append(frac); c_err.append(ce); phases.append(phase); freqs.append(freq)
        # Clamp the per-symbol correction so a noisy decision cannot skip a symbol.
        position += sps * (1.0 + min(0.25, max(-0.25, timing_gain * e)))
        z1, d1 = z, d
        k += 1
    symbols = np.asarray(out, np.complex64)
    return JointResult(TimingResult(symbols, np.asarray(t_err, np.float32), np.asarray(mus, np.float32), position),
                       CostasResult(symbols, np.asarray(c_err, np.float32), np.asarray(phases, np.float32),
                                    (np.asarray(freqs) * (symbol_rate / (2 * np.pi))).astype(np.float32), phase, freq))


def symbol_rms(samples: np.ndarray, sps: int) -> float:
    """RMS of matched-filter samples at the best sampling phase (≈ symbol RMS)."""
    x = np.asarray(samples)
    sps = max(1, int(sps))
    if not len(x):
        return 1.0
    m = (len(x) // sps) * sps
    if m < sps:
        return float(np.sqrt(np.mean(np.abs(x) ** 2) + 1e-30))
    power = np.mean(np.abs(x[:m].reshape(-1, sps)) ** 2, axis=0)
    return float(np.sqrt(np.max(power) + 1e-30))


# Slope of the timing error detectors for unit-RMS root-raised-cosine symbols,
# per symbol of timing offset (measured S-curves; independent of oversampling).
# Mueller & Müller: about 1.6–1.95 for roll-off 0.2–0.5; Gardner: about 2.9·β.
TED_GAIN_MM = 1.7


def timing_settling_symbols(gain: float, detector: str, rolloff: float = 0.35) -> int:
    """Symbols for a first-order timing loop to settle (≈ 4 time constants).

    The loop updates the timing offset τ (in symbols) by ``gain·e`` with a
    detector output ``e ≈ K·τ``, so it converges as ``(1 − gain·K)ᵏ``.
    """
    k = TED_GAIN_MM if detector == "mueller-muller" else max(0.3, 2.9 * rolloff)
    return int(np.ceil(4.0 / max(gain * k, 1e-4)))


def loop_settling_symbols(alpha: float, beta: float, detector_gain: float) -> int:
    """Symbols for a second-order carrier loop to settle (≈ 4 / ζωn).

    The loop updates ``f += β·e`` and ``φ += f + α·e`` with ``e ≈ Kd·θ``, so
    ``ωn = √(β·Kd)`` and ``ζ = α·Kd / (2ωn)`` in radians per symbol.
    """
    wn = np.sqrt(max(beta * detector_gain, 1e-12))
    zeta = max(alpha * detector_gain / (2 * wn), 1e-3)
    return int(np.ceil(4.0 / (zeta * wn)))


def matched_filter(samples: np.ndarray, sps: int, rolloff: float, span_symbols: int = 10) -> np.ndarray:
    taps = rrc_taps(sps, rolloff, span_symbols)
    return signal.lfilter(taps, [1.0], samples).astype(np.complex64)


def symbol_sample(samples: np.ndarray, sps: int, phase: float = 0.0, group_delay: int = 0) -> np.ndarray:
    """Known-rate symbol sampler with a user-tunable fractional symbol phase."""
    x = np.asarray(samples)
    offset = group_delay + int(round((phase + 0.5) * (sps-1)))
    offset = int(np.clip(offset, 0, max(0, sps-1 + group_delay)))
    return x[offset::sps].astype(np.complex64)


def eye_segments(samples: np.ndarray, sps: int, traces: int = 128, symbols_per_trace: int = 2) -> np.ndarray:
    x = np.asarray(samples)
    length = max(2, sps*symbols_per_trace)
    if x.size < length:
        return np.empty((0, length), dtype=np.complex64)
    starts = np.linspace(0, x.size-length, min(traces, max(1, x.size//length)), dtype=int)
    return np.stack([x[s:s+length] for s in starts]).astype(np.complex64)


def gardner(samples: np.ndarray, sps: float, gain: float = .015, phase: float = 0, position0: float | None = None) -> TimingResult:
    """Non-data-aided timing detector with linear fractional interpolation.

    ``gain`` is in symbols per unit detector output (see :func:`timing_settling_symbols`).
    """
    x=np.asarray(samples,dtype=np.complex128)
    x=x/np.sqrt(np.mean(np.abs(x)**2)+1e-15)
    def interp(t):
        k=int(t); u=t-k
        return x[k]*(1-u)+x[k+1]*u
    t=max(sps, sps+phase*sps) if position0 is None else max(float(sps), float(position0))
    previous=interp(t-sps)
    out=[]; errors=[]; mus=[]
    while t+sps/2+1<len(x):
        current=interp(t); middle=interp(t-sps/2)
        error=float(np.real((previous-current)*np.conj(middle)))
        correction=sps*float(np.clip(gain*error,-.2,.2))
        out.append(current); errors.append(error); mus.append((t % sps)/sps)
        previous=current
        t+=sps+correction
    return TimingResult(np.asarray(out,np.complex64),np.asarray(errors,np.float32),np.asarray(mus,np.float32),float(t))


@lru_cache(maxsize=8)
def _hann(n: int) -> np.ndarray:
    w = np.hanning(n)
    w.flags.writeable = False
    return w


def feedforward_phase(symbols, order, predicted=0.0):
    """Carrier phase of unit-RMS symbols by the M-th power (Viterbi & Viterbi)
    estimator, on the branch of its 2π/M ambiguity nearest ``predicted``."""
    z=np.asarray(symbols,dtype=np.complex128)
    if not len(z):
        return float(predicted)
    estimate=float(np.angle(np.sum(z**order)))/order
    step=2*math.pi/order
    k=round((predicted-estimate)/step)
    return (estimate+k*step+math.pi)%(2*math.pi)-math.pi


def spectral_centre(samples, fs, width, search_hz=None):
    """Carrier of a band-limited signal from its power spectrum.

    The band of ``width`` Hz (the occupied bandwidth, e.g. Rs(1 + β)) that
    holds the most power above the noise floor is located with a Welch
    periodogram, and its power centroid returned.  Its centre may lie at most
    ``search_hz`` from 0 Hz (default: the whole band), so a stronger
    neighbouring signal outside the search range is ignored.

    Unlike an M-th power estimator it has no ambiguity, but a payload whose
    spectrum is not symmetric (a short repeating text) biases it by a few
    percent of the symbol rate: it arbitrates between the branches of the
    precise estimators rather than replacing them.
    """
    x = np.asarray(samples)[:1 << 18]
    n = len(x)
    if n < 64 or width <= 0:
        return 0.0
    nper = int(min(n, max(256, 1 << int(np.ceil(np.log2(8.0 * fs / width))))))
    f, p = signal.welch(x, fs=fs, nperseg=nper, noverlap=nper // 2, return_onesided=False, detrend=False, window="hann")
    f, p = sfft.fftshift(f), sfft.fftshift(p)
    q = np.maximum(p - np.median(p), 0.0)
    m = max(1, int(round(width / (f[1] - f[0]))))
    if m >= len(q):
        return float(np.sum(q * f) / max(float(np.sum(q)), 1e-300))
    c = np.r_[0.0, np.cumsum(q)]
    energy = c[m:] - c[:-m]                     # window starting at bin k
    centres = f[:len(energy)] + (m - 1) * (f[1] - f[0]) / 2
    if search_hz is not None:
        energy = np.where(np.abs(centres) <= max(search_hz, f[1] - f[0]), energy, -1.0)
    k = int(np.argmax(energy))
    sl = slice(max(0, k - m // 2), min(len(q), k + m + m // 2))
    w = q[sl]
    total = float(np.sum(w))
    return float(np.sum(w * f[sl]) / total) if total > 0 else float(centres[k])


def differential_cfo(symbols, symbol_rate, modulation):
    """Residual carrier frequency from consecutive-symbol products.

    For differential PSK the phase steps are multiples of a known angle, so
    the M-th power of z[n]·z*[n−1] is a constant rotated by M·2πΔf/Rs, with
    no dependence on the carrier phase.  Range ±Rs/(2M) (π/4-DQPSK and
    DQPSK: ±Rs/8; DBPSK: ±Rs/4).
    """
    z = np.asarray(symbols, dtype=np.complex128)
    if len(z) < 8:
        return 0.0
    d = z[1:] * np.conj(z[:-1])
    if modulation == "dbpsk":
        m, offset = 2, 0.0
    elif modulation == "pi4-dqpsk":
        m, offset = 4, math.pi            # steps ±π/4, ±3π/4: (·)^4 = −1
    else:
        m, offset = 4, 0.0                 # DQPSK steps are multiples of π/2
    moment = np.mean(d ** m) * cmath.exp(-1j * offset)
    return float(np.angle(moment) / (2 * math.pi * m) * symbol_rate)


def coarse_carrier(samples, fs, order=4):
    """M-th power FFT estimator; acquisition range is ±Fs/(2M)."""
    x=np.asarray(samples)[:131072]
    nfft=1<<int(np.ceil(np.log2(max(256,len(x)))))
    p=x.astype(np.complex128)
    if order in (2,4,8):
        for _ in range(int(math.log2(order))):p=p*p
    else:
        p=p**order
    p*=_hann(len(p))
    power=np.abs(sfft.fft(p,nfft,workers=-1))
    k=int(np.argmax(power))
    delta=0.
    if power[(k-1)%nfft]>0 and power[(k+1)%nfft]>0:
        a,b,c=np.log(power[[(k-1)%nfft,k,(k+1)%nfft]]+1e-30)
        delta=float(np.clip(.5*(a-c)/(a-2*b+c+1e-30),-.5,.5))
    index=(k+delta+nfft/2)%nfft-nfft/2
    return float(index*fs/(nfft*order))


def decision_carrier(samples, fs, refs, cfg, phase0=0.0, freq0=0.0):
    x=np.asarray(samples,dtype=np.complex64)
    x=x/np.sqrt(np.mean(np.abs(x)**2)+1e-15)
    out=np.empty_like(x); errors=np.empty(len(x)); phases=np.empty(len(x)); frequencies=np.empty(len(x))
    phase=float(phase0); frequency=float(freq0)
    ref_list=np.asarray(refs,np.complex128).tolist()
    for i,s in enumerate(x.astype(np.complex128).tolist()):
        z=s*cmath.exp(-1j*phase)
        err=cmath.phase(z*_nearest_ref(z,ref_list).conjugate())
        frequency+=cfg.beta*err
        phase=(phase+frequency+cfg.alpha*err+math.pi)%(2*math.pi)-math.pi
        out[i]=z; errors[i]=err; phases[i]=phase; frequencies[i]=frequency*fs/(2*np.pi)
    return CostasResult(out,errors,phases,frequencies,phase,frequency)


def cma_equalizer(samples, taps=7, gain=.002, weights0=None):
    """Symbol-spaced blind constant-modulus equalizer; best suited to PSK.

    ``weights0`` continues from a previous block's weights (default: a pass-through).
    """
    x=np.asarray(samples,dtype=np.complex128)
    x=x/np.sqrt(np.mean(np.abs(x)**2)+1e-15)
    taps=int(taps)|1
    weights=np.zeros(taps,complex); weights[0]=1
    if weights0 is not None and len(weights0)==taps and np.all(np.isfinite(weights0)):
        weights=np.array(weights0,dtype=complex)
    padded=np.pad(x,(taps-1,0)); y=np.zeros(len(x),complex); error=np.zeros(len(x))
    for n in range(len(x)):
        u=padded[n:n+taps][::-1]
        z=np.dot(weights.conj(),u)
        e=float(np.clip(1-abs(z)**2,-4,4))
        weights+=gain*e*np.conj(z)*u/(np.vdot(u,u).real+1e-9)
        if np.linalg.norm(weights)>8: weights*=8/np.linalg.norm(weights)
        y[n]=z;error[n]=e
    return y.astype(np.complex64),error,weights
