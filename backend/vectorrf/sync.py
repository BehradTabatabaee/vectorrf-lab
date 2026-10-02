from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal

from .dsp import add_awgn, fractional_delay, rrc_taps
from .models import ChannelConfig, CostasConfig


@dataclass
class CostasResult:
    output: np.ndarray
    error: np.ndarray
    phase: np.ndarray
    freq_hz: np.ndarray


@dataclass
class TimingResult:
    output: np.ndarray
    error: np.ndarray
    mu: np.ndarray


def apply_channel(samples: np.ndarray, fs: float, cfg: ChannelConfig, seed: int = 123, start_s: float = 0) -> np.ndarray:
    x = np.asarray(samples, dtype=np.complex64).astype(np.complex128)
    n = np.arange(x.size)
    if cfg.timing_offset_samples:
        x = fractional_delay(x, cfg.timing_offset_samples).astype(np.complex128)
    if cfg.freq_offset_hz or cfg.phase_offset_deg:
        phase = 2*np.pi*cfg.freq_offset_hz*(n/fs+start_s) + np.deg2rad(cfg.phase_offset_deg)
        x *= np.exp(1j*phase)
    if cfg.iq_gain_db or cfg.iq_phase_deg:
        # Educational IQ imbalance model: unequal I/Q gains plus quadrature phase error.
        g = 10**(cfg.iq_gain_db/20)
        phi = np.deg2rad(cfg.iq_phase_deg)
        i = x.real * g
        q = x.imag / max(g, 1e-12)
        x = i + 1j*(q*np.cos(phi) + i*np.sin(phi))
    x += cfg.dc_i + 1j*cfg.dc_q
    if cfg.snr_db is not None:
        x = add_awgn(x, cfg.snr_db, np.random.default_rng(seed))
    return x.astype(np.complex64)


def costas_loop(samples: np.ndarray, fs: float, cfg: CostasConfig) -> CostasResult:
    x = np.asarray(samples, dtype=np.complex64)
    out = np.empty_like(x)
    err_log = np.empty(x.size, dtype=np.float32)
    phase_log = np.empty(x.size, dtype=np.float32)
    freq_log = np.empty(x.size, dtype=np.float32)
    phase = 0.0
    freq = 0.0
    for i, s in enumerate(x):
        z = s * np.exp(-1j*phase)
        out[i] = z
        if cfg.order == 2:
            err = float(z.real * z.imag)
        else:
            a = 1.0 if z.real >= 0 else -1.0
            b = 1.0 if z.imag >= 0 else -1.0
            err = float(a*z.imag - b*z.real)
        freq += cfg.beta * err
        phase += freq + cfg.alpha * err
        phase = (phase + np.pi) % (2*np.pi) - np.pi
        err_log[i] = err
        phase_log[i] = phase
        freq_log[i] = freq * fs / (2*np.pi)
    return CostasResult(out, err_log, phase_log, freq_log)


def mueller_muller(samples: np.ndarray, sps: int, gain: float = 0.30) -> TimingResult:
    """Educational Mueller & Müller symbol timing recovery.

    The implementation follows the decision-directed form commonly used in SDR
    teaching material.  It deliberately logs the timing error and fractional
    phase accumulator so the UI can show acquisition rather than only the final
    sampled constellation.
    """
    x = np.asarray(samples, dtype=np.complex64)
    sps = max(2, int(sps))
    max_out = x.size // sps + 16
    out = np.zeros(max_out, dtype=np.complex64)
    rail = np.zeros(max_out, dtype=np.complex64)
    err_log = np.zeros(max_out, dtype=np.float32)
    mu_log = np.zeros(max_out, dtype=np.float32)
    i_in = 0
    i_out = 2
    mu = 0.0
    while i_out < max_out and i_in + 2 < x.size:
        out[i_out] = x[i_in]
        rail[i_out] = (1.0 if out[i_out].real >= 0 else -1.0) + 1j * (1.0 if out[i_out].imag >= 0 else -1.0)
        a = (rail[i_out] - rail[i_out - 2]) * np.conj(out[i_out - 1])
        b = (out[i_out] - out[i_out - 2]) * np.conj(rail[i_out - 1])
        err = float(np.real(b - a))
        mu += sps + gain * err
        advance = max(1, int(np.floor(mu)))
        i_in += advance
        mu -= advance
        err_log[i_out] = err
        mu_log[i_out] = mu
        i_out += 1
    return TimingResult(out[2:i_out], err_log[2:i_out], mu_log[2:i_out])


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


def gardner(samples: np.ndarray, sps: float, gain: float = .1, phase: float = 0) -> TimingResult:
    """Non-data-aided timing detector with linear fractional interpolation."""
    x=np.asarray(samples,dtype=np.complex128)
    x=x/np.sqrt(np.mean(np.abs(x)**2)+1e-15)
    def interp(t):
        k=int(t); u=t-k
        return x[k]*(1-u)+x[k+1]*u
    t=max(sps, sps+phase*sps)
    previous=interp(t-sps)
    out=[]; errors=[]; mus=[]
    while t+sps/2+1<len(x):
        current=interp(t); middle=interp(t-sps/2)
        error=float(np.real((previous-current)*np.conj(middle)))
        correction=float(np.clip(gain*error,-.2*sps,.2*sps))
        out.append(current); errors.append(error); mus.append((t % sps)/sps)
        previous=current
        t+=sps+correction
    return TimingResult(np.asarray(out,np.complex64),np.asarray(errors,np.float32),np.asarray(mus,np.float32))


def coarse_carrier(samples, fs, order=4):
    """M-th power FFT estimator; acquisition range is ±Fs/(2M)."""
    x=np.asarray(samples)[:131072]
    nfft=1<<int(np.ceil(np.log2(max(256,len(x)))))
    p=x.astype(np.complex128)**order
    p*=np.hanning(len(p))
    power=np.abs(np.fft.fft(p,nfft))
    k=int(np.argmax(power))
    delta=0.
    if power[(k-1)%nfft]>0 and power[(k+1)%nfft]>0:
        a,b,c=np.log(power[[(k-1)%nfft,k,(k+1)%nfft]]+1e-30)
        delta=float(np.clip(.5*(a-c)/(a-2*b+c+1e-30),-.5,.5))
    index=(k+delta+nfft/2)%nfft-nfft/2
    return float(index*fs/(nfft*order))


def decision_carrier(samples, fs, refs, cfg):
    x=np.asarray(samples,dtype=np.complex64)
    x=x/np.sqrt(np.mean(np.abs(x)**2)+1e-15)
    out=np.empty_like(x); errors=np.empty(len(x)); phases=np.empty(len(x)); frequencies=np.empty(len(x))
    phase=0.; frequency=0.
    for i,s in enumerate(x):
        z=s*np.exp(-1j*phase)
        ref=refs[np.argmin(abs(z-refs))]
        err=float(np.angle(z*np.conj(ref)))
        frequency+=cfg.beta*err
        phase=(phase+frequency+cfg.alpha*err+np.pi)%(2*np.pi)-np.pi
        out[i]=z; errors[i]=err; phases[i]=phase; frequencies[i]=frequency*fs/(2*np.pi)
    return CostasResult(out,errors,phases,frequencies)


def cma_equalizer(samples, taps=7, gain=.002):
    """Symbol-spaced blind constant-modulus equalizer; best suited to PSK."""
    x=np.asarray(samples,dtype=np.complex128)
    x=x/np.sqrt(np.mean(np.abs(x)**2)+1e-15)
    taps=int(taps)|1
    weights=np.zeros(taps,complex); weights[0]=1
    padded=np.pad(x,(taps-1,0)); y=np.zeros(len(x),complex); error=np.zeros(len(x))
    for n in range(len(x)):
        u=padded[n:n+taps][::-1]
        z=np.dot(weights.conj(),u)
        e=float(np.clip(1-abs(z)**2,-4,4))
        weights+=gain*e*np.conj(z)*u/(np.vdot(u,u).real+1e-9)
        if np.linalg.norm(weights)>8: weights*=8/np.linalg.norm(weights)
        y[n]=z;error[n]=e
    return y.astype(np.complex64),error,weights
