from __future__ import annotations

import operator
import numpy as np
from scipy import signal

from .models import BlindResult
from .modem import bits_per_symbol, evm_with_decisions
from .spectrum import bandwidth_metrics
from .dsp import rrc_taps


# Every feature a rule can test, with the explanation the UI shows next to it.
FEATURES = {
    "spectral_flatness": ("Spectral flatness", "Geometric over arithmetic mean of the PSD: 1 for white noise, near 0 for line spectra."),
    "envelope_cv": ("Envelope variation", "Standard deviation of |x| divided by its mean: 0 for constant-envelope signals, 0.52 for Gaussian noise."),
    "papr_db": ("Peak-to-average power", "Crest factor of the analysed samples in dB."),
    "duty": ("Active duty", "Fraction of samples above 25 % of the peak magnitude."),
    "spectral_peaks": ("Spectral lines", "Prominent narrow peaks in the PSD (≥ 6 dB prominence, within 25 dB of the strongest)."),
    "envelope_kurtosis": ("Envelope kurtosis", "E|x|⁴ / (E|x|²)²: about 2 for Gaussian-like OFDM, below 1.6 for single-carrier PSK/QAM."),
    "phase_reversal_fraction": ("Steady/reversing phase", "Fraction of intra-pulse samples whose phase either holds or reverses by about π."),
    "phase_reversals": ("Phase reversals", "Near-π phase jumps inside the first pulse."),
    "steady_phase_fraction": ("Steady phase", "Fraction of intra-pulse samples with almost no phase change."),
    "frequency_r2": ("Linear frequency fit", "R² of a straight-line fit to the instantaneous frequency."),
    "frequency_resets": ("Sawtooth resets", "Abrupt downward jumps of the instantaneous frequency."),
    "rising_fraction": ("Rising ramp", "Fraction of steps where the smoothed instantaneous frequency increases."),
    "flat_fraction": ("Flat frequency", "Fraction of steps where the smoothed instantaneous frequency barely changes."),
    "frequency_span_fraction": ("Swept span", "Instantaneous-frequency range (2–98 %) as a fraction of the sample rate."),
    "fit_span_fraction": ("Fitted sweep", "Frequency range of the best straight-line fit to the instantaneous frequency, as a fraction of the sample rate."),
    "excess_envelope_cv2": ("Envelope variation beyond noise", "Envelope CV² minus the CV² additive noise alone would cause at the estimated SNR."),
    "fsk_levels": ("Frequency levels", "Clusters formed by symbol-averaged instantaneous frequency."),
    "fsk_separation": ("Level separation", "Fisher ratio of the frequency clusters (gap / pooled spread)."),
    "transition_fraction": ("Level transitions", "Fraction of symbols whose frequency level differs from the previous one; ≈1 means a periodic message."),
    "deviation_ratio": ("Deviation / (Rs/4)", "Measured peak deviation relative to MSK's Rs/4 (modulation index 0.5)."),
    "neighbour_isi": ("Neighbour ISI", "How much a symbol's frequency average depends on its neighbours; Gaussian (GMSK) shaping raises it."),
    "clock_peak_db": ("Symbol-clock line", "Prominence of the cyclostationary symbol-rate line in dB."),
    "samples_per_symbol": ("Samples per symbol", "Sample rate divided by the estimated symbol rate."),
    "constellation_evm": ("Constellation fit", "Residual EVM of the best constellation hypothesis after timing and carrier removal."),
}

_OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le, "==": operator.eq}


class _Stage:
    """One rule of the decision chain: its question, checks and outcome."""

    def __init__(self, record: dict):
        self.record = record

    def check(self, feature: str, value, op: str, threshold, unit: str = "") -> bool:
        value = float(value) if value is not None else float("nan")
        passed = bool(_OPS[op](value, threshold)) if np.isfinite(value) else False
        self.record["checks"].append({"feature": feature, "label": FEATURES.get(feature, (feature,))[0], "value": value,
                                      "op": op, "threshold": float(threshold), "unit": unit, "passed": passed})
        return passed

    def all(self, *passed: bool) -> bool:
        return all(passed)

    def conclude(self, outcome: str, detail: str = "", result: str | None = None) -> None:
        self.record.update(outcome=outcome, detail=detail, result=result)


class _Trace:
    """Ordered record of every rule the classifier evaluated."""

    def __init__(self):
        self.stages: list[dict] = []

    def stage(self, key: str, title: str, question: str) -> _Stage:
        record = {"key": key, "title": title, "question": question, "checks": [], "outcome": "rejected",
                  "detail": "", "result": None}
        self.stages.append(record)
        return _Stage(record)


def _periodogram_rate(feature: np.ndarray, fs: float, min_rate: float | None = None, max_rate: float | None = None) -> tuple[float | None, float]:
    v = np.asarray(feature, dtype=np.float64)
    if v.size < 128:
        return None, 0.0
    v = v - np.mean(v)
    if np.std(v) < 1e-12:
        return None, 0.0
    f, P = signal.periodogram(v, fs=fs, window="hann", detrend=False)
    lo = min_rate or fs/5000
    hi = max_rate or fs/3
    mask = (f >= lo) & (f <= hi)
    if not np.any(mask):
        return None, 0.0
    fm, pm = f[mask], P[mask]
    i = int(np.argmax(pm))
    floor = np.median(pm) + 1e-30
    prominence = float(10*np.log10((pm[i]+1e-30)/floor))
    if prominence < 5.0:
        return None, prominence
    rate = float(fm[i])
    if 0 < i < len(pm) - 1:
        # Sub-bin line frequency (parabolic fit of log power).
        a, b, c = np.log(pm[[i - 1, i, i + 1]] + 1e-30)
        rate += float(np.clip(0.5 * (a - c) / (a - 2 * b + c - 1e-30), -0.5, 0.5)) * float(f[1] - f[0])
    return rate, prominence


def estimate_symbol_rate_linear(x: np.ndarray, fs: float, bandwidth_hint_hz: float | None = None) -> tuple[float | None, float]:
    feat = np.abs(np.diff(x))**2
    if bandwidth_hint_hz and bandwidth_hint_hz > 0:
        # For raised-cosine linear modulation B_occ is approximately Rs*(1+alpha), alpha in [0,1].
        # Constraining the cyclostationary search by occupied bandwidth avoids random low-frequency peaks.
        lo = max(fs/5000, 0.40*bandwidth_hint_hz)
        hi = min(fs/3, 1.20*bandwidth_hint_hz)
        return _periodogram_rate(feat, fs, min_rate=lo, max_rate=hi)
    return _periodogram_rate(feat, fs, min_rate=fs/5000, max_rate=fs/3)


def estimate_symbol_rate_fsk(x: np.ndarray, fs: float) -> tuple[float | None, float]:
    dphi = np.angle(x[1:]*np.conj(x[:-1])) * fs/(2*np.pi)
    feat = np.abs(np.diff(dphi))**2
    return _periodogram_rate(feat, fs, min_rate=fs/5000, max_rate=fs/3)


def _psd_features(x: np.ndarray, fs: float):
    nfft = min(16384, max(1024, 1 << int(np.ceil(np.log2(min(len(x), 16384))))))
    if len(x) < nfft:
        y = np.pad(x, (0, nfft-len(x)))
    else:
        y = x[:nfft]
    w = signal.windows.hann(nfft, sym=False)
    X = np.fft.fftshift(np.fft.fft(y*w, nfft))
    P = np.abs(X)**2 + 1e-30
    f = np.fft.fftshift(np.fft.fftfreq(nfft, 1/fs))
    floor = np.median(P)
    Ps = np.maximum(P-floor, 0)
    bw3,bw99,peak_hz,peak_db = bandwidth_metrics(f, Ps if np.any(Ps) else P)
    sf = float(np.exp(np.mean(np.log(P))) / np.mean(P))
    db = 10*np.log10(P)
    peak_floor=max(float(np.median(db)+10.0), float(np.max(db)-25.0))
    peaks,_ = signal.find_peaks(db, prominence=6.0, height=peak_floor, distance=max(2,nfft//2048))
    # SNR via robust white-noise floor estimate; intentionally a model, not a lab-calibrated measurement.
    sig = float(np.sum(Ps))
    noise = float(floor * len(P))
    snr = 10*np.log10(sig/noise) if sig > 0 and noise > 0 else None
    return f,P,peaks,sf,float(bw3),float(bw99),snr


def _active_duty(x: np.ndarray) -> float:
    a=np.abs(x)
    if not np.any(a): return 0.0
    return float(np.mean(a > 0.25*np.max(a)))


def _radar_kind(x: np.ndarray, fs: float, stage: _Stage) -> tuple[str,float,list[str]]:
    a=np.abs(x)
    active=a>0.35*np.max(a)
    idx=np.flatnonzero(active)
    if idx.size<8:
        return "pulse",0.5,["Too few active samples for intra-pulse classification."]
    # First contiguous active run.
    gaps=np.flatnonzero(np.diff(idx)>1)
    stop=(gaps[0]+1) if gaps.size else idx.size
    seg=x[idx[:stop]]
    if len(seg)<8:
        return "pulse",0.5,[]
    dphi=np.unwrap(np.angle(seg))[1:]-np.unwrap(np.angle(seg))[:-1]
    f=dphi*fs/(2*np.pi)
    t=np.arange(len(f))/fs
    phase_jumps=np.abs(np.angle(seg[1:]*np.conj(seg[:-1])))
    coded=stage.check("phase_reversal_fraction",np.mean((phase_jumps<0.25)|(phase_jumps>2.5)),">",0.85)
    reversals=stage.check("phase_reversals",np.sum(phase_jumps>2.5),">=",2)
    if coded and reversals:
        return "barker/pn-phase-coded",0.86,["Detected discrete near-π phase reversals inside a pulse."]
    if stage.check("steady_phase_fraction",np.mean(phase_jumps<0.25),">",0.88):
        return "simple-pulse",0.90,["Pulse has essentially constant intra-pulse phase/frequency."]
    if len(f)>10:
        co=np.polyfit(t,f,1)
        fit=np.polyval(co,t)
        ssr=np.sum((f-fit)**2); sst=np.sum((f-np.mean(f))**2)+1e-30
        r2=1-ssr/sst
        if stage.check("frequency_r2",r2,">",0.92):
            return "lfm",float(min(0.98,0.75+0.2*r2)),[f"Intra-pulse instantaneous frequency is strongly linear (R²={r2:.3f})."]
        return "nlfm",0.76,[f"Intra-pulse frequency is swept but not well fit by a line (R²={r2:.3f})."]
    return "radar-pulse",0.6,[]


def _continuous_chirp_kind(x: np.ndarray, fs: float, stage: _Stage) -> tuple[str|None,float,list[str]]:
    if len(x)<64: return None,0.0,[]
    inst=np.angle(x[1:]*np.conj(x[:-1]))*fs/(2*np.pi)
    if np.std(inst)<fs*1e-5: return None,0.0,["Instantaneous frequency is essentially constant."]
    k=5 if len(inst)>=5 else 3
    sm=signal.medfilt(inst,kernel_size=k)
    lo,hi=np.quantile(sm,[.02,.98]); span=float(hi-lo)
    if not stage.check("frequency_span_fraction",span/fs,">=",0.01): return None,0.0,[]
    d=np.diff(sm)
    reset_idx=np.flatnonzero(d < -0.35*span)
    keep=np.ones_like(d,dtype=bool); keep[reset_idx]=False
    if np.any(keep):
        positive=float(np.mean(d[keep] > max(1e-12,0.001*span)))
        nearflat=float(np.mean(np.abs(d[keep]) < 0.01*span))
    else:
        positive=0.0; nearflat=1.0
    if stage.all(stage.check("frequency_resets",len(reset_idx),">=",2),stage.check("rising_fraction",positive,">",0.55),
                 stage.check("flat_fraction",nearflat,"<",0.75)):
        return "fmcw",float(np.clip(0.72+0.04*len(reset_idx),0.72,0.96)),[f"Detected {len(reset_idx)} sawtooth frequency resets with predominantly rising ramps."]
    t=np.arange(len(sm))/fs
    co=np.polyfit(t,sm,1); fit=np.polyval(co,t)
    r2=1-np.sum((sm-fit)**2)/(np.sum((sm-np.mean(sm))**2)+1e-30)
    if stage.all(stage.check("frequency_r2",r2,">",0.94),stage.check("fit_span_fraction",np.ptp(fit)/fs,">",0.03)):
        return "chirp",float(min(0.98,0.78+0.18*r2)),[f"Instantaneous frequency follows a linear ramp (R²={r2:.3f})."]
    return None,0.0,[]

def _kmeans1d(v: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, float]:
    v=np.asarray(v,dtype=float)
    qlo,qhi=np.quantile(v,[0.02,0.98])
    centers=np.linspace(qlo,qhi,k)
    labels=np.zeros(len(v),dtype=int)
    for _ in range(20):
        labels=np.argmin(np.abs(v[:,None]-centers[None,:]),axis=1)
        new=centers.copy()
        for i in range(k):
            if np.any(labels==i): new[i]=np.mean(v[labels==i])
        if np.allclose(new,centers,rtol=1e-5,atol=1e-6): break
        centers=new
    order=np.argsort(centers)
    remap=np.empty_like(order); remap[order]=np.arange(k)
    labels=remap[labels]; centers=centers[order]
    resid=v-centers[labels]
    mse=float(np.mean(resid*resid))
    return centers,labels,mse

def _symbol_averages(inst: np.ndarray, sps: float, phase: float) -> np.ndarray:
    """Integrate-and-dump of a sampled frequency trace over fractional-length symbols."""
    c = np.r_[0.0, np.cumsum(inst, dtype=np.float64)]
    edges = phase + sps * np.arange(int((len(inst) - phase) // sps) + 1)
    edges = edges[edges <= len(inst)]
    if len(edges) < 3:
        return np.zeros(0)
    s = np.interp(edges, np.arange(len(c)), c)
    return np.diff(s) / sps


def _fisher_separation(values: np.ndarray, k: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Smallest adjacent-centre gap divided by the pooled within-cluster spread."""
    centers, labels, mse = _kmeans1d(values, k)
    if len(centers) < 2:
        return 0.0, centers, labels
    gap = float(np.min(np.diff(centers)))
    return gap / max(np.sqrt(mse), 1e-12 * max(1.0, float(np.ptp(values)))), centers, labels


def occupied_band(f: np.ndarray, P: np.ndarray, above_floor_db: float = 6.0) -> tuple[float, float]:
    """Edges of the band where the smoothed PSD exceeds the noise floor by ``above_floor_db``.

    Unlike a 99 % power bandwidth this is not inflated by broadband noise
    (which holds a large share of total power at moderate SNR).
    """
    k = max(1, len(P) // 256)
    smooth = np.convolve(P, np.ones(k) / k, mode="same")
    floor = float(np.median(smooth))
    above = np.flatnonzero(smooth > floor * 10 ** (above_floor_db / 10))
    if not above.size:
        return float(f[0]), float(f[-1])
    peak = int(np.argmax(smooth))
    # Contiguous run around the strongest bin, tolerating small gaps (spectral nulls).
    gaps = np.flatnonzero(np.diff(above) > max(2, len(P) // 64))
    runs = np.split(above, gaps + 1)
    run = next((r for r in runs if r[0] <= peak <= r[-1]), above)
    return float(f[run[0]]), float(f[run[-1]])


def occupied_bands(f: np.ndarray, P: np.ndarray, above_floor_db: float = 10.0, min_fraction: float = 0.03) -> list[dict]:
    """Separate occupied bands (candidate emitters), strongest first.

    Runs of the smoothed PSD more than ``above_floor_db`` above the noise floor
    are merged across small gaps; runs carrying less than ``min_fraction`` of
    the excess power are ignored.
    """
    k = max(1, len(P) // 256)
    smooth = np.convolve(P, np.ones(k) / k, mode="same")
    floor = float(np.median(smooth))
    above = np.flatnonzero(smooth > floor * 10 ** (above_floor_db / 10))
    if not above.size:
        return []
    gaps = np.flatnonzero(np.diff(above) > max(2, len(P) // 64))
    excess = np.maximum(P - floor, 0)
    total = float(np.sum(excess)) or 1.0
    bands = []
    for run in np.split(above, gaps + 1):
        power = float(np.sum(excess[run[0]:run[-1] + 1])) / total
        if power >= min_fraction:
            lo, hi = float(f[run[0]]), float(f[run[-1]])
            bands.append({"center_hz": (lo + hi) / 2, "width_hz": hi - lo, "power_fraction": power})
    return sorted(bands, key=lambda b: b["power_fraction"], reverse=True)


def _cpfsk_hypothesis(x: np.ndarray, fs: float, band_hz: tuple[float, float] | None) -> dict | None:
    """Symbol-synchronous continuous-phase FSK test.

    1. Band-limit to the occupied bandwidth: FM-discriminator noise grows with
       the sampled bandwidth, so this removes most of it without knowing the
       symbol rate.
    2. Estimate the symbol rate from the cyclostationary line of the
       discriminator's transition energy.
    3. Integrate-and-dump the instantaneous frequency over each symbol at the
       timing phase with the widest eye; averaging over a symbol is what makes
       the levels separable at moderate SNR.
    4. Accept M = 2 or 4 levels when the symbol averages form well separated
       clusters (Fisher ratio).  A perfectly alternating symbol sequence is
       periodic modulation (e.g. sinusoidal FM), which is physically
       indistinguishable from 2-FSK carrying 0101 data, and is not accepted.
    """
    def band_limited(cutoff):
        y = np.asarray(x, dtype=np.complex128)
        if cutoff < 0.45 * fs:
            h = signal.firwin(129, cutoff, fs=fs)
            y = signal.fftconvolve(y, h, mode="same")[64:-64]
        return y

    half = 0.5 * fs if band_hz is None else max(abs(band_hz[0]), abs(band_hz[1]), 1e-3 * fs)
    # The symbol-rate cyclic feature lives at the band edges, so candidate
    # rates come from a wider filter than the level averaging.
    wide = band_limited(1.5 * half)
    candidates = _fsk_rate_candidates(wide, fs)
    y = band_limited(1.1 * half)
    inst = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi)
    if len(inst) < 256 or candidates is None:
        return None
    # Transition lines appear at every multiple of the symbol rate (the
    # strongest is sometimes 2·Rs), and structured payloads such as repeating
    # text add genuine lines at submultiples.  Starting from the strongest
    # line, step down to a subharmonic line only while symbol-synchronous
    # clustering stays as good: at a true subharmonic of a harmonic lock it
    # does, while a data-periodicity line blurs several symbols together.
    lines = candidates
    rs, prominence = lines.strongest
    h = _cpfsk_at_rate(inst, fs, rs)
    for _ in range(3):
        moved = False
        for m in (2, 3):
            sub = lines.near(rs / m)
            if sub is None:
                continue
            # A subharmonic of the current line is exactly rs/m; the measured
            # line only confirms it exists (its own bin is less precise).
            hs = _cpfsk_at_rate(inst, fs, rs / m)
            # Undoing a harmonic lock preserves the level count; averaging
            # several binary symbols instead creates new levels (±1, ±3 → a
            # spurious 4-FSK), so a level change is not accepted.
            if hs is not None and (h is None or (hs["levels"] == h["levels"] and hs["separation"] >= 0.9 * h["separation"])):
                rs, prominence, h, moved = rs / m, sub[1], hs, True
                break
        if not moved:
            break
    if h is None:
        for m in (2, 3, 4):
            up = lines.near(rs * m)
            if up is not None and (hu := _cpfsk_at_rate(inst, fs, rs * m)) is not None:
                rs, prominence, h = rs * m, up[1], hu
                break
    if h is not None:
        h["clock_peak_db"] = float(prominence)
        if h["levels"] == 2 and not h["periodic"]:
            # The narrower level-averaging filter also shrinks frequency
            # excursions, so measure deviation and ISI on the wider filter's
            # discriminator output.  Both are means over many symbols, so the
            # extra noise averages out.
            inst_wide = np.angle(wide[1:] * np.conj(wide[:-1])) * fs / (2 * np.pi)
            avg = _symbol_averages(inst_wide, fs / rs, h["phase"])
            peak, isi = _neighbour_statistics(avg)
            if peak is not None:
                h["peak_deviation_hz"], h["isi"] = peak, isi
    return h


def _neighbour_statistics(avg: np.ndarray) -> tuple[float | None, float]:
    """Full deviation and neighbour ISI of binary CPFSK symbol averages.

    Gaussian frequency pulses (GMSK) leak into neighbouring symbols: a symbol
    between two opposite neighbours averages well below one between equal
    neighbours.  Conditioning on the neighbours makes the measure immune to
    noise, which only spreads both groups symmetrically.
    """
    if len(avg) < 8:
        return None, 0.0
    # Midpoint of the two level centres: unlike the median it does not move
    # with an unbalanced payload (e.g. ASCII text, whose MSB is always 0).
    centers, _, _ = _kmeans1d(avg, 2)
    centre = float(np.mean(centers))
    sign = np.where(avg >= centre, 1, -1)
    mag = np.abs(avg - centre)
    same = (sign[:-2] == sign[1:-1]) & (sign[2:] == sign[1:-1])
    opposite = (sign[:-2] != sign[1:-1]) & (sign[2:] != sign[1:-1])
    if not np.any(same) or not np.any(opposite):
        return None, 0.0
    full = float(np.mean(mag[1:-1][same]))
    return full, (full - float(np.mean(mag[1:-1][opposite]))) / max(full, 1e-30)


class _CyclicLines:
    """Spectral lines of the FM-discriminator transition energy."""

    def __init__(self, y: np.ndarray, fs: float):
        dphi = np.angle(y[1:] * np.conj(y[:-1])) * fs / (2 * np.pi)
        v = np.abs(np.diff(dphi)) ** 2
        self.ok = v.size >= 128 and np.std(v) > 1e-12
        if not self.ok:
            return
        self.f, self.P = signal.periodogram(v - np.mean(v), fs=fs, window="hann", detrend=False)
        self.lo, self.hi = fs / 5000, fs / 3
        band = (self.f >= self.lo) & (self.f <= self.hi)
        self.ok = bool(np.any(band))
        if not self.ok:
            return
        self.floor = float(np.median(self.P[band])) + 1e-30
        self.df = float(self.f[1] - self.f[0])
        k = int(np.argmax(np.where(band, self.P, -1)))
        self.strongest = (self._refine(k), float(10 * np.log10(self.P[k] / self.floor)))
        self.ok = self.strongest[1] >= 5.0

    def _refine(self, k: int) -> float:
        """Sub-bin line frequency by parabolic interpolation of log power.

        Symbol averaging over N symbols tolerates a rate error well below
        1/N; whole-bin quantization (fs/len) would otherwise drift a long
        capture by a large fraction of a symbol.
        """
        if 0 < k < len(self.P) - 1:
            a, b, c = np.log(self.P[[k - 1, k, k + 1]] + 1e-30)
            delta = float(np.clip(0.5 * (a - c) / (a - 2 * b + c - 1e-30), -0.5, 0.5))
            return float(self.f[k] + delta * self.df)
        return float(self.f[k])

    def near(self, target: float) -> tuple[float, float] | None:
        """A genuine narrow line close to ``target``, judged against its neighbourhood."""
        if not self.lo <= target <= self.hi:
            return None
        f, P = self.f, self.P
        win = np.abs(f - target) <= max(2 * self.df, 0.01 * target)
        if not np.any(win):
            return None
        j = int(np.argmax(np.where(win, P, -1)))
        near = (np.abs(f - f[j]) < 0.2 * target) & (np.abs(f - f[j]) > 0.03 * target)
        local = float(np.median(P[near])) + 1e-30 if np.any(near) else self.floor
        prominence = float(10 * np.log10(P[j] / self.floor))
        # The caller confirms any line with symbol-synchronous clustering, so a
        # moderate test suffices here: clearly above the floor, within 15 dB of
        # the strongest line, and locally peaked.
        if prominence > max(8.0, self.strongest[1] - 15.0) and 10 * np.log10(P[j] / local) > 3.0:
            return self._refine(j), prominence
        return None


def _fsk_rate_candidates(y: np.ndarray, fs: float) -> "_CyclicLines | None":
    lines = _CyclicLines(y, fs)
    return lines if lines.ok else None


def _cpfsk_at_rate(inst: np.ndarray, fs: float, rs: float) -> dict | None:
    sps = fs / rs
    if sps < 2 or len(inst) / sps < 32:
        return None
    best = None
    best_phase = 0.0
    for phase in np.linspace(0, sps, 8, endpoint=False):
        avg = _symbol_averages(inst, sps, phase)
        if len(avg) >= 32 and (best is None or np.var(avg) > np.var(best)):
            best, best_phase = avg, float(phase)
    if best is None:
        return None
    lo, hi = np.quantile(best, [0.01, 0.99])
    trimmed = best[(best >= lo) & (best <= hi)]
    j2, c2, l2 = _fisher_separation(trimmed, 2)
    # Gap between the two clusters relative to their centre spacing.  Pulse
    # shaping (GMSK) widens clusters without closing the gap; a continuous
    # analog message has no gap at all (uniform values give j2 ≈ 3.5, gap ≈ 0).
    low2, high2 = trimmed[l2 == 0], trimmed[l2 == 1]
    opening = (float(np.quantile(high2, 0.02) - np.quantile(low2, 0.98)) / max(float(c2[1] - c2[0]), 1e-30)
               if low2.size and high2.size else 0.0)
    j4, c4, l4 = _fisher_separation(trimmed, 4)
    occupancy4 = np.bincount(l4, minlength=4) / max(1, len(l4))
    span = float(c4[-1] - c4[0]) if len(c4) > 1 else 0.0
    # A 2-level signal forced into four clusters leaves two nearly empty, so
    # the occupancy test (not a j4/j2 ratio) separates M = 4 from M = 2.
    if j4 > 5.0 and np.min(occupancy4) > 0.05 and np.min(np.diff(c4)) > 0.2 * span:
        levels, centers, labels, separation = 4, c4, l4, j4
    elif j2 > 5.0 or (j2 > 3.0 and opening > 0.25):
        levels, centers, labels, separation = 2, c2, l2, j2
    else:
        return None
    transitions = float(np.mean(labels[1:] != labels[:-1])) if len(labels) > 1 else 0.0
    if transitions > 0.95:
        return {"periodic": True, "rate": rs, "levels": levels, "separation": separation, "transition_fraction": transitions}
    deviation = float(centers[-1] - centers[0]) / 2
    peak_deviation, isi = deviation, 0.0
    if levels == 2:
        peak, isi = _neighbour_statistics(trimmed)
        peak_deviation = peak if peak is not None else deviation
    return {"periodic": False, "rate": float(rs), "levels": levels, "centers": centers, "separation": float(separation),
            "deviation_hz": deviation, "peak_deviation_hz": peak_deviation, "opening": opening, "isi": isi,
            "transition_fraction": transitions, "phase": best_phase}

_ORDER = {"bpsk": 2, "qpsk": 4, "16qam": 4, "64qam": 4, "8psk": 8}
# Phase of E[s^M] for each constellation in its reference orientation: BPSK
# s^2 = 1, 8-PSK s^8 = 1, QPSK/square QAM have a negative real fourth moment.
_MOMENT_PHASE = {2: 0.0, 4: np.pi, 8: 0.0}


def _derotate(syms: np.ndarray, order: int, frequency: bool = True) -> np.ndarray:
    """Remove residual carrier frequency and phase with an M-th power estimator.

    Raising symbols to the constellation's rotational order M strips the data
    modulation, leaving a tone at M·Δf; its FFT peak (parabolically refined)
    gives Δf in cycles/symbol, unambiguous within ±1/(2M).  The remaining
    common phase comes from the argument of the mean M-th power.  With
    ``frequency=False`` only the common phase is removed: periodic payloads put
    data lines into the M-th power spectrum that the frequency search can
    mistake for an offset, so callers compare both hypotheses.
    """
    z = np.asarray(syms, dtype=np.complex128)
    if z.size < 16:
        return z
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-30)
    if frequency:
        p = z ** order
        nfft = 1 << int(np.ceil(np.log2(4 * len(p))))
        spec = np.abs(np.fft.fft(p * np.hanning(len(p)), nfft))
        k = int(np.argmax(spec))
        a, b, c = np.log(spec[[(k - 1) % nfft, k, (k + 1) % nfft]] + 1e-30)
        delta = float(np.clip(0.5 * (a - c) / (a - 2 * b + c - 1e-30), -0.5, 0.5))
        cycles = (((k + delta) / nfft + 0.5) % 1.0 - 0.5) / order
        z = z * np.exp(-2j * np.pi * cycles * np.arange(len(z)))
    phase = (np.angle(np.mean(z ** order)) - _MOMENT_PHASE[order]) / order
    return z * np.exp(-1j * phase)


def _symbol_samples(mf: np.ndarray, period: float, phase: float, skip: int, count: int) -> np.ndarray:
    """Linearly interpolated samples at ``phase + k·period`` (fractional samples per symbol)."""
    pos = phase + period * np.arange(skip, skip + count)
    pos = pos[pos < len(mf) - 1]
    i = pos.astype(np.int64)
    frac = pos - i
    return mf[i] * (1 - frac) + mf[i + 1] * frac


def _eye_energy(mf: np.ndarray, period: float, phases: np.ndarray, count: int) -> float:
    return max(float(np.mean(np.abs(_symbol_samples(mf, period, ph, 12, count)) ** 2)) for ph in phases)


def _best_linear_mod(x: np.ndarray, fs: float, rs: float) -> tuple[str,float,dict]:
    estimate=fs/rs
    if estimate<2 or estimate>256:
        return "digital-linear",0.3,{}
    # Interactive AMC does not need hundreds of thousands of samples for a
    # constellation hypothesis. Keep a central stationary block and search a
    # bounded set of timing phases before the more expensive EVM comparisons.
    if len(x)>65536:
        mid=len(x)//2
        x=x[max(0,mid-32768):mid+32768]
    taps_sps=max(2,int(round(estimate)))
    count=int(min(1800,len(x)/estimate-14))
    if count<32:
        return "digital-linear",0.3,{}
    # Fine symbol-period search.  The clock line is accurate to ~1e-4, which
    # can still drift a long block by a large fraction of a symbol when the
    # samples per symbol are fractional (a band decimated by the analyzer);
    # keep the period whose eye is most open.  An integer period stays a
    # candidate, as generated signals use integer samples per symbol.
    probe=signal.lfilter(rrc_taps(taps_sps,0.35,8),[1.0],x)
    phases=np.linspace(0,estimate,12,endpoint=False)
    periods=[estimate*(1+d) for d in np.linspace(-1,1,9)/(2*count)]
    if abs(estimate-round(estimate))<0.05:
        periods.append(float(round(estimate)))
    sps=max(periods,key=lambda period:_eye_energy(probe,period,phases,count))
    best=(1e9,"digital-linear",0,None)
    scores = {}
    candidates=("bpsk","qpsk","8psk","16qam","64qam")
    step=sps/min(2*taps_sps,24)
    phase_grid=np.arange(min(2*taps_sps,24))*step
    for rolloff in (0.2,0.35,0.5):
        taps=rrc_taps(taps_sps,rolloff,8)
        mf=signal.lfilter(taps,[1.0],x)
        # Pick the strongest eye-opening phases first; this cuts repeated EVM
        # work substantially while preserving nearby timing hypotheses.  The
        # best grid phase is refined by a parabola through its neighbours.
        energy=np.array([np.mean(np.abs(_symbol_samples(mf,sps,ph,12,count))**2) for ph in phase_grid])
        k=int(np.argmax(energy))
        a,b,c=energy[[(k-1)%len(energy),k,(k+1)%len(energy)]]
        refined=(phase_grid[k]+step*float(np.clip(0.5*(a-c)/(a-2*b+c-1e-30),-0.5,0.5)))%sps
        candidates_phase=[refined,*phase_grid[np.argsort(energy)[::-1][:4]]]
        scored=[]
        for phase in candidates_phase:
            syms=_symbol_samples(mf,sps,phase,12,count)
            if len(syms)>=32:
                scored.append((float(np.mean(np.abs(syms)**2)),float(phase),syms))
        for _,phase,syms in scored:
            # Residual CFO would smear any constellation into a ring that fits
            # the densest grid best; remove it per rotational order first.
            derotated={order:(_derotate(syms,order),_derotate(syms,order,frequency=False)) for order in set(_ORDER.values())}
            for mod in candidates:
                # Residual-offset and phase-only hypotheses; keep the better fit.
                (m,idx)=min((evm_with_decisions(z,mod,robust=False) for z in derotated[_ORDER[mod]]),key=lambda fit:fit[0]["evm_rms_percent"])
                evm=m["evm_rms_percent"]
                # Nested constellations (BPSK ⊂ QPSK ⊂ 8-PSK after rotation)
                # fit a denser grid equally well, so minimum EVM alone always
                # prefers higher orders.  Random data on a true M-ary signal
                # uses all M points; a hypothesis that leaves points unused is
                # not supported by the observation.
                points=1<<bits_per_symbol(mod)
                used=float(np.mean(np.bincount(idx,minlength=points)>=0.2*len(idx)/points))
                score=evm+(100.0 if used<0.75 else 0.0)
                if score<scores.get(mod,(1e9,))[0]:
                    scores[mod]=(score,evm,used)
                if score<best[0]: best=(score,mod,phase,(rolloff,m))
    if best[3] is None:
        return "digital-linear",0.3,{}
    evm,mod,phase,(ro,m)=best
    conf=float(np.clip(1.0-evm/70.0,0.35,0.96))
    # ``score`` adds 100 points when fewer than 75 % of the hypothesis' points
    # are used: random data on a true M-ary signal visits all M points.
    return mod,conf,{"evm_rms_percent":float(m["evm_rms_percent"]),"sampling_phase":float(phase),"rrc_rolloff_candidate":ro,
                     "sps":float(sps),"symbol_rate":fs/sps,
                     "candidates":[{"modulation":name,"evm_percent":float(e),"score":float(sc),"points_used":u,"penalized":u<0.75}
                                   for name,(sc,e,u) in sorted(scores.items(),key=lambda item:item[1][0])]}


def isolate_band(x: np.ndarray, fs: float, center_hz: float, width_hz: float) -> tuple[np.ndarray, float]:
    """Mix ``center_hz`` to DC and low-pass to ``width_hz``; return samples and their rate.

    Narrow bands are also decimated by the spectrum analyzer's multistage
    down-converter (≥ 110 dB alias rejection) to at least eight times the band
    width, so the classifier sees about ten samples per symbol rather than
    thousands.  Wide bands, bands touching Nyquist and very short captures
    keep the full rate (linear-phase FIR, no start-up transient).
    """
    from .models import SpectrumConfig
    from .spectrum import down_convert, zoom_plan
    plan = zoom_plan(fs, SpectrumConfig(center_hz=center_hz, span_hz=min(fs, 1.15 * width_hz)), min_rate=8 * width_hz)
    if plan.stages and len(x) // plan.decimation >= 4096:
        # The down-converter only keeps the band itself alias-free; other
        # emitters inside the decimated Nyquist zone still need the band-edge
        # low-pass below.
        y, rate = down_convert(x, plan), plan.rate
    else:
        from .dsp import phasor
        y, rate = np.asarray(x, dtype=np.complex128) * phasor(-center_hz, fs, len(x)), fs
    h = signal.firwin(255, min(0.49 * rate, 0.5 * width_hz * 1.15), fs=rate)
    return signal.fftconvolve(y, h, mode="same").astype(np.complex64), rate


def analyze(samples: np.ndarray, fs: float, band: tuple[float, float] | None = None) -> BlindResult:
    """Heuristic single-emitter classification of a capture.

    ``band = (center_hz, width_hz)`` isolates one emitter first.  Without it,
    several separate occupied bands are reported (and the score lowered)
    because the features then describe their combination.  The result's
    ``trace`` lists every rule evaluated, with feature values and thresholds.
    """
    x=np.asarray(samples,dtype=np.complex64)
    if x.size<256:
        raise ValueError("blind analysis needs at least 256 complex samples")
    if band is not None:
        center,width=float(band[0]),float(band[1])
        if not 0<width<=fs:raise ValueError("Analysis band width must be positive and at most the sample rate")
        y,rate=isolate_band(x,fs,center,width)
        result=_analyze(y,rate)
        # The isolated band has no noise floor of its own (outside it the
        # filters leave only stopband), so measure SNR in the original capture:
        # band power above the capture's median noise density.
        f,P,*_=_psd_features(x[:131072],fs)
        inside=np.abs(f-center)<=width/2
        if np.any(inside):
            floor=float(np.median(P))
            signal_power=float(np.sum(np.maximum(P[inside]-floor,0)))
            noise_power=floor*int(np.sum(inside))
            result.snr_db=float(10*np.log10(signal_power/noise_power)) if signal_power>0 and noise_power>0 else None
        result.notes.insert(0,f"Isolated band {center:,.0f} Hz ± {width/2:,.0f} Hz (mixed to DC and low-pass filtered"
                            +(f", analysed at {rate:,.0f} S/s" if rate!=fs else "")+") before classification.")
        result.features["analysis_band"]={"center_hz":center,"width_hz":width,"analysis_rate":rate}
        return result
    result=_analyze(x,fs)
    probe=x[:131072]
    f,P,*_=_psd_features(probe,fs)
    bands=occupied_bands(f,P)
    result.features["bands"]=bands[:8]
    if len(bands)>1:
        result.confidence=float(result.confidence*0.6)
        result.notes.insert(0,f"{len(bands)} separate occupied bands detected: the features describe their combination. Choose an analysis band to classify one emitter.")
    return result


def _analyze(samples: np.ndarray, fs: float) -> BlindResult:
    x=np.asarray(samples,dtype=np.complex64)
    trace=_Trace()
    # Keep computational work bounded for interactive analysis while retaining stationarity cues.
    # 131k complex samples is enough for the current feature set and keeps the
    # classifier responsive when the source runs at several MS/s.
    if x.size>131072:
        start=(x.size-131072)//2; x=x[start:start+131072]
    # Bursty emitters (on for 30-90 % of the capture) are classified from their
    # longest burst: the on/off envelope would otherwise defeat the
    # constant-envelope and constellation tests.  Short-duty pulse trains are
    # left whole for the radar rules.
    burst_note=None
    smooth=np.convolve(np.abs(x),np.ones(64)/64,mode="same")
    on=smooth>0.25*float(np.max(smooth))
    on_fraction=float(np.mean(on))
    if 0.3<=on_fraction<=0.9:
        edges=np.flatnonzero(np.diff(on.astype(np.int8)))+1
        starts=np.r_[0,edges];stops=np.r_[edges,len(x)]
        lengths=np.where(on[starts],stops-starts,0)
        k=int(np.argmax(lengths))
        if lengths[k]>=2048+128:
            x=x[starts[k]+64:stops[k]-64]
            burst_note=f"Bursty emitter (on {100*on_fraction:.0f}% of the capture): classified its longest burst ({len(x):,} samples)."
    a=np.abs(x); p=a*a
    papr=10*np.log10((np.max(p)+1e-30)/(np.mean(p)+1e-30))
    cv=float(np.std(a)/(np.mean(a)+1e-30))
    duty=_active_duty(x)
    f,P,peaks,sflat,bw3,bw99,snr=_psd_features(x,fs)
    kurtosis=float(np.mean(p*p)/(np.mean(p)**2+1e-30))
    notes=[f"PAPR {papr:.1f} dB; envelope CV {cv:.2f}; spectral flatness {sflat:.3f}."]+([burst_note] if burst_note else [])
    features={"spectral_flatness":float(sflat),"envelope_kurtosis":kurtosis,"duty":duty,"spectral_peaks":int(len(peaks))}

    def result(modulation,family,confidence,stage,detail,symbol_rate=None,snr_db=snr,extra_notes=(),extra_features=None,**fields):
        stage.conclude("accepted",detail,modulation)
        return BlindResult(modulation=modulation,family=family,confidence=confidence,bandwidth_3db_hz=bw3,
            bandwidth_99_hz=bw99,symbol_rate_baud=symbol_rate,snr_db=snr_db,notes=notes+list(extra_notes)+[detail],
            trace=trace.stages,features={**features,**(extra_features or {})},**fields)

    # AWGN is checked before duty-cycle logic because a Rayleigh envelope naturally crosses amplitude thresholds often.
    s=trace.stage("noise","Noise","Is the capture spectrally flat with a Rayleigh-like envelope?")
    if s.all(s.check("spectral_flatness",sflat,">",0.45),s.check("envelope_cv",cv,">",0.42),s.check("papr_db",papr,">",6.0,"dB")):
        return result("awgn","noise",0.94,s,"Broad, spectrally flat complex noise with no stable carrier structure.",snr_db=None)

    # OFDM is noise-like in amplitude but has many stable occupied carriers;
    # distinguish it from sparse pulsed radar by duty.  Its sum of many
    # subcarriers makes I/Q nearly Gaussian, so the envelope kurtosis
    # E|x|^4/(E|x|^2)^2 is close to 2, whereas single-carrier PSK/QAM stays
    # below about 1.6 even for 64-QAM.  (Spectral peak counts alone mistook
    # 64-QAM periodogram fluctuations for subcarriers.)
    s=trace.stage("ofdm","OFDM","Is it noise-like in amplitude yet made of many stable subcarriers?")
    if s.all(s.check("papr_db",papr,">",6.0,"dB"),s.check("envelope_cv",cv,">",0.42),s.check("duty",duty,">",0.30),
             s.check("spectral_peaks",len(peaks),">",10),s.check("envelope_kurtosis",kurtosis,">",1.8)):
        return result("ofdm","digital/multicarrier",0.82,s,"High crest factor and many occupied spectral components are consistent with OFDM.")

    s=trace.stage("pulsed","Pulsed radar","Is the signal on for a small fraction of the time, in high-crest pulses?")
    if s.all(s.check("duty",duty,"<",0.30),s.check("papr_db",papr,">",5.0,"dB")):
        kind,conf,n=_radar_kind(x,fs,s)
        return result(kind,"radar/pulsed",conf,s,n[-1] if n else "Pulsed emission.",extra_notes=n[:-1])

    s=trace.stage("chirp","Continuous chirp / FMCW","Does a constant-envelope signal sweep its frequency linearly or in sawtooth ramps?")
    chirp,conf,n=_continuous_chirp_kind(x,fs,s)
    if s.check("envelope_cv",cv,"<",0.18) and chirp is not None:
        return result(chirp,"chirp/radar",conf,s,n[-1],extra_notes=n[:-1])
    if n:
        s.conclude("rejected",n[-1])

    # Sparse line spectra are intentionally handled before symbol-clock heuristics.
    # With an unknown message, tone/SSB and multitone/AM/DSB can be physically ambiguous.
    s=trace.stage("tone","Single tone","Is there one spectral line and almost no other content?")
    if s.all(s.check("spectral_peaks",len(peaks),"<=",1),s.check("spectral_flatness",sflat,"<",0.08)):
        return result("single-tone/ssb-like","tone/analog",0.72,s,"A pure SSB sinusoidal-message waveform is physically indistinguishable from a complex tone without side information.")
    s=trace.stage("lines","Line spectrum","Are there a few discrete lines (multitone or analog modulation of a tone)?")
    if s.all(s.check("spectral_peaks",len(peaks),">=",2),s.check("spectral_peaks",len(peaks),"<=",6),s.check("spectral_flatness",sflat,"<",0.08)):
        return result("multitone/am/dsb-like","tone/analog",0.64,s,"A sparse line spectrum can be a multitone waveform or analog modulation of a sinusoidal message; classification is ambiguous.")

    # Coarse tuning: move the occupied band's power centroid to DC, as a
    # receiver would before classifying an off-centre emission.  Symbol-rate
    # and constellation features are defined about the carrier, not DC.
    floor=float(np.median(P))
    excess=np.maximum(P-floor,0)
    offset=float(np.sum(f*excess)/np.sum(excess)) if np.sum(excess)>0 else 0.0
    if abs(offset)>2e-3*fs:
        x=(x*np.exp(-2j*np.pi*offset*np.arange(len(x))/fs)).astype(np.complex64)
        notes.append(f"Coarse-tuned the band centroid ({offset:,.0f} Hz) to DC before symbol analysis.")
    features["center_offset_hz"]=offset

    # Constant-envelope, non-chirp waveforms: detect FSK/MSK before analog angle
    # modulation.  Additive noise alone gives a constant-envelope signal an
    # envelope CV of about sqrt(1/(2(SNR+1))) (0.125 at 15 dB), so the test uses
    # the CV in excess of that noise contribution (capped where the SNR
    # estimate itself becomes unreliable).
    snr_lin=10**(snr/10) if snr is not None else np.inf
    noise_cv2=min(0.5/(snr_lin+1.0),0.04)
    s=trace.stage("angle","Constant envelope (FSK / angle modulation)","Beyond noise, is the envelope constant so information lives in phase or frequency?")
    if s.check("excess_envelope_cv2",cv*cv-noise_cv2,"<",0.12**2):
        lo_edge,hi_edge=occupied_band(f,P)
        h=_cpfsk_hypothesis(x,fs,(lo_edge-offset,hi_edge-offset))
        periodic=False
        if h is not None:
            s.check("fsk_levels",h["levels"],">=",2)
            s.check("fsk_separation",h["separation"],">",3.0)
            periodic=not s.check("transition_fraction",h.get("transition_fraction",0.0),"<=",0.95)
        if h is not None and not periodic:
            rs=h["rate"];levels=h["levels"];mod=f"{levels}fsk"
            s.check("clock_peak_db",h["clock_peak_db"],">=",5.0,"dB")
            ratio=h["peak_deviation_hz"]/max(rs/4,1)
            if levels==2 and s.all(s.check("deviation_ratio",ratio,">",0.75),s.check("deviation_ratio",ratio,"<",1.25)):
                # Peak deviation Rs/4 is MSK's modulation index 0.5; Gaussian
                # pulse shaping shows as symbol averages well inside the peak.
                mod="gmsk" if s.check("neighbour_isi",h["isi"],">",0.45) else "msk"
            conf=float(np.clip(0.55+0.04*h["separation"],0.55,0.94))
            detail=(f"Symbol rate from the frequency-transition cyclostationary line ({h['clock_peak_db']:.1f} dB)."
                    +(" Peak deviation ≈ Rs/4 (h = 0.5): minimum-shift keying." if mod in ("msk","gmsk") else "")
                    +(f" Neighbour-dependent symbol averages (ISI {h['isi']:.2f}) indicate Gaussian frequency-pulse shaping." if mod=="gmsk" else ""))
            return result(mod,"digital/cpfsk",conf,s,detail,symbol_rate=rs,
                extra_notes=[f"Symbol-synchronous frequency averages form {levels} clusters (Fisher separation {h['separation']:.1f}); peak deviation ±{h['peak_deviation_hz']:,.0f} Hz."],
                extra_features={"deviation_hz":h["peak_deviation_hz"],"level_separation":h["separation"],"neighbour_isi":h["isi"],"clock_peak_db":h["clock_peak_db"]})
        extra=["Frequency changes periodically: consistent with angle modulation by a periodic message, or FSK carrying a strictly alternating pattern."] if periodic else []
        # FM and PM are not identifiable from an arbitrary unknown message in general.
        return result("fm/pm-like","analog/angle",0.58,s,"Constant envelope with continuous phase/frequency variation; blind FM-vs-PM is message-dependent.",extra_notes=extra)

    # Linear digital signals have strong symbol-clock cyclostationarity in transition energy.
    rs,rprom=estimate_symbol_rate_linear(x,fs,bw99)
    s=trace.stage("linear","Linear digital (PSK / QAM)","Is there a symbol clock, and which constellation fits best?")
    clock=s.check("clock_peak_db",rprom,">",8.0,"dB")
    if rs and clock and s.all(s.check("samples_per_symbol",fs/rs,">=",2),s.check("samples_per_symbol",fs/rs,"<=",256)):
        mod,conf,extra=_best_linear_mod(x,fs,rs)
        s.check("constellation_evm",extra.get("evm_rms_percent",float("nan")),"<",70.0,"%")
        return result(mod,"digital/linear",conf,s,f"Best constellation fit: {mod}, estimated EVM {extra.get('evm_rms_percent',float('nan')):.1f}%.",
            symbol_rate=extra.get("symbol_rate",rs),extra_notes=[f"Estimated symbol rate from transition-energy cyclostationarity ({rprom:.1f} dB peak)."],
            candidates=extra.get("candidates",[]),extra_features={"clock_peak_db":float(rprom)})

    s=trace.stage("unknown","No rule matched","Fallback when no rule produced a reliable class.")
    return result("unknown","unknown",0.25,s,"No current MVP rule produced a reliable class; inspect the feature views or use a trained AMC plugin.",symbol_rate=rs)
