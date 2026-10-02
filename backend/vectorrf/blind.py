from __future__ import annotations

import math
import numpy as np
from scipy import signal

from .models import BlindResult
from .modem import evm_rms
from .spectrum import bandwidth_metrics, stft_ridge
from .dsp import rrc_taps


def _periodogram_rate(feature: np.ndarray, fs: float, min_rate: float | None = None, max_rate: float | None = None) -> tuple[float | None, float]:
    v = np.asarray(feature, dtype=np.float64)
    if v.size < 128:
        return None, 0.0
    v = v - np.mean(v)
    if np.std(v) < 1e-12:
        return None, 0.0
    f, P = signal.periodogram(v, fs=fs, window="hann", detrend=False)
    lo = min_rate or max(100.0, fs/2000)
    hi = max_rate or min(fs/3, 1_000_000.0)
    mask = (f >= lo) & (f <= hi)
    if not np.any(mask):
        return None, 0.0
    fm, pm = f[mask], P[mask]
    i = int(np.argmax(pm))
    floor = np.median(pm) + 1e-30
    prominence = float(10*np.log10((pm[i]+1e-30)/floor))
    if prominence < 5.0:
        return None, prominence
    return float(fm[i]), prominence


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


def _radar_kind(x: np.ndarray, fs: float) -> tuple[str,float,list[str]]:
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
    phase_coded=np.mean((phase_jumps<0.25)|(phase_jumps>2.5))>0.85 and np.sum(phase_jumps>2.5)>=2
    if phase_coded:
        return "barker/pn-phase-coded",0.86,["Detected discrete near-π phase reversals inside a pulse."]
    quiet=np.mean(phase_jumps<0.25)
    if quiet>0.88:
        return "simple-pulse",0.90,["Pulse has essentially constant intra-pulse phase/frequency."]
    if len(f)>10:
        co=np.polyfit(t,f,1)
        fit=np.polyval(co,t)
        ssr=np.sum((f-fit)**2); sst=np.sum((f-np.mean(f))**2)+1e-30
        r2=1-ssr/sst
        if r2>0.92:
            return "lfm",float(min(0.98,0.75+0.2*r2)),[f"Intra-pulse instantaneous frequency is strongly linear (R²={r2:.3f})."]
        return "nlfm",0.76,[f"Intra-pulse frequency is swept but not well fit by a line (R²={r2:.3f})."]
    return "radar-pulse",0.6,[]


def _continuous_chirp_kind(x: np.ndarray, fs: float) -> tuple[str|None,float,list[str]]:
    if len(x)<64: return None,0.0,[]
    inst=np.angle(x[1:]*np.conj(x[:-1]))*fs/(2*np.pi)
    if np.std(inst)<fs*1e-5: return None,0.0,[]
    k=5 if len(inst)>=5 else 3
    sm=signal.medfilt(inst,kernel_size=k)
    lo,hi=np.quantile(sm,[.02,.98]); span=float(hi-lo)
    if span<0.01*fs: return None,0.0,[]
    d=np.diff(sm)
    reset_idx=np.flatnonzero(d < -0.35*span)
    keep=np.ones_like(d,dtype=bool); keep[reset_idx]=False
    if np.any(keep):
        positive=float(np.mean(d[keep] > max(1e-12,0.001*span)))
        nearflat=float(np.mean(np.abs(d[keep]) < 0.01*span))
    else:
        positive=0.0; nearflat=1.0
    if len(reset_idx)>=2 and positive>0.55 and nearflat<0.75:
        return "fmcw",float(np.clip(0.72+0.04*len(reset_idx),0.72,0.96)),[f"Detected {len(reset_idx)} sawtooth frequency resets with predominantly rising ramps."]
    t=np.arange(len(sm))/fs
    co=np.polyfit(t,sm,1); fit=np.polyval(co,t)
    r2=1-np.sum((sm-fit)**2)/(np.sum((sm-np.mean(sm))**2)+1e-30)
    if r2>0.94 and np.ptp(fit)>0.03*fs:
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

def _fsk_levels(x: np.ndarray, fs: float) -> tuple[int,float,np.ndarray,np.ndarray,float]:
    inst=np.angle(x[1:]*np.conj(x[:-1]))*fs/(2*np.pi)
    if len(inst)<100: return 0,0.0,inst,np.array([]),0.0
    kmed=min(11,max(3,(len(inst)//10000)*2+3))
    if kmed%2==0:kmed+=1
    sm=signal.medfilt(inst,kernel_size=kmed)
    lo,hi=np.quantile(sm,[.01,.99]); span=float(hi-lo)
    if span<fs*1e-4: return 1,0.0,sm,np.array([np.median(sm)]),1.0
    trim=sm[(sm>=lo)&(sm<=hi)]
    c2,l2,m2=_kmeans1d(trim,2)
    c4,l4,m4=_kmeans1d(trim,4)
    sep2=float(np.diff(c2).min())
    minsep4=float(np.diff(c4).min())
    occ4=np.array([np.mean(l4==i) for i in range(4)])
    # A true 4-FSK cloud has four well separated, meaningfully occupied plateaus.
    use4=(minsep4/span>0.14 and np.min(occ4)>0.05 and m4 < 0.45*m2)
    centers=c4 if use4 else c2
    levels=4 if use4 else 2
    labels=np.argmin(np.abs(sm[:,None]-centers[None,:]),axis=1)
    residual=sm-centers[labels]
    spacing=float(np.min(np.diff(centers))) if len(centers)>1 else span
    noise=float(np.std(residual)+1e-12)
    cluster_score=float(np.clip(spacing/(6*noise),0,1))
    # CPFSK spends most samples on plateaus; continuous FM/PM does not.
    d=np.abs(np.diff(sm)); plateau=float(np.mean(d < 0.02*span)) if len(d) else 0.0
    score=cluster_score*plateau
    return levels,score,sm,centers,plateau

def _estimate_rate_from_labels(sm: np.ndarray, centers: np.ndarray, fs: float) -> tuple[float|None,float]:
    if len(sm)<32 or len(centers)<2: return None,0.0
    labels=np.argmin(np.abs(sm[:,None]-centers[None,:]),axis=1)
    change=np.flatnonzero(labels[1:]!=labels[:-1])+1
    if len(change)<5: return None,0.0
    runs=np.diff(np.r_[0,change,len(labels)])
    runs=runs[runs>=2]
    if len(runs)<3: return None,0.0
    # Random symbols create runs of 1,2,3... symbol periods. The lower mode estimates one symbol.
    q=np.quantile(runs,0.35)
    short=runs[runs<=max(q,2)]
    if len(short)==0: short=np.sort(runs)[:max(1,len(runs)//4)]
    sps=float(np.median(short))
    if sps<2: return None,0.0
    rs=fs/sps
    # Consistency: run lengths should be close to integer multiples of sps.
    mult=runs/sps
    err=np.median(np.abs(mult-np.rint(mult)))
    conf=float(np.clip(1-2*err,0,1))
    return float(rs),conf

def _best_linear_mod(x: np.ndarray, fs: float, rs: float) -> tuple[str,float,dict]:
    sps=int(round(fs/rs))
    if sps<2 or sps>256:
        return "digital-linear",0.3,{}
    # Interactive AMC does not need hundreds of thousands of samples for a
    # constellation hypothesis. Keep a central stationary block and search a
    # bounded set of timing phases before the more expensive EVM comparisons.
    if len(x)>65536:
        mid=len(x)//2
        x=x[max(0,mid-32768):mid+32768]
    best=(1e9,"digital-linear",0,None)
    scores = {}
    candidates=("bpsk","qpsk","8psk","16qam","64qam")
    phase_candidates=np.unique(np.linspace(0,max(0,sps-1),min(sps,12),dtype=int))
    for rolloff in (0.2,0.35,0.5):
        taps=rrc_taps(sps,rolloff,8)
        mf=signal.lfilter(taps,[1.0],x)
        # Pick the strongest eye-opening phases first; this cuts repeated EVM
        # work substantially while preserving nearby timing hypotheses.
        scored=[]
        for phase in phase_candidates:
            syms=mf[int(phase)::sps]
            syms=syms[min(12,len(syms)):min(len(syms),1800)]
            if len(syms)>=32:
                scored.append((float(np.mean(np.abs(syms)**2)),int(phase),syms))
        for _,phase,syms in sorted(scored,reverse=True)[:5]:
            for mod in candidates:
                m=evm_rms(syms,mod)
                evm=m["evm_rms_percent"]
                scores[mod] = min(evm,scores.get(mod,1e9))
                if evm<best[0]: best=(evm,mod,phase,(rolloff,m))
    if best[3] is None:
        return "digital-linear",0.3,{}
    evm,mod,phase,(ro,m)=best
    conf=float(np.clip(1.0-evm/70.0,0.35,0.96))
    return mod,conf,{"evm_rms_percent":float(evm),"sampling_phase":int(phase),"rrc_rolloff_candidate":ro,"sps":sps,"candidates":[{"modulation":m,"evm_percent":float(e)} for m,e in sorted(scores.items(),key=lambda item:item[1])]}


def analyze(samples: np.ndarray, fs: float) -> BlindResult:
    x=np.asarray(samples,dtype=np.complex64)
    if x.size<256:
        raise ValueError("blind analysis needs at least 256 complex samples")
    # Keep computational work bounded for interactive analysis while retaining stationarity cues.
    # 131k complex samples is enough for the current feature set and keeps the
    # classifier responsive when the source runs at several MS/s.
    if x.size>131072:
        start=(x.size-131072)//2; x=x[start:start+131072]
    a=np.abs(x); p=a*a
    papr=10*np.log10((np.max(p)+1e-30)/(np.mean(p)+1e-30))
    cv=float(np.std(a)/(np.mean(a)+1e-30))
    duty=_active_duty(x)
    f,P,peaks,sflat,bw3,bw99,snr=_psd_features(x,fs)
    notes=[f"PAPR {papr:.1f} dB; envelope CV {cv:.2f}; spectral flatness {sflat:.3f}."]

    # AWGN is checked before duty-cycle logic because a Rayleigh envelope naturally crosses amplitude thresholds often.
    if sflat>0.45 and cv>0.42 and papr>6.0:
        return BlindResult(modulation="awgn",family="noise",confidence=0.94,
            bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=None,snr_db=None,
            notes=notes+["Broad, spectrally flat complex noise with no stable carrier structure."])

    # OFDM is noise-like in amplitude but has many stable occupied carriers; distinguish it from sparse pulsed radar by duty.
    if papr>6.0 and cv>0.42 and duty>0.30 and len(peaks)>10:
        return BlindResult(modulation="ofdm",family="digital/multicarrier",confidence=0.82,
            bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=None,snr_db=snr,
            notes=notes+["High crest factor and many occupied spectral components are consistent with OFDM."])

    if duty<0.30 and papr>5.0:
        kind,conf,n=_radar_kind(x,fs)
        return BlindResult(modulation=kind,family="radar/pulsed",confidence=conf,
            bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=None,snr_db=snr,notes=notes+n)

    chirp,conf,n=_continuous_chirp_kind(x,fs)
    if chirp is not None and cv<0.18:
        return BlindResult(modulation=chirp,family="chirp/radar",confidence=conf,
            bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=None,snr_db=snr,notes=notes+n)

    # Sparse line spectra are intentionally handled before symbol-clock heuristics.
    # With an unknown message, tone/SSB and multitone/AM/DSB can be physically ambiguous.
    if len(peaks)<=1 and sflat<0.08:
        return BlindResult(modulation="single-tone/ssb-like",family="tone/analog",confidence=0.72,
            bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=None,snr_db=snr,
            notes=notes+["A pure SSB sinusoidal-message waveform is physically indistinguishable from a complex tone without side information."])
    if 2<=len(peaks)<=6 and sflat<0.08:
        return BlindResult(modulation="multitone/am/dsb-like",family="tone/analog",confidence=0.64,
            bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=None,snr_db=snr,
            notes=notes+["A sparse line spectrum can be a multitone waveform or analog modulation of a sinusoidal message; classification is ambiguous."])

    # Constant-envelope, non-chirp waveforms: detect FSK/MSK before analog angle modulation.
    if cv<0.12:
        levels,lscore,inst,centers,plateau=_fsk_levels(x,fs)
        rs,runconf=_estimate_rate_from_labels(inst,centers,fs)
        if levels in (2,4) and lscore>0.07 and rs:
            mod=f"{levels}fsk"
            is_msk=False
            if levels==2:
                dev=abs(centers[-1]-centers[0])/2
                is_msk=abs(dev-rs/4)/max(rs/4,1)<0.30
                if is_msk: mod="msk"
            if plateau>0.70 or (is_msk and plateau>0.42):
                return BlindResult(modulation=mod,family="digital/cpfsk",confidence=float(np.clip(0.58+0.22*lscore+0.12*runconf,0.58,0.94)),
                    bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=rs,snr_db=snr,
                    notes=notes+[f"Instantaneous-frequency clustering supports {levels} plateaus (plateau fraction {plateau:.2f}).",f"Run-length symbol-clock consistency {runconf:.2f}."])
        # FM and PM are not identifiable from an arbitrary unknown message in general.
        return BlindResult(modulation="fm/pm-like",family="analog/angle",confidence=0.58,
            bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=None,snr_db=snr,
            notes=notes+["Constant envelope with continuous phase/frequency variation; blind FM-vs-PM is message-dependent."])

    # Linear digital signals have strong symbol-clock cyclostationarity in transition energy.
    rs,rprom=estimate_symbol_rate_linear(x,fs,bw99)
    if rs and rprom>8.0 and 2 <= fs/rs <= 256:
        mod,conf,extra=_best_linear_mod(x,fs,rs)
        return BlindResult(modulation=mod,family="digital/linear",confidence=conf,candidates=extra.get("candidates",[]),features={"clock_peak_db":float(rprom)},
            bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=rs,snr_db=snr,
            notes=notes+[f"Estimated symbol rate from transition-energy cyclostationarity ({rprom:.1f} dB peak).",
                         f"Best constellation fit: {mod}, estimated EVM {extra.get('evm_rms_percent',float('nan')):.1f}%."])

    return BlindResult(modulation="unknown",family="unknown",confidence=0.25,
        bandwidth_3db_hz=bw3,bandwidth_99_hz=bw99,symbol_rate_baud=rs,snr_db=snr,
        notes=notes+["No current MVP rule produced a reliable class; inspect the feature views or use a trained AMC plugin."])
