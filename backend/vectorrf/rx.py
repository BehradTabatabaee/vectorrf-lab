from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy import signal

from .dsp import bits_to_bytes, rrc_taps
from .models import RxConfig
from .modem import evm_rms, slice_symbols, reference_constellation
from .sync import costas_loop, eye_segments, mueller_muller, gardner, coarse_carrier, decision_carrier, cma_equalizer


@dataclass
class RxResult:
    recovered: np.ndarray
    symbols: np.ndarray | None
    bits: np.ndarray | None
    text: str | None
    metrics: dict
    telemetry: dict


def _digital(samples: np.ndarray, cfg: RxConfig) -> RxResult:
    fs = cfg.sample_rate
    sps = max(2, int(round(fs/cfg.symbol_rate)))
    actual_rs = fs/sps
    mod = cfg.modulation
    base = {"dbpsk":"bpsk", "dqpsk":"qpsk", "pi4-dqpsk":"qpsk", "oqpsk":"qpsk"}.get(mod,mod)
    telemetry = {}
    cfo = 0.
    if cfg.coarse_cfo and mod in {"bpsk","qpsk","dbpsk","dqpsk","8psk"}:
        order = 2 if base == "bpsk" else 8 if base == "8psk" else 4
        cfo = coarse_carrier(samples,fs,order)
        samples = samples*np.exp(-2j*np.pi*cfo*np.arange(len(samples))/fs)
    telemetry["coarse_cfo_hz"] = cfo
    taps = rrc_taps(sps, cfg.rolloff, cfg.filter_span_symbols) if cfg.pulse_shape == "rrc" else np.ones(sps)/sps
    filtered=signal.fftconvolve(samples,taps)[:len(samples)] if len(taps)>257 else signal.lfilter(taps,[1.0],samples)
    mf = filtered.astype(np.complex64)
    gd = len(taps)-1 if cfg.pulse_shape == "rrc" else sps-1
    if mod == "oqpsk":
        q = np.interp(np.arange(len(mf))+sps/2,np.arange(len(mf)),mf.imag,right=0)
        mf = mf.real+1j*q
    if cfg.timing_recovery in {"mueller-muller", "gardner"}:
        if cfg.timing_recovery == "gardner":
            tr = gardner(mf[max(0,gd-sps):], sps, cfg.timing_gain, cfg.timing_phase)
        else:
            tr = mueller_muller(mf[max(0, gd):], sps, cfg.timing_gain)
        syms = tr.output
        telemetry["timing_error"] = tr.error[:4000].astype(float).tolist()
        telemetry["timing_mu"] = tr.mu[:4000].astype(float).tolist()
    else:
        positions = np.arange(max(0,gd+cfg.timing_phase*sps),len(mf)-1,sps)
        syms = np.interp(positions,np.arange(len(mf)),mf.real)+1j*np.interp(positions,np.arange(len(mf)),mf.imag)
    if syms.size > 50000:
        syms = syms[:50000]
    # Preserve a bounded view of the symbols before carrier recovery.  This is
    # used by the educational receiver UI to show the actual acquisition path,
    # not a synthetic before/after illustration.
    if syms.size:
        take_pre = 1
        pre = syms[::take_pre][:2500]
        telemetry["pre_costas_i"] = pre.real.astype(float).tolist()
        telemetry["pre_costas_q"] = pre.imag.astype(float).tolist()
    refs = reference_constellation(base,cfg.apsk_ratio2,cfg.apsk_ratio3)
    if cfg.use_costas and cfg.carrier_algorithm != "off" and syms.size and mod not in {"dbpsk","dqpsk","pi4-dqpsk"}:
        if cfg.carrier_algorithm == "costas" and base in {"bpsk","qpsk"}:
            c = cfg.costas.model_copy(update={"order": 2 if base == "bpsk" else 4})
            cr = costas_loop(syms,actual_rs,c)
        else:
            cr = decision_carrier(syms,actual_rs,refs,cfg.costas)
        syms = cr.output
        telemetry["costas_freq_hz"] = cr.freq_hz[:4000].astype(float).tolist()
        telemetry["costas_error"] = cr.error[:4000].astype(float).tolist()
        telemetry["costas_phase_deg"] = np.rad2deg(cr.phase[:4000]).astype(float).tolist()
    if mod in {"dbpsk","dqpsk","pi4-dqpsk"} and len(syms):
        syms=syms*np.conj(np.r_[1+0j,syms[:-1]])
        if mod == "dqpsk": syms*=np.exp(1j*np.pi/4)
    if cfg.equalizer == "cma" and syms.size:
        telemetry["pre_equalizer_i"] = syms.real[:2500].tolist()
        telemetry["pre_equalizer_q"] = syms.imag[:2500].tolist()
        syms, eq_error, weights = cma_equalizer(syms,cfg.equalizer_taps,cfg.equalizer_gain)
        telemetry["equalizer_error"] = eq_error[:4000].tolist()
        telemetry["equalizer_taps"] = np.abs(weights).tolist()
    phase_rot = np.exp(-1j*np.deg2rad(cfg.manual_phase_deg))
    syms = syms * phase_rot
    decisions, bits = slice_symbols(syms, base, cfg.apsk_ratio2, cfg.apsk_ratio3)
    # Symbol-resolution quality telemetry, similar to a vector signal analyzer's
    # error-vs-time views.  Keep it bounded before serializing to the browser.
    if syms.size:
        z = syms.astype(np.complex128)
        z /= np.sqrt(np.mean(np.abs(z) ** 2) + 1e-15)
        ref = decisions.astype(np.complex128)
        common_phase = np.angle(np.vdot(z, ref))
        z *= np.exp(1j * common_phase)
        err = z - ref
        ref_rms = np.sqrt(np.mean(np.abs(ref) ** 2) + 1e-15)
        evm_time = 100.0 * np.abs(err) / ref_rms
        mag_error = 100.0 * (np.abs(z) - np.abs(ref)) / max(ref_rms, 1e-15)
        phase_error = np.rad2deg(np.angle(z * np.conj(ref)))
        take = max(1, len(z) // 2000)
        telemetry["evm_time_percent"] = evm_time[::take].astype(float).tolist()
        telemetry["magnitude_error_percent"] = mag_error[::take].astype(float).tolist()
        telemetry["phase_error_deg"] = phase_error[::take].astype(float).tolist()
    raw = bits_to_bytes(bits)
    text = raw.decode("utf-8", errors="replace")[:4096]
    metrics = evm_rms(syms, base, cfg.apsk_ratio2, cfg.apsk_ratio3)
    metrics.update({"samples_per_symbol":sps, "actual_symbol_rate":actual_rs, "symbol_count":int(syms.size)})
    eyes = eye_segments(mf[max(0, gd-sps):], sps, traces=64)
    if eyes.size:
        telemetry["eye_i"] = eyes.real.astype(np.float32).tolist()
        telemetry["eye_q"] = eyes.imag.astype(np.float32).tolist()
    return RxResult(mf.astype(np.complex64), syms.astype(np.complex64), bits, text, metrics, telemetry)

def _fsk(samples: np.ndarray, cfg: RxConfig, levels: int, msk: bool = False) -> RxResult:
    fs = cfg.sample_rate
    sps = max(2, int(round(fs/cfg.symbol_rate)))
    dphi = np.angle(samples[1:]*np.conj(samples[:-1])) * fs/(2*np.pi)
    # Average discriminator over each symbol; manual timing phase controls the start.
    start = int(np.clip(round(cfg.timing_phase*sps), 0, sps-1))
    m = (len(dphi)-start)//sps
    if m <= 0:
        return RxResult(dphi.astype(np.complex64), np.array([]), np.array([],dtype=np.uint8), "", {}, {})
    freq_sym = dphi[start:start+m*sps].reshape(m,sps).mean(axis=1)
    if msk:
        idx = (freq_sym > 0).astype(int)
        levels = 2
    else:
        centers = np.linspace(-cfg.fsk_deviation_hz, cfg.fsk_deviation_hz, levels)
        idx = np.argmin(np.abs(freq_sym[:,None]-centers[None,:]), axis=1)
    k = int(round(np.log2(levels)))
    shifts = np.arange(k-1,-1,-1)
    bits = ((idx[:,None]>>shifts)&1).astype(np.uint8).reshape(-1)
    text = bits_to_bytes(bits).decode("utf-8", errors="replace")[:4096]
    return RxResult(dphi.astype(np.complex64), freq_sym.astype(np.complex64), bits, text,
                    {"samples_per_symbol":sps,"actual_symbol_rate":fs/sps}, {})


def _ofdm(samples: np.ndarray, cfg: RxConfig) -> RxResult:
    from .ofdm import receive
    syms,bits,metrics,telemetry=receive(samples[cfg.ofdm_timing_offset:],cfg.sample_rate,cfg.ofdm_fft,cfg.ofdm_cp,cfg.ofdm_occupied,cfg.ofdm_design,cfg.coarse_cfo,cfg.ofdm_start_s)
    text=bits_to_bytes(bits).decode('utf-8',errors='replace')[:4096]
    return RxResult(samples,syms,bits,text,metrics,telemetry)


def _analog(samples: np.ndarray, cfg: RxConfig) -> RxResult:
    x=np.asarray(samples,dtype=np.complex64)
    if cfg.modulation=="am":
        y=np.abs(x)-np.mean(np.abs(x))
    elif cfg.modulation in {"dsb-sc","ssb"}:
        y=x.real
    elif cfg.modulation=="fm":
        y=np.angle(x[1:]*np.conj(x[:-1]))*cfg.sample_rate/(2*np.pi*cfg.fm_deviation_hz)
    elif cfg.modulation=="pm":
        y=np.unwrap(np.angle(x))
        y-=np.mean(y)
    else:
        raise ValueError(cfg.modulation)
    y=np.asarray(y,dtype=np.float64)
    take=max(1,len(x)//2000)
    amp=np.abs(x)
    inst=np.r_[0.0,np.angle(x[1:]*np.conj(x[:-1]))*cfg.sample_rate/(2*np.pi)] if len(x)>1 else np.zeros(len(x))
    telemetry={"amplitude":amp[::take].astype(float).tolist(),"frequency_hz":inst[::take].astype(float).tolist()}
    metrics={"rms":float(np.sqrt(np.mean(np.abs(y)**2)+1e-15)),"peak":float(np.max(np.abs(y))) if len(y) else 0.0,"mean_frequency_hz":float(np.mean(inst[1:])) if len(inst)>1 else 0.0}
    return RxResult(np.asarray(y,dtype=np.complex64),None,None,None,metrics,telemetry)



def _basic(samples: np.ndarray, cfg: RxConfig) -> RxResult:
    """Analyze an unframed/basic complex waveform without inventing symbols.

    Fundamentals do not have a bit clock to recover, so expose measurements that
    are meaningful for the actual complex samples instead: envelope, phase and
    instantaneous frequency.  The UI can therefore follow tone/noise/chirp/custom
    sources through the same tuner/channel/processing path as digital signals.
    """
    x=np.asarray(samples,dtype=np.complex64)
    amp=np.abs(x).astype(np.float64)
    phase=np.unwrap(np.angle(x).astype(np.float64))
    if len(x)>1:
        freq=np.r_[0.0,np.angle(x[1:]*np.conj(x[:-1]))*cfg.sample_rate/(2*np.pi)]
    else:
        freq=np.zeros(len(x),dtype=float)
    take=max(1,len(x)//2000)
    telemetry={
        "amplitude":amp[::take].astype(float).tolist(),
        "phase_deg":np.rad2deg(phase[::take]).astype(float).tolist(),
        "frequency_hz":freq[::take].astype(float).tolist(),
    }
    power=np.abs(x)**2
    metrics={
        "rms":float(np.sqrt(np.mean(power)+1e-15)) if len(x) else 0.0,
        "peak":float(np.max(np.abs(x))) if len(x) else 0.0,
        "mean_frequency_hz":float(np.mean(freq[1:])) if len(freq)>1 else 0.0,
        "frequency_std_hz":float(np.std(freq[1:])) if len(freq)>1 else 0.0,
        "sample_count":int(len(x)),
    }
    return RxResult(x,None,None,None,metrics,telemetry)

def demodulate(samples: np.ndarray, cfg: RxConfig) -> RxResult:
    x=np.asarray(samples,dtype=np.complex64)
    if cfg.tuner_hz:
        x=x*np.exp(-2j*np.pi*cfg.tuner_hz*np.arange(len(x))/cfg.sample_rate)
    if cfg.modulation in {"bpsk","qpsk","8psk","16qam","64qam","dbpsk","dqpsk","pi4-dqpsk","oqpsk","16apsk","32apsk"}:
        return _digital(x,cfg)
    if cfg.modulation=="2fsk":
        return _fsk(x,cfg,2)
    if cfg.modulation=="4fsk":
        return _fsk(x,cfg,4)
    if cfg.modulation in {"msk","gmsk"}:
        return _fsk(x,cfg,2,msk=True)
    if cfg.modulation=="ofdm":
        return _ofdm(x,cfg)
    if cfg.modulation in {"am","fm","pm","ssb","dsb-sc"}:
        return _analog(x,cfg)
    return _basic(x,cfg)


def receive_frame(samples: np.ndarray, cfg: RxConfig, *, power_dbm: float = 0.0, reference_rms: float = 1.0) -> dict:
    """Run generic frame acquisition and, when configured, decode each block.

    This is deliberately separate from ``demodulate`` so users can inspect
    acquisition failures (energy/correlation) instead of hiding them inside the
    symbol synchronizer.
    """
    from .frame import frame_acquisition

    # Acquisition operates in the receiver's tuned baseband, just like the
    # conventional demodulator.  Per-block demodulators below therefore use a
    # zero residual tuner offset rather than applying the same correction twice.
    x = np.asarray(samples, dtype=np.complex64)
    if cfg.tuner_hz and len(x):
        n = np.arange(len(x), dtype=np.float64)
        x = (x * np.exp(-2j*np.pi*float(cfg.tuner_hz)*n/float(cfg.sample_rate))).astype(np.complex64)
    acq = frame_acquisition(x, cfg, power_dbm=power_dbm, reference_rms=reference_rms)
    if not acq.get("detected") or not cfg.frame_blocks:
        acq["decoded_blocks"] = []
        return acq

    decoded = []
    for span, block in zip(acq.get("blocks", []), cfg.frame_blocks):
        a, b = int(span["start_sample"]), int(span["end_sample"])
        if b-a < 8:
            decoded.append({**span, "status":"incomplete", "bits":[], "metrics":{}})
            continue
        rate = float(block.symbol_rate or cfg.symbol_rate)
        block_cfg = cfg.model_copy(update={
            "modulation": block.modulation,
            "symbol_rate": rate,
            "frame_enabled": False,
            "timing_recovery": "manual",
            "timing_phase": 0.0,
            "use_costas": block.modulation in {"bpsk","qpsk","8psk","16qam","64qam","16apsk","32apsk"},
            "carrier_algorithm": "decision-directed" if block.modulation not in {"bpsk","qpsk"} else "costas",
            "tuner_hz": 0.0,
        })
        try:
            result = demodulate(x[a:b], block_cfg)
            bits = result.bits[:block.bit_count].astype(int).tolist() if result.bits is not None else []
            decoded.append({
                **span, "status":"decoded", "bits":bits,
                "bit_count":len(bits), "metrics":result.metrics,
                "text":result.text[:256] if result.text else None,
            })
        except (ValueError, IndexError, FloatingPointError) as exc:
            decoded.append({**span, "status":"error", "bits":[], "metrics":{}, "error":str(exc)[:160]})
    acq["decoded_blocks"] = decoded
    return acq
