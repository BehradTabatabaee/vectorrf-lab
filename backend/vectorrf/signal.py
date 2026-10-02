from __future__ import annotations

from dataclasses import dataclass, field
import math
import numpy as np
from scipy import signal

from .dsp import add_awgn, bytes_to_bits, normalize_rms
from .models import TxConfig, WaveformKind, RADAR_WAVEFORMS
from .modem import map_bits, pulse_shape, bits_per_symbol


@dataclass
class GeneratedSignal:
    samples: np.ndarray
    sample_rate: float
    metadata: dict = field(default_factory=dict)
    tx_bits: np.ndarray | None = None
    tx_symbols: np.ndarray | None = None


def _source_bits(cfg: TxConfig, needed: int, rng: np.random.Generator, start_bit: int = 0) -> np.ndarray:
    """Payload-source bits without adding any framing."""
    if cfg.data_source == "bits":
        pattern = np.array([int(b) for b in cfg.custom_bits], dtype=np.uint8)
    elif cfg.data_source == "text" and cfg.message:
        pattern = bytes_to_bits(cfg.message)
    else:
        # Maximal-length PRBS, explicit order and repeatable nonzero state.
        seed = rng.integers(0, 2, cfg.pn_order, dtype=np.int8)
        if not np.any(seed): seed[0] = 1
        pattern, _ = signal.max_len_seq(cfg.pn_order, state=seed, length=min(cfg.pn_bits, (1 << cfg.pn_order)-1))
    return pattern[(np.arange(needed)+start_bit)%len(pattern)].astype(np.uint8)


def _data_bits(cfg: TxConfig, needed: int, rng: np.random.Generator, start_bit: int = 0) -> np.ndarray:
    if not (cfg.frame_enabled and cfg.frame_mode == "simple"):
        return _source_bits(cfg, needed, rng, start_bit)
    payload=_source_bits(cfg,cfg.payload_bits,rng,0)
    frame=[np.array([int(b) for b in cfg.preamble_bits],dtype=np.uint8),payload]
    if cfg.frame_crc:
        import binascii
        crc=binascii.crc_hqx(np.packbits(payload).tobytes(),0xffff)
        frame.append(np.unpackbits(np.frombuffer(crc.to_bytes(2,'big'),dtype=np.uint8)))
    pattern=np.concatenate(frame)
    return pattern[(np.arange(needed)+start_bit)%len(pattern)].astype(np.uint8)


def _digital_linear(cfg: TxConfig, n: int, rng: np.random.Generator, mod: str) -> GeneratedSignal:
    sps = max(2, int(round(cfg.sample_rate / cfg.symbol_rate)))
    actual_rs = cfg.sample_rate / sps
    base = {"dbpsk":"bpsk", "dqpsk":"qpsk", "pi4-dqpsk":"qpsk", "oqpsk":"qpsk"}.get(mod,mod)
    k = bits_per_symbol(base)
    n_sym = int(np.ceil(n / sps)) + cfg.filter_span_symbols + 4
    bits = _data_bits(cfg, n_sym * k, rng)
    symbols = map_bits(bits, base, cfg.apsk_ratio2, cfg.apsk_ratio3)
    if mod in {"dbpsk", "dqpsk", "pi4-dqpsk"}:
        if mod == "dqpsk": symbols = symbols * np.exp(-1j*np.pi/4)
        symbols = np.cumprod(symbols.astype(np.complex128)).astype(np.complex64)
    if cfg.pulse_shape == "rect":
        taps = np.ones(sps)
        y = np.repeat(symbols, sps)
    else:
        y, taps = pulse_shape(symbols, sps, cfg.rolloff, cfg.filter_span_symbols)
    if mod == "oqpsk":
        # Fractional half-symbol delay remains valid at odd samples/symbol.
        q = np.interp(np.arange(len(y))-sps/2, np.arange(len(y)), y.imag, left=0)
        y = y.real + 1j*q
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    y = y[:n]
    return GeneratedSignal(
        samples=y,
        sample_rate=cfg.sample_rate,
        metadata={"modulation": mod, "samples_per_symbol": sps, "symbol_rate": actual_rs, "rrc_taps": len(taps), "pulse_shape": cfg.pulse_shape, "data_source": cfg.data_source},
        tx_bits=bits,
        tx_symbols=symbols,
    )


def _cpfsk(cfg: TxConfig, n: int, rng: np.random.Generator, levels: int, msk: bool = False) -> GeneratedSignal:
    sps = max(2, int(round(cfg.sample_rate / cfg.symbol_rate)))
    actual_rs = cfg.sample_rate / sps
    k = int(round(math.log2(levels)))
    n_sym = int(np.ceil(n / sps)) + 2
    bits = _data_bits(cfg, n_sym * k, rng)
    groups = bits[: (bits.size // k) * k].reshape(-1, k)
    idx = groups @ (1 << np.arange(k-1, -1, -1))
    if msk:
        freqs = np.where(idx > 0, actual_rs / 4.0, -actual_rs / 4.0)
    else:
        centered = idx.astype(float) - (levels - 1) / 2.0
        spacing = 2.0 * cfg.fsk_deviation_hz / max(1, levels - 1)
        freqs = centered * spacing
    inst_f = np.repeat(freqs, sps)[:n]
    if cfg.waveform == WaveformKind.GMSK:
        from scipy.ndimage import gaussian_filter1d
        sigma = np.sqrt(np.log(2)) * sps / (2*np.pi*cfg.gaussian_bt)
        inst_f = gaussian_filter1d(inst_f.astype(float), sigma=sigma, mode="nearest")
    phase = 2 * np.pi * np.cumsum(inst_f) / cfg.sample_rate
    y = np.exp(1j * phase)
    return GeneratedSignal(
        y.astype(np.complex64), cfg.sample_rate,
        {"modulation": "msk" if msk else f"{levels}fsk", "samples_per_symbol": sps, "symbol_rate": actual_rs},
        tx_bits=bits,
    )


def _ofdm(cfg: TxConfig, n: int, rng: np.random.Generator, start_s: float = 0) -> GeneratedSignal:
    from .ofdm import synthesize
    y,bits,metadata=synthesize(cfg,n,lambda count,offset:_data_bits(cfg,count,rng,offset),start_s)
    return GeneratedSignal(y,cfg.sample_rate,metadata,tx_bits=bits)


def _message_tone(cfg: TxConfig, t: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * cfg.message_freq_hz * t).astype(np.float64)


def _analog(cfg: TxConfig, t: np.ndarray) -> np.ndarray:
    m = _message_tone(cfg, t)
    w = cfg.waveform
    if w == WaveformKind.AM:
        return (1.0 + cfg.modulation_index * m).astype(np.complex128)
    if w == WaveformKind.DSB_SC:
        return m.astype(np.complex128)
    if w == WaveformKind.SSB:
        analytic = signal.hilbert(m)
        return analytic if cfg.sideband == "usb" else np.conj(analytic)
    if w == WaveformKind.FM:
        phase = 2*np.pi*cfg.fm_deviation_hz*np.cumsum(m)/cfg.sample_rate
        return np.exp(1j*phase)
    if w == WaveformKind.PM:
        return np.exp(1j*cfg.pm_deviation_rad*m)
    raise ValueError(w)


_BARKER = {
    2:[1,-1], 3:[1,1,-1], 4:[1,1,-1,1], 5:[1,1,1,-1,1],
    7:[1,1,1,-1,-1,1,-1], 11:[1,1,1,-1,-1,-1,1,-1,-1,1,-1],
    13:[1,1,1,1,1,-1,-1,1,1,-1,1,-1,1],
}


def _pulse_envelope(length: int, kind: str) -> np.ndarray:
    if length <= 1 or kind == "rect":
        return np.ones(length, dtype=np.float64)
    x = np.linspace(-1.0, 1.0, length, endpoint=True)
    if kind == "gaussian":
        # ~-24 dB near the pulse edges, useful for seeing spectral tradeoffs.
        return np.exp(-0.5 * (x / 0.33) ** 2)
    if kind == "raised-cosine":
        return 0.5 * (1.0 + np.cos(np.pi * x))
    return np.ones(length, dtype=np.float64)


def _polyphase_code(kind: WaveformKind, cfg: TxConfig) -> np.ndarray:
    if kind == WaveformKind.BARKER:
        return np.asarray(_BARKER[cfg.barker_length], dtype=np.complex128)
    if kind == WaveformKind.CUSTOM_PHASE:
        phase = np.deg2rad(np.asarray(cfg.custom_phase_deg, dtype=np.float64))
        if phase.size < 2:
            phase = np.asarray([0.0, np.pi], dtype=np.float64)
        return np.exp(1j * phase)
    if kind == WaveformKind.ZADOFF_CHU:
        n = np.arange(cfg.phase_code_chips, dtype=np.float64)
        # Standard ZC definition differs for odd/even sequence lengths.
        if cfg.phase_code_chips % 2:
            phase = -np.pi * cfg.zc_root * n * (n + 1.0) / cfg.phase_code_chips
        else:
            phase = -np.pi * cfg.zc_root * n * n / cfg.phase_code_chips
        return np.exp(1j * phase)
    if kind in {WaveformKind.P3, WaveformKind.P4}:
        n = np.arange(cfg.phase_code_chips, dtype=np.float64)
        phase = np.pi * n * n / cfg.phase_code_chips
        if kind == WaveformKind.P4:
            phase -= np.pi * n
        return np.exp(1j * phase)

    m = cfg.polyphase_order
    i, j = np.meshgrid(np.arange(m, dtype=np.float64), np.arange(m, dtype=np.float64), indexing="ij")
    if kind == WaveformKind.FRANK:
        phase = 2.0 * np.pi * i * j / m
    elif kind == WaveformKind.P1:
        # Lewis/Kretschmer P1, flattened in frequency-step groups.
        phase = -np.pi / m * (m - (2.0 * j + 1.0)) * (j * m + i)
    elif kind == WaveformKind.P2:
        phase = -np.pi / (2.0 * m) * (2.0 * i + 1.0 - m) * (2.0 * j + 1.0 - m)
    else:
        raise ValueError(f"not a phase code: {kind}")
    return np.exp(1j * phase.reshape(-1))


def _primitive_root(p: int) -> int:
    """Small primitive-root helper for Welch Costas arrays (p must be prime)."""
    phi = p - 1
    factors = []
    x = phi
    q = 2
    while q * q <= x:
        if x % q == 0:
            factors.append(q)
            while x % q == 0:
                x //= q
        q += 1
    if x > 1:
        factors.append(x)
    for g in range(2, p):
        if all(pow(g, phi // f, p) != 1 for f in factors):
            return g
    return 2


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _costas_permutation(requested: int) -> np.ndarray:
    # Welch construction yields a true Costas permutation of length p-1.
    p = max(5, requested + 1)
    while p > 5 and not _is_prime(p):
        p -= 1
    if not _is_prime(p):
        p = 5
    g = _primitive_root(p)
    return np.asarray([pow(g, k, p) - 1 for k in range(p - 1)], dtype=int)


def _pri_starts(cfg: TxConfig, n: int, rng: np.random.Generator) -> list[int]:
    fs = cfg.sample_rate
    starts: list[int] = []
    pos = 0
    k = 0
    stagger = cfg.pri_stagger or [1.0]
    while pos < n:
        starts.append(pos)
        if cfg.pri_mode == "staggered":
            scale = max(0.05, float(stagger[k % len(stagger)]))
        elif cfg.pri_mode == "jittered":
            scale = max(0.05, 1.0 + rng.uniform(-cfg.pri_jitter_fraction, cfg.pri_jitter_fraction))
        else:
            scale = 1.0
        step = max(1, int(round(cfg.pri_s * scale * fs)))
        pos += step
        k += 1
    return starts


def _pulse_contents(cfg: TxConfig, length: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    fs = cfg.sample_rate
    tau = np.arange(length, dtype=np.float64) / fs
    w = cfg.waveform
    meta: dict = {}

    if w == WaveformKind.PULSE:
        p = np.ones(length, dtype=np.complex128)
    elif w in {WaveformKind.LFM, WaveformKind.NLFM}:
        T = max(length / fs, 1.0 / fs)
        if w == WaveformKind.LFM:
            f = -cfg.sweep_bw_hz / 2.0 + cfg.sweep_bw_hz * tau / T
        else:
            x = np.clip(2.0 * tau / T - 1.0, -1.0, 1.0)
            f = (cfg.sweep_bw_hz / 2.0) * np.sign(x) * np.abs(x) ** cfg.nlfm_exponent
        phase = 2.0 * np.pi * np.cumsum(f) / fs
        p = np.exp(1j * phase)
        meta.update({"sweep_bw_hz": cfg.sweep_bw_hz, "time_bandwidth_product": cfg.sweep_bw_hz * T})
    elif w in {
        WaveformKind.BARKER, WaveformKind.FRANK, WaveformKind.P1, WaveformKind.P2,
        WaveformKind.P3, WaveformKind.P4, WaveformKind.ZADOFF_CHU, WaveformKind.CUSTOM_PHASE,
    }:
        code = _polyphase_code(w, cfg)
        chip_idx = np.minimum((np.arange(length) * len(code) / max(1, length)).astype(int), len(code) - 1)
        p = code[chip_idx]
        meta.update({"code_length": int(len(code)), "code_phase_deg": np.rad2deg(np.angle(code)).astype(float).tolist()})
    elif w == WaveformKind.COSTAS:
        perm = _costas_permutation(cfg.costas_hops)
        hops = len(perm)
        centered = perm.astype(float) - (hops - 1) / 2.0
        spacing = cfg.sweep_bw_hz / max(1, hops - 1)
        freqs = centered * spacing
        hop_idx = np.minimum((np.arange(length) * hops / max(1, length)).astype(int), hops - 1)
        f = freqs[hop_idx]
        p = np.exp(1j * 2.0 * np.pi * np.cumsum(f) / fs)
        meta.update({"costas_hops": hops, "costas_permutation": perm.astype(int).tolist(), "hop_frequencies_hz": freqs.astype(float).tolist()})
    elif w == WaveformKind.STEPPED_FREQ:
        steps = cfg.stepped_steps
        idx = np.minimum((np.arange(length) * steps / max(1, length)).astype(int), steps - 1)
        freqs = (np.arange(steps, dtype=float) - (steps - 1) / 2.0) * cfg.stepped_spacing_hz
        f = freqs[idx]
        p = np.exp(1j * 2.0 * np.pi * np.cumsum(f) / fs)
        meta.update({"frequency_steps": steps, "step_spacing_hz": cfg.stepped_spacing_hz})
    else:
        raise ValueError(w)

    p = p * _pulse_envelope(length, cfg.pulse_envelope)
    return p.astype(np.complex128), meta


def _radar(cfg: TxConfig, n: int, rng: np.random.Generator) -> GeneratedSignal:
    fs = cfg.sample_rate
    t = np.arange(n, dtype=np.float64) / fs
    w = cfg.waveform

    if w == WaveformKind.CW:
        return GeneratedSignal(np.ones(n, dtype=np.complex64), fs, {"waveform": w.value, "radar_family": "cw"})

    if w == WaveformKind.ARBITRARY_RADAR:
        durs = cfg.arbitrary_duration_us or [10.0]
        f0s = cfg.arbitrary_start_freq_hz or [0.0]
        f1s = cfg.arbitrary_stop_freq_hz or f0s
        amps = cfg.arbitrary_amplitude or [1.0]
        phs = cfg.arbitrary_phase_deg or [0.0]
        pieces = []
        phase_acc = 0.0
        count = max(len(durs), len(f0s), len(f1s), len(amps), len(phs))
        segment_meta = []
        for kseg in range(count):
            dur_us = float(durs[kseg % len(durs)])
            m = max(1, int(round(dur_us * 1e-6 * fs)))
            f0 = float(f0s[kseg % len(f0s)])
            f1 = float(f1s[kseg % len(f1s)])
            amp = max(0.0, float(amps[kseg % len(amps)]))
            jump = np.deg2rad(float(phs[kseg % len(phs)]))
            freq = np.linspace(f0, f1, m, endpoint=False, dtype=np.float64)
            phase = phase_acc + jump + 2.0 * np.pi * np.cumsum(freq) / fs
            part = amp * np.exp(1j * phase)
            pieces.append(part)
            if m:
                phase_acc = float(phase[-1])
            segment_meta.append({"duration_us": dur_us, "start_hz": f0, "stop_hz": f1, "amplitude": amp, "phase_jump_deg": float(phs[kseg % len(phs)])})
        seq = np.concatenate(pieces) if pieces else np.ones(1, dtype=np.complex128)
        y = np.resize(seq, n)
        return GeneratedSignal(y.astype(np.complex64), fs, {
            "waveform": w.value, "radar_family": "arbitrary", "segment_count": count,
            "sequence_duration_s": len(seq) / fs, "segments": segment_meta,
        })

    if w == WaveformKind.FMCW:
        T = max(cfg.pri_s, 1.0 / fs)
        tau = np.mod(t, T)
        if cfg.fmcw_shape == "triangle":
            x = tau / T
            f = np.where(x < 0.5,
                         -cfg.sweep_bw_hz / 2.0 + 2.0 * cfg.sweep_bw_hz * x,
                         cfg.sweep_bw_hz / 2.0 - 2.0 * cfg.sweep_bw_hz * (x - 0.5))
        else:
            direction = 1.0 if cfg.fmcw_shape == "up" else -1.0
            f = direction * (-cfg.sweep_bw_hz / 2.0 + cfg.sweep_bw_hz * tau / T)
        phase = 2.0 * np.pi * np.cumsum(f) / fs
        return GeneratedSignal(np.exp(1j * phase).astype(np.complex64), fs, {
            "waveform": w.value, "radar_family": "fmcw", "sweep_period_s": T,
            "sweep_bw_hz": cfg.sweep_bw_hz, "fmcw_shape": cfg.fmcw_shape,
        })

    if w == WaveformKind.PMCW:
        chips = max(2, cfg.phase_code_chips)
        chip_sps = max(1, int(round(fs / cfg.pmcw_chip_rate)))
        # A reproducible BPSK PN sequence; arbitrary/custom PMCW codes are represented by CUSTOM_PHASE pulses.
        code = rng.choice(np.asarray([-1.0, 1.0]), size=chips).astype(np.complex128)
        seq = np.repeat(code, chip_sps)
        y = np.resize(seq, n)
        return GeneratedSignal(y.astype(np.complex64), fs, {
            "waveform": w.value, "radar_family": "pmcw", "code_length": chips,
            "chip_rate_hz": fs / chip_sps, "chip_sps": chip_sps,
            "code_phase_deg": np.rad2deg(np.angle(code)).astype(float).tolist(),
        })

    pw_n = max(1, int(round(cfg.pulse_width_s * fs)))
    y = np.zeros(n, dtype=np.complex128)
    starts = _pri_starts(cfg, n, rng)
    pulse_meta: dict = {}
    for start in starts:
        length = min(pw_n, n - start)
        if length <= 0:
            continue
        p, pulse_meta = _pulse_contents(cfg, length, rng)
        y[start:start + length] = p

    meta = {
        "waveform": w.value,
        "radar_family": "pulsed",
        "pri_s": cfg.pri_s,
        "prf_hz": 1.0 / cfg.pri_s,
        "pulse_width_s": cfg.pulse_width_s,
        "duty_cycle": min(1.0, cfg.pulse_width_s / cfg.pri_s),
        "pri_mode": cfg.pri_mode,
        "pulse_envelope": cfg.pulse_envelope,
        "pulse_count": len(starts),
    }
    meta.update(pulse_meta)
    return GeneratedSignal(y.astype(np.complex64), fs, meta)


def generate(cfg: TxConfig, start_s: float = 0, frame: int = 0) -> GeneratedSignal:
    fs = cfg.sample_rate
    n = max(32, int(round(cfg.duration * fs)))
    rng = np.random.default_rng(cfg.seed)
    t = np.arange(n, dtype=np.float64) / fs
    w = cfg.waveform

    if cfg.frame_enabled and cfg.frame_mode == "blocks":
        from .frame import synthesize_frame
        y, bits, frame_meta = synthesize_frame(
            cfg, n, rng, lambda count, offset: _source_bits(cfg, count, rng, offset)
        )
        frame_meta.update({"waveform":"mixed-frame", "configured_waveform":w.value})
        out = GeneratedSignal(y, fs, frame_meta, tx_bits=bits)
    elif w == WaveformKind.CUSTOM:
        from .programs import program_samples
        out=GeneratedSignal(program_samples(cfg.program,fs,n,start_s,frame,cfg.seed),fs,{'waveform':'custom','program_mode':cfg.program.mode})
    elif w == WaveformKind.TONE:
        y = np.ones(n, dtype=np.complex64)
        out = GeneratedSignal(y, fs, {"waveform":"tone"})
    elif w == WaveformKind.MULTITONE:
        y = np.zeros(n, dtype=np.complex128)
        for f in cfg.tones_hz:
            if abs(f) < fs/2:
                y += np.exp(1j*2*np.pi*f*t)
        out = GeneratedSignal(normalize_rms(y), fs, {"waveform":"multitone", "tones_hz":cfg.tones_hz})
    elif w == WaveformKind.AWGN:
        y = (rng.standard_normal(n) + 1j*rng.standard_normal(n))/np.sqrt(2)
        out = GeneratedSignal(y.astype(np.complex64), fs, {"waveform":"awgn"})
    elif w == WaveformKind.CHIRP:
        T = max(cfg.duration, 1/fs)
        k = (cfg.chirp_stop_hz-cfg.chirp_start_hz)/T
        y = np.exp(1j*2*np.pi*(cfg.chirp_start_hz*t + 0.5*k*t*t))
        out = GeneratedSignal(y.astype(np.complex64), fs, {"waveform":"chirp", "start_hz":cfg.chirp_start_hz, "stop_hz":cfg.chirp_stop_hz})
    elif w.value in {"bpsk","qpsk","8psk","16qam","64qam","dbpsk","dqpsk","pi4-dqpsk","oqpsk","16apsk","32apsk"}:
        out = _digital_linear(cfg, n, rng, w.value)
    elif w == WaveformKind.FSK2:
        out = _cpfsk(cfg, n, rng, 2)
    elif w == WaveformKind.FSK4:
        out = _cpfsk(cfg, n, rng, 4)
    elif w in {WaveformKind.MSK, WaveformKind.GMSK}:
        out = _cpfsk(cfg, n, rng, 2, msk=True)
    elif w == WaveformKind.OFDM:
        out = _ofdm(cfg, n, rng,start_s)
    elif w in {WaveformKind.AM, WaveformKind.FM, WaveformKind.PM, WaveformKind.SSB, WaveformKind.DSB_SC}:
        out = GeneratedSignal(_analog(cfg, t).astype(np.complex64), fs, {"waveform":w.value, "message_freq_hz":cfg.message_freq_hz})
    elif w in RADAR_WAVEFORMS:
        out = _radar(cfg, n, rng)
    else:
        raise ValueError(f"unsupported waveform: {w}")

    y = out.samples.astype(np.complex128, copy=False)
    if w == WaveformKind.AWGN and cfg.noise_bandwidth_hz:
        bw = min(cfg.noise_bandwidth_hz, fs*.98)
        h = signal.firwin(129, bw/2, fs=fs)
        y = signal.lfilter(h, [1], y)
    if abs(cfg.phase_offset_deg) > 1e-12:
        y *= np.exp(1j*np.deg2rad(cfg.phase_offset_deg))
    if abs(cfg.center_hz) > 1e-12:
        y *= np.exp(1j*2*np.pi*cfg.center_hz*(t[:len(y)]+start_s))
    y = normalize_rms(y, cfg.amplitude) if cfg.normalize_output else y*cfg.amplitude
    if cfg.snr_db is not None and w != WaveformKind.AWGN:
        y = add_awgn(y, cfg.snr_db, rng)
    if cfg.iq_gain_db or cfg.iq_phase_deg:
        gain = 10**(cfg.iq_gain_db/20)
        i, q = y.real*gain, y.imag/gain
        y = i + 1j*(q*np.cos(np.deg2rad(cfg.iq_phase_deg))+i*np.sin(np.deg2rad(cfg.iq_phase_deg)))
    y += cfg.dc_i + 1j*cfg.dc_q
    if cfg.burst_enabled:
        y *= (t % cfg.burst_period_s) < cfg.burst_period_s*cfg.burst_duty
    out.samples = y.astype(np.complex64)
    reference_rms = float(np.sqrt(np.mean(np.abs(out.samples.astype(np.complex128))**2))) if len(out.samples) else 0.0
    out.metadata.update({
        "sample_rate":fs, "duration":len(y)/fs, "center_hz":cfg.center_hz, "snr_db":cfg.snr_db,
        "power_dbm":float(cfg.power_dbm), "power_reference_rms":reference_rms,
        "power_calibration":"Average complex-sample power is calibrated to the configured dBm value; 1 mW reference."
    })
    return out
