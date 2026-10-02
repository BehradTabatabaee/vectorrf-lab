from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import math
import numpy as np
from pydantic import BaseModel, Field, model_validator

from .models import MAX_SAMPLE_RATE, TxConfig
from .signal import generate
from .spectrum import analyze as spectrum_analyze
from .models import SpectrumConfig


class RadarPulseConfig(BaseModel):
    name: str = "Pulse 1"
    waveform: Literal[
        "pulse", "lfm", "nlfm", "barker", "frank", "p1", "p2", "p3", "p4",
        "zadoff-chu", "custom-phase", "costas", "stepped-frequency"
    ] = "lfm"
    width_s: float = Field(default=20e-6, gt=1e-8, le=0.5)
    amplitude: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_offset_hz: float = 0.0
    phase_offset_deg: float = 0.0
    envelope: Literal["rect", "gaussian", "raised-cosine"] = "rect"
    sweep_bw_hz: float = Field(default=200_000.0, gt=1.0)
    nlfm_exponent: float = Field(default=3.0, ge=1.0, le=9.0)
    barker_length: Literal[2, 3, 4, 5, 7, 11, 13] = 13
    polyphase_order: int = Field(default=4, ge=2, le=32)
    phase_code_chips: int = Field(default=32, ge=2, le=4096)
    zc_root: int = Field(default=5, ge=1, le=4095)
    custom_phase_deg: list[float] = Field(default_factory=lambda: [0.0, 180.0, 0.0, 0.0, 180.0, 0.0, 180.0])
    costas_hops: int = Field(default=7, ge=3, le=64)
    stepped_steps: int = Field(default=8, ge=2, le=256)
    stepped_spacing_hz: float = Field(default=20_000.0, gt=0.0)


class RadarSequenceItem(BaseModel):
    kind: Literal["pulse", "off"] = "pulse"
    name: str = "Pulse"
    pulse: RadarPulseConfig | None = Field(default_factory=RadarPulseConfig)
    repeat: int = Field(default=1, ge=1, le=100_000)
    pri_s: float = Field(default=100e-6, gt=1e-8, le=2.0)
    off_time_s: float = Field(default=100e-6, gt=1e-8, le=2.0)
    pri_jitter_fraction: float = Field(default=0.0, ge=0.0, le=0.8)
    pri_wobble_fraction: float = Field(default=0.0, ge=0.0, le=0.8)
    pri_wobble_period: int = Field(default=16, ge=2, le=4096)
    scale: float = Field(default=1.0, ge=0.0, le=2.0)
    frequency_offset_hz: float = 0.0
    phase_offset_deg: float = 0.0

    @model_validator(mode="after")
    def _validate_item(self):
        if self.kind == "pulse":
            if self.pulse is None:
                raise ValueError("pulse item requires pulse configuration")
            if self.pulse.width_s > self.pri_s:
                raise ValueError("pulse width must be <= PRI")
        return self


class RadarSegment(BaseModel):
    duration_us: float = Field(default=25,gt=0,le=100000)
    start_hz: float = 0
    stop_hz: float = 0
    amplitude: float = Field(default=1,ge=0,le=2)
    phase_deg: float = 0


class RadarStudioConfig(BaseModel):
    mode: Literal["continuous", "sequence", "segments"] = "sequence"
    sample_rate: float = Field(default=1_000_000.0, gt=1_000.0, le=MAX_SAMPLE_RATE)
    duration_s: float = Field(default=0.02, gt=1e-5, le=1.0)
    seed: int = 7
    segments: list[RadarSegment] = Field(default_factory=lambda:[RadarSegment(start_hz=-100000,stop_hz=100000),RadarSegment(amplitude=0)],min_length=1,max_length=128)

    # Continuous-wave radar modes.
    continuous_waveform: Literal["cw", "fmcw", "pmcw", "stepped-cw"] = "fmcw"
    center_offset_hz: float = 0.0
    sweep_bw_hz: float = Field(default=200_000.0, gt=1.0)
    sweep_period_s: float = Field(default=250e-6, gt=1e-8, le=2.0)
    sweep_shape: Literal["up", "down", "triangle"] = "triangle"
    pmcw_chip_rate: float = Field(default=250_000.0, gt=10.0)
    pmcw_chips: int = Field(default=127, ge=2, le=65_536)
    stepped_start_hz: float = -100_000.0
    stepped_stop_hz: float = 100_000.0
    stepped_points: int = Field(default=16, ge=2, le=4096)
    stepped_dwell_s: float = Field(default=20e-6, gt=1e-8, le=1.0)

    sequence: list[RadarSequenceItem] = Field(max_length=64, default_factory=lambda: [
        RadarSequenceItem(
            kind="pulse",
            name="LFM burst",
            pulse=RadarPulseConfig(name="LFM burst", waveform="lfm", width_s=30e-6, sweep_bw_hz=250_000.0),
            repeat=8,
            pri_s=150e-6,
        ),
        RadarSequenceItem(kind="off", name="Off time", pulse=None, repeat=1, off_time_s=400e-6),
    ])

    @model_validator(mode="after")
    def _validate_cfg(self):
        if self.duration_s*self.sample_rate>2_000_000:
            raise ValueError("Radar record is limited to 2 million samples")
        nyq = self.sample_rate / 2.0
        if self.mode == "segments" and any(max(abs(x.start_hz),abs(x.stop_hz))>=nyq for x in self.segments):
            raise ValueError("Segment frequencies must lie inside Nyquist")
        if abs(self.center_offset_hz) >= nyq:
            raise ValueError("center offset must lie inside Nyquist")
        if self.sweep_bw_hz >= self.sample_rate:
            raise ValueError("sweep bandwidth must be below sample rate")
        if max(abs(self.stepped_start_hz), abs(self.stepped_stop_hz)) >= nyq:
            raise ValueError("stepped CW frequencies must lie inside Nyquist")
        if sum(s.duration_us for s in self.segments)*1e-6*self.sample_rate>2_000_000:
            raise ValueError("Segment sequence exceeds 2 million samples")
        budget=sum(item.repeat*(item.off_time_s if item.kind=='off' else item.pri_s*(1+item.pri_jitter_fraction+item.pri_wobble_fraction)) for item in self.sequence)*self.sample_rate
        if budget>2_000_000:
            raise ValueError("Logical radar sequence exceeds 2 million samples; reduce repetition or PRI")
        for item in self.sequence:
            if item.kind == "pulse" and item.pulse is not None:
                if abs(item.frequency_offset_hz + item.pulse.frequency_offset_hz) + item.pulse.sweep_bw_hz / 2.0 >= nyq and item.pulse.waveform in {"lfm", "nlfm", "costas"}:
                    raise ValueError(f"{item.name}: pulse spectrum exceeds Nyquist")
        return self


@dataclass
class RadarStudioResult:
    samples: np.ndarray
    sample_rate: float
    metadata: dict


def _pulse_samples(pulse: RadarPulseConfig, fs: float, seed: int) -> np.ndarray:
    # Reuse the normal waveform engine for each pulse definition.  Setting duration
    # equal to PRI/width yields exactly one active pulse at t=0.
    cfg = TxConfig(
        waveform=pulse.waveform,
        sample_rate=fs,
        duration=max(pulse.width_s,32/fs,1.00001e-5),
        amplitude=1.0,
        center_hz=pulse.frequency_offset_hz,
        phase_offset_deg=pulse.phase_offset_deg,
        snr_db=None,
        seed=seed,
        pulse_width_s=pulse.width_s,
        pri_s=pulse.width_s,
        pulse_envelope=pulse.envelope,
        sweep_bw_hz=pulse.sweep_bw_hz,
        nlfm_exponent=pulse.nlfm_exponent,
        barker_length=pulse.barker_length,
        polyphase_order=pulse.polyphase_order,
        phase_code_chips=pulse.phase_code_chips,
        zc_root=pulse.zc_root,
        custom_phase_deg=pulse.custom_phase_deg,
        costas_hops=pulse.costas_hops,
        stepped_steps=pulse.stepped_steps,
        stepped_spacing_hz=pulse.stepped_spacing_hz,
    )
    return generate(cfg).samples[:max(1,int(round(pulse.width_s*fs)))].astype(np.complex128) * pulse.amplitude


def _continuous(cfg: RadarStudioConfig) -> RadarStudioResult:
    fs = cfg.sample_rate
    n = max(32, int(round(cfg.duration_s * fs)))
    t = np.arange(n, dtype=np.float64) / fs
    rng = np.random.default_rng(cfg.seed)

    if cfg.continuous_waveform == "cw":
        y = np.exp(1j * 2.0 * np.pi * cfg.center_offset_hz * t)
        meta = {"mode": "continuous", "waveform": "cw"}
    elif cfg.continuous_waveform == "fmcw":
        tx = TxConfig(
            waveform="fmcw", sample_rate=fs, duration=cfg.duration_s, amplitude=1.0,
            center_hz=cfg.center_offset_hz, snr_db=None, pri_s=cfg.sweep_period_s,
            sweep_bw_hz=cfg.sweep_bw_hz, fmcw_shape=cfg.sweep_shape, seed=cfg.seed,
        )
        g = generate(tx)
        y = g.samples.astype(np.complex128)
        meta = {"mode": "continuous", **g.metadata}
    elif cfg.continuous_waveform == "pmcw":
        chip_sps = max(1, int(round(fs / cfg.pmcw_chip_rate)))
        code = rng.choice(np.asarray([-1.0, 1.0]), size=cfg.pmcw_chips)
        seq = np.repeat(code, chip_sps).astype(np.complex128)
        y = np.resize(seq, n)
        if cfg.center_offset_hz:
            y *= np.exp(1j * 2.0 * np.pi * cfg.center_offset_hz * t)
        meta = {"mode": "continuous", "waveform": "pmcw", "chip_rate_hz": fs / chip_sps, "chips": cfg.pmcw_chips}
    else:  # stepped-cw
        freqs = np.linspace(cfg.stepped_start_hz, cfg.stepped_stop_hz, cfg.stepped_points)
        dwell = max(1, int(round(cfg.stepped_dwell_s * fs)))
        f = np.resize(np.repeat(freqs, dwell), n) + cfg.center_offset_hz
        phase = 2.0 * np.pi * np.cumsum(f) / fs
        y = np.exp(1j * phase)
        meta = {"mode": "continuous", "waveform": "stepped-cw", "points": cfg.stepped_points, "dwell_s": dwell / fs}
    return RadarStudioResult(np.asarray(y, dtype=np.complex64), fs, meta)


def _sequence(cfg: RadarStudioConfig) -> RadarStudioResult:
    fs = cfg.sample_rate
    target_n = max(32, int(round(cfg.duration_s * fs)))
    rng = np.random.default_rng(cfg.seed)
    pieces: list[np.ndarray] = []
    timeline: list[dict] = []
    total_n = 0
    occurrence = 0

    # Build one logical sequence then repeat it to the requested capture length.
    for item_idx, item in enumerate(cfg.sequence):
        if item.kind == "off":
            for r in range(item.repeat):
                m = max(1, int(round(item.off_time_s * fs)))
                pieces.append(np.zeros(m, dtype=np.complex128))
                timeline.append({"kind": "off", "name": item.name, "start_s": total_n / fs, "duration_s": m / fs})
                total_n += m
            continue

        pulse = item.pulse
        assert pulse is not None
        base = _pulse_samples(pulse, fs, cfg.seed + item_idx)
        for r in range(item.repeat):
            wobble = item.pri_wobble_fraction * math.sin(2.0 * math.pi * occurrence / max(2, item.pri_wobble_period))
            jitter = rng.uniform(-item.pri_jitter_fraction, item.pri_jitter_fraction) if item.pri_jitter_fraction else 0.0
            pri = max(pulse.width_s, item.pri_s * (1.0 + wobble + jitter))
            slot_n = max(len(base), int(round(pri * fs)))
            slot = np.zeros(slot_n, dtype=np.complex128)
            m = min(len(base), slot_n)
            q = base[:m] * item.scale
            phase_deg = item.phase_offset_deg
            freq_hz = item.frequency_offset_hz
            if phase_deg or freq_hz:
                tt = np.arange(m, dtype=float) / fs
                q = q * np.exp(1j * (np.deg2rad(phase_deg) + 2.0 * np.pi * freq_hz * tt))
            slot[:m] = q
            pieces.append(slot)
            timeline.append({
                "kind": "pulse", "name": item.name, "waveform": pulse.waveform,
                "start_s": total_n / fs, "width_s": m / fs, "pri_s": slot_n / fs,
                "scale": item.scale, "frequency_offset_hz": item.frequency_offset_hz + pulse.frequency_offset_hz,
            })
            total_n += slot_n
            occurrence += 1

    logical = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.complex128)
    if logical.size < target_n:
        y = np.resize(logical, target_n)
    else:
        y = logical[:target_n]
    meta = {
        "mode": "sequence", "waveform": "pulse-sequence", "sequence_items": len(cfg.sequence),
        "logical_sequence_s": logical.size / fs, "timeline": timeline[:512],
    }
    return RadarStudioResult(y.astype(np.complex64), fs, meta)


def generate_radar(cfg: RadarStudioConfig) -> RadarStudioResult:
    if cfg.mode == "segments":
        fields={"arbitrary_duration_us":[s.duration_us for s in cfg.segments],
            "arbitrary_start_freq_hz":[s.start_hz for s in cfg.segments],
            "arbitrary_stop_freq_hz":[s.stop_hz for s in cfg.segments],
            "arbitrary_amplitude":[s.amplitude for s in cfg.segments],
            "arbitrary_phase_deg":[s.phase_deg for s in cfg.segments]}
        g=generate(TxConfig(waveform="radar-arbitrary",sample_rate=cfg.sample_rate,duration=cfg.duration_s,snr_db=None,amplitude=1,**fields))
        timeline=[];start=0
        for i,seg in enumerate(cfg.segments):
            timeline.append({'kind':'pulse' if seg.amplitude else 'off','name':f'Segment {i+1}','start_s':start,'duration_s':seg.duration_us*1e-6})
            start+=seg.duration_us*1e-6
        return RadarStudioResult(g.samples,g.sample_rate,{**g.metadata,'mode':'segments','waveform':'segmented','timeline':timeline,'logical_sequence_s':start})
    return _continuous(cfg) if cfg.mode == "continuous" else _sequence(cfg)


def radar_products(cfg: RadarStudioConfig, max_points: int = 4096) -> dict:
    result = generate_radar(cfg)
    x = result.samples.astype(np.complex128)
    fs = result.sample_rate
    stride = max(1, int(np.ceil(len(x) / max_points)))
    xd = x[::stride][:max_points]
    inst_full = np.r_[0.0, np.angle(x[1:]*np.conj(x[:-1])) * fs/(2*np.pi)]
    instf = inst_full[::stride][:max_points]

    sp = spectrum_analyze(x, fs, SpectrumConfig(rbw_hz=max(100.0, fs / 4096.0), averaging=2, max_fft=32768))

    # Pulse-compression view: use the first active island when possible.
    amp = np.abs(x)
    active = np.flatnonzero(amp > 0.08 * max(1e-12, float(np.max(amp))))
    if active.size:
        gaps = np.flatnonzero(np.diff(active) > 1)
        end_i = gaps[0] + 1 if gaps.size else active.size
        ref = x[active[:end_i]]
    else:
        ref = x[:min(len(x), 4096)]
    ref_stride=max(1,int(np.ceil(len(ref)/8192)))
    ref = ref[::ref_stride]
    ref_fs=fs/ref_stride
    if len(ref) >= 2:
        ac = np.correlate(ref, ref, mode="full")[len(ref)-1:]
        ac = np.abs(ac)
        ac = ac / max(1e-12, float(ac[0]))
        ac_db = 20.0 * np.log10(np.maximum(ac, 1e-6))
    else:
        ac_db = np.zeros(1)

    # Bounded delay-Doppler ambiguity surface.  This is a visualization product,
    # not a metrology-grade ambiguity implementation.
    amb_ref = ref
    amb_stride=max(1,int(np.ceil(len(amb_ref)/768)))
    amb_ref=amb_ref[::amb_stride]
    amb_fs=ref_fs/amb_stride
    ndelay = min(97, max(17, 2 * (len(amb_ref) // 24) + 1))
    max_lag = max(1, min(len(amb_ref) - 2, ndelay // 2)) if len(amb_ref) > 3 else 1
    lags = np.linspace(-max_lag, max_lag, ndelay).astype(int)
    dopplers = np.linspace(-0.25 * amb_fs, 0.25 * amb_fs, 49)
    ambiguity = np.zeros((len(dopplers), len(lags)), dtype=np.float32)
    if len(amb_ref) >= 4:
        n = np.arange(len(amb_ref), dtype=float)
        for di, fd in enumerate(dopplers):
            mixed = amb_ref * np.exp(-1j * 2.0 * np.pi * fd * n / amb_fs)
            for li, lag in enumerate(lags):
                if lag >= 0:
                    a, b = mixed[lag:], amb_ref[:len(amb_ref)-lag]
                else:
                    q = -lag
                    a, b = mixed[:len(amb_ref)-q], amb_ref[q:]
                ambiguity[di, li] = abs(np.vdot(b, a)) if len(a) else 0.0
        peak = float(np.max(ambiguity))
        if peak > 0:
            ambiguity /= peak
        ambiguity = 20.0 * np.log10(np.maximum(ambiguity, 1e-4))

    return {
        "metadata": result.metadata,
        "sample_rate": fs,
        "time_s": (np.arange(len(xd)) * stride / fs).astype(float).tolist(),
        "envelope": np.abs(xd).astype(float).tolist(),
        "i": xd.real.astype(float).tolist(),
        "q": xd.imag.astype(float).tolist(),
        "inst_freq_hz": instf.astype(float).tolist(),
        "phase_deg": np.rad2deg(np.angle(xd)).astype(float).tolist(),
        "freq_hz": sp.freq_hz.astype(float).tolist(),
        "psd_db": sp.psd_db.astype(float).tolist(),
        "autocorr_db": ac_db.astype(float).tolist(),
        "autocorr_delay_s": (np.arange(len(ac_db))/ref_fs).tolist(),
        "ambiguity_delay_s": (lags/amb_fs).tolist(),
        "ambiguity_db": ambiguity.reshape(-1).astype(float).tolist(),
        "ambiguity_shape": [int(ambiguity.shape[0]), int(ambiguity.shape[1])],
        "ambiguity_delay_samples": lags.astype(int).tolist(),
        "ambiguity_doppler_hz": dopplers.astype(float).tolist(),
        "metrics": {
            "peak_hz": sp.peak_hz,
            "peak_db": sp.peak_db,
            "occupied_bw_hz": sp.bandwidth_99_hz,
            "rbw_hz": sp.rbw_actual_hz,
            "crest_factor_db": float(20.0 * np.log10((np.max(np.abs(x)) + 1e-12) / (np.sqrt(np.mean(np.abs(x) ** 2)) + 1e-12))),
        },
    }
