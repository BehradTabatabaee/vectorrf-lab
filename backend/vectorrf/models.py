from __future__ import annotations

from enum import Enum
import os
from typing import Literal
from pydantic import BaseModel, Field, model_validator, ConfigDict, ValidationInfo

from .agility import FrequencyAgility, frequency_extent
from .ofdm import OFDMDesign
from .programs import WaveProgram
from .units import eng


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except ValueError:
        value = default
    return max(lo, min(hi, value))


# Sample rate is a scale in a simulator: the engine bounds the number of
# samples per capture, not the rate, so a capture simply spans less time at
# higher rates.  5 GS/s covers RFSoC-class direct-sampling ADCs; every DSP path
# is written in samples/symbols and validated with GS/s-rate tests.
MAX_SAMPLE_RATE = 5_000_000_000.0
MAX_SPECTRUM_SPAN = MAX_SAMPLE_RATE
# Samples in one live/HTTP capture.  8 Mi complex samples = 64 MiB of complex64
# (4.2 ms at 2 GS/s).  Raise with VECTORRF_MAX_CAPTURE_SAMPLES on large machines.
MAX_CAPTURE_SAMPLES = _env_int("VECTORRF_MAX_CAPTURE_SAMPLES", 1 << 23, 1 << 16, 1 << 28)
# Exports may render one longer contiguous record.
MAX_EXPORT_SAMPLES = max(MAX_CAPTURE_SAMPLES, _env_int("VECTORRF_MAX_EXPORT_SAMPLES", 1 << 25, 1 << 16, 1 << 30))
RECOMMENDED_LIVE_SAMPLE_RATE = 5_000_000.0
# Signals in one composite transmitter (each is a complete transmitter design).
MAX_COMPONENTS = 16
# Pulse shaping and the receiver's polyphase decimator cost the same per sample at
# any oversampling, so the bound only keeps filters a sensible size (50 kBd at 3.3 GS/s).
MAX_SAMPLES_PER_SYMBOL = 65_536


class WaveformKind(str, Enum):
    # Sum of independently designed signals
    COMPOSITE = "composite"

    # Fundamentals
    CUSTOM = "custom"
    TONE = "tone"
    MULTITONE = "multitone"
    AWGN = "awgn"
    CHIRP = "chirp"

    # Digital communications
    BPSK = "bpsk"
    QPSK = "qpsk"
    DBPSK = "dbpsk"
    DQPSK = "dqpsk"
    OQPSK = "oqpsk"
    PI4DQPSK = "pi4-dqpsk"
    APSK16 = "16apsk"
    APSK32 = "32apsk"
    GMSK = "gmsk"
    PSK8 = "8psk"
    QAM16 = "16qam"
    QAM64 = "64qam"
    FSK2 = "2fsk"
    FSK4 = "4fsk"
    MSK = "msk"
    OFDM = "ofdm"

    # Analog communications
    AM = "am"
    FM = "fm"
    PM = "pm"
    SSB = "ssb"
    DSB_SC = "dsb-sc"

    # Radar waveform workbench
    CW = "radar-cw"
    PULSE = "pulse"
    LFM = "lfm"
    NLFM = "nlfm"
    BARKER = "barker"
    FRANK = "frank"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"
    ZADOFF_CHU = "zadoff-chu"
    CUSTOM_PHASE = "custom-phase"
    COSTAS = "costas"
    STEPPED_FREQ = "stepped-frequency"
    PMCW = "pmcw"
    FMCW = "fmcw"
    ARBITRARY_RADAR = "radar-arbitrary"


RADAR_WAVEFORMS = {
    WaveformKind.CW, WaveformKind.PULSE, WaveformKind.LFM, WaveformKind.NLFM,
    WaveformKind.BARKER, WaveformKind.FRANK, WaveformKind.P1, WaveformKind.P2,
    WaveformKind.P3, WaveformKind.P4, WaveformKind.ZADOFF_CHU,
    WaveformKind.CUSTOM_PHASE, WaveformKind.COSTAS, WaveformKind.STEPPED_FREQ,
    WaveformKind.PMCW, WaveformKind.FMCW, WaveformKind.ARBITRARY_RADAR,
}




FRAME_MODULATIONS = (
    "bpsk", "qpsk", "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk", "8psk",
    "16qam", "64qam", "16apsk", "32apsk", "2fsk", "4fsk", "msk", "gmsk", "ofdm",
)


class FrameBlock(BaseModel):
    """One ordered block in a generic communications frame.

    The frame builder intentionally models generic laboratory framing rather than
    any named air interface.  Blocks may use different digital modulations and
    symbol rates while sharing the transmitter sample clock.
    """
    model_config = ConfigDict(allow_inf_nan=False)
    name: str = Field(default="Block", min_length=1, max_length=64)
    role: Literal["preamble", "payload", "bits", "pn"] = "payload"
    modulation: Literal[
        "bpsk", "qpsk", "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk", "8psk",
        "16qam", "64qam", "16apsk", "32apsk", "2fsk", "4fsk", "msk", "gmsk", "ofdm",
    ] = "qpsk"
    bit_count: int = Field(default=128, ge=1, le=65536)
    bits: str = Field(default="1010101010101010", max_length=65536)
    symbol_rate: float | None = Field(default=None, gt=10.0)
    power_offset_db: float = Field(default=0.0, ge=-120.0, le=120.0)
    repeat: int = Field(default=1, ge=1, le=1024)

    @model_validator(mode="after")
    def _validate_block(self):
        if self.role in {"preamble", "bits"} and (not self.bits or set(self.bits) - {"0", "1"}):
            raise ValueError("Frame block bits must contain only 0 and 1")
        return self


class TxConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    waveform: WaveformKind = WaveformKind.QPSK
    sample_rate: float = Field(default=1_000_000.0, gt=1_000.0, le=MAX_SAMPLE_RATE)
    # Capture length in seconds.  Its sample count (duration × rate) is bounded
    # by MAX_CAPTURE_SAMPLES rather than by a fixed time limit.
    duration: float = Field(default=0.05, gt=1e-9, le=600.0)
    # Legacy sample-unit controls are retained for API compatibility.  Absolute
    # transmit power is carried separately and calibrates instrument readouts.
    normalize_output: bool = True
    amplitude: float = Field(default=0.8, gt=0.0, le=2.0)
    power_dbm: float = 0.0
    center_hz: float = 0.0
    snr_db: float | None = Field(default=None, ge=-30.0, le=100.0)
    seed: int = 7
    # RF frequency that complex baseband 0 Hz represents (0 = plain baseband).
    # It labels instrument axes and exports; every other frequency here is a
    # baseband offset from it.
    rf_center_hz: float = Field(default=0.0, ge=0.0, le=1e12)
    # Hopping or sweeping applied to the finished signal (any waveform).
    agility: FrequencyAgility = Field(default_factory=FrequencyAgility)
    # Composite waveform: the sum of complete component designs.  Components
    # share this sample clock and capture length; each keeps its own absolute
    # power, frequency offset, burst gating and agility.
    components: list["TxConfig"] = Field(default_factory=list, max_length=MAX_COMPONENTS)
    label: str = Field(default="", max_length=64)
    enabled: bool = True

    # Message / digital modulation
    data_source: Literal["text", "pn", "bits"] = "text"
    message: str = Field(default="VECTORRF LAB", max_length=65536)
    custom_bits: str = Field(default="11010010", max_length=65536)
    pn_order: Literal[7, 9, 11, 15] = 15
    frame_enabled: bool = False
    frame_mode: Literal["simple", "blocks"] = "simple"
    frame_blocks: list[FrameBlock] = Field(default_factory=list, max_length=64)
    preamble_bits: str = Field(default="10101010101010101101001110010001",max_length=1024)
    payload_bits: int = Field(default=256,ge=8,le=65536)
    frame_crc: bool = False
    pulse_shape: Literal["rrc", "rect"] = "rrc"
    gaussian_bt: float = Field(default=0.3, ge=0.1, le=1.0)
    apsk_ratio2: float = Field(default=2.85, gt=1.0, le=8)
    apsk_ratio3: float = Field(default=5.27, gt=1.0, le=12)
    burst_enabled: bool = False
    burst_period_s: float = Field(default=0.01, ge=0.00001, le=2)
    burst_duty: float = Field(default=0.5, gt=0, le=1)
    noise_bandwidth_hz: float | None = Field(default=None, gt=1)
    iq_gain_db: float = Field(default=0, ge=-12, le=12)
    iq_phase_deg: float = Field(default=0, ge=-45, le=45)
    dc_i: float = Field(default=0, ge=-1, le=1)
    dc_q: float = Field(default=0, ge=-1, le=1)
    pn_bits: int = Field(default=4096, ge=32, le=10_000_000)
    symbol_rate: float = Field(default=50_000.0, gt=10.0)
    rolloff: float = Field(default=0.35, ge=0.0, le=1.0)
    filter_span_symbols: int = Field(default=10, ge=2, le=64)
    fsk_deviation_hz: float = Field(default=25_000.0, gt=1.0)
    phase_offset_deg: float = 0.0

    # Analog message source
    message_freq_hz: float = Field(default=5_000.0, gt=0.0)
    modulation_index: float = Field(default=0.6, ge=0.0, le=10.0)
    fm_deviation_hz: float = Field(default=25_000.0, ge=0.0)
    pm_deviation_rad: float = Field(default=1.0, ge=0.0, le=20.0)
    sideband: Literal["usb", "lsb"] = "usb"

    # Multi-tone / chirp
    tones_hz: list[float] = Field(default_factory=lambda: [-80_000.0, 45_000.0, 130_000.0])
    chirp_start_hz: float = -100_000.0
    chirp_stop_hz: float = 100_000.0
    # Sweep period of the (sawtooth) chirp; None = the capture length.
    chirp_period_s: float | None = Field(default=None, gt=1e-9, le=600.0)

    program: WaveProgram = Field(default_factory=WaveProgram)
    ofdm_design: OFDMDesign = Field(default_factory=OFDMDesign)

    # OFDM
    ofdm_fft: int = Field(default=64, ge=16, le=4096)
    ofdm_cp: int = Field(default=16, ge=0, le=1024)
    ofdm_occupied: int = Field(default=52, ge=4, le=4094)

    # Radar timing / train controls
    pulse_width_s: float = Field(default=20e-6, gt=1e-8, le=1.0)
    pri_s: float = Field(default=100e-6, gt=1e-8, le=2.0)
    pri_mode: Literal["fixed", "staggered", "jittered"] = "fixed"
    pri_stagger: list[float] = Field(default_factory=lambda: [1.0, 1.08, 0.94])
    pri_jitter_fraction: float = Field(default=0.08, ge=0.0, le=0.8)
    pulse_envelope: Literal["rect", "gaussian", "raised-cosine"] = "rect"

    # Radar frequency/chirp controls
    sweep_bw_hz: float = Field(default=200_000.0, gt=1.0)
    fmcw_shape: Literal["up", "down", "triangle"] = "up"
    nlfm_exponent: float = Field(default=3.0, ge=1.0, le=9.0)
    stepped_steps: int = Field(default=8, ge=2, le=256)
    stepped_spacing_hz: float = Field(default=20_000.0, gt=0.0)

    # Radar coding controls
    barker_length: Literal[2, 3, 4, 5, 7, 11, 13] = 13
    polyphase_order: int = Field(default=4, ge=2, le=32)
    phase_code_chips: int = Field(default=32, ge=2, le=4096)
    zc_root: int = Field(default=5, ge=1, le=4095)
    # Barker 7 (+ + + − − + −): peak sidelobes at −16.9 dB.
    custom_phase_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 180.0, 180.0, 0.0, 180.0])
    costas_hops: int = Field(default=7, ge=3, le=64)
    pmcw_chip_rate: float = Field(default=250_000.0, gt=10.0)

    # Generic segmented radar waveform. Values are cycled if lists differ in length.
    arbitrary_duration_us: list[float] = Field(default_factory=lambda: [20.0, 30.0, 20.0, 30.0])
    arbitrary_start_freq_hz: list[float] = Field(default_factory=lambda: [-50_000.0, 0.0, 50_000.0, 0.0])
    arbitrary_stop_freq_hz: list[float] = Field(default_factory=lambda: [50_000.0, 0.0, -50_000.0, 0.0])
    arbitrary_amplitude: list[float] = Field(default_factory=lambda: [1.0, 0.0, 0.8, 0.0])
    arbitrary_phase_deg: list[float] = Field(default_factory=lambda: [0.0, 0.0, 180.0, 0.0])

    @model_validator(mode="before")
    @classmethod
    def _share_clock(cls, data):
        """Components always run on the parent's sample clock and capture length."""
        if isinstance(data, dict) and data.get("components"):
            shared = {key: data.get(key, cls.model_fields[key].default)
                      for key in ("sample_rate", "duration", "rf_center_hz")}
            components = [{**(c.model_dump() if isinstance(c, BaseModel) else dict(c)), **shared}
                          for c in data["components"]]
            data = {**data, "components": components}
        return data

    @model_validator(mode="after")
    def _validate(self, info: ValidationInfo):
        if self.frame_enabled:
            if self.frame_mode == "blocks" and not self.frame_blocks:
                raise ValueError("Block frame mode requires at least one frame block")
            for block in self.frame_blocks:
                rate = block.symbol_rate or self.symbol_rate
                if rate >= self.sample_rate / 1.5:
                    raise ValueError(f"Frame block {block.name!r} symbol_rate must be comfortably below sample_rate")
                if self.sample_rate / rate > MAX_SAMPLES_PER_SYMBOL:
                    raise ValueError(f"Frame block {block.name!r} supports at most {MAX_SAMPLES_PER_SYMBOL:,} samples per symbol")
            if set(self.preamble_bits)-{'0','1'}:
                raise ValueError("Preamble must contain 0 and 1 only")
            if self.frame_crc and self.payload_bits % 8:
                raise ValueError("CRC requires a whole number of payload bytes")
        if self.data_source == "text" and not self.message:
            raise ValueError("Text payload cannot be empty; select PN or enter a message")
        if self.data_source == "bits" and (not self.custom_bits or set(self.custom_bits) - {"0", "1"}):
            raise ValueError("custom_bits must contain only 0 and 1")
        if self.waveform == WaveformKind.APSK32 and self.apsk_ratio3 <= self.apsk_ratio2:
            raise ValueError("APSK outer radius must exceed the middle radius")
        limit = int((info.context or {}).get("max_samples", MAX_CAPTURE_SAMPLES)) if info is not None else MAX_CAPTURE_SAMPLES
        if round(self.sample_rate * self.duration) > limit:
            raise ValueError(f"capture length {eng(self.duration, 's')} is {round(self.sample_rate * self.duration):,} samples at "
                             f"{eng(self.sample_rate, 'S/s')}; a capture holds at most {limit:,} samples ({eng(limit / self.sample_rate, 's', 4)} at this rate)")
        nyq = self.sample_rate / 2.0
        band = f"Nyquist: at {eng(self.sample_rate, 'S/s')} the sampled band is ±{eng(nyq)}"
        if abs(self.center_hz) >= nyq:
            raise ValueError(f"frequency offset {eng(self.center_hz)} is beyond {band}")
        if self.waveform == WaveformKind.COMPOSITE:
            if not self.components:
                raise ValueError("A composite signal needs at least one component")
            for c in self.components:
                if c.waveform == WaveformKind.COMPOSITE or c.components:
                    raise ValueError("Composite components cannot themselves be composite")
            if not any(c.enabled for c in self.components):
                raise ValueError("Enable at least one composite component")
        if self.waveform == WaveformKind.MULTITONE:
            outside = [f for f in self.tones_hz if abs(f) >= nyq]
            if outside:
                raise ValueError(f"tone {eng(outside[0])} is beyond {band}")
        lo, hi = carrier_extent(self)
        if beyond_nyquist(self):
            def reach(a, b):
                return eng(a) if a == b else f"{eng(a)} … {eng(b)}"
            culprits = []
            if self.waveform == WaveformKind.COMPOSITE:
                shift = frequency_extent(self.agility)
                for i, c in enumerate(self.components):
                    a, b = carrier_extent(c)
                    a, b = self.center_hz + a + shift[0], self.center_hz + b + shift[1]
                    if c.enabled and beyond_nyquist(c, self.center_hz, shift):
                        culprits.append(f"{c.label or f'Signal {i + 1}'} reaches {reach(a, b)}")
            what = "; ".join(culprits) if culprits else f"the signal reaches {reach(lo, hi)}"
            raise ValueError(f"{what} (carrier, offsets and hopping/sweep combined), beyond {band}")
        linear={"bpsk","qpsk","dbpsk","dqpsk","pi4-dqpsk","oqpsk","8psk","16qam","64qam","16apsk","32apsk"}
        if self.waveform.value in linear | {"2fsk","4fsk","msk","gmsk"} and self.sample_rate/self.symbol_rate>MAX_SAMPLES_PER_SYMBOL:
            raise ValueError(f"symbol rate {eng(self.symbol_rate, 'Bd')} gives {self.sample_rate/self.symbol_rate:,.0f} samples per symbol at "
                             f"{eng(self.sample_rate, 'S/s')}; at most {MAX_SAMPLES_PER_SYMBOL:,} are supported (use at least {eng(self.sample_rate/MAX_SAMPLES_PER_SYMBOL, 'Bd', 4)})")
        if self.waveform.value in linear | {"2fsk","4fsk","msk","gmsk"} and self.symbol_rate >= self.sample_rate / 1.5:
            raise ValueError(f"symbol rate {eng(self.symbol_rate, 'Bd')} needs more than 1.5 samples per symbol; at {eng(self.sample_rate, 'S/s')} use less than {eng(self.sample_rate / 1.5, 'Bd', 4)}")
        if self.waveform.value in {"am","fm","pm","ssb","dsb-sc"} and self.message_freq_hz >= nyq:
            raise ValueError(f"message tone {eng(self.message_freq_hz)} is beyond {band}")
        if self.waveform == WaveformKind.OFDM and not self.ofdm_design.enabled and self.ofdm_occupied >= self.ofdm_fft:
            raise ValueError("ofdm_occupied must be smaller than ofdm_fft")
        if self.waveform == WaveformKind.OFDM and not self.ofdm_design.cp_lengths and self.ofdm_cp >= self.ofdm_fft:
            raise ValueError("ofdm_cp must be smaller than ofdm_fft")
        if self.waveform in RADAR_WAVEFORMS:
            pulsed = self.waveform not in {WaveformKind.CW, WaveformKind.FMCW, WaveformKind.PMCW, WaveformKind.ARBITRARY_RADAR}
            if pulsed and self.pulse_width_s > self.pri_s:
                raise ValueError("pulse_width_s must be <= pri_s for pulsed radar waveforms")
            if self.waveform in {WaveformKind.LFM, WaveformKind.NLFM, WaveformKind.FMCW, WaveformKind.COSTAS} and self.sweep_bw_hz >= self.sample_rate:
                raise ValueError(f"sweep bandwidth {eng(self.sweep_bw_hz)} must be below the sample rate ({eng(self.sample_rate, 'S/s')})")
            if self.waveform == WaveformKind.P2 and self.polyphase_order % 2:
                raise ValueError("P2 requires an even polyphase_order")
            if self.waveform == WaveformKind.ZADOFF_CHU:
                import math
                if math.gcd(self.zc_root, self.phase_code_chips) != 1:
                    raise ValueError("zc_root must be relatively prime to phase_code_chips")
            if self.waveform == WaveformKind.ARBITRARY_RADAR:
                if not self.arbitrary_duration_us or any(d <= 0 for d in self.arbitrary_duration_us):
                    raise ValueError("arbitrary radar segment durations must be positive")
                freqs = self.arbitrary_start_freq_hz + self.arbitrary_stop_freq_hz
                if freqs and max(abs(float(f)) for f in freqs) >= nyq:
                    raise ValueError(f"arbitrary radar segment frequencies must lie inside {band}")
        if self.waveform == WaveformKind.OFDM:
            from .ofdm import layout
            layout(self.ofdm_fft,self.ofdm_cp,self.ofdm_occupied,self.ofdm_design)
        return self


def carrier_extent(cfg: TxConfig, sweeps: bool = True) -> tuple[float, float]:
    """Lowest and highest carrier offset (Hz) the configuration visits.

    Carriers are the signal's frequency offset plus the tones of a multitone,
    the end points of a chirp, the peak instantaneous frequency of FSK, FM and
    PM, every component of a composite and the full hop/sweep range: a carrier
    beyond Nyquist would alias, so validation rejects it.  Modulation
    bandwidth is not included; see ``bands.nominal_band`` for that estimate
    (its excess only produces a warning).

    ``sweeps=False`` leaves out swept frequencies (chirps and agility sweeps):
    validation lets a sweep end exactly at ±fs/2 (a full-band chirp), while a
    static carrier there is ambiguous and must lie strictly inside.
    """
    if cfg.waveform == WaveformKind.COMPOSITE:
        spans = [carrier_extent(c, sweeps) for c in cfg.components if c.enabled] or [(0.0, 0.0)]
        lo, hi = min(s[0] for s in spans), max(s[1] for s in spans)
    elif cfg.waveform == WaveformKind.MULTITONE and cfg.tones_hz:
        lo, hi = min(cfg.tones_hz), max(cfg.tones_hz)
    elif cfg.waveform == WaveformKind.CHIRP:
        lo, hi = (min(cfg.chirp_start_hz, cfg.chirp_stop_hz), max(cfg.chirp_start_hz, cfg.chirp_stop_hz)) if sweeps else (0.0, 0.0)
    elif cfg.waveform in (WaveformKind.FSK2, WaveformKind.FSK4):
        lo, hi = -cfg.fsk_deviation_hz, cfg.fsk_deviation_hz
    elif cfg.waveform == WaveformKind.FM:
        lo, hi = -cfg.fm_deviation_hz, cfg.fm_deviation_hz
    elif cfg.waveform == WaveformKind.PM:
        lo, hi = -cfg.pm_deviation_rad * cfg.message_freq_hz, cfg.pm_deviation_rad * cfg.message_freq_hz
    else:
        lo = hi = 0.0
    a, b = frequency_extent(cfg.agility) if sweeps or cfg.agility.mode == "hop" else (0.0, 0.0)
    return cfg.center_hz + lo + a, cfg.center_hz + hi + b


def beyond_nyquist(cfg: TxConfig, offset: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> bool:
    """Whether a carrier of ``cfg`` (moved by ``offset`` and an outer hop/sweep
    ``shift``) leaves the sampled band: sweeps may touch ±fs/2, static carriers
    must stay strictly inside."""
    nyq = cfg.sample_rate / 2.0
    lo, hi = carrier_extent(cfg)
    slo, shi = carrier_extent(cfg, sweeps=False)
    return (max(abs(offset + lo + shift[0]), abs(offset + hi + shift[1])) > nyq
            or max(abs(offset + slo + shift[0]), abs(offset + shi + shift[1])) >= nyq)


TxConfig.model_rebuild()


class ChannelBand(BaseModel):
    """An integration band for channel power, density and occupied bandwidth."""
    model_config = ConfigDict(allow_inf_nan=False)
    label: str = Field(default="CH", max_length=32)
    center_hz: float = 0.0
    width_hz: float = Field(default=100_000.0, gt=0.0)


class SpectrumConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    # Analyzer tuning in complex-baseband Hz.  Spans narrower than a quarter of
    # the sample rate are digitally down-converted (zoom FFT), so the finest
    # RBW is set by the capture length rather than the FFT-size limit.
    center_hz: float = 0.0
    span_hz: float | None = Field(default=None, gt=0, le=MAX_SPECTRUM_SPAN)
    rbw_hz: float = Field(default=2_000.0, gt=0.1)
    # Blackman-Harris (4-term, -92 dB sidelobes) approximates an analyzer's
    # Gaussian RBW filter: a tone shows one clean peak above the noise floor.
    window: Literal["hann", "blackmanharris", "flattop", "kaiser", "rect"] = "blackmanharris"
    detector: Literal["average", "peak", "sample"] = "average"
    averaging: int = Field(default=4, ge=1, le=128)
    max_fft: int = Field(default=131_072, ge=256, le=1_048_576)
    channels: list[ChannelBand] = Field(default_factory=list, max_length=8)


class ChannelConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    # Disable to pass recordings (or clean transmitter captures) through untouched.
    enabled: bool = True
    snr_db: float | None = Field(default=20.0, ge=-30.0, le=100.0)
    freq_offset_hz: float = 0.0
    phase_offset_deg: float = 0.0
    timing_offset_samples: float = Field(default=0.0, ge=-0.95, le=0.95)
    iq_gain_db: float = Field(default=0.0, ge=-12.0, le=12.0)
    iq_phase_deg: float = Field(default=0.0, ge=-45.0, le=45.0)
    dc_i: float = Field(default=0.0, ge=-1.0, le=1.0)
    dc_q: float = Field(default=0.0, ge=-1.0, le=1.0)
    echo_gain: float = Field(default=0, ge=0, le=0.9)
    echo_delay_symbols: int = Field(default=1, ge=1, le=8)


class CostasConfig(BaseModel):
    """Second-order carrier loop gains for unit-RMS symbols.

    The receiver normalizes symbols before its loops, so these gains set the
    same loop bandwidth at any signal level or samples per symbol.  Defaults
    give ζ ≈ 0.75 and ωn ≈ 0.038 rad/symbol for a QPSK Costas detector.
    """
    order: Literal[2, 4] = 4
    alpha: float = Field(default=0.04, gt=0.0, le=1.0)
    beta: float = Field(default=0.001, gt=0.0, le=0.2)


class RxConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    modulation: Literal[
        "bpsk", "qpsk", "8psk", "16qam", "64qam", "2fsk", "4fsk", "msk",
        "ofdm", "am", "fm", "pm", "ssb", "dsb-sc", "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk", "16apsk", "32apsk", "gmsk",
        "tone", "multitone", "awgn", "chirp", "custom"
    ] = "qpsk"
    sample_rate: float = Field(default=1_000_000.0, gt=1_000.0, le=MAX_SAMPLE_RATE)
    symbol_rate: float = Field(default=50_000.0, gt=10.0)
    rolloff: float = Field(default=0.35, ge=0.0, le=1.0)
    timing_phase: float = Field(default=0.0, ge=-0.5, le=0.5)
    timing_recovery: Literal["manual", "mueller-muller", "gardner"] = "manual"
    # Linear demodulation analyses at most this many symbols from the start of each capture.
    max_symbols: int = Field(default=8192, ge=64, le=1_000_000)
    # Timing loop gain in symbols per unit detector output (independent of oversampling).
    timing_gain: float = Field(default=0.015, gt=0.0, le=0.25)
    tuner_hz: float = 0
    manual_phase_deg: float = Field(default=0.0, ge=-180.0, le=180.0)
    fsk_deviation_hz: float = Field(default=25_000.0, gt=1.0)
    fm_deviation_hz: float = Field(default=25_000.0, gt=0.0)
    costas: CostasConfig = Field(default_factory=CostasConfig)
    use_costas: bool = True
    coarse_cfo: bool = False
    carrier_algorithm: Literal["costas", "decision-directed", "off"] = "costas"
    equalizer: Literal["off", "cma"] = "off"
    equalizer_gain: float = Field(default=0.002, gt=0, le=0.03)
    equalizer_taps: int = Field(default=7, ge=3, le=21)
    filter_span_symbols: int = Field(default=10, ge=2, le=64)
    pulse_shape: Literal["rrc", "rect"] = "rrc"
    apsk_ratio2: float = Field(default=2.85, gt=1, le=8)
    apsk_ratio3: float = Field(default=5.27, gt=1, le=12)
    ofdm_timing_offset:int=Field(default=0,ge=0,le=8192)
    ofdm_start_s: float = Field(default=0,ge=0)
    ofdm_design: OFDMDesign = Field(default_factory=OFDMDesign)
    ofdm_fft: int = Field(default=64, ge=16, le=4096)
    ofdm_cp: int = Field(default=16, ge=0, le=1024)
    ofdm_occupied: int = Field(default=52, ge=4, le=4094)

    # Generic frame acquisition / parsing.  These settings are independent of
    # the carrier/timing loops so the user can inspect acquisition explicitly.
    frame_enabled: bool = False
    frame_blocks: list[FrameBlock] = Field(default_factory=list, max_length=64)
    energy_detection: bool = True
    energy_threshold_dbm: float = -80.0
    energy_window_symbols: float = Field(default=1.0, gt=0.05, le=256.0)
    preamble_correlation: bool = True
    preamble_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    # Synchronized de-hopping: a cooperative receiver that knows the
    # transmitter's hop/sweep schedules removes them before demodulation
    # (outermost schedule first when receiving a hopping composite component).
    dehop: list[FrequencyAgility] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _validate_receiver(self):
        linear={"bpsk","qpsk","dbpsk","dqpsk","pi4-dqpsk","oqpsk","8psk","16qam","64qam","16apsk","32apsk"}
        if self.modulation in linear and not 1.5 < self.sample_rate/self.symbol_rate <= MAX_SAMPLES_PER_SYMBOL:
            raise ValueError(f"The receiver needs 1.5 to {MAX_SAMPLES_PER_SYMBOL:,} samples per symbol: "
                             f"{self.symbol_rate:,.0f} Bd at {self.sample_rate:,.0f} S/s is {self.sample_rate/self.symbol_rate:,.1f}")
        if self.modulation == "32apsk" and self.apsk_ratio3 <= self.apsk_ratio2:
            raise ValueError("APSK outer radius must exceed the middle radius")
        if self.modulation == "ofdm" and not self.ofdm_design.enabled and (self.ofdm_cp >= self.ofdm_fft or self.ofdm_occupied >= self.ofdm_fft):
            raise ValueError("OFDM prefix and occupied carrier count must be smaller than FFT length")
        if self.modulation == "ofdm":
            from .ofdm import layout
            layout(self.ofdm_fft,self.ofdm_cp,self.ofdm_occupied,self.ofdm_design)
        if self.frame_enabled:
            for block in self.frame_blocks:
                rate = block.symbol_rate or self.symbol_rate
                if not 1.5 < self.sample_rate / rate <= MAX_SAMPLES_PER_SYMBOL:
                    raise ValueError(f"Frame block {block.name!r} has an invalid symbol rate for this sample rate")
        return self


class BlindResult(BaseModel):
    modulation: str
    family: str
    confidence: float
    bandwidth_3db_hz: float
    bandwidth_99_hz: float
    symbol_rate_baud: float | None
    snr_db: float | None
    notes: list[str] = Field(default_factory=list)
    candidates: list[dict] = Field(default_factory=list)
    features: dict = Field(default_factory=dict)
    score_kind: str = "heuristic agreement; not a calibrated probability"
    # Every rule evaluated, in order: question, feature checks against their
    # thresholds and outcome.  The accepted stage produced ``modulation``.
    trace: list[dict] = Field(default_factory=list)
