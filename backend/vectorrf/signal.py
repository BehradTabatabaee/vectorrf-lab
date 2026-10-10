"""Transmitter signal synthesis.

The transmitter output is one stream x[i], i = 0, 1, 2, … from switch-on at
sample 0, and a deterministic function of the design and the absolute sample
index.  ``generate(cfg, start_sample=a)`` returns the slice x[a : a + N] with
N = ``capture_samples(cfg)``, so consecutive captures join exactly and a
capture of any length at any position is the same slice of the same stream:

* data (text, bit patterns, PRBS, simple frames) is one periodic bit stream
  indexed by symbol number; pulse-shaping filters and differential encoders
  carry the state of every earlier symbol;
* FSK/MSK/GMSK phase is the exact integral of the instantaneous frequency
  since switch-on (GMSK includes the Gaussian filter's memory);
* pulse trains, sweeps, bursts and hopping follow absolute time, and a pulse
  cut by a capture edge is the start (or end) of the full pulse;
* noise is a function of the absolute sample index (``dsp.stream_noise``).

Power: ``power_dbm`` is the signal's mean power while it is on, the level
setting of a pulse-modulated generator.  The scale that maps the clean
waveform onto it (``reference_power``) is a property of the design, never of
one capture, so a capture's measured power varies with its content as a real
signal's does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import math
import numpy as np
from scipy import signal

from .dsp import bytes_to_bits, noise_key, phase_cycles, phasor, rrc_taps, shape_pulses, stream_noise
from .models import TxConfig, WaveformKind, RADAR_WAVEFORMS
from .modem import map_bits, bits_per_symbol

# Independent noise streams of one design.
_NOISE_IMPAIRMENT, _NOISE_SOURCE = 1, 2


@dataclass
class GeneratedSignal:
    samples: np.ndarray
    sample_rate: float
    metadata: dict = field(default_factory=dict)
    tx_bits: np.ndarray | None = None
    tx_symbols: np.ndarray | None = None


def capture_samples(cfg: TxConfig) -> int:
    """Number of samples in one capture of ``cfg`` (the record length)."""
    return max(32, int(round(cfg.duration * cfg.sample_rate)))


def component_seed(parent_seed: int, index: int, own_seed: int) -> int:
    """Distinct, reproducible seed for composite component ``index``.

    Two otherwise identical components must not carry identical noise or PN
    data (they would add coherently).
    """
    return (int(parent_seed) * 1_000_003 + 7_919 * (index + 1) + int(own_seed)) % (2 ** 32)


def _sps(cfg: TxConfig, rate: float | None = None) -> int:
    return max(2, int(round(cfg.sample_rate / float(rate or cfg.symbol_rate))))


def _take(segment: np.ndarray, origin: int, start: int, n: int) -> np.ndarray:
    """``n`` samples of a segment whose first sample is absolute ``origin``, from ``start`` (zeros outside)."""
    out = np.zeros(n, dtype=np.result_type(segment.dtype, np.complex64))
    a, b = max(start, origin), min(start + n, origin + len(segment))
    if b > a:
        out[a - start:b - start] = segment[a - origin:b - origin]
    return out


# ------------------------------------------------------------------ data bits

@lru_cache(maxsize=64)
def _source_pattern(data_source: str, message: str, custom_bits: str, pn_order: int, pn_bits: int, seed: int) -> np.ndarray:
    if data_source == "bits":
        return np.array([int(b) for b in custom_bits], dtype=np.uint8)
    if data_source == "text" and message:
        return bytes_to_bits(message)
    # Maximal-length PRBS of the chosen order from a reproducible nonzero state.
    state = np.random.default_rng(seed).integers(0, 2, pn_order, dtype=np.int8)
    if not np.any(state):
        state[0] = 1
    pattern, _ = signal.max_len_seq(pn_order, state=state, length=min(pn_bits, (1 << pn_order) - 1))
    return pattern.astype(np.uint8)


def source_pattern(cfg: TxConfig) -> np.ndarray:
    """One period of the payload source bits (text, bit pattern or PRBS)."""
    return _source_pattern(cfg.data_source, cfg.message, cfg.custom_bits, cfg.pn_order, cfg.pn_bits, int(cfg.seed))


@lru_cache(maxsize=64)
def _framed_pattern(payload: bytes, preamble: str, payload_bits: int, crc: bool) -> np.ndarray:
    source = np.frombuffer(payload, dtype=np.uint8)
    body = source[np.arange(payload_bits) % len(source)]
    parts = [np.array([int(b) for b in preamble], dtype=np.uint8), body]
    if crc:
        import binascii
        value = binascii.crc_hqx(np.packbits(body).tobytes(), 0xFFFF)
        parts.append(np.unpackbits(np.frombuffer(value.to_bytes(2, "big"), dtype=np.uint8)))
    return np.concatenate(parts).astype(np.uint8)


def data_pattern(cfg: TxConfig) -> np.ndarray:
    """One period of the transmitted bit stream: the payload source, or
    preamble + payload (+ CRC-16) frames in the simple frame mode."""
    source = source_pattern(cfg)
    if cfg.frame_enabled and cfg.frame_mode == "simple":
        return _framed_pattern(source.tobytes(), cfg.preamble_bits, cfg.payload_bits, cfg.frame_crc)
    return source


def data_bits(cfg: TxConfig, start_bit: int, count: int) -> np.ndarray:
    """Bits ``start_bit … start_bit + count − 1`` of the periodic data stream."""
    pattern = data_pattern(cfg)
    return pattern[(np.arange(int(count), dtype=np.int64) + int(start_bit)) % len(pattern)]


def _source_bits(cfg: TxConfig, needed: int, rng: np.random.Generator | None = None, start_bit: int = 0) -> np.ndarray:
    """Payload-source bits without framing (``rng`` is unused: the PRBS state follows the design seed)."""
    pattern = source_pattern(cfg)
    return pattern[(np.arange(int(needed), dtype=np.int64) + int(start_bit)) % len(pattern)]


def _data_bits(cfg: TxConfig, needed: int, rng: np.random.Generator | None = None, start_bit: int = 0) -> np.ndarray:
    return data_bits(cfg, start_bit, needed)


def _symbol_period(cfg: TxConfig, k: int) -> int:
    """Symbols after which the symbol sequence repeats (k bits per symbol)."""
    p = len(data_pattern(cfg))
    return p // math.gcd(p, k)


# --------------------------------------------------------- linear modulations

_BASE = {"dbpsk": "bpsk", "dqpsk": "qpsk", "pi4-dqpsk": "qpsk", "oqpsk": "qpsk"}
_DIFFERENTIAL = {"dbpsk", "dqpsk", "pi4-dqpsk"}
LINEAR = {"bpsk", "qpsk", "8psk", "16qam", "64qam", "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk", "16apsk", "32apsk"}


def _phase_steps(symbols: np.ndarray, mod: str) -> np.ndarray:
    """Differential phase increments in units of π/4 (DQPSK uses the π/4-rotated QPSK points)."""
    z = symbols * np.exp(-1j * np.pi / 4) if mod == "dqpsk" else symbols
    return np.mod(np.rint(np.angle(z) / (np.pi / 4)).astype(np.int64), 8)


def _differential_offset(cfg: TxConfig, mod: str, symbol: int) -> int:
    """Accumulated phase (π/4 units, mod 8) of all symbols before ``symbol``."""
    k = bits_per_symbol(_BASE[mod])
    period = _symbol_period(cfg, k)
    steps = _phase_steps(map_bits(data_bits(cfg, 0, period * k), _BASE[mod], cfg.apsk_ratio2, cfg.apsk_ratio3), mod)
    full, rest = divmod(int(symbol), period)
    return int((full * int(steps.sum()) + int(steps[:rest].sum())) % 8)


def _linear_taps(cfg: TxConfig, sps: int) -> np.ndarray:
    return np.ones(sps, dtype=np.float32) if cfg.pulse_shape == "rect" else rrc_taps(sps, cfg.rolloff, cfg.filter_span_symbols)


def _linear_symbols(cfg: TxConfig, mod: str, first: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Bits and complex symbols ``first … first + count − 1`` of the stream."""
    base = _BASE.get(mod, mod)
    k = bits_per_symbol(base)
    bits = data_bits(cfg, first * k, count * k)
    symbols = map_bits(bits, base, cfg.apsk_ratio2, cfg.apsk_ratio3)
    if mod in _DIFFERENTIAL:
        index = (_differential_offset(cfg, mod, first) + np.cumsum(_phase_steps(symbols, mod))) % 8
        symbols = np.exp(1j * np.pi / 4 * index).astype(np.complex64)
    return bits, symbols


def _linear(cfg: TxConfig, a: int, n: int, mod: str) -> GeneratedSignal:
    """Samples ``a … a + n − 1`` of a pulse-shaped linear modulation.

    Symbol m's pulse h starts at sample m·sps (y[i] = Σ s_m h[i − m·sps]); the
    slice includes every earlier symbol whose pulse still reaches it.
    """
    sps = _sps(cfg)
    taps = _linear_taps(cfg, sps)
    span = len(taps)
    guard = sps if mod == "oqpsk" else 0          # OQPSK delays Q by half a symbol
    lo = a - guard
    first = max(0, -(-(lo - (span - 1)) // sps))  # first symbol whose pulse reaches sample lo
    last = max(first, (a + n - 1) // sps)
    bits, symbols = _linear_symbols(cfg, mod, first, last - first + 1)
    if mod == "oqpsk":
        ext = shape_pulses(symbols, first, taps, sps, lo, n + guard)
        positions = np.arange(lo, a + n, dtype=np.float64)
        q = np.interp(positions - sps / 2, positions, ext.imag, left=0.0)
        y = (ext.real + 1j * q)[guard:]
    else:
        y = shape_pulses(symbols, first, taps, sps, a, n)
    # Symbols whose pulse centre lies in this capture (from the first one on).
    centre = (span - 1) // 2 if cfg.pulse_shape == "rrc" else 0
    own = max(first, -(-(a - centre) // sps))
    k = bits_per_symbol(_BASE.get(mod, mod))
    return GeneratedSignal(
        samples=y, sample_rate=cfg.sample_rate,
        metadata={"modulation": mod, "samples_per_symbol": sps, "symbol_rate": cfg.sample_rate / sps, "rrc_taps": int(span),
                  "pulse_shape": cfg.pulse_shape, "data_source": cfg.data_source, "first_symbol": int(own)},
        tx_bits=bits[(own - first) * k:], tx_symbols=symbols[own - first:],
    )


def _linear_power(cfg: TxConfig, mod: str) -> float:
    """Long-run mean power: E|s|² over the symbol period × Σh² / sps (exact for
    rectangular pulses, and for RRC up to the truncated tails at nonzero symbol lags)."""
    sps = _sps(cfg)
    k = bits_per_symbol(_BASE.get(mod, mod))
    period = min(_symbol_period(cfg, k), 1 << 20)
    _, symbols = _linear_symbols(cfg, mod, 0, period)
    taps = _linear_taps(cfg, sps).astype(np.float64)
    return float(np.mean(np.abs(symbols.astype(np.complex128)) ** 2) * np.sum(taps * taps) / sps)


# ------------------------------------------------------- CPFSK, MSK and GMSK

def _fsk_levels(cfg: TxConfig) -> tuple[int, bool]:
    w = cfg.waveform
    return (4, False) if w == WaveformKind.FSK4 else (2, w in {WaveformKind.MSK, WaveformKind.GMSK})


def _fsk_frequencies(cfg: TxConfig, first: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Bits and per-symbol frequencies (Hz) of symbols ``first … first + count − 1``."""
    levels, msk = _fsk_levels(cfg)
    k = 2 if levels == 4 else 1
    bits = data_bits(cfg, first * k, count * k)
    idx = bits.reshape(-1, k) @ (1 << np.arange(k - 1, -1, -1))
    if msk:
        rate = cfg.sample_rate / _sps(cfg)
        return bits, np.where(idx > 0, rate / 4.0, -rate / 4.0)
    spacing = 2.0 * cfg.fsk_deviation_hz / max(1, levels - 1)
    return bits, (idx.astype(np.float64) - (levels - 1) / 2.0) * spacing


def _frequency_cycles(cfg: TxConfig, end: int) -> float:
    """Phase in cycles (mod 1) accumulated by samples 0 … end − 1 (inclusive cumulative sum)."""
    if end <= 0:
        return 0.0
    sps, fs = _sps(cfg), cfg.sample_rate
    levels, _ = _fsk_levels(cfg)
    period = _symbol_period(cfg, 2 if levels == 4 else 1)
    _, freqs = _fsk_frequencies(cfg, 0, period)
    cycles = freqs * (sps / fs)                       # cycles per whole symbol
    prefix = np.concatenate([[0.0], np.cumsum(cycles)])
    symbols, rest = divmod(int(end), sps)
    full, part = divmod(symbols, period)
    total = math.fmod(full * math.fmod(float(prefix[-1]), 1.0), 1.0) + float(prefix[part])
    if rest:
        total += rest * float(freqs[part]) / fs
    return math.fmod(total, 1.0)


def _gaussian_kernel(cfg: TxConfig, sps: int) -> np.ndarray:
    """GMSK frequency-pulse kernel (truncated at ±4σ, unit sum)."""
    sigma = math.sqrt(math.log(2)) * sps / (2 * math.pi * cfg.gaussian_bt)
    half = max(1, int(math.ceil(4 * sigma)))
    u = np.arange(-half, half + 1, dtype=np.float64)
    g = np.exp(-0.5 * (u / sigma) ** 2)
    return g / g.sum()


def _cpfsk(cfg: TxConfig, a: int, n: int) -> GeneratedSignal:
    """Continuous-phase FSK: φ[i] = 2π/fs · Σ_{j≤i} f[j], with f the (Gaussian-filtered for GMSK) frequency."""
    fs = cfg.sample_rate
    sps = _sps(cfg)
    levels, msk = _fsk_levels(cfg)
    gmsk = cfg.waveform == WaveformKind.GMSK
    kernel = _gaussian_kernel(cfg, sps) if gmsk else np.ones(1)
    half = len(kernel) // 2
    lo, hi = a - 2 * half, a + n + half                 # unfiltered frequency needed
    first, last = max(0, lo // sps), max(0, (hi - 1) // sps)
    bits, freqs = _fsk_frequencies(cfg, first, last - first + 1)
    index = np.arange(lo, hi, dtype=np.int64)
    f = np.where(index >= 0, freqs[np.clip(index // sps - first, 0, len(freqs) - 1)], 0.0)
    if gmsk:
        # Σ_{t ≤ a−h−1} (g∗f)[t] = Σ_j f[j]·G(a−h−1−j), G the kernel's cumulative sum:
        # whole symbols far enough back contribute fully, the last 2h samples partially.
        cumulative = np.cumsum(kernel)                   # G(d) for d = −h … h
        before = f[:2 * half]                            # samples a−2h … a−1
        d = (a - half - 1) - np.arange(a - 2 * half, a)  # h−1 … −h
        start_cycles = _frequency_cycles(cfg, a - 2 * half) + float(np.sum(before * cumulative[d + half])) / fs
        filtered = np.convolve(f, kernel, mode="valid")  # samples a−h … a+n−1
        cycles = start_cycles + np.cumsum(filtered) / fs
        cycles = cycles[half:]
    else:
        cycles = _frequency_cycles(cfg, a) + np.cumsum(f) / fs
    y = np.exp(2j * np.pi * np.mod(cycles, 1.0))
    k = 2 if levels == 4 else 1
    own = max(first, -(-a // sps))
    return GeneratedSignal(
        y, fs, {"modulation": "msk" if msk else f"{levels}fsk", "samples_per_symbol": sps, "symbol_rate": fs / sps,
                "first_symbol": int(own)},
        tx_bits=bits[(own - first) * k:],
    )


# ------------------------------------------------------------- frame blocks

@lru_cache(maxsize=8)
def _frame(config_json: str):
    from .frame import frame_period
    cfg = TxConfig.model_validate_json(config_json)
    return frame_period(cfg, np.random.default_rng(cfg.seed), lambda count, offset: _source_bits(cfg, count, None, offset))


def _frame_blocks(cfg: TxConfig, a: int, n: int) -> GeneratedSignal:
    frame, bits, meta = _frame(_waveform_json(cfg))
    y = frame[(np.arange(n, dtype=np.int64) + a) % len(frame)].astype(np.complex128)
    meta = {**meta, "waveform": "mixed-frame", "configured_waveform": cfg.waveform.value}
    return GeneratedSignal(y, cfg.sample_rate, meta, tx_bits=bits)


# ------------------------------------------------------------ OFDM, analog

def _ofdm(cfg: TxConfig, a: int, n: int) -> GeneratedSignal:
    from .ofdm import synthesize
    y, bits, metadata = synthesize(cfg, n, lambda count, offset: data_bits(cfg, offset, count), a / cfg.sample_rate)
    return GeneratedSignal(np.asarray(y, dtype=np.complex128), cfg.sample_rate, metadata, tx_bits=bits)


def _analog(cfg: TxConfig, n: int, start_sample: int = 0) -> np.ndarray:
    """Analog modulation of the sinusoidal message m(t) = sin(2π fm t).

    The message phase uses the absolute sample index, so successive captures
    join continuously.  Because the message is a sinusoid, every modulation has
    a closed form: SSB uses the exact analytic signal -j·exp(j2π fm t) instead
    of a finite-length Hilbert transform (which rings at record edges), and FM
    uses the exact phase integral (Δf/fm)(1 - cos 2π fm t) instead of a
    rectangular-rule cumulative sum.
    """
    cycles = phase_cycles(cfg.message_freq_hz, cfg.sample_rate, n, start_sample)
    theta = 2.0 * np.pi * cycles
    m = np.sin(theta)
    w = cfg.waveform
    if w == WaveformKind.AM:
        return (1.0 + cfg.modulation_index * m).astype(np.complex128)
    if w == WaveformKind.DSB_SC:
        return m.astype(np.complex128)
    if w == WaveformKind.SSB:
        analytic = -1j * np.exp(1j * theta)  # sin(θ) + j·H{sin θ} = sin θ - j cos θ
        return analytic if cfg.sideband == "usb" else np.conj(analytic)
    if w == WaveformKind.FM:
        phase = (cfg.fm_deviation_hz / cfg.message_freq_hz) * (1.0 - np.cos(theta))
        return np.exp(1j*phase)
    if w == WaveformKind.PM:
        return np.exp(1j*cfg.pm_deviation_rad*m)
    raise ValueError(w)


# -------------------------------------------------------------------- radar

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
    # g = frequency-step group (outer, slow index), k = sample within the group
    # (inner, fast index).  Flattening row-major therefore emits group 0, then
    # group 1, ... exactly as the codes are defined (Levanon & Mozeson, ch. 6).
    g, k = np.meshgrid(np.arange(m, dtype=np.float64), np.arange(m, dtype=np.float64), indexing="ij")
    if kind == WaveformKind.FRANK:
        phase = 2.0 * np.pi * g * k / m
    elif kind == WaveformKind.P1:
        # Lewis/Kretschmer P1: phi = -(pi/M)[M-(2g+1)][gM+k] (zero-based g, k).
        # The previous implementation flattened this array transposed, which
        # produced a different sequence ~5 dB worse in peak sidelobe level.
        phase = -np.pi / m * (m - (2.0 * g + 1.0)) * (g * m + k)
    elif kind == WaveformKind.P2:
        phase = -np.pi / (2.0 * m) * (2.0 * k + 1.0 - m) * (2.0 * g + 1.0 - m)
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


def _unit_uniform(seed: int, index: np.ndarray) -> np.ndarray:
    """Counter-based uniforms in [0, 1) for integer ``index`` (pulse jitter)."""
    from .dsp import _splitmix64
    return (_splitmix64(np.asarray(index, dtype=np.uint64), noise_key(seed, 3)) >> np.uint64(11)).astype(np.float64) * 2.0 ** -53


class _PulseTrain:
    """Start sample of pulse p, for any p, in O(1).

    * fixed: p·P with P = round(PRI·fs);
    * staggered: the PRI pattern repeats (its period sum and prefix sums);
    * jittered: pulse p sits at (p + j_p)·PRI with j_p uniform in ±fraction/2,
      so consecutive PRIs deviate by at most ±fraction and the train never
      drifts from its nominal grid.
    """

    def __init__(self, cfg: TxConfig):
        fs = cfg.sample_rate
        self.cfg, self.fs = cfg, fs
        self.pri = max(1, int(round(cfg.pri_s * fs)))
        if cfg.pri_mode == "staggered":
            scales = [max(0.05, float(s)) for s in (cfg.pri_stagger or [1.0])]
            self.steps = np.array([max(1, int(round(cfg.pri_s * s * fs))) for s in scales], dtype=np.int64)
            self.prefix = np.concatenate([[0], np.cumsum(self.steps)])
        self.min_step = int(self.steps.min()) if cfg.pri_mode == "staggered" else max(1, int(self.pri * (1 - cfg.pri_jitter_fraction)) if cfg.pri_mode == "jittered" else self.pri)

    def start(self, p: np.ndarray) -> np.ndarray:
        p = np.asarray(p, dtype=np.int64)
        mode = self.cfg.pri_mode
        if mode == "staggered":
            full, rest = np.divmod(p, len(self.steps))
            return full * int(self.prefix[-1]) + self.prefix[rest]
        if mode == "jittered":
            jitter = (_unit_uniform(self.cfg.seed, p) - 0.5) * self.cfg.pri_jitter_fraction
            return np.maximum(0, np.rint((p + jitter) * self.cfg.pri_s * self.fs)).astype(np.int64)
        return p * self.pri

    def _first(self, predicate, upper: int) -> int:
        lo, hi = 0, upper
        while lo < hi:
            mid = (lo + hi) // 2
            if predicate(int(self.start(mid))):
                hi = mid
            else:
                lo = mid + 1
        return lo

    def overlapping(self, a: int, n: int, width: int) -> np.ndarray:
        """Indices of pulses that overlap samples a … a + n − 1 (start is increasing in p)."""
        upper = (a + n) // self.min_step + 2
        first = self._first(lambda s: s + width > a, upper)    # first pulse ending after a
        stop = self._first(lambda s: s >= a + n, upper + 1)     # first pulse starting at or after the end
        return np.arange(first, max(first, stop), dtype=np.int64)


def _pulse_contents(cfg: TxConfig, length: int, rng: np.random.Generator | None = None) -> tuple[np.ndarray, dict]:
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


@lru_cache(maxsize=16)
def _arbitrary_sequence(config_json: str) -> tuple[np.ndarray, np.ndarray, list]:
    """One period of the segmented radar waveform and its on-mask."""
    cfg = TxConfig.model_validate_json(config_json)
    fs = cfg.sample_rate
    durs = cfg.arbitrary_duration_us or [10.0]
    f0s = cfg.arbitrary_start_freq_hz or [0.0]
    f1s = cfg.arbitrary_stop_freq_hz or f0s
    amps = cfg.arbitrary_amplitude or [1.0]
    phs = cfg.arbitrary_phase_deg or [0.0]
    pieces, masks, segment_meta = [], [], []
    phase_acc = 0.0
    count = max(len(durs), len(f0s), len(f1s), len(amps), len(phs))
    for kseg in range(count):
        dur_us = float(durs[kseg % len(durs)])
        m = max(1, int(round(dur_us * 1e-6 * fs)))
        f0 = float(f0s[kseg % len(f0s)])
        f1 = float(f1s[kseg % len(f1s)])
        amp = max(0.0, float(amps[kseg % len(amps)]))
        jump = np.deg2rad(float(phs[kseg % len(phs)]))
        freq = np.linspace(f0, f1, m, endpoint=False, dtype=np.float64)
        phase = phase_acc + jump + 2.0 * np.pi * np.cumsum(freq) / fs
        pieces.append(amp * np.exp(1j * phase))
        masks.append(np.full(m, amp > 0))
        phase_acc = float(phase[-1])
        segment_meta.append({"duration_us": dur_us, "start_hz": f0, "stop_hz": f1, "amplitude": amp, "phase_jump_deg": float(phs[kseg % len(phs)])})
    return np.concatenate(pieces), np.concatenate(masks), segment_meta


@lru_cache(maxsize=8)
def _pmcw_code(chips: int, seed: int) -> np.ndarray:
    # A reproducible BPSK PN sequence; arbitrary/custom PMCW codes are represented by CUSTOM_PHASE pulses.
    return np.random.default_rng(seed).choice(np.asarray([-1.0, 1.0]), size=chips).astype(np.complex128)


def _fmcw_frequency(cfg: TxConfig, tau: np.ndarray, T: float) -> np.ndarray:
    if cfg.fmcw_shape == "triangle":
        x = tau / T
        return np.where(x < 0.5, -cfg.sweep_bw_hz / 2.0 + 2.0 * cfg.sweep_bw_hz * x, cfg.sweep_bw_hz / 2.0 - 2.0 * cfg.sweep_bw_hz * (x - 0.5))
    direction = 1.0 if cfg.fmcw_shape == "up" else -1.0
    return direction * (-cfg.sweep_bw_hz / 2.0 + cfg.sweep_bw_hz * tau / T)


def _radar(cfg: TxConfig, a: int, n: int) -> GeneratedSignal:
    fs = cfg.sample_rate
    w = cfg.waveform
    index = np.arange(n, dtype=np.int64) + a

    if w == WaveformKind.CW:
        return GeneratedSignal(np.ones(n, dtype=np.complex128), fs, {"waveform": w.value, "radar_family": "cw"})

    if w == WaveformKind.ARBITRARY_RADAR:
        seq, _, segment_meta = _arbitrary_sequence(_waveform_json(cfg))
        return GeneratedSignal(seq[index % len(seq)], fs, {
            "waveform": w.value, "radar_family": "arbitrary", "segment_count": len(segment_meta),
            "sequence_duration_s": len(seq) / fs, "segments": segment_meta,
        })

    if w == WaveformKind.FMCW:
        # Sweeps repeat every P = round(T·fs) samples with a continuous phase:
        # sweep r starts with the phase of r whole sweeps.
        period = max(1, int(round(cfg.pri_s * fs)))
        T = period / fs
        sweep_f = _fmcw_frequency(cfg, np.arange(period) / fs, T)
        prefix = np.concatenate([[0.0], np.cumsum(sweep_f) / fs])        # cycles after j samples of a sweep
        r, j = np.divmod(index, period)
        per_sweep = math.fmod(float(prefix[-1]), 1.0)
        cycles = np.mod(np.mod(r.astype(np.float64) * per_sweep, 1.0) + prefix[j + 1], 1.0)
        return GeneratedSignal(np.exp(2j * np.pi * cycles), fs, {
            "waveform": w.value, "radar_family": "fmcw", "sweep_period_s": T,
            "sweep_bw_hz": cfg.sweep_bw_hz, "fmcw_shape": cfg.fmcw_shape,
        })

    if w == WaveformKind.PMCW:
        chips = max(2, cfg.phase_code_chips)
        chip_sps = max(1, int(round(fs / cfg.pmcw_chip_rate)))
        code = _pmcw_code(chips, int(cfg.seed))
        y = code[(index // chip_sps) % chips]
        return GeneratedSignal(y, fs, {
            "waveform": w.value, "radar_family": "pmcw", "code_length": chips,
            "chip_rate_hz": fs / chip_sps, "chip_sps": chip_sps,
            "code_phase_deg": np.rad2deg(np.angle(code)).astype(float).tolist(),
        })

    width = max(1, int(round(cfg.pulse_width_s * fs)))
    pulse, pulse_meta = _pulse_contents(cfg, width)
    train = _PulseTrain(cfg)
    pulses = train.overlapping(a, n, width)
    y = np.zeros(n, dtype=np.complex128)
    for start in train.start(pulses):
        lo, hi = max(a, int(start)), min(a + n, int(start) + width)
        y[lo - a:hi - a] = pulse[lo - int(start):hi - int(start)]
    meta = {
        "waveform": w.value,
        "radar_family": "pulsed",
        "pri_s": cfg.pri_s,
        "prf_hz": 1.0 / cfg.pri_s,
        "pulse_width_s": cfg.pulse_width_s,
        "duty_cycle": min(1.0, cfg.pulse_width_s / cfg.pri_s),
        "pri_mode": cfg.pri_mode,
        "pulse_envelope": cfg.pulse_envelope,
        "pulse_count": int(len(pulses)),
    }
    meta.update(pulse_meta)
    return GeneratedSignal(y, fs, meta)


# ------------------------------------------------------------- the stream

def chirp_period_samples(cfg: TxConfig) -> int:
    """Samples per sweep of the basic chirp (its period, or the capture length when unset)."""
    return max(1, int(round(float(cfg.chirp_period_s or cfg.duration) * cfg.sample_rate)))


def capture_prefix(cfg: TxConfig, n: int) -> TxConfig:
    """A design whose captures are the first ``n`` samples of ``cfg``'s captures.

    Only the capture length changes: a chirp without a period of its own (in
    the design or a component) keeps sweeping over ``cfg``'s capture.
    """
    def pin(c: TxConfig) -> TxConfig:
        return c if c.chirp_period_s is not None else c.model_copy(update={"chirp_period_s": cfg.duration})
    return pin(cfg).model_copy(update={"duration": n / cfg.sample_rate, "components": [pin(c) for c in cfg.components]})


# Fields that change the clean waveform's shape (not its scale, offset, gating or impairments).
_SCALE_FIELDS = {"power_dbm", "amplitude", "normalize_output", "snr_db", "center_hz", "phase_offset_deg", "iq_gain_db",
                 "iq_phase_deg", "dc_i", "dc_q", "burst_enabled", "burst_period_s", "burst_duty", "agility", "label",
                 "enabled", "rf_center_hz", "components"}


def _waveform_json(cfg: TxConfig) -> str:
    """The design without fields that do not shape the clean waveform (a cache key)."""
    return cfg.model_dump_json(exclude=_SCALE_FIELDS)


def _clean(cfg: TxConfig, a: int, n: int, frame: int = 0) -> GeneratedSignal:
    """Samples ``a … a + n − 1`` of the clean, unscaled waveform (before offset, power, agility and impairments)."""
    fs = cfg.sample_rate
    w = cfg.waveform
    if cfg.frame_enabled and cfg.frame_mode == "blocks":
        return _frame_blocks(cfg, a, n)
    if w == WaveformKind.CUSTOM:
        from .programs import program_samples
        return GeneratedSignal(program_samples(cfg.program, fs, n, a / fs, frame, cfg.seed, start_sample=a).astype(np.complex128), fs,
                               {"waveform": "custom", "program_mode": cfg.program.mode})
    if w == WaveformKind.TONE:
        return GeneratedSignal(np.ones(n, dtype=np.complex128), fs, {"waveform": "tone"})
    if w == WaveformKind.MULTITONE:
        y = np.zeros(n, dtype=np.complex128)
        for f in cfg.tones_hz:
            y += phasor(f, fs, n, a)
        return GeneratedSignal(y, fs, {"waveform": "multitone", "tones_hz": cfg.tones_hz})
    if w == WaveformKind.AWGN:
        key = noise_key(cfg.seed, _NOISE_SOURCE)
        if cfg.noise_bandwidth_hz:
            # Band-limit with a linear-phase FIR fed by the noise of the preceding samples.
            h = _noise_filter(cfg)
            y = signal.oaconvolve(stream_noise(key, a - (len(h) - 1), n + len(h) - 1), h.astype(np.float32), mode="valid")
        else:
            y = stream_noise(key, a, n)
        return GeneratedSignal(np.asarray(y, dtype=np.complex128) * math.sqrt(0.5), fs, {"waveform": "awgn"})
    if w == WaveformKind.CHIRP:
        # Sawtooth: the frequency ramps start → stop over each period and jumps
        # back, with a continuous phase (like a VCO): sweep r starts with the
        # phase of r whole sweeps, so the flyback has no phase step.
        period = chirp_period_samples(cfg)
        T = period / fs
        r, j = np.divmod(np.arange(n, dtype=np.int64) + a, period)
        a1 = cfg.chirp_start_hz / fs                                       # cycles per sample at the start
        a2 = 0.5 * (cfg.chirp_stop_hz - cfg.chirp_start_hz) / (period * fs)  # cycles per sample²
        per_sweep = math.fmod(a1 * period + a2 * period * period, 1.0)
        jf = j.astype(np.float64)
        cycles = np.mod(r.astype(np.float64) * per_sweep, 1.0) + np.mod(a1 * jf, 1.0) + np.mod(a2 * jf * jf, 1.0)
        y = np.exp(2j * np.pi * np.mod(cycles, 1.0))
        return GeneratedSignal(y, fs, {"waveform": "chirp", "start_hz": cfg.chirp_start_hz, "stop_hz": cfg.chirp_stop_hz, "sweep_period_s": T})
    if w.value in LINEAR:
        return _linear(cfg, a, n, w.value)
    if w in {WaveformKind.FSK2, WaveformKind.FSK4, WaveformKind.MSK, WaveformKind.GMSK}:
        return _cpfsk(cfg, a, n)
    if w == WaveformKind.OFDM:
        return _ofdm(cfg, a, n)
    if w in {WaveformKind.AM, WaveformKind.FM, WaveformKind.PM, WaveformKind.SSB, WaveformKind.DSB_SC}:
        return GeneratedSignal(_analog(cfg, n, a), fs, {"waveform": w.value, "message_freq_hz": cfg.message_freq_hz})
    if w in RADAR_WAVEFORMS:
        return _radar(cfg, a, n)
    raise ValueError(f"unsupported waveform: {w}")


def _noise_filter(cfg: TxConfig) -> np.ndarray:
    bw = min(cfg.noise_bandwidth_hz, cfg.sample_rate * .98)
    return signal.firwin(129, bw / 2, fs=cfg.sample_rate)


@lru_cache(maxsize=64)
def _reference_power(config_json: str) -> float:
    cfg = TxConfig.model_validate_json(config_json)
    fs = cfg.sample_rate
    w = cfg.waveform
    if cfg.frame_enabled and cfg.frame_mode == "blocks":
        frame, _, _ = _frame(config_json)
        return float(np.mean(np.abs(frame.astype(np.complex128)) ** 2))
    if w.value in LINEAR:
        return _linear_power(cfg, w.value)
    if w == WaveformKind.MULTITONE:
        # Distinct tones add in power over time; coincident ones add coherently.
        _, counts = np.unique(np.asarray(cfg.tones_hz, dtype=np.float64), return_counts=True)
        return float(np.sum(counts.astype(np.float64) ** 2)) if len(counts) else 1.0
    if w == WaveformKind.AWGN:
        return float(np.sum(_noise_filter(cfg) ** 2)) if cfg.noise_bandwidth_hz else 1.0
    if w == WaveformKind.AM:
        return 1.0 + cfg.modulation_index ** 2 / 2.0
    if w == WaveformKind.DSB_SC:
        return 0.5
    if w in RADAR_WAVEFORMS and w not in {WaveformKind.CW, WaveformKind.FMCW, WaveformKind.PMCW, WaveformKind.ARBITRARY_RADAR}:
        # Pulsed: the mean power during one pulse.
        pulse, _ = _pulse_contents(cfg, max(1, int(round(cfg.pulse_width_s * fs))))
        return float(np.mean(np.abs(pulse) ** 2))
    if w == WaveformKind.ARBITRARY_RADAR:
        seq, on, _ = _arbitrary_sequence(config_json)
        return float(np.mean(np.abs(seq[on]) ** 2)) if on.any() else 1.0
    if w in {WaveformKind.OFDM, WaveformKind.CUSTOM}:
        # Numerically, over a whole program period / several OFDM symbols; while on.
        if w == WaveformKind.CUSTOM:
            length = int(min(1 << 18, max(4096, round(cfg.program.period_s * fs))))
        else:
            length = int(min(1 << 18, max(1 << 14, 8 * (cfg.ofdm_fft + cfg.ofdm_cp))))
        y = _clean(cfg, 0, length).samples
        power = np.abs(y) ** 2
        on = power > 1e-12 * max(float(power.max()), 1e-300)
        return float(np.mean(power[on])) if on.any() else 1.0
    return 1.0  # constant-envelope waveforms: tone, chirp, FSK/MSK/GMSK, FM, PM, SSB, CW, FMCW, PMCW


def reference_power(cfg: TxConfig) -> float:
    """Mean power of the clean waveform while it is on (maps ``power_dbm`` onto samples)."""
    return max(_reference_power(_waveform_json(cfg)), 1e-30)


def _component_config(cfg: TxConfig, index: int, c: TxConfig, duration: float | None = None) -> TxConfig:
    """Component ``index`` as a clean stand-alone design on the parent's clock.

    Components run on the parent's clock even when ``cfg`` was derived with
    ``model_copy`` (which skips the validator that copies the clock), and get
    a seed of their own, so two copies never share noise or PN data.
    """
    update = {"snr_db": None, "normalize_output": True, "amplitude": 1.0,
              "iq_gain_db": 0.0, "iq_phase_deg": 0.0, "dc_i": 0.0, "dc_q": 0.0,
              "sample_rate": cfg.sample_rate, "duration": cfg.duration if duration is None else duration,
              "rf_center_hz": cfg.rf_center_hz, "seed": component_seed(cfg.seed, index, c.seed)}
    if duration is not None and c.waveform == WaveformKind.CHIRP and c.chirp_period_s is None:
        update["chirp_period_s"] = cfg.duration  # the sweep the capture length defines
    return c.model_copy(update=update)


# Samples over which components are compared for coherence (a design property).
_COHERENCE_SAMPLES = 1 << 16


@lru_cache(maxsize=16)
def _composite_power(config_json: str) -> tuple[float, float]:
    """``P_ref`` (dBm) and the composite's on-power relative to it.

    Independent components add in power.  Components that carry the same
    deterministic content (a duplicated tone, two copies of one text
    payload at one frequency) add coherently: the total is
    ``Σᵢ Σⱼ aᵢ aⱼ Re ρᵢⱼ`` with ``ρ`` the correlation coefficients of the
    clean component signals, measured over the first 65,536 samples of the
    stream (independent signals give |ρ| of a few thousandths there).
    """
    cfg = TxConfig.model_validate_json(config_json)
    enabled = [(i, c) for i, c in enumerate(cfg.components) if c.enabled]
    reference_dbm = max(c.power_dbm for _, c in enabled)
    a = np.array([10.0 ** ((c.power_dbm - reference_dbm) / 20.0) for _, c in enabled])
    if len(enabled) < 2:
        return float(reference_dbm), float(np.sum(a * a))
    n = _COHERENCE_SAMPLES
    s = np.stack([generate(_component_config(cfg, i, c, n / cfg.sample_rate), start_sample=0).samples.astype(np.complex128)
                  for i, c in enabled])
    gram = (s.conj() @ s.T) / n
    scale = np.sqrt(np.maximum(np.real(np.diag(gram)), 1e-300))
    rho = np.real(gram) / np.outer(scale, scale)
    np.fill_diagonal(rho, 1.0)
    return float(reference_dbm), float(max(a @ rho @ a, 1e-30))


def composite_power(cfg: TxConfig) -> tuple[float, float]:
    """(``P_ref`` dBm, on-power relative to ``P_ref``) of a composite design."""
    return _composite_power(cfg.model_dump_json(exclude={"label", "snr_db", "iq_gain_db", "iq_phase_deg", "dc_i", "dc_q",
                                                         "burst_enabled", "burst_period_s", "burst_duty", "agility",
                                                         "phase_offset_deg", "center_hz", "amplitude", "normalize_output"}))


def _composite(cfg: TxConfig, frame: int, start_sample: int) -> tuple[GeneratedSignal, float, float]:
    """Sum of the enabled components at their absolute powers.

    Each component is generated clean at unit on-power and scaled by
    ``10**((P_i - P_ref)/20)`` with ``P_ref`` the strongest component, so a
    component of amplitude 1 corresponds to ``P_ref``.  Returns the summed
    signal, ``P_ref`` and the expected total on-power relative to ``P_ref``
    (see ``composite_power``).
    """
    from .bands import nominal_band
    enabled = [(i, c) for i, c in enumerate(cfg.components) if c.enabled]
    reference_dbm, power_sum = composite_power(cfg)
    y = np.zeros(capture_samples(cfg), dtype=np.complex128)
    parts = []
    for i, c in enabled:
        g = generate(_component_config(cfg, i, c), frame=frame, start_sample=start_sample)
        amplitude = 10.0 ** ((c.power_dbm - reference_dbm) / 20.0)
        y += g.samples.astype(np.complex128) * amplitude
        band = nominal_band(c)
        parts.append({"index": i, "label": c.label or f"Component {i + 1}", "waveform": c.waveform.value,
                      "power_dbm": c.power_dbm, "center_hz": c.center_hz + cfg.center_hz,
                      "band_hz": None if band is None else [band[0] + cfg.center_hz, band[1] + cfg.center_hz],
                      "symbol_rate": g.metadata.get("symbol_rate"), "agile": c.agility.enabled,
                      "burst": c.burst_enabled})
    meta = {"waveform": "composite", "components": parts,
            "total_power_dbm": float(reference_dbm + 10.0 * np.log10(power_sum))}
    return GeneratedSignal(y, cfg.sample_rate, meta), float(reference_dbm), power_sum


def generate(cfg: TxConfig, start_s: float = 0, frame: int = 0, start_sample: int | None = None) -> GeneratedSignal:
    """Samples ``start_sample … start_sample + N − 1`` of the transmitter stream (N = ``capture_samples``).

    ``start_sample`` defaults to ``round(start_s * sample_rate)``.  ``frame``
    (the capture number) only reaches custom programs that use it explicitly.
    """
    fs = cfg.sample_rate
    n = capture_samples(cfg)
    start_sample = int(round(start_s * fs)) if start_sample is None else int(start_sample)
    w = cfg.waveform

    if w == WaveformKind.COMPOSITE:
        out, reference_dbm, power_sum = _composite(cfg, frame, start_sample)
        y = out.samples
    else:
        out = _clean(cfg, start_sample, n, frame)
        y = out.samples
    # complex64 from here on: phases are computed in float64 (phasor) and the
    # 24-bit mantissa sits ~144 dB below the signal, under every modelled impairment.
    y = np.asarray(y, dtype=np.complex64)
    if abs(cfg.phase_offset_deg) > 1e-12:
        y = y * np.complex64(np.exp(1j*np.deg2rad(cfg.phase_offset_deg)))
    if abs(cfg.center_hz) > 1e-12:
        y = y * phasor(cfg.center_hz, fs, len(y), start_sample).astype(np.complex64)
    # The calibration reference is the clean signal's on-power, a property of
    # the design: added AWGN, DC and I/Q imbalance read on top of the
    # configured power, and burst, hop or pulse off-time reads the configured
    # power while the signal is on, as a pulsed generator's level setting does.
    if w == WaveformKind.COMPOSITE:
        # Independent components add in power: scale by the expected on-power.
        gain = cfg.amplitude / np.sqrt(power_sum) if cfg.normalize_output else cfg.amplitude
        y = y * np.float32(gain)
        reference_rms = float(gain)
        signal_power = float(gain * gain * power_sum)
        power_dbm = reference_dbm
    else:
        scale = cfg.amplitude / math.sqrt(reference_power(cfg)) if cfg.normalize_output else cfg.amplitude
        y = y * np.float32(scale)
        reference_rms = float(cfg.amplitude) if cfg.normalize_output else float(cfg.amplitude * math.sqrt(reference_power(cfg)))
        signal_power = reference_rms ** 2
        power_dbm = float(cfg.power_dbm)
    if cfg.agility.enabled:
        from .agility import apply as apply_agility
        y = apply_agility(y, cfg.agility, fs, start_sample).astype(np.complex64, copy=False)
    if cfg.snr_db is not None and w != WaveformKind.AWGN:
        noise_power = signal_power / (10.0 ** (cfg.snr_db / 10.0))
        noise = stream_noise(noise_key(cfg.seed, _NOISE_IMPAIRMENT), start_sample, len(y))
        noise *= np.float32(math.sqrt(noise_power / 2.0))
        y = y + noise
    if cfg.iq_gain_db or cfg.iq_phase_deg:
        gain = 10**(cfg.iq_gain_db/20)
        i, q = y.real*np.float32(gain), y.imag/np.float32(gain)
        phi = np.deg2rad(cfg.iq_phase_deg)
        y = (i + 1j*(q*np.float32(np.cos(phi))+i*np.float32(np.sin(phi)))).astype(np.complex64)
    if cfg.dc_i or cfg.dc_q:
        y = y + np.complex64(cfg.dc_i + 1j*cfg.dc_q)
    if cfg.burst_enabled:
        # Gate on absolute time so the burst pattern continues across captures.
        y = y * (phase_cycles(1.0 / cfg.burst_period_s, fs, len(y), start_sample) < cfg.burst_duty)
    out.samples = y.astype(np.complex64, copy=False)
    out.metadata.update({
        "sample_rate":fs, "duration":len(y)/fs, "center_hz":cfg.center_hz, "snr_db":cfg.snr_db,
        "center_frequency_hz": cfg.rf_center_hz or None,
        "start_sample":start_sample, "capture_samples":len(y),
        "power_dbm":power_dbm, "power_reference_rms":reference_rms,
        "power_calibration":("Each component's on-power is calibrated to its own dBm setting; 1 mW reference."
                             if w == WaveformKind.COMPOSITE else
                             "The clean signal's mean power while on (before AWGN, I/Q imbalance, DC, burst and hop gating) is calibrated to the configured dBm value; 1 mW reference. Pulses, bursts and hops read the configured power while on.")
    })
    if cfg.agility.enabled:
        from .agility import summary
        out.metadata["agility"] = {k: v for k, v in summary(cfg.agility, fs, max_segments=0).items() if k != "plan"}
    from .bands import band_warnings
    warnings = band_warnings(cfg)
    if warnings:
        out.metadata["warnings"] = warnings
    return out
