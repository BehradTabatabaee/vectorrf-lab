from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from functools import lru_cache
import math
import os
import numpy as np
from scipy import signal

_EPS = 1e-15


def phase_cycles(freq_hz: float, fs: float, n: int, start_sample: int = 0, start_time: Fraction | None = None) -> np.ndarray:
    """Fractional carrier cycles ``frac(f·(t0 + k/fs))`` for k = 0…n-1.

    ``t0`` is ``start_sample / fs`` or, when given, the exact rational
    ``start_time`` in seconds.  The capture-start phase is evaluated with exact
    rational arithmetic, so a tone keeps sub-microradian phase accuracy after
    arbitrarily long simulated time or at multi-GS/s rates.  Evaluating
    ``f*(t+start_s)`` in float64 instead loses precision once ``f·start_s``
    grows large; at 2 GS/s and 10^4 s simulated time that appears as roughly
    −50 dBc phase noise.
    """
    start, step = _cycle_terms(freq_hz, fs, start_sample, start_time)
    k = np.arange(int(n), dtype=np.float64)
    return np.mod(start + np.mod(k * step, 1.0), 1.0)


def _cycle_terms(freq_hz: float, fs: float, start_sample: int = 0, start_time: Fraction | None = None) -> tuple[float, float]:
    """Start phase and per-sample increment, in cycles, with the start evaluated exactly."""
    f = Fraction(float(freq_hz))
    ratio = f / Fraction(float(fs))
    t0 = Fraction(int(start_sample)) / Fraction(float(fs)) if start_time is None else Fraction(start_time)
    return float((f * t0) % 1), float(ratio % 1)


_PHASOR_BLOCK = 1024


def phasor(freq_hz: float, fs: float, n: int, start_sample: int = 0, phase_rad: float = 0.0,
           start_time: Fraction | None = None, dtype=np.complex128) -> np.ndarray:
    """Complex exponential ``exp(j(2π f (t0 + k/fs) + phase))`` with exact start phase.

    Long records are built as an outer product: sample ``k = b·B + i`` is
    ``exp(j2π(start + frac(b·B·step))) · exp(j2π·frac(i·step))``.  Each factor
    carries the same float64 phase rounding as evaluating ``k·step`` directly,
    but only ``n/B + B`` complex exponentials are computed instead of ``n``.
    With ``dtype=np.complex64`` the product is formed in single precision
    (phase error about 10⁻⁷ rad, at half the memory traffic).
    """
    n = int(n)
    if not freq_hz:
        return np.full(n, np.exp(1j * phase_rad), dtype=dtype)
    start, step = _cycle_terms(freq_hz, fs, start_sample, start_time)
    if n <= 4 * _PHASOR_BLOCK:
        k = np.arange(n, dtype=np.float64)
        return np.exp(1j * (2.0 * np.pi * np.mod(start + np.mod(k * step, 1.0), 1.0) + phase_rad)).astype(dtype, copy=False)
    blocks = -(-n // _PHASOR_BLOCK)
    inner = np.exp(2j * np.pi * np.mod(np.arange(_PHASOR_BLOCK, dtype=np.float64) * step, 1.0))
    heads = np.arange(0, blocks * _PHASOR_BLOCK, _PHASOR_BLOCK, dtype=np.float64)
    outer = np.exp(1j * (2.0 * np.pi * np.mod(start + np.mod(heads * step, 1.0), 1.0) + phase_rad))
    return (outer.astype(dtype, copy=False)[:, None] * inner.astype(dtype, copy=False)[None, :]).reshape(-1)[:n]


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(x) ** 2) + _EPS))


def normalize_rms(x: np.ndarray, target: float = 1.0) -> np.ndarray:
    r = rms(x)
    return (x * (target / r)).astype(np.complex64, copy=False) if r > 0 else x.astype(np.complex64)


_NOISE_CHUNK = 1 << 17
_noise_pool: ThreadPoolExecutor | None = None


def complex_normal(n: int, rng: np.random.Generator) -> np.ndarray:
    """``n`` complex64 samples whose real and imaginary parts are independent N(0, 1).

    Long records are drawn in fixed chunks of 131,072 samples from independent
    generators seeded by ``rng``, filled in parallel threads (numpy releases the
    GIL while filling).  The result depends only on ``rng``'s state and ``n``,
    never on the number of threads, so captures stay reproducible.
    """
    n = int(n)
    out = np.empty(2 * n, dtype=np.float32)
    chunks = -(-n // _NOISE_CHUNK)
    if chunks <= 1:
        rng.standard_normal(2 * n, dtype=np.float32, out=out)
        return out.view(np.complex64)
    keys = rng.integers(0, 2**63 - 1, size=chunks, dtype=np.int64)

    def fill(i: int) -> None:
        a, b = 2 * i * _NOISE_CHUNK, min(2 * n, 2 * (i + 1) * _NOISE_CHUNK)
        np.random.default_rng(int(keys[i])).standard_normal(b - a, dtype=np.float32, out=out[a:b])

    list(_pool().map(fill, range(chunks)))
    return out.view(np.complex64)


_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _splitmix64(counter: np.ndarray, key: np.uint64) -> np.ndarray:
    """SplitMix64 of ``key + (counter + 1)·φ`` (a counter-based generator: the
    output for a counter depends on nothing else)."""
    with np.errstate(over="ignore"):
        z = key + (counter + np.uint64(1)) * _GOLDEN
        z ^= z >> np.uint64(30)
        z *= _MIX1
        z ^= z >> np.uint64(27)
        z *= _MIX2
        z ^= z >> np.uint64(31)
    return z


def noise_key(seed: int, stream: int) -> np.uint64:
    """Key of the noise stream ``stream`` of a design with ``seed``."""
    return _splitmix64(np.array([int(seed) & 0xFFFFFFFF], np.uint64), np.uint64((int(stream) * 0x632BE59BD9B4E019) & (2**64 - 1)))[0]


def _noise_chunk(key: np.uint64, chunk: int, out: np.ndarray) -> None:
    """Fill ``out`` with the start of noise chunk ``chunk`` (Philox counter stream ``chunk``)."""
    g = np.random.Generator(np.random.Philox(key=int(key), counter=[0, int(chunk), 0, 0]))
    out.view(np.float32)[:] = g.standard_normal(2 * _NOISE_CHUNK, dtype=np.float32)[:2 * len(out)]


def _pool() -> ThreadPoolExecutor:
    global _noise_pool
    if _noise_pool is None:
        _noise_pool = ThreadPoolExecutor(max_workers=max(1, min(8, (os.cpu_count() or 2) // 2)), thread_name_prefix="vectorrf-dsp")
    return _noise_pool


def interpolate(taps: np.ndarray, symbols: np.ndarray, up: int) -> np.ndarray:
    """``scipy.signal.upfirdn(taps, symbols, up)`` (the full output), computed in
    parallel threads for long records.

    The symbols are split into blocks of a fixed size whose outputs are
    overlap-added, so the result does not depend on the number of threads.
    """
    up = int(up)
    n_out = (len(symbols) - 1) * up + len(taps)
    if len(symbols) < 2 or n_out < (1 << 18):
        return signal.upfirdn(taps, symbols, up=up)
    block = max(1, (1 << 16) // up)
    starts = range(0, len(symbols), block)
    parts = list(_pool().map(lambda b: signal.upfirdn(taps, symbols[b:b + block], up=up), starts))
    out = np.zeros(n_out, dtype=np.result_type(parts[0].dtype, np.complex64))
    for b, part in zip(starts, parts):
        out[b * up:b * up + len(part)] += part
    return out


_SHAPE_BLOCK = 1 << 16   # samples per pulse-shaping product, at most 256 symbol rows (see shape_pulses)


def shape_pulses(symbols: np.ndarray, first: int, taps: np.ndarray, sps: int, start: int, n: int) -> np.ndarray:
    """Samples ``start … start + n − 1`` of y[i] = Σ_m s_m · h[i − m·sps].

    ``symbols`` holds s_m for m = first, first + 1, …; symbols outside it are 0.
    In polyphase form the output of symbol row m and phase r = i − m·sps is
    Σ_k s_{m−k} · H[k, r] with H = h reshaped to (taps/sps) × sps, a small
    matrix product: the cost is n × (filter span in symbols) multiply-adds for
    any samples per symbol, and no filter ramps outside the slice are computed.
    """
    sps, n, start = int(sps), int(n), int(start)
    k = -(-len(taps) // sps)                              # symbols spanned by the pulse
    h = np.zeros(k * sps, dtype=np.float32)
    h[:len(taps)] = taps
    H = h.reshape(k, sps)
    # Symbol rows covering the slice, widened to whole blocks of ``block`` rows
    # (a function of ``sps`` alone) at absolute multiples: BLAS rounds a row
    # differently depending on the shape of the product and the row's place in
    # it, so every row is always computed in the same block, and any slice (a
    # capture or its prefix) holds exactly the samples of the whole stream.
    block = max(1, min(256, _SHAPE_BLOCK // sps))
    row0 = (start // sps) // block * block
    row1 = -(-((start + n - 1) // sps + 1) // block) * block   # exclusive
    lo = row0 - (k - 1)                                   # earliest symbol that reaches row0
    ext = np.zeros(row1 - lo, dtype=np.complex64)
    a, b = max(lo, first), min(row1, first + len(symbols))
    if b > a:
        ext[a - lo:b - lo] = symbols[a - first:b - first]
    # Row j holds s_{row0+j−(k−1)} … s_{row0+j}; against the tap rows in reverse
    # order that is Σ_k s_{row0+j−k}·H[k].  Real and imaginary parts go through
    # one batched real matrix product (BLAS needs contiguous operands).
    windows = np.lib.stride_tricks.sliding_window_view(ext, k)
    rows = len(windows)
    stacked = np.concatenate([windows.real, windows.imag]).astype(np.float32).reshape(-1, block, k)
    product = (stacked @ np.ascontiguousarray(H[::-1])).reshape(2 * rows, sps)
    out = np.empty((rows, sps), dtype=np.complex64)
    out.real = product[:rows]
    out.imag = product[rows:]
    offset = start - row0 * sps
    return out.reshape(-1)[offset:offset + n]


def stream_noise(key: np.uint64, start: int, n: int) -> np.ndarray:
    """Complex Gaussian noise for absolute samples ``start … start + n − 1``.

    The stream is divided into chunks of 131,072 samples; chunk c is drawn
    from its own Philox counter stream (key, c), so every sample is a pure
    function of (key, absolute index): any capture of a stream, of any length
    at any position, holds exactly the noise of those samples.  Real and
    imaginary parts are independent N(0, 1).  Chunks are filled in parallel
    threads (numpy releases the GIL while filling); the result does not
    depend on the threading.
    """
    n, start = int(n), int(start)
    out = np.empty(n, dtype=np.complex64)
    if n <= 0:
        return out
    first, last = start // _NOISE_CHUNK, (start + n - 1) // _NOISE_CHUNK

    def fill(c: int) -> None:
        lo, hi = max(start, c * _NOISE_CHUNK), min(start + n, (c + 1) * _NOISE_CHUNK)
        block = np.empty(hi - c * _NOISE_CHUNK, dtype=np.complex64)
        _noise_chunk(key, c, block)
        out[lo - start:hi - start] = block[lo - c * _NOISE_CHUNK:]

    if first == last:
        fill(first)
        return out
    list(_pool().map(fill, range(first, last + 1)))
    return out


def add_awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator, signal_power: float | None = None) -> np.ndarray:
    """Add complex white Gaussian noise at ``snr_db`` relative to the signal power.

    ``signal_power`` defaults to the mean power of ``x``.  Pass it explicitly
    when ``x`` contains gated/burst samples whose average does not represent
    the signal power the SNR refers to.
    """
    x = np.asarray(x)
    p = (float(np.vdot(x, x).real) / max(x.size, 1) if signal_power is None else float(signal_power)) + _EPS
    noise_p = p / (10.0 ** (snr_db / 10.0))
    # Interleaved float32 normals viewed as complex64: real and imaginary parts
    # are independent N(0, 1) draws, scaled to noise_p / 2 each.
    n = complex_normal(x.size, rng)
    n *= np.float32(math.sqrt(noise_p / 2.0))
    return (x + n).astype(np.complex64, copy=False)


def rrc_taps(sps: int, rolloff: float, span_symbols: int) -> np.ndarray:
    """Root-raised-cosine FIR sampled at integer samples/symbol (unit energy)."""
    return _rrc_taps(max(2, int(sps)), float(rolloff), int(span_symbols)).copy()


@lru_cache(maxsize=64)
def _rrc_taps(sps: int, beta: float, span_symbols: int) -> np.ndarray:
    n = span_symbols * sps
    if n % 2:
        n += 1
    t = np.arange(-n // 2, n // 2 + 1, dtype=np.float64) / sps
    h = np.empty_like(t)
    centre = np.abs(t) < 1e-12
    # t = ±1/(4β) is a removable singularity of the closed form.
    edge = (np.abs(np.abs(t) - 1.0 / (4.0 * beta)) < 1e-10) if beta > 0 else np.zeros_like(centre)
    rest = ~(centre | edge)
    tr = t[rest]
    h[rest] = (np.sin(np.pi * tr * (1 - beta)) + 4 * beta * tr * np.cos(np.pi * tr * (1 + beta)))         / (np.pi * tr * (1 - (4 * beta * tr) ** 2))
    h[centre] = 1.0 - beta + 4.0 * beta / np.pi
    if beta > 0:
        h[edge] = (beta / np.sqrt(2.0)) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                            + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
    h /= np.sqrt(np.sum(h * h) + _EPS)
    return h.astype(np.float32)


def fractional_delay(x: np.ndarray, delay_samples: float, ntaps: int = 41) -> np.ndarray:
    """Delay ``x`` by ``delay_samples`` (|delay| < 1) with a Hamming-windowed sinc.

    The filter is centred, so the output keeps the input's length and
    alignment; the error is about −65 dB for content within 0.8·Nyquist.
    The record is treated as zero outside its ends, so only the
    ``ntaps // 2`` samples nearest each end are approximate.
    """
    if abs(delay_samples) < 1e-8:
        return x.astype(np.complex64, copy=True)
    n = np.arange(ntaps) - (ntaps - 1) / 2
    h = np.sinc(n - delay_samples) * np.hamming(ntaps)
    h /= np.sum(h)
    return filter_same(np.asarray(x, dtype=np.complex64), h.astype(np.float32))


def resample(x: np.ndarray, up: int, down: int, window=("kaiser", 10.0)) -> np.ndarray:
    """``scipy.signal.resample_poly(x, up, down, window=window)``, computed over
    overlapping chunks in parallel threads for long records.

    Chunks start at multiples of ``down`` input samples, so each lands on the
    same output phase as the whole record; margins wider than the filter
    keep every kept output identical to the one-call result.
    """
    g = math.gcd(int(up), int(down))
    up, down = int(up) // g, int(down) // g
    if len(x) < (1 << 16) or up == down:
        return signal.resample_poly(x, up, down, window=window)
    half = 10 * max(up, down)                       # resample_poly's half filter length (upsampled domain)
    margin = -(-(-(-half // up) + 2 * down) // down) * down
    block = max(down, ((1 << 15) // down) * down, -(-4 * margin // down) * down)   # mostly payload, not margin
    n_out = -(-len(x) * up // down)
    out = np.empty(n_out, dtype=np.result_type(x.dtype, np.complex64) if np.iscomplexobj(x) else np.float64)

    def run(s: int) -> None:
        lo, hi = max(0, s - margin), min(len(x), s + block + margin)
        y = signal.resample_poly(x[lo:hi], up, down, window=window)
        base = lo * up // down
        o0, o1 = s * up // down, min(n_out, (s + block) * up // down)
        out[o0:o1] = y[o0 - base:o1 - base]

    list(_pool().map(run, range(0, len(x), block)))
    return out


def filter_same(x: np.ndarray, h: np.ndarray, block: int = 1 << 16) -> np.ndarray:
    """``oaconvolve(x, h, mode="same")`` (zeros outside the record), computed
    over fixed blocks in parallel threads for long records."""
    n, taps = len(x), len(h)
    if n < 4 * block:
        return signal.oaconvolve(x, h, mode="same").astype(np.result_type(x.dtype, np.complex64), copy=False)
    left = (taps - 1) // 2
    padded = np.concatenate([np.zeros(left, x.dtype), x, np.zeros(taps - 1 - left, x.dtype)])
    out = np.empty(n, dtype=np.result_type(x.dtype, np.complex64))

    def run(b: int) -> None:
        e = min(n, b + block)
        out[b:e] = signal.oaconvolve(padded[b:e + taps - 1], h, mode="valid")

    list(_pool().map(run, range(0, n, block)))
    return out


def next_pow2(n: float) -> int:
    return 1 << max(0, int(math.ceil(math.log2(max(1.0, n)))))


def bytes_to_bits(message: str) -> np.ndarray:
    b = message.encode("utf-8")
    if not b:
        return np.zeros(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8)).astype(np.uint8)


def decode_text(bits: np.ndarray, limit: int = 4096) -> tuple[str, int]:
    """UTF-8 text of a bit stream, aligned to the byte boundary that reads most like text.

    A capture of a continuous transmission starts at an arbitrary bit, so the
    eight possible byte alignments are tried and the one with the most
    printable ASCII bytes (ties: the earliest) is decoded.  Returns the text
    and the chosen bit offset.
    """
    best, _ = text_alignment(bits)
    return bits_to_bytes(np.asarray(bits, dtype=np.uint8)[best:]).decode("utf-8", errors="replace")[:limit], best


def text_alignment(bits: np.ndarray) -> tuple[int, float]:
    """Bit offset (0–7) whose bytes read most like text, and that share of printable bytes."""
    bits = np.asarray(bits, dtype=np.uint8)
    best, best_score = 0, -1.0
    for offset in range(8):
        raw = np.frombuffer(bits_to_bytes(bits[offset:]), dtype=np.uint8)
        if not raw.size:
            break
        score = float(np.mean(((raw >= 0x20) & (raw < 0x7F)) | (raw == 0x0A) | (raw == 0x0D) | (raw == 0x09)))
        if score > best_score + 1e-12:
            best, best_score = offset, score
    return best, max(best_score, 0.0)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8)
    n = (bits.size // 8) * 8
    if n == 0:
        return b""
    return np.packbits(bits[:n]).tobytes()


def gray_to_binary(g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.int64)
    b = g.copy()
    shift = 1
    while np.max(g, initial=0) >> shift:
        b ^= g >> shift
        shift += 1
    return b


# ---------------------------------------------------------------------------
# Display payloads
#
# JSON products carry decimated series for plotting.  REST responses keep full
# double precision.  The live stream passes ``digits`` to round its plot traces
# to that many significant figures of each series' largest magnitude: far
# below what a plot resolves, and about half the bytes of full precision.


def display_values(x, digits: int | None = None) -> list:
    """``x`` as a JSON list, rounded to ``digits`` significant figures of its
    largest finite magnitude when given.  Integer and boolean arrays are exact."""
    a = np.asarray(x)
    if a.dtype.kind in "biu":
        return a.tolist()
    if digits is None:
        a = a.astype(float)
        mask = np.isfinite(a)
        return a.tolist() if mask.all() else [v if ok else None for v, ok in zip(a.tolist(), mask.tolist())]
    a = a.astype(float)
    mask = np.isfinite(a)
    finite = np.abs(a[mask])
    top = float(finite.max()) if finite.size else 0.0
    if top > 0:
        a = np.round(a, int(digits - 1 - math.floor(math.log10(top))))
    if not mask.all():
        # JSON has no NaN or infinity: an undefined point is a gap (null) in the plot.
        return [v if ok else None for v, ok in zip(a.tolist(), mask.tolist())]
    return a.tolist()


def display_indices(n: int, maximum: int) -> np.ndarray:
    """At most ``maximum`` indices evenly spread over ``0 … n-1`` (both ends kept)."""
    return np.unique(np.linspace(0, n - 1, min(maximum, n), dtype=int)) if n else np.zeros(0, int)


def display_real(x, maximum: int = 4000, digits: int | None = None) -> list:
    y = np.asarray(x).real
    return display_values(y[display_indices(len(y), maximum)], digits)


def display_complex(x, maximum: int = 2000, digits: int | None = None) -> dict | None:
    """``{'i': [...], 'q': [...]}`` with at most ``maximum`` evenly spread points."""
    if x is None:
        return None
    z = np.asarray(x)
    z = z[display_indices(len(z), maximum)]
    return {"i": display_values(z.real, digits), "q": display_values(z.imag, digits)}


def display_tree(value, maximum: int = 2200, digits: int | None = None):
    """A telemetry value for a JSON payload: series longer than ``maximum`` are
    decimated (and float series rounded when ``digits`` is given); dicts are
    handled recursively and scalars pass unchanged."""
    if isinstance(value, dict):
        return {k: display_tree(v, maximum, digits) for k, v in value.items()}
    if not isinstance(value, (list, tuple, np.ndarray)):
        return value
    if len(value) > maximum:
        value = value[::max(1, len(value) // maximum)][:maximum]
    try:
        a = np.asarray(value)
    except ValueError:  # ragged nested lists
        return list(value)
    if a.dtype.kind == "f":
        return display_values(a, digits)
    return a.tolist() if a.dtype.kind in "biu" else list(value)
