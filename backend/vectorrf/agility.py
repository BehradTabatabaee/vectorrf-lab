"""Frequency agility: hopping and swept translation of any baseband signal.

A schedule is a periodic sequence of *segments*.  During the first part of a
segment (the dwell) the signal is on and its frequency either stays constant
(a hop) or moves linearly (a sweep); during the optional off-time that follows
the output is gated off while the oscillator holds its last frequency.

The oscillator is modelled as a 64-bit direct digital synthesizer (DDS):
every frequency is a tuning word ``W = round(f / fs * 2**64)`` and the phase is
an unsigned 64-bit accumulator.  Unsigned wrap-around arithmetic is exactly
modulo 2**64, so the phase of absolute sample ``k`` is exact integer
arithmetic at any sample rate and after any simulated time.  Frequencies are
quantised to the float64 value of ``f / fs`` (relative error below 1e-16), far
below every other model error.

Phase options for hops:

* ``continuous`` integrates frequency through every hop, like a VCO or a
  phase-accumulating DDS: no phase steps, minimal spectral splatter.
* ``coherent`` gives every hop the phase it would have had if its oscillator
  had run since time zero, like a bank of coherent synthesizers: phase steps at
  hop boundaries, but each hop's phase is independent of the hop order.

Sweeps are always phase continuous.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

MAX_HOPS = 262_144
_TWO64 = float(2 ** 64)
_MASK = (1 << 64) - 1
_CHUNK = 1 << 20


class FrequencyAgility(BaseModel):
    """Hopping or sweeping applied on top of any transmitter signal.

    Frequencies are complex-baseband offsets in Hz (relative to the
    transmitter's RF centre), added to the signal's own frequency offset.
    """
    model_config = ConfigDict(allow_inf_nan=False)
    enabled: bool = False
    mode: Literal["hop", "sweep"] = "hop"
    # Hop frequency set: a linear grid start..stop in step increments, an
    # explicit list, or a seeded pseudo-random permutation of the grid.
    plan: Literal["linear", "list", "random"] = "linear"
    start_hz: float = -100_000.0
    stop_hz: float = 100_000.0
    step_hz: float = Field(default=20_000.0, gt=0.0)
    frequencies_hz: list[float] = Field(default_factory=lambda: [-50_000.0, 25_000.0, 75_000.0], max_length=16_384)
    # Traversal: start→stop, stop→start, or start→stop→start (triangle).
    order: Literal["up", "down", "up-down"] = "up"
    dwell_s: float = Field(default=1e-3, gt=0.0, le=60.0)
    off_s: float = Field(default=0.0, ge=0.0, le=60.0)
    ramp_s: float = Field(default=0.0, ge=0.0, le=60.0)
    offset_s: float = Field(default=0.0, ge=0.0, le=3600.0)
    phase: Literal["continuous", "coherent"] = "continuous"
    seed: int = Field(default=1, ge=0, le=2**32 - 1)

    @model_validator(mode="after")
    def _validate_agility(self):
        if not self.enabled:
            return self
        if self.mode == "sweep" and self.start_hz == self.stop_hz:
            raise ValueError("A frequency sweep needs different start and stop frequencies")
        if self.mode == "hop" and self.plan == "list" and not self.frequencies_hz:
            raise ValueError("A hop list needs at least one frequency")
        if self.mode == "hop" and self.plan in {"linear", "random"}:
            count = hop_count(self)
            if count > MAX_HOPS:
                raise ValueError(f"The hop grid has {count:,} frequencies; at most {MAX_HOPS:,} are supported. "
                                 "Increase the step or narrow the range.")
        return self


def hop_count(cfg: FrequencyAgility) -> int:
    """Number of distinct frequencies in a linear/random hop grid."""
    span = abs(float(cfg.stop_hz) - float(cfg.start_hz))
    # A small tolerance keeps start..stop inclusive when the span is an exact
    # multiple of the step despite float rounding (900 MHz..1 GHz in 100 kHz).
    return int(np.floor(span / float(cfg.step_hz) * (1 + 1e-12) + 1e-9)) + 1


def hop_frequencies(cfg: FrequencyAgility) -> np.ndarray:
    """Hop frequencies in transmission order for one schedule period."""
    if cfg.plan == "list":
        f = np.asarray(cfg.frequencies_hz, dtype=np.float64)
    else:
        n = hop_count(cfg)
        direction = 1.0 if cfg.stop_hz >= cfg.start_hz else -1.0
        f = float(cfg.start_hz) + direction * float(cfg.step_hz) * np.arange(n, dtype=np.float64)
        if cfg.plan == "random":
            return f[np.random.default_rng(cfg.seed).permutation(n)]
    if cfg.order == "down":
        f = f[::-1]
    elif cfg.order == "up-down" and f.size > 2:
        f = np.concatenate([f, f[-2:0:-1]])
    return f


@dataclass(frozen=True)
class Schedule:
    """One period of a frequency schedule in samples and DDS tuning words.

    Every segment has the same dwell and off-time, so a sample's segment
    follows from integer division instead of a search.
    """
    on: int               # dwell samples per segment
    off: int              # off-time samples per segment
    word0: np.ndarray     # uint64 tuning word at the start of each dwell
    slope: np.ndarray     # uint64 per-sample tuning-word increment (0 for hops)
    word_end: np.ndarray  # uint64 tuning word held during the off-time
    phase0: np.ndarray    # uint64 accumulator at each segment start (continuous mode)
    period_phase: int     # accumulator advance per period (continuous mode)
    ramp: int             # raised-cosine ramp samples at each dwell edge
    offset: int           # schedule delay in samples
    coherent: bool
    sweep: bool
    f_start: np.ndarray   # float64 Hz at the start of each dwell
    f_stop: np.ndarray    # float64 Hz at the end of each dwell

    @property
    def segment(self) -> int:
        return self.on + self.off

    @property
    def count(self) -> int:
        return len(self.word0)

    @property
    def period(self) -> int:
        return self.segment * self.count

    @property
    def duty(self) -> float:
        return self.on / self.segment


def _words(freq_hz: np.ndarray, fs: float) -> np.ndarray:
    """Signed tuning words as uint64 (two's complement wraps like the DDS)."""
    w = np.round(np.asarray(freq_hz, dtype=np.float64) / fs * _TWO64)
    # |f| < fs/2 keeps |w| < 2**63; the clip only guards float rounding at the edge.
    return np.clip(w, -(2.0 ** 63), 2.0 ** 63 - 1024).astype(np.int64).view(np.uint64)


def _triangular(u: np.ndarray) -> np.ndarray:
    """``u*(u-1)/2`` modulo 2**64 for uint64 ``u`` without losing the halving."""
    u = np.asarray(u, dtype=np.uint64)
    one = np.uint64(1)
    even = (u & one) == 0
    return np.where(even, (u >> one) * (u - one), u * ((u - one) >> one))


@lru_cache(maxsize=32)
def _schedule(config_json: str, fs: float) -> Schedule:
    cfg = FrequencyAgility.model_validate_json(config_json)
    on = max(1, int(round(cfg.dwell_s * fs)))
    off = int(round(cfg.off_s * fs))
    sweep = cfg.mode == "sweep"
    if sweep:
        a, b = float(cfg.start_hz), float(cfg.stop_hz)
        f0, f1 = {"up": ([a], [b]), "down": ([b], [a]), "up-down": ([a, b], [b, a])}[cfg.order]
        f0, f1 = np.asarray(f0), np.asarray(f1)
    else:
        f0 = f1 = hop_frequencies(cfg)
    w0 = _words(f0, fs)
    if sweep:
        # Per-sample increment so the accumulated frequency reaches f1 after
        # the dwell; float64 keeps 53 bits of the increment, ample for a ramp.
        increment = np.round((f1 - f0) / fs * _TWO64 / max(on, 2))
        slope = np.clip(increment, -(2.0 ** 63), 2.0 ** 63 - 1024).astype(np.int64).view(np.uint64)
    else:
        slope = np.zeros(len(f0), dtype=np.uint64)
    with np.errstate(over="ignore"):
        on_u, off_u = np.uint64(on), np.uint64(off)
        word_end = w0 + slope * on_u
        segment_phase = w0 * on_u + slope * _triangular(np.array([on_u]))[0] + word_end * off_u
        cumulative = np.cumsum(segment_phase, dtype=np.uint64)
        phase0 = np.concatenate([np.zeros(1, np.uint64), cumulative[:-1]]).astype(np.uint64)
    return Schedule(on, off, w0, slope, word_end, phase0, int(cumulative[-1]) & _MASK,
                    min(int(round(cfg.ramp_s * fs)), on // 2), int(round(cfg.offset_s * fs)),
                    not sweep and cfg.phase == "coherent", sweep, f0.astype(np.float64), f1.astype(np.float64))


def schedule(cfg: FrequencyAgility, fs: float) -> Schedule:
    return _schedule(cfg.model_dump_json(), float(fs))


def _gate(u: np.ndarray, on: int, ramp: int) -> np.ndarray:
    g = (u < on).astype(np.float64)
    if ramp > 0:
        rise = u < ramp
        fall = (u >= on - ramp) & (u < on)
        g[rise] = 0.5 * (1.0 - np.cos(np.pi * (u[rise] + 0.5) / ramp))
        g[fall] = 0.5 * (1.0 - np.cos(np.pi * (on - u[fall] - 0.5) / ramp))
    return g


def _oscillator_chunk(s: Schedule, start_sample: int, n: int, out: np.ndarray) -> np.ndarray:
    """Write the oscillator for ``n`` samples into ``out``; return the gate."""
    k = np.arange(n, dtype=np.int64) + (start_sample - s.offset)
    before = k < 0
    k = np.maximum(k, 0)
    index = k // s.segment
    u = k - index * s.segment
    seg = index % s.count
    with np.errstate(over="ignore"):
        if s.coherent:
            # Each hop's phase is W·k as if its oscillator ran since t = 0.
            acc = s.word0[seg] * (k + s.offset).astype(np.uint64)
        else:
            uu = u.astype(np.uint64)
            acc = (index // s.count).astype(np.uint64) * np.uint64(s.period_phase) + s.phase0[seg]
            if s.sweep:
                v = np.minimum(uu, np.uint64(s.on))
                acc += s.word0[seg] * v + s.slope[seg] * _triangular(v) + s.word_end[seg] * (uu - v)
            else:
                acc += s.word0[seg] * uu
    angle = (acc >> np.uint64(11)).astype(np.float64)
    angle *= 2.0 * np.pi * 2.0 ** -53
    np.cos(angle, out=out.real)
    np.sin(angle, out=out.imag)
    gate = _gate(u, s.on, s.ramp)
    gate[before] = 0.0
    return gate


def oscillator(cfg: FrequencyAgility, fs: float, n: int, start_sample: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Complex local oscillator and amplitude gate for samples ``start_sample …``.

    Returns ``(lo, gate)``; the agile signal is ``x * gate * lo``.  Before the
    schedule offset the oscillator is gated off.
    """
    s = schedule(cfg, fs)
    lo = np.empty(int(n), dtype=np.complex128)
    gate = np.empty(int(n), dtype=np.float64)
    # Bounded chunks keep the integer phase arrays small for 8 Mi-sample captures.
    for a in range(0, int(n), _CHUNK):
        b = min(int(n), a + _CHUNK)
        gate[a:b] = _oscillator_chunk(s, int(start_sample) + a, b - a, lo[a:b])
    return lo, gate


def apply(x: np.ndarray, cfg: FrequencyAgility, fs: float, start_sample: int = 0) -> np.ndarray:
    """Translate and gate ``x`` along the schedule (no-op when disabled)."""
    if not cfg.enabled:
        return x
    lo, gate = oscillator(cfg, fs, len(x), start_sample)
    lo *= gate
    lo *= x
    return lo


def frequency_extent(cfg: FrequencyAgility) -> tuple[float, float]:
    """Lowest and highest translation frequency the schedule visits."""
    if not cfg.enabled:
        return 0.0, 0.0
    if cfg.mode == "sweep":
        return min(cfg.start_hz, cfg.stop_hz), max(cfg.start_hz, cfg.stop_hz)
    f = hop_frequencies(cfg)
    return float(np.min(f)), float(np.max(f))


def summary(cfg: FrequencyAgility, fs: float, max_segments: int = 1024) -> dict:
    """Schedule description for the UI time–frequency plan."""
    s = schedule(cfg, fs)
    shown = min(s.count, max_segments)
    lo, hi = frequency_extent(cfg)
    return {
        "mode": cfg.mode, "segments": s.count, "period_s": s.period / fs, "duty": s.duty,
        "dwell_s": s.on / fs, "off_s": s.off / fs, "ramp_s": s.ramp / fs, "offset_s": s.offset / fs,
        "min_hz": lo, "max_hz": hi, "phase": "coherent" if s.coherent else "continuous",
        "hop_rate_hz": fs / s.segment if not s.sweep else None,
        "sweep_rate_hz_s": float((s.f_stop[0] - s.f_start[0]) * fs / s.on) if s.sweep else None,
        "plan": {"start_s": (np.arange(shown) * s.segment / fs).tolist(), "f_start_hz": s.f_start[:shown].tolist(),
                 "f_stop_hz": s.f_stop[:shown].tolist(), "truncated": bool(s.count > shown)},
    }
