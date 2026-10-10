from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import numpy as np
from scipy import signal

from .dsp import bits_to_bytes, decode_text, phasor, resample, rrc_taps, text_alignment
from .models import RxConfig
from .modem import _nearest, evm_rms, fit_constellation, radius_scales, slice_symbols, reference_constellation
from .sync import (costas_loop, eye_segments, mueller_muller, mueller_muller_carrier, gardner, coarse_carrier,
                   decision_carrier, cma_equalizer, differential_cfo, feedforward_phase, loop_settling_symbols,
                   spectral_centre, symbol_rms, timing_settling_symbols)

# Linear demodulation runs at this many samples per symbol at most.  Timing,
# carrier and equalizer loops work per symbol, so more oversampling only adds
# cost (a 10 MS/s capture of a 50 kBd signal has 200 samples per symbol).
WORK_SPS = 16


@dataclass
class RxResult:
    recovered: np.ndarray
    symbols: np.ndarray | None
    bits: np.ndarray | None
    text: str | None
    metrics: dict
    telemetry: dict


def _symmetric_text(syms: np.ndarray, base: str, cfg: RxConfig) -> tuple[str, int, float]:
    """Text of the decisions, resolving the constellation's rotation ambiguity.

    A blind carrier loop may lock at any of the constellation's symmetric
    rotations (90° for QPSK and QAM), and a capture starts at an arbitrary bit;
    real links resolve both with a known sequence.  For the text view only,
    every rotation and byte alignment is tried and the most text-like reading
    shown (the raw bits and hex are not rotated).
    """
    order = {"bpsk": 2, "8psk": 8}.get(base, 4)  # rotational symmetry (QAM and APSK: 4)
    best = (-1.0, 0, 0)
    for q in range(order):
        _, rotated = slice_symbols(syms * np.exp(2j * np.pi * q / order), base, cfg.apsk_ratio2, cfg.apsk_ratio3)
        offset, score = text_alignment(rotated)
        if score > best[0] + 1e-9:
            best = (score, q, offset)
    _, q, offset = best
    _, bits = slice_symbols(syms * np.exp(2j * np.pi * q / order), base, cfg.apsk_ratio2, cfg.apsk_ratio3)
    return bits_to_bytes(bits[offset:]).decode("utf-8", errors="replace")[:4096], int(offset), 360.0 * q / order


def blind_phase_search(symbols: np.ndarray, refs: np.ndarray, center: float | None = None, steps: int = 48) -> float:
    """Carrier phase of square-QAM symbols by blind phase search.

    Every rotation over one 90° sector (around ``center`` when given, so the
    result takes the branch nearest it) is scored by the mean squared distance
    of the rotated symbols to their decisions, jointly with the candidate
    gains that match the symbol magnitudes to the constellation's radii
    (``radius_scales``); the best rotation, refined by a parabola through its
    neighbours, is the phase.  Unlike the fourth-power estimator it does not
    depend on how a payload uses the constellation.
    """
    z = np.asarray(symbols, dtype=np.complex128)[:512]
    refs = np.asarray(refs, dtype=np.complex128)
    base = 0.0 if center is None else float(center)
    step = np.pi / 2 / steps
    thetas = base + (np.arange(steps) - steps // 2) * step
    rotations = np.exp(-1j * thetas)
    best = None
    for gain in radius_scales(z, refs):
        rotated = (z / gain)[None, :] * rotations[:, None]
        distance = np.abs(rotated - refs[_nearest(rotated, refs).reshape(rotated.shape)]) ** 2
        costs = distance.mean(axis=1)
        k = int(np.argmin(costs))
        if best is None or costs[k] < best[0]:
            best = (float(costs[k]), k, costs)
    _, k, costs = best
    a, b, c = costs[(k - 1) % steps], costs[k], costs[(k + 1) % steps]
    delta = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c > 1e-15 else 0.0
    theta = thetas[k] + float(np.clip(delta, -0.5, 0.5)) * step
    return float((theta + np.pi) % (2 * np.pi) - np.pi)


def symbol_errors(symbols: np.ndarray, refs: np.ndarray, fit: np.ndarray | None = None) -> dict:
    """Per-symbol EVM, magnitude error (both % of the reference RMS) and phase
    error (degrees), as a vector signal analyzer's error-vs-time traces.

    The decision-directed least-squares gain and phase come from ``fit``
    (default: all symbols), the same fit the summary EVM uses, so the traces
    and the summary describe the same errors.
    """
    refs = np.asarray(refs, dtype=np.complex128)
    z = np.asarray(symbols, dtype=np.complex128)
    basis = z if fit is None else np.asarray(fit, dtype=np.complex128)
    if not len(z) or not len(basis):
        return {"evm_time_percent": np.zeros(0), "magnitude_error_percent": np.zeros(0), "phase_error_deg": np.zeros(0)}
    fitted, _ = fit_constellation(basis, refs, complex_gain=True)
    gain = np.vdot(fitted, basis) / max(float(np.vdot(fitted, fitted).real), 1e-30)
    z = z / gain
    ref = refs[_nearest(z, refs)]
    ref_rms = max(float(np.sqrt(np.mean(np.abs(ref) ** 2))), 1e-15)
    return {"evm_time_percent": 100.0 * np.abs(z - ref) / ref_rms,
            "magnitude_error_percent": 100.0 * (np.abs(z) - np.abs(ref)) / ref_rms,
            "phase_error_deg": np.rad2deg(np.angle(z * np.conj(ref)))}


def _strongest_phase(mf: np.ndarray, gd: int, sps: int, symbols: int = 2048) -> int:
    """Sampling phase (0 … sps−1 after the filter delay ``gd``) with the most
    matched-filter energy over the first ``symbols`` symbols."""
    if len(mf) - gd < 2 * sps:
        return 0
    return int(np.argmax([np.mean(np.abs(mf[gd + k:gd + k + symbols * sps:sps]) ** 2) for k in range(sps)]))


def _inner_ring_phase(symbols: np.ndarray, refs: np.ndarray, predicted: float) -> float:
    """Carrier phase of APSK symbols from the inner ring's fourth power, on the
    90° branch nearest ``predicted`` (``predicted`` when the ring is empty)."""
    z = np.asarray(symbols, dtype=np.complex128)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-30)
    refs = np.asarray(refs, dtype=np.complex128)
    radii = np.unique(np.round(np.abs(refs), 6))
    inner = z[np.abs(z) < 0.5 * (radii[0] + radii[1])] if len(radii) > 1 else z
    if len(inner) < 4:
        return float(predicted)
    reference = refs[np.abs(refs) < 0.5 * (radii[0] + radii[1])] if len(radii) > 1 else refs
    moment = np.mean((inner / np.abs(inner)) ** 4) / np.mean((reference / np.abs(reference)) ** 4)
    estimate = float(np.angle(moment)) / 4
    branch = round((predicted - estimate) / (np.pi / 2))
    return float((estimate + branch * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi)


def _inner_ring_coherence(symbols: np.ndarray, cfo_hz: float, symbol_rate: float, refs: np.ndarray) -> float:
    """Fourth-power coherence of an APSK constellation's inner ring after
    removing ``cfo_hz``: near 1 at the true carrier frequency, near 0 a branch
    (2π/M per symbol) away, where only the outer ring's symmetry holds."""
    z = np.asarray(symbols, dtype=np.complex128)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2) + 1e-30)
    radii = np.unique(np.round(np.abs(np.asarray(refs, dtype=np.complex128)), 6))
    if len(radii) < 2:
        return 0.0
    index = np.flatnonzero(np.abs(z) < 0.5 * (radii[0] + radii[1]))
    if len(index) < 8:
        return 0.0
    u = (z[index] / np.abs(z[index])) ** 4 * np.exp(-8j * np.pi * cfo_hz * index / symbol_rate)
    return float(np.abs(np.mean(u)))


def _track_key(cfg: RxConfig) -> str:
    return cfg.model_dump_json(exclude={"ofdm_start_s", "max_symbols"})


# Warm starts across a gap (see _digital): refine the predicted carrier phase
# only across gaps of at least WARM_REFINE_GAP symbols, and weigh the drift it
# reveals against the loop's averaging time when correcting the frequency.
WARM_REFINE_GAP = 64
WARM_AVERAGING_SYMBOLS = 1000


def _digital(samples: np.ndarray, cfg: RxConfig, start_sample: int | None = None, track: dict | None = None) -> RxResult:
    """Linear demodulation of one block.

    With ``track`` (a dict kept between calls) and the block's absolute
    ``start_sample``, the receiver keeps tracking from block to block instead
    of re-acquiring: the symbol clock continues on its absolute grid, the
    coarse CFO correction stays anchored to absolute time, and the carrier
    loop starts from its predicted phase (refined by an M-th power estimate
    on the new block, which also resolves the phase ambiguity consistently)
    and frequency.  Blocks then need no acquisition, which matters when real
    time keeps only a few symbols per capture.
    """
    fs = cfg.sample_rate
    sps = max(2, int(round(fs/cfg.symbol_rate)))
    actual_rs = fs/sps
    mod = cfg.modulation
    base = {"dbpsk":"bpsk", "dqpsk":"qpsk", "pi4-dqpsk":"qpsk", "oqpsk":"qpsk"}.get(mod,mod)
    telemetry = {}
    warm = None
    if track is not None and start_sample is not None:
        key = _track_key(cfg)
        if track.get("key") == key and "next" in track:
            warm = dict(track)
        else:
            track.clear()
            track["key"] = key
    # Bounded work per capture: the loops run per symbol, so analyse at most
    # ``max_symbols`` (plus the filter span) from the start of the capture.
    capture_symbols = len(samples) // sps
    samples = samples[:(cfg.max_symbols + 2 * cfg.filter_span_symbols + 8) * sps]
    input_sps = sps
    if sps > WORK_SPS:
        # Rational polyphase decimation to WORK_SPS samples/symbol first: the
        # loops work per symbol, and every carrier estimate below then sees
        # thousands of symbols however high the oversampling (131,072 input
        # samples are only three symbols at 40,000 samples/symbol).  The
        # Kaiser (β = 10, ≈ 100 dB) anti-alias filter passes ±0.68 of the new
        # Nyquist, about ±5 symbol rates, so a carrier offset of up to ±4 Rs
        # passes intact; the output is time-aligned with the input.
        g = math.gcd(WORK_SPS, sps)
        samples = resample(samples, WORK_SPS // g, sps // g, window=("kaiser", 10.0))
        sps = WORK_SPS
    per_work = input_sps / sps  # input samples per working sample
    work_rate = actual_rs * sps
    # Absolute time of the block's first sample: frequency corrections are
    # anchored to it, so a constant correction is continuous across blocks.
    t0 = Fraction(int(start_sample or 0)) / Fraction(float(fs))

    def mix(x, f):
        return (x * phasor(-f, work_rate, len(x), start_time=t0)).astype(np.complex64)

    decimated = samples
    cfo = 0.
    # Coarse carrier acquisition, in up to three stages:
    # 1. the power spectrum's centroid over the occupied band: no ambiguity,
    #    but a few percent of Rs of bias for payloads whose spectrum is not
    #    symmetric (a short repeating text); it arbitrates the others;
    # 2. the M-th power line at the working rate (PSK and QAM; range ±Fs/2M).
    #    A periodic payload or a neighbouring signal can make a false line
    #    stronger, so it is kept only where the spectrum agrees.  APSK and
    #    π/4-DQPSK have symbol-clock harmonics stronger than their carrier
    #    line and skip this stage;
    # 3. after matched filtering, a fine estimate at the symbol instants,
    #    whose Rs/M ambiguity stages 1 and 2 resolve (see below).
    cfo_order = {"bpsk": 2, "dbpsk": 2, "qpsk": 4, "dqpsk": 4, "8psk": 8,
                 "16qam": 4, "64qam": 4, "oqpsk": 4}.get(mod)
    occupied = actual_rs * (1 + cfg.rolloff) if cfg.pulse_shape == "rrc" else 2 * actual_rs
    reference = 0.0
    precise = False   # the M-th power estimate was taken (stage 3 is then not needed)
    sample_rate_estimate = None
    if cfg.coarse_cfo and warm is not None:
        cfo = reference = warm["cfo"]
        samples = mix(samples, cfo)
    elif cfg.coarse_cfo:
        search = max(0.0, min(work_rate / 2 - occupied / 2, max(occupied, 0.05 * fs)))
        spectral = spectral_centre(samples, work_rate, occupied, search_hz=search)
        cfo = reference = spectral
        if cfo_order:
            estimate = sample_rate_estimate = coarse_carrier(samples, work_rate, cfo_order)
            if abs(estimate - reference) <= 0.05 * actual_rs:
                cfo = reference = estimate
                precise = True
        telemetry["spectral_cfo_hz"] = float(spectral)
        samples = mix(samples, cfo)
    taps = rrc_taps(sps, cfg.rolloff, cfg.filter_span_symbols) if cfg.pulse_shape == "rrc" else np.ones(sps)/sps

    def matched(x):
        return (signal.fftconvolve(x, taps)[:len(x)] if len(taps) > 257 else signal.lfilter(taps, [1.0], x)).astype(np.complex64)

    mf = matched(samples)
    gd = len(taps)-1 if cfg.pulse_shape == "rrc" else sps-1
    refs = reference_constellation(base,cfg.apsk_ratio2,cfg.apsk_ratio3)
    # The sampling phase with the most matched-filter energy is the symbol
    # instant to within a fraction of a sample: the estimates below read the
    # symbols there, and a cold timing loop starts from it.
    best = _strongest_phase(mf, gd, sps)
    fine_order = {"bpsk": 2, "qpsk": 4, "8psk": 8, "16qam": 4, "64qam": 4, "16apsk": 12, "32apsk": 16,
                  "dbpsk": 2, "dqpsk": 4, "pi4-dqpsk": 4}.get(mod)
    if cfg.coarse_cfo and warm is None and not precise and fine_order and len(mf) - gd >= 64 * sps:
        symbols0 = mf[gd + best::sps][:2048].astype(np.complex128)
        if mod in {"dbpsk", "dqpsk", "pi4-dqpsk"}:
            residual = differential_cfo(symbols0, actual_rs, mod)
        else:
            residual = coarse_carrier(symbols0, actual_rs, fine_order)
        # The estimate repeats every Rs/M (Rs/4 for π/4-DQPSK); a repeating
        # payload can also put a false QAM line half a branch away.  Turning by
        # one branch per symbol is a symmetry of PSK, QAM and π/4-DQPSK, so the
        # branch nearest the coarse estimate is taken.  For APSK it is a
        # symmetry of the outer ring only: the inner ring alone (four points)
        # keeps a coherent fourth power on the right branch.
        spacing = actual_rs / fine_order / (2 if mod in {"16qam", "64qam"} else 1)
        target = reference - cfo
        candidates = [residual + k * spacing for k in range(-8, 9)]
        if mod in {"16apsk", "32apsk"}:
            plausible = [c for c in candidates if abs(c - target) <= 0.06 * actual_rs + spacing] or candidates
            residual = max(plausible, key=lambda c: _inner_ring_coherence(symbols0, c, actual_rs, refs))
        else:
            residual = min(candidates, key=lambda c: abs(c - target))
        if sample_rate_estimate is not None and abs(sample_rate_estimate - (cfo + residual)) <= 0.02 * actual_rs:
            # The branch is settled; the M-th power estimate (rejected only
            # because the spectrum's centroid was too uncertain) is the more precise.
            residual = sample_rate_estimate - cfo
        cfo += residual
        if abs(residual) >= 0.01 * actual_rs:
            # A false first estimate shifts the matched filter off centre:
            # filter again at the resolved CFO (rotating the matched filter's
            # output alone leaves avoidable ISI on QAM).
            mf = matched(mix(decimated, cfo))
        elif residual:
            mf = mix(mf, residual)
    telemetry["coarse_cfo_hz"] = cfo
    front_phase = warm.get("front_phase", 0.0) if warm is not None else 0.0
    phase_order = {"16qam": 4, "64qam": 4, "16apsk": 12, "32apsk": 16, "oqpsk": 4}.get(mod)
    if (warm is None or (mod == "oqpsk" and cfg.coarse_cfo)) and phase_order and len(mf) - gd >= 32 * sps:
        probe = mf[gd + best::sps][16:528].astype(np.complex128)
        reference_moment = np.mean(refs.astype(np.complex128) ** phase_order)
        if mod in {"16qam", "64qam"} and len(probe) >= 16:
            # A repeating payload makes the fourth-power moment data dependent
            # (it can point anywhere); the blind phase search does not care.
            front_phase = blind_phase_search(probe, refs)
        elif len(probe) >= 16 and abs(reference_moment) > 1e-12:
            estimate = float(np.angle(np.mean(probe ** phase_order) / reference_moment) / phase_order)
            if mod in {"16apsk", "32apsk"}:
                # The outer ring's M-fold symmetry leaves M/4 possible phases
                # per 90 degrees.  Inner rings resolve the ambiguity.
                candidates = (estimate + k * 2 * np.pi / phase_order
                              for k in range(-phase_order // 8 - 1, phase_order // 8 + 2))
                front_phase = min(candidates,
                    key=lambda p: (evm_rms(probe * np.exp(-1j * p), base,
                                           cfg.apsk_ratio2, cfg.apsk_ratio3)["evm_rms_percent"], abs(p)))
            else:
                if warm is not None and mod == "oqpsk":
                    estimate += round((warm["front_phase"] - estimate) / (np.pi / 2)) * (np.pi / 2)
                front_phase = estimate
    if front_phase:
        # OQPSK must be derotated before undoing its half-symbol Q delay:
        # a carrier phase otherwise mixes the staggered I and Q streams.
        mf *= np.complex64(np.exp(-1j * front_phase))
    telemetry["feedforward_phase_deg"] = float(np.rad2deg(front_phase))
    if mod == "oqpsk":
        q = np.interp(np.arange(len(mf))+sps/2,np.arange(len(mf)),mf.imag,right=0)
        mf = mf.real+1j*q
    # Symbol-level AGC: the timing and carrier loop gains refer to unit-RMS
    # symbols, so loop dynamics do not change with signal level, filter gain
    # or samples per symbol.
    level = symbol_rms(mf[gd:], sps)
    mf = (mf / level).astype(np.complex64)
    if mod in {"16qam", "64qam", "16apsk", "32apsk"} and len(mf) - gd >= 32 * sps:
        # A finite payload need not use all constellation points equally.
        # Its measured RMS can differ substantially from the reference RMS
        # (a repeating text payload biases 64-QAM by ~16%).  A decision loop
        # fed that scale error drifts even with zero carrier error.
        probe = mf[gd + best::sps][8:520]
        prior = warm["scale"] / level if warm is not None and warm.get("scale") else 1.0
        fitted, _ = fit_constellation(probe / prior, refs)
        gain = prior * np.vdot(fitted, probe / prior) / max(float(np.vdot(fitted, fitted).real), 1e-30)
        if warm is not None and warm.get("scale"):
            # A tracked block continues from the scale it inherits and moves it
            # a quarter of the way to its own fit: a short block's fit can
            # settle on another match of a dense constellation's rings (0.81 of
            # the true scale for 16-QAM), and fits accepted whole let the scale
            # wander block by block until decisions fall on the wrong grid and
            # the carrier loop loses lock.  A real level change is followed
            # over a few blocks.
            ratio = abs(gain) / prior
            gain = prior * (1.0 + 0.25 * (ratio - 1.0)) if 0.9 < ratio < 1.1 else prior
        if np.isfinite(gain) and 0.25 < abs(gain) < 4:
            mf = (mf / abs(gain)).astype(np.complex64)
            level = level * abs(gain)
    differential = mod in {"dbpsk","dqpsk","pi4-dqpsk"}
    tracking = cfg.use_costas and cfg.carrier_algorithm != "off" and not differential and not (mod == "oqpsk" and cfg.coarse_cfo)
    costas = cfg.carrier_algorithm == "costas" and base in {"bpsk","qpsk"}
    order = 2 if base == "bpsk" else 4
    carrier = None
    settle = []
    # Loop input offset (working samples into mf) and, when tracking, the
    # first symbol instant on the absolute symbol grid of the previous block.
    # Sign-rail M&M decisions are wrong for differential QPSK (axis states and
    # alternating pi/4 states), and the staggered OQPSK rails bias its error.
    # Gardner uses pulse transitions without a fixed constellation orientation.
    timing = "gardner" if mod in {"dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk"} and cfg.timing_recovery == "mueller-muller" else cfg.timing_recovery
    telemetry["timing_detector"] = timing
    offset = max(0, gd - sps) if timing == "gardner" else max(0, gd)
    position0 = phase0 = freq0 = None
    # Cold start: the loop's first symbol at the strongest sampling phase
    # (half a symbol of initial error takes a decision-directed loop on dense
    # QAM hundreds of symbols, if it converges at all).
    cold_start = float(gd + _strongest_phase(mf, gd, sps) - offset)  # mf as the loops see it (OQPSK realigned)
    if warm is not None and timing != "manual":
        earliest = start_sample + (2 * sps + offset) * per_work
        k = math.ceil((earliest - warm["next"]) / input_sps)
        first = warm["next"] + k * input_sps
        position0 = (first - start_sample) / per_work - offset
        elapsed = (first - warm["last"]) / input_sps
        phase0 = warm["phase"] if mod == "oqpsk" else warm["phase"] + elapsed * warm["freq"]
        freq0 = warm["freq"]
        # Refine the predicted carrier phase on the block's first symbols.
        m = {"bpsk": 2, "8psk": 8, "16apsk": 12, "32apsk": 16}.get(base, 4)
        moment = complex(np.mean(refs.astype(np.complex128) ** m))
        # After a gap of a few symbols the loop's own phase is better than any
        # block estimate: refine only across real gaps.
        if mod != "oqpsk" and elapsed >= WARM_REFINE_GAP and abs(moment) > 0.1 * float(np.mean(np.abs(refs.astype(np.complex128)) ** m)) and tracking:
            # QAM's fourth-power estimate needs more symbols for the same accuracy,
            # and APSK's comes from its inner ring alone (4 of 32-APSK's points):
            # 64 symbols left fewer than four on it in one block of 25.
            count = 256 if mod in {"16qam", "64qam", "16apsk", "32apsk"} else 64
            pos = position0 + offset + sps * np.arange(min(count, max(0, int((len(mf) - position0 - offset - 2) // sps))))
            if len(pos) >= 8:
                probe = np.interp(pos, np.arange(len(mf)), mf.real) + 1j * np.interp(pos, np.arange(len(mf)), mf.imag)
                # Removing the constellation's own M-th moment phase leaves M·θ, so the
                # estimate is the carrier rotation θ the loop derotates by.
                if mod in {"16qam", "64qam"}:
                    measured = blind_phase_search(probe, refs, center=phase0)
                elif mod in {"16apsk", "32apsk"}:
                    # The inner ring's fourth power has 90° branches, not the
                    # outer ring's 30° or 22.5°: far safer to resolve by prediction.
                    measured = _inner_ring_phase(probe, refs, phase0)
                else:
                    probe = probe * np.exp(-1j * np.angle(moment) / m)
                    measured = feedforward_phase(probe, m, phase0)
                # The prediction error is phase drift accumulated over a known number of
                # symbols: it measures the residual carrier frequency far more precisely
                # than one short block can, so later predictions (and the ambiguity
                # branch they select) stay right across long gaps.  It is weighed
                # against the loop's own averaging time: over a short gap the
                # estimate's noise (a fraction of a degree) would otherwise become a
                # frequency error of hundreds of hertz and throw a dense
                # constellation's loop out of lock.
                drift = (measured - phase0 + math.pi) % (2 * math.pi) - math.pi
                freq0 = freq0 + 0.5 * drift / (elapsed + WARM_AVERAGING_SYMBOLS)
                phase0 = measured
    if timing == "mueller-muller" and tracking:
        # Decision-directed timing needs carrier-corrected decisions.
        joint = mueller_muller_carrier(mf[offset:], sps, cfg.timing_gain, cfg.costas, actual_rs,
                                       order=order, refs=None if costas else refs,
                                       position0=cold_start if position0 is None else position0,
                                       phase0=phase0 or 0.0, freq0=freq0 or 0.0)
        tr, carrier = joint.timing, joint.carrier
    elif timing == "mueller-muller":
        tr = mueller_muller(mf[offset:], sps, cfg.timing_gain, position0=cold_start if position0 is None else position0)
    elif timing == "gardner":
        tr = gardner(mf[offset:], sps, cfg.timing_gain, cfg.timing_phase, position0=cold_start if position0 is None else position0)
    else:
        tr = None
        # Captures can begin between transmitter symbol instants (their length
        # is independent of samples/symbol).  Manual timing is relative to the
        # absolute symbol grid, not the left edge of each capture.
        alignment = ((-int(start_sample)) % input_sps) / per_work if start_sample is not None else 0.0
        positions = np.arange(max(0,gd+alignment+cfg.timing_phase*sps),len(mf)-1,sps)
        syms = np.interp(positions,np.arange(len(mf)),mf.real)+1j*np.interp(positions,np.arange(len(mf)),mf.imag)
    if tr is not None:
        syms = tr.output
        telemetry["timing_error"] = tr.error[:4000].astype(float).tolist()
        telemetry["timing_mu"] = tr.mu[:4000].astype(float).tolist()
        settle.append(0 if position0 is not None else timing_settling_symbols(cfg.timing_gain, timing, cfg.rolloff))
    syms = syms[:cfg.max_symbols]
    # Preserve a bounded view of the symbols before carrier recovery.  This is
    # used by the educational receiver UI to show the actual acquisition path,
    # not a synthetic before/after illustration.
    if syms.size:
        pre = (tr.output if carrier is not None else syms)[:2500]
        if carrier is not None:
            # The joint loop derotates while it samples; undo that for the
            # "before carrier recovery" view.
            pre = pre * np.exp(1j * carrier.phase[:len(pre)])
        telemetry["pre_costas_i"] = pre.real.astype(float).tolist()
        telemetry["pre_costas_q"] = pre.imag.astype(float).tolist()
    if tracking and syms.size and carrier is None:
        if costas:
            carrier = costas_loop(syms, actual_rs, cfg.costas.model_copy(update={"order": order}), phase0 or 0.0, freq0 or 0.0)
        else:
            carrier = decision_carrier(syms, actual_rs, refs, cfg.costas, phase0 or 0.0, freq0 or 0.0)
    if carrier is not None:
        syms = carrier.output[:len(syms)]
        telemetry["costas_freq_hz"] = carrier.freq_hz[:4000].astype(float).tolist()
        telemetry["costas_error"] = carrier.error[:4000].astype(float).tolist()
        telemetry["costas_phase_deg"] = np.rad2deg(carrier.phase[:4000]).astype(float).tolist()
        # Detector gain for unit-RMS symbols: Costas BPSK 1, Costas QPSK √2, DD 1.
        settle.append(0 if phase0 is not None else loop_settling_symbols(cfg.costas.alpha, cfg.costas.beta, np.sqrt(2.0) if costas and order == 4 else 1.0))
    if track is not None and start_sample is not None and tr is not None and syms.size:
        # Where the next block continues: the symbol instant after this block's
        # last one, the carrier loop's final phase and frequency.
        following = start_sample + (tr.next_position + offset) * per_work
        track.update(cfo=cfo, front_phase=front_phase, next=following, last=following - input_sps,
                     phase=carrier.final_phase if carrier is not None else 0.0,
                     freq=carrier.final_freq if carrier is not None else 0.0, scale=float(level))
    telemetry["tracking"] = position0 is not None
    if mod in {"dbpsk","dqpsk","pi4-dqpsk"} and len(syms):
        rotation = 0.0
        if cfg.carrier_algorithm != "off" and len(syms) >= 16:
            # Differential detection tolerates a residual frequency offset but
            # turns every product by 2πΔf/Rs; that rotation is removed.
            rotation = 2 * np.pi * differential_cfo(syms, actual_rs, mod) / actual_rs
        telemetry["differential_rotation_deg"] = float(np.rad2deg(rotation))
        syms=syms*np.conj(np.r_[1+0j,syms[:-1]])*np.exp(-1j*rotation)
        if mod == "dqpsk": syms*=np.exp(1j*np.pi/4)
        # At the start of an arbitrary later capture, the predecessor is
        # unavailable.  Capture zero (and standalone frame blocks) may use
        # their known initial differential state to decode every payload bit.
        if start_sample is not None and start_sample > 0:
            syms=syms[1:]
    if cfg.equalizer == "cma" and syms.size:
        telemetry["pre_equalizer_i"] = syms.real[:2500].tolist()
        telemetry["pre_equalizer_q"] = syms.imag[:2500].tolist()
        # A tracking receiver keeps its converged equalizer from block to block.
        previous = warm.get("cma") if warm is not None else None
        weights0 = previous if previous is not None and len(previous) == (int(cfg.equalizer_taps) | 1) else None
        syms, eq_error, weights = cma_equalizer(syms,cfg.equalizer_taps,cfg.equalizer_gain,weights0)
        if track is not None:
            track["cma"] = weights
        telemetry["equalizer_error"] = eq_error[:4000].tolist()
        telemetry["equalizer_taps"] = np.abs(weights).tolist()
        telemetry["equalizer_warm"] = weights0 is not None
    phase_rot = np.exp(-1j*np.deg2rad(cfg.manual_phase_deg))
    syms = syms * phase_rot
    _, bits = slice_symbols(syms, base, cfg.apsk_ratio2, cfg.apsk_ratio3)
    if cfg.equalizer == "cma" and not telemetry.get("equalizer_warm"):
        settle.append(int(np.ceil(1.0 / max(cfg.equalizer_gain, 1e-6))))
    # A vector signal analyzer measures EVM once its loops are synchronized.
    # The acquisition transient is excluded from the headline EVM (and still
    # reported, together with the all-symbol EVM, so nothing is hidden).
    settle_n = min(max(settle, default=0), syms.size // 2)
    tracked = syms[settle_n:]
    metrics = evm_rms(tracked, base, cfg.apsk_ratio2, cfg.apsk_ratio3)
    metrics.update({"samples_per_symbol":input_sps, "working_samples_per_symbol":sps,
                    "actual_symbol_rate":actual_rs, "symbol_count":int(syms.size), "capture_symbols":int(capture_symbols),
                    "evm_settle_symbols":int(settle_n),
                    "evm_all_percent":evm_rms(syms, base, cfg.apsk_ratio2, cfg.apsk_ratio3)["evm_rms_percent"]})
    # Symbol-resolution quality telemetry, similar to a vector signal analyzer's
    # error-vs-time views.  Keep it bounded before serializing to the browser.
    if syms.size:
        traces = symbol_errors(syms, refs, tracked)
        take = max(1, len(syms) // 2000)
        telemetry.update({key: value[::take].tolist() for key, value in traces.items()})
        telemetry["evm_settle_index"] = int(settle_n // take)
    text, metrics["text_bit_offset"], metrics["text_rotation_deg"] = _symmetric_text(syms, base, cfg)
    eyes = eye_segments(mf[max(0, gd-sps):], sps, traces=64)
    if eyes.size:
        telemetry["eye_i"] = eyes.real.astype(np.float32).tolist()
        telemetry["eye_q"] = eyes.imag.astype(np.float32).tolist()
    return RxResult(mf.astype(np.complex64), syms.astype(np.complex64), bits, text, metrics, telemetry)

def _fsk_timing(dphi: np.ndarray, sps: int) -> int:
    """Symbol boundary (0 … sps−1) that maximizes the spread of the per-symbol
    frequency averages: a window straddling two symbols averages their tones
    towards each other.  Non-data-aided, so it works on any payload."""
    candidates = range(0, sps, max(1, sps // 32))
    best, best_spread = 0, -1.0
    for start in candidates:
        m = (len(dphi) - start) // sps
        if m < 2:
            continue
        averages = dphi[start:start + m * sps].reshape(m, sps).mean(axis=1)
        spread = float(np.mean((averages - np.median(averages)) ** 2))
        if spread > best_spread:
            best, best_spread = start, spread
    return best


def _fsk_offset(averages: np.ndarray, centres: np.ndarray) -> float:
    """Frequency offset of FSK symbol averages from their nominal tones.

    Starts midway between the extreme tones actually seen (robust to a payload
    that does not use the tones equally), then alternates nearest-tone
    decisions with a least-squares offset."""
    if len(averages) < 4:
        return 0.0
    offset = 0.5 * float(np.percentile(averages, 2) + np.percentile(averages, 98))
    for _ in range(6):
        index = np.argmin(np.abs(averages[:, None] - offset - centres[None, :]), axis=1)
        new = float(np.mean(averages - centres[index]))
        if abs(new - offset) < 1e-9 * (abs(offset) + 1.0):
            break
        offset = new
    return offset


def _fsk(samples: np.ndarray, cfg: RxConfig, levels: int, msk: bool = False, start_sample: int | None = None) -> RxResult:
    """Discriminator FSK/MSK/GMSK demodulator.

    Symbol boundaries come from the transmitter's absolute symbol grid (manual
    timing, offset by ``timing_phase``) or from a non-data-aided search for
    the boundary that best separates the tones (any timing loop selected).
    With ``coarse_cfo`` the tones' common frequency offset is estimated and
    removed before the decisions.
    """
    fs = cfg.sample_rate
    sps = max(2, int(round(fs/cfg.symbol_rate)))
    actual_rs = fs / sps
    if len(samples) < 2:
        return RxResult(np.zeros(0, np.complex64), np.array([]), np.array([], dtype=np.uint8), "", {}, {})
    # The phase difference between samples k-1 and k measures the frequency of
    # sample k, so index the discriminator by k: dphi[k] <-> sample k.  (Indexing
    # it by k-1 shifted every symbol average by one sample, mixing 1/sps of the
    # next symbol into each decision: 25 % noise-free BER at 2 samples/symbol.)
    d = np.angle(samples[1:]*np.conj(samples[:-1])) * fs/(2*np.pi)
    dphi = np.r_[d[:1], d]
    if cfg.timing_recovery == "manual":
        # A capture can begin anywhere in a symbol: align to the absolute grid.
        grid = (-int(start_sample)) % sps if start_sample is not None else 0
        start = (grid + int(round(cfg.timing_phase*sps))) % sps
    else:
        start = _fsk_timing(dphi, sps)
    m = (len(dphi)-start)//sps
    if m <= 0:
        return RxResult(dphi.astype(np.complex64), np.array([]), np.array([],dtype=np.uint8), "", {}, {})
    freq_sym = dphi[start:start+m*sps].reshape(m,sps).mean(axis=1)
    if msk:
        levels = 2
        centres = np.array([-actual_rs / 4, actual_rs / 4])
    else:
        centres = np.linspace(-cfg.fsk_deviation_hz, cfg.fsk_deviation_hz, levels)
    offset = _fsk_offset(freq_sym, centres) if cfg.coarse_cfo else 0.0
    if msk:
        idx = (freq_sym - offset > 0).astype(int)
    else:
        idx = np.argmin(np.abs(freq_sym[:, None] - offset - centres[None, :]), axis=1)
    k = int(round(np.log2(levels)))
    shifts = np.arange(k-1,-1,-1)
    bits = ((idx[:,None]>>shifts)&1).astype(np.uint8).reshape(-1)
    text, text_offset = decode_text(bits)
    return RxResult(dphi.astype(np.complex64), (freq_sym - offset).astype(np.complex64), bits, text,
                    {"samples_per_symbol":sps,"actual_symbol_rate":actual_rs,"text_bit_offset":text_offset,
                     "symbol_count":int(m),"frequency_offset_hz":float(offset),"symbol_boundary_sample":int(start)},
                    {"coarse_cfo_hz":float(offset)})


def _ofdm(samples: np.ndarray, cfg: RxConfig) -> RxResult:
    from .ofdm import receive
    syms,bits,metrics,telemetry=receive(samples[cfg.ofdm_timing_offset:],cfg.sample_rate,cfg.ofdm_fft,cfg.ofdm_cp,cfg.ofdm_occupied,cfg.ofdm_design,cfg.coarse_cfo,cfg.ofdm_start_s)
    if telemetry.get('ofdm_blind_phase') and len(syms):
        # Without pilots nothing fixes the 90° rotation: the text view tries all four.
        text,metrics['text_bit_offset'],metrics['text_rotation_deg']=_symmetric_text(syms,'qpsk',cfg)
    else:
        text,metrics['text_bit_offset']=decode_text(bits)
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

def demodulate(samples: np.ndarray, cfg: RxConfig, start_sample: int | None = None, track: dict | None = None) -> RxResult:
    """Demodulate one block.  ``start_sample`` (the block's absolute first
    sample) and ``track`` (state kept between blocks) let linear receivers keep
    tracking across blocks, including blocks separated by dropped samples."""
    x=np.asarray(samples,dtype=np.complex64)
    if cfg.tuner_hz:
        x=x*phasor(-cfg.tuner_hz,cfg.sample_rate,len(x),int(start_sample or 0)).astype(np.complex64)
    if cfg.modulation in {"bpsk","qpsk","8psk","16qam","64qam","dbpsk","dqpsk","pi4-dqpsk","oqpsk","16apsk","32apsk"}:
        previous = track.get("evm") if track is not None else None
        result = _digital(x,cfg,start_sample,track)
        evm = result.metrics.get("evm_rms_percent")
        if result.telemetry.get("tracking") and previous is not None and evm is not None and evm > max(2.0 * previous + 1.0, 6.0):
            # A tracked block much worse than its predecessor lost lock (a
            # long gap or a phase slip): acquire it from scratch instead.
            track.clear()
            result = _digital(x,cfg,start_sample,track)
            result.telemetry["reacquired"] = True
            evm = result.metrics.get("evm_rms_percent")
        if track is not None and evm is not None:
            track["evm"] = float(evm)
        return result
    if cfg.modulation=="2fsk":
        return _fsk(x,cfg,2,start_sample=start_sample)
    if cfg.modulation=="4fsk":
        return _fsk(x,cfg,4,start_sample=start_sample)
    if cfg.modulation in {"msk","gmsk"}:
        return _fsk(x,cfg,2,msk=True,start_sample=start_sample)
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
        segment = x[a:b]
        linear = block.modulation in {"bpsk","qpsk","dbpsk","dqpsk","pi4-dqpsk","oqpsk","8psk","16qam","64qam","16apsk","32apsk"}
        if linear and cfg.pulse_shape == "rrc":
            # Frame blocks are pulse-shaped with a centred (zero-delay) RRC so
            # block boundaries stay exact, whereas ``demodulate`` expects the
            # causal transmit-filter delay of a continuous stream.  Supplying
            # that many samples of context before (and after) the block puts
            # symbol 0 where the demodulator samples it; without it every
            # linear block lost its first filter_span/2 symbols.
            sps = max(2, int(round(cfg.sample_rate / rate)))
            lead = (len(rrc_taps(sps, cfg.rolloff, cfg.filter_span_symbols)) - 1) // 2
            lo, hi = a - lead, b + lead
            segment = np.concatenate([np.zeros(max(0, -lo), np.complex64), x[max(0, lo):min(len(x), hi)],
                                      np.zeros(max(0, hi - len(x)), np.complex64)])
        try:
            result = demodulate(segment, block_cfg)
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


def channel_symbols(result: RxResult) -> np.ndarray | None:
    """The receiver's symbols before any equalizer (what the channel produced)."""
    t = result.telemetry
    if t.get("pre_equalizer_i") is not None:
        return np.asarray(t["pre_equalizer_i"], dtype=np.float64) + 1j * np.asarray(t["pre_equalizer_q"], dtype=np.float64)
    return result.symbols


def reference_channel_estimate(reference: np.ndarray, measured: np.ndarray, taps: int = 5, search: int = 64,
                               limit: int = 4096) -> dict | None:
    """Symbol-spaced least-squares channel taps from known transmitted symbols.

    The receiver's first decision need not be the capture's first transmitted
    symbol (the capture starts mid-stream and the loops settle), so the
    reference is first aligned by the lag (±``search`` symbols) that maximizes
    the cross-correlation magnitude.  Returns ``None`` when there is too little
    overlap to fit.
    """
    ref = np.asarray(reference, dtype=np.complex128)[:limit + search]
    meas = np.asarray(measured, dtype=np.complex128)[:limit]
    if len(ref) < 64 or len(meas) < 64:
        return None
    best_lag, best = 0, -1.0
    for lag in range(-search, search + 1):
        a, b = (ref[lag:], meas) if lag >= 0 else (ref, meas[-lag:])
        m = min(len(a), len(b))
        if m < 64:
            continue
        value = abs(np.vdot(a[:m], b[:m])) / m
        if value > best:
            best_lag, best = lag, value
    a, b = (ref[best_lag:], meas) if best_lag >= 0 else (ref, meas[-best_lag:])
    m = min(len(a), len(b))
    if m < 64:
        return None
    from scipy.linalg import convolution_matrix
    A = convolution_matrix(a[:m], taps, mode="full")[:m]
    h, *_ = np.linalg.lstsq(A[32:], b[32:m], rcond=None)
    return {"channel_estimate_taps": np.abs(h).astype(float).tolist(),
            "channel_estimate_phase_deg": np.rad2deg(np.angle(h)).astype(float).tolist(),
            "channel_estimate_lag_symbols": int(best_lag),
            "channel_estimate_kind": "reference-aided, correlation-aligned 5-tap least squares"}
