"""The transmitter is one stream indexed by absolute sample.

``generate(cfg, start_sample=a)`` must return exactly x[a : a + N] of a single
stream x that does not depend on how it is cut into captures: consecutive
captures join without a seam, captures of different lengths agree wherever
they overlap, and this holds with noise, impairments, hopping, bursts and
composites.  Calibration follows the design, not the capture.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vectorrf.models import TxConfig, WaveformKind
from vectorrf.rx import demodulate
from vectorrf.models import RxConfig
from vectorrf.signal import capture_samples, generate, reference_power

FS = 1e6
WAVEFORMS = [w.value for w in WaveformKind if w.value != "composite"]


def design(waveform: str, n: int, **extra) -> TxConfig:
    base = dict(waveform=waveform, sample_rate=FS, duration=n / FS, snr_db=None, symbol_rate=37_000,
                data_source="pn", chirp_period_s=0.004321)
    base.update(extra)
    return TxConfig(**base)


def stream(cfg: TxConfig, start: int, count: int) -> np.ndarray:
    """``count`` samples from ``start`` through captures of ``cfg``'s own length."""
    n = capture_samples(cfg)
    first, last = start // n, (start + count - 1) // n
    joined = np.concatenate([generate(cfg, start_sample=k * n, frame=k).samples for k in range(first, last + 1)])
    return joined[start - first * n:start - first * n + count]


def assert_same(a: np.ndarray, b: np.ndarray, what: str):
    scale = max(float(np.max(np.abs(b))), 1e-12)
    error = float(np.max(np.abs(a.astype(np.complex128) - b.astype(np.complex128)))) / scale
    assert error < 2e-6, f"{what}: max relative difference {error:.2e}"


@pytest.mark.parametrize("waveform", WAVEFORMS)
def test_consecutive_captures_join_exactly(waveform):
    n = 3001
    short, long = design(waveform, n), design(waveform, 2 * n)
    for k in (0, 5):
        joined = np.concatenate([generate(short, start_sample=k * n).samples, generate(short, start_sample=(k + 1) * n).samples])
        assert_same(joined, generate(long, start_sample=k * n).samples, f"{waveform} captures {k},{k + 1}")


@pytest.mark.parametrize("waveform", WAVEFORMS)
def test_capture_length_does_not_change_the_stream(waveform):
    """Captures of 1 024 and 4 999 samples agree on every sample they share (with noise)."""
    a, b = design(waveform, 1024, snr_db=12), design(waveform, 4999, snr_db=12)
    start, count = 7 * 4999 + 13, 3000
    assert_same(stream(a, start, count), stream(b, start, count), waveform)


@pytest.mark.parametrize("rate,sps", [(2e6, 2), (10e6, 7), (100e6, 2000)])
@pytest.mark.parametrize("waveform", ["qpsk", "16qam", "oqpsk", "pi4-dqpsk", "gmsk", "4fsk"])
def test_digital_streams_are_continuous_at_any_oversampling(waveform, rate, sps):
    n = 4 * sps + 333
    cfg = TxConfig(waveform=waveform, snr_db=None, data_source="pn", sample_rate=rate, symbol_rate=rate / sps,
                   fsk_deviation_hz=rate / sps / 2, duration=n / rate)
    long = cfg.model_copy(update={"duration": 3 * n / rate})
    joined = np.concatenate([generate(cfg, start_sample=(10 + k) * n).samples for k in range(3)])
    assert_same(joined, generate(long, start_sample=10 * n).samples, f"{waveform} at {sps} samples/symbol")


def test_impairments_hopping_bursts_and_composites_are_continuous():
    n = 2500
    base = dict(sample_rate=FS, snr_db=20, iq_gain_db=1.5, iq_phase_deg=4, dc_i=0.02, burst_enabled=True,
                burst_period_s=0.0011, burst_duty=0.6, center_hz=12_345,
                agility={"enabled": True, "mode": "hop", "plan": "random", "start_hz": -200e3, "stop_hz": 200e3,
                         "step_hz": 25e3, "dwell_s": 0.00037, "off_s": 0.00005, "ramp_s": 0.00002})
    components = [dict(waveform="qpsk", symbol_rate=50e3, power_dbm=-3, center_hz=-150e3, label="A"),
                  dict(waveform="2fsk", symbol_rate=20e3, fsk_deviation_hz=10e3, power_dbm=-10, center_hz=100e3,
                       burst_enabled=True, burst_period_s=0.0007, burst_duty=0.3, label="B"),
                  dict(waveform="multitone", tones_hz=[-20e3, 33e3], power_dbm=-20, label="C")]
    for cfg in (TxConfig(waveform="16qam", symbol_rate=40e3, duration=n / FS, **base),
                TxConfig(waveform="composite", components=components, duration=n / FS, **{**base, "center_hz": 0})):
        long = cfg.model_copy(update={"duration": 2 * n / FS})
        joined = np.concatenate([generate(cfg, start_sample=9 * n).samples, generate(cfg, start_sample=10 * n).samples])
        assert_same(joined, generate(long, start_sample=9 * n).samples, cfg.waveform.value)


def test_noise_is_white_gaussian_at_the_configured_snr():
    cfg = TxConfig(waveform="tone", sample_rate=FS, duration=0.2, snr_db=10)
    clean = generate(cfg.model_copy(update={"snr_db": None})).samples.astype(np.complex128)
    noise = generate(cfg).samples.astype(np.complex128) - clean
    snr = 10 * math.log10(np.mean(np.abs(clean) ** 2) / np.mean(np.abs(noise) ** 2))
    assert snr == pytest.approx(10, abs=0.05)
    assert abs(np.mean(noise.real * noise.imag)) < 0.01 * np.mean(np.abs(noise) ** 2)
    assert abs(np.vdot(noise[1:], noise[:-1])) / len(noise) < 0.01 * np.mean(np.abs(noise) ** 2)


@pytest.mark.parametrize("waveform,extra", [
    ("tone", {}), ("multitone", {"tones_hz": [-100e3, 20e3, 20e3, 250e3]}), ("awgn", {}), ("awgn", {"noise_bandwidth_hz": 100e3}),
    ("chirp", {}), ("qpsk", {}), ("16qam", {}), ("64qam", {}), ("32apsk", {}), ("dqpsk", {}), ("qpsk", {"pulse_shape": "rect"}),
    ("2fsk", {}), ("gmsk", {}), ("ofdm", {}), ("am", {"modulation_index": 0.8}), ("dsb-sc", {}), ("ssb", {}), ("fm", {}),
])
def test_long_run_power_equals_the_configured_power(waveform, extra):
    cfg = TxConfig(waveform=waveform, sample_rate=FS, duration=0.25, snr_db=None, power_dbm=-7, data_source="pn", **extra)
    x = generate(cfg, start_sample=123_457).samples.astype(np.complex128)
    reference = generate(cfg).metadata["power_reference_rms"] ** 2
    tolerance = 0.1 if waveform != "awgn" else 0.05
    assert 10 * math.log10(np.mean(np.abs(x) ** 2) / reference) == pytest.approx(0, abs=tolerance)


@pytest.mark.parametrize("waveform", ["pulse", "lfm", "barker", "frank", "costas", "stepped-frequency"])
def test_pulses_read_the_configured_power_while_on_in_every_capture(waveform):
    """Power refers to the pulse; a capture's pulse count must not rescale it."""
    cfg = TxConfig(waveform=waveform, sample_rate=FS, duration=0.00033, pulse_width_s=40e-6, pri_s=150e-6, snr_db=None,
                   sweep_bw_hz=200e3, pulse_envelope="rect")
    peaks = []
    for k in range(6):
        x = generate(cfg, start_sample=k * capture_samples(cfg)).samples
        on = np.abs(x) > 1e-6
        if on.any():
            peaks.append(float(np.mean(np.abs(x[on]) ** 2)))
    reference = generate(cfg).metadata["power_reference_rms"] ** 2
    assert np.allclose(peaks, reference, rtol=1e-5), (peaks, reference)


def test_a_pulse_cut_by_a_capture_edge_is_part_of_the_full_pulse():
    cfg = TxConfig(waveform="lfm", sample_rate=FS, duration=150e-6 + 25e-6, pulse_width_s=60e-6, pri_s=150e-6, snr_db=None,
                   sweep_bw_hz=300e3)
    long = cfg.model_copy(update={"duration": 900e-6})
    edge = generate(cfg, start_sample=0).samples        # ends 25 µs into the second pulse
    assert_same(edge, generate(long).samples[:len(edge)], "truncated pulse")


def test_jittered_pri_is_reproducible_and_bounded():
    cfg = TxConfig(waveform="pulse", sample_rate=FS, duration=0.02, pulse_width_s=10e-6, pri_s=100e-6, pri_mode="jittered",
                   pri_jitter_fraction=0.3, snr_db=None)
    x = np.abs(generate(cfg).samples) > 0.5
    starts = np.flatnonzero(x[1:] & ~x[:-1]) + 1
    pri = np.diff(starts)
    assert np.all(pri >= 100 * 0.7 - 1) and np.all(pri <= 100 * 1.3 + 1)
    assert np.std(pri) > 3
    assert np.array_equal(generate(cfg).samples, generate(cfg).samples)


def test_chirp_period_defaults_to_the_capture_and_can_be_pinned():
    one = TxConfig(waveform="chirp", sample_rate=FS, duration=0.002, snr_db=None)
    x = generate(one).samples
    freq = np.angle(x[1:] * np.conj(x[:-1])) * FS / (2 * np.pi)
    # 200 kHz in 2000 samples: 100 Hz per sample, start → stop over the one-capture period.
    assert freq[0] == pytest.approx(-99_950, abs=1) and freq[-1] == pytest.approx(99_850, abs=1)
    pinned = one.model_copy(update={"chirp_period_s": 0.001})
    y = generate(pinned).samples
    assert_same(y[:1000], y[1000:2000], "pinned chirp period")


def test_text_decodes_from_a_capture_that_starts_mid_stream():
    cfg = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3, duration=0.03, snr_db=None, message="HELLO VECTORRF ")
    g = generate(cfg, start_sample=17 * capture_samples(cfg) + 333)
    r = demodulate(g.samples, RxConfig(modulation="qpsk", sample_rate=1e6, symbol_rate=50e3, timing_recovery="mueller-muller"))
    assert "HELLO VECTORRF" in r.text, r.text[:60]
    assert 0 <= r.metrics["text_bit_offset"] < 8


def test_same_capture_is_identical_and_designs_differ_by_seed():
    cfg = TxConfig(waveform="qpsk", snr_db=15, data_source="pn")
    a, b = generate(cfg, start_sample=50_000), generate(cfg, start_sample=50_000)
    assert np.array_equal(a.samples, b.samples)
    c = generate(cfg.model_copy(update={"seed": cfg.seed + 1}), start_sample=50_000)
    assert not np.allclose(a.samples, c.samples)


def test_reference_power_is_a_property_of_the_design():
    cfg = TxConfig(waveform="custom", sample_rate=FS, duration=0.0013, snr_db=None)
    assert reference_power(cfg) == reference_power(cfg.model_copy(update={"duration": 0.0007, "power_dbm": 9, "center_hz": 1e3}))


def test_sawtooth_chirp_is_phase_continuous_through_the_flyback():
    fs = 1e6
    cfg = TxConfig(waveform="chirp", sample_rate=fs, duration=0.01, chirp_start_hz=-100e3, chirp_stop_hz=100e3,
                   chirp_period_s=1e-3, snr_db=None)
    x = generate(cfg).samples.astype(np.complex128)
    step = np.angle(x[1:] * np.conj(x[:-1])) * fs / (2 * np.pi)
    # Frequency ramps over each period and jumps back; the phase has no step of its own.
    assert step[0] == pytest.approx(-100e3 + 100.0, abs=1.0) and step[998] == pytest.approx(99.7e3, abs=1.0)
    assert step[999] == pytest.approx(99.9e3, abs=1.0) and step[1000] == pytest.approx(-99.9e3, abs=1.0)
    y = generate(cfg.model_copy(update={"duration": 0.004}), start_sample=3000).samples
    np.testing.assert_array_equal(y, x[3000:7000].astype(np.complex64))
