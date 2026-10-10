"""Receiver acquisition across rate, modulation, phase, and carrier offset.

The windows at 100 MS/s are intentionally short, like real-time SDR buffers.
"""
import json

import numpy as np
import pytest

from vectorrf.engine import MeasurementSpec, measurement_product
from vectorrf.measurements import zero_span
from vectorrf.models import ChannelConfig, RxConfig, TxConfig
from vectorrf.modem import evm_rms
from vectorrf.rx import demodulate
from vectorrf.source import source_record
from vectorrf.sync import apply_channel


LINEAR = [
    "bpsk", "qpsk", "8psk", "16qam", "64qam", "16apsk", "32apsk",
    "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk",
]


@pytest.mark.parametrize("rate", [1e6, 10e6, 100e6])
@pytest.mark.parametrize("modulation", LINEAR)
def test_linear_acquisition_matrix(modulation, rate):
    tx = TxConfig(waveform=modulation, sample_rate=rate, symbol_rate=50e3,
                  duration=min(0.02, 524288 / rate), snr_db=None)
    rx = RxConfig(modulation=modulation, sample_rate=rate, symbol_rate=50e3,
                  timing_recovery="mueller-muller", coarse_cfo=True)
    capture = source_record(tx, 0)
    for cfo, phase in [(0, 0), (300, 0), (0, 20), (300, 20), (-300, -20)]:
        impaired = apply_channel(capture.samples, rate,
            ChannelConfig(snr_db=None, freq_offset_hz=cfo, phase_offset_deg=phase))
        received = demodulate(impaired, rx)
        assert received.metrics["symbol_count"] >= 100
        assert received.metrics["evm_rms_percent"] < 3.5, (modulation, rate, cfo, phase, received.metrics)
        assert abs(received.telemetry["coarse_cfo_hz"] - cfo) < 25
        if modulation in {"dqpsk", "pi4-dqpsk", "oqpsk"}:
            assert received.telemetry["timing_detector"] == "gardner"


@pytest.mark.parametrize("modulation", ["qpsk", "16qam", "64qam", "16apsk", "32apsk",
                                         "dqpsk", "pi4-dqpsk", "oqpsk"])
def test_tracking_through_contiguous_blocks_and_dropped_buffers(modulation):
    rate = 10e6
    tx = TxConfig(waveform=modulation, sample_rate=rate, symbol_rate=50e3,
                  duration=0.02, snr_db=None)
    rx = RxConfig(modulation=modulation, sample_rate=rate, symbol_rate=50e3,
                  timing_recovery="mueller-muller", coarse_cfo=True)
    track = {}
    for index, capture_id in enumerate([0, 1, 3, 4]):
        capture = source_record(tx, capture_id, limit=40_000)
        start = capture.metadata["start_sample"]
        impaired = apply_channel(capture.samples, rate,
            ChannelConfig(snr_db=None, freq_offset_hz=300, phase_offset_deg=20),
            start_sample=start)
        received = demodulate(impaired, rx, start, track)
        assert received.telemetry["tracking"] is (index > 0)
        assert abs(received.telemetry["coarse_cfo_hz"] - 300) < 2
        assert received.metrics["evm_rms_percent"] < 3.5


def test_short_high_rate_quality_uses_absolute_symbol_clock_and_valid_json():
    rate = 5e9
    tx = TxConfig(waveform="qpsk", sample_rate=rate, symbol_rate=50e6,
                  duration=524288 / rate, snr_db=None)
    capture = source_record(tx, 7, limit=111_331)  # capture begins between symbols
    quality = measurement_product(capture, tx, False, 7,
                                  MeasurementSpec(quality=True, duration_s=0.002))
    assert quality["quality"]["kind"] == "evm"
    assert quality["quality"]["metrics"]["evm_rms_percent"] < 3
    assert quality["capture_duration_s"] == pytest.approx(len(capture.samples) / rate)  # the whole capture
    json.dumps(quality, allow_nan=False)


def test_zero_span_never_claims_out_of_window_samples():
    samples = np.ones(8192, dtype=np.complex64)
    with pytest.raises(ValueError, match="outside"):
        zero_span(samples, 5e9, offset=0.001)
    tx = TxConfig(waveform="tone", sample_rate=5e9, duration=8192 / 5e9)
    capture = source_record(tx, 2)
    live = measurement_product(capture, tx, False, 2,
                               MeasurementSpec(offset_s=0.001, duration_s=0.002))
    assert live["interval_clipped"]
    assert live["capture_start_s"] == 0
    assert live["capture_duration_s"] == pytest.approx(len(capture.samples) / 5e9)


def test_zero_span_rejects_tuning_that_would_alias_at_nyquist():
    samples = np.ones(8192, dtype=np.complex64)
    with pytest.raises(ValueError, match="Nyquist"):
        zero_span(samples, 1e6, center_hz=500e3)
    with pytest.raises(ValueError, match="Nyquist"):
        zero_span(samples, 1e6, center_hz=450e3, bandwidth_hz=200e3)


def test_wideband_zero_span_resolves_requested_band_without_an_enormous_fir():
    rate, count = 5e9, 524_288
    k = np.arange(count, dtype=np.float64)
    wanted = 0.5 * np.exp(2j * np.pi * 500e6 * k / rate)
    interferer = 0.5 * np.exp(2j * np.pi * 501e6 * k / rate)
    measured = zero_span((wanted + interferer).astype(np.complex64), rate,
                         duration=count / rate, center_hz=500e6, bandwidth_hz=100e3)
    assert rate / measured["analysis_rate"] > 1000
    assert measured["bandwidth_actual_hz"] == 100e3
    assert np.median(measured["amplitude"]) == pytest.approx(0.5, rel=0.02)
    assert abs(np.nanmedian(measured["frequency_hz"][10:-10])) < 20
    too_narrow = zero_span(wanted.astype(np.complex64), rate,
                           duration=count / rate, center_hz=500e6, bandwidth_hz=1)
    assert too_narrow["bandwidth_actual_hz"] >= 4 * rate / count


def test_zero_span_tuning_phase_uses_absolute_capture_position():
    rate, center, count = 1e6, 125e3, 8192
    left = np.exp(2j * np.pi * center * np.arange(count) / rate)
    right = np.exp(2j * np.pi * center * (count + np.arange(count)) / rate)
    a = zero_span(left, rate, duration=count / rate, center_hz=center, start_sample=0)
    b = zero_span(right, rate, duration=count / rate, center_hz=center, start_sample=count)
    np.testing.assert_allclose(a["i"], b["i"], atol=1e-6)
    np.testing.assert_allclose(a["q"], b["q"], atol=1e-6)


def test_zero_span_does_not_invent_frequency_after_a_signal_null():
    x = np.ones(64, dtype=np.complex64)
    x[20] = 0
    x[21:] = 1j
    result = zero_span(x, 1000, duration=len(x) / 1000, full_band=True)
    f, phase = np.asarray(result["frequency_hz"], float), np.asarray(result["phase_deg"], float)
    assert np.isnan(f[20]) and np.isnan(f[21]) and np.isnan(phase[20])
    assert f[22] == 0 and phase[22] == pytest.approx(90)


def test_basic_quality_masks_phase_and_frequency_during_pulse_off_time():
    tx = TxConfig(waveform="pulse", sample_rate=1e6, duration=0.01,
                  pulse_width_s=100e-6, pri_s=1000e-6, snr_db=None)
    capture = source_record(tx, 0)
    quality = measurement_product(capture, tx, False, 0,
                                  MeasurementSpec(quality=True, duration_s=0.01))["quality"]
    assert quality["kind"] == "basic"
    telemetry = quality["telemetry"]
    for key in ("phase_deg", "frequency_hz"):   # undefined while the pulse is off: gaps
        assert any(v is None for v in telemetry[key]) and not all(v is None for v in telemetry[key])
    assert quality["metrics"]["mean_frequency_hz"] == pytest.approx(0, abs=1)
    json.dumps(quality, allow_nan=False)


def test_empty_evm_is_unavailable_not_a_nan_token():
    metrics = evm_rms(np.array([], dtype=np.complex64), "qpsk")
    assert metrics["evm_rms_percent"] is None
    json.dumps(metrics, allow_nan=False)
