"""Receiver acquisition and tracking over the whole carrier-offset range of the UI.

The live receiver (``engine.receiver_product``) is run over consecutive and
skipped captures with a realistic channel: 30 dB SNR, a 20° phase, a 0.15
sample timing offset and carrier offsets up to ±5 kHz (±10 % of 50 kBd).
Text payloads matter as much as PRBS: a short repeating text uses a few
constellation points, puts false lines into M-th power spectra and biases
RMS-based scale estimates, which used to make 16/64-QAM and APSK lock wrongly.
"""
import json

import numpy as np
import pytest

from vectorrf.engine import MeasurementSpec, measurement_product, receiver_product
from vectorrf.models import ChannelConfig, RxConfig, TxConfig
from vectorrf.pipeline import Pipeline
from vectorrf.source import acquire

LINEAR = ["bpsk", "qpsk", "8psk", "16qam", "64qam", "16apsk", "32apsk", "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk"]
CHANNEL = dict(snr_db=30, phase_offset_deg=20, timing_offset_samples=0.15)


def run(modulation, cfo, data, rate=1e6, captures=(0, 1, 2, 4), **rx):
    tx = TxConfig(waveform=modulation, sample_rate=rate, symbol_rate=50e3, duration=min(524288, rate / 30) / rate,
                  snr_db=None, data_source=data)
    receiver = RxConfig(modulation=modulation, sample_rate=rate, symbol_rate=50e3, timing_recovery="mueller-muller",
                        coarse_cfo=True, fsk_deviation_hz=tx.fsk_deviation_hz, **rx)
    channel = ChannelConfig(freq_offset_hz=cfo, **CHANNEL)
    memory = {}
    return [receiver_product(acquire(tx, None, c), channel, Pipeline(), receiver, c, memory) for c in captures]


@pytest.mark.parametrize("data", ["pn", "text"])
@pytest.mark.parametrize("cfo", [-5000.0, 2500.0, 5000.0])
@pytest.mark.parametrize("modulation", LINEAR)
def test_linear_receivers_lock_on_every_capture(modulation, cfo, data):
    for product in run(modulation, cfo, data):
        evm = product["metrics"]["evm_rms_percent"]
        assert evm is not None and evm < 3.0, (modulation, cfo, data, product["capture_id"], evm)


@pytest.mark.parametrize("modulation", ["16qam", "64qam", "16apsk", "32apsk"])
def test_dense_constellations_lock_cold_on_a_repeating_text(modulation):
    """Every capture acquired from scratch (no tracking state) still locks."""
    tx = TxConfig(waveform=modulation, sample_rate=10e6, symbol_rate=50e3, duration=333333 / 10e6, snr_db=None)
    receiver = RxConfig(modulation=modulation, sample_rate=10e6, symbol_rate=50e3, timing_recovery="mueller-muller", coarse_cfo=True)
    channel = ChannelConfig(freq_offset_hz=-2500, **CHANNEL)
    for capture in (1, 2, 4, 7):
        product = receiver_product(acquire(tx, None, capture), channel, Pipeline(), receiver, capture, None)
        assert product["metrics"]["evm_rms_percent"] < 3.0, (modulation, capture, product["metrics"]["evm_rms_percent"])


@pytest.mark.parametrize("modulation", ["2fsk", "4fsk", "msk", "gmsk"])
@pytest.mark.parametrize("cfo", [0.0, 2000.0, -6000.0])
def test_cpfsk_decodes_captures_that_start_mid_symbol(modulation, cfo):
    """Captures of 33,333 samples start anywhere in a 20-sample symbol."""
    for product in run(modulation, cfo, "text", captures=(0, 1, 3)):
        assert "VECTORRF LAB" in product["text"], (modulation, cfo, product["capture_id"], product["text"][:40])


def test_pilotless_ofdm_follows_the_carrier_phase():
    for product in run("ofdm", 2000.0, "text", captures=(0, 1, 3)):
        assert "VECTORRF LAB" in product["text"], product["text"][:40]


def test_a_capture_shorter_than_a_symbol_gives_valid_json():
    """2 GS/s at 50 kBd: 94,000 samples are two symbols, fewer than the filters' transient."""
    tx = TxConfig(waveform="qpsk", sample_rate=2e9, symbol_rate=50e3, duration=524288 / 2e9, snr_db=30)
    receiver = RxConfig(modulation="qpsk", sample_rate=2e9, symbol_rate=50e3, timing_recovery="mueller-muller", coarse_cfo=True)
    product = receiver_product(acquire(tx, None, 3, 94_000), ChannelConfig(**CHANNEL), Pipeline(), receiver, 3, {})
    json.dumps(product, allow_nan=False)
    assert product["metrics"]["evm_rms_percent"] is None


def test_extreme_oversampling_estimates_the_carrier_on_decimated_symbols():
    """At 40,000 samples per symbol the first 131,072 input samples hold three
    symbols: the carrier estimates run after decimation, on every symbol."""
    rate = 2e9
    samples = int((2 * 16 + 32) * rate / 50e3)
    tx = TxConfig(waveform="qpsk", sample_rate=rate, symbol_rate=50e3, duration=samples / rate, snr_db=40, data_source="pn",
                  filter_span_symbols=16)
    memory = {}
    for capture in (10, 90, 170, 250):
        quality = measurement_product(acquire(tx, None, capture), tx, False, capture, MeasurementSpec(quality=True), memory)["quality"]
    assert quality["kind"] == "evm"
    assert quality["metrics"]["evm_rms_percent"] < 6
    assert quality["metrics"]["symbol_count"] > 100  # accumulated over captures


def test_quality_accumulates_tracked_symbols_and_stays_small():
    tx = TxConfig(waveform="16qam", sample_rate=1e6, symbol_rate=50e3, duration=33333 / 1e6, snr_db=35, data_source="pn")
    memory = {}
    for capture in range(4):
        out = measurement_product(acquire(tx, None, capture), tx, False, capture, MeasurementSpec(quality=True), memory, digits=6)
    quality = out["quality"]
    assert quality["metrics"]["captures"] == 4 and quality["metrics"]["symbol_count"] == 2048
    assert quality["metrics"]["evm_rms_percent"] < 3
    assert len(json.dumps(out)) < 150_000  # the quality page used to receive ~600 kB per frame
    assert "time_s" not in out  # zero-span traces are not sent to the quality page


def test_equalizer_page_gets_the_channel_estimate_and_a_converging_cma():
    tx = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3, duration=0.0333, snr_db=None, data_source="pn")
    channel = ChannelConfig(freq_offset_hz=300, echo_gain=0.3, echo_delay_symbols=2, **CHANNEL)
    receiver = RxConfig(modulation="qpsk", sample_rate=1e6, symbol_rate=50e3, timing_recovery="mueller-muller",
                        coarse_cfo=True, equalizer="cma", equalizer_gain=0.005)
    memory, evm = {}, []
    for capture in range(6):
        product = receiver_product(acquire(tx, None, capture), channel, Pipeline(), receiver, capture, memory)
        evm.append(product["metrics"]["evm_rms_percent"])
    taps = product["telemetry"]["channel_estimate_taps"]
    assert taps[2] / taps[0] == pytest.approx(0.3, abs=0.05)  # the echo, two symbols late
    assert max(taps[1], taps[3], taps[4]) < 0.05 * taps[0]
    assert evm[-1] < 8 and evm[-1] < 0.5 * evm[0]  # the equalizer keeps converging from capture to capture
    assert product["telemetry"]["equalizer_taps"][2] == pytest.approx(0.3, abs=0.06)
