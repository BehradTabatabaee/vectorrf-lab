import numpy as np

from vectorrf.models import CostasConfig, RxConfig, SpectrumConfig, TxConfig
from vectorrf.rx import demodulate
from vectorrf.signal import generate
from vectorrf.spectrum import analyze
from vectorrf.sync import apply_channel
from vectorrf.models import ChannelConfig


def test_spectrum_finds_tone():
    cfg = TxConfig(waveform="tone", sample_rate=500_000, duration=0.03, center_hz=73_000, snr_db=45)
    g = generate(cfg)
    s = analyze(g.samples, g.sample_rate, SpectrumConfig(rbw_hz=1000, detector="average"))
    assert abs(s.peak_hz - 73_000) < 2_000
    assert s.rbw_actual_hz < 2_000


def test_qpsk_loopback_recovers_message_without_impairment():
    tx = TxConfig(waveform="qpsk", sample_rate=500_000, symbol_rate=25_000, duration=.08,
                  message="HELLO VECTORRF ", snr_db=40)
    g = generate(tx)
    rx = RxConfig(modulation="qpsk", sample_rate=tx.sample_rate, symbol_rate=tx.symbol_rate,
                  rolloff=tx.rolloff, use_costas=True)
    r = demodulate(g.samples, rx)
    assert r.text.startswith("HELLO VECTORRF")
    assert r.metrics["evm_rms_percent"] < 5.0


def test_costas_tracks_cfo_even_with_phase_ambiguity():
    tx = TxConfig(waveform="qpsk", sample_rate=500_000, symbol_rate=25_000, duration=.10, snr_db=None)
    g = generate(tx)
    y = apply_channel(g.samples, g.sample_rate, ChannelConfig(snr_db=25, freq_offset_hz=1000, phase_offset_deg=35))
    rx = RxConfig(modulation="qpsk", sample_rate=tx.sample_rate, symbol_rate=tx.symbol_rate,
                  costas=CostasConfig(alpha=.08, beta=.003), use_costas=True)
    r = demodulate(y, rx)
    f = np.asarray(r.telemetry["costas_freq_hz"])
    np.testing.assert_allclose(np.median(f[-100:]), 1000, atol=80)
    assert r.metrics["evm_rms_percent"] < 10.0


def test_rx_exposes_error_vs_time_telemetry():
    tx = TxConfig(waveform="qpsk", sample_rate=500_000, symbol_rate=25_000, duration=.06, snr_db=28)
    g = generate(tx)
    rx = RxConfig(modulation="qpsk", sample_rate=tx.sample_rate, symbol_rate=tx.symbol_rate, use_costas=True)
    r = demodulate(g.samples, rx)
    assert len(r.telemetry["evm_time_percent"]) > 10
    assert len(r.telemetry["magnitude_error_percent"]) == len(r.telemetry["evm_time_percent"])
    assert len(r.telemetry["phase_error_deg"]) == len(r.telemetry["evm_time_percent"])


def test_mueller_muller_timing_recovery_tracks_fractional_delay():
    tx = TxConfig(waveform="qpsk", sample_rate=500_000, symbol_rate=25_000, duration=.08,
                  message="HELLO VECTORRF ", snr_db=None)
    g = generate(tx)
    y = apply_channel(g.samples, g.sample_rate, ChannelConfig(snr_db=30, timing_offset_samples=.72))
    rx = RxConfig(modulation="qpsk", sample_rate=tx.sample_rate, symbol_rate=tx.symbol_rate,
                  timing_recovery="mueller-muller", timing_gain=.30, use_costas=True)
    r = demodulate(y, rx)
    assert r.text.startswith("HELLO VECTORRF")
    assert len(r.telemetry["timing_error"]) > 20
    assert len(r.telemetry["timing_mu"]) == len(r.telemetry["timing_error"])
