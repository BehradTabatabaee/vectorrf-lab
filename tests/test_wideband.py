"""Multi-GS/s operation: every DSP path is written in samples/symbols, so results
at 2-5 GS/s must match the same experiment scaled to kS/s-MS/s rates."""
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vectorrf.app import app
from vectorrf.blind import analyze as classify
from vectorrf.measurements import zero_span
from vectorrf.models import MAX_CAPTURE_SAMPLES, MAX_SAMPLE_RATE, RxConfig, SpectrumConfig, TxConfig
from vectorrf.ofdm import OFDMDesign
from vectorrf.pipeline import Pipeline, process_chain
from vectorrf.radar_processing import RadarProcessing
from vectorrf.radar_sim import ArrayModel, RadarExperiment, SceneTarget, simulate
from vectorrf.rx import demodulate
from vectorrf.signal import generate
from vectorrf.source import source_record
from vectorrf.spectrum import analyze

client = TestClient(app)
GIGA = [2e9, 5e9]


def test_rate_limits_are_sample_count_based():
    assert MAX_SAMPLE_RATE >= 2e9
    TxConfig(sample_rate=MAX_SAMPLE_RATE, duration=1e-4, symbol_rate=MAX_SAMPLE_RATE / 10)
    with pytest.raises(ValidationError):
        TxConfig(sample_rate=MAX_SAMPLE_RATE * 1.01, duration=1e-6)
    with pytest.raises(ValidationError, match="at most .* samples"):
        TxConfig(sample_rate=2e9, duration=(MAX_CAPTURE_SAMPLES + 2) / 2e9, symbol_rate=2e8)
    assert client.get("/api/health").json()["max_sample_rate"] == MAX_SAMPLE_RATE


@pytest.mark.parametrize("fs", GIGA)
def test_qpsk_scales_exactly_to_ghz_rates(fs):
    """The same 10-samples-per-symbol experiment at 1 MS/s and at GS/s gives identical symbols."""
    def run(rate):
        tx = TxConfig(waveform="qpsk", sample_rate=rate, symbol_rate=rate / 10, duration=200_000 / rate,
                      message="WIDEBAND ", snr_db=None, center_hz=0.137 * rate)
        g = generate(tx)
        r = demodulate(g.samples, RxConfig(modulation="qpsk", sample_rate=rate, symbol_rate=rate / 10, tuner_hz=tx.center_hz))
        return g, r
    g_lo, r_lo = run(1e6)
    g_hi, r_hi = run(fs)
    assert r_hi.text.startswith("WIDEBAND")
    np.testing.assert_allclose(r_hi.symbols, r_lo.symbols, atol=2e-3)
    assert r_hi.metrics["evm_rms_percent"] == pytest.approx(r_lo.metrics["evm_rms_percent"], abs=0.05)


@pytest.mark.parametrize("fs", GIGA)
def test_spectrum_power_and_frequency_at_ghz(fs):
    f0 = 0.3141 * fs
    g = generate(TxConfig(waveform="tone", sample_rate=fs, duration=262_144 / fs, center_hz=f0, power_dbm=-12.5, snr_db=40))
    r = analyze(g.samples, fs, SpectrumConfig(rbw_hz=fs / 40_000, window="flattop", averaging=4, max_fft=1 << 18),
                reference_power_dbm=-12.5, reference_rms=g.metadata["power_reference_rms"])
    assert r.peak_hz == pytest.approx(f0, abs=r.rbw_actual_hz)
    assert r.peak_db == pytest.approx(-12.5, abs=0.05)
    assert r.freq_hz[0] == pytest.approx(-fs / 2, rel=1e-6)


def test_ofdm_numerology_at_2gsps():
    d = OFDMDesign(enabled=True, modulation="64qam", guard_left=100, guard_right=99,
                   pilot_carriers=[-400, -200, -100, 100, 200, 400], cp_lengths=[72, 64])
    tx = TxConfig(waveform="ofdm", sample_rate=2e9, ofdm_fft=1024, ofdm_cp=72, ofdm_design=d,
                  duration=1e-4, snr_db=None, message="OFDM AT 2 GS/S ")
    g = generate(tx)
    assert g.metadata["subcarrier_spacing_hz"] == pytest.approx(2e9 / 1024)
    r = demodulate(g.samples, RxConfig(modulation="ofdm", sample_rate=2e9, ofdm_fft=1024, ofdm_cp=72, ofdm_design=d))
    assert r.text.startswith("OFDM AT 2 GS/S")
    assert r.metrics["evm_rms_percent"] < 1e-3


@pytest.mark.parametrize("fs", GIGA)
def test_contiguous_captures_at_ghz_keep_exact_carrier_phase(fs):
    tx = TxConfig(waveform="tone", sample_rate=fs, duration=4096 / fs, center_hz=0.2123456789 * fs, snr_db=None,
                  normalize_output=False, amplitude=1)
    later = 1_000_000_000  # capture far into simulated time
    a, b = source_record(tx, later), source_record(tx, later + 1)
    joined = np.r_[a.samples, b.samples]
    step = np.angle(joined[1:] * np.conj(joined[:-1]))
    np.testing.assert_allclose(step, 2 * np.pi * 0.2123456789, atol=2e-6)
    assert b.metadata["start_sample"] - a.metadata["start_sample"] == 4096


@pytest.mark.parametrize("fs", GIGA)
def test_classifier_and_zero_span_are_rate_invariant(fs):
    tx = TxConfig(waveform="16qam", sample_rate=fs, symbol_rate=fs / 8, duration=60_000 / fs, data_source="pn",
                  snr_db=30, center_hz=0.05 * fs)
    b = classify(generate(tx).samples, fs)
    assert b.modulation == "16qam"
    assert b.symbol_rate_baud == pytest.approx(fs / 8, rel=0.01)
    z = zero_span(np.exp(2j * np.pi * 0.123 * np.arange(20_000)), fs, duration=20_000 / fs)
    assert np.median(z["frequency_hz"][1:]) == pytest.approx(0.123 * fs, rel=1e-9)


def test_pipeline_mixer_and_resampler_at_ghz():
    fs = 4e9
    x = np.exp(2j * np.pi * 1.1e9 * np.arange(65536) / fs)
    cfg = Pipeline.model_validate({"blocks": [{"kind": "mixer", "frequency_hz": 1.0e9},
                                              {"kind": "lowpass", "cutoff_hz": 4e8, "taps": 127},
                                              {"kind": "resample", "up": 1, "down": 4}]})
    y, out_fs, stages, _ = process_chain(x, fs, cfg, start_sample=123_456_789_012)
    assert out_fs == 1e9 and len(stages) == 3
    f = np.median(np.angle(y[200:] * np.conj(y[199:-1]))) * out_fs / (2 * np.pi)
    assert f == pytest.approx(1e8, rel=1e-6)


def test_fmcw_radar_at_2gsps_resolves_decimetre_range():
    """1 GHz chirp bandwidth → c/2B = 15 cm range resolution."""
    fs, pri, bw = 2e9, 4e-6, 1e9
    slope = bw / pri
    exp = RadarExperiment(
        sample_rate=fs, samples=65536, carrier_hz=77e9, noise_db=-90, max_range_m=300, max_velocity_mps=40,
        program={"period_s": pri, "expression": f"exp(j*2*pi*({-bw / 2}*t + {slope / 2}*t*t))"},
        antenna=ArrayModel(geometry="linear", columns=1, element="isotropic"),
        targets=[SceneTarget(position=[42.3, 0, 0], velocity=[0, 0, 0], gain_db=0)],
        processing=RadarProcessing(method="fmcw", pri_s=pri, chirp_slope_hz_s=slope, doppler_window="hann"))
    d = simulate(exp)
    assert d["processor_info"]["range_bin_m"] == pytest.approx(299792458 / (2 * bw), rel=0.01)
    assert d["peak_range_m"] == pytest.approx(42.3, abs=0.3)


def test_radar_limits_bound_memory_not_rate():
    with pytest.raises(ValidationError, match="channel-samples"):
        RadarExperiment(sample_rate=2e9, samples=262_144, antenna=ArrayModel(geometry="planar", columns=8, rows=4))
    RadarExperiment(sample_rate=5e9, samples=131_072, antenna=ArrayModel(geometry="linear", columns=4))


def test_live_stream_at_2gsps_reports_wideband_axis():
    with client.websocket_connect("/ws/live") as ws:
        hello = ws.receive_json()
        assert hello["max_sample_rate"] == MAX_SAMPLE_RATE and hello["max_capture_samples"] == MAX_CAPTURE_SAMPLES
        ws.send_json({"action": "start", "tx": {"waveform": "qpsk", "sample_rate": 2e9, "symbol_rate": 2e8,
                                                 "duration": 262_144 / 2e9, "center_hz": 3e8},
                      "spectrum": {"span_hz": 2e9, "rbw_hz": 1e5, "max_fft": 65536}})
        for _ in range(20):
            msg = ws.receive()
            if msg.get("text") and json.loads(msg["text"])["type"] == "metrics":
                m = json.loads(msg["text"])
                break
        assert m["freq_hz"][0] == pytest.approx(-1e9, rel=1e-3) and m["freq_hz"][-1] == pytest.approx(1e9, rel=1e-3)
        assert 2e8 < m["bw99_hz"] < 3e8
        assert m["peak_hz"] == pytest.approx(3e8, abs=1.5e8)


def test_wideband_export_formats():
    tx = TxConfig(waveform="tone", sample_rate=2e9, duration=8192 / 2e9, center_hz=1e8, snr_db=None).model_dump(mode="json")
    wav = client.post("/api/files/export", json={"tx": tx, "format": "ci16_le", "container": "wav"})
    assert wav.status_code == 200 and wav.content[:4] == b"RIFF"
    tx5 = {**tx, "sample_rate": 5e9, "duration": 8192 / 5e9}
    assert client.post("/api/files/export", json={"tx": tx5, "format": "ci16_le", "container": "wav"}).status_code == 422
    sig = client.post("/api/files/export", json={"tx": tx5, "format": "ci16_le", "container": "sigmf"})
    assert sig.status_code == 200 and int(sig.headers["X-VectorRF-Samples"]) == 8192
