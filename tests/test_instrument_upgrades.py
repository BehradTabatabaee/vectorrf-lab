"""Spectrum zoom/channel power, receiver loops, de-hopping, classifier trace, radar and API upgrades."""
import io
import json
import tarfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorrf.app import app
from vectorrf.blind import FEATURES, analyze as classify
from vectorrf.measurements import zero_span
from vectorrf.models import ChannelBand, ChannelConfig, CostasConfig, RxConfig, SpectrumConfig, TxConfig
from vectorrf.radar_sim import RadarExperiment
from vectorrf.radar_studio import RadarStudioConfig, design_summary
from vectorrf.rx import demodulate
from vectorrf.signal import generate
from vectorrf.spectrum import analyze, required_samples, zoom_plan
from vectorrf.sync import apply_channel, loop_settling_symbols

client = TestClient(app)


# ---------------------------------------------------------------- spectrum --

def _tones(fs, n, freqs, amp=0.5):
    k = np.arange(n)
    return sum(amp * np.exp(2j * np.pi * f * k / fs) for f in freqs).astype(np.complex64)


def test_zoom_plan_decimates_narrow_spans_only():
    assert zoom_plan(100e6, SpectrumConfig(span_hz=100e6)).stages == ()
    assert zoom_plan(100e6, SpectrumConfig(span_hz=30e6)).stages == ()
    p = zoom_plan(100e6, SpectrumConfig(center_hz=12e6, span_hz=400e3))
    assert p.decimation > 1 and 1.25 * 400e3 <= p.rate < 2.5 * 400e3
    # A window that would cross Nyquist is analysed full band instead.
    assert zoom_plan(100e6, SpectrumConfig(center_hz=49.9e6, span_hz=1e6)).stages == ()


def test_zoom_keeps_tone_level_and_frequency_and_rejects_aliases():
    fs, n = 100e6, 1 << 22
    bin_centred = 12.3e6 + 64 * (781250.0 / 16384)
    x = _tones(fs, n, [bin_centred]) + _tones(fs, n, [12.8e6, 13.1e6, 11.0e6])  # three tones outside the span
    cfg = SpectrumConfig(center_hz=12.3e6, span_hz=400e3, rbw_hz=100, max_fft=16384, averaging=2, window="flattop")
    r = analyze(x, fs, cfg, reference_power_dbm=0.0, reference_rms=0.5)
    assert r.decimation > 1 and r.freq_hz[0] >= 12.1e6 and r.freq_hz[-1] <= 12.5e6
    assert r.peak_hz == pytest.approx(bin_centred, abs=r.rbw_actual_hz) and r.peak_db == pytest.approx(0, abs=0.05)
    others = r.psd_db[np.abs(r.freq_hz - bin_centred) > 20 * r.rbw_actual_hz]
    assert np.max(others) < -100  # out-of-span tones fold below the 110 dB stopband


def test_zoom_reaches_rbw_the_fft_limit_cannot():
    fs = 100e6
    cfg = dict(rbw_hz=100, max_fft=65536, averaging=1)
    x = _tones(fs, required_samples(fs, SpectrumConfig(center_hz=5e6, span_hz=200e3, **cfg)), [5e6])
    zoomed = analyze(x, fs, SpectrumConfig(center_hz=5e6, span_hz=200e3, **cfg))
    full = analyze(x, fs, SpectrumConfig(**cfg))
    assert zoomed.rbw_actual_hz < 200 < full.rbw_actual_hz


def test_channel_power_of_white_noise_matches_its_bandwidth():
    fs, n = 10e6, 1 << 20
    rng = np.random.default_rng(3)
    noise = ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)).astype(np.complex64)
    band = ChannelBand(center_hz=2e6, width_hz=1e6)
    for cfg in (SpectrumConfig(rbw_hz=5e3, averaging=64, channels=[band]),
                SpectrumConfig(center_hz=2e6, span_hz=2e6, rbw_hz=5e3, averaging=64, channels=[band])):
        ch = analyze(noise, fs, cfg, reference_power_dbm=0.0, reference_rms=1.0).channels[0]
        assert ch["power_db"] == pytest.approx(10 * np.log10(0.1), abs=0.1)
        assert ch["density_db_hz"] == pytest.approx(-10 * np.log10(fs), abs=0.1)
        assert ch["covered"] and ch["obw99_hz"] == pytest.approx(0.99e6, rel=0.02)


def test_channel_power_of_a_tone_is_its_power_for_every_window():
    fs, n = 1e6, 1 << 16
    x = _tones(fs, n, [123_456.7], amp=1.0)
    for window in ("hann", "blackmanharris", "flattop", "kaiser", "rect"):
        cfg = SpectrumConfig(rbw_hz=200, window=window, channels=[ChannelBand(center_hz=123e3, width_hz=20e3)])
        ch = analyze(x, fs, cfg, reference_power_dbm=-30.0, reference_rms=1.0).channels[0]
        assert ch["power_db"] == pytest.approx(-30, abs=0.02), window
        assert ch["peak_hz"] == pytest.approx(123_456.7, abs=200)


def test_band_outside_the_analysed_span_is_reported_uncovered():
    x = _tones(1e6, 1 << 15, [0])
    ch = analyze(x, 1e6, SpectrumConfig(center_hz=0, span_hz=100e3, channels=[ChannelBand(center_hz=300e3, width_hz=10e3)])).channels[0]
    assert ch["covered"] is False and "power_db" not in ch


def test_live_stream_reports_tuning_and_channel_power():
    with client.websocket_connect('/ws/live') as ws:
        ws.receive_json()
        ws.send_json({'action': 'start', 'tx': {'waveform': 'tone', 'center_hz': 100e3, 'power_dbm': -5, 'duration': .01},
                      'spectrum': {'center_hz': 100e3, 'span_hz': 100e3, 'rbw_hz': 200,
                                   'channels': [{'label': 'T', 'center_hz': 100e3, 'width_hz': 20e3}]}})
        while True:
            m = ws.receive()
            if m.get('text') and json.loads(m['text'])['type'] == 'metrics':
                d = json.loads(m['text'])
                break
    assert d['center_hz'] == 100e3 and d['span_hz'] == 100e3 and d['decimation'] > 1
    assert d['channels'][0]['power_db'] == pytest.approx(-5, abs=0.1) and d['channels'][0]['label'] == 'T'


# ---------------------------------------------------------------- receiver --

def _qpsk(cfo=0.0, timing=0.0, snr=30, seconds=.04):
    tx = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3, duration=seconds, message="VECTORRF LAB ", snr_db=None)
    y = apply_channel(generate(tx).samples, 1e6, ChannelConfig(snr_db=snr, freq_offset_hz=cfo, timing_offset_samples=timing,
                                                               phase_offset_deg=20))
    return y


@pytest.mark.parametrize("timing", ["manual", "gardner", "mueller-muller"])
@pytest.mark.parametrize("cfo", [0.0, 50.0, 300.0])
def test_timing_loops_tolerate_carrier_offset(timing, cfo):
    """M&M decisions are taken on carrier-corrected symbols (it failed at 50 Hz before)."""
    r = demodulate(_qpsk(cfo, 0.15 if timing != "manual" else 0.0),
                   RxConfig(modulation="qpsk", sample_rate=1e6, symbol_rate=50e3, timing_recovery=timing))
    assert r.metrics["evm_rms_percent"] < 6, r.metrics
    assert np.median(r.telemetry["costas_freq_hz"][-200:]) == pytest.approx(cfo, abs=5)


def test_loop_gains_are_independent_of_signal_level():
    rx = RxConfig(modulation="qpsk", sample_rate=1e6, symbol_rate=50e3)
    y = _qpsk(200.0)
    a, b = demodulate(y, rx), demodulate(y * 37.0, rx)
    assert a.telemetry["costas_freq_hz"] == pytest.approx(b.telemetry["costas_freq_hz"], rel=1e-4, abs=1e-3)


def test_demodulation_does_not_depend_on_oversampling():
    """The receiver works at most 16 samples/symbol: the same link (same
    in-band SNR) reads the same EVM at 20, 80 or 200 samples/symbol."""
    evm = {}
    for fs in (1e6, 4e6, 10e6):
        tx = TxConfig(waveform="qpsk", sample_rate=fs, symbol_rate=50e3, duration=0.04, snr_db=None, data_source="text")
        # Full-band SNR falls with the rate so the noise density, and the in-band SNR, stay the same.
        channel = ChannelConfig(snr_db=25 - 10 * np.log10(fs / 1e6), freq_offset_hz=300, phase_offset_deg=20,
                                timing_offset_samples=0.15)
        y = apply_channel(generate(tx).samples, fs, channel, seed=5)
        r = demodulate(y, RxConfig(modulation="qpsk", sample_rate=fs, symbol_rate=50e3, coarse_cfo=True))
        assert r.metrics["samples_per_symbol"] == round(fs / 50e3)
        assert r.metrics["working_samples_per_symbol"] == 16
        assert "VECTORRF LAB" in r.text
        evm[fs] = r.metrics["evm_rms_percent"]
    assert max(evm.values()) < 1.25 * min(evm.values()), evm


@pytest.mark.parametrize("sps", [4, 16, 64])
def test_timing_loop_converges_in_symbols_whatever_the_oversampling(sps):
    """Timing gains are in symbols per unit detector output."""
    from vectorrf.dsp import rrc_taps
    from vectorrf.sync import mueller_muller_carrier, timing_settling_symbols
    from scipy import signal as sig
    rng = np.random.default_rng(2)
    symbols = ((rng.integers(0, 2, 1500) * 2 - 1) + 1j * (rng.integers(0, 2, 1500) * 2 - 1)) / np.sqrt(2)
    h = rrc_taps(sps, 0.35, 10)
    y = np.convolve(sig.upfirdn(h, symbols, up=sps), h)[len(h) - 1:]
    y /= np.sqrt(np.mean(np.abs(y[::sps]) ** 2))
    start = int(0.4 * sps)  # begin 0.4 symbol late
    gain = 0.015
    out = mueller_muller_carrier(y[start:], sps, gain, CostasConfig(alpha=1e-6, beta=1e-9), 50e3)
    error = np.abs(np.asarray(out.timing.error, float))
    settle = timing_settling_symbols(gain, "mueller-muller", 0.35)
    early, late = np.mean(error[:20]), np.mean(error[settle:settle + 200])
    assert late < 0.1 * early, (early, late, settle)


def test_evm_excludes_the_reported_acquisition_transient():
    r = demodulate(_qpsk(400.0, snr=35), RxConfig(modulation="qpsk", sample_rate=1e6, symbol_rate=50e3))
    m = r.metrics
    assert m["evm_settle_symbols"] == loop_settling_symbols(0.04, 0.001, np.sqrt(2))
    assert m["evm_all_percent"] > m["evm_rms_percent"] and m["evm_rms_percent"] < 3
    assert r.telemetry["evm_settle_index"] == m["evm_settle_symbols"]


def test_default_demodulation_page_decodes_cleanly():
    """The first-open Demodulation view: UI channel and receiver defaults."""
    tx = TxConfig(waveform="qpsk", sample_rate=1e6, duration=0.02, amplitude=1, snr_db=30, symbol_rate=50e3, message="VECTORRF LAB")
    r = client.post('/api/rx/loopback', json={
        'tx': tx.model_dump(), 'capture_id': 3,
        'channel': {'snr_db': 25, 'freq_offset_hz': 300, 'phase_offset_deg': 20, 'timing_offset_samples': 0.15},
        'rx': {'modulation': 'qpsk', 'sample_rate': 1e6, 'symbol_rate': 50e3, 'timing_recovery': 'mueller-muller',
               'coarse_cfo': True}}).json()
    assert r['metrics']['evm_rms_percent'] < 5 and 'VECTORRF LAB' in r['text']


def test_synchronized_dehopping_recovers_a_hopping_qpsk_link():
    agility = dict(enabled=True, start_hz=-200e3, stop_hz=200e3, step_hz=100e3, dwell_s=2e-3)
    tx = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3, duration=0.02, message="HOPPING LINK ",
                  snr_db=None, agility=agility)
    body = {'tx': tx.model_dump(), 'channel': {'snr_db': 30}, 'capture_id': 2,
            'rx': {'modulation': 'qpsk', 'sample_rate': 1e6, 'symbol_rate': 50e3, 'coarse_cfo': True}}
    blind = client.post('/api/rx/loopback', json=body).json()
    body['rx']['dehop'] = [agility]
    locked = client.post('/api/rx/loopback', json=body).json()
    assert locked['metrics']['evm_rms_percent'] < 5 < blind['metrics']['evm_rms_percent']
    assert 'HOPPING LINK' in locked['text']


def test_measurement_quality_dehops_continuous_hopping():
    tx = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3, duration=0.02, snr_db=35,
                  agility=dict(enabled=True, start_hz=-200e3, stop_hz=200e3, step_hz=100e3, dwell_s=2e-3))
    q = client.post('/api/measurements', json={'tx': tx.model_dump(), 'quality': True}).json()['quality']
    assert q['kind'] == 'evm' and q['metrics']['evm_rms_percent'] < 6
    gated = tx.model_copy(update={'agility': tx.agility.model_copy(update={'off_s': 1e-4})})
    q = client.post('/api/measurements', json={'tx': gated.model_dump(), 'quality': True}).json()['quality']
    assert q['kind'] == 'basic' and 'off-time' in q['method']


# -------------------------------------------------------------- zero span --

def test_zero_span_tunes_and_band_limits():
    fs, n = 1e6, 20_000
    x = _tones(fs, n, [100e3]) + _tones(fs, n, [300e3], amp=0.5)
    plain = zero_span(x, fs, 0.002, 0.004)
    tuned = zero_span(x, fs, 0.002, 0.004, center_hz=100e3, bandwidth_hz=50e3)
    assert np.std(plain['amplitude']) > 0.2  # two tones beat
    assert np.std(tuned['amplitude']) < 0.01 and np.median(tuned['amplitude']) == pytest.approx(0.5, rel=0.01)
    assert np.allclose(tuned['frequency_hz'][1:], 0.0, atol=50) and tuned['center_hz'] == 100e3


# --------------------------------------------------------------- classifier --

def test_feature_glossary_labels_are_distinct():
    """The decision path shows labels: two measurements must never read alike."""
    labels = [label for label, _ in FEATURES.values()]
    assert len(labels) == len(set(labels))
    assert all(description.endswith(".") for _, description in FEATURES.values())
    served = client.get("/api/blind/features").json()["features"]
    assert served == {key: {"label": label, "description": text} for key, (label, text) in FEATURES.items()}


@pytest.mark.parametrize("waveform", ["awgn", "qpsk", "2fsk", "lfm", "tone", "ofdm"])
def test_classifier_explains_its_decision(waveform):
    cfg = TxConfig(waveform=waveform, sample_rate=500e3, symbol_rate=25e3, duration=.06, data_source="pn",
                   snr_db=None if waveform == "awgn" else 30, sweep_bw_hz=100e3, pulse_width_s=40e-6, pri_s=200e-6)
    r = classify(generate(cfg).samples, 500e3)
    accepted = [s for s in r.trace if s["outcome"] == "accepted"]
    assert len(accepted) == 1 and accepted[0]["result"] == r.modulation and r.trace[-1] is accepted[0]
    assert all(s["outcome"] == "rejected" for s in r.trace[:-1])
    for stage in r.trace:
        for c in stage["checks"]:
            assert c["feature"] in FEATURES and np.isfinite(c["threshold"]) and isinstance(c["passed"], bool)
    # Every earlier stage was rejected by at least one failing check (or a failed sub-test).
    assert all(any(not c["passed"] for c in s["checks"]) or s["detail"] for s in r.trace[:-1])


def test_constellation_candidates_explain_subset_payloads():
    """A repeating 12-byte ASCII payload visits only 9 of the 16 QAM points.

    The 16-QAM fit itself is excellent; the candidates say that the unused
    points (not the EVM) ruled it out.
    """
    cfg = TxConfig(waveform="16qam", sample_rate=500e3, symbol_rate=25e3, duration=.06, snr_db=30, message="VECTORRF LAB")
    r = classify(generate(cfg).samples, 500e3)
    c = {x["modulation"]: x for x in r.candidates}
    assert c["16qam"]["penalized"] and c["16qam"]["points_used"] == pytest.approx(9 / 16, abs=0.07)
    assert c["16qam"]["evm_percent"] < 6 and c["16qam"]["score"] >= 100


def test_classifier_isolates_a_narrow_band_in_a_wide_capture():
    fs = 50e6
    tx = TxConfig(waveform="composite", sample_rate=fs, duration=4e-3, snr_db=40, components=[
        {"waveform": "qpsk", "symbol_rate": 100e3, "center_hz": 8e6, "power_dbm": -10, "data_source": "pn"},
        {"waveform": "2fsk", "symbol_rate": 200e3, "fsk_deviation_hz": 300e3, "center_hz": -10e6, "power_dbm": -5, "data_source": "pn"}])
    x = generate(tx).samples
    r = classify(x, fs, (8e6, 150e3))
    assert r.modulation == "qpsk" and r.symbol_rate_baud == pytest.approx(100e3, rel=0.02)
    assert r.features["analysis_band"]["analysis_rate"] < fs


# -------------------------------------------------------------------- radar --

def test_radar_waveform_record_follows_the_simulation_rate():
    exp = RadarExperiment(source="legacy", sample_rate=2e6, legacy=RadarStudioConfig(sample_rate=1e6))
    assert exp.legacy.sample_rate == 2e6
    with pytest.raises(ValueError):
        # The design is revalidated at the simulation rate (sweep wider than 1 MS/s).
        RadarExperiment(source="legacy", sample_rate=1e6, legacy=RadarStudioConfig(sample_rate=5e6, continuous_waveform="fmcw",
                                                                                      mode="continuous", sweep_bw_hz=2e6))


def test_radar_design_summary_figures():
    cfg = RadarStudioConfig()  # default: 8 × LFM 30 µs, 250 kHz, PRI 150 µs, then 400 µs off
    s = design_summary(cfg, 3e9)
    item = s["items"][0]
    assert item["prf_hz"] == pytest.approx(1 / 150e-6) and item["duty"] == pytest.approx(0.2)
    assert item["range_resolution_m"] == pytest.approx(299_792_458 / (2 * 250e3))
    assert item["compression_ratio"] == pytest.approx(30e-6 * 250e3)
    assert item["unambiguous_velocity_mps"] == pytest.approx(0.1 / (4 * 150e-6), rel=1e-3)
    assert s["sequence_s"] == pytest.approx(8 * 150e-6 + 400e-6)
    r = client.post('/api/studio/radar', json={'source': 'legacy', 'samples': 1024}).json()
    assert r['design']['items'][0]['prf_hz'] == pytest.approx(1 / 150e-6)


# ---------------------------------------------------------------------- API --

def test_plan_endpoint():
    design = {'waveform': 'composite', 'sample_rate': 60e6, 'duration': 0.002, 'components': [
        {'waveform': 'tone', 'power_dbm': -10.0}, {'waveform': 'qpsk', 'power_dbm': -20.0, 'center_hz': 20e6, 'symbol_rate': 7.5e6}]}
    plan = client.post('/api/tx/plan', json=design).json()
    assert plan['config']['waveform'] == 'composite' and len(plan['components']) == 2
    bad = client.post('/api/tx/plan', json={'waveform': 'composite', 'components': []})
    assert bad.status_code == 422


def test_sigmf_export_embeds_the_design():
    tx = {'waveform': 'composite', 'sample_rate': 1e6, 'duration': .002, 'rf_center_hz': 433.92e6,
          'components': [{'waveform': 'tone', 'power_dbm': -3}]}
    r = client.post('/api/files/export', json={'tx': tx, 'format': 'cf32_le', 'container': 'sigmf'})
    with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
        meta = json.load(tar.extractfile(next(m for m in tar.getmembers() if m.name.endswith('.sigmf-meta'))))
    design = meta['global']['vectorrf:tx_design']
    assert design['components'][0]['power_dbm'] == -3 and TxConfig.model_validate(design).rf_center_hz == 433.92e6
    assert meta['captures'][0]['core:frequency'] == 433.92e6


# ---------------------------------------------------------- display payloads --

def test_display_values_round_relative_to_the_series_peak():
    from vectorrf.dsp import display_values
    rng = np.random.default_rng(0)
    for scale in (1e-9, 3e-3, 1.0, 2.5e6):
        x = scale * rng.standard_normal(1000)
        y = np.asarray(display_values(x, 6))
        top = np.max(np.abs(x))
        quantum = 10.0 ** (np.floor(np.log10(top)) - 5)
        assert np.max(np.abs(y - x)) <= 0.5 * quantum * (1 + 1e-9)
    # Exact without digits, for integers and for degenerate series.
    x = np.array([0.1234567890123, -2.0])
    assert display_values(x) == x.tolist()
    assert display_values(np.arange(5, dtype=np.int64), 2) == [0, 1, 2, 3, 4]
    assert display_values(np.zeros(3), 6) == [0.0, 0.0, 0.0]
    # JSON has no NaN: undefined points become null (a gap in the plot).
    y = display_values(np.array([np.nan, 1.23456789, np.inf]), 3)
    assert y == [None, 1.23, None]
    json.dumps(y, allow_nan=False)


def test_display_indices_and_tree():
    from vectorrf.dsp import display_complex, display_indices, display_tree
    ix = display_indices(10_001, 640)
    assert len(ix) <= 640 and ix[0] == 0 and ix[-1] == 10_000 and np.all(np.diff(ix) > 0)
    assert len(display_indices(0, 10)) == 0 and display_indices(3, 10).tolist() == [0, 1, 2]
    assert display_complex(None) is None
    z = display_complex(np.exp(1j * np.linspace(0, 6, 5000)), 100)
    assert len(z["i"]) == len(z["q"]) == 100
    tree = display_tree({"a": np.linspace(0, 1, 5000), "n": 3, "s": ["x", "y"], "ragged": [[1, 2], [3]],
                         "nested": {"ints": list(range(4))}}, maximum=50, digits=4)
    assert len(tree["a"]) == 50 and tree["n"] == 3 and tree["s"] == ["x", "y"] and tree["ragged"] == [[1, 2], [3]]
    assert tree["nested"]["ints"] == [0, 1, 2, 3]
    json.dumps(tree)


def test_live_receiver_frames_round_plot_traces_but_rest_stays_exact():
    from vectorrf.engine import LIVE_DIGITS, receiver_product
    from vectorrf.pipeline import Pipeline
    from vectorrf.source import acquire
    tx = TxConfig(waveform="qpsk", sample_rate=1e6, symbol_rate=50e3, duration=0.02, snr_db=30)
    rx = RxConfig(modulation="qpsk", sample_rate=1e6, symbol_rate=50e3, coarse_cfo=True)
    channel = ChannelConfig(snr_db=25, freq_offset_hz=300, phase_offset_deg=20, timing_offset=0.15)
    product = receiver_product(acquire(tx, None, 4), channel, Pipeline(), rx, 4)
    body = json.dumps(product)
    assert len(body) < 400_000  # bounded per frame at ~10 frames/s
    i = np.asarray(product["symbols"]["i"])
    quantum = 10.0 ** (np.floor(np.log10(np.max(np.abs(i)))) - (LIVE_DIGITS - 1))
    assert np.allclose(i / quantum, np.round(i / quantum), atol=1e-6)
    # REST measurements are not display-rounded: a zero-span frequency keeps full resolution.
    fs, f0 = 10e6, 1_234_567.0
    x = np.exp(2j * np.pi * f0 * np.arange(20_000) / fs)
    out = zero_span(x, fs, 0, 1e-3)
    assert np.median(out["frequency_hz"][1:]) == pytest.approx(f0, abs=1e-3)


# ------------------------------------------------------- channel and OFDM --

def _band_limited(n=1 << 14, edge=0.41, seed=1):
    rng = np.random.default_rng(seed)
    spectrum = np.fft.fft(rng.standard_normal(n) + 1j * rng.standard_normal(n))
    f = np.fft.fftfreq(n)
    spectrum[np.abs(f) > edge] = 0
    return spectrum, f


@pytest.mark.parametrize("delay", [0.15, 0.5, -0.3])
def test_fractional_delay_is_accurate_and_keeps_every_sample(delay):
    """The channel delay used to zero the last 20 samples of every capture."""
    from vectorrf.dsp import fractional_delay
    spectrum, f = _band_limited()
    x = np.fft.ifft(spectrum)
    exact = np.fft.ifft(spectrum * np.exp(-2j * np.pi * f * delay))
    y = fractional_delay(x, delay)
    assert len(y) == len(x)
    interior = slice(100, -100)
    assert np.linalg.norm(y[interior] - exact[interior]) / np.linalg.norm(exact[interior]) < 10 ** (-55 / 20)
    tail = slice(-20, None)  # only the edge sees the record's end
    assert np.linalg.norm(y[tail] - exact[tail]) / np.linalg.norm(exact[tail]) < 0.2


def test_ofdm_channel_interpolation_is_exact_for_gain_and_delay():
    from vectorrf.ofdm import interpolate_channel
    carriers = np.r_[-26:0, 1:27]
    pilots = np.array([-21, -7, 7, 21])
    truth = lambda k: 0.7 * np.exp(1j * (0.4 - 2 * np.pi * 0.37 * np.asarray(k) / 64))
    np.testing.assert_allclose(interpolate_channel(carriers, pilots, truth(pilots)), truth(carriers), atol=1e-12)
    assert np.allclose(interpolate_channel(carriers, [5], [2j]), 2j)


@pytest.mark.parametrize("delay", [0.15, 0.5, -0.3])
def test_ofdm_receiver_removes_a_fractional_timing_offset(delay):
    """A pure delay is a linear phase across carriers: the pilots remove it.

    Before 2.0 carriers outside the outermost pilots kept the edge pilot's
    phase and the FFT window ended on the next symbol's smeared edge, so a
    noiseless 0.5-sample delay read 9 % EVM.
    """
    from vectorrf.models import OFDMDesign
    tx = TxConfig(waveform="ofdm", sample_rate=1e6, duration=0.02, snr_db=None, data_source="pn",
                  ofdm_design=OFDMDesign(enabled=True))
    x = np.asarray(generate(tx).samples, complex)
    f = np.fft.fftfreq(len(x))
    y = np.fft.ifft(np.fft.fft(x) * np.exp(-2j * np.pi * f * delay)).astype(np.complex64)
    rx = RxConfig(modulation="ofdm", sample_rate=1e6, ofdm_fft=tx.ofdm_fft, ofdm_cp=tx.ofdm_cp,
                  ofdm_occupied=tx.ofdm_occupied, ofdm_design=tx.ofdm_design, coarse_cfo=True)
    assert demodulate(y, rx).metrics["evm_rms_percent"] < 1.0
    # The window advance is compensated exactly: no impairment, no error.
    assert demodulate(x.astype(np.complex64), rx).metrics["evm_rms_percent"] < 1e-3
