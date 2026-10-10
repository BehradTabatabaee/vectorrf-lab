import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorrf.app import app
from vectorrf.engine import reduce_spectrum
from vectorrf.measurements import zero_span
from vectorrf.models import SpectrumConfig, TxConfig
from vectorrf.signal import generate
from vectorrf.spectrum import analyze

client = TestClient(app)


def _next_json(ws, wanted, limit=60):
    for _ in range(limit):
        message = ws.receive()
        if not message.get('text'):
            continue
        data = json.loads(message['text'])
        if data.get('type') == wanted:
            return data
    raise AssertionError(f'no {wanted!r} message received')


def test_release_metadata_and_help_page_are_published():
    from vectorrf import __version__
    from vectorrf.models import MAX_SAMPLE_RATE, MAX_CAPTURE_SAMPLES
    health = client.get('/api/health')
    assert health.status_code == 200
    assert health.json()['version'] == __version__ == '2.0.2'
    assert health.json()['max_spectrum_span'] == MAX_SAMPLE_RATE
    assert health.json()['max_capture_samples'] == MAX_CAPTURE_SAMPLES
    help_page = client.get('/help')
    assert help_page.status_code == 200
    text = help_page.text
    assert 'VectorRF Lab' in text and 'User guide' in text
    assert 'Spectrum analyzer' in text
    assert 'OUTSIDE SOURCE NYQUIST' in text
    assert 'Recordings' in text


@pytest.mark.parametrize('fs, span, expected_half', [
    (1_000_000, 100_000, 50_000),
    (1_000_000, 1_000_000, 500_000),
    (1_000_000, 10_000_000, 500_000),  # DSP data cannot exceed source Nyquist
    (10_000_000, 10_000_000, 5_000_000),
])
def test_spectrum_frequency_data_respects_requested_span_and_source_nyquist(fs, span, expected_half):
    tx = TxConfig(waveform='tone', sample_rate=fs, duration=.004, center_hz=0, snr_db=None)
    g = generate(tx)
    result = analyze(g.samples, fs, SpectrumConfig(span_hz=span, rbw_hz=max(500, fs / 4096), averaging=1, max_fft=8192))
    assert result.freq_hz[0] >= -expected_half - result.rbw_actual_hz
    assert result.freq_hz[-1] <= expected_half + result.rbw_actual_hz
    assert result.freq_hz[0] < 0 < result.freq_hz[-1]


def test_wide_10mhz_spectrum_endpoint_returns_bounded_display_data():
    response = client.post('/api/spectrum', json={
        'tx': {'waveform': 'tone', 'sample_rate': 10_000_000, 'duration': .004, 'center_hz': 1_000_000, 'snr_db': None},
        'spectrum': {'span_hz': 10_000_000, 'rbw_hz': 5_000, 'averaging': 1, 'max_fft': 32768},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['peak_hz'] == pytest.approx(1_000_000, abs=10_000)
    assert len(body['freq_hz']) <= 32768
    assert len(body['freq_hz']) == len(body['psd_db'])


def test_reduce_spectrum_is_peak_preserving_and_bounded_for_browser():
    f = np.linspace(-5e6, 5e6, 32768, endpoint=False)
    y = np.full(f.shape, -130.0)
    y[12345] = 42.0
    fr, yr = reduce_spectrum(f, y)
    assert len(fr) <= 2048
    assert len(fr) == len(yr)
    assert np.max(yr) == 42.0
    assert fr[0] >= f[0]
    assert fr[-1] <= f[-1]


def test_zero_span_default_payload_is_display_bounded_and_frequency_is_correct_at_10msps():
    fs = 10_000_000
    f0 = 1_234_567.0
    n = int(.004 * fs)
    t = np.arange(n) / fs
    x = np.exp(2j * np.pi * f0 * t)
    out = zero_span(x, fs, duration=.004, full_band=True)
    assert len(out['i']) <= 900
    assert len(out['i']) == len(out['q']) == len(out['frequency_hz']) == len(out['i_max']) == len(out['time_s'])
    assert np.nanmedian(out['frequency_hz']) == pytest.approx(f0, abs=1.0)
    # Tuned to the tone, the default RBW reads a steady carrier at 0 Hz.
    tuned = zero_span(x, fs, duration=.004, center_hz=f0)
    assert np.nanmax(np.abs(tuned['frequency_hz_max'])) < 1.0
    assert np.median(tuned['amplitude']) == pytest.approx(1.0, rel=1e-3)


def test_direct_api_rejects_oversized_span_without_poisoning_live_websocket_state():
    bad = client.post('/api/spectrum', json={
        'tx': {'waveform': 'tone', 'sample_rate': 1_000_000, 'duration': .004},
        'spectrum': {'span_hz': 50_000_000_000, 'rbw_hz': 2_000},
    })
    assert bad.status_code == 422

    with client.websocket_connect('/ws/live') as ws:
        assert ws.receive_json()['type'] == 'hello'
        ws.send_json({
            'action': 'start', 'fps': 20,
            'tx': {'waveform': 'tone', 'sample_rate': 1_000_000, 'duration': .004, 'center_hz': 40_000},
            'spectrum': {'span_hz': 1_000_000, 'rbw_hz': 2_000},
        })
        first = _next_json(ws, 'metrics')
        assert first['peak_hz'] == pytest.approx(40_000, abs=4_000)

        ws.send_json({'action': 'update', 'spectrum': {'span_hz': 50_000_000_000, 'rbw_hz': 2_000}})
        err = _next_json(ws, 'error')
        assert 'Span' in err['message'] and '5 GHz' in err['message']

        ws.send_json({
            'action': 'update',
            'tx': {'waveform': 'tone', 'sample_rate': 1_000_000, 'duration': .004, 'center_hz': -75_000},
            'spectrum': {'span_hz': 1_000_000, 'rbw_hz': 2_000},
        })
        _next_json(ws, 'meta')
        for _ in range(20):
            metrics = _next_json(ws, 'metrics')
            if abs(metrics['peak_hz'] + 75_000) < 4_000:
                break
        else:
            pytest.fail('live stream did not recover after oversized span validation error')

COMM_WAVEFORMS = [
    'tone', 'multitone', 'awgn', 'chirp',
    'bpsk', 'qpsk', 'dbpsk', 'dqpsk', 'pi4-dqpsk', 'oqpsk', '8psk',
    '16qam', '64qam', '16apsk', '32apsk', '2fsk', '4fsk', 'msk', 'gmsk',
    'ofdm', 'am', 'fm', 'pm', 'ssb', 'dsb-sc',
]


@pytest.mark.parametrize('waveform', COMM_WAVEFORMS)
def test_every_communications_waveform_runs_through_spectrum_and_quality_endpoints(waveform):
    tx = {
        'waveform': waveform, 'sample_rate': 500_000, 'duration': .004,
        'symbol_rate': 25_000, 'snr_db': None,
    }
    spectrum = client.post('/api/spectrum', json={
        'tx': tx,
        'spectrum': {'span_hz': 500_000, 'rbw_hz': 2_000, 'averaging': 1, 'max_fft': 4096},
    })
    assert spectrum.status_code == 200, (waveform, spectrum.text)
    body = spectrum.json()
    assert body['freq_hz'] and len(body['freq_hz']) == len(body['psd_db'])
    assert np.all(np.isfinite(body['psd_db']))

    quality = client.post('/api/measurements', json={
        'tx': tx, 'duration_s': .001, 'quality': True,
    })
    assert quality.status_code == 200, (waveform, quality.text)
    q = quality.json()['quality']
    assert q is not None and q['kind'] in {'evm', 'basic'}


@pytest.mark.parametrize('power_dbm', [-1000.0, 1000.0])
def test_absolute_power_path_has_no_artificial_dbm_clamp(power_dbm):
    tx = TxConfig(waveform='tone', sample_rate=500_000, duration=.004, power_dbm=power_dbm, snr_db=None)
    g = generate(tx)
    result = analyze(
        g.samples, g.sample_rate, SpectrumConfig(span_hz=500_000, rbw_hz=2_000, averaging=1),
        reference_power_dbm=g.metadata['power_dbm'],
        reference_rms=g.metadata['power_reference_rms'],
    )
    assert np.isfinite(result.peak_db)
    assert result.peak_db == pytest.approx(power_dbm, abs=.05)
