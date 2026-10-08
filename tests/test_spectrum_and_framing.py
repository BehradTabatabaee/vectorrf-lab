import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from vectorrf.app import app
from vectorrf.models import RxConfig, SpectrumConfig, TxConfig
from vectorrf.rx import receive_frame
from vectorrf.signal import generate
from vectorrf.spectrum import analyze


client = TestClient(app)


def test_spectrum_span_limit_follows_maximum_sample_rate():
    from vectorrf.models import MAX_SAMPLE_RATE, MAX_SPECTRUM_SPAN
    assert MAX_SPECTRUM_SPAN == MAX_SAMPLE_RATE >= 2e9
    assert SpectrumConfig(span_hz=10_000_000).span_hz == 10_000_000
    assert SpectrumConfig(span_hz=MAX_SPECTRUM_SPAN).span_hz == MAX_SPECTRUM_SPAN
    with pytest.raises(ValidationError):
        SpectrumConfig(span_hz=MAX_SPECTRUM_SPAN * 1.0001)


@pytest.mark.parametrize('power_dbm', [-50.0, 100.0])
def test_tone_peak_is_calibrated_to_absolute_tx_dbm(power_dbm):
    tx = TxConfig(
        waveform='tone', sample_rate=1_000_000, duration=.02,
        center_hz=0, power_dbm=power_dbm, snr_db=None,
    )
    generated = generate(tx)
    result = analyze(
        generated.samples, generated.sample_rate, SpectrumConfig(rbw_hz=1_000),
        reference_power_dbm=generated.metadata['power_dbm'],
        reference_rms=generated.metadata['power_reference_rms'],
    )
    assert result.unit == 'dBm'
    assert result.peak_hz == pytest.approx(0.0, abs=result.rbw_actual_hz)
    assert result.peak_db == pytest.approx(power_dbm, abs=.02)


def test_basic_signal_modulation_quality_returns_direct_waveform_metrics():
    response = client.post('/api/measurements', json={
        'tx': {'waveform': 'tone', 'sample_rate': 200_000, 'duration': .02, 'center_hz': 12_500},
        'duration_s': .002,
        'quality': True,
    })
    assert response.status_code == 200, response.text
    quality = response.json()['quality']
    assert quality['kind'] == 'basic'
    assert quality['symbols']['i']
    assert quality['telemetry']['amplitude']
    assert quality['telemetry']['frequency_hz']
    assert 'rms' in quality['metrics']


def test_mixed_modulation_frame_and_rx_acquisition_share_block_plan():
    blocks = [
        {'name': 'Sync', 'role': 'preamble', 'modulation': 'bpsk', 'bit_count': 24,
         'bits': '101101001110010110100011', 'symbol_rate': 50_000},
        {'name': 'OFDM header', 'role': 'bits', 'modulation': 'ofdm', 'bit_count': 104,
         'bits': '11010010', 'symbol_rate': 50_000},
        {'name': 'Payload', 'role': 'payload', 'modulation': 'qpsk', 'bit_count': 64,
         'symbol_rate': 50_000},
    ]
    tx = TxConfig(
        waveform='qpsk', sample_rate=500_000, symbol_rate=50_000, duration=.02,
        frame_enabled=True, frame_mode='blocks', frame_blocks=blocks, snr_db=None,
    )
    generated = generate(tx)
    assert generated.metadata['waveform'] == 'mixed-frame'
    assert [b['modulation'] for b in generated.metadata['frame_blocks'][:3]] == ['bpsk', 'ofdm', 'qpsk']

    rx = RxConfig(
        modulation='qpsk', sample_rate=tx.sample_rate, symbol_rate=tx.symbol_rate,
        frame_enabled=True, frame_blocks=blocks,
        energy_threshold_dbm=-120, preamble_threshold=.6,
    )
    acquisition = receive_frame(
        generated.samples, rx, power_dbm=generated.metadata['power_dbm'],
        reference_rms=generated.metadata['power_reference_rms'],
    )
    assert acquisition['detected']
    assert acquisition['correlation']['peak'] > .95
    assert [b['name'] for b in acquisition['decoded_blocks']] == ['Sync', 'OFDM header', 'Payload']
    assert acquisition['decoded_blocks'][1]['bit_count'] == 104


def _next_json(ws, wanted, limit=30):
    for _ in range(limit):
        message = ws.receive()
        if not message.get('text'):
            continue
        data = json.loads(message['text'])
        if data.get('type') == wanted:
            return data
    raise AssertionError(f'no {wanted!r} message received')


def test_live_stream_recovers_after_invalid_negative_span():
    with client.websocket_connect('/ws/live') as ws:
        assert ws.receive_json()['type'] == 'hello'
        ws.send_json({
            'action': 'start', 'fps': 10,
            'tx': {'waveform': 'tone', 'sample_rate': 500_000, 'duration': .02, 'center_hz': 10_000},
            'spectrum': {'span_hz': 400_000, 'rbw_hz': 2_000},
        })
        first = _next_json(ws, 'metrics')
        assert first['peak_hz'] == pytest.approx(10_000, abs=3_000)

        ws.send_json({'action': 'update', 'spectrum': {'span_hz': -1, 'rbw_hz': 2_000}})
        err = _next_json(ws, 'error')
        assert 'Span' in err['message'] and err['scope'] == 'spectrum'

        ws.send_json({
            'action': 'update',
            'tx': {'waveform': 'tone', 'sample_rate': 500_000, 'duration': .02, 'center_hz': 55_000},
            'spectrum': {'span_hz': 400_000, 'rbw_hz': 2_000},
        })
        # Ignore any already queued old metrics; a regenerated record announces meta.
        _next_json(ws, 'meta')
        for _ in range(10):
            metrics = _next_json(ws, 'metrics')
            if abs(metrics['peak_hz'] - 55_000) < 3_000:
                break
        else:
            pytest.fail('stream did not resume with the edited transmitter after invalid span')
