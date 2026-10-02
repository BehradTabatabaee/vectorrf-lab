import json

import pytest
from fastapi.testclient import TestClient

from vectorrf.app import app
from vectorrf.radar_sim import RadarExperiment


client = TestClient(app)


@pytest.mark.parametrize('waveform', ['tone', 'multitone', 'awgn', 'chirp', 'custom'])
def test_basic_sources_use_real_receiver_path(waveform):
    response = client.post('/api/rx/loopback', json={
        'tx': {'waveform': waveform, 'sample_rate': 200_000, 'duration': 0.004},
        'channel': {'snr_db': None},
        'rx': {'modulation': waveform},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['symbols'] is None
    assert body['bits'] == []
    assert body['recovered']
    assert body['telemetry']['amplitude']
    assert body['telemetry']['frequency_hz']
    assert body['metrics']['sample_count'] > 0


@pytest.mark.parametrize('waveform', ['am', 'fm', 'pm', 'ssb', 'dsb-sc'])
def test_analog_sources_keep_actual_demodulated_capture(waveform):
    response = client.post('/api/rx/loopback', json={
        'tx': {'waveform': waveform, 'sample_rate': 200_000, 'duration': 0.004, 'message_freq_hz': 3_217.35},
        'channel': {'snr_db': None},
        'rx': {'modulation': waveform, 'fm_deviation_hz': 25_000},
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['recovered']
    assert body['telemetry']['amplitude']
    assert body['telemetry']['frequency_hz']
    assert 'rms' in body['metrics']


def test_receiver_pipeline_edits_change_the_same_capture_data():
    base = {
        'tx': {'waveform': 'tone', 'sample_rate': 200_000, 'duration': 0.004, 'normalize_output': False, 'amplitude': 1.0},
        'channel': {'snr_db': None},
        'rx': {'modulation': 'tone'},
        'capture_id': 11,
    }
    direct = client.post('/api/rx/loopback', json=base).json()
    attenuated = client.post('/api/rx/loopback', json={
        **base,
        'pipeline': {'blocks': [{'name': 'Live gain', 'kind': 'gain', 'gain_db': -20}]},
    }).json()
    assert attenuated['metrics']['rms'] == pytest.approx(direct['metrics']['rms'] * 0.1, rel=0.03)


def test_live_receiver_accepts_60hz_and_streams_basic_source():
    with TestClient(app).websocket_connect('/ws/live') as ws:
        ws.receive_json()  # hello
        ws.send_json({
            'action': 'start',
            'fps': 60,
            'acknowledge_frames': True,
            'tx': {'waveform': 'chirp', 'sample_rate': 200_000, 'duration': 0.004},
            'channel': {'snr_db': None},
            'pipeline': {'blocks': []},
            'rx': {'modulation': 'chirp'},
        })
        seen_metrics = seen_receiver = False
        for _ in range(12):
            message = ws.receive()
            if not message.get('text'):
                continue
            data = json.loads(message['text'])
            if data['type'] == 'metrics':
                assert data['display_fps'] == 60.0
                seen_metrics = True
            elif data['type'] == 'receiver':
                assert data['recovered']
                assert data['telemetry']['frequency_hz']
                seen_receiver = True
            elif data['type'] == 'frame_end':
                ws.send_json({'action': 'ack', 'seq': data['seq']})
            if seen_metrics and seen_receiver:
                break
        assert seen_metrics and seen_receiver


def test_radar_default_is_60fps_friendly_without_reducing_available_capture_size():
    assert RadarExperiment().samples == 4096
    assert RadarExperiment(samples=32768).samples == 32768
