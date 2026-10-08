import pytest
from fastapi.testclient import TestClient
from vectorrf.app import app

client = TestClient(app)


def test_health_and_catalog():
    h = client.get('/api/health')
    assert h.status_code == 200 and h.json()['ok']
    c = client.get('/api/catalog')
    assert c.status_code == 200
    body = c.json()
    assert body['modes'] == ['tx', 'files', 'spectrum', 'rx', 'blind', 'radar']
    assert 'demos' not in body
    # Every communications waveform is listed exactly once and every listed name is valid.
    from vectorrf.models import RADAR_WAVEFORMS, WaveformKind
    listed = [w for group in body['waveforms'] for w in group['items']]
    assert len(listed) == len(set(listed))
    assert set(listed) == {w.value for w in WaveformKind if w not in RADAR_WAVEFORMS}
    assert body['agility']['max_hops'] > 0 and body['agility']['max_components'] == 16
    assert 'fmcw' in body['radar']['continuous']
    assert 'lfm' in body['radar']['pulses']
    assert 'iq_formats' not in body


def test_live_websocket_protocol_header():
    with client.websocket_connect('/ws/live') as ws:
        hello = ws.receive_json()
        assert hello['protocol'] == 'VRF1'
        ws.send_json({"action":"start","tx":{"waveform":"qpsk","sample_rate":250000,"duration":0.05,"symbol_rate":12500},"spectrum":{"rbw_hz":2000}})
        assert ws.receive_json()["type"] == "accepted"
        meta = ws.receive_json()
        assert meta['type'] == 'meta'
        metrics = ws.receive_json()
        assert metrics["type"] == "metrics" and metrics["freq_hz"]
        frame = ws.receive_bytes()
        assert frame[:4] == b'VRF1'
        assert len(frame) > 16


def test_live_socket_refuses_other_origins():
    from starlette.websockets import WebSocketDisconnect
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/ws/live', headers={'origin': 'http://evil.example'}) as ws:
            ws.receive_json()
    with client.websocket_connect('/ws/live', headers={'origin': 'http://testserver'}) as ws:
        assert ws.receive_json()['type'] == 'hello'


def test_defaults_cover_every_editable_model():
    d = client.get('/api/defaults').json()
    assert {'tx', 'rx', 'channel', 'spectrum', 'agility', 'radar_experiment', 'radar_design', 'radar_processing'} <= set(d)
    assert d['spectrum']['window'] == 'blackmanharris' and d['radar_processing']['detector'] == 'ca-cfar'


def test_radar_waveform_metadata():
    from vectorrf.signal import generate
    from vectorrf.models import TxConfig
    g = generate(TxConfig(waveform="frank", sample_rate=500000, duration=0.01, pulse_width_s=80e-6, pri_s=250e-6,
                          polyphase_order=4, snr_db=None))
    assert g.metadata['code_length'] == 16 and g.metadata['pulse_count'] == 40


def test_radar_studio_designs_drive_the_simulation():
    design = {"mode": "sequence", "sample_rate": 500000, "duration_s": 0.006, "sequence": [{
        "kind": "pulse", "name": "LFM", "repeat": 5, "pri_s": 200e-6,
        "pulse": {"name": "LFM", "waveform": "lfm", "width_s": 40e-6, "sweep_bw_hz": 100000}}]}
    r = client.post('/api/studio/radar', json={"source": "legacy", "legacy": design, "sample_rate": 500000, "samples": 4096})
    assert r.status_code == 200
    body = r.json()
    assert body['design']['items'][0]['compression_ratio'] == pytest.approx(4.0)
    assert len(body['autocorr_db']) > 10 and len(body['tx_frequency_hz']) > 10


def test_radar_studio_continuous():
    from vectorrf.radar_studio import RadarStudioConfig, generate_radar
    result = generate_radar(RadarStudioConfig(mode="continuous", continuous_waveform="stepped-cw", sample_rate=500000,
                                              duration_s=0.004, stepped_start_hz=-50000, stepped_stop_hz=50000,
                                              stepped_points=8, stepped_dwell_s=20e-6))
    assert result.metadata['waveform'] == 'stepped-cw' and len(result.samples) == 2000


def test_file_routes_are_not_exposed():
    # File I/O is intentionally outside the current internal-workflow baseline.
    assert client.post('/api/tx/export', json={}).status_code == 404
    assert client.post('/api/iq/blind').status_code == 404
    assert client.post('/api/radar/blind', json={}).status_code == 404
