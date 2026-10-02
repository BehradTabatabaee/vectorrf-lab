from fastapi.testclient import TestClient
from vectorrf.app import app

client = TestClient(app)


def test_health_and_catalog():
    h = client.get('/api/health')
    assert h.status_code == 200 and h.json()['ok']
    c = client.get('/api/catalog')
    assert c.status_code == 200
    body = c.json()
    assert body['modes'] == ['tx', 'spectrum', 'rx', 'blind', 'radar']
    assert 'demos' not in body
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


def test_radar_preview_endpoint():
    r = client.post('/api/radar/preview', json={
        "waveform":"frank", "sample_rate":500000, "duration":0.01,
        "pulse_width_s":80e-6, "pri_s":250e-6, "polyphase_order":4,
        "sweep_bw_hz":100000, "snr_db":None,
    })
    assert r.status_code == 200
    body = r.json()
    assert body['metadata']['code_length'] == 16
    assert len(body['autocorr_db']) > 2
    assert len(body['inst_freq_hz']) > 10


def test_radar_studio_sequence_and_ambiguity():
    r = client.post('/api/radar/studio', json={
        "mode": "sequence",
        "sample_rate": 500000,
        "duration_s": 0.006,
        "sequence": [{
            "kind": "pulse", "name": "LFM", "repeat": 5, "pri_s": 200e-6,
            "pulse": {"name": "LFM", "waveform": "lfm", "width_s": 40e-6, "sweep_bw_hz": 100000}
        }]
    })
    assert r.status_code == 200
    body = r.json()
    assert body['metadata']['mode'] == 'sequence'
    assert body['metadata']['timeline']
    assert len(body['ambiguity_db']) == body['ambiguity_shape'][0] * body['ambiguity_shape'][1]
    assert len(body['envelope']) > 10


def test_radar_studio_continuous():
    r = client.post('/api/radar/studio', json={
        "mode": "continuous", "continuous_waveform": "stepped-cw",
        "sample_rate": 500000, "duration_s": 0.004,
        "stepped_start_hz": -50000, "stepped_stop_hz": 50000,
        "stepped_points": 8, "stepped_dwell_s": 20e-6,
    })
    assert r.status_code == 200
    assert r.json()['metadata']['waveform'] == 'stepped-cw'


def test_file_routes_are_not_exposed():
    # File I/O is intentionally outside the current internal-workflow baseline.
    assert client.post('/api/tx/export', json={}).status_code == 404
    assert client.post('/api/iq/blind').status_code == 404
    assert client.post('/api/radar/blind', json={}).status_code == 404
