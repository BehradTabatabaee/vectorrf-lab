"""Live captures are contiguous, advance, and are the same samples every observer sees."""
import json

import numpy as np
from fastapi.testclient import TestClient

from vectorrf.app import app
from vectorrf.models import TxConfig
from vectorrf.source import source_record

client = TestClient(app)


def _collect(ws, wanted, count, ack=True):
    out = []
    while len(out) < count:
        msg = ws.receive()
        if msg.get("text") is None:
            continue
        d = json.loads(msg["text"])
        if d["type"] == wanted:
            out.append(d)
        if d["type"] == "frame_end" and ack:
            ws.send_json({"action": "ack", "seq": d["seq"]})
    return out


def test_transmitter_captures_are_exactly_contiguous_in_absolute_samples():
    tx = TxConfig(waveform="fm", sample_rate=1e6, duration=.0123456, snr_db=None, normalize_output=False, amplitude=1)
    a, b = source_record(tx, 41), source_record(tx, 42)
    n = len(a.samples)
    assert a.metadata["start_sample"] == 41 * n and b.metadata["start_sample"] == 42 * n
    long = TxConfig(**{**tx.model_dump(), "duration": 2 * n / 1e6})
    from vectorrf.signal import generate
    joined = generate(long, start_sample=41 * n).samples
    np.testing.assert_allclose(np.r_[a.samples, b.samples], joined, atol=1e-5)


def test_chain_active_stream_consumes_one_new_capture_per_frame():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "pipeline": {"blocks": []},
                      "tx": {"waveform": "tone", "sample_rate": 500_000, "duration": .004}})
        products = _collect(ws, "pipeline", 4)
    ids = [p["capture_id"] for p in products]
    assert np.all(np.diff(ids) == 1)


def test_spectrum_only_stream_advances_after_showing_whole_capture_and_observers_match():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True,
                      "tx": {"waveform": "qpsk", "sample_rate": 1e6, "duration": .01, "snr_db": 20},
                      "spectrum": {"rbw_hz": 5000, "averaging": 1}})
        metrics = _collect(ws, "metrics", 12)
    ids = [m["capture_id"] for m in metrics]
    assert ids == sorted(ids) and ids[-1] > ids[0], ids
    assert len(set(ids)) < len(ids)  # several spectrum frames per capture
    # An HTTP observer naming the reported capture sees the same record.
    last = metrics[-1]
    tx = {"waveform": "qpsk", "sample_rate": 1e6, "duration": .01, "snr_db": 20}
    z = client.post("/api/measurements", json={"tx": tx, "capture_id": last["capture_id"]}).json()
    assert z["capture_start_sample"] == last["capture_start_sample"]
    assert z["source_revision"] == last["source_revision"]


def _wall_clock_frames(pacing, chain, seconds=1.2):
    """Frames over ``seconds`` of wall time: (elapsed, capture id, capture duration)."""
    import time
    tx = {"waveform": "tone", "sample_rate": 500_000, "duration": .004}
    frames = []
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "pacing": pacing, "tx": tx,
                      "pipeline": {"blocks": []} if chain else None, "spectrum": {"rbw_hz": 5000, "averaging": 1}})
        start = time.monotonic()
        while time.monotonic() - start < seconds:
            msg = ws.receive()
            if msg.get("text") is None:
                continue  # binary spectrum frame
            d = json.loads(msg["text"])
            if d["type"] == "metrics":
                frames.append((time.monotonic() - start, d["capture_id"], d["pacing"]))
            elif d["type"] == "frame_end":
                ws.send_json({"action": "ack", "seq": d["seq"]})
    return frames, tx["duration"]


def test_realtime_pacing_follows_the_wall_clock_and_skips_captures():
    frames, duration = _wall_clock_frames("realtime", chain=False)
    assert all(p == "realtime" for _, _, p in frames)
    elapsed, first, last = frames[-1][0] - frames[0][0], frames[0][1], frames[-1][1]
    # About one capture per capture duration of wall time, far more than frames shown.
    assert abs((last - first) * duration - elapsed) < 0.25, (first, last, elapsed)
    assert last - first > len(frames)


def test_realtime_pacing_never_feeds_a_chain_the_same_capture_twice():
    frames, _ = _wall_clock_frames("realtime", chain=True)
    ids = [c for _, c, _ in frames]
    assert len(ids) > 5 and all(b > a for a, b in zip(ids, ids[1:])), ids


def test_contiguous_pacing_is_the_api_default_and_bad_pacing_is_rejected():
    frames, _ = _wall_clock_frames("contiguous", chain=True, seconds=0.6)
    ids = [c for _, c, _ in frames]
    assert all(b - a == 1 for a, b in zip(ids, ids[1:])), ids
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "pacing": "warp", "tx": {"waveform": "tone"}})
        errors = [m for m in (json.loads(t) for t in [ws.receive_text()]) if m["type"] == "error"]
        assert errors and "pacing" in errors[0]["message"]
