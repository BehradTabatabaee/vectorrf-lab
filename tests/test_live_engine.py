"""Live engine: real-time pacing keeps the display rate and drops what it cannot process.

Under real-time pacing every live view updates at the display rate.  A
product that cannot process a whole capture in one frame period works on the
prefix of the capture it can afford and the rest of the capture is dropped,
like an SDR dropping buffers.  The prefix is an exact slice of the capture
every other observer sees.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorrf.app import app
from vectorrf.engine import MIN_WORK_SAMPLES, _WorkSize
from vectorrf.models import TxConfig
from vectorrf.source import source_record

client = TestClient(app)


def _collect(ws, wanted, count, ack=True, limit_s=60):
    out, end = [], time.monotonic() + limit_s
    while len(out) < count and time.monotonic() < end:
        msg = ws.receive()
        if msg.get("text") is None:
            continue
        d = json.loads(msg["text"])
        if d["type"] == wanted:
            out.append(d)
        if d["type"] == "frame_end" and ack:
            ws.send_json({"action": "ack", "seq": d["seq"]})
    return out


def test_a_capture_prefix_is_the_exact_start_of_the_capture():
    tx = TxConfig(waveform="16qam", sample_rate=20e6, symbol_rate=400e3, duration=0.01, snr_db=18)
    full, prefix = source_record(tx, 9), source_record(tx, 9, limit=50_000)
    assert len(prefix.samples) == 50_000
    np.testing.assert_array_equal(prefix.samples, full.samples[:50_000])
    assert prefix.metadata["start_sample"] == full.metadata["start_sample"]


def test_work_size_tracks_the_frame_budget():
    work = _WorkSize()
    capture = 1 << 20
    assert work.limit(capture, 30) is None                      # unknown cost: whole capture
    work.observe(1 << 20, 400.0)                                 # 400 ms for a full capture
    limit = work.limit(capture, 30)
    assert MIN_WORK_SAMPLES <= limit < capture
    assert limit * work.cost_ms <= 1.1 * 1000 / 30 + 1e-9        # fits the per-frame budget
    for _ in range(20):
        work.observe(limit, limit * 1e-6)                        # suddenly cheap
        limit = work.limit(capture, 30) or capture
    assert limit == capture                                      # grows back to whole captures
    slow = _WorkSize()
    slow.observe(1000, 1000.0)                                   # absurdly slow: never below the floor
    assert slow.limit(capture, 30) == MIN_WORK_SAMPLES


def test_realtime_receiver_drops_samples_and_reports_it():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "pacing": "realtime", "fps": 30,
                      "tx": {"waveform": "qpsk", "sample_rate": 200e6, "symbol_rate": 1e6, "duration": 524288 / 200e6, "snr_db": 30},
                      "rx": {"modulation": "qpsk", "sample_rate": 200e6, "symbol_rate": 1e6, "timing_recovery": "mueller-muller"}})
        metrics = _collect(ws, "metrics", 25)
        products = _collect(ws, "receiver", 5)
    assert all(m["window_samples"] == 524288 for m in metrics)
    assert all(MIN_WORK_SAMPLES <= m["processed_samples"] <= m["window_samples"] for m in metrics)
    ids = [p["capture_id"] for p in products]
    assert ids == sorted(set(ids)), ids                          # never the same capture twice
    assert all(p["metrics"]["symbol_count"] > 0 for p in products)


def test_every_sample_pacing_never_drops():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "pacing": "contiguous",
                      "tx": {"waveform": "qpsk", "sample_rate": 50e6, "symbol_rate": 1e6, "duration": 0.004, "snr_db": 30},
                      "rx": {"modulation": "qpsk", "sample_rate": 50e6, "symbol_rate": 1e6}})
        metrics = _collect(ws, "metrics", 6)
        products = _collect(ws, "receiver", 4)
    assert all(m["processed_samples"] == m["window_samples"] == 200_000 for m in metrics)
    ids = [p["capture_id"] for p in products]
    assert np.all(np.diff(ids) == 1), ids


def test_every_sample_zero_span_sweeps_consecutive_intervals():
    """Every sample: sweep after sweep covers the stream without gaps, ten
    2 ms sweeps per 20 ms capture, then the next capture."""
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "pacing": "contiguous",
                      "tx": {"waveform": "tone", "sample_rate": 1e6, "duration": 0.02},
                      "measurement": {"offset_s": 0.001, "duration_s": 0.002}})
        metrics = _collect(ws, "metrics", 4)
        products = _collect(ws, "measurement", 14)
    assert all(m["processed_samples"] == m["window_samples"] == 20_000 for m in metrics)
    starts = [p["capture_start_sample"] + round(p["capture_start_s"] * 1e6) for p in products]
    assert np.all(np.diff(starts) == 2000), starts
    assert all(p["capture_duration_s"] == 0.002 and not p["interval_clipped"] for p in products)


@pytest.mark.parametrize("pacing", ["contiguous", "realtime"])
def test_overlapping_low_rate_file_windows_advance_by_the_step(tmp_path, monkeypatch, pacing):
    from vectorrf import recordings

    monkeypatch.setenv("VECTORRF_CAPTURES_DIR", str(tmp_path / "captures"))
    recordings.close_readers()
    try:
        library = recordings.library_dir()
        (library / "slow.cf32").write_bytes(np.ones(1024, dtype=np.complex64).tobytes())
        with client.websocket_connect("/ws/live") as ws:
            ws.receive_json()
            ws.send_json({"action": "start", "acknowledge_frames": True, "pacing": pacing,
                          "source": {"name": "slow.cf32", "sample_rate": 1001,
                                     "capture_samples": 64, "step_samples": 33, "loop": True},
                          "pipeline": {"blocks": []}})
            metrics = _collect(ws, "metrics", 5)
        assert all(m["window_samples"] == m["processed_samples"] == 64 for m in metrics)
        steps = np.diff([m["capture_start_sample"] for m in metrics])
        if pacing == "contiguous":
            assert np.all(steps == 33)
        else:
            assert np.all((steps > 0) & (steps % 33 == 0))
    finally:
        recordings.close_readers()


def test_zero_span_and_quality_are_live_products():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "pacing": "realtime",
                      "tx": {"waveform": "qpsk", "sample_rate": 1e6, "duration": 0.02, "snr_db": 30},
                      "measurement": {"offset_s": 0.001, "duration_s": 0.002}})
        zero = _collect(ws, "measurement", 3)
        ws.send_json({"action": "update", "measurement": {"offset_s": 0, "duration_s": 0.002, "quality": True}})
        quality = [m for m in _collect(ws, "measurement", 6) if m["quality"] is not None]
    assert all(z["quality"] is None and len(z["i"]) > 100 and "i_max" in z for z in zero)
    assert zero[0]["capture_start_s"] == 0.001
    assert quality and quality[-1]["quality"]["kind"] == "evm"
    assert quality[-1]["quality"]["metrics"]["evm_rms_percent"] < 8


def test_live_products_and_metrics_carry_the_edit_revision():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "client_revision": 7,
                      "tx": {"waveform": "tone", "sample_rate": 1e6, "duration": 0.01},
                      "measurement": {"duration_s": 0.002}})
        first = {}
        while not {"accepted", "metrics", "measurement"} <= first.keys():
            msg = ws.receive()
            if msg.get("text") is None:
                continue
            data = json.loads(msg["text"])
            first.setdefault(data["type"], data)
            if data["type"] == "frame_end":
                ws.send_json({"action": "ack", "seq": data["seq"]})
        assert all(first[k]["client_revision"] == 7 for k in ("accepted", "metrics", "measurement"))
        ws.send_json({"action": "update", "client_revision": 8,
                      "measurement": {"duration_s": 0.001}})
        second = {}
        while not {"accepted", "metrics", "measurement"} <= second.keys():
            msg = ws.receive()
            if msg.get("text") is None:
                continue
            data = json.loads(msg["text"])
            if data["type"] in {"accepted", "metrics", "measurement"} and data.get("client_revision") == 8:
                second.setdefault(data["type"], data)
            if data["type"] == "frame_end":
                ws.send_json({"action": "ack", "seq": data["seq"]})
    assert second["measurement"]["capture_duration_s"] == 0.001


def test_measurement_errors_disable_only_that_product():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True,
                      "tx": {"waveform": "tone", "sample_rate": 1e6, "duration": 0.01},
                      "measurement": {"duration_s": -1}})
        errors, metrics = [], []
        while len(metrics) < 3:
            msg = ws.receive()
            if msg.get("text") is None:
                continue  # binary spectrum frames
            d = json.loads(msg["text"])
            if d["type"] == "error":
                errors.append(d)
            if d["type"] == "metrics":
                metrics.append(d)
            if d["type"] == "frame_end":
                ws.send_json({"action": "ack", "seq": d["seq"]})
    assert errors and errors[0]["scope"] == "measurement" and "Measurement" in errors[0]["message"]


def test_runtime_measurement_fault_reports_once_and_recovers_on_edit():
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "client_revision": 1,
                      "tx": {"waveform": "tone", "sample_rate": 1e6, "duration": 0.01},
                      "measurement": {"center_hz": 700e3, "bandwidth_hz": 10e3}})
        errors, metrics = [], []
        while len(metrics) < 6:
            msg = ws.receive()
            if msg.get("text") is None:
                continue
            data = json.loads(msg["text"])
            if data["type"] == "error":
                errors.append(data)
            if data["type"] == "metrics":
                metrics.append(data)
            if data["type"] == "frame_end":
                ws.send_json({"action": "ack", "seq": data["seq"]})
        assert len(errors) == 1 and errors[0]["scope"] == "product"
        assert "Nyquist" in errors[0]["message"]
        assert all(m["client_revision"] == 1 for m in metrics)
        ws.send_json({"action": "update", "client_revision": 2,
                      "measurement": {"center_hz": 0, "bandwidth_hz": 10e3}})
        repaired = None
        while repaired is None:
            msg = ws.receive()
            if msg.get("text") is None:
                continue
            data = json.loads(msg["text"])
            if data["type"] == "measurement" and data["client_revision"] == 2:
                repaired = data
            if data["type"] == "frame_end":
                ws.send_json({"action": "ack", "seq": data["seq"]})
    assert repaired["bandwidth_actual_hz"] == 10e3


def test_transient_product_fault_retries_and_clears_without_an_edit(monkeypatch):
    import vectorrf.engine as engine

    original = engine.measurement_product
    calls = 0

    def flaky(*args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("temporary product failure")
        return original(*args)

    monkeypatch.setattr(engine, "measurement_product", flaky)
    monkeypatch.setattr(engine, "PRODUCT_RETRY_S", 0.05)
    with client.websocket_connect("/ws/live") as ws:
        ws.receive_json()
        ws.send_json({"action": "start", "acknowledge_frames": True, "client_revision": 3,
                      "tx": {"waveform": "tone", "sample_rate": 1e6, "duration": 0.01},
                      "measurement": {"duration_s": 0.002}})
        seen = set()
        while not {"error", "measurement", "product_recovered"} <= seen:
            msg = ws.receive()
            if msg.get("text") is None:
                continue
            data = json.loads(msg["text"])
            seen.add(data["type"])
            if data["type"] == "frame_end":
                ws.send_json({"action": "ack", "seq": data["seq"]})
    assert calls >= 2
