"""Recording import/export: formats, containers, library API and file-driven workflows."""
import json
import struct
import tarfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from vectorrf import iqfile, recordings
from vectorrf.app import app
from vectorrf.iqfile import RecordingReader, SampleFormat, guess_from_filename, inspect, sniff_raw
from vectorrf.models import TxConfig
from vectorrf.signal import generate


@pytest.fixture()
def library(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTORRF_CAPTURES_DIR", str(tmp_path / "captures"))
    recordings.close_readers()
    yield recordings.library_dir()
    recordings.close_readers()


@pytest.fixture()
def client(library):
    return TestClient(app)


def _signal(n=20_000, seed=3):
    """Band-limited complex test signal with |I|,|Q| < 1 (correlated, like real captures)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n) + 1j * rng.normal(size=n)
    x = np.convolve(x, np.hanning(17), mode="same")
    return (0.4 * x / np.max(np.abs(np.r_[x.real, x.imag]))).astype(np.complex64)


# ---------------------------------------------------------------------------
# Sample formats


@pytest.mark.parametrize("name, step", [("cf32_le", 1e-7), ("cf64_le", 1e-7), ("cf32_be", 1e-7), ("ci32_le", 2 ** -31),
                                        ("ci16_le", 2 ** -15), ("ci16_be", 2 ** -15), ("ci8", 2 ** -7), ("cu8", 2 ** -7)])
def test_sample_formats_roundtrip_within_one_lsb(name, step):
    fmt = SampleFormat.parse(name)
    x = _signal(4096)
    raw, clipped = fmt.encode(x)
    assert clipped == 0
    y = fmt.decode(np.frombuffer(raw.tobytes(), dtype=fmt.dtype))
    assert np.max(np.abs(y - x)) <= step * 0.75 + 1e-7
    assert fmt.bytes_per_sample * len(x) == len(raw.tobytes())


def test_sample_format_names_scale_and_offset_binary():
    assert SampleFormat.parse("CF32").name == "cf32_le"
    assert SampleFormat.parse("ci8").bytes_per_sample == 2
    cu8 = SampleFormat.parse("cu8")
    # RTL-SDR offset binary: codes 127/128 straddle zero symmetrically.
    np.testing.assert_allclose(cu8.decode(np.array([127, 128, 0, 255], np.uint8)), [-0.5 / 128 + 0.5j / 128, -127.5 / 128 + 127.5j / 128])
    full = SampleFormat.parse("ci16_le").decode(np.array([-32768, 32767], "<i2"))
    assert full[0] == pytest.approx(-1 + (32767 / 32768) * 1j)
    with pytest.raises(ValueError):
        SampleFormat.parse("cq16")
    with pytest.raises(ValueError):
        SampleFormat.parse("ci64_le")


def test_integer_encoding_reports_clipping():
    raw, clipped = SampleFormat.parse("ci16_le").encode(np.array([1.5 + 0j, -2j, 0.5], np.complex64))
    assert clipped == 2
    assert raw.max() == 32767 and raw.min() == -32768


# ---------------------------------------------------------------------------
# Containers


def test_raw_reader_windows_wrap_and_end(tmp_path):
    x = _signal(10_000)
    path = tmp_path / "test.cf32"
    path.write_bytes(x.tobytes())
    r = RecordingReader(inspect(path, sample_rate=1e6))
    assert r.num_samples == 10_000
    np.testing.assert_array_equal(r.read(1234, 100), x[1234:1334])
    np.testing.assert_array_equal(r.read(9950, 100), x[9950:])
    np.testing.assert_array_equal(r.read_wrapped(9950, 100), np.r_[x[9950:], x[:50]])
    assert len(r.read(20_000, 10)) == 0
    r.close()


def test_header_offset_and_trailing_bytes(tmp_path):
    x = _signal(1000)
    path = tmp_path / "skip.bin"
    path.write_bytes(b"HEADER!!" + SampleFormat.parse("ci16_le").encode(x)[0].tobytes() + b"\x01")
    info = inspect(path, "ci16_le", 2e6, header_bytes=8)
    assert info.num_samples == 1000
    assert any("trailing" in n for n in info.notes)
    np.testing.assert_allclose(RecordingReader(info).read(0, 1000), x, atol=2 ** -15)


@pytest.mark.parametrize("fmt_name", ["ci16_le", "cf32_le", "cu8"])
def test_wav_iq_roundtrip_with_center_frequency(tmp_path, fmt_name):
    x = _signal(5000)
    data, clipped = iqfile.encode_recording(x, SampleFormat.parse(fmt_name), 2_048_000, "wav", 1.0, center_hz=433_920_000)
    assert clipped == 0
    path = tmp_path / "sdr_IQ.wav"
    path.write_bytes(data)
    info = inspect(path)
    assert info.container == "wav" and info.sample_rate == 2_048_000
    assert info.center_frequency_hz == 433_920_000  # SpectraVue/SDR# 'auxi' chunk
    assert info.num_samples == 5000
    tol = {"ci16_le": 2 ** -15, "cf32_le": 1e-7, "cu8": 2 ** -7}[fmt_name]
    np.testing.assert_allclose(RecordingReader(info).read(0, 5000), x, atol=tol)


def test_wav_8bit_pcm_uses_128_as_zero(tmp_path):
    data, _ = iqfile.encode_recording(np.zeros(16, np.complex64), SampleFormat(True, "u", 8, unsigned_zero=128.0), 8000, "wav")
    assert set(data[-32:]) == {128}
    path = tmp_path / "zero.wav"
    path.write_bytes(data)
    np.testing.assert_array_equal(RecordingReader(inspect(path)).read(0, 16), 0)


def _riff(chunks, rf64=False, data_size=None):
    body = b"WAVE" + b"".join(chunks)
    if rf64:
        return b"RF64" + struct.pack("<I", 0xFFFFFFFF) + body
    return b"RIFF" + struct.pack("<I", len(body)) + body


def test_wav_24bit_extensible_mono_and_rf64(tmp_path):
    rng = np.random.default_rng(0)
    vals = rng.integers(-(1 << 23), 1 << 23, size=2 * 300)
    packed = b"".join(int(v & 0xFFFFFF).to_bytes(3, "little") for v in vals)
    ext = struct.pack("<HHIIHHHHI16s", 0xFFFE, 2, 96000, 96000 * 6, 6, 24, 22, 24, 3, struct.pack("<H", 1) + bytes(14))
    fmt_chunk = b"fmt " + struct.pack("<I", len(ext)) + ext
    data_chunk = b"data" + struct.pack("<I", len(packed)) + packed
    p24 = tmp_path / "x24.wav"
    p24.write_bytes(_riff([fmt_chunk, data_chunk]))
    got = RecordingReader(inspect(p24)).read(0, 300)
    np.testing.assert_allclose(got, (vals[0::2] + 1j * vals[1::2]) / (1 << 23), atol=1e-7)

    mono = struct.pack("<HHIIHH", 1, 1, 8000, 16000, 2, 16)
    pm = tmp_path / "mono.wav"
    pm.write_bytes(_riff([b"fmt " + struct.pack("<I", 16) + mono, b"data" + struct.pack("<I", 8) + np.array([0, 16384, -16384, 0], "<i2").tobytes()]))
    info = inspect(pm)
    assert not info.sample_format.complex and any("real" in n for n in info.notes)
    np.testing.assert_allclose(RecordingReader(info).read(0, 4), [0, 0.5, -0.5, 0])

    stereo = struct.pack("<HHIIHH", 3, 2, 10_000_000, 80_000_000, 8, 32)
    x = _signal(64)
    payload = np.column_stack([x.real, x.imag]).astype("<f4").tobytes()
    ds64 = b"ds64" + struct.pack("<IQQQI", 28, 0, len(payload), 64, 0)
    prf = tmp_path / "big.wav"
    prf.write_bytes(_riff([ds64, b"fmt " + struct.pack("<I", 16) + stereo, b"data" + struct.pack("<I", 0xFFFFFFFF) + payload], rf64=True))
    info = inspect(prf)
    assert info.extra["rf64"] and info.num_samples == 64
    np.testing.assert_allclose(RecordingReader(info).read(0, 64), x, atol=1e-7)


def test_rf64_header_is_written_beyond_4gib():
    header = iqfile._wav_header(SampleFormat.parse("cf32_le"), 2e9, (1 << 30), 2.4e9)
    assert header[:4] == b"RF64" and header[12:16] == b"ds64"
    data_size = struct.unpack("<Q", header[28:36])[0]
    assert data_size == (1 << 30) * 8


def test_wav_rejects_rates_it_cannot_store():
    with pytest.raises(ValueError, match="SigMF"):
        iqfile.encode_recording(np.zeros(8, np.complex64), SampleFormat.parse("ci16_le"), 5e9, "wav")


def test_sigmf_pair_and_archive(tmp_path):
    x = _signal(3000)
    fmt = SampleFormat.parse("ci16_le")
    data = fmt.encode(x)[0].tobytes()
    meta = iqfile.sigmf_meta(fmt, 20e6, 915e6, "unit test", {"vectorrf:full_scale_dbm": -10.0})
    (tmp_path / "rec.sigmf-data").write_bytes(data)
    (tmp_path / "rec.sigmf-meta").write_text(json.dumps(meta))
    info = inspect(tmp_path / "rec.sigmf-meta")
    assert (info.container, info.sample_rate, info.center_frequency_hz, info.num_samples) == ("sigmf", 20e6, 915e6, 3000)
    assert info.extra["vectorrf:full_scale_dbm"] == -10.0
    np.testing.assert_allclose(RecordingReader(info).read(0, 3000), x, atol=2 ** -15)
    archive = tmp_path / "rec.sigmf"
    archive.write_bytes(iqfile.sigmf_archive("rec", meta, data))
    with tarfile.open(archive) as tar:
        assert sorted(m.name for m in tar.getmembers()) == ["rec/rec.sigmf-data", "rec/rec.sigmf-meta"]
    ainfo = inspect(archive)
    assert ainfo.path == archive and ainfo.num_samples == 3000
    np.testing.assert_allclose(RecordingReader(ainfo).read(100, 50), x[100:150], atol=2 ** -15)


@pytest.mark.parametrize("truth", ["cf32_le", "ci16_le", "ci8", "cu8"])
def test_raw_format_sniffer_ranks_the_true_encoding_first(tmp_path, truth):
    fmt = SampleFormat.parse(truth)
    x = _signal(200_000, seed=11)
    path = tmp_path / "unknown.iq"
    path.write_bytes(fmt.encode(x)[0].tobytes())
    ranked = sniff_raw(path)
    assert ranked[0]["format"] == truth, ranked


def test_filename_hints():
    assert guess_from_filename("gqrx_20240101_120000_433920000_2000000_fc.raw") == {
        "center_frequency_hz": 433_920_000.0, "sample_rate": 2_000_000.0, "source": "gqrx file name"}
    assert guess_from_filename("SDRSharp_20240101_120000Z_100500000Hz_IQ.wav")["center_frequency_hz"] == 100_500_000
    assert guess_from_filename("capture_2.5Msps.cs16")["sample_rate"] == 2.5e6
    assert guess_from_filename("wideband_2Gsps.cf32")["sample_rate"] == 2e9


# ---------------------------------------------------------------------------
# Library API


@pytest.mark.parametrize("name", ["../escape.iq", "a/b.iq", "..\\x.iq", ".hidden.iq", "x.exe", "", "con/..iq"])
def test_library_rejects_unsafe_names(library, name):
    with pytest.raises(ValueError):
        recordings.resolve(name, must_exist=False)


def test_upload_list_inspect_overview_delete(client, library):
    x = _signal(50_000)
    body = SampleFormat.parse("ci16_le").encode(x)[0].tobytes()
    r = client.put("/api/files/upload", params={"name": "probe_1Msps.iq"}, content=body)
    assert r.status_code == 200, r.text
    assert r.json()["name"] == "probe_1Msps.iq" and r.json()["bytes"] == len(body)
    again = client.put("/api/files/upload", params={"name": "probe_1Msps.iq"}, content=body).json()
    assert again["name"] == "probe_1Msps (2).iq"  # never overwrites silently
    listing = client.get("/api/files").json()
    assert {f["name"] for f in listing["files"]} == {"probe_1Msps.iq", "probe_1Msps (2).iq"}
    entry = next(f for f in listing["files"] if f["name"] == "probe_1Msps.iq")
    assert entry["needs_format"] and entry["sample_rate"] == 1e6  # rate parsed from the file name
    info = client.post("/api/files/inspect", json={"name": "probe_1Msps.iq"}).json()
    assert info["format_candidates"][0]["format"] == "ci16_le"
    ov = client.post("/api/files/overview", json={"source": {"name": "probe_1Msps.iq", "format": "ci16_le"}, "rows": 32, "nfft": 256})
    assert ov.status_code == 200
    d = ov.json()
    assert d["spectrogram_shape"] == [32, 256] and len(d["spectrogram_db"]) == 32 * 256
    assert 0 < d["coverage"] <= 1
    assert client.delete("/api/files/probe_1Msps (2).iq").json()["removed"] == ["probe_1Msps (2).iq"]
    assert client.put("/api/files/upload", params={"name": "../evil.iq"}, content=b"0000").status_code == 422
    assert client.post("/api/files/inspect", json={"name": "missing.iq"}).status_code == 422


def test_library_lists_a_sigmf_pair_once_with_its_full_size(client, library):
    x = _signal(40_000)
    fmt = SampleFormat.parse("ci16_le")
    data, _ = iqfile.encode_recording(x, fmt, 1e6, "raw", 1.0)
    (library / "pair.sigmf-data").write_bytes(data)
    (library / "pair.sigmf-meta").write_text(json.dumps(iqfile.sigmf_meta(fmt, 1e6, None, "pair", {})), encoding="utf-8")
    files = client.get("/api/files").json()["files"]
    assert [f["name"] for f in files] == ["pair.sigmf-meta"]
    on_disk = (library / "pair.sigmf-data").stat().st_size + (library / "pair.sigmf-meta").stat().st_size
    assert files[0]["bytes"] == on_disk and files[0]["data_file"] == "pair.sigmf-data"


def test_full_scale_tone_reads_0_dbfs_and_offset_calibrates(client, library):
    fs = 1e6
    x = (0.5 * np.exp(2j * np.pi * 100e3 * np.arange(65536) / fs)).astype(np.complex64)
    (library / "tone.cf32").write_bytes(x.tobytes())
    src = {"name": "tone.cf32", "sample_rate": fs, "capture_samples": 65536}
    spec = {"rbw_hz": 1000, "averaging": 4, "window": "flattop"}  # flat top: negligible scalloping
    d = client.post("/api/spectrum", json={"source": src, "spectrum": spec}).json()
    assert d["power_unit"] == "dBFS"
    assert d["peak_hz"] == pytest.approx(100e3, abs=d["rbw_actual_hz"])
    assert d["peak_db"] == pytest.approx(20 * np.log10(0.5), abs=0.05)
    d = client.post("/api/spectrum", json={"source": {**src, "power_offset_db": 10, "power_unit": "dBm"}, "spectrum": spec}).json()
    assert d["power_unit"] == "dBm" and d["peak_db"] == pytest.approx(20 * np.log10(0.5) + 10, abs=0.05)


def test_raw_recording_without_rate_asks_for_it(client, library):
    (library / "norate.bin").write_bytes(np.zeros(1000, np.complex64).tobytes())
    r = client.post("/api/spectrum", json={"source": {"name": "norate.bin", "format": "cf32_le"}})
    assert r.status_code == 422 and "sample rate" in r.json()["detail"]


@pytest.mark.parametrize("container, fmt", [("sigmf", "ci16_le"), ("wav", "ci16_le"), ("raw", "cf32_le"), ("sigmf", "cu8")])
def test_export_then_import_decodes_the_same_message(client, library, container, fmt):
    tx = TxConfig(waveform="qpsk", sample_rate=500_000, symbol_rate=25_000, duration=.04, message="ROUND TRIP ", snr_db=None)
    r = client.post("/api/files/export", json={"tx": tx.model_dump(mode="json"), "format": fmt, "container": container})
    assert r.status_code == 200, r.text
    assert int(r.headers["X-VectorRF-Clipped"]) == 0
    name = r.headers["Content-Disposition"].split('filename="')[1].rstrip('"')
    up = client.put("/api/files/upload", params={"name": name}, content=r.content).json()
    src = {"name": up["name"], "capture_samples": 20_000}
    if container == "raw":
        src["sample_rate"] = 500_000
    out = client.post("/api/rx/loopback", json={"source": src, "channel": {"enabled": False},
                                                "rx": {"modulation": "qpsk", "symbol_rate": 25_000, "use_costas": False}}).json()
    assert out["text"].startswith("ROUND TRIP"), out["text"][:40]
    assert out["metrics"]["evm_rms_percent"] < (8 if fmt == "cu8" else 2)


def test_export_records_absolute_calibration_for_reimport(client, library):
    tx = TxConfig(waveform="tone", sample_rate=1e6, duration=.02, power_dbm=-23.5, snr_db=None, center_hz=50_000)
    r = client.post("/api/files/export", json={"tx": tx.model_dump(mode="json"), "format": "ci16_le",
                                               "container": "sigmf", "destination": "library", "name": "cal"})
    assert r.status_code == 200, r.text
    meta_name = r.json()["saved"]
    info = client.post("/api/files/inspect", json={"name": meta_name}).json()
    offset = info["extra"]["vectorrf:full_scale_dbm"]
    d = client.post("/api/spectrum", json={"source": {"name": meta_name, "power_offset_db": offset, "power_unit": "dBm", "capture_samples": 20000},
                                           "spectrum": {"rbw_hz": 1000, "window": "flattop"}}).json()
    assert d["peak_db"] == pytest.approx(-23.5, abs=0.05)


def test_export_validates_limits_and_scaling(client, library):
    from vectorrf.models import MAX_EXPORT_SAMPLES
    big = TxConfig(sample_rate=1e6, duration=.01).model_dump(mode="json")
    big["duration"] = (MAX_EXPORT_SAMPLES + 10) / 1e6
    assert client.post("/api/files/export", json={"tx": big}).status_code == 422
    loud = TxConfig(waveform="tone", sample_rate=1e6, duration=.001, amplitude=1.5, snr_db=None).model_dump(mode="json")
    assert client.post("/api/files/export", json={"tx": loud, "format": "ci16_le", "scaling": "unity"}).status_code == 422
    ok = client.post("/api/files/export", json={"tx": loud, "format": "cf32_le", "scaling": "unity", "container": "raw"})
    x = np.frombuffer(ok.content, np.complex64)
    assert np.max(np.abs(x)) == pytest.approx(1.5, rel=1e-5)


def test_export_excerpt_converts_recording_format(client, library):
    x = _signal(40_000)
    (library / "src.cu8").write_bytes(SampleFormat.parse("cu8").encode(x)[0].tobytes())
    r = client.post("/api/files/export", json={"source": {"name": "src.cu8", "sample_rate": 2.4e6, "start_sample": 1000},
                                               "samples": 5000, "format": "cf32_le", "container": "raw", "scaling": "unity"})
    y = np.frombuffer(r.content, np.complex64)
    np.testing.assert_allclose(y, SampleFormat.parse("cu8").decode(SampleFormat.parse("cu8").encode(x)[0])[1000:6000], atol=1e-7)


def test_file_capture_windows_are_contiguous_and_shared_by_observers(client, library):
    fs = 2e6
    x = _signal(100_000)
    (library / "w.cf32").write_bytes(x.tobytes())
    src = {"name": "w.cf32", "sample_rate": fs, "capture_samples": 10_000, "start_sample": 500}
    a = client.post("/api/measurements", json={"source": src, "capture_id": 3, "duration_s": 10_000 / fs}).json()
    assert a["capture_start_sample"] == 500 + 3 * 10_000
    np.testing.assert_allclose(a["i"][0], x[30_500].real, atol=1e-7)
    tail = client.post("/api/measurements", json={"source": src, "capture_id": 10, "duration_s": 1e-3}).json()
    assert tail["capture_start_sample"] == (500 + 100_000) % 100_000  # loops by default
    stop = client.post("/api/measurements", json={"source": {**src, "loop": False}, "capture_id": 10}).json()
    assert stop["capture_start_sample"] == 90_000  # clamped to the last full window


def test_file_source_feeds_classifier_and_pipeline(client, library):
    tx = TxConfig(waveform="qpsk", sample_rate=500_000, symbol_rate=25_000, duration=.06, data_source="pn", snr_db=30)
    raw = SampleFormat.parse("cf32_le").encode(generate(tx).samples)[0].tobytes()
    (library / "q.cf32").write_bytes(raw)
    src = {"name": "q.cf32", "sample_rate": 500_000, "capture_samples": 30_000}
    blind = client.post("/api/blind/analyze", json={"source": src}).json()
    assert blind["modulation"] == "qpsk" and blind["source_kind"] == "file"
    pipe = client.post("/api/studio/pipeline", json={"source": src, "channel": {"enabled": False},
                                                     "pipeline": {"blocks": [{"kind": "resample", "up": 1, "down": 2}]}}).json()
    assert pipe["output_rate"] == 250_000


def test_live_stream_plays_recording_contiguously(client, library):
    fs = 1e6
    x = _signal(200_000)
    (library / "live.cf32").write_bytes(x.tobytes())
    src = {"name": "live.cf32", "sample_rate": fs, "capture_samples": 8192}
    with client.websocket_connect("/ws/live") as ws:
        assert ws.receive_json()["type"] == "hello"
        ws.send_json({"action": "start", "acknowledge_frames": True, "source": src, "pipeline": {"blocks": []},
                      "spectrum": {"rbw_hz": 2000, "averaging": 1}})
        starts, meta = [], None
        while len(starts) < 4:
            msg = ws.receive()
            if msg.get("text") is None:
                continue
            d = json.loads(msg["text"])
            if d["type"] == "meta":
                meta = d
            if d["type"] == "metrics":
                assert d["power_unit"] == "dBFS" and d["source_kind"] == "file"
                starts.append(d["capture_start_sample"])
            if d["type"] == "frame_end":
                ws.send_json({"action": "ack", "seq": d["seq"]})
        assert meta["source_kind"] == "file" and meta["metadata"]["file"] == "live.cf32"
        assert np.all(np.diff(starts) == 8192)


def test_radar_uses_recorded_waveform_as_transmitter(client, library):
    from vectorrf.radar_sim import RadarExperiment, ArrayModel, SceneTarget, simulate
    fs = 2e6
    t = np.arange(400) / fs
    pulse = np.zeros(1000, np.complex64)
    pulse[:400] = np.exp(1j * np.pi * 1e6 / t[-1] * (t - t[-1] / 2) ** 2)  # LFM pulse, 500 µs PRI
    (library / "pulse.cf32").write_bytes(np.tile(pulse, 4).tobytes())
    exp = RadarExperiment(source="file", file={"name": "pulse.cf32", "sample_rate": fs, "capture_samples": 1000},
                          sample_rate=fs, samples=8192, noise_db=-90, max_range_m=60_000,
                          antenna=ArrayModel(geometry="linear", columns=1, element="isotropic"),
                          targets=[SceneTarget(position=[9000, 0, 0], velocity=[0, 0, 0], gain_db=0)])
    d = simulate(exp)
    assert d["peak_range_m"] == pytest.approx(9000, abs=2 * 299792458 / (2 * fs))
    with pytest.raises(ValueError):
        RadarExperiment(source="file")


def test_sigmf_pair_is_listed_and_addressed_by_its_metadata_name(client, library):
    x = _signal(2000)
    fmt = SampleFormat.parse("cf32_le")
    (library / "pair.sigmf-data").write_bytes(fmt.encode(x)[0].tobytes())
    (library / "pair.sigmf-meta").write_text(json.dumps(iqfile.sigmf_meta(fmt, 1e6, 1e8, "pair")))
    files = client.get("/api/files").json()["files"]
    assert [f["name"] for f in files] == ["pair.sigmf-meta"] and files[0]["data_file"] == "pair.sigmf-data"
    info = client.post("/api/files/inspect", json={"name": "pair.sigmf-meta"}).json()
    assert info["name"] == "pair.sigmf-meta" and info["center_frequency_hz"] == 1e8
    assert client.delete("/api/files/pair.sigmf-meta").json()["removed"] == ["pair.sigmf-meta", "pair.sigmf-data"]


def test_wrong_float_format_is_flagged_and_never_produces_nonfinite_samples(client, library):
    x = _signal(50_000, seed=5)
    (library / "really_cu8.raw").write_bytes(SampleFormat.parse("cu8").encode(x)[0].tobytes())
    info = client.post("/api/files/inspect", json={"name": "really_cu8.raw", "sample_rate": 1e6}).json()
    assert info["format"] == "cf32_le" and "cu8" in info["format_warning"]
    d = client.post("/api/spectrum", json={"source": {"name": "really_cu8.raw", "sample_rate": 1e6, "capture_samples": 4096}})
    assert d.status_code == 200 and all(np.isfinite(d.json()["psd_db"]))
    assert client.post("/api/files/overview", json={"source": {"name": "really_cu8.raw", "sample_rate": 1e6}, "rows": 16, "nfft": 256}).status_code == 200
