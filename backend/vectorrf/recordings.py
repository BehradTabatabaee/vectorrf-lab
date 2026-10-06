"""Recording library and file-backed capture sources.

Recordings live in one managed folder (``VECTORRF_CAPTURES_DIR``).  Network
requests select recordings by file name inside that folder; they never pass
arbitrary file-system paths, mirroring the extension catalog's design.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import os
from pathlib import Path
import re
from threading import RLock
from typing import AsyncIterator, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .iqfile import (AMBIGUOUS_EXTENSIONS, LIBRARY_EXTENSIONS, RecordingInfo, RecordingReader,
                     SampleFormat, inspect, sniff_raw)
from .models import MAX_CAPTURE_SAMPLES, MAX_SAMPLE_RATE
from .signal import GeneratedSignal

_ROOT = Path(__file__).resolve().parents[2]
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+\-]{0,199}$")
MAX_UPLOAD_BYTES = int(os.environ.get("VECTORRF_MAX_UPLOAD_GB", "256")) * (1 << 30)


def library_dir() -> Path:
    """Folder holding recordings; created on first use.

    ``VECTORRF_CAPTURES_DIR`` overrides it.  A source checkout uses
    ``<repository>/captures``; an installed package uses
    ``~/.vectorrf-lab/captures``.
    """
    configured = os.environ.get("VECTORRF_CAPTURES_DIR")
    if configured:
        path = Path(configured).expanduser()
    elif (_ROOT / "pyproject.toml").exists() and (_ROOT / "backend").is_dir():
        path = _ROOT / "captures"
    else:
        path = Path.home() / ".vectorrf-lab" / "captures"
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def safe_name(name: str) -> str:
    """Validate a library file name (no directories, no hidden files)."""
    base = Path(str(name)).name
    if base != str(name) or not _NAME_RE.match(base) or base in {".", ".."}:
        raise ValueError("Use a plain file name: letters, digits, spaces and . _ - ( ) +")
    lower = base.lower()
    if not any(lower.endswith(ext) for ext in LIBRARY_EXTENSIONS):
        raise ValueError("Supported recordings: .iq .bin .dat .raw .cf32 .cfile .cs16 .cs8 .cu8 .wav .sigmf-meta/.sigmf-data")
    return base


def resolve(name: str, must_exist: bool = True) -> Path:
    path = (library_dir() / safe_name(name)).resolve()
    if path.parent != library_dir():
        raise ValueError("Recording must be inside the capture library")
    if must_exist and not path.is_file():
        raise ValueError(f"Recording {name!r} is not in the capture library")
    return path


def list_library() -> list[dict]:
    out = []
    for p in sorted(library_dir().iterdir(), key=lambda q: q.name.lower()):
        if not p.is_file():
            continue
        lower = p.name.lower()
        if lower.endswith(".sigmf-data") or not any(lower.endswith(ext) for ext in LIBRARY_EXTENSIONS):
            continue  # a SigMF pair is listed once, by its metadata file
        item = {"name": p.name, "bytes": p.stat().st_size, "modified": p.stat().st_mtime}
        try:
            info = inspect(p)
            item.update({**info.as_dict(), "name": p.name, "data_file": info.path.name})
            if info.path.resolve() != p.resolve():
                item["bytes"] += info.path.stat().st_size  # a SigMF pair occupies both files
            item["needs_format"] = info.container == "raw" and p.suffix.lower() in AMBIGUOUS_EXTENSIONS
            item["needs_sample_rate"] = info.sample_rate is None
        except (ValueError, OSError) as exc:
            item["error"] = str(exc)[:200]
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Capture specification shared by the live engine and HTTP endpoints


class FileSource(BaseModel):
    """A recording used as the live source, addressed in capture windows.

    Capture ``k`` covers ``capture_samples`` samples starting at
    ``start_sample + k * step_samples`` (wrapping when ``loop`` is set).  With
    the default step equal to the window, successive captures are contiguous.
    """
    model_config = ConfigDict(allow_inf_nan=False)
    kind: Literal["file"] = "file"
    name: str = Field(min_length=1, max_length=200)
    format: str | None = Field(default=None, max_length=16)
    sample_rate: float | None = Field(default=None, gt=0, le=MAX_SAMPLE_RATE)
    center_frequency_hz: float | None = None
    header_bytes: int = Field(default=0, ge=0)
    start_sample: int = Field(default=0, ge=0)
    capture_samples: int = Field(default=65536, ge=64, le=MAX_CAPTURE_SAMPLES)
    step_samples: int | None = Field(default=None, ge=1)
    loop: bool = True
    power_offset_db: float = Field(default=0.0, ge=-400, le=400)
    power_unit: Literal["dBFS", "dBm"] = "dBFS"


_lock = RLock()
_readers: "OrderedDict[tuple, RecordingReader]" = OrderedDict()
_MAX_READERS = 8


def _reader_key(path: Path, spec: FileSource) -> tuple:
    st = path.stat()
    return (str(path), st.st_size, st.st_mtime_ns, spec.format, spec.header_bytes, spec.sample_rate, spec.center_frequency_hz)


def open_reader(spec: FileSource) -> RecordingReader:
    path = resolve(spec.name)
    key = _reader_key(path, spec)
    with _lock:
        reader = _readers.get(key)
        if reader is not None:
            _readers.move_to_end(key)
            return reader
        info = inspect(path, spec.format, spec.sample_rate, spec.center_frequency_hz, spec.header_bytes)
        reader = RecordingReader(info)
        _readers[key] = reader
        while len(_readers) > _MAX_READERS:
            _, old = _readers.popitem(last=False)
            old.close()
        return reader


def close_readers(name: str | None = None) -> None:
    """Release memory maps (needed before deleting/replacing files on Windows)."""
    with _lock:
        for key in list(_readers):
            if name is None or Path(key[0]).name == name or Path(key[0]).name.startswith(Path(name).stem):
                _readers.pop(key).close()


def describe(spec: FileSource) -> dict:
    reader = open_reader(spec)
    info = {**reader.info.as_dict(), "name": spec.name, "data_file": reader.info.path.name}
    if reader.info.container == "raw":
        ranked = sniff_raw(reader.info.path, spec.header_bytes)
        info["format_candidates"] = ranked
        current = next((c["score"] for c in ranked if c["format"] == reader.info.sample_format.name), None)
        best = ranked[0] if ranked else None
        if best and best["format"] != reader.info.sample_format.name and best["score"] > 0.4 and (current is None or current < 0.25 * best["score"]):
            info["format_warning"] = (f"The content looks like {best['format']} ({100*best['score']:.0f}%), but it is being read as "
                                      f"{reader.info.sample_format.name}" + (f" ({100*current:.0f}%)" if current is not None else "")
                                      + ". Choose the sample format explicitly.")
            info["notes"] = [*info["notes"], info["format_warning"]]
    info["revision"] = recording_revision(spec, reader.info)
    return info


def recording_revision(spec: FileSource, info: RecordingInfo) -> str:
    st = info.path.stat()
    key = f"{info.path.name}|{st.st_size}|{st.st_mtime_ns}|{info.sample_format.name}|{info.data_offset}|{info.sample_rate}|{spec.power_offset_db}|{spec.power_unit}"
    return sha256(key.encode()).hexdigest()[:12]


def capture_window(spec: FileSource, capture_id: int, total: int) -> tuple[int, bool]:
    """Start sample of capture ``capture_id`` and whether playback reached the end."""
    step = spec.step_samples or spec.capture_samples
    start = spec.start_sample + int(capture_id) * step
    if total <= 0:
        return 0, True
    if spec.loop:
        return start % total, False
    last = max(0, total - spec.capture_samples)
    return min(start, last), start >= last


def file_capture(spec: FileSource, capture_id: int = 0, limit: int | None = None) -> GeneratedSignal:
    """Read one capture window of a recording (its first ``limit`` samples when given)."""
    reader = open_reader(spec)
    info = reader.info
    if not info.sample_rate:
        raise ValueError(f"Set the sample rate of {spec.name}: raw recordings do not store it")
    total = reader.num_samples
    if total < 32:
        raise ValueError(f"{spec.name} contains fewer than 32 samples")
    count = min(spec.capture_samples, total) if not spec.loop else spec.capture_samples
    if limit is not None:
        count = max(1, min(count, int(limit)))
    start, at_end = capture_window(spec, capture_id, total)
    replaced_before = reader.nonfinite_replaced
    samples = reader.read_wrapped(start, count) if spec.loop else reader.read(start, count)
    samples.flags.writeable = False
    fs = float(info.sample_rate)
    meta = {
        "waveform": "recording", "source_kind": "file", "file": info.path.name, "container": info.container,
        "format": info.sample_format.name, "real_valued": not info.sample_format.complex,
        "sample_rate": fs, "duration": len(samples) / fs, "center_hz": 0.0,
        "center_frequency_hz": info.center_frequency_hz, "start_sample": int(start), "capture_samples": len(samples),
        "recording_samples": int(total), "recording_duration_s": total / fs, "at_end": bool(at_end), "loop": spec.loop,
        "capture_id": int(capture_id), "simulation_time_s": start / fs, "source_seed": 0,
        "source_revision": recording_revision(spec, info), "snr_db": None,
        "power_dbm": float(spec.power_offset_db), "power_reference_rms": 1.0, "power_unit": spec.power_unit,
        "nonfinite_samples_replaced": int(reader.nonfinite_replaced - replaced_before),
        "power_calibration": ("Full scale: a complex tone of magnitude 1.0 reads 0 dBFS"
                              + (f", offset {spec.power_offset_db:+g} dB to {spec.power_unit}" if spec.power_offset_db or spec.power_unit != "dBFS" else "")
                              + "."),
        "playback": "file windows" + (" (looping)" if spec.loop else ""),
    }
    for key in ("description", "datetime"):
        value = getattr(info, key)
        if value:
            meta[f"recording_{key}"] = value
    return GeneratedSignal(samples, fs, meta)


def overview(spec: FileSource, rows: int = 192, nfft: int = 1024, columns: int = 512) -> dict:
    """Whole-recording overview from ``rows`` evenly spaced FFT segments.

    Cost is independent of file length: at most ``rows × nfft`` samples are
    read, so a multi-gigabyte wideband recording still opens instantly.  The
    power trace and spectrogram therefore *sample* the recording; short events
    between segments can be missed (``coverage`` reports the examined fraction).
    """
    from scipy import fft as sfft
    from scipy.signal import windows
    reader = open_reader(spec)
    info = reader.info
    total = reader.num_samples
    if total < 16:
        raise ValueError("Recording is too short for an overview")
    fs = float(info.sample_rate or spec.sample_rate or 1.0)
    nfft = int(min(nfft, 1 << int(np.floor(np.log2(total)))))
    rows = int(max(1, min(rows, total // nfft)))
    starts = np.linspace(0, total - nfft, rows).astype(np.int64)
    w = windows.hann(nfft, sym=False).astype(np.float32)
    block = np.stack([reader.read(int(s), nfft) for s in starts])
    power = np.mean(np.abs(block) ** 2, axis=1)
    peaks = np.max(np.abs(block), axis=1)
    X = sfft.fftshift(sfft.fft(block * w[None, :], axis=1, workers=-1), axes=1)
    # Power per bin relative to a full-scale complex tone (0 dBFS).
    P = (np.abs(X) ** 2) / float(np.sum(w)) ** 2
    if nfft > columns:
        edges = np.linspace(0, nfft, columns + 1).astype(int)
        P = np.maximum.reduceat(P, edges[:-1], axis=1)
        f = sfft.fftshift(sfft.fftfreq(nfft, 1 / fs))
        freq = (f[edges[:-1]] + f[edges[1:] - 1]) / 2
    else:
        freq = sfft.fftshift(sfft.fftfreq(nfft, 1 / fs))
    offset = float(spec.power_offset_db)
    spectrogram = 10 * np.log10(P + 1e-30) + offset
    avg = 10 * np.log10(np.mean(P, axis=0) + 1e-30) + offset
    return {
        "name": info.path.name, "sample_rate": fs, "recording_samples": int(total), "duration_s": total / fs,
        "center_frequency_hz": info.center_frequency_hz, "unit": spec.power_unit,
        "time_s": (starts / fs).tolist(), "start_samples": starts.tolist(),
        "power_db": (10 * np.log10(power + 1e-30) + offset).tolist(),
        "peak_magnitude": peaks.astype(float).tolist(),
        "max_magnitude": float(np.max(peaks)),
        "freq_hz": np.asarray(freq, dtype=np.float64).tolist(), "avg_psd_db": avg.astype(np.float32).tolist(),
        "spectrogram_db": np.round(spectrogram, 2).astype(np.float32).ravel().tolist(),
        "spectrogram_shape": [int(rows), int(len(freq))], "nfft": nfft,
        "rbw_hz": 1.5 * fs / nfft, "coverage": float(rows * nfft / total),
        "method": f"{rows} evenly spaced {nfft}-point Hann FFTs; display-only sampling of the recording.",
    }


# ---------------------------------------------------------------------------
# Uploads


def unique_name(name: str) -> str:
    base = safe_name(name)
    path = library_dir() / base
    if not path.exists():
        return base
    lower = base.lower()
    suffix = next((s for s in (".sigmf-meta", ".sigmf-data") if lower.endswith(s)), Path(base).suffix)
    stem = base[: len(base) - len(suffix)]
    for i in range(2, 10_000):
        candidate = f"{stem} ({i}){suffix}"
        if not (library_dir() / candidate).exists():
            return candidate
    raise ValueError("Too many recordings with this name")


async def save_upload(name: str, chunks: AsyncIterator[bytes], overwrite: bool = False) -> dict:
    """Stream an upload to the library without holding it in memory."""
    target_name = safe_name(name) if overwrite else unique_name(name)
    target = resolve(target_name, must_exist=False)
    partial = target.with_name(target.name + ".partial")
    written = 0
    try:
        with partial.open("wb") as f:
            async for chunk in chunks:
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    raise ValueError(f"Upload exceeds the {MAX_UPLOAD_BYTES >> 30} GiB limit (VECTORRF_MAX_UPLOAD_GB)")
                f.write(chunk)
        if overwrite:
            close_readers(target_name)
        os.replace(partial, target)
    finally:
        if partial.exists():
            partial.unlink()
    return {"name": target_name, "bytes": written}


def delete_recording(name: str) -> list[str]:
    path = resolve(name)
    removed = []
    close_readers(path.name)
    lower = path.name.lower()
    targets = [path]
    for suffix in (".sigmf-meta", ".sigmf-data"):
        if lower.endswith(suffix):
            stem = path.name[: -len(suffix)]
            targets = [path.with_name(stem + ".sigmf-meta"), path.with_name(stem + ".sigmf-data")]
    for t in targets:
        if t.exists():
            t.unlink()
            removed.append(t.name)
    return removed


def write_library_file(name: str, payload: bytes) -> str:
    target_name = unique_name(name)
    resolve(target_name, must_exist=False).write_bytes(payload)
    return target_name


__all__ = [
    "FileSource", "SampleFormat", "library_dir", "list_library", "resolve", "file_capture", "describe",
    "save_upload", "delete_recording", "write_library_file", "unique_name", "close_readers", "open_reader",
]
