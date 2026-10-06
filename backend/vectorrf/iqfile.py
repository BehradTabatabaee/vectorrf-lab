"""I/Q recording formats: raw interleaved binary (.iq/.bin/...), WAV/RF64 and SigMF.

Readers memory-map the sample region and decode only the requested window, so
multi-gigabyte wideband recordings (for example 2 GS/s captures) can be
scrolled and analysed without loading the file into RAM.  All integer formats
are scaled to a common full scale: a complex sample of magnitude 1.0 is
full-scale, i.e. 0 dBFS for a full-scale complex tone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
import struct
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Sample encodings

_SAMPLE_RE = re.compile(r"^(?P<cr>[rc])(?P<kind>[fiu])(?P<bits>8|16|24|32|64)(?:_(?P<endian>le|be))?$")


@dataclass(frozen=True)
class SampleFormat:
    """Interleaved sample encoding using SigMF datatype names (e.g. ``cf32_le``)."""

    complex: bool
    kind: str          # 'f' float, 'i' signed int, 'u' unsigned int
    bits: int
    endian: str = "le"  # irrelevant for 8-bit types
    unsigned_zero: float | None = None  # zero code for unsigned data; default 2^(b-1) - 0.5

    @classmethod
    def parse(cls, text: str) -> "SampleFormat":
        m = _SAMPLE_RE.match(str(text).strip().lower())
        if not m:
            raise ValueError(f"Unknown sample format {text!r}; use SigMF names such as cf32_le, ci16_le, ci8 or cu8")
        kind, bits = m["kind"], int(m["bits"])
        if kind == "f" and bits not in (32, 64):
            raise ValueError("Floating-point samples must be 32 or 64 bit")
        if kind in "iu" and bits == 64:
            raise ValueError("64-bit integer samples are not supported")
        if bits == 24 and (kind != "i" or (m["endian"] or "le") != "le"):
            raise ValueError("24-bit samples are supported as packed little-endian signed integers (ci24_le / ri24_le)")
        return cls(m["cr"] == "c", kind, bits, m["endian"] or "le")

    @property
    def name(self) -> str:
        base = f"{'c' if self.complex else 'r'}{self.kind}{self.bits}"
        return base if self.bits == 8 else f"{base}_{self.endian}"

    @property
    def zero_code(self) -> float:
        return float(self.unsigned_zero) if self.unsigned_zero is not None else (1 << (self.bits - 1)) - 0.5

    @property
    def dtype(self) -> np.dtype:
        if self.bits == 24:
            raise ValueError("Packed 24-bit samples have no NumPy dtype; use RecordingReader")
        order = "<" if self.endian == "le" else ">"
        return np.dtype(f"{order if self.bits > 8 else '|'}{self.kind}{self.bits // 8}")

    @property
    def components(self) -> int:
        return 2 if self.complex else 1

    @property
    def bytes_per_sample(self) -> int:
        return self.components * self.bits // 8

    def decode(self, raw: np.ndarray) -> np.ndarray:
        """Decode interleaved components to complex64 at the common full scale."""
        return self.decode_counted(raw)[0]

    def decode_counted(self, raw: np.ndarray) -> tuple[np.ndarray, int]:
        """Decode and report how many invalid float components were zeroed."""
        values = np.asarray(raw)
        replaced = 0
        if self.kind == "f":
            # Float data read with the wrong format (or corrupt files) contains
            # NaN/Inf and absurd magnitudes (up to 3e38) whose FFTs overflow;
            # no DSP stage or JSON product can carry them, so zero them.
            # RecordingReader counts how many were replaced.
            with np.errstate(invalid="ignore", over="ignore"):
                v = values.astype(np.float32)
                bad = ~(np.abs(v) <= INVALID_FLOAT)  # also true for NaN
            replaced = int(np.count_nonzero(bad))
            if replaced:
                v[bad] = 0.0
        elif self.kind == "i":
            v = values.astype(np.float32) * np.float32(1.0 / (1 << (self.bits - 1)))
        else:
            # Offset binary.  By default the mid-scale code 2^(b-1) - 0.5 is
            # zero, as for RTL-SDR 8-bit captures (127.5); 8-bit WAV PCM uses 128.
            v = (values.astype(np.float32) - np.float32(self.zero_code)) * np.float32(1.0 / (1 << (self.bits - 1)))
        if self.complex:
            v = v[: (v.size // 2) * 2]
            out = np.empty(v.size // 2, dtype=np.complex64)
            out.real = v[0::2]
            out.imag = v[1::2]
            return out, replaced
        return v.astype(np.complex64), replaced

    def encode(self, samples: np.ndarray, scale: float = 1.0) -> tuple[np.ndarray, int]:
        """Encode complex samples (multiplied by ``scale``); returns raw values and clip count."""
        x = np.asarray(samples, dtype=np.complex64) * np.float32(scale)
        if self.complex:
            comp = np.empty(x.size * 2, dtype=np.float64)
            comp[0::2] = x.real
            comp[1::2] = x.imag
        else:
            comp = x.real.astype(np.float64)
        if self.kind == "f":
            return comp.astype(self.dtype), 0
        full = float(1 << (self.bits - 1))
        if self.kind == "i":
            q = np.rint(comp * full)
            lo, hi = -full, full - 1
        else:
            q = np.rint(comp * full + self.zero_code)
            lo, hi = 0.0, 2 * full - 1
        clipped = int(np.count_nonzero((q < lo) | (q > hi)))
        return np.clip(q, lo, hi).astype(self.dtype), clipped


# Float sample magnitudes beyond this are treated as invalid (wrong format).
INVALID_FLOAT = 1e18

RAW_FORMATS = ("cf32_le", "cf64_le", "ci16_le", "ci32_le", "ci8", "cu8", "cf32_be", "ci16_be", "rf32_le", "ri16_le")

# Extension conventions used by GNU Radio, gqrx, HackRF, RTL-SDR, UHD and others.
EXTENSION_FORMATS = {
    ".cf32": "cf32_le", ".fc32": "cf32_le", ".cfile": "cf32_le", ".cf64": "cf64_le",
    ".cs16": "ci16_le", ".sc16": "ci16_le", ".ci16": "ci16_le", ".cs8": "ci8", ".sc8": "ci8",
    ".ci8": "ci8", ".cu8": "cu8", ".raw": "cf32_le",
}
AMBIGUOUS_EXTENSIONS = {".iq", ".bin", ".dat"}
RAW_EXTENSIONS = set(EXTENSION_FORMATS) | AMBIGUOUS_EXTENSIONS
LIBRARY_EXTENSIONS = RAW_EXTENSIONS | {".wav", ".sigmf-data", ".sigmf-meta", ".sigmf"}


# ---------------------------------------------------------------------------
# Metadata

@dataclass
class RecordingInfo:
    path: Path
    container: str               # 'raw' | 'wav' | 'sigmf'
    sample_format: SampleFormat
    data_offset: int
    num_samples: int
    sample_rate: float | None
    center_frequency_hz: float | None = None
    channels: int = 2
    datetime: str | None = None
    description: str | None = None
    annotations: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float | None:
        return self.num_samples / self.sample_rate if self.sample_rate else None

    def as_dict(self) -> dict:
        return {
            "name": self.path.name, "container": self.container, "format": self.sample_format.name,
            "data_offset": int(self.data_offset), "num_samples": int(self.num_samples),
            "sample_rate": self.sample_rate, "center_frequency_hz": self.center_frequency_hz,
            "duration_s": self.duration_s, "channels": self.channels, "real_valued": not self.sample_format.complex,
            "bytes_per_sample": self.sample_format.bytes_per_sample, "datetime": self.datetime,
            "description": self.description, "annotations": self.annotations[:256], "notes": self.notes,
            "extra": self.extra,
        }


def guess_from_filename(name: str) -> dict:
    """Center frequency / sample rate hints embedded by common SDR applications.

    * gqrx: ``gqrx_YYYYMMDD_HHMMSS_<center>_<rate>_fc.raw``
    * SDR#, SDRuno, HDSDR: ``..._<center>Hz_IQ.wav`` (rate is in the WAV header)
    * generic: ``..._<n>[k|M|G]sps`` / ``..._<n>[k|M|G]Hz``
    """
    hints: dict = {}
    stem = Path(name).name
    m = re.search(r"_(\d{4,12})_(\d{3,11})_fc\.[a-z0-9]+$", stem, re.I)
    if m:
        hints.update(center_frequency_hz=float(m[1]), sample_rate=float(m[2]), source="gqrx file name")
        return hints
    m = re.search(r"_(\d{3,12})Hz_IQ\.", stem, re.I)
    if m:
        hints.update(center_frequency_hz=float(m[1]), source="SDR# / SDRuno file name")
    units = {"": 1.0, "k": 1e3, "m": 1e6, "g": 1e9}
    m = re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*([kmg]?)s(?:ps|/s)(?![a-z])", stem, re.I)
    if m and "sample_rate" not in hints:
        hints.update(sample_rate=float(m[1]) * units[m[2].lower()], source=hints.get("source", "file name"))
    return hints


def _sigmf_paths(path: Path) -> tuple[Path, Path]:
    name = path.name
    for suffix in (".sigmf-meta", ".sigmf-data", ".sigmf"):
        if name.endswith(suffix):
            base = path.with_name(name[: -len(suffix)])
            return base.with_name(base.name + ".sigmf-meta"), base.with_name(base.name + ".sigmf-data")
    return path.with_suffix(".sigmf-meta"), path.with_suffix(".sigmf-data")


def read_sigmf(path: Path) -> RecordingInfo:
    path = Path(path)
    if path.name.lower().endswith(".sigmf") and path.is_file():
        return _read_sigmf_archive(path)
    meta_path, data_path = _sigmf_paths(path)
    if not meta_path.exists():
        raise ValueError(f"SigMF metadata {meta_path.name} not found")
    if not data_path.exists():
        raise ValueError(f"SigMF dataset {data_path.name} not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return _sigmf_info(meta, data_path, 0, data_path.stat().st_size)


def _read_sigmf_archive(path: Path) -> RecordingInfo:
    """SigMF Archive (``.sigmf``): an uncompressed tar holding a meta/data pair.

    Tar stores member data contiguously, so the dataset is memory-mapped in
    place at the member's offset rather than extracted.
    """
    import tarfile
    try:
        with tarfile.open(path, mode="r:") as tar:
            members = [m for m in tar.getmembers() if m.isfile()]
            meta_m = next((m for m in members if m.name.lower().endswith(".sigmf-meta")), None)
            if meta_m is None:
                raise ValueError("SigMF archive has no .sigmf-meta member")
            stem = meta_m.name[: -len(".sigmf-meta")]
            data_m = next((m for m in members if m.name == stem + ".sigmf-data"), None)
            if data_m is None:
                raise ValueError("SigMF archive has no matching .sigmf-data member")
            meta = json.loads(tar.extractfile(meta_m).read().decode("utf-8"))
            return _sigmf_info(meta, path, int(data_m.offset_data), int(data_m.size))
    except tarfile.TarError as exc:
        raise ValueError(f"Not a valid SigMF archive: {exc}") from exc


def _sigmf_info(meta: dict, data_path: Path, base_offset: int, data_bytes: int) -> RecordingInfo:
    g = meta.get("global", {})
    fmt = SampleFormat.parse(g.get("core:datatype", "cf32_le"))
    captures = meta.get("captures") or [{}]
    first = captures[0]
    header = int(first.get("core:header_bytes", 0))
    offset = base_offset + header
    size = data_bytes - header
    n = max(0, size // fmt.bytes_per_sample)
    notes = []
    if len(captures) > 1:
        notes.append(f"{len(captures)} SigMF capture segments; the first segment's frequency is used for display.")
    if g.get("core:num_channels", 1) not in (1, None):
        notes.append("Multi-channel SigMF datasets are read as their first channel interleave only.")
    extra = {k: v for k, v in g.items() if k.startswith("vectorrf:")}
    return RecordingInfo(
        path=data_path, container="sigmf", sample_format=fmt, data_offset=offset, num_samples=n,
        sample_rate=float(g["core:sample_rate"]) if g.get("core:sample_rate") else None,
        center_frequency_hz=float(first["core:frequency"]) if first.get("core:frequency") is not None else None,
        datetime=first.get("core:datetime"), description=g.get("core:description"),
        annotations=[a for a in meta.get("annotations", []) if isinstance(a, dict)], extra=extra, notes=notes,
    )


# ---------------------------------------------------------------------------
# WAV / RF64

_WAVE_FORMAT_PCM = 1
_WAVE_FORMAT_FLOAT = 3
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


def _iter_chunks(f: BinaryIO, start: int, end: int, ds64_data: int | None) -> Iterable[tuple[bytes, int, int]]:
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            return
        cid, size = header[:4], struct.unpack("<I", header[4:])[0]
        if cid == b"data" and size == 0xFFFFFFFF and ds64_data is not None:
            size = ds64_data
        yield cid, pos + 8, size
        pos += 8 + size + (size & 1)


def read_wav(path: Path) -> RecordingInfo:
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[8:12] != b"WAVE" or head[:4] not in (b"RIFF", b"RF64", b"BW64"):
            raise ValueError("Not a RIFF/RF64 WAVE file")
        rf64 = head[:4] in (b"RF64", b"BW64")
        ds64_data = None
        if rf64:
            f.seek(12)
            cid, size = f.read(4), struct.unpack("<I", f.read(4))[0]
            if cid != b"ds64":
                raise ValueError("RF64 file is missing its ds64 chunk")
            riff_size, data_size = struct.unpack("<QQ", f.read(16))
            ds64_data = data_size
        fmt_info = None
        data = None
        center = None
        recorded = None
        for cid, at, size in _iter_chunks(f, 12, file_size, ds64_data):
            if cid == b"fmt ":
                f.seek(at)
                raw = f.read(min(size, 40))
                tag, channels, rate, _, align, bits = struct.unpack("<HHIIHH", raw[:16])
                if tag == _WAVE_FORMAT_EXTENSIBLE and len(raw) >= 40:
                    tag = struct.unpack("<H", raw[24:26])[0]
                fmt_info = (tag, channels, rate, align, bits)
            elif cid == b"data":
                data = (at, min(size, file_size - at))
            elif cid == b"auxi" and size >= 36:
                # SpectraVue / SDR# / HDSDR auxiliary chunk: two SYSTEMTIMEs then
                # CenterFreq (uint32 Hz).
                f.seek(at)
                raw = f.read(36)
                yr, mo, _, dy, hh, mi, ss, _ = struct.unpack("<8H", raw[:16])
                center = float(struct.unpack("<I", raw[32:36])[0]) or None
                if 1900 < yr < 3000:
                    recorded = f"{yr:04d}-{mo:02d}-{dy:02d}T{hh:02d}:{mi:02d}:{ss:02d}"
    if fmt_info is None or data is None:
        raise ValueError("WAV file needs both fmt and data chunks")
    tag, channels, rate, align, bits = fmt_info
    if channels not in (1, 2):
        raise ValueError(f"WAV files with {channels} channels are not I/Q recordings; use 1 (real) or 2 (I/Q) channels")
    if tag == _WAVE_FORMAT_FLOAT and bits in (32, 64):
        kind = "f"
    elif tag == _WAVE_FORMAT_PCM and bits == 8:
        kind = "u"  # WAV 8-bit PCM is unsigned with 128 as zero
    elif tag == _WAVE_FORMAT_PCM and bits in (16, 24, 32):
        kind = "i"
    else:
        raise ValueError(f"Unsupported WAV encoding (format tag {tag}, {bits} bits)")
    notes = []
    if channels == 1:
        notes.append("Mono WAV is read as a real-valued signal; its spectrum is symmetric about DC.")
    sf = SampleFormat(channels == 2, kind, bits, unsigned_zero=128.0 if kind == "u" else None)
    at, size = data
    bps = channels * bits // 8
    return RecordingInfo(path=path, container="wav", sample_format=sf, data_offset=at, num_samples=size // bps,
                         sample_rate=float(rate), center_frequency_hz=center, channels=channels,
                         datetime=recorded, notes=notes, extra={"rf64": rf64})


# ---------------------------------------------------------------------------
# Raw inspection and sniffing

def sniff_raw(path: Path, header_bytes: int = 0, probe_bytes: int = 1 << 20) -> list[dict]:
    """Rank plausible sample encodings of a headerless file.

    Raw ``.iq``/``.bin`` files carry no format information: RTL-SDR tools write
    cu8, HackRF ci8, UHD ci16 and GNU Radio cf32 under the same extensions.
    This returns heuristic scores for the user to confirm; it never silently
    decides.
    """
    path = Path(path)
    size = path.stat().st_size - header_bytes
    with path.open("rb") as f:
        f.seek(header_bytes)
        raw = f.read(min(max(0, size), probe_bytes))
    out = []
    for name in ("cf32_le", "ci16_le", "ci8", "cu8", "cf64_le"):
        fmt = SampleFormat.parse(name)
        reasons = []
        if size % fmt.bytes_per_sample:
            out.append({"format": name, "score": 0.0, "reasons": ["file size is not a whole number of samples"]})
            continue
        usable = (len(raw) // fmt.bytes_per_sample) * fmt.bytes_per_sample
        if usable < 64:
            out.append({"format": name, "score": 0.1, "reasons": ["too little data to judge"]})
            continue
        v = np.frombuffer(raw[:usable], dtype=fmt.dtype)
        if fmt.kind == "f":
            # Integer data reinterpreted as float contains signalling NaNs;
            # that is exactly what this test detects, so silence the cast warning.
            with np.errstate(invalid="ignore", over="ignore"):
                fv = v.astype(np.float64)
            finite = np.isfinite(fv)
            mag = np.abs(fv[finite])
            plausible = np.mean((mag == 0) | ((mag > 1e-7) & (mag < 1e4))) if mag.size else 0.0
            score = float(np.mean(finite) * plausible) ** 4
            reasons.append(f"{100*np.mean(finite):.1f}% finite, {100*plausible:.1f}% in a plausible float range")
        else:
            iv = v.astype(np.float64)
            if fmt.kind == "u":
                iv = iv - ((1 << (fmt.bits - 1)) - 0.5)
            full = float(1 << (fmt.bits - 1))
            mean = float(np.mean(iv)) / full
            std = float(np.std(iv)) / full
            # Smooth signals have strongly correlated successive I samples;
            # the wrong integer width destroys that correlation.
            i = iv[0::2]
            corr = float(np.corrcoef(i[:-1], i[1:])[0, 1]) if i.size > 3 and np.std(i) > 0 else 0.0
            dc_ok = math.exp(-abs(mean) / 0.05)
            spread_ok = 1.0 if 1e-4 < std < 0.9 else 0.2
            score = dc_ok * spread_ok * (0.5 + 0.5 * max(0.0, corr))
            reasons.append(f"mean {mean:+.3f} FS, std {std:.3f} FS, lag-1 correlation {corr:.2f}")
        out.append({"format": name, "score": round(float(score), 4), "reasons": reasons})
    out.sort(key=lambda r: r["score"], reverse=True)
    return out


def inspect(path: Path, format: str | None = None, sample_rate: float | None = None,
            center_frequency_hz: float | None = None, header_bytes: int = 0) -> RecordingInfo:
    """Describe a recording, applying any explicit user overrides."""
    path = Path(path)
    lower = path.name.lower()
    if lower.endswith((".sigmf-meta", ".sigmf-data", ".sigmf")):
        info = read_sigmf(path)
    elif lower.endswith(".wav"):
        info = read_wav(path)
    else:
        ext = path.suffix.lower()
        chosen = format or EXTENSION_FORMATS.get(ext)
        notes = []
        if chosen is None:
            ranked = sniff_raw(path, header_bytes)
            chosen = ranked[0]["format"] if ranked and ranked[0]["score"] > 0 else "cf32_le"
            notes.append(f"{ext or 'This'} files carry no sample format; {chosen} was suggested by content heuristics. Confirm it.")
        fmt = SampleFormat.parse(chosen)
        size = path.stat().st_size - int(header_bytes)
        if size < 0:
            raise ValueError("Header offset is larger than the file")
        hints = guess_from_filename(path.name)
        info = RecordingInfo(path=path, container="raw", sample_format=fmt, data_offset=int(header_bytes),
                             num_samples=size // fmt.bytes_per_sample, sample_rate=hints.get("sample_rate"),
                             center_frequency_hz=hints.get("center_frequency_hz"), notes=notes,
                             extra={"filename_hints": hints} if hints else {})
        if size % fmt.bytes_per_sample:
            info.notes.append(f"{size % fmt.bytes_per_sample} trailing bytes do not form a whole {fmt.name} sample and are ignored.")
        format = None  # already applied
    if format and info.container != "raw":
        # Explicit override of a container's own description is allowed but noted.
        fmt = SampleFormat.parse(format)
        info.notes.append(f"Sample format overridden to {fmt.name}.")
        size = info.num_samples * info.sample_format.bytes_per_sample
        info.sample_format = fmt
        info.num_samples = size // fmt.bytes_per_sample
    if sample_rate:
        info.sample_rate = float(sample_rate)
    if center_frequency_hz is not None:
        info.center_frequency_hz = float(center_frequency_hz)
    return info


# ---------------------------------------------------------------------------
# Reading

class RecordingReader:
    """Memory-mapped random access to the samples of one recording."""

    def __init__(self, info: RecordingInfo):
        self.info = info
        self.nonfinite_replaced = 0
        fmt = info.sample_format
        self._packed24 = fmt.bits == 24
        if info.num_samples <= 0:
            self._map = np.zeros(0, dtype=np.uint8)
        elif self._packed24:
            self._map = np.memmap(info.path, dtype=np.uint8, mode="r", offset=info.data_offset,
                                  shape=(info.num_samples * fmt.bytes_per_sample,))
        else:
            self._map = np.memmap(info.path, dtype=fmt.dtype, mode="r", offset=info.data_offset,
                                  shape=(info.num_samples * fmt.components,))

    def close(self) -> None:
        m = getattr(self, "_map", None)
        if isinstance(m, np.memmap):
            mm = getattr(m, "_mmap", None)
            if mm is not None:
                mm.close()
        self._map = np.zeros(0, dtype=np.uint8)

    @property
    def num_samples(self) -> int:
        return int(self.info.num_samples)

    def read(self, start: int, count: int) -> np.ndarray:
        """Return ``count`` complex64 samples from ``start`` (fewer at end of file)."""
        start = max(0, int(start))
        stop = min(self.num_samples, start + max(0, int(count)))
        if stop <= start:
            return np.zeros(0, dtype=np.complex64)
        fmt = self.info.sample_format
        if self._packed24:
            b = np.asarray(self._map[start * fmt.bytes_per_sample: stop * fmt.bytes_per_sample]).reshape(-1, 3)
            v = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
            v = np.where(v >= 1 << 23, v - (1 << 24), v).astype(np.float32) / np.float32(1 << 23)
            if fmt.complex:
                out = np.empty(v.size // 2, np.complex64)
                out.real, out.imag = v[0::2], v[1::2]
                return out
            return v.astype(np.complex64)
        c = fmt.components
        out, replaced = fmt.decode_counted(np.asarray(self._map[start * c: stop * c]))
        self.nonfinite_replaced += replaced
        return out

    def read_wrapped(self, start: int, count: int) -> np.ndarray:
        """Read ``count`` samples treating the file as a loop."""
        n = self.num_samples
        if n == 0:
            return np.zeros(0, dtype=np.complex64)
        start %= n
        parts = []
        remaining = int(count)
        while remaining > 0:
            part = self.read(start, min(remaining, n - start))
            parts.append(part)
            remaining -= len(part)
            start = 0
        return np.concatenate(parts) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# Writing

def _wav_header(fmt: SampleFormat, rate: float, num_samples: int, center_hz: float | None) -> bytes:
    if fmt.kind == "f":
        tag = _WAVE_FORMAT_FLOAT
    elif fmt.kind == "i" and fmt.bits in (16, 32):
        tag = _WAVE_FORMAT_PCM
    elif fmt.kind == "u" and fmt.bits == 8:
        tag = _WAVE_FORMAT_PCM
    else:
        raise ValueError("WAV export supports cu8, ci16, ci32, cf32 and cf64 samples")
    if fmt.endian != "le":
        raise ValueError("WAV samples are little-endian")
    if not 0 < rate <= 0xFFFFFFFF or abs(rate - round(rate)) > 1e-6:
        raise ValueError("WAV stores an integer sample rate up to 4,294,967,295 S/s; use SigMF for other rates")
    channels = fmt.components
    align = channels * fmt.bits // 8
    data_size = num_samples * align
    # nAvgBytesPerSec is 32-bit; at GS/s rates it saturates (readers derive the
    # rate from nSamplesPerSec, which is what this format actually needs).
    byte_rate = min(int(round(rate)) * align, 0xFFFFFFFF)
    fmt_chunk = struct.pack("<4sIHHIIHH", b"fmt ", 16, tag, channels, int(round(rate)), byte_rate, align, fmt.bits)
    auxi = b""
    if center_hz is not None and 0 <= center_hz <= 0xFFFFFFFF:
        body = bytes(32) + struct.pack("<IIIII", int(round(center_hz)), int(round(rate)), 0, int(round(rate)), 0) + bytes(16)
        auxi = struct.pack("<4sI", b"auxi", len(body)) + body
    if 4 + len(fmt_chunk) + len(auxi) + 8 + data_size <= 0xFFFFFFFF:
        riff = struct.pack("<4sI4s", b"RIFF", 4 + len(fmt_chunk) + len(auxi) + 8 + data_size, b"WAVE")
        return riff + fmt_chunk + auxi + struct.pack("<4sI", b"data", data_size)
    # RF64 (EBU Tech 3306) for data beyond the 4 GiB RIFF limit.
    ds64 = struct.pack("<4sIQQQI", b"ds64", 28, 4 + 36 + len(fmt_chunk) + len(auxi) + 8 + data_size, data_size, num_samples, 0)
    return struct.pack("<4sI4s", b"RF64", 0xFFFFFFFF, b"WAVE") + ds64 + fmt_chunk + auxi + struct.pack("<4sI", b"data", 0xFFFFFFFF)


def sigmf_meta(fmt: SampleFormat, rate: float, center_hz: float | None, description: str,
               extra: dict | None = None, datetime: str | None = None) -> dict:
    from . import __version__
    capture = {"core:sample_start": 0}
    if center_hz is not None:
        capture["core:frequency"] = float(center_hz)
    if datetime:
        capture["core:datetime"] = datetime
    g = {"core:datatype": fmt.name, "core:sample_rate": float(rate), "core:version": "1.2.0",
         "core:description": description, "core:recorder": f"VectorRF Lab {__version__}"}
    g.update(extra or {})
    return {"global": g, "captures": [capture], "annotations": []}


def encode_recording(samples: np.ndarray, fmt: SampleFormat, rate: float, container: str,
                     scale: float = 1.0, center_hz: float | None = None) -> tuple[bytes, int]:
    """Serialize a capture to bytes for download (header + interleaved samples)."""
    if container == "wav" and fmt.kind == "u":
        # 8-bit WAV PCM defines 128 as zero (raw cu8 uses the RTL-SDR 127.5 midpoint).
        fmt = SampleFormat(fmt.complex, "u", fmt.bits, fmt.endian, unsigned_zero=128.0)
    raw, clipped = fmt.encode(samples, scale)
    body = raw.tobytes()
    if container == "wav":
        return _wav_header(fmt, rate, len(samples), center_hz) + body, clipped
    if container == "raw":
        return body, clipped
    raise ValueError("Unknown container")


def sigmf_archive(stem: str, meta: dict, data: bytes) -> bytes:
    """SigMF Archive (uncompressed tar): ``stem/stem.sigmf-meta`` + ``stem/stem.sigmf-data``."""
    import io
    import tarfile
    import time
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for suffix, payload in ((".sigmf-meta", json.dumps(meta, indent=2).encode("utf-8")), (".sigmf-data", data)):
            member = tarfile.TarInfo(f"{stem}/{stem}{suffix}")
            member.size = len(payload)
            member.mtime = int(time.time())
            tar.addfile(member, io.BytesIO(payload))
    return buf.getvalue()


def peak_scale(samples: np.ndarray, headroom_db: float = 1.0) -> float:
    """Scale placing the largest I or Q component ``headroom_db`` below full scale."""
    x = np.asarray(samples)
    peak = float(max(np.max(np.abs(x.real), initial=0.0), np.max(np.abs(x.imag), initial=0.0)))
    return (10 ** (-headroom_db / 20)) / peak if peak > 0 else 1.0
