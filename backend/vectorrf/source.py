"""Deterministic capture identity shared by observers; live captures advance.

A capture is addressed by its source and an integer capture ID.  Transmitter
capture ``k`` starts at absolute sample ``k * N`` (N = samples per capture), so
consecutive captures are contiguous in time and every observer (spectrum,
zero span, receiver, classifier) that names the same capture sees identical
samples.  Recordings are addressed the same way through ``FileSource``.
"""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import os
from threading import RLock

from .models import TxConfig
from .signal import GeneratedSignal, capture_samples, generate

_lock = RLock()
_cache: "OrderedDict[tuple[str, int], GeneratedSignal]" = OrderedDict()
_cache_bytes = 0
CACHE_BYTES = max(64, int(os.environ.get("VECTORRF_CACHE_MB", "512"))) << 20
CACHE_ENTRIES = 12


def _nbytes(g: GeneratedSignal) -> int:
    total = g.samples.nbytes
    for extra in (g.tx_bits, g.tx_symbols):
        if extra is not None:
            total += extra.nbytes
    return total


def _record(config_json: str, frame: int, limit: int | None = None) -> GeneratedSignal:
    global _cache_bytes
    key = (config_json, frame, limit)
    with _lock:
        hit = _cache.get(key)
        if hit is not None:
            _cache.move_to_end(key)
            return hit
    cfg = TxConfig.model_validate_json(config_json)
    # The transmitter is one stream indexed by absolute sample (noise included),
    # so capture ``frame`` is simply its slice starting at frame·N.
    start = frame * capture_samples(cfg)
    if limit is not None and limit < capture_samples(cfg):
        # Only the first ``limit`` samples of the capture: the same samples, the rest never made.
        g = generate(cfg.model_copy(update={"duration": limit / cfg.sample_rate}), frame=frame, start_sample=start)
    else:
        g = generate(cfg, frame=frame, start_sample=start)
    g.samples.flags.writeable = False
    g.metadata.update(source_revision=sha256(config_json.encode()).hexdigest()[:12],
                      capture_id=frame, source_seed=cfg.seed, tx_symbol_rate=cfg.symbol_rate,
                      simulation_time_s=start / cfg.sample_rate, source_kind="transmitter",
                      playback="advancing deterministic simulation captures")
    with _lock:
        if key not in _cache:
            _cache[key] = g
            _cache_bytes += _nbytes(g)
            while _cache and (len(_cache) > CACHE_ENTRIES or _cache_bytes > CACHE_BYTES) and len(_cache) > 1:
                _, old = _cache.popitem(last=False)
                _cache_bytes -= _nbytes(old)
        return _cache[key]


def source_record(cfg: TxConfig, frame: int = 0, limit: int | None = None) -> GeneratedSignal:
    """Capture ``frame`` of the transmitter (only its first ``limit`` samples when given)."""
    if not 0 <= frame <= 2147483647:
        raise ValueError("Capture ID outside supported range")
    return _record(cfg.model_dump_json(), int(frame), None if limit is None else max(32, int(limit)))


def acquire(tx: TxConfig | None, file=None, capture_id: int = 0, limit: int | None = None) -> GeneratedSignal:
    """One capture from a recording (when ``file`` is given) or the transmitter;
    with ``limit``, only its first ``limit`` samples (a real-time receiver that
    drops what it cannot process)."""
    if file is not None:
        from .recordings import file_capture
        return file_capture(file, capture_id, limit)
    if tx is None:
        raise ValueError("A transmitter configuration or a recording source is required")
    return source_record(tx, capture_id, limit)


def cache_clear() -> None:
    global _cache_bytes
    with _lock:
        _cache.clear()
        _cache_bytes = 0
