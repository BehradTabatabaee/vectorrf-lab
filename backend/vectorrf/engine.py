from __future__ import annotations
import asyncio
import math
import json
import logging
import time
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict, Field
from .models import TxConfig, SpectrumConfig, ChannelConfig, RxConfig, RECOMMENDED_LIVE_SAMPLE_RATE, MAX_SAMPLE_RATE, MAX_CAPTURE_SAMPLES
from .source import acquire
from .pipeline import Pipeline,pipeline_product,channel_samples,dehop,process_chain,preview
from .spectrum import analyze, required_samples
from .stream import FrameType, pack_f32
from .rx import channel_symbols, demodulate, receive_frame, reference_channel_estimate
from .dsp import display_complex, display_real, display_tree, display_values
from .signal import capture_samples
from .errors import describe

log = logging.getLogger(__name__)

# Significant figures of the plot traces in live receiver frames (see dsp.display_values).
LIVE_DIGITS = 6


def reduce_spectrum(freq, db, maximum=2048):
    """Fixed frequency buckets preserve trace-hold alignment across frames.

    Bucket centres stay float64: at GHz offsets float32 cannot resolve the
    bin spacing of a large FFT (its spacing near 2.5 GHz is 256 Hz).
    """
    if len(db) <= maximum:
        return np.asarray(freq, dtype=np.float64), db
    edges = np.linspace(0, len(db), maximum+1, dtype=int)
    freq = np.asarray(freq, dtype=np.float64)
    centers=(freq[edges[:-1]]+freq[edges[1:]-1])/2
    return centers, np.maximum.reduceat(db,edges[:-1])


def calibrated_analyze(samples, g, spec):
    """Spectrum of samples from capture ``g`` using that capture's power calibration."""
    return analyze(
        samples, g.sample_rate, spec,
        reference_power_dbm=g.metadata.get('power_dbm'),
        reference_rms=g.metadata.get('power_reference_rms'),
        unit=g.metadata.get('power_unit'),
    )


def frame_product(g, spec, cursor):
    """Analyse the next spectrum chunk of capture ``g``.

    Returns ``(result, freq, psd, next_cursor, exhausted)``.  ``exhausted`` is
    true once the chunk reached the end of the capture, so the caller can move
    on to the next (contiguous) capture instead of re-showing the same samples.
    """
    count = min(len(g.samples), required_samples(g.sample_rate, spec))
    # Never wrap a DSP window across the record boundary.  The former circular
    # splice created a discontinuity that appeared as occasional broadband
    # "pulses" on PSK/QAM spectra.
    if count >= len(g.samples):
        chunk = g.samples
        next_cursor = 0
        exhausted = True
    else:
        max_start = len(g.samples) - count
        cursor = int(np.clip(cursor, 0, max_start))
        chunk = g.samples[cursor:cursor+count]
        exhausted = cursor >= max_start
        next_cursor = 0 if exhausted else min(max_start, cursor + max(1, count//3))
    result = calibrated_analyze(chunk, g, spec)
    f, p = reduce_spectrum(result.freq_hz, result.psd_db)
    return result, f, p, next_cursor, exhausted


def receiver_product(g, channel, pipeline, rx, frame, memory=None):
    """One real source -> channel -> user pipeline -> demod product for live UI."""
    y=dehop(channel_samples(g,channel,frame),rx.dehop,g.sample_rate,g.metadata.get('start_sample'))
    start=float(g.metadata.get('simulation_time_s',frame*g.metadata.get('duration',0)))
    z,fs,stages,probe=process_chain(y,g.sample_rate,pipeline,start,frame,memory,start_sample=g.metadata.get('start_sample'))
    cfg=rx.model_copy(update={'sample_rate':fs,'ofdm_start_s':start})
    # The receiver keeps tracking from capture to capture (also across captures
    # real time dropped) while the processing chain keeps the sample clock.
    a=g.metadata.get('start_sample')
    tracked=memory is not None and a is not None and abs(fs-g.sample_rate)<1e-9*g.sample_rate
    r=demodulate(z,cfg,int(a) if tracked else None,memory.setdefault('rx_track',{}) if tracked else None)
    frame_info = receive_frame(
        z, cfg, power_dbm=float(g.metadata.get('power_dbm', 0.0)),
        reference_rms=float(g.metadata.get('power_reference_rms', 1.0) or 1.0),
    ) if cfg.frame_enabled else None
    telemetry={k:display_tree(v,digits=LIVE_DIGITS) for k,v in dict(r.telemetry).items()}
    pre_i=telemetry.pop('pre_costas_i',None);pre_q=telemetry.pop('pre_costas_q',None)
    if pre_i is not None and pre_q is not None:
        raw_symbols={'i':pre_i,'q':pre_q}
    else:
        raw_sps=max(1,int(round(fs/max(cfg.symbol_rate,1.0)))) if cfg.modulation not in {'tone','multitone','awgn','chirp','custom'} else max(1,len(z)//1600)
        raw_symbols=display_complex(z[::raw_sps],2500,LIVE_DIGITS)
    syms=display_complex(r.symbols,2500,LIVE_DIGITS)
    if syms is not None and r.symbols is not None and len(r.symbols):
        syms['scale']=float(np.sqrt(np.mean(np.abs(r.symbols)**2)))
    # Keep the reference-aided channel view available without creating a second capture.
    # (Linear captures name their format in ``modulation``; the receiver must decode that same format.)
    if g.tx_symbols is not None and r.symbols is not None and len(r.symbols)>64 and g.metadata.get('modulation') == cfg.modulation in {'bpsk','qpsk','8psk','16qam','64qam'}:
        try:
            telemetry.update(reference_channel_estimate(g.tx_symbols,channel_symbols(r),limit=2048) or {})
        except (ValueError,np.linalg.LinAlgError):
            pass
    scfg=SpectrumConfig(rbw_hz=max(1.0,fs/1024),averaging=1,max_fft=2048)
    sp=analyze(z,fs,scfg);freq,db=reduce_spectrum(sp.freq_hz,sp.psd_db,640)
    # Bound the per-frame JSON cost when captures are large.
    bits=r.bits[:2048].astype(int).tolist() if r.bits is not None else []
    packed=np.packbits(r.bits[:2048]).tobytes().hex(' ') if r.bits is not None else ''
    recovered=np.asarray(r.recovered)
    return {
        'capture_id':frame,
        'tx_metadata':{k:v for k,v in g.metadata.items() if not k.startswith('grid_')},
        'channel':channel.model_dump(),
        'metrics':r.metrics,'text':r.text[:4096] if r.text else r.text,
        'symbols':syms,'raw_symbols':raw_symbols,'bits':bits,'hex':packed,
        'recovered':display_real(recovered,1600,LIVE_DIGITS),
        'recovered_complex':display_complex(recovered,1600,LIVE_DIGITS),
        'telemetry':telemetry,'frame':display_tree(frame_info,digits=LIVE_DIGITS) if frame_info is not None else None,
        'input':preview(y,g.sample_rate),'output':probe,'stages':stages,'output_rate':float(fs),
        'freq_hz':np.round(freq,3).tolist(),'psd_db':np.round(db,4).tolist(),
        'rms':float(np.sqrt(np.mean(np.abs(z)**2))) if len(z) else 0.0,
    }


def _public_metadata(g):
    return {k:v for k,v in g.metadata.items() if not k.startswith('grid_')}


MAX_IN_FLIGHT = 2
# Two frame envelopes can be in flight while spectrum and transport overlap a
# product.  _ordered_product serializes the stateful DSP itself.
REALTIME_PRODUCT_DEPTH = 2
PRODUCT_RETRY_S = 2.0


# Real-time processing chains work on at most the prefix of each capture they
# can process at the display rate; the rest of the capture is dropped, like an
# SDR dropping buffers.  The prefix never falls below MIN_WORK_SAMPLES.
MIN_WORK_SAMPLES = 8192


class _WorkSize:
    """Samples per capture a real-time chain processes, from its measured cost."""

    def __init__(self):
        self.cost_ms = None   # milliseconds per sample (smoothed)
        self.samples = None

    def observe(self, samples: int, ms: float) -> None:
        cost = ms / max(1, samples)
        self.cost_ms = cost if self.cost_ms is None else 0.7 * self.cost_ms + 0.3 * cost

    def limit(self, capture: int, fps: float) -> int | None:
        """The prefix to process (None = the whole capture)."""
        if self.cost_ms is None:
            return None
        # Leave time for capture generation, the spectrum, transport and paint.
        target = int(0.75 * 1000.0 / fps / self.cost_ms)
        target = max(MIN_WORK_SAMPLES, min(capture, target))
        # Hysteresis keeps the prefix (and the prefetch) steady.
        if self.samples is not None and 0.8 * self.samples <= target <= 1.25 * self.samples:
            target = self.samples
        self.samples = target
        return None if target >= capture else target


class _Flow:
    """Send-side state shared by the frame loop and the product sender."""

    def __init__(self, ws):
        self.ws = ws
        self.lock = asyncio.Lock()
        self.require_ack = False
        self.in_flight = 0
        self.sent_at = 0.0
        self.pending = 0                  # products queued or being sent
        self.product_fault = None          # revision with a failed runtime product
        self.product_retry_at = 0.0
        self.idle = asyncio.Event()
        self.idle.set()
        self.queue: asyncio.Queue = asyncio.Queue()

    async def send(self, message):
        async with self.lock:
            if isinstance(message, (bytes, bytearray)):
                await self.ws.send_bytes(message)
            elif isinstance(message, str):
                await self.ws.send_text(message)
            else:
                await self.ws.send_json(message)

    async def frame_end(self, seq, capture_id):
        if self.require_ack:
            await self.send({'type': 'frame_end', 'seq': seq, 'capture_id': capture_id})
            self.in_flight += 1
            self.sent_at = time.monotonic()

    def enqueue(self, seq, capture_id, revision, task):
        self.pending += 1
        self.idle.clear()
        self.queue.put_nowait((seq, capture_id, revision, task))

    async def sender(self):
        """Sends finished products in frame order, each followed by its frame_end."""
        try:
            while True:
                seq, capture_id, revision, task = await self.queue.get()
                try:
                    # Only computing the product can fail as a product; a failed
                    # send (the browser went away) ends the sender below.
                    try:
                        message = await task
                    except Exception as exc:
                        self.product_retry_at = time.monotonic() + PRODUCT_RETRY_S
                        if self.product_fault != revision:
                            self.product_fault = revision
                            if isinstance(exc, (ValueError, IndexError)):
                                log.warning("Live product rejected: %s", exc)
                            else:
                                log.exception("Live product failed")
                            await self.send({'type': 'error', 'message': str(exc)[:500], 'scope': 'product',
                                             'client_revision': revision})
                    else:
                        await self.send(message)
                        if self.product_fault == revision:
                            self.product_fault = None
                            await self.send({'type': 'product_recovered', 'client_revision': revision})
                    await self.frame_end(seq, capture_id)
                finally:
                    self.pending -= 1
                    if not self.pending:
                        self.idle.set()
        except (WebSocketDisconnect, RuntimeError):
            return  # the session ends; the frame loop sees the disconnect too


class _Invalid(ValueError):
    """A rejected part of a live update, with a readable message and the part it concerns."""
    def __init__(self, message: str, scope: str):
        super().__init__(message)
        self.scope = scope


def _parse(model, data, prefix: str, scope: str):
    try:
        return model.model_validate(data)
    except ValueError as exc:
        raise _Invalid(describe(exc, data, prefix), scope) from None


class _Pacer:
    """Maps wall-clock time to stream samples for real-time pacing."""

    def __init__(self):
        self.anchor = None  # (wall time, stream sample)

    def reset(self):
        self.anchor = None

    def position(self, capture_id, n, rate):
        """Stream sample at the current wall time (anchored at ``capture_id`` on first use)."""
        now = time.monotonic()
        if self.anchor is None:
            self.anchor = (now, capture_id * n)
        return self.anchor[1] + int((now - self.anchor[0]) * rate)

    def time_of(self, sample, rate):
        return self.anchor[0] + (sample - self.anchor[1]) / rate


def _stream_window(tx, file, record):
    """Samples between captures and sample rate (the file window may overlap)."""
    if file is None:
        return capture_samples(tx), float(tx.sample_rate)
    rate = record.sample_rate if record is not None else file.sample_rate
    return file.step_samples or file.capture_samples, (float(rate) if rate else None)


class MeasurementSpec(BaseModel):
    """Zero span and modulation quality of the live capture (``quality`` adds the demodulator view)."""
    model_config = ConfigDict(allow_inf_nan=False)
    offset_s: float = Field(default=0, ge=0)
    duration_s: float = Field(default=.002, gt=0, le=1)
    # Zero span is tuned to ``center_hz`` and filtered by a Gaussian resolution
    # bandwidth ``bandwidth_hz`` (None: the RBW the display resolves) before
    # AM/PM/FM detection; ``full_band`` detects the whole sampled band instead.
    center_hz: float = 0.0
    bandwidth_hz: float | None = Field(default=None, gt=0)
    full_band: bool = False
    quality: bool = False
    # Modulation quality of a composite measures one component (None: the first enabled one).
    component: int | None = Field(default=None, ge=0)


# Modulation quality keeps the most recent tracked symbols of consecutive
# captures: a constellation fills in even when real time leaves only a few
# symbols per capture (2 GS/s at 50 kBd: 13 per 524,288-sample capture).
QUALITY_SYMBOLS = 2048
QUALITY_TRACE_POINTS = 1000
LINEAR_MODULATIONS = {"bpsk", "qpsk", "8psk", "16qam", "64qam", "16apsk", "32apsk", "dbpsk", "dqpsk", "pi4-dqpsk", "oqpsk"}
EVM_MODULATIONS = LINEAR_MODULATIONS | {"ofdm"}   # the formats whose demodulator reports an EVM


def _evm_quality(r, rx, store, digits=None):
    """EVM view of a demodulated capture, accumulated over captures in ``store``."""
    from .modem import evm_rms, reference_constellation
    from .rx import symbol_errors
    base = {"dbpsk": "bpsk", "dqpsk": "qpsk", "pi4-dqpsk": "qpsk", "oqpsk": "qpsk"}.get(rx.modulation, rx.modulation)
    symbols = np.asarray(r.symbols if r.symbols is not None else [], dtype=np.complex64)
    settle = int(r.metrics.get("evm_settle_symbols") or 0)
    if rx.modulation in LINEAR_MODULATIONS:
        if r.telemetry.get("reacquired") or not r.telemetry.get("tracking"):
            store["symbols"] = np.zeros(0, np.complex64)   # a fresh lock starts a fresh record
            store["captures"] = 0
        record = np.concatenate([store.get("symbols", np.zeros(0, np.complex64)), symbols[settle:]])[-QUALITY_SYMBOLS:]
        store["symbols"] = record
        store["captures"] = store.get("captures", 0) + 1
        metrics = dict(r.metrics)
        metrics.update(evm_rms(record, base, rx.apsk_ratio2, rx.apsk_ratio3))
        metrics.update({"symbol_count": int(len(record)), "evm_settle_symbols": 0, "captures": int(store["captures"]),
                        "capture_symbols_tracked": int(max(0, len(symbols) - settle))})
        traces = symbol_errors(record, reference_constellation(base, rx.apsk_ratio2, rx.apsk_ratio3))
        settle_index = 0
    else:
        record = symbols
        metrics = dict(r.metrics)
        traces = {k: np.asarray(r.telemetry.get(k, [])) for k in ("evm_time_percent", "magnitude_error_percent", "phase_error_deg")}
        settle_index = int(r.telemetry.get("evm_settle_index") or 0)
    take = max(1, len(traces["evm_time_percent"]) // QUALITY_TRACE_POINTS)
    telemetry = {k: display_values(np.asarray(v)[::take], digits) for k, v in traces.items()}
    telemetry["evm_settle_index"] = settle_index // take
    for key in ("coarse_cfo_hz", "spectral_cfo_hz", "tracking", "reacquired", "ofdm_cfo_hz"):
        if key in r.telemetry:
            telemetry[key] = r.telemetry[key]
    return record, metrics, telemetry


ZERO_SPAN_TRACES = ('time_s', 'i', 'q', 'amplitude', 'phase_deg', 'frequency_hz')


def _level(mean_power, g):
    """Mean power of record ``g``'s samples on its calibrated scale (dBm, or dBFS)."""
    reference_dbm, reference_rms = g.metadata.get('power_dbm'), g.metadata.get('power_reference_rms')
    if not mean_power > 0:
        return None
    if reference_dbm is None or not reference_rms:
        return 10 * math.log10(mean_power)
    return 10 * math.log10(mean_power / reference_rms ** 2) + float(reference_dbm)


def _surroundings(g, tx, recording):
    """``around(lo, hi)``: the source's samples at positions ``lo … hi`` of
    record ``g``, generated beyond the record's ends (the transmitter is one
    stream); recordings give zeros there."""
    n, origin = len(g.samples), int(g.metadata.get('start_sample', 0))

    def stream(start, count):
        from .signal import capture_prefix, generate
        if recording or count <= 0 or origin + start < 0:
            return np.zeros(max(0, count), dtype=np.complex64)
        return generate(capture_prefix(tx, max(32, count)), start_sample=origin + start).samples[:count]

    def around(lo, hi):
        inside = np.asarray(g.samples[max(lo, 0):max(min(hi, n), 0)], dtype=np.complex64)
        return np.concatenate([stream(lo, min(0, hi) - lo), inside, stream(max(n, lo), hi - max(n, lo))])
    return around


def _quality_design(tx, component):
    """The design modulation quality measures: ``tx``, or one of its components.

    Returns ``(design, index, tuner_hz, schedules)``: the component's own
    design on the transmitter's clock, its index (None for a plain design),
    the frequency it sits at and the hop schedules to remove.
    """
    if tx.waveform.value != 'composite' or not tx.components:
        return tx, None, float(tx.center_hz), [tx.agility]
    enabled = [i for i, c in enumerate(tx.components) if c.enabled] or [0]
    index = component if component is not None and component in enabled else enabled[0]
    c = tx.components[index].model_copy(update={'sample_rate': tx.sample_rate})
    return c, index, float(tx.center_hz + c.center_hz), [tx.agility, c.agility]


def measurement_product(g, tx, recording, capture_id, m: MeasurementSpec, memory=None, digits=None, segment=None):
    """Zero span (and, with ``m.quality``, modulation quality) of capture ``g``.

    ``segment`` is the ``[start, stop)`` of the record to measure (every-sample
    pacing sweeps a record in consecutive segments); by default zero span
    measures its interval and quality the whole record.  ``memory`` (kept
    between captures by the live engine) lets the quality receiver keep
    tracking from capture to capture and accumulate the most recent
    ``QUALITY_SYMBOLS`` symbols.  ``digits`` rounds the plot traces (live
    frames); REST responses keep full precision.
    """
    from .measurements import zero_span
    fs = g.sample_rate
    origin = int(g.metadata.get('start_sample', 0))
    if m.quality:
        a, b = segment if segment is not None else (0, len(g.samples))
        out = {'sample_rate': fs, 'capture_start_s': a / fs, 'capture_duration_s': (b - a) / fs}
    else:
        if segment is not None:
            offset, duration = segment[0] / fs, (segment[1] - segment[0]) / fs
        else:
            # A high-rate real-time capture can be shorter than the chosen
            # interval: start at its beginning rather than measuring its last samples.
            offset = m.offset_s if int(m.offset_s * fs) < len(g.samples) - 1 else 0.0
            duration = m.duration_s
        out = zero_span(g.samples, fs, offset, duration, m.center_hz, m.bandwidth_hz, m.full_band, start_sample=origin,
                        around=_surroundings(g, tx, recording))
        out['level_db'] = _level(out['rms'] ** 2, g)
        out['level_unit'] = (g.metadata.get('power_unit') or 'dBm') if g.metadata.get('power_reference_rms') else 'dBFS'
        for key in ZERO_SPAN_TRACES:
            for k in (key, f'{key}_max'):
                if k in out:
                    out[k] = display_values(out[k], digits)
        # The interval was cut to the captured window (the browser then shows that window).
        out['interval_clipped'] = bool(abs(out['capture_duration_s'] - m.duration_s) * fs > 1
                                       or (segment is None and abs(out['capture_start_s'] - m.offset_s) * fs > 1))
    out['source_revision'] = g.metadata['source_revision']
    out['capture_id'] = capture_id
    out['capture_start_sample'] = g.metadata.get('start_sample')
    out['quality'] = None
    if not m.quality:
        if memory is not None:
            memory.pop('quality', None)
        return out
    x = np.asarray(g.samples[a:b], dtype=np.complex64)
    start = origin + a
    design, index, tuner, schedules = (tx, None, 0.0, []) if recording else _quality_design(tx, m.component)
    # Mixed frames intentionally do not pretend that one demodulator applies
    # to the whole record.  Their detailed quality belongs in RX framing.
    mixed = not recording and design.frame_enabled and design.frame_mode == "blocks"
    # Gated hops leave empty dwell gaps that no symbol-based EVM describes.
    gated = any(s.enabled and (s.off_s > 0 or s.ramp_s > 0) for s in schedules)
    schedules = [s for s in schedules if s.enabled]
    if schedules:
        x = dehop(x, schedules, fs, start).astype(np.complex64)
    record = None
    if not recording and not mixed and not gated and design.waveform.value in EVM_MODULATIONS:
        rx = RxConfig(modulation=design.waveform.value, sample_rate=fs, tuner_hz=tuner,
                      symbol_rate=design.symbol_rate, rolloff=design.rolloff, pulse_shape=design.pulse_shape,
                      filter_span_symbols=design.filter_span_symbols, apsk_ratio2=design.apsk_ratio2,
                      apsk_ratio3=design.apsk_ratio3, ofdm_fft=design.ofdm_fft, ofdm_cp=design.ofdm_cp,
                      ofdm_occupied=design.ofdm_occupied, ofdm_design=design.ofdm_design,
                      ofdm_start_s=start / fs, fsk_deviation_hz=design.fsk_deviation_hz,
                      fm_deviation_hz=max(1, design.fm_deviation_hz), coarse_cfo=True, timing_recovery="mueller-muller")
        store = memory.setdefault('quality', {}) if memory is not None else {}
        key = rx.model_dump_json(exclude={'ofdm_start_s'})
        if store.get('key') != key:
            store.clear()
            store['key'] = key
        r = demodulate(x, rx, start, store.setdefault('track', {}) if memory is not None else None)
        if r.metrics.get("evm_rms_percent") is not None or design.waveform.value in LINEAR_MODULATIONS:
            record, metrics, telemetry = _evm_quality(r, rx, store, digits)
    label = None if index is None else f'component {index + 1} ({tx.components[index].label or design.waveform.value})'
    if record is not None and len(record) >= 16:
        out['quality'] = {'kind': 'evm', 'metrics': metrics, 'telemetry': telemetry, 'component': index,
                          'symbols': display_complex(record, 2000, digits),
                          'method': 'Decision-directed EVM with the known modulation, rate, filter and tuning, over the '
                                    f'{len(record):,} most recent tracked symbols'
                                    f'{" of " + label if label else ""}; not a standards conformance result.'}
        return out
    quality = _waveform_quality(x, fs, start, design, tuner, recording, digits,
                                _surroundings(g, tx, recording) if not schedules else None, a)
    quality['component'] = index
    if record is not None:
        quality['method'] = (f'{len(record)} of the 16 symbols an EVM needs recovered so far: the record fills as '
                             'captures arrive (or raise the symbol rate). Meanwhile, the waveform.')
    elif mixed:
        quality['method'] = 'Mixed-modulation frame: the waveform. Use Receiver → Framing for per-block acquisition and decoding.'
    elif gated:
        quality['method'] = 'Hopping with off-time or ramps: the waveform after removing the known hop schedule. EVM needs continuous dwells.'
    elif label:
        quality['method'] = f'{label[0].upper()}{label[1:]}: {quality["method"]}'
    out['quality'] = quality
    return out


QUALITY_POINTS = 600


def _waveform_quality(x, fs, start, design, tuner, recording, digits, around=None, offset=0):
    """Modulation quality of a signal without symbols: tuned to its carrier and
    limited to its band, its I/Q trajectory, envelope, frequency and phase."""
    from .bands import content_band
    from .dsp import phasor
    from .measurements import auto_bandwidth, buckets, rbw_filter
    duration = len(x) / fs
    band = None if recording else content_band(design)
    lo, hi = band if band is not None else (0.0, 0.0)
    width = hi - lo
    center = tuner + (lo + hi) / 2
    # Twice the occupied width keeps the band's edges within 0.7 dB of the
    # Gaussian filter's peak; a tone (no width) gets the RBW the display resolves.
    rbw = max(2.0 * width, auto_bandwidth(duration, QUALITY_POINTS), 2.0 / duration)
    if band is None or rbw >= fs / 6 or abs(center) + 3 * rbw > fs / 2:
        # Unknown or too wide a band to narrow down: the full band, tuned.
        y, rate, rbw = x, fs, None
        if center:
            y = (x * phasor(-center, fs, len(x), start)).astype(np.complex64)
    else:
        # ``around`` addresses the record; ``x`` starts ``offset`` samples into it.
        shifted = None if around is None else (lambda lo, hi: around(lo + offset, hi + offset))
        y, _, rate = rbw_filter(x, fs, 0, len(x), center, rbw, start, shifted)
    amp = np.abs(y).astype(np.float64)
    valid = amp > max(float(amp.max(initial=0.0)) * .01, 1e-12)
    step = np.angle(y[1:] * np.conj(y[:-1])).astype(np.float64) if len(y) > 1 else np.zeros(0)
    phase = (float(np.angle(y[0])) + np.r_[0.0, np.cumsum(step)]) if len(y) else np.zeros(0)
    freq = np.r_[np.nan, step * rate / (2 * np.pi)] if len(y) else np.zeros(0)
    frequency_valid = valid.copy()
    if len(frequency_valid):
        frequency_valid[0] = False
        frequency_valid[1:] &= valid[:-1]
    freq[~frequency_valid] = np.nan
    phase[~valid] = np.nan
    measured = freq[frequency_valid]
    times = (start + np.arange(len(y)) * (len(x) / max(len(y), 1))) / fs
    t_low, t_high = buckets(times, QUALITY_POINTS)
    telemetry = {'time_s': display_values(t_low if t_high is None else (t_low + t_high) / 2, digits)}
    for key, values in (('amplitude', amp), ('frequency_hz', freq), ('phase_deg', np.rad2deg(phase))):
        low, high = buckets(values, QUALITY_POINTS)
        telemetry[key] = display_values(low, digits)
        if high is not None:
            telemetry[f'{key}_max'] = display_values(high, digits)
    on = amp[valid]
    if recording:
        method = ('Recording: I/Q trajectory, envelope, instantaneous frequency and phase over the whole band. '
                  'Decode it in Receiver with an explicit format.')
    elif rbw is None:
        method = ('The full band, tuned to the signal: I/Q trajectory, envelope, instantaneous frequency and phase. '
                  'EVM is not defined for this signal type.')
    else:
        method = (f'Tuned to {"the carrier" if not width else "the centre of the band"} and filtered by a {_hz(rbw)} '
                  'Gaussian RBW: I/Q trajectory, envelope, instantaneous frequency and phase. '
                  'EVM is not defined for this signal type.')
    return {
        'kind': 'basic',
        'metrics': {
            'rms': float(np.sqrt(np.mean(amp * amp))) if len(amp) else 0.0,
            'peak': float(amp.max(initial=0.0)),
            'am_percent': float(100 * np.std(on) / np.mean(on)) if len(on) > 1 and np.mean(on) > 0 else None,
            'mean_frequency_hz': float(np.mean(measured)) if len(measured) else None,
            'frequency_std_hz': float(np.std(measured)) if len(measured) else None,
            'center_hz': center, 'bandwidth_hz': rbw, 'analysis_rate': rate,
        },
        'telemetry': telemetry,
        'symbols': display_complex(y, maximum=2000, digits=digits),
        'method': method,
    }


def _hz(value):
    for scale, prefix in ((1e9, 'G'), (1e6, 'M'), (1e3, 'k')):
        if abs(value) >= scale:
            return f'{value / scale:.4g} {prefix}Hz'
    return f'{value:.4g} Hz'


def _timed(work, samples, fn, *args):
    """``fn(*args)``, reporting its duration per processed sample to ``work``."""
    start = time.perf_counter()
    result = fn(*args)
    work.observe(samples, 1000 * (time.perf_counter() - start))
    return result


def _message(kind, product, revision, *args):
    """``{'type': kind, **product(*args)}`` as compact JSON text."""
    return json.dumps({'type': kind, 'client_revision': revision, **product(*args)},
                      separators=(',', ':'), ensure_ascii=False, allow_nan=False)


async def _ordered_product(previous, work, samples, kind, product, revision, *args):
    """Compute stateful products in capture order, even with two frames in flight."""
    if previous is not None:
        try:
            await previous
        except Exception:
            pass  # sender reports the failed predecessor; this frame can recover
    return await asyncio.to_thread(_timed, work, samples, _message, kind, product, revision, *args)


def _symbol_floor(tx) -> int:
    """Samples a receiver needs per capture of ``tx``: the matched filters'
    transient plus 32 symbols (0 when the design has no symbol clock)."""
    designs = [c for c in tx.components if c.enabled] if tx.waveform.value == "composite" else [tx]
    floors = [int((2 * c.filter_span_symbols + 32) * c.sample_rate / c.symbol_rate)
              for c in designs if c.waveform.value in LINEAR_MODULATIONS]
    return max(floors, default=0)


class _CaptureStream:
    """Loads captures of the current source, prefetching the next one."""

    def __init__(self):
        self.key = None
        self.task = None
        self.task_key = None
        self.tx = None
        self.file = None
        self.limit = None  # samples per capture to load (None = all)

    def configure(self, tx, file):
        self.key = (tx.model_dump_json() if file is None else None, file.model_dump_json() if file is not None else None)
        self.tx, self.file = tx, file
        self.cancel()

    def cancel(self):
        if self.task is not None and not self.task.done():
            self.task.cancel()
        self.task = None
        self.task_key = None

    async def load(self, capture_id):
        key = (self.key, capture_id, self.limit)
        if self.task is not None and self.task_key == key:
            task, self.task, self.task_key = self.task, None, None
            try:
                return await task
            except (ValueError, IndexError, OSError, asyncio.CancelledError):
                pass
        self.cancel()
        return await asyncio.to_thread(acquire, self.tx, self.file, capture_id, self.limit)

    def prepared(self, done=False):
        """Capture id the prefetch is preparing (or, with ``done``, has finished) for this source."""
        if self.task is None or self.task_key is None or self.task_key[0] != self.key or self.task_key[2] != self.limit:
            return None
        if done and not self.task.done():
            return None
        return self.task_key[1]

    def prefetch(self, capture_id, replace=True):
        """Start preparing ``capture_id``; without ``replace`` a preparation
        already running for this source is left to finish."""
        key = (self.key, capture_id, self.limit)
        if self.task_key == key:
            return
        if not replace and self.task is not None and not self.task.done() and self.task_key[0] == self.key and self.task_key[2] == self.limit:
            return
        self.cancel()
        self.task_key = key
        self.task = asyncio.ensure_future(asyncio.to_thread(acquire, self.tx, self.file, capture_id, self.limit))
        # Errors surface when the capture is actually loaded.
        self.task.add_done_callback(lambda t: t.cancelled() or t.exception())


async def live_session(ws: WebSocket) -> None:
    """Scheduled live stream of spectrum / receiver / processing products.

    Capture identity is explicit: ``capture_id`` in every message names the
    source capture whose samples produced it.

    ``pacing`` selects how the stream (transmitter or recording) advances:

    * ``contiguous`` (default): while a receiver or processing chain is active
      every frame consumes the next capture, so state carried by the chain
      (filter memory, loop phase) is physically continuous; with only the
      spectrum displayed, frames step through the current capture and then
      advance.  Simulated time runs slower than real time when processing
      cannot keep up.
    * ``realtime``: the stream position follows the wall clock.  Each frame
      shows the capture at the current position and whatever could not be
      processed is skipped, like an SDR dropping buffers; a chain never sees
      the same capture twice and waits when it is faster than real time.
      A recording advances by its windows and holds its last window at the
      end unless it loops.
    """
    from .recordings import FileSource
    await ws.accept()
    try:
        await ws.send_json({'type':'hello','protocol':'VRF1','recommended_live_sample_rate':RECOMMENDED_LIVE_SAMPLE_RATE,
                            'max_sample_rate':MAX_SAMPLE_RATE,'max_capture_samples':MAX_CAPTURE_SAMPLES})
    except (WebSocketDisconnect, RuntimeError):
        return  # the page went away before the session started
    tx, spec = TxConfig(duration=.06), SpectrumConfig(max_fft=32768, averaging=2)
    file = None
    fps, seq, capture_id, cursor, dirty, running = 30., 0, 0, 0, True, True
    record = None
    # Acknowledged backpressure with one frame of lookahead: the next frame is
    # computed while the browser draws the previous one, and never more than
    # MAX_IN_FLIGHT frames await acknowledgement.
    flow = _Flow(ws)
    sender = asyncio.ensure_future(flow.sender())
    work = _WorkSize()
    work_signature = None
    previous_product = None
    pipeline=None;channel=ChannelConfig();rx=None;measure=None;memory={};pipe_signature=None
    deadline = time.monotonic() + .05
    axis_signature = None
    stream = _CaptureStream()
    stream.configure(tx, file)
    advance_next = False
    sweep = 0           # next sample of the record an every-sample measurement measures
    pacing = 'contiguous'
    client_revision = 0
    pacer = _Pacer()
    processed = None    # last capture a chain consumed under real-time pacing
    frame_period = 1 / fps   # smoothed time between frame starts (real-time prefetch)
    last_frame_start = None
    # One receive is kept pending across iterations: cancelling and re-creating
    # it on every timeout made the event loop compete for the GIL with the
    # product threads (a GIL convoy that slowed products several-fold).
    recv = asyncio.ensure_future(ws.receive_text())
    idle = None
    try:
        while True:
            # Deadline is independent of incoming edits, so controls cannot starve the stream.
            # While the browser holds MAX_IN_FLIGHT unacknowledged frames, block
            # until an acknowledgement arrives (or the recovery timeout); while a
            # product is still being computed, until it has been sent.  Never poll.
            blocked = flow.in_flight >= MAX_IN_FLIGHT
            busy = (rx is not None or pipeline is not None or measure is not None) and flow.pending >= (REALTIME_PRODUCT_DEPTH if pacing == 'realtime' else 1)
            waiters = {recv}
            if blocked:
                wake = flow.sent_at + 2
            elif busy:
                wake = None
                if idle is None or idle.done():
                    idle = asyncio.ensure_future(flow.idle.wait())
                waiters.add(idle)
            else:
                wake = deadline
            try:
                done, _ = await asyncio.wait(waiters, timeout=None if wake is None else max(.001, wake-time.monotonic()),
                                             return_when=asyncio.FIRST_COMPLETED)
                if recv not in done:
                    raise asyncio.TimeoutError
                message = recv.result()
                recv = asyncio.ensure_future(ws.receive_text())
                req = json.loads(message)
                if req.get('action')=='ack':
                    flow.in_flight=max(0,flow.in_flight-1)
                elif req.get('action') == 'stop':
                    running = False
                elif req.get('action') in {'start','update'}:
                    if req.get('action')=='start' and 'capture_id' in req:
                        capture_id=max(0,min(2147483647,int(req['capture_id'])))
                    new_pacing = req.get('pacing', pacing)
                    new_client_revision = max(0, min(2147483647, int(req.get('client_revision', client_revision))))
                    if new_pacing not in ('contiguous', 'realtime'):
                        raise ValueError("pacing must be 'contiguous' or 'realtime'")
                    flow.require_ack=bool(req.get('acknowledge_frames',flow.require_ack))
                    # A bad receiver or chain setting disables only that product; the
                    # source and spectrum still follow the update.
                    product_errors = []
                    try:
                        new_pipeline=_parse(Pipeline,req['pipeline'],'Processing chain','pipeline') if req.get('pipeline') is not None else None
                    except _Invalid as exc:
                        new_pipeline=None;product_errors.append(exc)
                    new_channel=_parse(ChannelConfig,req.get('channel') or channel.model_dump(),'Channel','channel')
                    try:
                        new_rx=_parse(RxConfig,req['rx'],'Receiver','rx') if req.get('rx') is not None else None
                    except _Invalid as exc:
                        new_rx=None;product_errors.append(exc)
                    try:
                        new_measure=_parse(MeasurementSpec,req['measurement'],'Measurement','measurement') if req.get('measurement') is not None else None
                    except _Invalid as exc:
                        new_measure=None;product_errors.append(exc)
                    new_tx = _parse(TxConfig,req.get('tx',tx.model_dump()),'Transmitter','tx')
                    if 'source' in req:
                        new_file = _parse(FileSource,req['source'],'Recording','source') if req.get('source') else None
                    else:
                        new_file = file
                    new_spec = _parse(SpectrumConfig,req.get('spectrum',spec.model_dump()),'Spectrum','spectrum')
                    signature=json.dumps([new_pipeline.model_dump(exclude={'probe'}) if new_pipeline else None,new_channel.model_dump(),
                                          new_tx.model_dump() if new_file is None else None,new_file.model_dump() if new_file else None],sort_keys=True)
                    if signature!=pipe_signature:memory={};pipe_signature=signature
                    budget_signature=(signature, new_rx.model_dump_json() if new_rx else None,
                                      new_measure.model_dump_json() if new_measure else None)
                    if budget_signature != work_signature:
                        work = _WorkSize()
                        work_signature = budget_signature
                    pipeline,channel,rx,measure=new_pipeline,new_channel,new_rx,new_measure
                    # Spectrum-only edits do not require regenerating the source.
                    source_changed = (new_file != file) if (new_file is not None or file is not None) else (new_tx != tx)
                    if source_changed and new_file is not None and (file is None or new_file.name != file.name or new_file.start_sample != file.start_sample):
                        capture_id = 0
                    if req.get('seek_capture') is not None:
                        capture_id = max(0, min(2147483647, int(req['seek_capture'])))
                        source_changed = True
                    if source_changed:
                        dirty = True
                        cursor = 0
                    if source_changed or new_pacing != pacing or req.get('action') == 'start':
                        pacer.reset()
                        processed = None
                    pacing = new_pacing
                    client_revision = new_client_revision
                    flow.product_fault = None
                    flow.product_retry_at = 0.0
                    tx, spec, file = new_tx, new_spec, new_file
                    if source_changed:
                        stream.configure(tx, file)
                    fps = float(np.clip(req.get('fps',fps),2,60))
                    running = True
                    await flow.send({'type':'accepted','capture_id':capture_id,'client_revision':client_revision})
                    for problem in product_errors:
                        await flow.send({'type':'error','message':str(problem)[:1000],'scope':problem.scope})
            except asyncio.TimeoutError:
                pass
            except _Invalid as exc:
                await flow.send({'type':'error','message':str(exc)[:1000],'scope':exc.scope})
                continue
            except (ValueError,TypeError) as exc:
                await flow.send({'type':'error','message':str(exc)[:1000],'scope':'request'})
                continue
            if flow.in_flight >= MAX_IN_FLIGHT:
                if time.monotonic()-flow.sent_at<2:
                    continue
                flow.in_flight=0  # an acknowledgement was lost: recover
            now = time.monotonic()
            if now < deadline:
                continue
            # Frames follow an ideal 1/fps grid: timer wake-ups are coarse (15.6 ms
            # on Windows) and re-arming from "now" would add that error to every
            # frame period; a stream that fell behind restarts the grid.
            deadline = deadline + 1/fps if now - deadline < 1/fps else now + 1/fps
            if not running:
                continue
            try:
                # Every product that consumes whole captures (receiver, chain,
                # zero span / quality) is paced like a processing chain.
                chain_active = rx is not None or pipeline is not None or measure is not None
                n, rate = _stream_window(tx, file, record)
                capture_window = file.capture_samples if file is not None else n
                realtime = pacing == 'realtime' and rate is not None
                depth = REALTIME_PRODUCT_DEPTH if realtime else 1
                if chain_active and flow.pending >= depth:
                    continue  # the loop waits for the product sender (see above)
                # The measurement is the product only without a receiver or chain.
                measuring = measure is not None and rx is None and pipeline is None
                zero = measuring and not measure.quality
                # Real time drops what the chain cannot process at the display rate.
                # A zero-span sweep costs what its interval costs: it reads that
                # interval whole and generates no further than its end.
                limit = work.limit(capture_window, fps) if realtime and chain_active and not zero else None
                if realtime and zero and rate:
                    from .measurements import settling_samples
                    need = math.ceil((measure.offset_s + measure.duration_s) * rate) + settling_samples(
                        rate, measure.duration_s, measure.bandwidth_hz, measure.full_band)
                    need = min(capture_window, max(MIN_WORK_SAMPLES, need))
                    limit = need if need < capture_window else None
                if limit is not None and file is None and (rx is not None or (measure is not None and measure.quality)):
                    # A receiver needs a few symbols beyond its filters' transient in
                    # every capture it keeps, however expensive the samples are.
                    minimum = _symbol_floor(tx)
                    if minimum:
                        limit = None if minimum >= capture_window else max(limit, minimum)
                if limit != stream.limit:
                    stream.limit = limit
                    if record is not None and len(record.samples) != (limit or capture_window):
                        record = None
                next_capture = (capture_id + 1) & 0x7fffffff
                if realtime:
                    position = pacer.position(capture_id, n, rate)
                    target = (position // n) & 0x7fffffff
                    held = file is not None and not file.loop and record is not None and record.metadata.get('at_end')
                    if held:
                        target = min(target, capture_id)  # the recording has ended: hold its last window
                    if chain_active and target == processed:
                        # Faster than real time (or at the end of a recording): wait for the next capture.
                        deadline = time.monotonic() + 0.25 if held else max(time.monotonic() + 0.005, pacer.time_of((target + 1) * n, rate))
                        continue
                    # Show the capture the prefetcher already prepared rather than
                    # generating the exact current one in series, when it lies
                    # within one frame period of real time (either side: frames
                    # jitter around the predicted time, and a window that only
                    # accepted late predictions made half the frames wait).
                    ready = stream.prepared()
                    ahead = max(1, math.ceil(frame_period * rate / n))
                    if ready is not None and target - ahead <= ready <= target + ahead and (processed is None or ready > processed):
                        target = ready
                    if not chain_active and record is not None and target != capture_id and stream.prepared(done=True) != target:
                        # The spectrum alone does not wait for generation: it keeps
                        # stepping through the current capture until the next is ready.
                        pass
                    else:
                        if target != capture_id or record is None:
                            capture_id, record = target, None
                        # The newest spectrum chunk ends at the current position.
                        cursor = max(0, position % n - required_samples(rate, spec))
                    next_capture = ((position + int(frame_period * rate)) // n) & 0x7fffffff
                    if next_capture <= capture_id:
                        next_capture = (capture_id + 1) & 0x7fffffff
                elif advance_next and record is not None and not dirty and not record.metadata.get('at_end'):
                    capture_id = (capture_id + 1) & 0x7fffffff
                    next_capture = (capture_id + 1) & 0x7fffffff
                    record = None
                    cursor = 0
                advance_next = False
                frame_started = time.monotonic()
                if last_frame_start is not None:
                    frame_period += 0.3 * (min(frame_started - last_frame_start, 2.0) - frame_period)
                last_frame_start = frame_started
                if dirty or record is None:
                    record = await stream.load(capture_id)
                    sweep = 0
                    if dirty:
                        dirty, axis_signature = False, None
                        # Products of the previous source go out before its successor's metadata.
                        await flow.idle.wait()
                        await flow.send({'type':'meta','client_revision':client_revision,'sample_rate':record.sample_rate,'waveform':record.metadata.get('waveform',tx.waveform.value),
                                         'source_kind':record.metadata.get('source_kind','transmitter'),'metadata':_public_metadata(record),'spectrum':spec.model_dump()})
                stream.prefetch(next_capture, replace=not realtime)
                # Every-sample pacing measures a record in consecutive segments: zero
                # span sweeps intervals of its length, quality takes what it can
                # process per frame (at least the symbols a receiver needs), so
                # every sample is measured and the stream runs slower than real time.
                segment = None
                if measuring and not realtime:
                    total = len(record.samples)
                    if zero:
                        size = max(2, int(round(measure.duration_s * record.sample_rate)))
                    else:
                        size = max(work.limit(total, fps) or total, MIN_WORK_SAMPLES, _symbol_floor(tx) if file is None else 0)
                    size = min(size, total)
                    # The last segment of a record ends at its end (overlapping the one before).
                    segment = (min(sweep, total - size), min(sweep, total - size) + size)
                    sweep = segment[1]
                start = time.perf_counter()
                sr, f, p, cursor, exhausted = await asyncio.to_thread(frame_product,record,spec,cursor)
                # Frequency coordinates are invariant until source-rate/analyzer
                # settings change.  Sending them once removes most live JSON work.
                current_axis=(float(record.sample_rate),spec.model_dump_json(),len(f),float(f[0]) if len(f) else 0.0,float(f[-1]) if len(f) else 0.0)
                axis_changed=current_axis!=axis_signature
                axis_signature=current_axis
                meta=record.metadata
                metrics={'type':'metrics','seq':seq,'client_revision':client_revision,'source_revision':meta['source_revision'],
                    'rbw_actual_hz':sr.rbw_actual_hz,'rbw_limited_by_capture':sr.rbw_limited_by_capture,'nfft':sr.nfft,
                    'bw3_hz':sr.bandwidth_3db_hz,'bw99_hz':sr.bandwidth_99_hz,'peak_hz':sr.peak_hz,'peak_db':sr.peak_db,'power_unit':sr.unit,
                    'capture_id':capture_id,'capture_start_sample':meta.get('start_sample'),'capture_samples':len(record.samples),
                    'simulation_time_s':meta.get('simulation_time_s'),'source_kind':meta.get('source_kind','transmitter'),
                    'at_end':bool(meta.get('at_end')),'recording_samples':meta.get('recording_samples'),
                    'center_frequency_hz':meta.get('center_frequency_hz'),
                    'center_hz':sr.center_hz,'span_hz':sr.span_hz,'decimation':sr.decimation,'analysis_rate':sr.analysis_rate,
                    'channels':sr.channels,
                    'display_fps':fps,'compute_ms':1000*(time.perf_counter()-start),'cursor':cursor,'pacing':'realtime' if realtime else 'contiguous',
                    'window_samples':capture_window,'processed_samples':len(record.samples)}
                if axis_changed: metrics['freq_hz']=f.tolist()
                await flow.send(metrics)
                await flow.send(pack_f32(FrameType.PSD,seq,p))
                # Products are computed and serialized in worker threads (so the
                # event loop stays free for control messages) and sent in frame
                # order by the product sender, which also ends the frame.
                if rx is not None:
                    product_kind, product_fn, product_args = 'receiver', receiver_product, (record,channel,pipeline or Pipeline(),rx,capture_id,memory)
                elif pipeline is not None:
                    product_kind, product_fn, product_args = 'pipeline', pipeline_product, (record,channel,pipeline,capture_id,memory)
                elif measure is not None:
                    product_kind, product_fn, product_args = 'measurement', measurement_product, (record,tx,file is not None,capture_id,measure,memory,LIVE_DIGITS,segment)
                else:
                    product_kind = None
                if product_kind is not None and (flow.product_fault != client_revision or
                                                 time.monotonic() >= flow.product_retry_at):
                    measured = segment[1] - segment[0] if segment is not None else len(record.samples)
                    previous_product = asyncio.ensure_future(_ordered_product(
                        previous_product, work, measured, product_kind, product_fn, client_revision, *product_args))
                    flow.enqueue(seq, capture_id, client_revision, previous_product)
                else:
                    await flow.frame_end(seq, capture_id)
                seq=(seq+1)&0x7fffffff
                if chain_active:
                    processed = capture_id
                # A processing chain consumes one contiguous capture per frame, a
                # measurement once it has swept the whole record; the spectrum
                # alone advances once it has shown the whole capture.
                advance_next = sweep >= len(record.samples) if segment is not None else (chain_active or exhausted)
            except (ValueError,IndexError,OSError) as exc:
                # Keep the last valid stream alive after a bad transient edit.
                await flow.send({'type':'error','message':str(exc)[:500]})
                dirty=True;record=None
                stream.configure(tx, file)
    except (WebSocketDisconnect, RuntimeError):
        return
    finally:
        stream.cancel()
        sender.cancel()
        recv.cancel()
        if idle is not None:
            idle.cancel()
