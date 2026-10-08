/* The live source (transmitter or recording) every observer shares, and the
   request fragments that name it. */
import {publish} from './events.js';
import {state, clone} from './state.js';

/**
 * The transmitter design the backend last accepted.  Edits go into
 * ``state.tx`` (the draft); only a draft the backend validated becomes the
 * applied design, which the live stream, every instrument and their labels
 * use.  A rejected edit therefore never leaves a display labelled with a
 * design it is not showing.
 */
export function appliedTx() {
  return state.txApplied || designRequest();
}

/** Make a validated design (a ``designRequest()``) the one on air. */
export function applyTx(design) {
  state.txApplied = clone(design);
  publish('tx:applied', state.txApplied);
}

export function activeFile() {
  return state.source.kind === 'file' && state.source.file ? state.source.file : null;
}

export function activeRate() {
  const file = activeFile();
  const rate = appliedTx().sample_rate;
  return file ? Number(state.source.info?.sample_rate || file.sample_rate) || rate : rate;
}

/** RF frequency that baseband 0 Hz represents for the active source (0 = plain baseband). */
export function sourceRfCenter() {
  const file = activeFile();
  if (file) return Number(file.center_frequency_hz ?? state.source.info?.center_frequency_hz) || 0;
  return Number(appliedTx().rf_center_hz) || 0;
}

/** Offset the spectrum axis adds to baseband frequencies. */
export function axisOffset() {
  return state.specUi?.rfAxis ? sourceRfCenter() : 0;
}

export function powerUnit(metrics) {
  const unit = metrics?.power_unit;
  if (unit && unit !== 'dBFS/Hz') return unit;
  return activeFile()?.power_unit || 'dBm';
}

/** Capture length in samples allowed at a rate. */
export function maxCaptureSeconds(rate = state.tx.sample_rate) {
  return state.limits.max_capture_samples / rate;
}

/* Under real-time pacing a live capture holds at most this many samples, so the
   work per update is bounded at any sample rate (5.2 ms at 100 MS/s); captures
   that cannot be processed are skipped.  "Every sample" pacing uses the full
   designed capture. */
export const LIVE_WINDOW_SAMPLES = 524_288;
const LIVE_FPS = 30;
const liveLimit = (rate, minimum = 32) => Math.max(minimum, Math.min(LIVE_WINDOW_SAMPLES, Math.floor(rate / LIVE_FPS)));

/** The transmitter design as written (design files, the design plan). */
export function designRequest(tx = state.tx) {
  const cfg = clone(tx);
  if (!state.txUi.noise || cfg.waveform === 'awgn') cfg.snr_db = null;
  cfg.duration = Math.min(cfg.duration, maxCaptureSeconds(cfg.sample_rate));
  return cfg;
}

const LINEAR = new Set(['bpsk', 'qpsk', 'dbpsk', 'dqpsk', 'pi4-dqpsk', 'oqpsk', '8psk', '16qam', '64qam', '16apsk', '32apsk']);

/**
 * Samples a symbol receiver needs in one capture: its matched filters'
 * transient plus 32 symbols (0 for designs without a linear symbol clock).
 * At 2 GS/s and 50 kBd that is 2.6 M samples, five times the real-time window.
 */
export function symbolFloor(cfg = appliedTx()) {
  const designs = cfg.waveform === 'composite' ? cfg.components.filter(c => c.enabled) : [cfg];
  return Math.max(0, ...designs.filter(c => LINEAR.has(c.waveform))
    .map(c => Math.ceil((2 * c.filter_span_symbols + 32) * cfg.sample_rate / c.symbol_rate)));
}

/** Whether the open view demodulates symbols (receiver pages, modulation quality). */
function demodulating() {
  return (state.mode === 'rx' && !['monitor', 'pipeline'].includes(state.rxUi?.page)) || (state.mode === 'spectrum' && state.specUi?.page === 'quality');
}

/** The applied transmitter as the live instruments observe it (live stream and every HTTP observer). */
export function txRequest() {
  const cfg = clone(appliedTx());
  if (state.pacing === 'realtime') {
    // A receiver gets captures long enough to hold symbols (fewer updates per
    // second at extreme oversampling); every other view the 1/30 s window.
    const floor = demodulating() ? Math.min(symbolFloor(cfg), state.limits.max_capture_samples) : 0;
    const window = Math.max(liveLimit(cfg.sample_rate), floor) / cfg.sample_rate;
    if (cfg.duration > window) {
      // A shorter live window shows slices of the same stream; only a chirp whose
      // sweep period is "the capture length" depends on it, so pin that period.
      for (const c of [cfg, ...(cfg.components || [])]) if (c.waveform === 'chirp' && c.chirp_period_s == null) c.chirp_period_s = cfg.duration;
      cfg.duration = window;
    }
  }
  return cfg;
}

/** Samples per live capture for a designed capture of ``samples``. */
export function liveCaptureSamples(samples, rate = activeRate()) {
  return state.pacing === 'realtime' ? Math.min(samples, liveLimit(rate)) : samples;
}

/** FileSource for the selected library recording (with optional overrides): its window and step as set. */
export function recordingSpec(overrides = {}) {
  const f = state.files.selected;
  if (!f) return null;
  const numberOrNull = v => (v === null || v === undefined || v === '' ? null : Number(v));
  const step = numberOrNull(state.playback.step);
  return {
    kind: 'file', name: f.name, format: f.format || null, sample_rate: numberOrNull(f.sample_rate),
    center_frequency_hz: numberOrNull(f.center_frequency_hz), header_bytes: Number(f.header_bytes) || 0,
    start_sample: Math.max(0, Math.round(Number(f.start_sample) || 0)),
    capture_samples: Math.max(64, Math.round(Number(f.capture_samples) || 65536)),
    step_samples: step ? Math.max(1, Math.round(step)) : null, loop: f.loop !== false, power_offset_db: Number(f.power_offset_db) || 0,
    power_unit: f.power_unit || 'dBFS', ...overrides,
  };
}

/** A recording spec as the live stream plays it under the current pacing. */
function paced(spec) {
  if (!spec || state.pacing !== 'realtime') return spec;
  const rate = spec.sample_rate || activeRate();
  const capture = Math.min(spec.capture_samples, liveLimit(rate, 64));
  // A low-rate recording needs overlapping windows when fewer than 64 new
  // samples arrive per display frame; otherwise windows advance without overlap.
  return {...spec, capture_samples: capture, step_samples: Math.min(capture, Math.max(1, Math.floor(rate / LIVE_FPS)))};
}

/** The recording the live source is pinned to (it stays live while another file is inspected). */
export function liveSource() {
  if (state.source.kind !== 'file' || !state.source.file) return null;
  return paced(state.files.selected?.name === state.source.file.name ? recordingSpec() : state.source.file);
}

/** Keep the live source in step with edits to the same recording. */
export function syncLiveSource() {
  if (state.source.kind === 'file' && state.files.selected?.name === state.source.file?.name) {
    state.source.file = recordingSpec();
  }
}

/** Body fragment naming the capture every observer shares. */
export function sourceRequest() {
  return {tx: txRequest(), source: liveSource(), capture_id: state.captureId || 0};
}

export function sourceLabel() {
  const file = activeFile();
  if (file) return {kind: 'file', text: file.name};
  const tx = appliedTx();
  const what = tx.waveform === 'composite' ? `composite · ${tx.components.filter(c => c.enabled).length} signals` : tx.waveform.toUpperCase();
  return {kind: 'transmitter', text: `${what}${tx.agility?.enabled ? ` · ${tx.agility.mode}` : ''}`};
}
