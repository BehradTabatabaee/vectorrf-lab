/* Pieces shared by the observer workspaces: the source switch, analyzer
   tuning requests and the host that connects an Analyzer to live frames. */
import {esc} from '../core/dom.js';
import {tip} from '../core/controls.js';
import {fmt, fmtCount} from '../core/format.js';
import {live, update as liveUpdate} from '../core/live.js';
import {subscribe} from '../core/events.js';
import {state} from '../core/state.js';
import {activeFile, activeRate, appliedTx, axisOffset, liveCaptureSamples, maxCaptureSeconds, powerUnit, recordingSpec, sourceRfCenter} from '../core/source.js';
import {renderStatus, setStatus, toast} from '../core/ui.js';
import {Analyzer} from '../plots/analyzer.js';

/** Source switch plus a readout of what the observers currently see. */
export function sourceSection() {
  const file = activeFile();
  const selected = state.files.selected;
  const toggle = `<div class="segmented" role="group" aria-label="Live source">`
    + `<button type="button" data-action="source-tx" class="${file ? '' : 'active'}">Transmitter</button>`
    + `<button type="button" data-action="source-file" class="${file ? 'active' : ''}"${selected || file ? '' : ' disabled title="Choose a recording in Recordings first"'}>`
    + `${esc(file ? `File · ${file.name}` : selected ? `File · ${selected.name}` : 'Recording')}</button></div>`;
  const m = live.metrics, applied = appliedTx();
  const rows = file
    ? [['Recording', file.name], ['Format', state.source.info?.format || file.format || '—'], ['Rate', fmt(activeRate(), 'S/s')],
       ['RF centre', sourceRfCenter() ? fmt(sourceRfCenter(), 'Hz') : '—'],
       ['Position', Number.isFinite(m.capture_start_sample) && m.source_kind === 'file' ? `${fmt(m.capture_start_sample / activeRate(), 's')} · capture ${m.capture_id}` : '—']]
    : [['Signal', applied.waveform === 'composite' ? `Composite · ${applied.components.filter(c => c.enabled).length} signals` : applied.waveform.toUpperCase()],
       ['Rate', fmt(applied.sample_rate, 'S/s')], ['RF centre', applied.rf_center_hz ? fmt(applied.rf_center_hz, 'Hz') : 'baseband'],
       ['Capture', captureText()]];
  return `${toggle}<div class="source-card">${rows.map(([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}</div>`
    + pacingControl();
}

const PACING_HELP = '<b>Real time</b>: the stream follows the wall clock; each update shows the capture at the current time and skips what could not be processed, like an SDR dropping buffers. <b>Every sample</b>: every capture is processed in order (filter and loop state stay continuous), so simulated time runs slower than real time when processing cannot keep up.';

export function pacingControl() {
  const current = state.pacing;
  return `<div class="field"><label${tip(PACING_HELP)}>Pacing</label><div class="segmented" role="group" aria-label="Live pacing">`
    + [['realtime', 'Real time'], ['contiguous', 'Every sample']].map(([value, label]) =>
      `<button type="button" data-action="pacing" data-pacing="${value}" class="${value === current ? 'active' : ''}">${label}</button>`).join('')
    + '</div></div>';
}

/** Set the live pacing everywhere ("realtime" or "contiguous"). */
export function setPacing(value) {
  state.pacing = value;
  for (const b of document.querySelectorAll('[data-action="pacing"]')) b.classList.toggle('active', b.dataset.pacing === value);
  state.revision += 1;
  liveUpdate(true);
  renderStatus();
}

/** Source switching and pacing actions shared by the observing workspaces. */
export function sourceActions(rerender) {
  return {
    'source-tx': () => switchSource('transmitter', {rerender}),
    'source-file': () => switchSource('file', {rerender}),
    pacing: el => { setPacing(el.dataset.pacing); rerender?.(); },
  };
}

function captureText() {
  const tx = appliedTx();
  const designed = Math.round(Math.min(tx.duration, maxCaptureSeconds(tx.sample_rate)) * tx.sample_rate);
  const used = liveCaptureSamples(designed, tx.sample_rate);
  return `${fmtCount(used)} samples${used < designed ? ' (real-time window)' : ''} · #${state.captureId ?? 0}`;
}

/** Switch every observer between the transmitter and the selected recording. */
export function switchSource(kind, {rerender} = {}) {
  const toFile = kind === 'file';
  if (toFile === (state.source.kind === 'file')) return;
  if (toFile) {
    const spec = state.files.selected && {...state.files.selected};
    if (!spec) { toast('Choose a recording in Recordings first'); return; }
    if (!(state.files.info?.sample_rate || spec.sample_rate)) { toast('Enter the sample rate of this raw recording first'); return; }
  }
  state.source = toFile ? {kind: 'file', file: recordingSpec(), info: state.files.info} : {kind: 'transmitter', file: null, info: null};
  // A recording already contains its real channel: bypass the simulated one by default.
  state.channel.enabled = !toFile;
  if (toFile) state.rxUi.follow = false;
  state.spectrum.center_hz = 0;
  state.spectrum.span_hz = null;
  state.captureId = 0;
  state.revision += 1;
  state.rxRevision += 1;
  liveUpdate(true);
  renderStatus();
  setStatus(toFile ? 'recording streaming' : 'transmitter streaming');
  toast(toFile ? `Live source: ${state.source.file.name}` : 'Live source: transmitter');
  rerender?.();
}

/** SpectrumConfig for a tuning object {center_hz, span_hz} plus the analyzer settings. */
export function spectrumRequest(tune = state.spectrum) {
  const s = state.spectrum, rate = activeRate();
  const span = Math.max(100, Math.min(state.limits.max_span, Number(tune.span_hz ?? rate) || rate));
  const rbw = state.specUi.autoRbw ? Math.max(0.5, span / state.specUi.rbwRatio) : s.rbw_hz;
  return {center_hz: Number(tune.center_hz) || 0, span_hz: span, rbw_hz: rbw, window: s.window, detector: s.detector,
          averaging: s.averaging, max_fft: s.max_fft, channels: tune === state.spectrum ? s.channels : []};
}

/** Full-span tuning (used by the transmitter and classifier monitors). */
export function fullTune() {
  return {center_hz: 0, span_hz: activeRate()};
}

/**
 * Connects an Analyzer to live frames and a tuning object.  ``tune`` is
 * mutated in place ({center_hz, span_hz}); the live request is refreshed
 * after the pointer settles.
 */
export class AnalyzerHost {
  constructor(root, {tune, ui, channels = null, selectionActions = [], onSelectionAction, bands, onBandClick, onTuned, waterfall = true, hint}) {
    this.tune = tune;
    this.channelsEditable = !!channels;
    this.analyzer = new Analyzer(root, ui, {
      waterfall,
      getTune: () => this.current(),
      onTune: (change, live) => this.retune(change, live),
      axisOffset: () => axisOffset(),
      unit: () => powerUnit(live.metrics),
      rbw: () => live.metrics.rbw_actual_hz,
      dataBounds: () => this.dataBounds(),
      channels: channels ? () => state.spectrum.channels : null,
      channelResults: channels ? () => live.metrics.channels || [] : null,
      onChannels: channels ? (index, change, done) => channels(index, change, done) : null,
      selectionActions, onSelectionAction, bands, onBandClick, hint,
      extraAnnotation: () => (live.metrics.decimation > 1 ? `   Zoom ×${live.metrics.decimation}` : ''),
      onChange: () => {},
    });
    this.onTuned = onTuned;
    this.unsubscribe = [
      subscribe('live:psd', psd => this.frame(psd)),
      subscribe('live:meta', () => this.analyzer.resetTraces()),
    ];
    this.timer = null;
    this.analyzer.renderStrip();
    this.analyzer.render();
  }

  current() {
    const rate = activeRate();
    const span = Math.max(100, Math.min(state.limits.max_span, Number(this.tune.span_hz ?? rate) || rate));
    return {center: Number(this.tune.center_hz) || 0, span};
  }

  dataBounds() {
    const f = live.freq;
    if (f?.length >= 2) return {left: f[0], right: f.at(-1)};
    const half = activeRate() / 2;
    return {left: -half, right: half};
  }

  /** Apply a tuning change at once on screen; send it once the pointer settles. */
  retune(change, settling = false) {
    const rate = activeRate();
    if (change === 'full') { this.tune.center_hz = 0; this.tune.span_hz = null; }
    else {
      if (change.span !== undefined) this.tune.span_hz = Math.max(100, Math.min(state.limits.max_span, change.span));
      if (change.center !== undefined) this.tune.center_hz = change.center;
      const span = this.current().span;
      // Keep the window over sampled data unless the requested span is wider than the source.
      if (span <= rate) this.tune.center_hz = Math.max(-rate / 2 + span / 2, Math.min(rate / 2 - span / 2, this.tune.center_hz));
      else this.tune.center_hz = 0;
      if (this.tune.span_hz >= rate * 0.999 && Math.abs(this.tune.center_hz) < 1) this.tune.span_hz = null;
    }
    this.analyzer.render();
    clearTimeout(this.timer);
    this.timer = setTimeout(() => { liveUpdate(true); this.onTuned?.(); }, settling ? 60 : 160);
  }

  frame(psd) {
    const m = live.metrics;
    const signature = JSON.stringify([m.source_revision, m.center_hz, m.span_hz, m.rbw_actual_hz, live.freq.length, live.freq[0]]);
    this.analyzer.ingest(live.freq, psd, signature);
    this.analyzer.renderStrip();
    this.analyzer.render();
  }

  destroy() {
    clearTimeout(this.timer);
    for (const off of this.unsubscribe) off();
  }
}

export function followBadge() {
  if (activeFile()) return '<span class="badge info" title="Receiving a recording">Recording</span>';
  return state.rxUi.follow
    ? '<span class="badge good" title="Receiver settings track the transmitter design">Following TX</span>'
    : '<span class="badge warn" title="Receiver settings were edited by hand">Manual · <button type="button" data-action="refollow">follow TX</button></span>';
}
