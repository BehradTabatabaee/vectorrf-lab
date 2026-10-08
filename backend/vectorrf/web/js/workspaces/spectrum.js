/* Spectrum analyzer workspace: swept display, zero span and modulation quality. */
import {$, esc, replaceHTML, setText} from '../core/dom.js';
import {bindControls, numberField, pane, readout, section, segmented, selectField, toggleField, views} from '../core/controls.js';
import {publish, subscribe} from '../core/events.js';
import {fmt, fmtCount, fmtFreq} from '../core/format.js';
import {live, update as liveUpdate} from '../core/live.js';
import {state} from '../core/state.js';
import {activeRate, appliedTx, axisOffset, powerUnit} from '../core/source.js';
import {toast} from '../core/ui.js';
import {analyzerMarkup, CHANNEL_COLORS} from '../plots/analyzer.js';
import {linePlot, scatter} from '../plots/canvas.js';
import {AnalyzerHost, sourceActions, sourceSection, spectrumRequest} from './common.js';
import {idealConstellation} from './constellations.js';

let root = null;
let host = null;
let unsubscribe = [];
let measurement = null;   // the zero span / quality product on screen

const PAGES = [['sweep', 'Spectrum'], ['zero', 'Zero span'], ['quality', 'Modulation quality']];
const TRACE_MODES = [['clear', 'Clear / write'], ['max', 'Max hold'], ['min', 'Min hold'], ['average', 'Average'], ['view', 'View (freeze)'], ['off', 'Off']];
const MEMORY_COLORS = ['#d7a9f7', '#f49ac1', '#9bc8f2', '#cfd6db'];

export const usesLive = true;

export function payload() {
  return {spectrum: spectrumRequest(state.spectrum), ...(measuring() ? {measurement: measurementRequest()} : {})};
}

function span() {
  return Math.max(100, Math.min(state.limits.max_span, Number(state.spectrum.span_hz ?? activeRate()) || activeRate()));
}

/* ---------------------------------------------------------------- dock -- */

function frequencySection() {
  const off = axisOffset();
  const s = span(), c = Number(state.spectrum.center_hz) || 0;
  state.specView = {start: c - s / 2, stop: c + s / 2};
  return section('Frequency', numberField('Center', 'spectrum.center_hz', {unit: 'Hz', offset: off, help: off ? `Shown as RF: ${fmt(off, 'Hz')} source centre + baseband offset.` : ''})
    + numberField('Span', 'spectrum.span_hz', {unit: 'Hz', value: s, min: 100, max: state.limits.max_span})
    + numberField('Start', 'specView.start', {unit: 'Hz', offset: off})
    + numberField('Stop', 'specView.stop', {unit: 'Hz', offset: off})
    + `<div class="btn-grid"><button class="btn small" data-action="full-span">Full span</button><button class="btn small" data-action="zoom-in">Span ÷ 2</button><button class="btn small" data-action="zoom-out">Span × 2</button></div>`
    + (off ? toggleField('Label axis with RF frequency', 'specUi.rfAxis') : ''),
  {id: 'spec-frequency', help: 'Spans up to a quarter of the sample rate are digitally down-converted (zoom FFT): the analyzer mixes the centre to DC, decimates with ≥110 dB alias rejection and resolves finer RBW than a full-band FFT of the same size. Wheel over the trace to zoom, right-drag to pan, double-click for full span.'});
}

function amplitudeSection() {
  return section('Amplitude', toggleField('Auto reference level', 'specUi.autoRef')
    + numberField('Reference', 'specUi.ref', {unit: powerUnit(live.metrics), step: 5, disabled: state.specUi.autoRef})
    + selectField('Scale', 'specUi.perDiv', [[1, '1 dB/div'], [2, '2 dB/div'], [5, '5 dB/div'], [10, '10 dB/div'], [15, '15 dB/div'], [20, '20 dB/div']]),
  {id: 'spec-amplitude', help: `Readings are ${esc(powerUnit(live.metrics))} per resolution bandwidth: a bin-centred tone reads its power. Transmitter captures read absolute simulated dBm; recordings read dBFS unless a calibration offset is set.`});
}

function bandwidthSection() {
  const ui = state.specUi, m = live.metrics;
  return section('Bandwidth / sweep', toggleField('Auto RBW (span / 500)', 'specUi.autoRbw')
    + (ui.autoRbw ? '' : numberField('RBW', 'spectrum.rbw_hz', {unit: 'Hz', min: 0.1}))
    + selectField('Window', 'spectrum.window', [['hann', 'Hann'], ['blackmanharris', 'Blackman-Harris'], ['flattop', 'Flat top (amplitude)'], ['kaiser', 'Kaiser β 8.6'], ['rect', 'Rectangular']])
    + selectField('Detector', 'spectrum.detector', [['average', 'Average (power)'], ['peak', 'Peak'], ['sample', 'Sample']])
    + numberField('FFT averages', 'spectrum.averaging', {min: 1, max: 128, integer: true})
    + selectField('FFT size limit', 'spectrum.max_fft', [[32768, '32 k'], [65536, '64 k'], [131072, '128 k'], [262144, '256 k'], [524288, '512 k'], [1048576, '1 M']])
    + (m.rbw_limited_by_capture ? `<div class="callout warn">RBW is limited by the capture length: ${esc(fmt(m.rbw_actual_hz, 'Hz'))} achieved. Lengthen the capture or narrow the span.</div>` : ''),
  {id: 'spec-bandwidth', open: false, help: 'The FFT length follows the requested RBW and the window’s equivalent noise bandwidth, but never extends past the captured samples: when a capture is too short, the achieved RBW is reported instead.'});
}

function tracesSection() {
  return section('Traces', state.specUi.traces.map((t, i) => selectField(`Trace ${i + 1}`, `specUi.traces.${i}.mode`, TRACE_MODES)).join('')
    + numberField('Average count', 'specUi.traceAverage', {min: 1, max: 256, integer: true})
    + toggleField('Hold display (stop updating)', 'specUi.hold')
    + toggleField('Waterfall', 'specUi.waterfall')
    + '<div class="btn-row"><button class="btn small" data-action="reset-traces">Reset traces</button><button class="btn small" data-action="capture-memory">Store trace in memory</button></div>'
    + (state.specUi.memories.length ? `<div class="list">${state.specUi.memories.map((m, i) => `<div class="list-item"><i style="width:9px;height:9px;border-radius:2px;background:${m.color}"></i><span class="grow">${esc(m.label)}</span><button class="btn small ghost" data-action="remove-memory" data-index="${i}" aria-label="Remove memory">×</button></div>`).join('')}</div>` : ''),
  {id: 'spec-traces', open: false});
}

function markersSection() {
  const ui = state.specUi, m = ui.markers[ui.activeMarker];
  const off = axisOffset();
  return section('Markers', `<div class="btn-grid">`
      + '<button class="btn small" data-action="marker-add" title="M">+ Marker</button><button class="btn small" data-action="marker-peak" title="P">Peak</button><button class="btn small" data-action="marker-next" title="N">Next peak</button>'
      + '<button class="btn small" data-action="marker-left" title="[">◀ Peak</button><button class="btn small" data-action="marker-right" title="]">Peak ▶</button><button class="btn small" data-action="marker-center" title="C">Mkr → CF</button>'
      + `<button class="btn small" data-action="marker-ref">Mkr → Ref</button><button class="btn small${m?.delta ? ' on' : ''}" data-action="marker-delta">Delta</button><button class="btn small${m?.noise ? ' on' : ''}" data-action="marker-noise">Noise /Hz</button></div>`
    + (m ? numberField(`M${ui.activeMarker + 1} frequency`, `specUi.markers.${ui.activeMarker}.hz`, {unit: 'Hz', offset: off})
      + (m.delta ? selectField('Delta reference', `specUi.markers.${ui.activeMarker}.ref`, ui.markers.map((_, i) => [i, `M${i + 1}`]).filter(([i]) => i !== ui.activeMarker)) : '')
      + selectField('On trace', `specUi.markers.${ui.activeMarker}.trace`, ui.traces.map((_, i) => [i, `Trace ${i + 1}`]))
      + '<div class="btn-row"><button class="btn small danger" data-action="marker-remove">Remove marker</button><button class="btn small ghost" data-action="marker-clear">Clear all</button></div>'
      : ''),
  {id: 'spec-markers', help: 'Keyboard (click the trace first): P peak · N next lower peak · [ / ] nearest peak left/right · ← → step · M new marker · C marker to centre · Del remove · F full span · Z zoom to selection · Esc clear selection. Noise markers read power density using the actual RBW (average detector).'});
}

function channelsSection() {
  const off = axisOffset();
  const rows = state.spectrum.channels.map((b, i) => `<div class="card" style="margin:6px 0"><header><i style="width:9px;height:9px;border-radius:2px;background:${CHANNEL_COLORS[i % 8]}"></i>${esc(b.label || `CH${i + 1}`)}<span class="tools"><button class="btn small ghost" data-action="channel-remove" data-index="${i}" aria-label="Remove channel">×</button></span></header><div class="card-body">`
    + numberField('Centre', `spectrum.channels.${i}.center_hz`, {unit: 'Hz', offset: off})
    + numberField('Width', `spectrum.channels.${i}.width_hz`, {unit: 'Hz', min: 1})
    + `</div></div>`).join('');
  return section('Channel power', `${rows}<div class="btn-row"><button class="btn small" data-action="channel-add">+ Channel</button>${state.specUi.selection ? '<button class="btn small" data-action="channel-from-selection">From selection</button>' : ''}</div>`
    ,
  {id: 'spec-channels', help: 'Integrated power over each band from the full-resolution spectrum (not the reduced display), with bins at the band edges weighted by their overlap. Density divides by the band width; OBW is the 99 % power bandwidth inside the band. Use the average detector for noise-like signals.'});
}

function zeroSections() {
  const m = state.measurement, off = axisOffset();
  return section('Zero-span tuning', toggleField('Tune and band-limit before detection', 'measurement.tuned')
      + (m.tuned ? numberField('Centre', 'measurement.center_hz', {unit: 'Hz', offset: off}) + numberField('Bandwidth', 'measurement.bandwidth_hz', {unit: 'Hz', min: 1, value: m.bandwidth_hz ?? Math.min(activeRate() / 4, 100e3)})
        + '<button class="btn small" data-action="zero-from-marker">Centre = active marker</button>' : ''),
    {id: 'zero-tune', help: 'Like an analyzer in zero span: the record is mixed to the centre frequency and limited to the bandwidth (a linear-phase FIR) before the AM, PM and FM detectors. Off: the full-band samples are detected directly.'})
    + section('Interval', numberField('Offset', 'measurement.offset_s', {unit: 's', min: 0})
      + numberField('Length', 'measurement.duration_s', {unit: 's', min: 1e-7, max: 1})
      + `<div class="btn-row"><button class="btn small${state.specUi.zeroHold ? ' on' : ''}" data-action="zero-hold">${state.specUi.zeroHold ? 'Resume live' : 'Hold capture'}</button></div>`, {id: 'zero-interval'});
}

function dockHTML() {
  const page = state.specUi.page;
  let body = section('Source', sourceSection(), {id: 'spec-source'});
  if (page === 'sweep') body += frequencySection() + amplitudeSection() + markersSection() + channelsSection() + bandwidthSection() + tracesSection();
  else if (page === 'zero') body += zeroSections();

  return body;
}

/* ---------------------------------------------------------------- main -- */

function mainHTML() {
  const page = state.specUi.page;
  if (page === 'sweep') {
    return views(PAGES, page, 'page')
      + `<div class="readout-bar">${readout('Center', 'sCenter')}${readout('Span', 'sSpan')}${readout('RBW', 'sRbw')}${readout('Peak', 'sPeakF', '—', 'accent')}${readout('Peak level', 'sPeakL')}${readout('99 % BW', 'sBw99')}${readout('Analysis', 'sZoom')}${readout('Engine', 'sCompute')}</div>`
      + analyzerMarkup({waterfall: state.specUi.waterfall, tools: '<button class="btn small ghost" data-action="marker-peak" title="Peak search (P)">Peak</button><button class="btn small ghost" data-action="marker-add">+ Mkr</button><button class="btn small ghost" data-action="full-span" title="Full span (F)">Full</button>'});
  }
  if (page === 'zero') {
    return views(PAGES, page, 'page') + `<div class="readout-bar">${readout('Capture', 'zCapture')}${readout('Interval', 'zInterval')}${readout('RMS', 'zRms')}${readout('Peak', 'zPeak')}${readout('Tuning', 'zTune', '—', 'grow')}</div>`
      + `<div class="stage quad">${pane('I / Q', 'zeroIQ')}${pane('AM · envelope', 'zeroAM')}${pane('PM · unwrapped phase', 'zeroPM')}${pane('FM · instantaneous frequency', 'zeroFM')}</div>`;
  }
  return views(PAGES, page, 'page') + `<div class="readout-bar">${readout('EVM', 'qEvm', '—', 'accent')}${readout('EVM peak', 'qPeak')}${readout('MER', 'qMer')}${readout('Record', 'qAll')}${readout('Capture', 'qCapture')}${readout('Method', 'qMethod', '—', 'grow')}</div>`
    + `<div class="stage quad">${pane('Constellation', 'qualityConst')}${pane('EVM vs symbol', 'qualityEvm')}${pane('Magnitude error', 'qualityMag')}${pane('Phase error', 'qualityPhase')}</div>`;
}

export function render(el) {
  root = el;
  stopHost();
  el.innerHTML = `<div class="instrument"><aside class="dock"><div class="dock-head"><h2>Spectrum analyzer</h2></div><div class="dock-scroll" data-keep-scroll="spectrum-dock"></div></aside><section class="main" id="specMain"></section></div>`;
  bindControls(el, {onChange, actions});
  renderDock();
  renderMain();
}

function renderDock() {
  replaceHTML($('.dock-scroll', root), dockHTML());
}

function renderMain() {
  stopHost();
  const main = $('#specMain', root);
  main.innerHTML = mainHTML();
  if (state.specUi.page === 'sweep') {
    host = new AnalyzerHost($('.analyzer', main), {
      tune: state.spectrum, ui: state.specUi, waterfall: state.specUi.waterfall,
      channels: onChannelDrag,
      selectionActions: [
        {id: 'zoom', label: 'Zoom', primary: true, title: 'Tune the analyzer to this band (Z)'},
        {id: 'channel', label: 'Channel power', title: 'Measure integrated power in this band'},
        {id: 'classify', label: 'Classify', title: 'Classify the emitter in this band'},
        {id: 'receive', label: 'Receive', title: 'Tune the receiver to this band'},
      ],
      onSelectionAction: selectionAction,
      onTuned: () => { replaceSection('spec-frequency', frequencySection()); updateReadouts(); },
    });
  }
  updateReadouts();
  drawMeasurement();
}

function replaceSection(id, html) {
  const old = $(`[data-fold="${id}"]`, root);
  if (!old) return;
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  old.replaceWith(tmp.firstElementChild);
}

/* ------------------------------------------------------------- actions -- */

function retune(change) {
  host?.retune(change);
  if (!host) liveUpdate();
  replaceSection('spec-frequency', frequencySection());
}

function selectionAction(id, sel) {
  const center = (sel.lo + sel.hi) / 2, width = sel.hi - sel.lo;
  if (id === 'zoom') { retune({center, span: width}); host?.analyzer.setSelection(null); }
  else if (id === 'channel') { addChannel(center, width); host?.analyzer.setSelection(null); }
  else if (id === 'classify') {
    Object.assign(state.blindUi.band, {enabled: true, center_hz: center, width_hz: width});
    publish('navigate', 'blind');
  } else if (id === 'receive') {
    state.rx.tuner_hz = center;
    state.rxUi.follow = false;
    state.rxUi.page = 'acquire';
    toast(`Receiver tuned to ${fmtFreq(center + axisOffset(), width / 100)}`);
    publish('navigate', 'rx');
  }
}

function addChannel(center = Number(state.spectrum.center_hz) || 0, width = span() / 10) {
  if (state.spectrum.channels.length >= 8) { toast('Up to eight channels'); return; }
  state.spectrum.channels.push({label: `CH${state.spectrum.channels.length + 1}`, center_hz: center, width_hz: width});
  replaceSection('spec-channels', channelsSection());
  host?.analyzer.renderStrip();
  liveUpdate();
}

function onChannelDrag(index, change, done) {
  if (change) Object.assign(state.spectrum.channels[index], change);
  if (done) { replaceSection('spec-channels', channelsSection()); liveUpdate(); }
}

const actions = {
  ...sourceActions(() => render(root)),
  'full-span': () => retune('full'),
  'zoom-in': () => retune({span: span() / 2}),
  'zoom-out': () => retune({span: span() * 2}),
  'reset-traces': () => host?.analyzer.resetTraces(),
  'capture-memory': () => {
    const t = host?.analyzer.traceData(0);
    if (!t) { toast('Waiting for a spectrum frame'); return; }
    if (state.specUi.memories.length >= 4) state.specUi.memories.shift();
    const n = state.specUi.memories.length;
    state.specUi.memories.push({x: Array.from(live.freq), y: Array.from(t), color: MEMORY_COLORS[n % 4], label: `Memory ${n + 1} · ${new Date().toLocaleTimeString()}`});
    replaceSection('spec-traces', tracesSection());
  },
  'remove-memory': el => { state.specUi.memories.splice(Number(el.dataset.index), 1); replaceSection('spec-traces', tracesSection()); },
  'marker-add': () => host?.analyzer.addMarker(state.spectrum.center_hz || 0),
  'marker-peak': () => host?.analyzer.peakSearch(),
  'marker-next': () => host?.analyzer.nextPeak(0),
  'marker-left': () => host?.analyzer.nextPeak(-1),
  'marker-right': () => host?.analyzer.nextPeak(1),
  'marker-center': () => { const m = host?.analyzer.activeMarker(); if (m) retune({center: m.hz}); },
  'marker-ref': () => {
    const m = host?.analyzer.activeMarker(), at = m && host.analyzer.levelAt(m.hz, m.trace);
    if (at) { state.specUi.autoRef = false; state.specUi.ref = Math.ceil(at.db); replaceSection('spec-amplitude', amplitudeSection()); host.analyzer.render(); }
  },
  'marker-delta': () => toggleMarker('delta'),
  'marker-noise': () => toggleMarker('noise'),
  'marker-remove': () => host?.analyzer.removeMarker(),
  'marker-clear': () => { state.specUi.markers = []; state.specUi.activeMarker = -1; host?.analyzer.changed(); },
  'channel-add': () => addChannel(),
  'channel-from-selection': () => { const s = state.specUi.selection; if (s) addChannel((s.lo + s.hi) / 2, s.hi - s.lo); },
  'channel-remove': el => {
    state.spectrum.channels.splice(Number(el.dataset.index), 1);
    replaceSection('spec-channels', channelsSection());
    host?.analyzer.renderStrip();
    liveUpdate();
  },
  'zero-hold': () => { state.specUi.zeroHold = !state.specUi.zeroHold; renderDock(); liveUpdate(true); },
  'zero-from-marker': () => {
    const m = state.specUi.markers[state.specUi.activeMarker];
    if (m) { state.measurement.center_hz = m.hz; renderDock(); liveUpdate(true); } else toast('Place a marker in the Spectrum view first');
  },
};

function toggleMarker(kind) {
  const m = host?.analyzer.activeMarker();
  if (!m) return;
  m[kind] = !m[kind];
  if (kind === 'delta') { m.noise = false; if (m.ref === state.specUi.activeMarker) m.ref = state.specUi.activeMarker === 0 ? 1 : 0; }
  if (kind === 'noise') m.delta = false;
  replaceSection('spec-markers', markersSection());
  host.analyzer.changed();
}

/* -------------------------------------------------------------- changes -- */

function onChange(path) {
  if (path === 'spectrum.span_hz' || path === 'spectrum.center_hz') { retune({}); return; }
  if (path === 'specView.start' || path === 'specView.stop') {
    const {start, stop} = state.specView;
    if (stop > start) retune({center: (start + stop) / 2, span: stop - start});
    else { toast('Stop must be above start'); replaceSection('spec-frequency', frequencySection()); }
    return;
  }
  if (path.startsWith('spectrum.channels')) { host?.analyzer.render(); host?.analyzer.renderStrip(); liveUpdate(); return; }
  if (path.startsWith('spectrum.')) { host?.analyzer.resetTraces(); liveUpdate(); if (path === 'spectrum.detector') renderDock(); return; }
  if (path === 'specUi.autoRbw') { liveUpdate(); replaceSection('spec-bandwidth', bandwidthSection()); return; }
  if (path === 'specUi.autoRef') { replaceSection('spec-amplitude', amplitudeSection()); host?.analyzer.render(); return; }
  if (path === 'specUi.waterfall') { renderMain(); return; }
  if (path === 'specUi.rfAxis') { renderDock(); host?.analyzer.render(); return; }
  if (path.startsWith('specUi.traces')) { host?.analyzer.resetTraces(); host?.analyzer.render(); return; }
  if (path.startsWith('specUi.markers')) { host?.analyzer.changed(); if (path.endsWith('.ref') || path.endsWith('.trace')) replaceSection('spec-markers', markersSection()); return; }
  if (path.startsWith('specUi.')) { host?.analyzer.render(); return; }
  if (path.startsWith('measurement.')) { if (path === 'measurement.tuned') renderDock(); liveUpdate(true); }
}

/* ------------------------------------------------------------ readouts -- */

function updateReadouts() {
  if (state.specUi.page !== 'sweep') return;
  const m = live.metrics, off = axisOffset(), unit = powerUnit(m);
  const c = Number(state.spectrum.center_hz) || 0;
  setText('sCenter', fmtFreq(c + off, span() / 1000));
  setText('sSpan', fmt(span(), 'Hz'));
  setText('sRbw', m.rbw_actual_hz ? `${fmt(m.rbw_actual_hz, 'Hz')}${m.rbw_limited_by_capture ? ' (capture-limited)' : ''}` : '—');
  setText('sPeakF', m.peak_hz == null ? '—' : fmtFreq(m.peak_hz + off, (m.rbw_actual_hz || 1) / 4));
  setText('sPeakL', m.peak_db == null ? '—' : `${m.peak_db.toFixed(2)} ${unit}`);
  setText('sBw99', fmt(m.bw99_hz, 'Hz'));
  setText('sZoom', m.decimation > 1 ? `DDC ÷${m.decimation} · ${fmt(m.analysis_rate, 'S/s')}` : `full band · ${fmt(m.nfft, '')} pt`);
  setText('sCompute', m.compute_ms == null ? '—' : `${m.compute_ms.toFixed(1)} ms`);
}

/* ---------------------------------------------------- zero span / quality -- */

/* Zero span and modulation quality are live products: the engine measures
   each capture it shows (at the display rate, dropping what it cannot
   process in real time) and sends the result with the frame. */
function measurementRequest() {
  const m = state.measurement;
  return {offset_s: m.offset_s, duration_s: m.duration_s, quality: state.specUi.page === 'quality',
    ...(m.tuned ? {center_hz: m.center_hz, bandwidth_hz: m.bandwidth_hz ?? Math.min(activeRate() / 4, 100e3)} : {})};
}

function measuring() {
  return state.specUi.page !== 'sweep' && !(state.specUi.zeroHold && state.specUi.page === 'zero');
}

function drawMeasurement() {
  const d = measurement;
  if (!d || !root) return;
  if (state.specUi.page === 'zero' && d.interval_clipped) {
    const changed = state.measurement.offset_s !== d.capture_start_s || state.measurement.duration_s !== d.capture_duration_s;
    state.measurement.offset_s = d.capture_start_s;
    state.measurement.duration_s = d.capture_duration_s;
    if (changed) { renderDock(); toast('Zero-span interval shortened to the captured window'); }
  }
  setText(state.specUi.page === 'zero' ? 'zCapture' : 'qCapture', `#${d.capture_id}`);
  if (state.specUi.page === 'zero') {
    const x = {x: d.time_s, xUnit: 's'};
    linePlot($('#zeroIQ'), [{y: d.i}, {y: d.q}], x);
    linePlot($('#zeroAM'), d.amplitude, {...x, min: 0});
    linePlot($('#zeroPM'), d.phase_deg.map((v, i) => (d.valid[i] ? v : NaN)), {...x, yUnit: '°'});
    linePlot($('#zeroFM'), d.frequency_hz.map((v, i) => ((d.frequency_valid || d.valid)[i] ? v : NaN)), {...x, yUnit: 'Hz'});
    setText('zInterval', `${fmt(d.capture_start_s, 's')} + ${fmt(d.capture_duration_s, 's')}`);
    setText('zRms', fmt(d.rms));
    setText('zPeak', fmt(d.peak));
    setText('zTune', d.bandwidth_hz
      ? `${fmtFreq(d.center_hz + axisOffset(), 1)} · ${fmt(d.bandwidth_actual_hz ?? d.bandwidth_hz, 'Hz')} RBW${d.bandwidth_actual_hz > d.bandwidth_hz * 1.01 ? ' (capture limited)' : ''}`
      : 'full band');
    return;
  }
  const q = d.quality;
  if (!q) return;
  if (q.kind === 'basic') {
    // No symbols to measure: the same four panes show the waveform itself.
    paneTitles(['I/Q trajectory', 'Envelope', 'Instantaneous frequency', 'Unwrapped phase']);
    const t = q.telemetry, x = t.time_s ? {x: t.time_s, xUnit: 's'} : {};
    scatter($('#qualityConst'), q.symbols?.i, q.symbols?.q);
    linePlot($('#qualityEvm'), t.amplitude || [], {...x, min: 0});
    linePlot($('#qualityMag'), (t.frequency_hz || []).map((v, i) => t.frequency_valid?.[i] === false ? NaN : v), {...x, yUnit: 'Hz'});
    linePlot($('#qualityPhase'), (t.phase_deg || []).map((v, i) => t.phase_valid?.[i] === false ? NaN : v), {...x, yUnit: '°'});
    setText('qEvm', '—'); setText('qPeak', fmt(q.metrics.peak)); setText('qMer', '—'); setText('qAll', '—');
    setText('qualityEvmValue', '');
  } else {
    paneTitles(['Constellation', 'EVM vs symbol', 'Magnitude error', 'Phase error']);
    const m = q.metrics, settle = q.telemetry.evm_settle_index || 0;
    const shade = settle ? [{x0: 0, x1: settle, color: 'rgba(236,123,114,.10)'}] : [];
    scatter($('#qualityConst'), q.symbols?.i, q.symbols?.q, {ideals: idealConstellation(appliedTx().waveform, appliedTx())});
    linePlot($('#qualityEvm'), q.telemetry.evm_time_percent || [], {min: 0, yUnit: '%', shade});
    linePlot($('#qualityMag'), q.telemetry.magnitude_error_percent || [], {yUnit: '%', shade});
    linePlot($('#qualityPhase'), q.telemetry.phase_error_deg || [], {yUnit: '°', shade});
    setText('qEvm', fmt(m.evm_rms_percent, '%'));
    setText('qPeak', fmt(m.evm_peak_percent, '%'));
    setText('qMer', fmt(m.mer_db, 'dB'));
    setText('qAll', m.captures != null ? `${fmtCount(m.symbol_count)} symbols · ${fmtCount(m.captures)} captures` : `${fmtCount(m.symbol_count)} symbols`);
    setText('qualityEvmValue', settle ? 'shaded: loop acquisition (excluded)' : 'most recent symbols on the right');
  }
  setText('qMethod', q.method);
}

/** Titles of the four quality panes (their content depends on the signal). */
function paneTitles(titles) {
  ['qualityConst', 'qualityEvm', 'qualityMag', 'qualityPhase'].forEach((id, i) => {
    const h = document.getElementById(id)?.closest('.pane')?.querySelector('.pane-head h3');
    if (h && h.textContent !== titles[i]) h.textContent = titles[i];
  });
}

function measurementError(error) {
  if (!error || !measuring() || !['measurement', 'product'].includes(error.scope)) return;
  measurement = null;
  if (state.specUi.page === 'zero') {
    for (const id of ['zeroIQ', 'zeroAM', 'zeroPM', 'zeroFM']) linePlot($(`#${id}`), []);
    setText('zCapture', 'Unavailable');
    for (const id of ['zInterval', 'zRms', 'zPeak', 'zTune']) setText(id, '—');
  } else {
    scatter($('#qualityConst'), [], []);
    for (const id of ['qualityEvm', 'qualityMag', 'qualityPhase']) linePlot($(`#${id}`), []);
    setText('qCapture', 'Unavailable');
    for (const id of ['qEvm', 'qPeak', 'qMer', 'qAll']) setText(id, '—');
    setText('qMethod', error.message);
  }
}

/* ----------------------------------------------------------- lifecycle -- */

export function start() {
  // stop() releases the analyzer (also when the page is hidden): rebuild it.
  if (!host && state.specUi.page === 'sweep' && root) renderMain();
  unsubscribe = [
    subscribe('live:metrics', () => updateReadouts()),
    // A product requested for the other page (zero span vs quality) may still arrive once.
    subscribe('live:measurement', m => { if (measuring() && (m.quality != null) === (state.specUi.page === 'quality')) { measurement = m; drawMeasurement(); } }),
    subscribe('live:error', measurementError),
    subscribe('live:meta', () => replaceSection('spec-source', section('Source', sourceSection(), {id: 'spec-source'}))),
  ];
}

function stopHost() {
  host?.destroy();
  host = null;
}

export function stop() {
  for (const off of unsubscribe) off();
  unsubscribe = [];
  stopHost();
}

export function resize() {
  host?.analyzer.render();
  drawMeasurement();
}

// View tabs live in the main area.
document.addEventListener('click', event => {
  const tab = event.target.closest('#specMain [data-page]');
  if (!tab || state.mode !== 'spectrum') return;
  state.specUi.page = tab.dataset.page;
  measurement = null;  // zero span and quality products differ: wait for the new page's
  stop();
  render(root);
  start();
  liveUpdate();
});
