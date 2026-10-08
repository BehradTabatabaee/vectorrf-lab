/* Transmitter: single signals and composites, frequency agility and export. */
import {$, esc, replaceHTML, setText} from '../core/dom.js';
import {post, errorText} from '../core/api.js';
import {bindControls, numberField, pane, readout, section, segmented, sliderField, textField, toggleField, views} from '../core/controls.js';
import {publish, subscribe} from '../core/events.js';
import {fmt, fmtCount} from '../core/format.js';
import {update as liveUpdate} from '../core/live.js';
import {clone, getPath, state} from '../core/state.js';
import {applyTx, designRequest, maxCaptureSeconds, txRequest} from '../core/source.js';
import {renderStatus, toast} from '../core/ui.js';
import {agilityEditor} from '../editors/agility.js';
import {exportFields, runExport} from '../editors/export.js';
import {frameActions} from '../editors/frame.js';
import {bindOfdmGrid, ofdmActions, refreshOfdmGrid} from '../editors/ofdm.js';
import {programActions} from '../editors/program.js';
import {structuredActions} from '../editors/structured.js';
import {burstSection, dataSections, familyOf, inBand, modulationSections, shapingSections, waveformForFamily, waveformLabel, waveformPicker} from '../editors/waveform.js';
import {analyzerMarkup} from '../plots/analyzer.js';
import {COMPONENT_COLORS, drawBandPlan, drawTimePlan} from '../plots/plan.js';
import {AnalyzerHost, spectrumRequest} from './common.js';

let root = null;
let host = null;
let planTimer = null;
let unsubscribe = [];
const monitor = {tune: {center_hz: 0, span_hz: null}, ui: null};

export const usesLive = true;

export function payload() {
  return {spectrum: spectrumRequest(monitor.tune)};
}

const tx = () => state.tx;
const composite = () => tx().waveform === 'composite';
const rf = () => Number(tx().rf_center_hz) || 0;

/** Path of the config currently being edited (the signal, or the selected component). */
function editPath() {
  if (!composite()) return 'tx';
  const i = Math.min(state.txUi.component, tx().components.length - 1);
  state.txUi.component = Math.max(0, i);
  return `tx.components.${state.txUi.component}`;
}

/** Absolute frequency of a config's 0 Hz offset (RF centre + parent offset). */
function carrierOf(path) {
  return path === 'tx' ? rf() : rf() + Number(tx().center_hz || 0);
}

/* ------------------------------------------------------------------ dock -- */

function dockHTML() {
  const t = tx(), plan = state.txUi.plan;
  const samples = Math.round(Math.min(t.duration, maxCaptureSeconds()) * t.sample_rate);
  return section('Clock and capture', numberField('Sample rate', 'tx.sample_rate', {unit: 'S/s', min: 2000, max: state.limits.max_sample_rate})
      + numberField('Capture length', 'tx.duration', {unit: 's', min: 1e-6, max: maxCaptureSeconds(), help: `${fmtCount(samples)} samples per capture; at most ${fmtCount(state.limits.max_capture_samples)} at any rate.`})
      + numberField('RF centre', 'tx.rf_center_hz', {unit: 'Hz', min: 0, max: 1e12, help: 'The RF frequency complex baseband 0 Hz represents. Instruments then label RF frequencies, and frequency fields accept absolute values (for example 900M–1G hops around a 950 MHz centre). 0 = plain baseband.'}),
    {id: 'tx-clock'})
    + section('Output', (composite()
      ? `<div class="field"><span class="label">Total power</span><b style="font:500 13px var(--mono);color:var(--accent-2)">${plan ? esc(fmt(plan.total_power_dbm, 'dBm')) : '—'}</b></div>`
      : numberField('Power', 'tx.power_dbm', {unit: 'dBm', help: 'Absolute simulator calibration: a bin-centred CW marker reads this value; modulated signals spread it over their bandwidth.'}))
      + numberField(composite() ? 'Composite offset' : 'Frequency offset', 'tx.center_hz', {unit: 'Hz', within: inBand(0), help: 'Baseband offset of the whole signal from the RF centre.'})
      + numberField('Phase', 'tx.phase_offset_deg', {unit: '°'}), {id: 'tx-output'})
    + section('Impairments', toggleField('Transmitter AWGN', 'txUi.noise')
      + (state.txUi.noise && t.waveform !== 'awgn' ? sliderField('SNR', 'tx.snr_db', {min: -10, max: 60, step: 1, unit: 'dB'}) : '')
      + sliderField('I/Q gain imbalance', 'tx.iq_gain_db', {min: -6, max: 6, step: 0.1, unit: 'dB'})
      + sliderField('Quadrature error', 'tx.iq_phase_deg', {min: -30, max: 30, step: 0.5, unit: '°'})
      + numberField('DC I', 'tx.dc_i', {min: -1, max: 1}) + numberField('DC Q', 'tx.dc_q', {min: -1, max: 1}),
    {id: 'tx-impair', open: false, help: 'Applied to the finished signal (after composition and hopping). SNR refers to the clean on-power.'})
    + section('Export I/Q file', `${exportFields(false)}<div class="btn-row"><button class="btn primary" data-action="export">Export capture</button></div>`,
      {id: 'tx-export', open: false, help: 'Writes the capture the instruments are showing (same seed and start sample), in formats SDR tools read. SigMF keeps rate, RF centre, calibration and the complete design.'});
}

/* ---------------------------------------------------------------- editor -- */

function componentList() {
  const plan = state.txUi.planError ? null : state.txUi.plan;
  const problems = state.txUi.planError ? state.txUi.componentProblems || [] : [];
  return `<div class="component-list">${tx().components.map((c, i) => {
    const p = plan?.components?.[i];
    const bw = p?.band_hz ? fmt(p.band_hz[1] - p.band_hz[0], 'Hz') : '—';
    const problem = c.enabled && problems[i];
    return `<div class="component${i === state.txUi.component ? ' active' : ''}${c.enabled ? '' : ' off'}${problem ? ' invalid' : ''}" data-action="select-component" data-index="${i}">`
      + `<span class="swatch" style="background:${COMPONENT_COLORS[i % 8]}"></span><div style="min-width:0"><div class="c-name">${esc(c.label || `Signal ${i + 1}`)}</div>`
      + `<div class="c-meta">${esc(waveformLabel(c.waveform))} · ${esc(fmt(c.power_dbm, 'dBm'))} · ${esc(fmt(rf() + Number(tx().center_hz) + c.center_hz, 'Hz'))} · BW ${esc(bw)}${c.agility.enabled ? ` · ${c.agility.mode}` : ''}${c.burst_enabled ? ' · burst' : ''}</div>`
      + `${problem ? `<div class="c-problem" title="${esc(problem)}">${esc(problem)}</div>` : ''}</div>`
      + `<div class="c-tools"><button type="button" class="btn small ghost" data-action="toggle-component" data-index="${i}" title="${c.enabled ? 'Mute' : 'Enable'}">${c.enabled ? 'On' : 'Off'}</button>`
      + `<button type="button" class="btn small ghost" data-action="duplicate-component" data-index="${i}" title="Duplicate">⧉</button>`
      + `<button type="button" class="btn small ghost" data-action="remove-component" data-index="${i}" aria-label="Remove component">×</button></div></div>`;
  }).join('')}</div><div class="btn-row"><button class="btn small" data-action="add-component">+ Add signal</button></div>`;
}

function componentHeader(path) {
  return section('Component', textField('Name', `${path}.label`)
    + numberField('Power', `${path}.power_dbm`, {unit: 'dBm', help: 'Absolute on-power of this component; the composite total is the sum.'})
    + numberField('Frequency', `${path}.center_hz`, {unit: 'Hz', offset: carrierOf(path), within: inBand(Number(tx().center_hz) || 0), help: 'Where this component’s carrier sits (absolute when an RF centre is set).'})
    + numberField('Phase', `${path}.phase_offset_deg`, {unit: '°'}), {id: `comp-head-${path}`});
}

function signalTabs(path) {
  const page = state.txUi.page;
  const cfg = getPath(state, path);
  const tabs = [['signal', 'Signal'], ['data', 'Data / framing'], ...(cfg.waveform === 'ofdm' ? [['shaping', 'OFDM shaping']] : []),
    ['agility', `Hopping / sweep${cfg.agility.enabled ? ' ●' : ''}`], ['burst', `Burst${cfg.burst_enabled ? ' ●' : ''}`]];
  const current = tabs.some(([k]) => k === page) ? page : 'signal';
  let body = '';
  if (current === 'signal') body = section('Waveform', waveformPicker(path), {id: `pick-${path}`}) + modulationSections(path, carrierOf(path) + (path === 'tx' ? 0 : cfg.center_hz));
  else if (current === 'data') body = dataSections(path);
  else if (current === 'shaping') body = shapingSections(path);
  else if (current === 'agility') body = agilityEditor(path, carrierOf(path) + cfg.center_hz, agilitySummary(path));
  else body = burstSection(path);
  return views(tabs, current, 'txpage') + body;
}

function agilitySummary(path) {
  const plan = state.txUi.plan;
  if (!plan) return null;
  if (path === 'tx') return plan.agility;
  return plan.components?.[Number(path.split('.').at(-1))]?.agility || null;
}

function editorHTML() {
  const t = tx();
  const count = composite() ? `${t.components.filter(c => c.enabled).length} of ${t.components.length} signals on` : '';
  const head = `<div class="editor-head">${segmented('txUi.family', [['single', 'Single signal'], ['composite', 'Composite']], {value: composite() ? 'composite' : 'single', label: 'Signal mode'})}`
    + `<span class="editor-count">${esc(count)}</span></div>`;
  let body;
  if (composite()) {
    const path = editPath();
    const sub = state.txUi.compositePage || 'components';
    body = views([['components', 'Components'], ['agility', `Composite hopping / sweep${t.agility.enabled ? ' ●' : ''}`], ['burst', `Composite burst${t.burst_enabled ? ' ●' : ''}`]], sub, 'txcomp')
      + (sub === 'components'
        ? `<div class="editor-body">${componentList()}${t.components.length ? componentHeader(path) + signalTabs(path) : ''}</div>`
        : `<div class="editor-body">${sub === 'agility' ? agilityEditor('tx', rf() + Number(t.center_hz), state.txUi.plan?.agility) : burstSection('tx')}</div>`);
  } else {
    body = `<div class="editor-body">${signalTabs('tx')}</div>`;
  }
  const warnings = [state.txUi.planError && `<div class="callout bad" role="alert"><b>Not applied.</b> ${esc(state.txUi.planError)} The instruments keep the last valid design.</div>`,
    ...(state.txUi.plan?.warnings || []).map(w => `<div class="callout warn">${esc(w)}</div>`)].filter(Boolean).join('');
  return head + (warnings ? `<div class="editor-body" style="padding-bottom:0">${warnings}</div>` : '') + body;
}

function mainHTML() {
  return `<div class="readout-bar">${readout('Signal', 'txSignal')}${readout('Power', 'txPower', '—', 'accent')}${readout('Occupied', 'txBand')}${readout('Carrier span', 'txExtent')}${readout('Agility', 'txAgility')}${readout('Capture', 'txCapture', '—', 'grow')}</div>`
    + `<div class="tx-layout"><div class="tx-editor" id="txEditor" data-keep-scroll="tx-editor"></div><div class="tx-monitor">`
    + `<section class="pane"><header class="pane-head"><h3>Live output · full span</h3><span class="pane-value" id="txMonitorValue"></span></header><div class="pane-body" style="display:flex">${analyzerMarkup({waterfall: true, compact: true})}</div></section>`
    + pane('Band plan · nominal occupied bands', 'txBandPlan')
    + pane('Hopping / sweep schedule', 'txTimePlan')
    + '</div></div>';
}

/* ---------------------------------------------------------------- render -- */

export function render(el) {
  root = el;
  monitor.ui ||= {...clone(state.specUi), traces: [{mode: 'clear', y: null}, {mode: 'off', y: null}, {mode: 'off', y: null}, {mode: 'off', y: null}], markers: [], activeMarker: -1, memories: [], selection: null, autoRef: true, perDiv: 10};
  el.innerHTML = `<div class="instrument"><aside class="dock"><div class="dock-head"><h2>Transmitter</h2></div><div class="dock-scroll" data-keep-scroll="tx-dock"></div></aside><section class="main" id="txMain">${mainHTML()}</section></div>`;
  bindControls(el, {onChange, actions});
  bindOfdmGrid(el, () => changed('tx.ofdm_design.cells'));
  renderDock();
  renderEditor();
  mountMonitor();
  requestPlan(true);
}

function mountMonitor() {
  host?.destroy();
  host = new AnalyzerHost($('#txMain .analyzer', root), {tune: monitor.tune, ui: monitor.ui, waterfall: true, bands: () => planBands(),
    hint: 'Wheel: zoom · double-click: full span'});
}

function renderDock() {
  replaceHTML($('.dock-scroll', root), dockHTML());
}

function renderEditor() {
  replaceHTML($('#txEditor', root), editorHTML());
  refreshOfdmGrid();
  drawPlans();
  updateReadouts();
}

/* --------------------------------------------------------------- planning -- */

function planBands() {
  const plan = state.txUi.plan;
  if (!plan || state.txUi.planError) return [];
  if (!composite()) return plan.band_hz ? [{lo: plan.band_hz[0], hi: plan.band_hz[1], color: COMPONENT_COLORS[0], fill: 'rgba(242,193,78,.04)'}] : [];
  return plan.components.filter(c => c.enabled && c.band_hz).map(c => ({lo: c.band_hz[0], hi: c.band_hz[1], label: c.label,
    color: COMPONENT_COLORS[c.index % 8], fill: `${COMPONENT_COLORS[c.index % 8]}0c`}));
}

/**
 * Validate the draft on the backend (debounced); a valid draft becomes the
 * applied design and goes on air, a rejected one stays in the editor with its
 * problems marked while the instruments keep the applied design.
 */
function requestPlan(immediate = false) {
  clearTimeout(planTimer);
  planTimer = setTimeout(async () => {
    planTimer = null;
    const revision = state.revision;
    const request = designRequest();
    try {
      const plan = await post('/api/tx/plan', request);
      if (revision !== state.revision) return;
      state.txUi.plan = plan;
      state.txUi.planError = '';
      state.txUi.componentProblems = [];
      applyTx(request);
      liveUpdate(true);
      renderStatus();
    } catch (error) {
      if (revision !== state.revision) return;
      state.txUi.planError = errorText(error);
      state.txUi.componentProblems = error.body?.components || [];
      renderStatus();
    }
    if (state.mode !== 'tx') return;
    const editorScroll = $('#txEditor', root);
    if (editorScroll) renderEditor();
    if (composite()) replaceSectionHTML('tx-output');
  }, immediate ? 0 : 100);
}

function replaceSectionHTML(id) {
  const old = $(`[data-fold="${id}"]`, root);
  if (!old) return;
  const tmp = document.createElement('div');
  tmp.innerHTML = dockHTML();
  const fresh = tmp.querySelector(`[data-fold="${id}"]`);
  if (fresh) old.replaceWith(fresh);
}

/** Replace a plot with a one-line explanation (or remove the explanation). */
function plotNote(canvas, text) {
  const host = canvas?.parentElement;
  if (!host) return;
  let note = host.querySelector('.plot-empty');
  if (!text) { note?.remove(); return; }
  if (!note) { note = Object.assign(document.createElement('div'), {className: 'plot-empty'}); host.append(note); }
  note.textContent = text;
}

function drawPlans() {
  const plan = state.txUi.plan, t = tx();
  const bandCanvas = $('#txBandPlan'), timeCanvas = $('#txTimePlan');
  if (!bandCanvas) return;
  if (!plan || state.txUi.planError) {
    // Never draw the previous design's plan next to an edited, rejected one.
    for (const canvas of [bandCanvas, timeCanvas]) { canvas?.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height); }
    plotNote(bandCanvas, state.txUi.planError ? 'No plan: the design is not valid' : '');
    plotNote(timeCanvas, '');
    setText('txTimePlanValue', '');
    return;
  }
  plotNote(bandCanvas, '');
  // band_hz already spans a signal's own hop/sweep range: an agile signal is drawn
  // as that hatched extent, a static one as a solid band.
  const item = (label, band, power, agile, extra = {}) => ({label, power, band: agile ? null : band, extent: agile ? band : null, ...extra});
  const items = composite()
    ? plan.components.map(c => item(c.label, c.band_hz || [c.center_hz, c.center_hz], c.power_dbm, !!c.agility,
      {enabled: c.enabled, active: c.index === state.txUi.component}))
    : [item(waveformLabel(t.waveform), plan.band_hz, t.power_dbm, !!plan.agility, {active: true})];
  drawBandPlan(bandCanvas, items, {fs: t.sample_rate, offset: rf(), extent: composite() && plan.agility ? plan.carrier_extent_hz : null});
  // Without any hop/sweep schedule the schedule pane is hidden and the monitor grows.
  const monitorPanes = $('.tx-monitor', root), agile = !!plan.agility || plan.components?.some(c => c.agility);
  if (monitorPanes && monitorPanes.classList.contains('static') === agile) {
    monitorPanes.classList.toggle('static', !agile);
    host?.analyzer.render();
  }
  const path = composite() ? editPath() : 'tx';
  const summary = composite() && state.txUi.compositePage !== 'agility' ? (agilitySummary(path) || plan.agility) : plan.agility;
  const carrier = path === 'tx' ? rf() + Number(t.center_hz) : carrierOf(path) + getPath(state, path).center_hz;
  setText('txTimePlanValue', summary ? `${summary.mode} · ${fmtCount(summary.segments)} segments · period ${fmt(summary.period_s, 's')}` : '');
  drawTimePlan(timeCanvas, summary, {offset: carrier, fs: t.sample_rate});
  plotNote(timeCanvas, summary ? '' : 'Static frequency');
}

function updateReadouts() {
  const t = tx(), plan = state.txUi.planError ? null : state.txUi.plan;
  setText('txSignal', composite() ? `Composite · ${t.components.filter(c => c.enabled).length}` : waveformLabel(t.waveform));
  setText('txPower', plan ? fmt(plan.total_power_dbm, 'dBm') : '—');
  setText('txBand', plan?.band_hz ? fmt(plan.band_hz[1] - plan.band_hz[0], 'Hz') : '—');
  const [lo, hi] = plan?.carrier_extent_hz || [0, 0];
  setText('txExtent', !plan ? '—' : hi - lo > 0 ? `${fmt(lo + rf(), 'Hz')} … ${fmt(hi + rf(), 'Hz')}` : fmt(lo + rf(), 'Hz'));
  const a = plan?.agility;
  setText('txAgility', a ? `${a.mode} · ${fmtCount(a.segments)} · ${fmt(a.period_s, 's')}` : 'static');
  setText('txCapture', `${fmtCount(Math.round(Math.min(t.duration, maxCaptureSeconds()) * t.sample_rate))} samples · ${fmt(t.sample_rate, 'S/s')}${rf() ? ` · RF ${fmt(rf(), 'Hz')}` : ''}`);
}

/* ---------------------------------------------------------------- changes -- */

/** A design edit: validate (which applies a valid draft and restreams it). */
function changed(path = '', structural = false) {
  state.revision += 1;
  publish('tx:changed');
  requestPlan();
  renderStatus();
  if (structural) renderEditor();
  if (path === 'tx.sample_rate' || path === 'tx.rf_center_hz') { monitor.tune.center_hz = 0; monitor.tune.span_hz = null; host?.analyzer.resetTraces(); renderDock(); }
}

const STRUCTURAL = /\.(waveform|data_source|frame_enabled|frame_mode|role|pulse_shape|burst_enabled|output_filter|window_kind|mode|plan|enabled)$/;

function onChange(path) {
  if (path === 'txUi.family') { setMode(state.txUi.family); return; }
  if (path === 'txUi.noise') { if (state.txUi.noise && tx().snr_db == null) tx().snr_db = 30; renderDock(); changed(path); return; }
  if (path === 'tx.duration' || path === 'tx.sample_rate') {
    const max = maxCaptureSeconds();
    if (tx().duration > max) { tx().duration = max; toast(`Capture shortened to ${fmt(max, 's')} at this rate`); }
    renderDock();
  }
  // Field limits depend on the sample rate and the offsets, so those re-render the editor.
  const structural = STRUCTURAL.test(path) || path.startsWith('tx.ofdm_fft') || /\.label$/.test(path) || path.endsWith('.power_dbm')
    || path.endsWith('.center_hz') || path === 'tx.sample_rate';
  changed(path, structural);
}

function setMode(mode) {
  const t = tx();
  if ((mode === 'composite') === composite()) return;
  if (mode === 'composite') {
    // The current signal (with its own hopping and burst) becomes component 1;
    // clock, RF centre, offset and transmitter impairments stay on the composite.
    const signal = {...clone(t), components: [], label: waveformLabel(t.waveform), enabled: true, center_hz: 0, snr_db: null,
      iq_gain_db: 0, iq_phase_deg: 0, dc_i: 0, dc_q: 0};
    state.tx = {...clone(t), waveform: 'composite', components: [signal], agility: clone(state.defaults.agility), burst_enabled: false};
    state.txUi.component = 0;
    state.txUi.compositePage = 'components';
  } else {
    const keep = clone(t.components[editPath() === 'tx' ? 0 : state.txUi.component] || state.defaults.tx);
    const shared = {sample_rate: t.sample_rate, duration: t.duration, rf_center_hz: t.rf_center_hz, snr_db: t.snr_db, seed: t.seed};
    state.tx = {...keep, ...shared, components: [], label: '', enabled: true, center_hz: Number(t.center_hz) + Number(keep.center_hz)};
  }
  state.txUi.page = 'signal';
  renderDock();
  changed('tx.waveform', true);
}

/**
 * A component offset where a signal of ``width`` Hz overlaps no other
 * component and stays inside the sampled band, or ``null`` when there is none.
 */
function freeOffset(width = 0) {
  const t = tx(), fs = Number(t.sample_rate), parent = Number(t.center_hz) || 0;
  const plan = state.txUi.planError ? null : state.txUi.plan;
  const bands = t.components.map((c, i) => {
    const b = plan?.components?.[i]?.band_hz;
    return b ? [b[0] - parent, b[1] - parent] : [c.center_hz, c.center_hz];
  });
  const guard = Math.max(fs / 40, width / 2);
  for (const k of [0, 2, -2, 4, -4, 6, -6, 1, -1, 3, -3, 5, -5, 7, -7]) {
    const f = k * fs / 16;
    if (Math.abs(f + parent) + width / 2 >= fs / 2) continue;
    if (bands.every(([lo, hi]) => f + guard < lo || f - guard > hi)) return f;
  }
  return null;
}

function newComponent() {
  const t = tx(), n = t.components.length;
  return {...clone(state.defaults.tx), sample_rate: t.sample_rate, duration: t.duration, rf_center_hz: t.rf_center_hz,
    waveform: 'tone', label: `Signal ${n + 1}`, power_dbm: -10, center_hz: freeOffset() ?? 0, snr_db: null, symbol_rate: Math.min(50e3, t.sample_rate / 8)};
}

/* ---------------------------------------------------------------- actions -- */

function afterStructure(path) { changed(path, true); }

const actions = {
  ...structuredActions(afterStructure),
  ...frameActions(afterStructure, path => {
    const cfg = getPath(state, path.replace(/\.frame_blocks$/, ''));
    return {modulation: cfg.waveform, symbolRate: cfg.symbol_rate};
  }),
  ...programActions(afterStructure),
  ...ofdmActions(afterStructure),
  'pick-waveform': el => pickWaveform(el.dataset.path, el.value ?? el.dataset.waveform),
  'pick-family': el => pickWaveform(el.dataset.path, waveformForFamily(el.value)),
  'select-component': (el, event) => {
    if (event.target.closest('button')) return;
    state.txUi.component = Number(el.dataset.index);
    renderEditor();
  },
  'add-component': () => {
    if (tx().components.length >= 16) { toast('Up to 16 components'); return; }
    tx().components.push(newComponent());
    state.txUi.component = tx().components.length - 1;
    state.txUi.page = 'signal';
    changed('tx.components', true);
  },
  'duplicate-component': el => {
    if (tx().components.length >= 16) { toast('Up to 16 components'); return; }
    const index = Number(el.dataset.index), copy = clone(tx().components[index]);
    copy.label = `${copy.label || 'Signal'} copy`.slice(0, 64);
    // A copy on top of its original adds coherently and looks like one signal:
    // place it where it overlaps nothing (its design is otherwise identical).
    const band = state.txUi.planError ? null : state.txUi.plan?.components?.[index]?.band_hz;
    const free = freeOffset(band ? band[1] - band[0] : 0);
    if (free !== null) copy.center_hz = free;
    else toast('No free frequency left: the copy sits on its original until you move it');
    tx().components.splice(index + 1, 0, copy);
    state.txUi.component = index + 1;
    changed('tx.components', true);
  },
  'remove-component': el => {
    if (tx().components.length <= 1) { toast('A composite keeps at least one signal; switch to Single signal instead'); return; }
    tx().components.splice(Number(el.dataset.index), 1);
    state.txUi.component = Math.min(state.txUi.component, tx().components.length - 1);
    changed('tx.components', true);
  },
  'toggle-component': el => {
    const c = tx().components[Number(el.dataset.index)];
    c.enabled = !c.enabled;
    changed('tx.components', true);
  },
  export: () => {
    const c = state.exportCfg, body = {tx: txRequest(), capture_id: state.captureId || 0, format: c.format, container: c.container,
      scaling: c.scaling, headroom_db: Number(c.headroom_db), destination: c.destination};
    if (c.duration_s) body.duration_s = Number(c.duration_s);
    runExport(body, 'transmitter capture');
  },
};

/** Change the waveform of the signal or component at ``path``. */
function pickWaveform(path, waveform) {
  const cfg = getPath(state, path);
  if (!waveform || cfg.waveform === waveform) return;
  // A component still named after its waveform follows the new one.
  if (path !== 'tx' && cfg.label === waveformLabel(cfg.waveform)) cfg.label = waveformLabel(waveform);
  cfg.waveform = waveform;
  if (path === 'tx') state.txUi.family = familyOf(cfg.waveform);
  if (cfg.waveform === 'ofdm') cfg.ofdm_design.enabled = true;
  changed(`${path}.waveform`, true);
}

/* -------------------------------------------------------------- lifecycle -- */

export function start() {
  // stop() releases the live monitor (also when the page is hidden): rebuild it.
  if (!host && $('#txMain .analyzer', root)) mountMonitor();
  unsubscribe = [subscribe('live:meta', () => updateReadouts())];
}

export function stop() {
  // An edit still waiting for validation is validated (and applied) at once
  // rather than lost when another workspace opens.
  if (planTimer) requestPlan(true);
  for (const off of unsubscribe) off();
  unsubscribe = [];
  host?.destroy();
  host = null;
}

export function resize() {
  host?.analyzer.render();
  drawPlans();
  refreshOfdmGrid();
}

document.addEventListener('click', event => {
  if (state.mode !== 'tx') return;
  const tab = event.target.closest('#txMain [data-txpage]');
  if (tab) { state.txUi.page = tab.dataset.txpage; renderEditor(); return; }
  const sub = event.target.closest('#txMain [data-txcomp]');
  if (sub) { state.txUi.compositePage = sub.dataset.txcomp; renderEditor(); }
});
