/* Classifier: heuristic modulation recognition with a transparent decision path. */
import {$, esc, replaceHTML, setText} from '../core/dom.js';
import {get, post, errorText} from '../core/api.js';
import {bindControls, numberField, readout, section, sliderField, textPane, toggleField, views} from '../core/controls.js';
import {subscribe} from '../core/events.js';
import {fmt, fmtFreq, percent} from '../core/format.js';
import {clone, state} from '../core/state.js';
import {axisOffset, sourceRequest} from '../core/source.js';
import {setStatus, toast} from '../core/ui.js';
import {analyzerMarkup, niceCenter, niceSpan} from '../plots/analyzer.js';
import {linePlot} from '../plots/canvas.js';
import {AnalyzerHost, fullTune, sourceActions, sourceSection, spectrumRequest} from './common.js';

let root = null;
let host = null;
let timer = null;
let busy = false;
let result = null;
let emitters = [];   // separate bands found in the last whole-capture observation
let unsubscribe = [];
const view = {tune: {center_hz: 0, span_hz: null}, ui: null};

export const usesLive = true;

export function payload() {
  return {spectrum: spectrumRequest(view.tune)};
}

const ui = () => state.blindUi;

function analysisBand() {
  const b = ui().band;
  return b.enabled ? {band_center_hz: Number(b.center_hz) || 0, band_width_hz: Number(b.width_hz)} : {};
}

/* ----------------------------------------------------------------- dock -- */

function bandsList() {
  const bands = emitters;
  if (!bands.length) return '';
  return `<div class="list">${bands.map((b, i) => `<div class="list-item"><span class="grow">${esc(fmtFreq(b.center_hz + axisOffset(), b.width_hz / 100))} · ${esc(fmt(b.width_hz, 'Hz'))} · ${percent(b.power_fraction)}</span>`
    + `<button type="button" class="btn small" data-action="use-band" data-index="${i}">Analyse</button></div>`).join('')}</div>`;
}

function dockHTML() {
  const b = ui().band, off = axisOffset();
  return section('Source', sourceSection(), {id: 'cl-source'})
    + section('Analysis band', toggleField('Isolate one band before classifying', 'blindUi.band.enabled')
      + (b.enabled ? numberField('Centre', 'blindUi.band.center_hz', {unit: 'Hz', offset: off}) + numberField('Width', 'blindUi.band.width_hz', {unit: 'Hz', min: 1}) : '')
      + `${bandsList() || '<p class="note">No separate emitters detected in the last observation.</p>'}`,
    {id: 'cl-band', help: 'The classifier assumes one emitter. A band is mixed to 0 Hz, decimated with the analyzer’s down-converter and low-pass filtered to its edges before any feature is measured.'})
    + section('Observation', numberField('Interval', 'blindUi.interval', {unit: 's', value: ui().interval / 1000, min: 0.25, max: 60, help: 'One classification per interval of a fresh capture (each takes 20–200 ms of CPU).'})
      + sliderField('Report threshold', 'blindUi.threshold', {min: 0.2, max: 0.95, step: 0.05, unit: ''})
      + `<div class="btn-row"><button class="btn small${ui().paused ? ' on' : ''}" data-action="pause">${ui().paused ? 'Resume' : 'Pause'}</button><button class="btn small" data-action="analyse-now">Analyse now</button><button class="btn small ghost" data-action="clear-history">Clear history</button></div>`,
    {id: 'cl-observe'})
;
}

/* ----------------------------------------------------------------- main -- */

function mainHTML() {
  const page = ui().page;
  const tabs = views([['overview', 'Observation'], ['decision', 'Decision path'], ['features', 'Features'], ['history', 'History']], page, 'clpage');
  const readouts = `<div class="readout-bar">${readout('Hypothesis', 'cClass', '—', 'accent')}${readout('Score', 'cScore')}${readout('Family', 'cFamily')}${readout('Symbol rate', 'cRate')}${readout('SNR estimate', 'cSnr')}${readout('99 % BW', 'cBw')}${readout('Analysed', 'cWhen', '—', 'grow')}</div>`;
  if (page === 'overview') {
    return tabs + readouts + `<div class="stage split"><section class="pane"><header class="pane-head"><h3>Spectrum · drag to choose the analysis band</h3><span class="pane-value" id="clSpecValue"></span></header><div class="pane-body" style="display:flex">${analyzerMarkup({waterfall: true})}</div></section>${textPane('Verdict', 'clVerdict')}</div>`;
  }
  if (page === 'decision') return tabs + readouts + `<div class="stage">${textPane('How the classifier decided', 'clTrace')}</div>`;
  if (page === 'features') return tabs + readouts + `<div class="stage split">${textPane('Measured features', 'clFeatures')}<div class="stage" style="padding:0;grid-template-rows:1fr 1fr">${textPane('Constellation hypotheses', 'clCandidates')}<section class="pane"><header class="pane-head"><h3>Amplitude distribution</h3></header><div class="pane-body"><canvas id="clHist"></canvas></div></section></div></div>`;
  return tabs + readouts + `<div class="stage">${textPane('Observation history', 'clHistory')}</div>`;
}

export function render(el) {
  root = el;
  view.ui ||= {...clone(state.specUi), traces: [{mode: 'clear', y: null}, {mode: 'max', y: null}, {mode: 'off', y: null}, {mode: 'off', y: null}], markers: [], activeMarker: -1, memories: [], selection: null};
  el.innerHTML = `<div class="instrument"><aside class="dock"><div class="dock-head"><h2>Classifier</h2></div><div class="dock-scroll" data-keep-scroll="cl-dock"></div></aside><section class="main" id="clMain"></section></div>`;
  bindControls(el, {onChange, actions});
  renderDock();
  renderMain();
}

function renderDock() {
  replaceHTML($('.dock-scroll', root), dockHTML());
}

function renderMain() {
  host?.destroy();
  host = null;
  $('#clMain', root).innerHTML = mainHTML();
  if (ui().page === 'overview') {
    syncSelection();
    host = new AnalyzerHost($('#clMain .analyzer', root), {
      tune: view.tune, ui: view.ui, waterfall: true,
      bands: () => emitters.map((b, i) => ({lo: b.center_hz - b.width_hz / 2, hi: b.center_hz + b.width_hz / 2, label: `E${i + 1}`, color: '#7fd99a', fill: 'rgba(127,217,154,.04)'})),
      selectionActions: [{id: 'classify', label: 'Classify this band', primary: true}, {id: 'zoom', label: 'Zoom'}],
      onSelectionAction: (id, sel) => {
        if (id === 'zoom') { host.retune({center: (sel.lo + sel.hi) / 2, span: sel.hi - sel.lo}); return; }
        const width = niceSpan(sel.hi - sel.lo);
        Object.assign(ui().band, {enabled: true, center_hz: niceCenter((sel.lo + sel.hi) / 2, width), width_hz: width});
        renderDock();
        analyse(true);
      },
      hint: 'Drag across an emitter and choose “Classify this band” · green brackets are detected emitters',
    });
  }
  draw();
}

/** Show the analysis band as the analyzer selection. */
function syncSelection() {
  const b = ui().band;
  view.ui.selection = b.enabled ? {lo: b.center_hz - b.width_hz / 2, hi: b.center_hz + b.width_hz / 2} : null;
}

/* -------------------------------------------------------------- analysis -- */

async function analyse(force = false) {
  if (state.mode !== 'blind' || busy || (ui().paused && !force)) return;
  busy = true;
  const started = performance.now();
  try {
    const d = await post('/api/blind/analyze', {...sourceRequest(), ...analysisBand()});
    d.elapsed_ms = performance.now() - started;
    result = d;
    if (d.features.bands) emitters = d.features.bands;
    remember(d);
    if (state.mode === 'blind') { draw(); setStatus(`classifier · capture ${d.capture_id} in ${d.elapsed_ms.toFixed(0)} ms`); }
  } catch (error) {
    setStatus(errorText(error), true);
  } finally {
    busy = false;
  }
}

function remember(d) {
  const history = ui().history, key = `${d.source_revision}/${d.modulation}/${JSON.stringify(d.features.analysis_band || null)}`;
  const last = history.at(-1), stamp = new Date().toLocaleTimeString();
  if (last?.key === key) { last.last = stamp; last.count += 1; last.score = d.confidence; return; }
  history.push({key, first: stamp, last: stamp, count: 1, label: d.modulation, score: d.confidence, band: d.features.analysis_band || null});
  if (history.length > 100) history.shift();
}

/* ---------------------------------------------------------------- drawing -- */

const OP = {'>': '>', '>=': '≥', '<': '<', '<=': '≤', '==': '='};

function valueText(c) {
  const v = c.value;
  if (!Number.isFinite(v)) return '—';
  if (c.unit === 'dB' || c.unit === '%') return `${v.toFixed(2)} ${c.unit}`;
  if (Math.abs(v) >= 100 || Number.isInteger(v)) return String(Math.round(v * 100) / 100);
  return v.toPrecision(3);
}

function traceHTML(d) {
  return `<div class="trace">${d.trace.map(stage => {
    const accepted = stage.outcome === 'accepted';
    return `<div class="trace-stage ${stage.outcome}"><div class="trace-title">${esc(stage.title)}<span class="badge ${accepted ? 'good' : 'bad'}">${accepted ? `→ ${esc(stage.result)}` : 'rejected'}</span></div>`
      + `<div class="trace-q">${esc(stage.question)}</div><div class="checks">${stage.checks.map(c => `<div class="check ${c.passed ? 'pass' : 'fail'}"><span class="mark">${c.passed ? '✓' : '✗'}</span>`
        + `<span class="what" title="${esc(featureHelp(c.feature))}">${esc(c.label)}</span><span class="val"><b>${esc(valueText(c))}</b> ${OP[c.op] || c.op} ${esc(String(c.threshold))}${c.unit && c.unit !== '%' && c.unit !== 'dB' ? ` ${esc(c.unit)}` : ''}</span></div>`).join('')}</div>`
      + `${stage.detail ? `<div class="trace-detail">${esc(stage.detail)}</div>` : ''}</div>`;
  }).join('')}</div>`;
}

let glossary = null;
function featureHelp(feature) {
  return glossary?.[feature]?.description || '';
}

function verdictHTML(d) {
  const reported = d.confidence >= ui().threshold ? d.modulation : 'uncertain';
  const accepted = d.trace.find(s => s.outcome === 'accepted');
  const rejected = d.trace.filter(s => s.outcome === 'rejected');
  return `<div class="verdict"><strong>${esc(reported)}</strong><span>${esc(d.family)}</span></div>`
    + `<div class="score-bar" title="Heuristic agreement score"><i style="width:${(100 * d.confidence).toFixed(0)}%"></i></div>`
    + `<p class="note">Score ${(100 * d.confidence).toFixed(0)} / 100 (heuristic, not a probability)${d.confidence < ui().threshold ? ` · below the ${(100 * ui().threshold).toFixed(0)} report threshold` : ''}.</p>`
    + (accepted ? `<h4 style="margin:12px 0 4px">Why ${esc(d.modulation)}</h4><div class="checks">${accepted.checks.map(c => `<div class="check ${c.passed ? 'pass' : 'fail'}"><span class="mark">${c.passed ? '✓' : '✗'}</span><span class="what">${esc(c.label)}</span><span class="val"><b>${esc(valueText(c))}</b> ${OP[c.op]} ${esc(String(c.threshold))}</span></div>`).join('')}</div>` : '')
    + (rejected.length ? `<h4 style="margin:12px 0 4px">Ruled out first</h4>${rejected.map(s => {
      const failing = s.checks.find(c => !c.passed);
      return `<div class="check fail"><span class="mark">✗</span><span class="what">${esc(s.title)}</span><span class="val">${failing ? `${esc(failing.label)} <b>${esc(valueText(failing))}</b> ${OP[failing.op]} ${esc(String(failing.threshold))} failed` : esc(s.detail)}</span></div>`;
    }).join('')}` : '')
    + `<h4 style="margin:12px 0 4px">Notes</h4><ul style="margin:0;padding-left:18px;color:var(--muted);font-size:12px;line-height:1.6">${d.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul>`;
}

function featuresHTML(d) {
  const f = d.features, rows = [
    ['papr_db', 'Peak-to-average power', f.papr_db, 'dB', 0, 15],
    ['envelope_cv', 'Envelope variation', f.envelope_cv, '', 0, 1],
    ['spectral_flatness', 'Spectral flatness', f.spectral_flatness, '', 0, 1],
    ['envelope_kurtosis', 'Envelope kurtosis', f.envelope_kurtosis, '', 0.8, 3],
    ['duty', 'Active duty', f.duty, '', 0, 1],
    ['spectral_peaks', 'Spectral lines', f.spectral_peaks, '', 0, 40],
    ['clock_peak_db', 'Symbol-clock line', f.clock_peak_db, 'dB', 0, 60],
  ].filter(r => Number.isFinite(r[2]));
  const thresholds = {};
  for (const stage of d.trace) for (const c of stage.checks) (thresholds[c.feature] ||= new Set()).add(c.threshold);
  return rows.map(([key, label, value, unit, lo, hi]) => {
    const pos = v => `${Math.max(0, Math.min(100, (v - lo) / (hi - lo) * 100))}%`;
    return `<div class="feature-row"><div class="f-top"><span>${esc(label)}</span><b>${esc(unit ? `${value.toFixed(2)} ${unit}` : value.toPrecision(3))}</b></div>`
      + `<div class="gauge"><b style="width:${pos(value)}"></b>${[...(thresholds[key] || [])].map(t => `<i style="left:${pos(t)}" title="rule threshold ${t}"></i>`).join('')}</div>`
      + `<p>${esc(featureHelp(key))}</p></div>`;
  }).join('');
}

function candidatesHTML(d) {
  if (!d.candidates?.length) return '<p class="note">No linear-constellation fit was needed for this observation.</p>';
  const worst = Math.max(...d.candidates.map(c => c.evm_percent), 1);
  return `<table class="table"><thead><tr><th>Constellation</th><th>Fit EVM</th><th>Points used</th><th></th></tr></thead><tbody>${d.candidates.map(c => `<tr class="${c.modulation === d.modulation ? 'selected' : ''}"><td>${esc(c.modulation)}</td>`
    + `<td class="num-cell">${c.evm_percent.toFixed(1)} %<div class="gauge" style="margin:3px 0 0"><b style="width:${(100 * c.evm_percent / worst).toFixed(0)}%"></b></div></td>`
    + `<td class="num-cell">${c.points_used != null ? percent(c.points_used) : '—'}</td><td>${c.penalized ? '<span class="badge warn" title="Random data would visit at least 75 % of the points">unused points</span>' : c.modulation === d.modulation ? '<span class="badge good">best</span>' : ''}</td></tr>`).join('')}</tbody></table>`
;
}

function draw() {
  if (!root || state.mode !== 'blind') return;
  const d = result;
  if (!d) return;
  const reported = d.confidence >= ui().threshold ? d.modulation : 'uncertain';
  setText('cClass', reported.toUpperCase());
  setText('cScore', `${(100 * d.confidence).toFixed(0)} / 100`);
  setText('cFamily', d.family);
  setText('cRate', fmt(d.symbol_rate_baud, 'Bd'));
  setText('cSnr', fmt(d.snr_db, 'dB'));
  setText('cBw', fmt(d.bandwidth_99_hz, 'Hz'));
  const band = d.features.analysis_band;
  setText('cWhen', `capture #${d.capture_id} · ${d.elapsed_ms?.toFixed(0)} ms${band ? ` · band ${fmtFreq(band.center_hz + axisOffset(), band.width_hz / 100)} ± ${fmt(band.width_hz / 2, 'Hz')}` : ' · whole capture'}`);
  if ($('#clVerdict')) replaceHTML($('#clVerdict'), verdictHTML(d));
  if ($('#clTrace')) replaceHTML($('#clTrace'), traceHTML(d));
  if ($('#clFeatures')) replaceHTML($('#clFeatures'), featuresHTML(d));
  if ($('#clCandidates')) replaceHTML($('#clCandidates'), candidatesHTML(d));
  if ($('#clHist')) linePlot($('#clHist'), d.features.amplitude_histogram, {x: d.features.amplitude_bins, min: 0});
  if ($('#clHistory')) {
    replaceHTML($('#clHistory'), `<table class="table"><thead><tr><th>First</th><th>Last</th><th>Hypothesis</th><th>Score</th><th>Count</th><th>Band</th></tr></thead><tbody>${ui().history.slice().reverse().map(h => `<tr><td>${h.first}</td><td>${h.last}</td><td>${esc(h.label)}</td><td class="num-cell">${(100 * h.score).toFixed(0)}</td><td class="num-cell">${h.count}</td><td>${h.band ? esc(`${fmt(h.band.center_hz, 'Hz')} ± ${fmt(h.band.width_hz / 2, 'Hz')}`) : 'whole'}</td></tr>`).join('')}</tbody></table>`);
  }
  const bandSection = $('[data-fold="cl-band"]', root);
  if (bandSection && !bandSection.contains(document.activeElement)) {
    const tmp = document.createElement('div');
    tmp.innerHTML = dockHTML();
    bandSection.replaceWith(tmp.querySelector('[data-fold="cl-band"]'));
  }
}

/* -------------------------------------------------------------- changes -- */

function onChange(path, value) {
  if (path === 'blindUi.interval') { ui().interval = Math.round(value * 1000); restartTimer(); return; }
  if (path.startsWith('blindUi.band')) {
    syncSelection();
    host?.analyzer.render();
    if (path === 'blindUi.band.enabled') renderDock();
    analyse(true);
    return;
  }
  draw();
}

const actions = {
  ...sourceActions(() => render(root)),
  pause: () => { ui().paused = !ui().paused; renderDock(); },
  'analyse-now': () => analyse(true),
  'clear-history': () => { ui().history = []; draw(); },
  'use-band': el => {
    const b = emitters[Number(el.dataset.index)];
    if (!b) return;
    const width = niceSpan(b.width_hz * 1.1);
    Object.assign(ui().band, {enabled: true, center_hz: niceCenter(b.center_hz, width), width_hz: width});
    syncSelection();
    renderDock();
    host?.analyzer.render();
    analyse(true);
    toast(`Analysing ${fmtFreq(b.center_hz + axisOffset(), b.width_hz / 100)}`);
  },
};

function restartTimer() {
  clearInterval(timer);
  timer = setInterval(() => analyse(false), Math.max(250, ui().interval));
}

/* ------------------------------------------------------------ lifecycle -- */

export async function start() {
  // stop() releases the analyzer (also when the page is hidden): rebuild it.
  if (!host && ui().page === 'overview' && root) renderMain();
  unsubscribe = [subscribe('live:meta', () => { result = null; analyse(true); })];
  if (!glossary) {
    try { glossary = (await get('/api/blind/features')).features; } catch { glossary = {}; }
  }
  analyse(true);
  restartTimer();
}

export function stop() {
  clearInterval(timer);
  for (const off of unsubscribe) off();
  unsubscribe = [];
  host?.destroy();
  host = null;
}

export function resize() {
  host?.analyzer.render();
  draw();
}

document.addEventListener('click', event => {
  if (state.mode !== 'blind') return;
  const tab = event.target.closest('#clMain [data-clpage]');
  if (tab) { ui().page = tab.dataset.clpage; renderMain(); }
});
