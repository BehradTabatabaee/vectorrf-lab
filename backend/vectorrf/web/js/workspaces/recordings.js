/* Recordings: capture library, interpretation, overview with playback, export. */
import {$, esc, replaceHTML, setText} from '../core/dom.js';
import {del, get, post, errorText} from '../core/api.js';
import {bindControls, numberField, readout, section, selectField, toggleField, views} from '../core/controls.js';
import {publish, subscribe} from '../core/events.js';
import {bytesText, fmt, fmtCount, fmtFreq} from '../core/format.js';
import {live, update as liveUpdate} from '../core/live.js';
import {state} from '../core/state.js';
import {LIVE_WINDOW_SAMPLES, activeFile, recordingSpec, syncLiveSource} from '../core/source.js';
import {loadRecordingSettings, saveRecordingSettings} from '../core/persist.js';
import {confirmDialog, renderStatus, setStatus, toast} from '../core/ui.js';
import {exportFields, runExport} from '../editors/export.js';
import {clear, colors, heatmap, linePlot, setup} from '../plots/canvas.js';
import {pacingControl, setPacing, switchSource} from './common.js';

const FORMATS = [['', 'Automatic / from file'], ['cf32_le', 'cf32 · complex float32 (GNU Radio)'], ['ci16_le', 'ci16 · complex int16 (UHD sc16)'],
  ['ci8', 'ci8 · complex int8 (HackRF)'], ['cu8', 'cu8 · complex uint8 (RTL-SDR)'], ['cf64_le', 'cf64 · complex float64'], ['ci32_le', 'ci32 · complex int32'],
  ['ci24_le', 'ci24 · packed int24'], ['cf32_be', 'cf32 big-endian'], ['ci16_be', 'ci16 big-endian'], ['rf32_le', 'rf32 · real float32'], ['ri16_le', 'ri16 · real int16']];
const ACCEPT = '.iq,.bin,.dat,.raw,.cf32,.cfile,.fc32,.cf64,.cs16,.sc16,.ci16,.cs8,.sc8,.ci8,.cu8,.wav,.sigmf,.sigmf-meta,.sigmf-data';

let root = null;
let unsubscribe = [];
let geometry = null;      // spectrogram plot geometry for hit testing
let region = null;        // {t0, t1} time selection on the overview
let dragging = null;

export const usesLive = true;
export function payload() { return {}; }

const files = () => state.files;
const fileCount = () => `${files().list.length} ${files().list.length === 1 ? 'file' : 'files'}`;
const sel = () => state.files.selected;
const isLive = name => state.source.kind === 'file' && state.source.file?.name === name;

/* ----------------------------------------------------------------- data -- */

async function refreshLibrary(select) {
  try {
    const d = await get('/api/files');
    Object.assign(files(), {list: d.files, directory: d.directory, maxUpload: d.max_upload_bytes, loaded: true, error: ''});
    if (select) await selectRecording(select);
  } catch (error) {
    files().error = errorText(error);
  }
  if (state.mode === 'files') { renderDock(); renderMain(); }
}

async function selectRecording(name) {
  const entry = files().list.find(f => f.name === name);
  if (!entry) return;
  const keep = sel()?.name === name ? sel() : null;
  files().selected = keep || {name, format: entry.needs_format ? (entry.format || '') : '', sample_rate: entry.sample_rate || '',
    center_frequency_hz: entry.center_frequency_hz ?? '', header_bytes: 0, start_sample: 0,
    capture_samples: Math.min(65536, state.limits.max_capture_samples), loop: true, power_offset_db: 0, power_unit: 'dBFS',
    ...(keep ? {} : loadRecordingSettings(name))};
  files().overview = null;
  region = null;
  await inspect();
}

async function inspect() {
  const spec = recordingSpec();
  if (!spec) return;
  files().busy = true;
  files().error = '';
  try {
    const info = await post('/api/files/inspect', spec);
    files().info = info;
    const row = files().list.find(r => r.name === spec.name);
    if (row) Object.assign(row, {format: info.format, sample_rate: info.sample_rate, duration_s: info.duration_s, needs_format: false});
    if (!sel().sample_rate && info.sample_rate) sel().sample_rate = info.sample_rate;
    if (isLive(spec.name)) { state.source.info = info; syncLiveSource(); }
    if (info.sample_rate) await refreshOverview();
  } catch (error) {
    files().info = null;
    files().error = errorText(error);
  } finally {
    files().busy = false;
    if (state.mode === 'files') { renderDock(); renderMain(); }
  }
}

async function refreshOverview() {
  const spec = recordingSpec();
  if (!spec) return;
  try { files().overview = await post('/api/files/overview', {source: spec, rows: 256, nfft: 1024}); }
  catch (error) { files().overview = null; files().error = errorText(error); }
}

function upload(fileList) {
  const queue = [...fileList].sort((a, b) => Number(b.name.endsWith('.sigmf-meta')) - Number(a.name.endsWith('.sigmf-meta')));
  if (!queue.length) return;
  let last = null;
  const next = () => {
    const file = queue.shift();
    if (!file) { files().upload = null; refreshLibrary(last); return; }
    files().upload = {name: file.name, loaded: 0, total: file.size};
    renderUpload();
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', `/api/files/upload?name=${encodeURIComponent(file.name)}`);
    xhr.upload.onprogress = e => { if (files().upload) { files().upload.loaded = e.loaded; renderUpload(); } };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const d = JSON.parse(xhr.responseText);
        last = d.name.endsWith('.sigmf-data') ? d.name.replace(/\.sigmf-data$/, '.sigmf-meta') : d.name;
        toast(`Uploaded ${d.name} · ${bytesText(d.bytes)}`, 'good');
      } else toast(`Upload failed: ${errorText(xhr.responseText)}`, 'bad');
      next();
    };
    xhr.onerror = () => { toast('Upload failed: connection error', 'bad'); next(); };
    xhr.send(file);
  };
  next();
}

function renderUpload() {
  const u = files().upload, el = $('#uploadProgress');
  if (!el) return;
  el.hidden = !u;
  if (u) el.innerHTML = `<i style="width:${u.total ? (100 * u.loaded / u.total).toFixed(1) : 0}%"></i>`;
  setText('uploadText', u ? `Uploading ${u.name} · ${bytesText(u.loaded)} / ${bytesText(u.total)}` : '');
}

/* ----------------------------------------------------------------- dock -- */

function libraryHTML() {
  const filter = files().filter.toLowerCase();
  const rows = files().list.filter(f => !filter || f.name.toLowerCase().includes(filter)).map(f => {
    const kind = f.container === 'raw' ? (f.format || 'raw').split('_')[0] : f.container || '?';
    const meta = [f.container === 'raw' && f.needs_format ? '<span class="warn">format?</span>' : esc(f.format || ''),
      f.sample_rate ? esc(fmt(f.sample_rate, 'S/s')) : '<span class="warn">rate?</span>', f.duration_s ? esc(fmt(f.duration_s, 's')) : '', esc(bytesText(f.bytes))].filter(Boolean).join(' · ');
    return `<div class="rec${f.name === sel()?.name ? ' active' : ''}${isLive(f.name) ? ' live' : ''}" data-action="select" data-name="${esc(f.name)}" role="button" tabindex="0">`
      + `<span class="rec-icon">${esc(kind.slice(0, 5))}</span><div style="min-width:0"><div class="rec-name">${esc(f.name)}${isLive(f.name) ? ' <span class="badge info">live</span>' : ''}</div>`
      + `<div class="rec-meta">${meta}</div>${f.error ? `<div class="rec-meta" style="color:var(--red)">${esc(f.error)}</div>` : ''}</div></div>`;
  }).join('');
  return `<div class="library"><div class="library-tools"><input type="search" placeholder="Filter recordings" value="${esc(files().filter)}" data-bind="files.filter" data-kind="text" aria-label="Filter recordings">`
    + `<button type="button" class="btn small" data-action="refresh" title="Rescan the library folder">↻</button></div>`
    + `<div class="drop-target" id="dropTarget"><strong>Drop I/Q recordings here</strong>raw .iq/.bin/.cf32/.cs16/.cu8 · WAV/RF64 · SigMF<br><button type="button" class="btn small" data-action="choose" style="margin-top:6px">Choose files…</button></div>`
    + `<div class="upload-progress" id="uploadProgress" hidden></div><p class="note" id="uploadText" style="margin:0 10px"></p>`
    + `<div class="library-list" data-keep-scroll="library">${rows || `<div class="empty">${files().loaded ? 'The library is empty.' : 'Loading…'}</div>`}</div>`
    + `<p class="note" style="margin:6px 10px 10px">Folder: <code>${esc(files().directory || '—')}</code></p></div>`;
}

/* ----------------------------------------------------------------- main -- */

function heroHTML() {
  const f = sel(), info = files().info;
  const cells = [['Recording', f.name], ['Format', info?.format || '—'], ['Sample rate', fmt(info?.sample_rate, 'S/s')], ['Duration', fmt(info?.duration_s, 's')],
    ['Samples', fmtCount(info?.num_samples)], ['RF centre', info?.center_frequency_hz ? fmtFreq(info.center_frequency_hz, 1e3) : '—'], ['Container', info?.container || '—']];
  return `<div class="hero">${cells.map(([k, v]) => `<div><span>${esc(k)}</span><b title="${esc(v)}">${esc(v)}</b></div>`).join('')}</div>`;
}

function transportHTML() {
  const f = sel(), info = files().info, live = isLive(f.name);
  const ready = !!info?.sample_rate;
  return `<div class="transport"><button type="button" class="btn ${live ? '' : 'primary'}" data-action="toggle-live"${ready ? '' : ' disabled title="Set the sample rate first"'}>${live ? '■ Return to transmitter' : '▶ Use as live source'}</button>`
    + `<button type="button" class="btn small" data-action="seek-start" title="Jump to the start">⏮</button><button type="button" class="btn small" data-action="step" data-dir="-1" title="Back one window">◀</button><button type="button" class="btn small" data-action="step" data-dir="1" title="Forward one window">▶</button>`
    + `<span class="position" id="recPosition">—</span><span class="grow"></span>`
    + `${toggleField('Loop', 'files.selected.loop')}`
    + `<div class="transport-pacing">${pacingControl()}</div>`
    + `<button type="button" class="btn small" data-action="open-spectrum"${ready ? '' : ' disabled'}>Spectrum ↗</button><button type="button" class="btn small" data-action="open-classifier"${ready ? '' : ' disabled'}>Classify ↗</button></div>`;
}

function interpretationHTML() {
  const f = sel(), info = files().info, raw = !info || info.container === 'raw';
  const candidates = info?.format_candidates?.length ? `<table class="table"><thead><tr><th>Format</th><th>Content score</th><th>Evidence</th></tr></thead><tbody>${info.format_candidates.slice(0, 6).map(c => `<tr class="clickable" data-action="use-format" data-format="${esc(c.format)}"><td>${esc(c.format)}</td><td class="num-cell">${Math.round(100 * c.score)} %</td><td>${esc((c.reasons || []).join('; '))}</td></tr>`).join('')}</tbody></table>` : '';
  return `<div class="page"><div class="columns"><div class="card"><header>Sample interpretation</header><div class="card-body">`
    + (raw ? selectField('Sample format', 'files.selected.format', FORMATS) : `<p class="note">${esc(info.container.toUpperCase())} header: ${esc(info.format)}${info.real_valued ? ' · real-valued' : ''}.</p>`)
    + numberField('Sample rate', 'files.selected.sample_rate', {unit: 'S/s', min: 1, max: state.limits.max_sample_rate, nullable: true, placeholder: 'required'})
    + numberField('RF centre', 'files.selected.center_frequency_hz', {unit: 'Hz', nullable: true, placeholder: 'unknown'})
    + (raw ? numberField('Header bytes', 'files.selected.header_bytes', {min: 0, integer: true}) : '')
    + (info?.format_warning ? `<div class="callout warn">${esc(info.format_warning)}</div>` : '')
    + '</div></div><div class="card"><header>Level calibration</header><div class="card-body">'
    + selectField('Display unit', 'files.selected.power_unit', [['dBFS', 'dBFS · relative to full scale'], ['dBm', 'dBm · absolute with an offset']])
    + numberField('Offset', 'files.selected.power_offset_db', {unit: 'dB'})
    + (info?.extra?.['vectorrf:full_scale_dbm'] != null ? `<button type="button" class="btn small" data-action="apply-calibration">Use recorded calibration (full scale = ${esc(fmt(info.extra['vectorrf:full_scale_dbm'], 'dBm'))})</button>` : '')

    + '</div></div></div>'
    + (candidates ? `<div class="card"><header>Content heuristics (headerless files store no format)</header><div class="card-body">${candidates}<p class="note">Click a row to read the file in that format.</p></div></div>` : '')
    + `<div class="card"><header>Capture window</header><div class="card-body">${numberField('Window', 'files.selected.capture_samples', {min: 64, max: state.limits.max_capture_samples, integer: true, help: `Samples per live capture. Real-time pacing plays windows of at most ${fmtCount(LIVE_WINDOW_SAMPLES)} samples on the wall clock.`})}`
    + numberField('Step', 'playback.step', {min: 1, integer: true, nullable: true, placeholder: 'one window', help: 'Every-sample pacing only: samples between successive windows (leave empty for contiguous playback; larger steps scan a long recording faster).'})
    + `${numberField('Start position', 'recView.start', {unit: 's', min: 0, value: info?.sample_rate ? Number(f.start_sample || 0) / info.sample_rate : 0})}</div></div></div>`;
}

function exportHTML() {
  const info = files().info;
  const rate = info?.sample_rate;
  const span = region && rate ? `${fmt(region.t0, 's')} … ${fmt(region.t1, 's')} (${fmtCount(Math.round((region.t1 - region.t0) * rate))} samples)` : '';
  return `<div class="page"><div class="card"><header>Export or convert an excerpt</header><div class="card-body">`
    + (span ? `<div class="callout good">Time region from the overview: ${esc(span)}<button type="button" class="btn small ghost" data-action="clear-region" style="margin-left:8px">clear</button></div>` : '<p class="note">Drag vertically on the Overview spectrogram to mark a time region, or export from the playback position.</p>')
    + exportFields(true) + '<div class="btn-row"><button type="button" class="btn primary" data-action="export">Export excerpt</button></div></div></div></div>';
}

function metadataHTML() {
  const info = files().info;
  if (!info) return '<div class="page"><p class="note">No metadata yet.</p></div>';
  const rows = [['Container', info.container], ['Sample format', info.format], ['Samples', fmtCount(info.num_samples)], ['Sample rate', fmt(info.sample_rate, 'S/s')],
    ['Duration', fmt(info.duration_s, 's')], ['RF centre', info.center_frequency_hz ? fmtFreq(info.center_frequency_hz, 1) : '—'], ['Data offset', `${info.data_offset} bytes`],
    ['Recorded', info.datetime || '—'], ['Description', info.description || '—'], ['Data file', info.data_file || '—']];
  return `<div class="page"><div class="card"><header>Recording</header><div class="card-body"><table class="table"><tbody>${rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td class="num-cell">${esc(v)}</td></tr>`).join('')}</tbody></table></div></div>`
    + (info.notes?.length ? `<div class="card"><header>Notes</header><div class="card-body"><ul style="margin:0;padding-left:18px;color:var(--muted);line-height:1.6">${info.notes.map(n => `<li>${esc(n)}</li>`).join('')}</ul></div></div>` : '')
    + (info.annotations?.length ? `<div class="card"><header>SigMF annotations (${info.annotations.length})</header><div class="card-body"><table class="table"><thead><tr><th>Label</th><th>Start</th><th>Length</th></tr></thead><tbody>${info.annotations.slice(0, 100).map(a => `<tr><td>${esc(a['core:label'] || a['core:comment'] || 'annotation')}</td><td class="num-cell">${fmtCount(a['core:sample_start'])}</td><td class="num-cell">${fmtCount(a['core:sample_count'])}</td></tr>`).join('')}</tbody></table></div></div>` : '')
    + (Object.keys(info.extra || {}).length ? `<div class="card"><header>Extra metadata</header><div class="card-body"><pre class="code">${esc(JSON.stringify(info.extra, null, 2))}</pre></div></div>` : '') + '</div>';
}

function mainHTML() {
  const f = sel();
  if (!f) {
    return `<div class="empty-state" id="emptyDrop"><h2>Open an I/Q recording</h2><p>Drop files anywhere on this page or choose them in the library. Raw interleaved files (cf32, ci16, ci8, cu8, …), WAV/RF64 and SigMF are memory-mapped, so multi-gigabyte wideband captures open instantly.</p>`
      + `<p>Pick a recording to see its whole-file spectrogram, set how it is interpreted, play it into Spectrum, Receiver and Classifier, and export excerpts.</p><button type="button" class="btn primary" data-action="choose">Choose files…</button></div>`;
  }
  const page = files().page || 'overview';
  const tabs = views([['overview', 'Overview'], ['interpretation', 'Interpretation'], ['export', 'Export'], ['metadata', 'Metadata']], page, 'recpage',
    `<button type="button" class="btn small danger" data-action="delete">Delete…</button>`);
  const errors = files().error ? `<div class="callout bad" style="margin:8px">${esc(files().error)}</div>` : '';
  let body;
  if (page === 'overview') {
    body = files().overview
      ? `<div class="overview"><section class="pane"><header class="pane-head"><h3>Spectrogram · whole recording · click to seek, drag to mark a region</h3><span class="pane-value" id="ovSpecValue"></span></header><div class="pane-body"><canvas id="ovSpec"></canvas></div></section>`
        + `<section class="pane"><header class="pane-head"><h3>Power</h3></header><div class="pane-body"><canvas id="ovPower"></canvas></div></section>`
        + `<section class="pane"><header class="pane-head"><h3>Average spectrum</h3><span class="pane-value" id="ovAvgValue"></span></header><div class="pane-body"><canvas id="ovAvg"></canvas></div></section><div></div></div>`
      : `<div class="empty-state"><p>${files().busy ? 'Reading the recording…' : files().info && !files().info.sample_rate ? 'Set the sample rate in Interpretation to see the overview.' : 'No overview available.'}</p></div>`;
  } else body = `<div class="scroll-page" data-keep-scroll="rec-${page}">${page === 'interpretation' ? interpretationHTML() : page === 'export' ? exportHTML() : metadataHTML()}</div>`;
  return heroHTML() + transportHTML() + tabs + errors + body;
}

export function render(el) {
  root = el;
  el.innerHTML = `<div class="instrument"><aside class="dock"><div class="dock-head"><h2>Recordings</h2><span class="badge" id="libCount">${fileCount()}</span></div><div class="dock-scroll" id="recDock" style="padding-bottom:0"></div></aside><section class="main" id="recMain"></section></div>`;
  bindControls(el, {onChange, actions});
  bindDrop(el);
  renderDock();
  renderMain();
  if (!files().loaded) refreshLibrary();
}

function renderDock() {
  replaceHTML($('#recDock', root), libraryHTML());
  setText('libCount', fileCount());
  renderUpload();
}

function renderMain() {
  replaceHTML($('#recMain', root), mainHTML());
  draw();
}

/* -------------------------------------------------------------- overview -- */

function livePosition() {
  const m = live.metrics, f = sel();
  return isLive(f?.name) && Number.isFinite(m.capture_start_sample) && m.source_kind === 'file' ? m.capture_start_sample : Number(f?.start_sample || 0);
}

function draw() {
  const o = files().overview, f = sel();
  const rate = files().info?.sample_rate;
  if (rate) setText('recPosition', `${fmt(livePosition() / rate, 's')} / ${fmt(files().info.duration_s, 's')}${isLive(f.name) ? ` · capture ${live.metrics.capture_id ?? 0}${live.metrics.at_end ? ' · end' : ''}` : ''}`);
  if (!o || !$('#ovSpec')) return;
  const offset = Number(f.center_frequency_hz || o.center_frequency_hz) || 0;
  const lo = Math.min(...o.avg_psd_db) - 10, hi = Math.max(...o.avg_psd_db) + 6;
  const position = livePosition() / o.recording_samples * o.duration_s;
  geometry = heatmap($('#ovSpec'), o.spectrogram_db, o.spectrogram_shape, {lo, hi, x: o.freq_hz.map(v => v + offset), xUnit: 'Hz', y: [0, o.duration_s], yUnit: 's', cursorY: position});
  if (geometry && region) {
    const s = setup($('#ovSpec'));
    if (s) {
      const y0 = geometry.Y(region.t0), y1 = geometry.Y(region.t1);
      s.ctx.fillStyle = 'rgba(233,176,75,.16)';
      s.ctx.fillRect(geometry.m.left, Math.min(y0, y1), geometry.pw, Math.abs(y1 - y0));
    }
  }
  drawPowerStrip($('#ovPower'), o, position);
  linePlot($('#ovAvg'), o.avg_psd_db, {x: o.freq_hz.map(v => v + offset), xUnit: 'Hz', yUnit: o.unit});
  setText('ovSpecValue', `${(100 * o.coverage).toFixed(o.coverage < 0.01 ? 3 : 1)} % of samples examined`);
  setText('ovAvgValue', `${o.unit} per ${fmt(o.rbw_hz, 'Hz')}`);
}

/** Power vs time drawn vertically, sharing the spectrogram's time axis. */
function drawPowerStrip(canvas, o, position) {
  const s = setup(canvas);
  if (!s || !geometry) return;
  const {ctx, w} = s;
  clear(ctx, w, s.h);
  const c = colors();
  const lo = Math.min(...o.power_db), hi = Math.max(...o.power_db), span = Math.max(1e-9, hi - lo);
  const X = v => 8 + (v - lo) / span * (w - 16);
  ctx.strokeStyle = c.grid;
  ctx.strokeRect(4.5, geometry.m.top + .5, w - 9, geometry.ph - 1);
  ctx.strokeStyle = c.cyan;
  ctx.beginPath();
  o.power_db.forEach((v, i) => {
    const y = geometry.Y(o.time_s[i]);
    if (i) ctx.lineTo(X(v), y); else ctx.moveTo(X(v), y);
  });
  ctx.stroke();
  ctx.strokeStyle = '#f0d49b';
  ctx.beginPath(); ctx.moveTo(4, geometry.Y(position)); ctx.lineTo(w - 4, geometry.Y(position)); ctx.stroke();
  ctx.fillStyle = c.text;
  ctx.font = `10px ${c.mono}`;
  ctx.textAlign = 'center';
  ctx.fillText(`${lo.toFixed(0)}…${hi.toFixed(0)} ${o.unit}`, w / 2, geometry.m.top + geometry.ph + 14);
}

function timeAt(event) {
  const canvas = $('#ovSpec'), o = files().overview;
  if (!canvas || !geometry || !o) return null;
  const rect = canvas.getBoundingClientRect();
  const y = event.clientY - rect.top;
  const frac = Math.max(0, Math.min(1, (y - geometry.m.top) / geometry.ph));
  return frac * o.duration_s;
}

function seekTo(seconds) {
  const f = sel(), o = files().overview;
  if (!f || !o) return;
  f.start_sample = Math.max(0, Math.min(o.recording_samples - 1, Math.round(seconds / o.duration_s * o.recording_samples)));
  applyPlayback();
}

function applyPlayback() {
  syncLiveSource();
  if (isLive(sel().name)) { state.revision += 1; liveUpdate(true); }
  draw();
}

function bindOverview(el) {
  if (el.dataset.overviewBound) return;
  el.dataset.overviewBound = '1';
  el.addEventListener('mousedown', event => {
    if (!event.target.closest('#ovSpec')) return;
    const t = timeAt(event);
    if (t === null) return;
    dragging = {t0: t, moved: false};
    event.preventDefault();
  });
  window.addEventListener('mousemove', event => {
    if (!dragging || state.mode !== 'files') return;
    const t = timeAt(event);
    if (t === null) return;
    if (Math.abs(t - dragging.t0) > files().overview.duration_s * 0.004) dragging.moved = true;
    if (dragging.moved) { region = {t0: Math.min(dragging.t0, t), t1: Math.max(dragging.t0, t)}; draw(); }
  });
  window.addEventListener('mouseup', event => {
    if (!dragging || state.mode !== 'files') return;
    const d = dragging;
    dragging = null;
    if (!d.moved) { region = null; seekTo(d.t0); toast(`Position ${fmt(d.t0, 's')}`); }
    else toast(`Region ${fmt(region.t0, 's')} … ${fmt(region.t1, 's')} marked for export`);
  });
}

function bindDrop(el) {
  bindOverview(el);
  if (el.dataset.dropBound) return;
  el.dataset.dropBound = '1';
  el.addEventListener('dragover', event => {
    if (state.mode !== 'files') return;
    event.preventDefault();
    $('#dropTarget')?.classList.add('over');
  });
  el.addEventListener('dragleave', event => { if (!el.contains(event.relatedTarget)) $('#dropTarget')?.classList.remove('over'); });
  el.addEventListener('drop', event => {
    if (state.mode !== 'files') return;
    event.preventDefault();
    $('#dropTarget')?.classList.remove('over');
    upload(event.dataTransfer.files);
  });
}

/* -------------------------------------------------------------- changes -- */

function onChange(path, value) {
  if (path === 'files.filter') { renderDock(); return; }
  const f = sel();
  if (!f) return;
  if (path === 'recView.start') { seekTo(value); return; }
  saveRecordingSettings(f);
  if (['files.selected.format', 'files.selected.sample_rate', 'files.selected.header_bytes', 'files.selected.center_frequency_hz'].includes(path)) { inspect(); return; }
  if (['files.selected.power_offset_db', 'files.selected.power_unit'].includes(path)) { refreshOverview().then(() => { if (state.mode === 'files') renderMain(); }); }
  if (path.startsWith('exportCfg.')) { if (path === 'exportCfg.scaling') renderMain(); return; }
  applyPlayback();
  renderStatus();
}

const actions = {
  select: el => selectRecording(el.dataset.name),
  refresh: () => refreshLibrary(sel()?.name),
  choose: () => {
    const input = Object.assign(document.createElement('input'), {type: 'file', multiple: true, accept: ACCEPT});
    input.onchange = () => upload(input.files);
    input.click();
  },
  'toggle-live': () => switchSource(isLive(sel().name) ? 'transmitter' : 'file', {rerender: () => { renderDock(); renderMain(); }}),
  pacing: el => { setPacing(el.dataset.pacing); applyPlayback(); },
  'seek-start': () => { sel().start_sample = 0; region = null; applyPlayback(); },
  step: el => {
    const f = sel(), info = files().info;
    if (!info) return;
    f.start_sample = Math.max(0, Math.min(info.num_samples - 1, Number(f.start_sample || 0) + Number(el.dataset.dir) * Number(f.capture_samples)));
    applyPlayback();
  },
  'open-spectrum': () => { if (!isLive(sel().name)) switchSource('file'); publish('navigate', 'spectrum'); },
  'open-classifier': () => { if (!isLive(sel().name)) switchSource('file'); publish('navigate', 'blind'); },
  'use-format': el => { sel().format = el.dataset.format; saveRecordingSettings(sel()); inspect(); },
  'apply-calibration': () => {
    Object.assign(sel(), {power_offset_db: files().info.extra['vectorrf:full_scale_dbm'], power_unit: 'dBm'});
    saveRecordingSettings(sel());
    refreshOverview().then(() => renderMain());
    applyPlayback();
  },
  'clear-region': () => { region = null; renderMain(); },
  export: () => {
    const c = state.exportCfg, info = files().info;
    const spec = region && info?.sample_rate ? recordingSpec({start_sample: Math.round(region.t0 * info.sample_rate)}) : recordingSpec();
    const samples = region && info?.sample_rate ? Math.max(32, Math.round((region.t1 - region.t0) * info.sample_rate)) : Math.round(Number(c.excerpt_samples) || spec.capture_samples);
    runExport({source: spec, samples, format: c.format, container: c.container, scaling: c.scaling, headroom_db: Number(c.headroom_db), destination: c.destination}, 'recording excerpt');
  },
  delete: async () => {
    const f = sel();
    if (!f || !(await confirmDialog('Delete recording', `Permanently delete ${f.name} from ${files().directory}?`, 'Delete', true))) return;
    try {
      if (isLive(f.name)) switchSource('transmitter');
      await del(`/api/files/${encodeURIComponent(f.name)}`);
      files().selected = null;
      files().info = null;
      files().overview = null;
      toast(`Deleted ${f.name}`);
      await refreshLibrary();
    } catch (error) {
      toast(errorText(error), 'bad');
    }
  },
};

/* ------------------------------------------------------------ lifecycle -- */

// Exports elsewhere can add files: rescan on the next visit (or now, if open).
subscribe('library:changed', name => {
  if (!state.ready) return;
  files().loaded = false;
  if (state.mode === 'files') refreshLibrary(name);
});

export function start() {
  unsubscribe = [subscribe('live:metrics', () => { if (isLive(sel()?.name)) draw(); })];
  setStatus(activeFile() ? 'recording streaming' : 'transmitter streaming');
}

export function stop() {
  for (const off of unsubscribe) off();
  unsubscribe = [];
}

export function resize() { draw(); }

document.addEventListener('click', event => {
  if (state.mode !== 'files') return;
  const tab = event.target.closest('#recMain [data-recpage]');
  if (tab) { files().page = tab.dataset.recpage; renderMain(); }
});
