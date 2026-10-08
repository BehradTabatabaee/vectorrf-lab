/* Radar Studio: waveform design, scene, array, processing and live measurements. */
import {$, esc, replaceHTML, setText} from '../core/dom.js';
import {post, errorText} from '../core/api.js';
import {bindControls, numberField, pane, readout, section, segmented, selectField, sliderField, textField, textPane, toggleField, views} from '../core/controls.js';
import {publish} from '../core/events.js';
import {fmt, fmtCount, formatQuantity, parseQuantity} from '../core/format.js';
import {clone, getPath, state} from '../core/state.js';
import {designRequest, recordingSpec} from '../core/source.js';
import {renderStatus, setStatus, toast} from '../core/ui.js';
import {pipelineActions, pipelineEditor} from '../editors/pipeline.js';
import {programActions, programEditor} from '../editors/program.js';
import {complexPairs, numberList, structuredActions, vectors3} from '../editors/structured.js';
import {clear, colors, heatmap, linePlot, setup} from '../plots/canvas.js';
import {SceneMap, TARGET_COLORS, setStateAt, stateAt} from '../plots/scene.js';

const C = 299_792_458;
const PULSES = [['pulse', 'Rectangular pulse'], ['lfm', 'LFM chirp'], ['nlfm', 'NLFM chirp'], ['barker', 'Barker code'], ['frank', 'Frank code'], ['p1', 'P1'], ['p2', 'P2'],
  ['p3', 'P3'], ['p4', 'P4'], ['zadoff-chu', 'Zadoff-Chu'], ['custom-phase', 'Custom phase code'], ['costas', 'Costas hops'], ['stepped-frequency', 'Stepped frequency']];
const SCOPES = [['tx', 'Transmitted I/Q'], ['frequency', 'Transmit instantaneous frequency'], ['rx', 'Received I/Q (beam)'], ['processed', 'Processing probe'],
  ['spectrum', 'Received spectrum'], ['range', 'Range profile / threshold'], ['rd', 'Range–Doppler'], ['angles', 'Angle response'], ['pattern', 'Array pattern'],
  ['autocorr', 'Transmit autocorrelation']];

let root = null;
let timer = null;
let debounce = null;
let busy = false;
let revision = 0;
let data = null;
let error = '';
let computeMs = 0;
let lastDone = 0;
let rate = 0;
let scene = null;

export const usesLive = false;
export function payload() { return {}; }

const s = () => state.studio;
const design = () => state.studio.legacy;
/** Simulation time of the capture on screen: target edits apply to the state at this instant. */
const shownTime = () => (data ? data.time_s : s().time_s);

/* -------------------------------------------------------------- request -- */

function request() {
  const st = s(), body = {};
  for (const key of Object.keys(state.defaults.radar_experiment)) body[key] = clone(st[key]);
  body.legacy = {...clone(st.legacy), sample_rate: st.sample_rate};
  body.ofdm = {...designRequest(), waveform: 'ofdm', snr_db: null, components: [], agility: clone(state.defaults.agility), frame_enabled: false,
    burst_enabled: false, sample_rate: st.sample_rate};
  body.file = st.source === 'file' ? recordingSpec({capture_samples: st.file_samples, loop: true}) : null;
  return body;
}

async function tick(force = false, advance = true) {
  const st = s();
  if (state.mode !== 'radar' || document.hidden || busy || (!st.running && !force)) return;
  if (!force && performance.now() - lastDone < 1000 / st.fps) return;
  if (st.source === 'file' && !state.files.selected) { error = 'Choose a recording in Recordings for the file waveform source.'; drawAll(); return; }
  busy = true;
  const mine = revision, started = performance.now();
  try {
    const d = await post('/api/studio/radar', request());
    if (mine !== revision) return;
    data = d;
    error = '';
    computeMs = performance.now() - started;
    rate = lastDone ? 1000 / (performance.now() - lastDone) : 0;
    lastDone = performance.now();
    if (advance) {
      st.capture_id += 1;
      st.time_s += st.samples / st.sample_rate * st.speed;
      d.truth.forEach((t, i) => { const trail = (st.trails[i] ||= []); trail.push([t.position[0], t.position[1]]); if (trail.length > 400) trail.shift(); });
    }
    drawAll();
    setStatus(`radar · capture ${d.capture_id} · t ${fmt(d.time_s, 's')}`);
  } catch (e) {
    if (mine !== revision) return;
    error = errorText(e);
    drawAll();
    setStatus('radar design invalid', true);
  } finally {
    busy = false;
  }
}

function changed(path = '', structural = false) {
  revision += 1;
  if (path.startsWith('studio.targets')) s().trails = {};
  if (structural) renderPage();
  clearTimeout(debounce);
  debounce = setTimeout(() => tick(true, false), 160);
}

/* ----------------------------------------------------------------- dock -- */

function dockHTML() {
  const st = s(), lam = C / st.carrier_hz;
  const sim = `<div class="sim-controls"><button type="button" class="btn ${st.running ? 'on' : 'primary'}" data-action="run">${st.running ? '❚❚ Pause' : '▶ Run'}</button><button type="button" class="btn" data-action="step">Step</button><button type="button" class="btn" data-action="reset">Reset</button></div>`
    + numberField('Time advance', 'studio.speed', {unit: '×', min: 0.01, max: 1000, help: 'Simulated seconds advance by capture duration × this factor per frame, so targets move visibly.'})
    + numberField('Sample rate', 'studio.sample_rate', {unit: 'S/s', min: 2000, max: state.limits.max_sample_rate, help: 'The one sample clock of the experiment: the waveform design, echoes and processing all run at it.'})
    + numberField('Samples / capture', 'studio.samples', {min: 512, max: 262144, integer: true, help: `${fmt(st.samples / st.sample_rate, 's')} per capture.`})
    + numberField('RF carrier', 'studio.carrier_hz', {unit: 'Hz', min: 1e6, max: 1e11, help: `λ = ${fmt(lam, 'm')}; sets Doppler and array phase.`})
    + numberField('Update target', 'studio.fps', {unit: 'Hz', min: 1, max: 60});
  const source = selectField('Waveform', 'studio.source', [['legacy', 'Pulses / CW / FMCW / segments'], ['program', 'Arbitrary waveform'], ['ofdm', 'OFDM (Transmitter design)'], ['file', 'Recording window']])
    + (st.source === 'file' ? numberField('Window', 'studio.file_samples', {min: 64, max: state.limits.max_capture_samples, integer: true, help: `Repeats a window of ${esc(state.files.selected?.name || 'the selected recording')} as the transmit waveform.`}) : '');
  const propagation = selectField('Power model', 'studio.power_model', [['relative', 'Relative · R⁻⁴'], ['radar-equation', 'Monostatic radar equation']])
    + (st.power_model === 'relative' ? sliderField('Receiver noise', 'studio.noise_db', {min: -120, max: -5, step: 1, unit: 'dB'})
      : numberField('Transmit power', 'studio.power_w', {unit: 'W', min: 1e-3, max: 1e9}) + numberField('TX gain', 'studio.tx_gain_db', {unit: 'dBi'})
        + numberField('RX gain', 'studio.rx_gain_db', {unit: 'dBi'}) + numberField('Noise figure', 'studio.noise_figure_db', {unit: 'dB', min: 0, max: 40}))
    + numberField('System losses', 'studio.loss_db', {unit: 'dB', min: 0, max: 100})
    + numberField('TX leakage', 'studio.leakage_db', {unit: 'dBc', min: -300, max: 0, nullable: true, placeholder: 'off',
      help: 'Transmit signal coupled straight into the receiver, relative to the transmitted signal (dBc). Empty = off.'})
    + numberField('Diffuse clutter', 'studio.clutter_db', {unit: 'dBc', min: -300, max: 0, nullable: true, placeholder: 'off',
      help: 'Noise-like clutter common to all elements, relative to the transmitted signal (dBc). Empty = off.'});
  return section('Simulation', sim, {id: 'r-sim'}) + section('Waveform source', source, {id: 'r-source'})
    + section('Propagation', propagation, {id: 'r-prop', open: st.page === 'scene', help: 'Narrowband far-field point targets with delay, Doppler, element and array gain. Relative mode scales echoes as R⁻⁴ from 3 km; the radar equation uses absolute power, gains, RCS and kTB·NF noise.'})
    + section('Search and detection', detectionHTML(), {id: 'r-detect', open: ['processing', 'live'].includes(st.page)});
}

const DETECTORS = [['ca-cfar', 'CA-CFAR · cell averaging'], ['go-cfar', 'GO-CFAR · greatest of'], ['so-cfar', 'SO-CFAR · smallest of'],
  ['os-cfar', 'OS-CFAR · ordered statistic'], ['manual', 'Fixed threshold']];
const DETECTOR_HELP = 'Applied along range in every Doppler row; above-threshold cells are grouped and each group is one detection. '
  + 'CA averages the training cells (best in uniform noise); GO takes the greater side (holds the false-alarm rate at clutter edges); '
  + 'SO the smaller side (resolves close targets); OS a chosen rank of the sorted cells (tolerates interfering targets).';

function detectionHTML() {
  const p = 'studio.processing', d = s().processing;
  return numberField('Maximum range', 'studio.max_range_m', {unit: 'm', min: 100, max: 1e7}) + numberField('Velocity span ±', 'studio.max_velocity_mps', {unit: 'm/s', min: 1, max: 3000})
    + numberField('Doppler bins', 'studio.doppler_bins', {min: 9, max: 81, integer: true, step: 2})
    + numberField('Minimum range', `${p}.min_range_m`, {unit: 'm', min: 0})
    + selectField('Detector', `${p}.detector`, DETECTORS, {help: DETECTOR_HELP})
    + (d.detector === 'manual' ? numberField('Threshold', `${p}.threshold_db`, {unit: 'dB', min: -300, max: 100})
      : numberField('Training cells', 'studio.cfar_train', {min: 2, max: 64, integer: true, help: 'On each side of the cell under test, along range.'})
        + numberField('Guard cells', 'studio.cfar_guard', {min: 1, max: 20, integer: true, help: 'Skipped on each side; widened automatically to the compressed mainlobe.'})
        + numberField('False-alarm probability', 'studio.cfar_pfa', {min: 1e-9, max: 0.49, help: 'Per cell, for noise-like cells: the threshold multiplier is solved exactly for it.'})
        + (d.detector === 'os-cfar' ? sliderField('Order statistic', `${p}.os_rank`, {min: 0.05, max: 1, step: 0.05, unit: '', help: 'Rank of the training cell that sets the noise level, as a fraction of the cells (0.75: the third quartile).'}) : ''));
}

/* ----------------------------------------------------------- waveform page -- */

function itemDuration(item) {
  return item.kind === 'off' ? item.off_time_s * item.repeat : item.pri_s * item.repeat;
}

function pulseFields(p, pulse) {
  const w = pulse.waveform;
  let out = selectField('Waveform', `${p}.waveform`, PULSES) + numberField('Width', `${p}.width_s`, {unit: 's', min: 1e-8, max: 0.5})
    + selectField('Envelope', `${p}.envelope`, [['rect', 'Rectangular'], ['gaussian', 'Gaussian'], ['raised-cosine', 'Raised cosine']])
    + numberField('Amplitude', `${p}.amplitude`, {min: 0, max: 2}) + numberField('Frequency offset', `${p}.frequency_offset_hz`, {unit: 'Hz'}) + numberField('Phase', `${p}.phase_offset_deg`, {unit: '°'});
  if (['lfm', 'nlfm', 'costas'].includes(w)) out += numberField('Sweep bandwidth', `${p}.sweep_bw_hz`, {unit: 'Hz', min: 1});
  if (w === 'nlfm') out += numberField('NLFM exponent', `${p}.nlfm_exponent`, {min: 1, max: 9});
  if (w === 'barker') out += selectField('Barker length', `${p}.barker_length`, [2, 3, 4, 5, 7, 11, 13].map(n => [n, String(n)]));
  if (['frank', 'p1', 'p2'].includes(w)) out += numberField('Polyphase order', `${p}.polyphase_order`, {min: 2, max: 32, integer: true});
  if (['p3', 'p4', 'zadoff-chu'].includes(w)) out += numberField('Code chips', `${p}.phase_code_chips`, {min: 2, max: 4096, integer: true});
  if (w === 'zadoff-chu') out += numberField('ZC root', `${p}.zc_root`, {min: 1, max: 4095, integer: true});
  if (w === 'custom-phase') out += numberList('Phase code', `${p}.custom_phase_deg`, {unit: '°', minLength: 2});
  if (w === 'costas') out += numberField('Costas hops', `${p}.costas_hops`, {min: 3, max: 64, integer: true});
  if (w === 'stepped-frequency') out += numberField('Steps', `${p}.stepped_steps`, {min: 2, max: 256, integer: true}) + numberField('Step spacing', `${p}.stepped_spacing_hz`, {unit: 'Hz', min: 1});
  return out;
}

function itemEditor() {
  const seq = design().sequence;
  const i = Math.min(s().selectedItem, seq.length - 1);
  s().selectedItem = Math.max(0, i);
  const item = seq[s().selectedItem], p = `studio.legacy.sequence.${s().selectedItem}`;
  if (!item) return '';
  if (item.kind === 'off') {
    return section(`Item ${s().selectedItem + 1} · off time`, textField('Name', `${p}.name`) + numberField('Off time', `${p}.off_time_s`, {unit: 's', min: 1e-8, max: 2})
      + numberField('Repeat', `${p}.repeat`, {min: 1, max: 100000, integer: true}), {id: 'r-item'});
  }
  return section(`Item ${s().selectedItem + 1} · pulse`, textField('Name', `${p}.name`) + pulseFields(`${p}.pulse`, item.pulse), {id: 'r-item'})
    + section('Repetition and timing', numberField('Repeat', `${p}.repeat`, {min: 1, max: 100000, integer: true}) + numberField('PRI', `${p}.pri_s`, {unit: 's', min: 1e-8, max: 2})
      + numberField('PRI jitter', `${p}.pri_jitter_fraction`, {min: 0, max: 0.8, help: 'Uniform random fraction of the PRI.'})
      + numberField('PRI wobble', `${p}.pri_wobble_fraction`, {min: 0, max: 0.8}) + numberField('Wobble period', `${p}.pri_wobble_period`, {min: 2, max: 4096, integer: true}), {id: 'r-timing'})
    + section('Item offsets', numberField('Scale', `${p}.scale`, {min: 0, max: 2}) + numberField('Frequency offset', `${p}.frequency_offset_hz`, {unit: 'Hz'}) + numberField('Phase', `${p}.phase_offset_deg`, {unit: '°'}), {id: 'r-offsets', open: false});
}

function figuresHTML() {
  const d = data?.design;
  if (!d) return '';
  if (d.mode !== 'sequence') {
    const rows = [['Bandwidth', fmt(d.bandwidth_hz, 'Hz')], ['Range resolution', fmt(d.range_resolution_m, 'm')], ['Period', fmt(d.period_s, 's')],
      ['Unambiguous range', fmt(d.unambiguous_range_m, 'm')], ['Unambiguous velocity', `±${fmt(d.unambiguous_velocity_mps, 'm/s')}`]];
    return `<div class="figure-grid">${rows.map(([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}</div>`;
  }
  const item = d.items[s().selectedItem];
  const rows = [['Sequence', fmt(d.sequence_s, 's')], ['Average duty', `${(100 * d.duty).toFixed(2)} %`], ['Wavelength', fmt(d.wavelength_m, 'm')]];
  if (item?.kind === 'pulse') {
    rows.push(['PRF', fmt(item.prf_hz, 'Hz')], ['Duty', `${(100 * item.duty).toFixed(2)} %`], ['Bandwidth', fmt(item.bandwidth_hz, 'Hz')],
      ['Compression', `${item.compression_ratio.toFixed(1)} × (${(10 * Math.log10(Math.max(1, item.compression_ratio))).toFixed(1)} dB)`],
      ['Range resolution', fmt(item.range_resolution_m, 'm')], ['Blind range', fmt(item.blind_range_m, 'm')],
      ['Unambiguous range', fmt(item.unambiguous_range_m, 'm')], ['Unambiguous velocity', `±${fmt(item.unambiguous_velocity_mps, 'm/s')}`],
      ['Velocity resolution', fmt(item.velocity_resolution_mps, 'm/s')]);
  }
  return `<div class="figure-grid">${rows.map(([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}</div>`;
}

function legacyEditor() {
  const d = design();
  let body = segmented('studio.legacy.mode', [['sequence', 'Pulse sequence'], ['continuous', 'Continuous'], ['segments', 'Segments']], {label: 'Design mode'});
  if (d.mode === 'sequence') {
    const rows = d.sequence.map((item, i) => `<tr class="clickable${i === s().selectedItem ? ' selected' : ''}" data-action="select-item" data-index="${i}"><td class="num-cell">${i + 1}</td><td>${esc(item.name)}</td><td>${item.kind === 'off' ? 'off' : esc(item.pulse.waveform)}</td><td class="num-cell">${item.repeat}</td><td class="num-cell">${esc(fmt(item.kind === 'off' ? item.off_time_s : item.pri_s, 's'))}</td><td class="num-cell">${esc(fmt(itemDuration(item), 's'))}</td></tr>`).join('');
    body += `<div class="sequence-timeline"><canvas id="seqTimeline" aria-label="Sequence timeline; click an item to edit it"></canvas></div>`
      + `<table class="table"><thead><tr><th>#</th><th>Name</th><th>Type</th><th>Repeat</th><th>PRI / time</th><th>Duration</th></tr></thead><tbody>${rows}</tbody></table>`
      + `<div class="btn-row"><button class="btn small" data-action="add-pulse">+ Pulse</button><button class="btn small" data-action="add-off">+ Off time</button><button class="btn small" data-action="move-item" data-dir="-1">↑</button><button class="btn small" data-action="move-item" data-dir="1">↓</button><button class="btn small" data-action="duplicate-item">Duplicate</button><button class="btn small danger" data-action="remove-item">Remove</button></div>`
      + '<div id="rFigures"></div>' + itemEditor();
    return section('Pulse sequence', body, {id: 'r-sequence', help: 'Items play in order and the whole sequence repeats. Engineering figures refer to the selected item at the RF carrier; jitter and wobble change individual PRIs around the nominal value.'});
  }
  if (d.mode === 'continuous') {
    const w = d.continuous_waveform;
    body += selectField('Waveform', 'studio.legacy.continuous_waveform', [['cw', 'CW'], ['fmcw', 'FMCW'], ['pmcw', 'PMCW (binary code)'], ['stepped-cw', 'Stepped CW']])
      + numberField('Centre offset', 'studio.legacy.center_offset_hz', {unit: 'Hz'})
      + (w === 'fmcw' ? numberField('Sweep bandwidth', 'studio.legacy.sweep_bw_hz', {unit: 'Hz', min: 1}) + numberField('Sweep period', 'studio.legacy.sweep_period_s', {unit: 's', min: 1e-8})
        + selectField('Shape', 'studio.legacy.sweep_shape', [['up', 'Up'], ['down', 'Down'], ['triangle', 'Triangle']]) : '')
      + (w === 'pmcw' ? numberField('Chip rate', 'studio.legacy.pmcw_chip_rate', {unit: 'Hz', min: 10}) + numberField('Code length', 'studio.legacy.pmcw_chips', {min: 2, integer: true}) : '')
      + (w === 'stepped-cw' ? numberField('Start', 'studio.legacy.stepped_start_hz', {unit: 'Hz'}) + numberField('Stop', 'studio.legacy.stepped_stop_hz', {unit: 'Hz'})
        + numberField('Points', 'studio.legacy.stepped_points', {min: 2, integer: true}) + numberField('Dwell', 'studio.legacy.stepped_dwell_s', {unit: 's', min: 1e-8}) : '')
      + numberField('Record length', 'studio.legacy.duration_s', {unit: 's', min: 1e-5, max: 1}) + '<div id="rFigures"></div>';
    return section('Continuous waveform', body, {id: 'r-continuous'});
  }
  const rows = d.segments.map((seg, i) => `<div class="card"><header>Segment ${i + 1}<span class="tools"><button type="button" class="btn small ghost" data-action="remove-segment" data-index="${i}" aria-label="Remove segment">×</button></span></header><div class="card-body">`
    + numberField('Duration', `studio.legacy.segments.${i}.duration_us`, {min: 0.01, max: 1e5, help: 'Microseconds.'}) + numberField('Start', `studio.legacy.segments.${i}.start_hz`, {unit: 'Hz'})
    + numberField('Stop', `studio.legacy.segments.${i}.stop_hz`, {unit: 'Hz'}) + numberField('Amplitude', `studio.legacy.segments.${i}.amplitude`, {min: 0, max: 2})
    + numberField('Phase jump', `studio.legacy.segments.${i}.phase_deg`, {unit: '°'}) + '</div></div>').join('');
  return section('Segments', body + rows + `<div class="btn-row"><button class="btn small" data-action="add-segment">+ Segment</button></div><div id="rFigures"></div>`, {id: 'r-segments',
    help: 'Constant-amplitude segments with linear frequency sweeps and phase jumps; phase accumulates through the sequence, which repeats.'});
}

function waveformEditor() {
  const st = s();
  if (st.source === 'legacy') return legacyEditor();
  if (st.source === 'program') return programEditor('studio.program');
  if (st.source === 'ofdm') return section('OFDM sensing', '<p class="note">The Transmitter’s OFDM resource grid, CP sequence, pilots and windowing drive the radar (joint radar/communications experiments).</p><button class="btn" data-action="open-ofdm">Open the OFDM editor</button>', {id: 'r-ofdm'});
  return section('Recording as transmit waveform', `<p class="note">${state.files.selected ? `One ${fmtCount(st.file_samples)}-sample window of ${esc(state.files.selected.name)} repeats as the transmitted signal. Match the simulation rate to the recording rate to avoid interpolation.` : 'Choose a recording in Recordings first.'}</p>`, {id: 'r-file'});
}

/* ----------------------------------------------------------------- scene -- */

function targetEditor() {
  const st = s(), i = Math.min(st.selectedTarget, st.targets.length - 1);
  st.selectedTarget = Math.max(0, i);
  const t = st.targets[st.selectedTarget], p = `studio.targets.${st.selectedTarget}`;
  if (!t) return '<p class="note">No targets. Double-click the scene to add one.</p>';
  const now = stateAt(t, shownTime());
  // Vectors show (and edit) the state at the displayed time: editing one keeps the others.
  const vec = (label, key, unit) => `<div class="field wide"><label>${esc(label)}</label><div class="vec3">${['x', 'y', 'z'].map((axis, j) =>
    `<input type="text" inputmode="decimal" spellcheck="false" data-change-action="target-vector" data-key="${key}" data-axis="${j}" data-unit="${unit}" value="${esc(formatQuantity(roundShown(now[key][j]), unit))}" aria-label="${esc(label)} ${axis}">`).join('')}</div></div>`;
  return section(`${t.name}`, textField('Name', `${p}.name`) + vec('Position now · x / y / z', 'position', 'm') + vec('Velocity now · x / y / z', 'velocity', 'm/s') + vec('Acceleration · x / y / z', 'acceleration', 'm/s²')
    + numberField('RCS', `${p}.rcs_m2`, {unit: 'm²', min: 1e-4, max: 1e6}) + numberField('Extra echo gain', `${p}.gain_db`, {unit: 'dB', min: -100, max: 30})
    + selectField('Fluctuation', `${p}.fluctuation`, [['none', 'None'], ['swerling1', 'Swerling I (per capture)']])
    + '<div class="btn-row"><button class="btn small danger" data-action="remove-target">Remove target</button></div>', {id: 'r-target'});
}

/** Display rounding of a kinematic value (6 significant digits; hides float noise from re-anchoring). */
function roundShown(v) {
  return Math.abs(v) < 1e-9 ? 0 : Number(v.toPrecision(6));
}

/** Refresh the editor's vector fields to the displayed time (not the one being typed in). */
function refreshTargetFields() {
  const t = s().targets[s().selectedTarget];
  if (!t || !root) return;
  const now = stateAt(t, shownTime());
  for (const input of root.querySelectorAll('[data-change-action="target-vector"]')) {
    if (input === document.activeElement) continue;
    input.value = formatQuantity(roundShown(now[input.dataset.key][Number(input.dataset.axis)]), input.dataset.unit);
  }
  const list = root.querySelector('.target-list');
  if (list) list.querySelectorAll('[data-index]').forEach(item => {
    const i = Number(item.dataset.index), target = s().targets[i], meta = item.querySelector('.meta');
    if (!target || !meta) return;
    const st = stateAt(target, shownTime()), range = Math.hypot(...st.position);
    const radial = range > 0 ? st.velocity.reduce((acc, v, k) => acc + v * st.position[k], 0) / range : 0;
    meta.textContent = `${fmt(range, 'm')} · ${radial.toFixed(1)} m/s · ${(Math.atan2(st.position[1], st.position[0]) * 180 / Math.PI).toFixed(1)}°`;
  });
}

function targetList() {
  const st = s();
  return `<div class="list target-list">${st.targets.map((t, i) => {
    return `<div class="list-item${i === st.selectedTarget ? ' active' : ''}" data-action="select-target" data-index="${i}"><span class="dot" style="background:${TARGET_COLORS[i % TARGET_COLORS.length]}"></span>`
      + `<span class="grow">${esc(t.name)}</span><span class="meta" style="font:11px var(--mono);color:var(--muted)"></span></div>`;
  }).join('')}</div><div class="btn-row"><button class="btn small" data-action="add-target">+ Target</button></div>`;
}

/* --------------------------------------------------------------- antenna -- */

function antennaEditor() {
  const a = s().antenna, p = 'studio.antenna';
  return section('Geometry', selectField('Layout', `${p}.geometry`, [['linear', 'Linear'], ['planar', 'Planar'], ['custom', 'Custom positions']])
      + (a.geometry === 'custom' ? vectors3('Element positions (λ)', `${p}.positions`)
        : numberField('Columns', `${p}.columns`, {min: 1, max: 16, integer: true}) + (a.geometry === 'planar' ? numberField('Rows', `${p}.rows`, {min: 1, max: 8, integer: true}) : '')
          + numberField('Spacing', `${p}.spacing_lambda`, {unit: 'λ', min: 0.05, max: 4})), {id: 'r-geometry'})
    + section('Element pattern', selectField('Pattern', `${p}.element`, [['isotropic', 'Isotropic'], ['cosine', 'Cosine power'], ['gaussian', 'Gaussian'], ['table', 'Table']])
      + (a.element === 'cosine' ? numberField('Exponent', `${p}.cosine_power`, {min: 0, max: 20}) : '')
      + (a.element === 'gaussian' ? numberField('Half-power width', `${p}.beamwidth_deg`, {unit: '°', min: 1, max: 360}) : '')
      + (a.element === 'table' ? complexPairs('Pattern table', `${p}.pattern_angles`, `${p}.pattern_db`, {labels: ['angle °', 'level dB'], minLength: 2}) : ''), {id: 'r-element'})
    + section('Steering and weights', numberField('Steer azimuth', `${p}.steer_az_deg`, {unit: '°', min: -90, max: 90}) + numberField('Steer elevation', `${p}.steer_el_deg`, {unit: '°', min: -60, max: 60})
      + numberField('Scan rate', `${p}.scan_deg_s`, {unit: '°/s', min: -180, max: 180}) + selectField('Beamformer', `${p}.beamformer`, [['conventional', 'Conventional'], ['mvdr', 'MVDR (loaded covariance)']])
      + complexPairs('Element weights', `${p}.weights_i`, `${p}.weights_q`, {help: 'empty = uniform'}), {id: 'r-steer',
      help: 'Azimuth and elevation are estimated from the strongest range/Doppler cell across the array. A linear array cannot resolve elevation; spacing above λ/2 produces grating lobes.'});
}

function processingEditor() {
  const p = 'studio.processing', d = s().processing;
  return section('Processor', selectField('Method', `${p}.method`, [['correlation', 'Delay/Doppler correlation (any waveform)'], ['pulse-doppler', 'Pulse-Doppler (matched filter + FFT)'], ['fmcw', 'FMCW (dechirp + FFT)']])
      + (d.method === 'correlation' ? '' : numberField('PRI / chirp interval', `${p}.pri_s`, {unit: 's', min: 1e-7})
        + (d.method === 'pulse-doppler' ? numberField('Pulse width', `${p}.pulse_width_s`, {unit: 's', min: 1e-8}) : numberField('Chirp slope', `${p}.chirp_slope_hz_s`, {unit: 'Hz/s'}))
        + selectField('Clutter cancellation', `${p}.clutter`, [['off', 'Off'], ['mean', 'Subtract coherent mean'], ['two-pulse', 'Two-pulse MTI'], ['three-pulse', 'Three-pulse MTI']])
        + selectField('Range window', `${p}.range_window`, ['boxcar', 'hann', 'hamming', 'blackman']) + selectField('Doppler window', `${p}.doppler_window`, ['boxcar', 'hann', 'hamming', 'blackman'])),
    {id: 'r-processor', help: 'Coherent processors need at least four complete intervals per capture at the configured PRI; correlation works for any waveform, including irregular timing.'})
    + pipelineEditor('studio.pipeline', {stages: data?.stages || []});
}

/* ------------------------------------------------------------------ pages -- */

function readoutsHTML() {
  return `<div class="readout-bar">${readout('Capture', 'rdCapture')}${readout('Sim time', 'rdTime')}${readout('Compute', 'rdCompute')}${readout('Peak range', 'rdRange', '—', 'accent')}${readout('Radial velocity', 'rdVel')}${readout('Az / El', 'rdAngle')}${readout('Detections', 'rdDet', '—', 'grow')}</div>`;
}

function pageHTML() {
  const page = s().page;
  const tabs = views([['waveform', 'Waveform'], ['scene', 'Scene'], ['antenna', 'Antenna / array'], ['processing', 'Processing'], ['live', 'Live measurements']], page, 'rpage');
  const err = `<div class="callout bad" id="radarError" style="margin:8px 8px 0"${error ? '' : ' hidden'}>${esc(error)}</div>`;
  const editor = html => `<div class="scroll-page" id="radarEditor" data-keep-scroll="radar-${page}" style="border-right:1px solid var(--line)"><div class="page" style="max-width:none">${html}</div></div>`;
  if (page === 'waveform') {
    return tabs + readoutsHTML() + err + `<div class="stage split" style="padding:0;gap:0">${editor(waveformEditor())}<div class="stage quad">${pane('Transmitted I/Q', 'rTx')}${pane('Instantaneous frequency', 'rFreq')}${pane('Autocorrelation', 'rAuto')}${pane('Transmit spectrum', 'rSpec')}</div></div>`;
  }
  if (page === 'scene') {
    return tabs + readoutsHTML() + err + `<div class="stage split" style="grid-template-columns:minmax(0,1.6fr) minmax(300px,.7fr)"><div class="stage" style="padding:0;grid-template-rows:minmax(0,2fr) minmax(0,1fr)">`
      + `<section class="pane"><header class="pane-head"><h3>Scene · top view · drag targets or their arrows · double-click to add</h3></header><div class="pane-body scene-wrap"><canvas id="rScene"></canvas>`
      + '<div class="scene-legend" style="top:auto;bottom:6px"><span><i style="background:#e9b04b"></i>beam &amp; estimate</span><span><i style="background:#ec7b72"></i>detected ranges</span></div></div></section>'
      + `${pane('Range profile and threshold', 'rRange')}</div><div class="scroll-page" data-keep-scroll="radar-targets" style="border:1px solid var(--line-2);border-radius:var(--radius);background:var(--panel)"><div class="page" style="padding:10px">${targetList()}${targetEditor()}</div></div></div>`;
  }
  if (page === 'antenna') return tabs + readoutsHTML() + err + `<div class="stage split" style="padding:0;gap:0">${editor(antennaEditor())}<div class="stage" style="grid-template-rows:1fr 1fr">${pane('Array pattern · current steering', 'rPattern')}${pane('Measured angle response', 'rAngles')}</div></div>`;
  if (page === 'processing') return tabs + readoutsHTML() + err + `<div class="stage split" style="padding:0;gap:0">${editor(processingEditor())}<div class="stage" style="grid-template-rows:repeat(3,minmax(150px,1fr))">${pane('Processing probe I/Q', 'rProbe')}${pane('Processed spectrum', 'rPSpec')}${pane('Range profile and threshold', 'rRange')}</div></div>`;
  const plots = s().plots.map((kind, i) => `<section class="pane"><header class="pane-head"><select data-bind="studio.plots.${i}" data-kind="select" aria-label="Scope ${i + 1}" style="width:auto;height:22px;font-size:11px">${SCOPES.map(([k, l]) => `<option value="${k}"${k === kind ? ' selected' : ''}>${esc(l)}</option>`).join('')}</select><span class="pane-value" id="rScope${i}Value"></span></header><div class="pane-body"><canvas id="rScope${i}"></canvas></div></section>`).join('');
  return tabs + readoutsHTML() + err + `<div class="stage split" style="grid-template-columns:minmax(0,1.6fr) minmax(280px,.6fr)"><div class="stage quad" style="padding:0">${plots}</div>${textPane('Detections, truth and model notes', 'rResults')}</div>`;
}

export function render(el) {
  root = el;
  el.innerHTML = `<div class="instrument"><aside class="dock"><div class="dock-head"><h2>Radar Studio</h2></div><div class="dock-scroll" data-keep-scroll="radar-dock"></div></aside><section class="main" id="radarMain"></section></div>`;
  bindControls(el, {onChange, actions});
  renderDock();
  renderPage();
}

function renderDock() {
  replaceHTML($('.dock-scroll', root), dockHTML());
}

function renderPage() {
  replaceHTML($('#radarMain', root), pageHTML());
  const canvas = $('#rScene', root);
  scene = canvas ? new SceneMap(canvas, {
    onSelect: i => { if (s().selectedTarget !== i) { s().selectedTarget = i; renderTargets(); } },
    onMove: (i, xy, done) => {
      const t = s().targets[i], time = shownTime();
      if (xy) { setStateAt(t, time, {position: [xy[0], xy[1], stateAt(t, time).position[2]]}); drawScene(); refreshTargetFields(); }
      if (done) { changed('studio.targets'); renderTargets(); }
    },
    onVelocity: (i, v, done) => {
      const t = s().targets[i], time = shownTime();
      if (v) { setStateAt(t, time, {velocity: [v[0], v[1], stateAt(t, time).velocity[2]]}); drawScene(); refreshTargetFields(); }
      if (done) { changed('studio.targets'); renderTargets(); }
    },
    onAdd: (x, y) => addTarget([x, y, 0]),
  }) : null;
  if (window.vectorrf) window.vectorrf.scene = scene;  // inspection handle for the browser tests
  drawAll();
}

function renderTargets() {
  const host = $('[data-keep-scroll="radar-targets"] .page', root);
  if (host) replaceHTML(host, targetList() + targetEditor());
  drawScene();
  refreshTargetFields();
}

/* ---------------------------------------------------------------- drawing -- */

function drawScene() {
  if (!scene) return;
  const st = s();
  scene.draw({targets: st.targets, selected: st.selectedTarget, data, maxRange: st.max_range_m, maxVelocity: st.max_velocity_mps,
    trails: st.trails, time: shownTime()});
}

/** Text clipped with an ellipsis to ``width`` pixels. */
function fitText(ctx, text, width) {
  if (ctx.measureText(text).width <= width) return text;
  let n = text.length;
  while (n > 1 && ctx.measureText(`${text.slice(0, n)}…`).width > width) n -= 1;
  return n > 1 ? `${text.slice(0, n)}…` : '';
}

/* Sequence timeline drawn to scale: a label row (name and timing of each
   item), the pulse track, and a time ruler.  Labels never overlap pulses. */
function drawTimeline() {
  const canvas = $('#seqTimeline');
  if (!canvas) return;
  const g = setup(canvas);
  if (!g) return;
  const {ctx, w, h} = g, c = colors();
  clear(ctx, w, h);
  const seq = design().sequence, total = seq.reduce((acc, item) => acc + itemDuration(item), 0) || 1;
  const labelH = 16, rulerH = 13, top = labelH + 2, bottom = h - rulerH - 2, trackH = Math.max(6, bottom - top);
  ctx.font = `10px ${c.mono}`;
  ctx.textBaseline = 'middle';
  let x = 0;
  seq.forEach((item, i) => {
    const width = itemDuration(item) / total * w;
    const selected = i === s().selectedItem;
    ctx.fillStyle = item.kind === 'off' ? c.bg : c.grid;
    ctx.fillRect(x, top, width, trackH);
    if (item.kind === 'pulse') {
      const pulseWidth = Math.max(1, item.pulse.width_s / total * w), pri = item.pri_s / total * w;
      ctx.fillStyle = selected ? c.accent : c.text;
      for (let k = 0; k < item.repeat && k * pri < width; k++) ctx.fillRect(x + k * pri, top + 3, pulseWidth, trackH - 6);
    }
    if (selected) {
      ctx.strokeStyle = c.accent;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, top + 0.5, Math.max(1, width - 1), trackH - 1);
    }
    ctx.fillStyle = c.gridMajor;
    ctx.fillRect(Math.round(x + width) - 1, 0, 1, bottom);
    const timing = item.kind === 'off' ? fmt(item.off_time_s * item.repeat, 's') : `${item.repeat} × ${fmt(item.pri_s, 's')}`;
    ctx.fillStyle = selected ? c.accent : c.text;
    ctx.textAlign = 'left';
    ctx.fillText(fitText(ctx, `${item.name} · ${timing}`, width - 8), x + 4, labelH / 2 + 1);
    x += width;
  });
  // Ruler: start, middle and end of the sequence.
  ctx.fillStyle = c.gridMajor;
  ctx.fillRect(0, bottom + 1, w, 1);
  ctx.fillStyle = c.text;
  [[0, 'left'], [0.5, 'center'], [1, 'right']].forEach(([f, align]) => {
    ctx.textAlign = align;
    ctx.fillText(fmt(f * total, 's'), f * w + (align === 'left' ? 2 : align === 'right' ? -2 : 0), h - rulerH / 2);
  });
  canvas.onclick = event => {
    const r = canvas.getBoundingClientRect(), t = (event.clientX - r.left) / r.width * total;
    let acc = 0;
    const index = seq.findIndex(item => (acc += itemDuration(item)) >= t);
    if (index >= 0) { s().selectedItem = index; renderPage(); }
  };
}

function drawScope(canvas, kind, d) {
  if (!canvas || !d) return;
  const iq = a => linePlot(canvas, [{y: a.i, x: a.time_s}, {y: a.q, x: a.time_s}], {xUnit: 's'});
  if (kind === 'tx') iq(d.tx);
  else if (kind === 'rx') iq(d.rx);
  else if (kind === 'processed') iq(d.processed);
  else if (kind === 'frequency') linePlot(canvas, d.tx_frequency_hz, {x: d.tx_time_s, xUnit: 's', yUnit: 'Hz'});
  else if (kind === 'spectrum') linePlot(canvas, d.psd_db, {x: d.freq_hz, xUnit: 'Hz', yUnit: 'dB'});
  else if (kind === 'range') linePlot(canvas, [{y: d.range_db, x: d.range_m}, {y: d.threshold_db, x: d.range_m, color: '#ec7b72', dash: [5, 4]}], {xUnit: 'm', yUnit: 'dB'});
  else if (kind === 'rd') heatmap(canvas, d.rd_db, d.rd_shape, {x: d.range_m, y: d.velocity_mps, xUnit: 'm', yUnit: 'm/s', rowsBottomUp: true,
    marks: d.truth.map(t => ({x: t.range_m, y: t.velocity_mps, color: '#ffffff', r: 5}))});
  else if (kind === 'angles') heatmap(canvas, d.angle_db, d.angle_shape, {x: d.angle_az, y: d.angle_el, xUnit: '°', yUnit: '°', rowsBottomUp: true, lo: -40, hi: 0});
  else if (kind === 'pattern') linePlot(canvas, d.pattern_db, {x: d.angle_az, xUnit: '°', yUnit: 'dB', min: -60, max: 3});
  else if (kind === 'autocorr') linePlot(canvas, d.autocorr_db, {x: d.autocorr_delay_s, xUnit: 's', yUnit: 'dB', min: -70, max: 1});
}

function drawAll() {
  if (state.mode !== 'radar' || !root) return;
  const err = $('#radarError');
  if (err) { err.hidden = !error; err.textContent = `${error}${data ? ' · showing the last valid capture' : ''}`; }
  drawTimeline();
  drawScene();
  refreshTargetFields();
  const d = data;
  if (!d) return;
  const figures = $('#rFigures');
  if (figures) figures.innerHTML = figuresHTML();
  setText('rdCapture', `#${d.capture_id}`);
  setText('rdTime', fmt(d.time_s, 's'));
  setText('rdCompute', `${computeMs.toFixed(0)} ms · ${rate.toFixed(1)} Hz`);
  setText('rdRange', fmt(d.peak_range_m, 'm'));
  setText('rdVel', fmt(d.peak_velocity_mps, 'm/s'));
  setText('rdAngle', `${d.azimuth_deg == null ? '—' : `${d.azimuth_deg.toFixed(1)}°`} / ${d.elevation_deg == null ? '—' : `${d.elevation_deg.toFixed(1)}°`}`);
  setText('rdDet', d.detections.length ? `${d.detections.length} · ${d.detections.slice(0, 3).map(x => fmt(x.range_m, 'm')).join(' · ')}` : 'none');
  drawScope($('#rTx'), 'tx', d); drawScope($('#rFreq'), 'frequency', d); drawScope($('#rAuto'), 'autocorr', d); drawScope($('#rSpec'), 'spectrum', d);
  drawScope($('#rRange'), 'range', d); drawScope($('#rPattern'), 'pattern', d); drawScope($('#rAngles'), 'angles', d);
  drawScope($('#rProbe'), 'processed', d); drawScope($('#rPSpec'), 'spectrum', d);
  s().plots.forEach((kind, i) => { drawScope($(`#rScope${i}`), kind, d); setText(`rScope${i}Value`, `capture ${d.capture_id}`); });
  const results = $('#rResults');
  if (results) {
    replaceHTML(results, `<h4 style="margin:0 0 6px">Strongest cell</h4><table class="table"><tbody><tr><td>Range</td><td class="num-cell">${esc(fmt(d.peak_range_m, 'm'))}</td></tr><tr><td>Radial velocity</td><td class="num-cell">${esc(fmt(d.peak_velocity_mps, 'm/s'))}</td></tr>`
      + `<tr><td>Azimuth / elevation</td><td class="num-cell">${esc(d.azimuth_deg?.toFixed(1) ?? '—')}° / ${esc(d.elevation_deg?.toFixed(1) ?? '—')}°</td></tr><tr><td>Estimated XYZ</td><td class="num-cell">${d.estimated_position ? d.estimated_position.map(v => esc(fmt(v, 'm'))).join(' · ') : 'needs azimuth and elevation'}</td></tr></tbody></table>`
      + `<h4 style="margin:12px 0 6px">Detections · ${esc((d.detector || '').toUpperCase())}</h4>${d.detections.length ? `<table class="table"><thead><tr><th>Range</th><th>Velocity</th><th>Power</th><th>Above threshold</th></tr></thead><tbody>${d.detections.map(x => `<tr><td class="num-cell">${esc(fmt(x.range_m, 'm'))}</td><td class="num-cell">${esc(fmt(x.velocity_mps, 'm/s'))}</td><td class="num-cell">${x.power_db.toFixed(1)} dB</td><td class="num-cell">${x.margin_db == null ? '—' : `${x.margin_db.toFixed(1)} dB`}</td></tr>`).join('')}</tbody></table>` : '<p class="note">No threshold crossings.</p>'}`
      + `<h4 style="margin:12px 0 6px">Simulation truth</h4><table class="table"><tbody>${d.truth.map(t => `<tr><td>${esc(t.name)}</td><td class="num-cell">${esc(fmt(t.range_m, 'm'))} · ${t.velocity_mps.toFixed(1)} m/s · ${t.az_deg.toFixed(1)}° · ${t.echo_power_db.toFixed(1)} dB</td></tr>`).join('')}</tbody></table>`
      + `${d.warnings.map(w => `<div class="callout warn">${esc(w)}</div>`).join('')}`);
  }
}

/* ---------------------------------------------------------------- changes -- */

const STRUCTURE = /\.(mode|continuous_waveform|waveform|kind|geometry|element|method|detector|power_model|source)$/;

function onChange(path) {
  if (path === 'studio.sample_rate') { toast('Waveform design, echoes and processing now run at the new rate'); renderStatus(); }
  if (path.startsWith('studio.plots')) { drawAll(); return; }
  if (['studio.speed', 'studio.fps'].includes(path)) return;
  if (path === 'studio.source' || path === 'studio.power_model' || path === 'studio.processing.detector') renderDock();
  changed(path, STRUCTURE.test(path) || path.endsWith('.name'));
  if (path.startsWith('studio.targets')) renderTargets();
}

function addTarget(position) {
  const st = s();
  if (st.targets.length >= 12) { toast('Up to 12 targets'); return; }
  const target = {...clone(state.defaults.radar_target), name: `Target ${st.targets.length + 1}`, acceleration: [0, 0, 0]};
  setStateAt(target, shownTime(), {position, velocity: [-20, 10, 0], acceleration: [0, 0, 0]});
  st.targets.push(target);
  st.selectedTarget = st.targets.length - 1;
  changed('studio.targets');
  renderTargets();
}

function newPulseItem() {
  const item = clone(state.defaults.radar_item);
  item.name = `Pulse ${design().sequence.filter(x => x.kind === 'pulse').length + 1}`;
  item.pulse.name = item.name;
  return item;
}

function afterStructure(path) { changed(path, true); }

const actions = {
  ...structuredActions(afterStructure),
  ...programActions(afterStructure),
  ...pipelineActions(afterStructure),
  run: () => { s().running = !s().running; renderDock(); if (s().running) tick(true); },
  step: () => { s().running = false; renderDock(); tick(true, true); },
  reset: () => { Object.assign(s(), {capture_id: 0, time_s: 0, trails: {}}); revision += 1; tick(true, false); },
  'select-item': el => { s().selectedItem = Number(el.dataset.index); renderPage(); },
  'add-pulse': () => { if (design().sequence.length >= 64) return; design().sequence.push(newPulseItem()); s().selectedItem = design().sequence.length - 1; changed('studio.legacy', true); },
  'add-off': () => {
    if (design().sequence.length >= 64) return;
    design().sequence.push({...clone(state.defaults.radar_item), kind: 'off', name: 'Off time', pulse: null, off_time_s: 100e-6});
    s().selectedItem = design().sequence.length - 1;
    changed('studio.legacy', true);
  },
  'move-item': el => {
    const a = design().sequence, i = s().selectedItem, j = i + Number(el.dataset.dir);
    if (j < 0 || j >= a.length) return;
    [a[i], a[j]] = [a[j], a[i]];
    s().selectedItem = j;
    changed('studio.legacy', true);
  },
  'duplicate-item': () => { const a = design().sequence; if (a.length >= 64) return; a.splice(s().selectedItem + 1, 0, clone(a[s().selectedItem])); s().selectedItem += 1; changed('studio.legacy', true); },
  'remove-item': () => {
    const a = design().sequence;
    if (a.length <= 1) { toast('Keep at least one sequence item'); return; }
    a.splice(s().selectedItem, 1);
    s().selectedItem = Math.max(0, s().selectedItem - 1);
    changed('studio.legacy', true);
  },
  'add-segment': () => { if (design().segments.length < 128) { design().segments.push(clone(state.defaults.radar_segment)); changed('studio.legacy', true); } },
  'remove-segment': el => { if (design().segments.length > 1) { design().segments.splice(Number(el.dataset.index), 1); changed('studio.legacy', true); } },
  'select-target': el => { s().selectedTarget = Number(el.dataset.index); renderTargets(); },
  'target-vector': el => {
    const t = s().targets[s().selectedTarget];
    if (!t) return;
    const value = parseQuantity(el.value, el.dataset.unit);
    const field = el.closest('.field');
    if (!Number.isFinite(value) || Math.abs(value) > 1e9) {
      field?.classList.add('invalid');
      toast(`Not a ${el.dataset.unit} value: ${el.value}`, 'bad');
      return;
    }
    field?.classList.remove('invalid');
    const time = shownTime(), now = stateAt(t, time), key = el.dataset.key;
    const vector = [...now[key]];
    vector[Number(el.dataset.axis)] = value;
    setStateAt(t, time, {[key]: vector});
    changed('studio.targets');
    renderTargets();
  },
  'add-target': () => addTarget([4500, -800, 300]),
  'remove-target': () => { s().targets.splice(s().selectedTarget, 1); s().selectedTarget = Math.max(0, s().selectedTarget - 1); changed('studio.targets'); renderTargets(); },
  'open-ofdm': () => {
    state.tx.waveform = 'ofdm';
    state.tx.ofdm_design.enabled = true;
    state.txUi.page = 'signal';
    publish('tx:changed');
    publish('navigate', 'tx');
  },
};

/* ------------------------------------------------------------ lifecycle -- */

export function start() {
  tick(true, false);
  clearInterval(timer);
  timer = setInterval(() => tick(), 16);
}

export function stop() {
  clearInterval(timer);
  clearTimeout(debounce);
}

export function resize() { drawAll(); }

document.addEventListener('click', event => {
  if (state.mode !== 'radar') return;
  const tab = event.target.closest('#radarMain [data-rpage]');
  if (tab) { s().page = tab.dataset.rpage; renderDock(); renderPage(); }
});
