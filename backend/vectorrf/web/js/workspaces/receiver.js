/* Receiver: live monitor, demodulation and synchronization, framing and the processing chain. */
import {$, esc, replaceHTML, setText} from '../core/dom.js';
import {bindControls, numberField, pane, readout, section, selectField, sliderField, textPane, toggleField, views} from '../core/controls.js';
import {subscribe} from '../core/events.js';
import {fmt, fmtCount, fmtFreq} from '../core/format.js';
import {update as liveUpdate} from '../core/live.js';
import {clone, state} from '../core/state.js';
import {activeFile, activeRate, appliedTx, axisOffset, powerUnit} from '../core/source.js';
import {renderStatus, setStatus, toast} from '../core/ui.js';
import {frameActions, frameBlocksEditor, newBlock} from '../editors/frame.js';
import {ofdmActions} from '../editors/ofdm.js';
import {pipelineActions, pipelineEditor} from '../editors/pipeline.js';
import {structuredActions, numberList} from '../editors/structured.js';
import {Waterfall, eye, linePlot, scatter} from '../plots/canvas.js';
import {followBadge, fullTune, sourceActions, sourceSection, spectrumRequest} from './common.js';
import {idealConstellation} from './constellations.js';

const LINEAR = new Set(['bpsk', 'qpsk', 'dbpsk', 'dqpsk', 'pi4-dqpsk', 'oqpsk', '8psk', '16qam', '64qam', '16apsk', '32apsk']);
const CPFSK = new Set(['2fsk', '4fsk', 'msk', 'gmsk']);
const RX_MODULATIONS = [...LINEAR, ...CPFSK, 'ofdm', 'am', 'fm', 'pm', 'ssb', 'dsb-sc', 'tone', 'multitone', 'awgn', 'chirp', 'custom'];
const DIFFERENTIAL = new Set(['dbpsk', 'dqpsk', 'pi4-dqpsk']);

let root = null;
let unsubscribe = [];
let animation = null;
let paintPending = false;
let data = null;      // last receiver (demodulation) product
let chain = null;     // newest product of either kind: the monitor and chain views follow it
const monitorFall = new Waterfall(512, 140);

export const usesLive = true;

const rx = () => state.rx;
const family = () => (LINEAR.has(rx().modulation) ? 'linear' : CPFSK.has(rx().modulation) ? 'cpfsk' : rx().modulation === 'ofdm' ? 'ofdm' : 'general');

function pages() {
  const f = family();
  const list = [['monitor', 'Live monitor'], ['acquire', 'Demodulation']];
  if (f === 'linear' || f === 'cpfsk') list.push(['framing', 'Framing']);
  if (f === 'linear') list.push(['timing', 'Timing'], ['carrier', 'Carrier'], ['equalizer', 'Equalizer']);
  list.push(['pipeline', 'Processing chain']);
  return list;
}

function page() {
  const p = state.rxUi.page;
  return pages().some(([k]) => k === p) ? p : 'acquire';
}

/* ------------------------------------------------------------ following -- */

/** Copy the transmitter design (or the chosen composite component) into the receiver. */
export function followTransmitter() {
  if (!state.rxUi.follow || activeFile()) return;
  const t = appliedTx(), r = rx();
  let signal = t, tuner = Number(t.center_hz) || 0;
  const schedules = t.agility?.enabled ? [t.agility] : [];
  if (t.waveform === 'composite') {
    state.rxUi.component = Math.max(0, Math.min(state.rxUi.component, t.components.length - 1));
    signal = t.components[state.rxUi.component];
    tuner += Number(signal.center_hz) || 0;
    if (signal.agility?.enabled) schedules.push(signal.agility);
  }
  r.modulation = RX_MODULATIONS.includes(signal.waveform) ? signal.waveform : 'custom';
  for (const key of ['symbol_rate', 'rolloff', 'filter_span_symbols', 'pulse_shape', 'apsk_ratio2', 'apsk_ratio3', 'ofdm_fft', 'ofdm_cp', 'ofdm_occupied', 'fsk_deviation_hz']) r[key] = signal[key];
  r.fm_deviation_hz = Math.max(1, signal.fm_deviation_hz);
  r.ofdm_design = clone(signal.ofdm_design);
  r.sample_rate = t.sample_rate;
  r.tuner_hz = tuner;
  r.dehop = state.rxUi.dehop ? clone(schedules) : [];
  if (r.modulation === 'ofdm') r.coarse_cfo = true;
  r.frame_enabled = !!signal.frame_enabled;
  if (signal.frame_enabled) {
    r.frame_blocks = signal.frame_mode === 'blocks' ? clone(signal.frame_blocks) : [
      {...newBlock('preamble', signal.waveform, signal.symbol_rate), bits: signal.preamble_bits, bit_count: Math.max(1, signal.preamble_bits.length)},
      {...newBlock('payload', signal.waveform, signal.symbol_rate), bit_count: Number(signal.payload_bits) + (signal.frame_crc ? 16 : 0)}];
  }
}

export function payload() {
  followTransmitter();
  const demod = !['monitor', 'pipeline'].includes(page());
  const r = demod ? {...clone(rx()), use_costas: rx().carrier_algorithm !== 'off', sample_rate: activeRate()} : null;
  if (r && activeFile()) r.dehop = [];
  return {pipeline: state.pipeline, channel: state.channel, rx: r, spectrum: spectrumRequest(fullTune())};
}

/* ----------------------------------------------------------------- dock -- */

function formatSection() {
  const t = appliedTx(), file = activeFile();
  let body = '';
  if (!file) body += toggleField('Follow the transmitter design', 'rxUi.follow');
  if (!file && t.waveform === 'composite') {
    body += selectField('Receive component', 'rxUi.component', t.components.map((c, i) => [i, `${i + 1} · ${c.label || c.waveform}${c.enabled ? '' : ' (off)'}`]),
      {help: 'The receiver tunes to this component; the others remain as adjacent signals.'});
  }
  if (!file && (t.agility?.enabled || (t.waveform === 'composite' && t.components.some(c => c.agility?.enabled)))) {
    body += toggleField('Remove the known hop / sweep schedule', 'rxUi.dehop');
  }
  body += selectField('Decode as', 'rx.modulation', RX_MODULATIONS.map(m => [m, m.toUpperCase()]))
    + (LINEAR.has(rx().modulation) || CPFSK.has(rx().modulation) ? numberField('Symbol rate', 'rx.symbol_rate', {unit: 'Bd', min: 10}) : '')
    + numberField('Tuner', 'rx.tuner_hz', {unit: 'Hz', offset: axisOffset(), help: 'Mixed to 0 Hz before demodulation (absolute when the source has an RF centre).'});
  return section('Demodulation format', body, {id: 'rx-format', help: 'Following copies the modulation, rates, filters, frame plan and tuning from the transmitter, and removes its hop schedule like a synchronized frequency-hopping receiver. Editing any of these fields switches to manual.'});
}

function demodSection() {
  const r = rx(), m = r.modulation;
  let body = '';
  if (LINEAR.has(m)) {
    body += selectField('Matched filter', 'rx.pulse_shape', [['rrc', 'Root raised cosine'], ['rect', 'Rectangular']])
      + sliderField('Roll-off β', 'rx.rolloff', {min: 0, max: 1, step: 0.01}) + numberField('Filter span', 'rx.filter_span_symbols', {min: 2, max: 64, integer: true});
    if (m.includes('apsk')) body += numberField('Ring 2 / ring 1', 'rx.apsk_ratio2', {min: 1.01, max: 8}) + (m === '32apsk' ? numberField('Ring 3 / ring 1', 'rx.apsk_ratio3', {min: 1.01, max: 12}) : '');
  }
  if (['2fsk', '4fsk'].includes(m)) body += numberField('Deviation', 'rx.fsk_deviation_hz', {unit: 'Hz', min: 1});
  if (m === 'fm') body += numberField('FM deviation', 'rx.fm_deviation_hz', {unit: 'Hz', min: 1});
  if (m === 'ofdm') {
    body += toggleField('CP frequency acquisition', 'rx.coarse_cfo') + numberField('Timing offset', 'rx.ofdm_timing_offset', {min: 0, max: 8192, integer: true, help: 'Samples skipped to compensate a known filter delay.'})
      + numberField('FFT length', 'rx.ofdm_fft', {min: 16, max: 4096, integer: true}) + numberField('Default CP', 'rx.ofdm_cp', {min: 0, max: 1024, integer: true})
      + selectField('Equalization', 'rx.ofdm_design.equalizer', [['pilot-ls', 'Pilot LS + interpolation'], ['off', 'Bypass']])
      + numberList('Pilot carriers', 'rx.ofdm_design.pilot_carriers', {defaultValue: 1});
  }
  return body ? section('Demodulator', body, {id: 'rx-demod'}) : '';
}

function loopFigures() {
  const r = rx(), kd = r.modulation === 'qpsk' && r.carrier_algorithm === 'costas' ? Math.SQRT2 : 1;
  const wn = Math.sqrt(r.costas.beta * kd), zeta = r.costas.alpha * kd / (2 * wn);
  const rs = r.sample_rate / Math.max(2, Math.round(r.sample_rate / r.symbol_rate));
  const settle = Math.ceil(4 / (zeta * wn)), lockIn = 2 * zeta * wn * rs / (2 * Math.PI);
  const noiseBw = wn / 2 * (zeta + 1 / (4 * zeta)) * rs;
  return `<div class="figure-grid"><div><span>Damping ζ</span><b>${zeta.toFixed(2)}</b></div><div><span>ωn</span><b>${wn.toFixed(4)} rad/sym</b></div>`
    + `<div><span>Lock-in range</span><b>±${esc(fmt(lockIn, 'Hz'))}</b></div><div><span>Settling</span><b>${settle} symbols</b></div><div><span>Noise bandwidth</span><b>${esc(fmt(noiseBw, 'Hz'))}</b></div></div>`;
}

function syncSections() {
  const r = rx(), m = r.modulation;
  if (!LINEAR.has(m)) return '';
  return section('Timing recovery', selectField('Algorithm', 'rx.timing_recovery', [['manual', 'Manual sampler'], ['mueller-muller', 'Mueller & Müller (joint with carrier)'], ['gardner', 'Gardner (interpolating)']])
      + (r.timing_recovery === 'manual' ? '' : sliderField('Loop gain', 'rx.timing_gain', {min: 0.001, max: 0.1, step: 0.001, help: 'Timing correction in symbols per unit of detector output; independent of oversampling. The loop settles in about 4 / (gain × detector slope) symbols.'}))
      + (r.timing_recovery === 'mueller-muller' ? '' : sliderField('Initial timing', 'rx.timing_phase', {min: -0.5, max: 0.5, step: 0.01, unit: ''})),
    {id: 'rx-timing', open: page() === 'timing', help: 'Mueller & Müller is decision-directed, so it runs jointly with the carrier loop on carrier-corrected decisions; Gardner is decision-free; manual samples at a fixed phase for eye-opening comparisons.'})
    + section('Carrier recovery', toggleField('Coarse CFO (M-th power FFT)', 'rx.coarse_cfo')
      + selectField('Tracking loop', 'rx.carrier_algorithm', [['costas', 'Costas (BPSK/QPSK), decision-directed otherwise'], ['decision-directed', 'Decision-directed PLL'], ['off', 'Bypass']])
      + numberField('Loop α', 'rx.costas.alpha', {min: 1e-4, max: 1}) + numberField('Loop β', 'rx.costas.beta', {min: 1e-5, max: 0.2})
      + loopFigures() + sliderField('Residual phase', 'rx.manual_phase_deg', {min: -180, max: 180, step: 1, unit: '°'})
      ,
    {id: 'rx-carrier', open: page() === 'carrier', help: 'Loop gains refer to unit-RMS symbols (the receiver normalizes them), so the figures hold at any level and samples per symbol. EVM is reported after the settling time; the shaded start of the EVM trace is the acquisition.'})
    + (page() === 'equalizer' ? section('Equalizer', selectField('Algorithm', 'rx.equalizer', [['off', 'Bypass'], ['cma', 'CMA (blind, PSK-oriented)']])
      + numberField('Taps', 'rx.equalizer_taps', {min: 3, max: 21, integer: true, step: 2}) + numberField('Adaptation gain', 'rx.equalizer_gain', {min: 1e-4, max: 0.03}),
    {id: 'rx-eq'}) : '');
}

function framingSection() {
  if (page() !== 'framing') return '';
  const r = rx(), unit = powerUnit();
  return section('Frame acquisition', toggleField('Enable frame acquisition', 'rx.frame_enabled')
      + toggleField('Energy detector', 'rx.energy_detection')
      + (r.energy_detection ? numberField('Energy threshold', 'rx.energy_threshold_dbm', {unit}) + numberField('Window', 'rx.energy_window_symbols', {min: 0.1, max: 256, help: 'In symbols.'}) : '')
      + toggleField('Preamble correlation', 'rx.preamble_correlation')
      + (r.preamble_correlation ? sliderField('Correlation threshold', 'rx.preamble_threshold', {min: 0, max: 1, step: 0.01}) : ''),
    {id: 'rx-frame', help: 'The first preamble correlation above threshold defines the frame start; energy detection is the fallback. Correlation is normalized, so its threshold is independent of power.'})
    + section('Expected frame', frameBlocksEditor('rx.frame_blocks'), {id: 'rx-blocks'});
}

function channelSection() {
  const file = activeFile();
  return section('Channel experiment', toggleField('Apply channel experiment', 'channel.enabled')
      + (file ? '<p class="note">A recording already contains its real channel; leave this off to analyse it as recorded.</p>' : '')
      + sliderField('SNR', 'channel.snr_db', {min: -10, max: 60, step: 1, unit: 'dB'})
      + sliderField('Carrier offset', 'channel.freq_offset_hz', {min: -5000, max: 5000, step: 10, unit: 'Hz'})
      + sliderField('Phase offset', 'channel.phase_offset_deg', {min: -180, max: 180, step: 1, unit: '°'})
      + sliderField('Timing offset', 'channel.timing_offset_samples', {min: -0.95, max: 0.95, step: 0.01, unit: ''})
      + sliderField('Echo gain', 'channel.echo_gain', {min: 0, max: 0.9, step: 0.01, unit: ''})
      + numberField('Echo delay', 'channel.echo_delay_symbols', {min: 1, max: 8, integer: true, help: 'In symbols.'})
      + sliderField('I/Q gain', 'channel.iq_gain_db', {min: -12, max: 12, step: 0.1, unit: 'dB'})
      + sliderField('Quadrature error', 'channel.iq_phase_deg', {min: -45, max: 45, step: 0.5, unit: '°'})
      + numberField('DC I', 'channel.dc_i', {min: -1, max: 1}) + numberField('DC Q', 'channel.dc_q', {min: -1, max: 1}),
  {id: 'rx-channel', open: false, help: 'Applied to each capture before the processing chain: AWGN, carrier frequency/phase offset (absolute-time exact), fractional timing delay, a delayed echo and I/Q imperfections.'});
}

function replaySection() {
  if (!['acquire', 'timing', 'carrier'].includes(page()) || !['linear', 'cpfsk'].includes(family())) return '';
  return section('Replay display', sliderField('Speed', 'rxUi.syncSpeed', {min: 0.2, max: 4, step: 0.1, unit: '×'})
    + sliderField('Window', 'rxUi.syncWindow', {min: 40, max: 600, step: 20, unit: ''}) + toggleField('Pause replay', 'rxUi.syncPause')
    + selectField('Loop trace', 'rxUi.loopTrace', [['frequency', 'Carrier frequency'], ['phase', 'Carrier phase'], ['error', 'Carrier error'], ['timing-error', 'Timing error'], ['timing-mu', 'Timing fraction μ']]),
  {id: 'rx-replay', open: false});
}

function dockHTML() {
  return section('Input', sourceSection(), {id: 'rx-source'}) + formatSection() + demodSection() + syncSections() + framingSection() + channelSection() + replaySection();
}

/* ----------------------------------------------------------------- main -- */

function mainHTML() {
  const p = page(), f = family();
  const tabs = views(pages(), p, 'rxpage');
  if (p === 'monitor') {
    return tabs + `<div class="readout-bar">${readout('Capture', 'mCapture')}${readout('Output rate', 'mRate')}${readout('RMS', 'mRms')}${readout('PAPR', 'mPapr')}</div>`
      + `<div class="stage quad">${pane('Processed I/Q', 'monIQ')}${pane('Sample distribution', 'monConst')}${pane('Processed spectrum', 'monSpec')}${pane('Processed waterfall · newest at top', 'monFall')}</div>`;
  }
  if (p === 'pipeline') {
    return tabs + `<div class="readout-bar">${readout('Capture', 'pCapture')}${readout('Output rate', 'pRate')}${readout('RMS', 'pRms')}${readout('PAPR', 'pPapr')}</div>`
      + `<div class="stage split"><div class="scroll-page" id="rxPipelineEditor" data-keep-scroll="rx-pipeline" style="border:1px solid var(--line-2);border-radius:var(--radius);background:var(--panel)"></div>`
      + `<div class="stage" style="padding:0;grid-template-rows:repeat(3,minmax(150px,1fr))">${pane('Input I/Q', 'pIn')}${pane('Probe I/Q', 'pOut')}${pane('Output spectrum', 'pSpec')}</div></div>`;
  }
  const demodReadouts = `<div class="readout-bar">${readout('Format', 'rFormat')}${readout('EVM (tracking)', 'rEvm', '—', 'accent')}${readout('All symbols', 'rEvmAll')}${readout('MER', 'rMer')}${readout('Symbols', 'rSyms')}${readout('Coarse CFO', 'rCoarse')}${readout('Loop frequency', 'rCfo')}${readout('Capture', 'rCapture', '—', 'grow')}</div>`;
  if (p === 'framing') {
    return tabs + `<div class="readout-bar">${readout('Detected', 'fDetected', '—', 'accent')}${readout('Frame start', 'fStart')}${readout('Correlation peak', 'fPeak')}${readout('Energy crossing', 'fEnergy')}</div>`
      + `<div class="stage quad">${pane(`Energy · ${powerUnit()}`, 'frEnergy')}${pane('Normalized preamble correlation', 'frCorr')}${pane('Received I/Q', 'frRaw')}${textPane('Frame blocks', 'frBlocks')}</div>`;
  }
  if (f === 'ofdm') {
    return tabs + demodReadouts + `<div class="stage quad">${pane('Equalized payload constellation', 'oConst')}${pane('Pilot channel estimate · magnitude', 'oChan')}${pane('EVM vs symbol', 'oEvm')}${textPane('Decoded payload', 'rxDecoded')}</div>`;
  }
  if (f === 'general') {
    return tabs + `<div class="readout-bar">${readout('Format', 'gFormat')}${readout('Capture', 'gCapture')}${readout('RMS', 'gRms')}${readout('Peak', 'gPeak')}${readout('Mean frequency', 'gFreq')}</div>`
      + `<div class="stage quad">${pane('Processed I/Q', 'gIQ')}${pane('Recovered / demodulated', 'gRec')}${pane('Spectrum', 'gSpec')}${pane('Envelope and instantaneous frequency', 'gTel')}</div>`;
  }
  const replay = pane('Synchronization replay · symbols', 'rxSync');
  if (p === 'timing') return tabs + demodReadouts + `<div class="stage quad">${pane('Eye · matched filter', 'rxEye')}${replay}${pane('Timing detector error', 'rxTimingErr')}${pane('Timing fraction μ', 'rxMu')}</div>`;
  if (p === 'carrier') return tabs + demodReadouts + `<div class="stage quad">${pane('Before carrier recovery', 'rxRaw')}${replay}${pane('Loop frequency', 'rxLoopF')}${pane('Loop phase error', 'rxLoopE')}</div>`;
  if (p === 'equalizer') return tabs + demodReadouts + `<div class="stage quad">${pane('Before equalizer', 'rxRaw')}${replay}${pane('Equalizer error', 'rxEqErr')}${pane('Channel and equalizer taps · magnitude', 'rxChan')}</div>`;
  return tabs + demodReadouts + `<div class="stage quad">${pane('Before carrier recovery', 'rxRaw')}${replay}${pane('EVM vs symbol', 'rxEvmT')}${textPane('Decoded output', 'rxDecoded')}</div>`;
}

export function render(el) {
  root = el;
  followTransmitter();
  el.innerHTML = `<div class="instrument"><aside class="dock"><div class="dock-head"><h2>Receiver</h2><span id="rxFollowBadge">${followBadge()}</span></div><div class="dock-scroll" data-keep-scroll="rx-dock"></div></aside><section class="main" id="rxMain"></section></div>`;
  bindControls(el, {onChange, actions});
  renderDock();
  renderMain();
}

function renderDock() {
  replaceHTML($('.dock-scroll', root), dockHTML());
  const badge = $('#rxFollowBadge', root);
  if (badge) badge.innerHTML = followBadge();
}

function renderMain() {
  $('#rxMain', root).innerHTML = mainHTML();
  monitorFall.reset();
  if (page() === 'pipeline') replaceHTML($('#rxPipelineEditor', root), `<div class="page">${pipelineEditor('pipeline', {stages: chain?.stages || []})}</div>`);
  draw();
}

/* -------------------------------------------------------------- changes -- */

const FORMAT_FIELDS = /^rx\.(modulation|symbol_rate|tuner_hz|rolloff|filter_span_symbols|pulse_shape|apsk_ratio|ofdm_|fsk_deviation|fm_deviation|frame_)/;

function onChange(path) {
  if (path === 'rxUi.follow' || path === 'rxUi.component' || path === 'rxUi.dehop') {
    state.rxRevision += 1;
    followTransmitter();
    renderDock();
    renderMain();
    liveUpdate();
    renderStatus();
    return;
  }
  if (path.startsWith('rxUi.')) { draw(); return; }
  if (FORMAT_FIELDS.test(path) && state.rxUi.follow && !activeFile()) {
    state.rxUi.follow = false;
    toast('Receiver set to manual: it no longer follows the transmitter. Use “follow TX” in the header to re-sync.');
  }
  state.rxRevision += 1;
  if (path === 'rx.modulation' || /\.(timing_recovery|carrier_algorithm|energy_detection|preamble_correlation|frame_enabled|role|enabled)$/.test(path) || path === 'channel.enabled') {
    renderDock();
    if (path === 'rx.modulation') renderMain();
  } else if (path.startsWith('rx.costas')) {
    const carrier = $('[data-fold="rx-carrier"] .figure-grid', root);
    if (carrier) carrier.outerHTML = loopFigures();
  }
  const badge = $('#rxFollowBadge', root);
  if (badge) badge.innerHTML = followBadge();
  if (path.startsWith('pipeline')) renderPipelineEditor();
  liveUpdate();
}

function renderPipelineEditor() {
  const el = $('#rxPipelineEditor', root);
  if (el) replaceHTML(el, `<div class="page">${pipelineEditor('pipeline', {stages: chain?.stages || []})}</div>`);
}

function afterStructure(path) {
  if (path.startsWith('rx.frame_blocks') && state.rxUi.follow) { state.rxUi.follow = false; toast('Receiver set to manual (frame plan edited)'); }
  state.rxRevision += 1;
  renderDock();
  renderPipelineEditor();
  liveUpdate();
}

const actions = {
  ...structuredActions(afterStructure),
  ...frameActions(afterStructure, () => ({modulation: rx().modulation, symbolRate: rx().symbol_rate})),
  ...pipelineActions(afterStructure),
  ...ofdmActions(afterStructure),
  ...sourceActions(() => render(root)),
  refollow: () => { state.rxUi.follow = true; onChange('rxUi.follow'); toast('Following the transmitter again'); },
};

/* ---------------------------------------------------------------- drawing -- */

function scheduleDraw() {
  if (paintPending) return;
  paintPending = true;
  requestAnimationFrame(() => { paintPending = false; draw(); });
}

function draw() {
  if (state.mode !== 'rx' || !root) return;
  const p = page(), f = family();
  if (p === 'monitor') return drawMonitor();
  if (p === 'pipeline') return drawPipeline();
  const d = data;
  if (!d) return;
  if (p === 'framing') return drawFraming(d);
  if (f === 'ofdm') return drawOfdm(d);
  if (f === 'general') return drawGeneral(d);
  drawDemod(d);
}

function captureText(d) {
  return `#${d.capture_id ?? '—'} · t ${fmt(d.tx_metadata?.simulation_time_s, 's')}`;
}

function drawMonitor() {
  const d = chain;
  if (!d) return;
  linePlot($('#monIQ'), [{y: d.output.i, x: d.output.time_s}, {y: d.output.q, x: d.output.time_s}], {xUnit: 's'});
  scatter($('#monConst'), d.output.i, d.output.q, {color: 'rgba(92,200,230,.5)'});
  linePlot($('#monSpec'), d.psd_db, {x: d.freq_hz, xUnit: 'Hz', yUnit: 'dB'});
  if (d.psd_db?.length) {
    const top = Math.max(...d.psd_db), row = Float32Array.from(d.psd_db);
    monitorFall.push(row, top - 80, top + 3);
    monitorFall.draw($('#monFall'), 54, 10);
  }
  setText('mCapture', `#${d.capture_id}`);
  setText('mRate', fmt(d.output_rate, 'S/s'));
  setText('mRms', fmt(d.rms));
  setText('mPapr', d.papr_db != null ? fmt(d.papr_db, 'dB') : '—');
}

function drawPipeline() {
  const d = chain;
  if (!d) return;
  linePlot($('#pIn'), [{y: d.input.i, x: d.input.time_s}, {y: d.input.q, x: d.input.time_s}], {xUnit: 's'});
  linePlot($('#pOut'), [{y: d.output.i, x: d.output.time_s}, {y: d.output.q, x: d.output.time_s}], {xUnit: 's'});
  linePlot($('#pSpec'), d.psd_db, {x: d.freq_hz, xUnit: 'Hz', yUnit: 'dB'});
  setText('pCapture', `#${d.capture_id}`);
  setText('pRate', fmt(d.output_rate, 'S/s'));
  setText('pRms', fmt(d.rms));
  setText('pPapr', d.papr_db != null ? fmt(d.papr_db, 'dB') : '—');
}

function demodReadouts(d) {
  const m = d.metrics || {}, t = d.telemetry || {};
  setText('rFormat', rx().modulation.toUpperCase());
  setText('rEvm', fmt(m.evm_rms_percent, '%'));
  setText('rEvmAll', m.evm_all_percent != null ? `${fmt(m.evm_all_percent, '%')} · ${m.evm_settle_symbols} settling` : '—');
  setText('rMer', fmt(m.mer_db, 'dB'));
  setText('rSyms', m.symbol_count == null ? '—' : m.capture_symbols > m.symbol_count + 32 ? `${fmtCount(m.symbol_count)} of ${fmtCount(m.capture_symbols)}` : fmtCount(m.symbol_count));
  setText('rCoarse', fmt(t.coarse_cfo_hz, 'Hz'));
  setText('rCfo', fmt(t.costas_freq_hz?.at(-1), 'Hz'));
  setText('rCapture', captureText(d));
}

/** Whether the received payload is text (a recording may be; PN and bit patterns are not). */
function textPayload() {
  if (activeFile()) return true;
  const t = appliedTx();
  const signal = t.waveform === 'composite' ? t.components[state.rxUi.component] || t : t;
  return signal.data_source === 'text';
}

function decodedText(d) {
  // Until a view is chosen: text for a text payload, hex for PN or bit patterns.
  const out = state.rxUi.output || (textPayload() ? 'text' : 'hex');
  const text = out === 'bits' ? (d.bits || []).join('').match(/.{1,64}/g)?.join('\n') : out === 'hex' ? d.hex : d.text;
  return `<div class="segmented" style="max-width:240px">${['text', 'hex', 'bits'].map(k => `<button type="button" data-rxout="${k}" class="${k === out ? 'active' : ''}">${k}</button>`).join('')}</div>`
    + `<pre class="code" style="max-height:none">${esc(text?.slice(0, 2400) || 'No decoded output for this waveform.')}</pre>`
    + `<p class="note">Capture ${esc(captureText(d))}${textPayload() ? '' : ' · PN / bit payload'}</p>`;
}

function drawDemod(d) {
  demodReadouts(d);
  const p = page(), t = d.telemetry || {}, ideals = idealConstellation(rx().modulation, rx());
  const before = p === 'equalizer' && t.pre_equalizer_i ? {i: t.pre_equalizer_i, q: t.pre_equalizer_q} : d.raw_symbols;
  if ($('#rxRaw')) scatter($('#rxRaw'), before?.i, before?.q, {ideals, dimIdeals: true});
  if ($('#rxEvmT')) {
    const settle = t.evm_settle_index || 0;
    linePlot($('#rxEvmT'), t.evm_time_percent || [], {min: 0, yUnit: '%', shade: settle ? [{x0: 0, x1: settle, color: 'rgba(236,123,114,.12)'}] : []});
    setText('rxEvmTValue', settle ? 'shaded: loop acquisition, excluded from EVM' : '');
  }
  if ($('#rxDecoded')) replaceHTML($('#rxDecoded'), decodedText(d));
  if ($('#rxEye')) eye($('#rxEye'), t.eye_i);
  if ($('#rxTimingErr')) linePlot($('#rxTimingErr'), t.timing_error || []);
  if ($('#rxMu')) linePlot($('#rxMu'), t.timing_mu || [], {min: 0, max: 1});
  if ($('#rxLoopF')) linePlot($('#rxLoopF'), t.costas_freq_hz || [], {yUnit: 'Hz'});
  if ($('#rxLoopE')) linePlot($('#rxLoopE'), t.costas_error || []);
  if ($('#rxEqErr')) linePlot($('#rxEqErr'), t.equalizer_error || []);
  if ($('#rxChan')) drawTaps(t);
  drawReplay();
}

/**
 * Symbol-spaced taps: the channel estimated against the known transmitted
 * symbols (amber) and, when the CMA equalizer runs, its own taps (cyan).
 */
function drawTaps(t) {
  const series = [];
  if (t.channel_estimate_taps?.length) series.push({y: t.channel_estimate_taps});
  if (t.equalizer_taps?.length) series.push({y: t.equalizer_taps, color: '#5cc8e6'});
  linePlot($('#rxChan'), series, {min: 0, xUnit: 'sym'});
  const parts = [t.channel_estimate_taps?.length ? 'amber: channel (known symbols)' : '', t.equalizer_taps?.length ? 'cyan: CMA equalizer' : ''].filter(Boolean);
  setText('rxChanValue', parts.length ? parts.join(' · ')
    : activeFile() ? 'a recording has no known symbols: enable the CMA equalizer for its taps'
    : 'needs a BPSK, QPSK, 8PSK or QAM transmitter, or the CMA equalizer');
}

/** Animated window of recovered symbols, with the loop trace cursor. */
function drawReplay() {
  const d = data, canvas = $('#rxSync');
  if (!d || !canvas) return;
  const I = d.symbols?.i || [], Q = d.symbols?.q || [];
  if (!I.length) return;
  const n = Math.min(I.length, Math.max(20, Math.round(state.rxUi.syncWindow)));
  const end = Math.max(1, Math.min(I.length, Math.floor((state.rxUi.replay % 1) * I.length)));
  const start = Math.max(0, end - n);
  scatter(canvas, I.slice(start, end), Q.slice(start, end), {ideals: idealConstellation(rx().modulation, rx()), scale: d.symbols.scale});
  const f = d.telemetry?.costas_freq_hz || [];
  setText('rxSyncValue', `symbol ${end} / ${I.length}${f.length ? ` · loop ${fmt(f[Math.min(f.length - 1, end - 1)], 'Hz')}` : ''}`);
}

function drawFraming(d) {
  const f = d.frame;
  scatter($('#frRaw'), d.raw_symbols?.i, d.raw_symbols?.q);
  if (!f) {
    replaceHTML($('#frBlocks'), '<p class="note">Enable frame acquisition to run energy detection and preamble correlation.</p>');
    return;
  }
  linePlot($('#frEnergy'), f.energy?.dbm || [], {threshold: f.energy?.threshold_dbm, yUnit: powerUnit()});
  linePlot($('#frCorr'), f.correlation?.score || [], {min: 0, max: 1, threshold: f.correlation?.threshold});
  setText('fDetected', f.detected ? 'YES' : 'NO');
  setText('fStart', f.start_sample == null ? '—' : `${fmtCount(f.start_sample)} · ${fmt(f.start_time_s, 's')}`);
  setText('fPeak', f.correlation?.peak == null ? '—' : Number(f.correlation.peak).toFixed(3));
  setText('fEnergy', f.energy?.first_crossing_sample == null ? '—' : fmtCount(f.energy.first_crossing_sample));
  const blocks = f.decoded_blocks?.length ? f.decoded_blocks : f.blocks || [];
  replaceHTML($('#frBlocks'), blocks.length
    ? `<table class="table"><thead><tr><th>Block</th><th>Role · modulation</th><th>Samples</th><th>Decode</th></tr></thead><tbody>${blocks.map((b, i) => `<tr><td>${esc(b.name || `Block ${i + 1}`)}</td><td>${esc(`${b.role || ''} · ${b.modulation || ''}`)}</td><td class="num-cell">${b.start_sample ?? '—'} … ${b.end_sample ?? '—'}</td><td>${esc(b.status || 'located')}${b.bit_count != null ? ` · ${b.bit_count} bits` : ''}${b.metrics?.evm_rms_percent != null ? ` · EVM ${fmt(b.metrics.evm_rms_percent, '%')}` : ''}${b.text ? ` · “${esc(b.text.slice(0, 40))}”` : ''}${b.error ? ` · ${esc(b.error)}` : ''}</td></tr>`).join('')}</tbody></table>`
    : `<p class="note">${f.detected ? 'Frame start found, but no expected blocks are configured.' : 'No frame start met the enabled detector thresholds.'}</p>`);
}

function drawOfdm(d) {
  demodReadouts(d);
  const t = d.telemetry || {};
  scatter($('#oConst'), d.symbols?.i, d.symbols?.q, {scale: 1, ideals: idealConstellation(rx().ofdm_design.modulation)});
  linePlot($('#oChan'), t.ofdm_channel_magnitude || [], {x: t.ofdm_carriers, min: 0});
  linePlot($('#oEvm'), t.evm_time_percent || [], {min: 0, yUnit: '%'});
  setText('rCoarse', fmt(t.ofdm_cfo_hz, 'Hz'));
  if ($('#rxDecoded')) replaceHTML($('#rxDecoded'), decodedText(d));
}

function drawGeneral(d) {
  const raw = d.raw_symbols || d.recovered_complex || {i: [], q: []}, rec = d.recovered_complex || raw, t = d.telemetry || {}, m = d.metrics || {};
  linePlot($('#gIQ'), [{y: raw.i}, {y: raw.q}]);
  linePlot($('#gRec'), rec.i?.length ? [{y: rec.i}, ...(rec.q ? [{y: rec.q}] : [])] : d.recovered || []);
  linePlot($('#gSpec'), d.psd_db || [], {x: d.freq_hz, xUnit: 'Hz', yUnit: 'dB'});
  linePlot($('#gTel'), t.frequency_hz?.length ? t.frequency_hz : t.amplitude || [], {yUnit: t.frequency_hz?.length ? 'Hz' : ''});
  setText('gFormat', rx().modulation.toUpperCase());
  setText('gCapture', captureText(d));
  setText('gRms', fmt(m.rms ?? d.rms));
  setText('gPeak', fmt(m.peak));
  setText('gFreq', fmtFreq(m.mean_frequency_hz, 1));
}

function animate() {
  let previous = performance.now(), lastPaint = 0;
  const loop = now => {
    if (state.mode !== 'rx') return;
    const dt = (now - previous) / 1000;
    previous = now;
    if (!state.rxUi.syncPause && data?.symbols?.i?.length) {
      state.rxUi.replay = (state.rxUi.replay + dt * state.rxUi.syncSpeed / Math.max(2, data.symbols.i.length / 700)) % 1;
    }
    if (now - lastPaint > 33 && $('#rxSync')) { lastPaint = now; drawReplay(); }
    animation = requestAnimationFrame(loop);
  };
  animation = requestAnimationFrame(loop);
}

/* ------------------------------------------------------------ lifecycle -- */

export function start() {
  unsubscribe = [
    subscribe('live:receiver', d => { data = d; chain = d; setStatus(`receiver · capture ${d.capture_id}`); scheduleDraw(); }),
    subscribe('live:pipeline', d => { chain = d; scheduleDraw(); }),
    subscribe('live:meta', () => replaceHTML($('[data-fold="rx-source"] .section-body', root), sourceSection())),
  ];
  animate();
}

export function stop() {
  for (const off of unsubscribe) off();
  unsubscribe = [];
  cancelAnimationFrame(animation);
}

export function resize() {
  draw();
}

document.addEventListener('click', event => {
  if (state.mode !== 'rx') return;
  const tab = event.target.closest('#rxMain [data-rxpage]');
  if (tab) {
    state.rxUi.page = tab.dataset.rxpage;
    data = null;
    renderDock();
    renderMain();
    liveUpdate(true);
    return;
  }
  const out = event.target.closest('#rxMain [data-rxout]');
  if (out) { state.rxUi.output = out.dataset.rxout; if (data) replaceHTML($('#rxDecoded'), decodedText(data)); }
});
