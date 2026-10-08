/* Waveform picker and per-waveform parameter sections.  Every function takes
   the path of a transmitter config (``tx`` or ``tx.components.i``), so the same
   editors serve the main signal and each composite component. */
import {esc} from '../core/dom.js';
import {numberField, section, selectField, sliderField, textField, toggleField} from '../core/controls.js';
import {fmt, formatQuantity} from '../core/format.js';
import {getPath, state} from '../core/state.js';
import {frameBlocksEditor} from './frame.js';
import {ofdmModulationEditor, ofdmShapingEditor} from './ofdm.js';
import {programEditor} from './program.js';
import {numberList} from './structured.js';

export const FAMILIES = [
  ['fundamentals', 'Basic', ['tone', 'multitone', 'awgn', 'chirp', 'custom']],
  ['linear', 'PSK / QAM / APSK', ['bpsk', 'qpsk', 'dbpsk', 'dqpsk', 'pi4-dqpsk', 'oqpsk', '8psk', '16qam', '64qam', '16apsk', '32apsk']],
  ['cpfsk', 'FSK / MSK', ['2fsk', '4fsk', 'msk', 'gmsk']],
  ['multicarrier', 'OFDM', ['ofdm']],
  ['analog', 'Analog', ['am', 'fm', 'pm', 'ssb', 'dsb-sc']],
];
export const LINEAR = new Set(FAMILIES[1][2]);
export const DIGITAL = new Set([...FAMILIES[1][2], ...FAMILIES[2][2], 'ofdm']);
const LABELS = {'pi4-dqpsk': 'π/4-DQPSK', 'dsb-sc': 'DSB-SC', awgn: 'Noise', multitone: 'Multitone', custom: 'Custom', chirp: 'Chirp', tone: 'Tone'};

export const waveformLabel = w => LABELS[w] || w.toUpperCase();

export function familyOf(waveform) {
  return FAMILIES.find(([, , items]) => items.includes(waveform))?.[0] || 'fundamentals';
}

/** Last waveform chosen in each family, so switching family and back restores it. */
const lastInFamily = {};

/** First waveform to show when ``family`` is chosen. */
export function waveformForFamily(family) {
  const items = FAMILIES.find(([key]) => key === family)?.[2] || [];
  return items.includes(lastInFamily[family]) ? lastInFamily[family] : items[0];
}

/** Family and modulation as two selects: only the chosen family's modulations are listed. */
export function waveformPicker(path) {
  const current = getPath(state, `${path}.waveform`), family = familyOf(current);
  lastInFamily[family] = current;
  const items = FAMILIES.find(([key]) => key === family)[2];
  const options = (list, selected) => list.map(([value, label]) => `<option value="${esc(value)}"${value === selected ? ' selected' : ''}>${esc(label)}</option>`).join('');
  return `<div class="field"><label for="fam-${esc(path)}">Family</label><select id="fam-${esc(path)}" data-change-action="pick-family" data-path="${esc(path)}" aria-label="Waveform family">`
    + `${options(FAMILIES.map(([key, label]) => [key, label]), family)}</select></div>`
    + `<div class="field"><label for="wf-${esc(path)}">Modulation</label><select id="wf-${esc(path)}" data-change-action="pick-waveform" data-path="${esc(path)}" aria-label="Waveform">`
    + `${options(items.map(w => [w, waveformLabel(w)]), current)}</select></div>`;
}

/** Transmitter sample rate (components always run on the composite's clock). */
const rate = () => Number(state.tx.sample_rate);

/** Baseband offset that frequencies inside the config at ``path`` add to (own and parent offsets). */
export function basebandOf(path) {
  const own = Number(getPath(state, `${path}.center_hz`)) || 0;
  return path === 'tx' ? own : own + (Number(state.tx.center_hz) || 0);
}

/**
 * Open interval for a frequency that lands at ``base + value`` in baseband: it
 * must stay inside the sampled band ±fs/2 (``closed`` allows the band edge).
 */
export function inBand(base = 0, closed = undefined) {
  const fs = rate(), half = fs / 2;
  return {lo: -half - base, hi: half - base, closed, base, note: `the sampled band at ${formatQuantity(fs, 'S/s')}`};
}

/** Symbol rates the generator and receiver accept at this sample rate (1.5 … 65,536 samples per symbol). */
export function symbolRateRange() {
  const fs = rate();
  return {lo: fs / 65536, hi: fs / 1.5, closed: 'lo', note: `1.5 to 65,536 samples per symbol at ${formatQuantity(fs, 'S/s')}`};
}

/** Generated symbol rate (integer samples per symbol). */
function actualRate(cfg, rate = cfg.symbol_rate) {
  return cfg.sample_rate / Math.max(2, Math.round(cfg.sample_rate / rate));
}

/** Modulation parameters; ``carrier`` is the absolute frequency of this config's 0 Hz offset. */
export function modulationSections(path, carrier = 0) {
  const cfg = getPath(state, path), w = cfg.waveform, fam = familyOf(w);
  const base = basebandOf(path), half = rate() / 2;
  const symbolRate = numberField('Symbol rate', `${path}.symbol_rate`, {unit: 'Bd', min: 10, within: symbolRateRange(),
    help: `Generated at an integer number of samples per symbol: actual ${fmt(actualRate(cfg), 'Bd')} (${Math.max(2, Math.round(cfg.sample_rate / cfg.symbol_rate))} samples/symbol).`});
  if (fam === 'linear') {
    let body = symbolRate;
    if (w.includes('apsk')) body += numberField('Ring 2 / ring 1', `${path}.apsk_ratio2`, {min: 1.01, max: 8}) + (w === '32apsk' ? numberField('Ring 3 / ring 1', `${path}.apsk_ratio3`, {min: 1.01, max: 12}) : '');
    const notes = {dbpsk: 'Differential binary phase increments.', dqpsk: 'Differential quadrature increments in multiples of π/2.',
      'pi4-dqpsk': 'Differential ±π/4 and ±3π/4 transitions.', oqpsk: 'Q delayed by half a symbol (fractional delay at odd samples/symbol).'};
    return section('Modulation', body, {id: `mod-${path}`, help: notes[w]})
      + section('Pulse shaping', selectField('Pulse', `${path}.pulse_shape`, [['rrc', 'Root raised cosine'], ['rect', 'Rectangular']])
        + (cfg.pulse_shape === 'rrc' ? sliderField('Roll-off β', `${path}.rolloff`, {min: 0, max: 1, step: 0.01}) + numberField('Filter span', `${path}.filter_span_symbols`, {min: 2, max: 64, integer: true, unit: ''}) : ''),
      {id: `shape-${path}`, help: `Occupied bandwidth ${fmt(actualRate(cfg) * (cfg.pulse_shape === 'rrc' ? 1 + cfg.rolloff : 2), 'Hz')}. The receiver uses the matching filter.`});
  }
  if (fam === 'cpfsk') {
    return section('Modulation', symbolRate
      + (['2fsk', '4fsk'].includes(w) ? numberField('Peak deviation', `${path}.fsk_deviation_hz`, {unit: 'Hz', min: 1,
        within: {lo: 0, hi: half - Math.abs(base), closed: 'hi', note: 'the outer tones stay inside the sampled band'}}) : '')
      + (w === 'gmsk' ? sliderField('Gaussian BT', `${path}.gaussian_bt`, {min: 0.1, max: 1, step: 0.01}) : '')
      , {id: `mod-${path}`, help: w === 'msk' || w === 'gmsk' ? 'Deviation ±Rs/4 (modulation index 0.5).' : undefined});
  }
  if (w === 'ofdm') return ofdmModulationEditor(path);
  if (fam === 'analog') {
    return section('Modulation', numberField('Message tone', `${path}.message_freq_hz`, {unit: 'Hz', min: 1, within: {lo: 0, hi: half, note: `the sampled band at ${formatQuantity(rate(), 'S/s')}`}})
      + (w === 'am' ? sliderField('AM index', `${path}.modulation_index`, {min: 0, max: 1, step: 0.01}) : '')
      + (w === 'fm' ? numberField('FM deviation', `${path}.fm_deviation_hz`, {unit: 'Hz', min: 0,
        within: {lo: 0, hi: half - Math.abs(base), closed: 'both', note: 'the peak frequency stays inside the sampled band'}}) : '')
      + (w === 'pm' ? numberField('PM deviation', `${path}.pm_deviation_rad`, {min: 0, max: 20, help: 'Peak phase deviation in radians.'}) : '')
      + (w === 'ssb' ? selectField('Sideband', `${path}.sideband`, [['usb', 'Upper'], ['lsb', 'Lower']]) : ''), {id: `mod-${path}`});
  }
  if (w === 'multitone') return section('Tones', numberList('Tone frequencies', `${path}.tones_hz`, {unit: 'Hz', offset: carrier, defaultValue: 0, minLength: 1, within: inBand(base)}), {id: `mod-${path}`});
  if (w === 'awgn') return section('Noise', numberField('Noise bandwidth', `${path}.noise_bandwidth_hz`, {unit: 'Hz', min: 10, placeholder: 'full band', nullable: true, help: 'Empty for white noise across the whole band; otherwise a linear-phase FIR limits it.'}), {id: `mod-${path}`});
  if (w === 'chirp') {
    return section('Chirp', numberField('Start', `${path}.chirp_start_hz`, {unit: 'Hz', offset: carrier, within: inBand(base, 'both')})
      + numberField('Stop', `${path}.chirp_stop_hz`, {unit: 'Hz', offset: carrier, within: inBand(base, 'both')})
      + numberField('Sweep period', `${path}.chirp_period_s`, {unit: 's', min: 1e-9, max: 600, nullable: true, placeholder: 'capture length'}),
      {id: `mod-${path}`, help: 'A sawtooth chirp: start → stop over each sweep period (empty: one sweep per capture). For schedule-controlled sweeps of any waveform use Hopping / sweep.'});
  }
  if (w === 'custom') return programEditor(`${path}.program`);
  return '';  // a tone has no parameters beyond its frequency offset and power
}

export function shapingSections(path) {
  const cfg = getPath(state, path);
  return cfg.waveform === 'ofdm' ? ofdmShapingEditor(path) : '';
}

export function dataSections(path) {
  const cfg = getPath(state, path);
  if (!DIGITAL.has(cfg.waveform)) return section('Payload', '<p class="note">This waveform carries no digital payload.</p>', {id: `data-${path}`});
  let body = selectField('Source', `${path}.data_source`, [['text', 'Text · UTF-8'], ['pn', 'PN / PRBS'], ['bits', 'Bit pattern']]);
  if (cfg.data_source === 'text') body += textField('Message', `${path}.message`, {help: 'UTF-8 bytes, most significant bit first, repeated to fill the capture.'});
  if (cfg.data_source === 'bits') body += textField('Bits', `${path}.custom_bits`, {help: 'Only 0 and 1; repeated to fill the capture.'});
  if (cfg.data_source === 'pn') {
    body += selectField('PRBS order', `${path}.pn_order`, [7, 9, 11, 15].map(n => [n, `PN${n} · period ${2 ** n - 1}`]))
      + numberField('Repeat length', `${path}.pn_bits`, {min: 32, max: 1e7, integer: true, unit: '', help: 'Bits before the sequence repeats (capped at its natural period).'})
      + numberField('Seed', `${path}.seed`, {integer: true});
  }
  let framing = toggleField('Frame builder', `${path}.frame_enabled`);
  if (cfg.frame_enabled) {
    framing += selectField('Frame model', `${path}.frame_mode`, [['simple', 'Preamble + payload + CRC'], ['blocks', 'Mixed-modulation blocks']]);
    framing += cfg.frame_mode === 'simple'
      ? textField('Preamble bits', `${path}.preamble_bits`) + numberField('Payload bits', `${path}.payload_bits`, {min: 8, max: 65536, integer: true, step: 8})
        + toggleField('Append CRC-16/CCITT-FALSE', `${path}.frame_crc`)
      : frameBlocksEditor(`${path}.frame_blocks`);
  }
  return section('Payload', body, {id: `data-${path}`})
    + section('Framing', framing, {id: `frame-${path}`, help: 'Generic laboratory framing (not a named standard). Block frames may mix modulations, rates and powers; the whole frame repeats to fill the capture. Receiver → Framing acquires it.'});
}

export function burstSection(path) {
  const cfg = getPath(state, path);
  return section('Burst gating', toggleField('Burst on/off gating', `${path}.burst_enabled`)
    + (cfg.burst_enabled ? numberField('Period', `${path}.burst_period_s`, {unit: 's', min: 1e-5, max: 2}) + sliderField('Duty', `${path}.burst_duty`, {min: 0.01, max: 1, step: 0.01}) : ''),
  {id: `burst-${path}`, help: 'Gates the output on absolute time, so the pattern continues across captures. Power reads the configured level while on.'});
}
