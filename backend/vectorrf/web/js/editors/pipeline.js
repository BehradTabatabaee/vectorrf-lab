/* Ordered DSP processing chain editor (receiver and radar). */
import {esc} from '../core/dom.js';
import {numberField, section, selectField, textField, toggleField} from '../core/controls.js';
import {clone, getPath, state} from '../core/state.js';
import {toast} from '../core/ui.js';
import {complexPairs, kvEditor} from './structured.js';

const KINDS = [['gain', 'Gain'], ['mixer', 'Frequency shift'], ['fir', 'Complex FIR'], ['lowpass', 'Low-pass FIR'], ['resample', 'Rational resampler'],
  ['dc', 'DC removal'], ['agc', 'RMS AGC'], ['expression', 'Expression'], ['costas', 'Costas loop'], ['gardner', 'Gardner timing'],
  ['mueller-muller', 'Mueller & Müller timing'], ['cma', 'CMA equalizer'], ['extension', 'Python extension']];
const KIND_LABEL = Object.fromEntries(KINDS);
const CARRIED = new Set(['fir', 'lowpass', 'costas', 'extension']);

function blockParams(p, b) {
  switch (b.kind) {
    case 'gain': return numberField('Gain', `${p}.gain_db`, {unit: 'dB', min: -120, max: 80});
    case 'mixer': return numberField('Shift down by', `${p}.frequency_hz`, {unit: 'Hz'});
    case 'fir': return complexPairs('Taps', `${p}.taps_i`, `${p}.taps_q`, {minLength: 1, help: 'Q may stay empty for a real FIR'});
    case 'lowpass': return numberField('Cutoff', `${p}.cutoff_hz`, {unit: 'Hz', min: 1}) + numberField('Taps', `${p}.taps`, {min: 3, max: 2047, integer: true, step: 2});
    case 'resample': return numberField('Up', `${p}.up`, {min: 1, max: 32, integer: true}) + numberField('Down', `${p}.down`, {min: 1, max: 128, integer: true});
    case 'expression': return textField('Output', `${p}.expression`, {textarea: true, rows: 2, help: '<code>x</code> samples, <code>t</code>/<code>time</code> absolute seconds, <code>fs</code>, <code>n</code>, <code>frame</code>.'});
    case 'costas': return numberField('Loop gain α', `${p}.loop_gain`, {min: 1e-5, max: 1}) + numberField('β', `${p}.beta`, {min: 1e-5, max: 0.2}) + selectField('Order', `${p}.order`, [[2, 'BPSK (2)'], [4, 'QPSK (4)']]);
    case 'gardner': case 'mueller-muller': return numberField('Loop gain', `${p}.loop_gain`, {min: 1e-5, max: 1}) + numberField('Symbol rate', `${p}.symbol_rate`, {unit: 'Bd', min: 1});
    case 'cma': return numberField('Gain', `${p}.loop_gain`, {min: 1e-5, max: 0.03}) + numberField('Taps', `${p}.taps`, {min: 3, max: 21, integer: true, step: 2});
    case 'extension': {
      const modules = state.extensions.filter(x => x.process).map(x => x.name);
      return selectField('Module', `${p}.extension`, [['', modules.length ? 'Choose a module' : 'No processing extensions installed'], ...modules]) + kvEditor('Parameters', `${p}.parameters`);
    }
    default: return '';
  }
}

// Block type each chain's Add button inserts, by chain path (kept across re-renders).
const nextKind = new Map();

export function pipelineEditor(path, {stages = []} = {}) {
  const cfg = getPath(state, path);
  const cards = cfg.blocks.map((b, i) => {
    const p = `${path}.blocks.${i}`, stage = stages.find(s => s.index === i);
    return `<div class="card${b.enabled ? '' : ' muted'}"><header>${i + 1} · ${esc(b.name || KIND_LABEL[b.kind])}`
      + `<span class="badge${CARRIED.has(b.kind) ? ' info' : ''}" title="${CARRIED.has(b.kind) ? 'State carries across live captures' : 'Restarts each capture'}">${CARRIED.has(b.kind) ? 'stateful' : 'per capture'}</span>`
      + `<span class="tools"><button type="button" class="btn small ghost" data-action="block-move" data-path="${esc(path)}" data-index="${i}" data-dir="-1" aria-label="Move up">↑</button>`
      + `<button type="button" class="btn small ghost" data-action="block-move" data-path="${esc(path)}" data-index="${i}" data-dir="1" aria-label="Move down">↓</button>`
      + `<button type="button" class="btn small ghost" data-action="block-remove" data-path="${esc(path)}" data-index="${i}" aria-label="Remove block">×</button></span></header>`
      + `<div class="card-body">${toggleField('Enabled', `${p}.enabled`)}${textField('Name', `${p}.name`)}${blockParams(p, b)}`
      + (stage ? `<p class="note">Output: ${stage.samples.toLocaleString()} samples · ${(stage.sample_rate / 1e3).toFixed(3)} kS/s · RMS ${stage.rms.toPrecision(3)}</p>` : '')
      + '</div></div>';
  }).join('');
  return section('Processing chain', `<div class="field"><label>Add block</label><div style="display:flex;gap:6px"><select data-change-action="block-kind" data-path="${esc(path)}" aria-label="Block type">${KINDS.map(([k, l]) => `<option value="${k}"${k === (nextKind.get(path) || 'gain') ? ' selected' : ''}>${esc(l)}</option>`).join('')}</select><button type="button" class="btn small" data-action="block-add" data-path="${esc(path)}">Add</button></div></div>`
    + selectField('Scope probe', `${path}.probe`, [[-1, 'Final output'], ...cfg.blocks.map((b, i) => [i, `${i + 1} · ${b.name || KIND_LABEL[b.kind]}`])])
    + `${cards || '<p class="note">No blocks: samples pass straight through.</p>'}<div class="btn-row"><button type="button" class="btn small" data-action="chain-reset" data-path="${esc(path)}">Reset loop/filter state</button></div>`,
  {id: `pipeline-${path}`, help: 'Samples flow through the enabled blocks in order. FIR, low-pass, Costas and Python extensions carry their state across live captures; timing, resampling and CMA restart per capture.'});
}

export function pipelineActions(after) {
  return {
    'block-kind': el => { nextKind.set(el.dataset.path, el.value); },
    'block-add': el => {
      const cfg = getPath(state, el.dataset.path);
      if (cfg.blocks.length >= 24) { toast('Up to 24 blocks'); return; }
      const kind = nextKind.get(el.dataset.path) || 'gain';
      cfg.blocks.push({...clone(state.defaults.dsp_block), kind, name: KIND_LABEL[kind], ...(kind === 'cma' ? {taps: 7, loop_gain: 0.002} : {})});
      after(el.dataset.path);
    },
    'block-remove': el => { getPath(state, el.dataset.path).blocks.splice(Number(el.dataset.index), 1); after(el.dataset.path); },
    'block-move': el => {
      const a = getPath(state, el.dataset.path).blocks, i = Number(el.dataset.index), j = i + Number(el.dataset.dir);
      if (j < 0 || j >= a.length) return;
      [a[i], a[j]] = [a[j], a[i]];
      after(el.dataset.path);
    },
    'chain-reset': el => { const cfg = getPath(state, el.dataset.path); cfg.reset_epoch = (cfg.reset_epoch || 0) + 1; after(el.dataset.path); },
  };
}
