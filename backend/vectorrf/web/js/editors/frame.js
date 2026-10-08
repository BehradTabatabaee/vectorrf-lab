/* Generic frame-block editor shared by the transmitter and receiver. */
import {esc} from '../core/dom.js';
import {numberField, selectField, textField} from '../core/controls.js';
import {clone, getPath, state} from '../core/state.js';
import {toast} from '../core/ui.js';

export const FRAME_MODULATIONS = ['bpsk', 'qpsk', 'dbpsk', 'dqpsk', 'pi4-dqpsk', 'oqpsk', '8psk', '16qam', '64qam', '16apsk', '32apsk', '2fsk', '4fsk', 'msk', 'gmsk', 'ofdm'];
const ROLES = [['preamble', 'Preamble / sync'], ['payload', 'Payload source'], ['bits', 'Exact bit pattern'], ['pn', 'Random PN bits']];

export function newBlock(role = 'payload', modulation = 'qpsk', symbolRate = 50e3) {
  return {...clone(state.defaults.frame_block), name: role === 'preamble' ? 'Preamble' : role === 'bits' ? 'Bits' : 'Payload', role,
    modulation: FRAME_MODULATIONS.includes(modulation) ? modulation : 'qpsk', symbol_rate: symbolRate,
    bit_count: role === 'preamble' ? 32 : 256, bits: role === 'preamble' ? '10101010101010101101001110010001' : '11010010'};
}

export function frameBlocksEditor(path) {
  const blocks = getPath(state, path) || [];
  const cards = blocks.map((b, i) => {
    const p = `${path}.${i}`;
    return `<div class="card"><header>${i + 1} · ${esc(b.name || b.role)}<span class="tools">`
      + `<button type="button" class="btn small ghost" data-action="frame-move" data-path="${esc(path)}" data-index="${i}" data-dir="-1" aria-label="Move up">↑</button>`
      + `<button type="button" class="btn small ghost" data-action="frame-move" data-path="${esc(path)}" data-index="${i}" data-dir="1" aria-label="Move down">↓</button>`
      + `<button type="button" class="btn small ghost" data-action="frame-remove" data-path="${esc(path)}" data-index="${i}" aria-label="Remove block">×</button></span></header><div class="card-body">`
      + textField('Name', `${p}.name`) + selectField('Role', `${p}.role`, ROLES) + selectField('Modulation', `${p}.modulation`, FRAME_MODULATIONS)
      + numberField('Bits', `${p}.bit_count`, {min: 1, max: 65536, integer: true})
      + (['preamble', 'bits'].includes(b.role) ? textField('Bit pattern', `${p}.bits`) : '')
      + numberField('Symbol rate', `${p}.symbol_rate`, {unit: 'Bd', min: 10})
      + numberField('Power offset', `${p}.power_offset_db`, {unit: 'dB', min: -120, max: 120})
      + numberField('Repeat', `${p}.repeat`, {min: 1, max: 1024, integer: true}) + '</div></div>';
  }).join('');
  return `${cards || '<p class="note">No blocks yet.</p>'}<div class="btn-row">`
    + ['preamble', 'payload', 'bits', 'pn'].map(role => `<button type="button" class="btn small" data-action="frame-add" data-path="${esc(path)}" data-role="${role}">+ ${role === 'pn' ? 'PN' : role[0].toUpperCase() + role.slice(1)}</button>`).join('')
    + '</div>';
}

/** ``context(path)`` returns {modulation, symbolRate} defaults for new blocks. */
export function frameActions(after, context) {
  return {
    'frame-add': el => {
      const blocks = getPath(state, el.dataset.path);
      if (blocks.length >= 64) { toast('Up to 64 frame blocks'); return; }
      const {modulation, symbolRate} = context(el.dataset.path);
      blocks.push(newBlock(el.dataset.role, modulation, symbolRate));
      after(el.dataset.path);
    },
    'frame-remove': el => { getPath(state, el.dataset.path).splice(Number(el.dataset.index), 1); after(el.dataset.path); },
    'frame-move': el => {
      const a = getPath(state, el.dataset.path), i = Number(el.dataset.index), j = i + Number(el.dataset.dir);
      if (j < 0 || j >= a.length) return;
      [a[i], a[j]] = [a[j], a[i]];
      after(el.dataset.path);
    },
  };
}
