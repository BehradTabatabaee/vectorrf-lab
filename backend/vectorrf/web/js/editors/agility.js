/* Frequency hopping / sweep editor and its summary. */
import {esc} from '../core/dom.js';
import {numberField, section, segmented, selectField, toggleField} from '../core/controls.js';
import {fmt, fmtCount, percent} from '../core/format.js';
import {getPath, state} from '../core/state.js';
import {numberList} from './structured.js';
import {basebandOf, inBand} from './waveform.js';

/**
 * ``path`` is the owning transmitter config (``tx`` or ``tx.components.i``);
 * ``carrier`` is the absolute frequency its offsets are relative to, so
 * frequencies are entered where the signal actually lands (RF when an RF
 * centre is set).  ``summary`` is the plan's agility summary, if known.
 */
export function agilityEditor(path, carrier, summary) {
  const a = getPath(state, `${path}.agility`);
  const p = `${path}.agility`;
  if (!a.enabled) {
    return section('Hopping / sweep', toggleField('Enable frequency agility', `${p}.enabled`),
      {id: `agility-${path}`, help: 'Hop the signal over a frequency grid or list, or sweep it linearly, with dwell time, off-time, edge ramps and phase control. Works on any waveform, including composites and their individual components.'});
  }
  const hop = a.mode === 'hop';
  const band = inBand(basebandOf(path), a.mode === 'sweep' ? 'both' : undefined);  // a sweep may end at the band edge
  const f = (label, key, help) => numberField(label, `${p}.${key}`, {unit: 'Hz', offset: carrier, help, within: band});
  let body = toggleField('Enable frequency agility', `${p}.enabled`)
    + segmented(`${p}.mode`, [['hop', 'Frequency hopping'], ['sweep', 'Linear sweep (chirp)']], {label: 'Agility mode'});
  if (hop) {
    body += selectField('Frequency set', `${p}.plan`, [['linear', 'Linear grid start → stop'], ['random', 'Pseudo-random order of the grid'], ['list', 'Explicit list']]);
    if (a.plan === 'list') body += numberList('Hop frequencies', `${p}.frequencies_hz`, {unit: 'Hz', offset: carrier, defaultValue: 0, minLength: 1, within: band});
    else body += f('Start', 'start_hz') + f('Stop', 'stop_hz') + numberField('Step', `${p}.step_hz`, {unit: 'Hz', min: 1e-3});
    if (a.plan === 'random') body += numberField('Pattern seed', `${p}.seed`, {integer: true, min: 0});
    if (a.plan !== 'random') body += selectField('Order', `${p}.order`, [['up', 'Start → stop'], ['down', 'Stop → start'], ['up-down', 'Start → stop → start']]);
  } else {
    body += f('Start', 'start_hz') + f('Stop', 'stop_hz')
      + selectField('Shape', `${p}.order`, [['up', 'Sawtooth start → stop'], ['down', 'Sawtooth stop → start'], ['up-down', 'Triangle start → stop → start']]);
  }
  body += numberField(hop ? 'Dwell' : 'Sweep time', `${p}.dwell_s`, {unit: 's', min: 1e-9, max: 60})
    + numberField('Off-time', `${p}.off_s`, {unit: 's', min: 0, max: 60, help: 'Output is gated off between dwells (transmitter noise continues).'})
    + numberField('Edge ramp', `${p}.ramp_s`, {unit: 's', min: 0, max: 60, help: 'Raised-cosine rise and fall inside each dwell: limits switching splatter.'})
    + numberField('Start delay', `${p}.offset_s`, {unit: 's', min: 0, max: 3600, help: 'Delays the schedule, e.g. to stagger components.'})
    + (hop ? selectField('Phase', `${p}.phase`, [['continuous', 'Continuous (VCO-like)'], ['coherent', 'Coherent (free-running synthesizers)']],
      {help: 'Continuous: phase never jumps at a hop. Coherent: each hop has the phase its own oscillator would have had since t = 0, independent of the hop order.'}) : '');
  if (summary) {
    const rows = hop
      ? [['Hops', fmtCount(summary.segments)], ['Hop rate', fmt(summary.hop_rate_hz, 'Hz')], ['Period', fmt(summary.period_s, 's')], ['Duty', percent(summary.duty, 1)]]
      : [['Sweep rate', `${fmt(summary.sweep_rate_hz_s, 'Hz')}/s`], ['Period', fmt(summary.period_s, 's')], ['Duty', percent(summary.duty, 1)], ['Span', fmt(summary.max_hz - summary.min_hz, 'Hz')]];
    body += `<div class="figure-grid">${rows.map(([k, v]) => `<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}</div>`;
  }
  return section('Hopping / sweep', body, {id: `agility-${path}`,
    help: 'The schedule is an exact 64-bit DDS: phase stays exact across captures at any rate. Frequencies here are where the signal’s carrier lands (RF centre + offsets). The schedule repeats with the period shown.'});
}
