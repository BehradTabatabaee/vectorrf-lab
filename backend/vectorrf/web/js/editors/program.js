/* Arbitrary complex waveform builder (expression, control points, I/Q table,
   local Python extension), shared by the transmitter and Radar Studio. */
import {esc} from '../core/dom.js';
import {numberField, section, selectField, textField} from '../core/controls.js';
import {formatQuantity} from '../core/format.js';
import {getPath, state} from '../core/state.js';
import {toast} from '../core/ui.js';
import {complexPairs, kvEditor} from './structured.js';

const MODES = [['expression', 'Complex expression'], ['points', 'Frequency / amplitude control points'], ['iq', 'I/Q sample table'], ['extension', 'Local Python waveform']];

function pointsTable(path) {
  const points = getPath(state, `${path}.points`) || [];
  const head = '<div class="structured-row" style="grid-template-columns:24px repeat(4,minmax(0,1fr)) 26px;font-size:10px;color:var(--dim)"><span></span><span>Time</span><span>Frequency</span><span>Amplitude</span><span>Phase</span><span></span></div>';
  const rows = points.map((k, i) => `<div class="structured-row" style="grid-template-columns:24px repeat(4,minmax(0,1fr)) 26px"><span class="idx">${i + 1}</span>`
    + [['time_us', 'µs'], ['frequency_hz', 'Hz'], ['amplitude', ''], ['phase_deg', '°']].map(([key, unit]) =>
      `<input type="text" inputmode="decimal" data-bind="${esc(path)}.points.${i}.${key}" data-kind="number" data-unit="${unit === 'µs' ? '' : unit}" value="${esc(formatQuantity(k[key], unit === 'µs' ? '' : unit))}" aria-label="Point ${i + 1} ${key}">`).join('')
    + `<button type="button" class="btn small ghost" data-action="point-remove" data-path="${esc(path)}" data-index="${i}" aria-label="Remove point">×</button></div>`).join('');
  return `<div class="structured"><div class="structured-head"><span>Control points (time in µs)</span><button type="button" class="btn small" data-action="point-add" data-path="${esc(path)}">+ Point</button></div><div class="structured-rows">${head}${rows}</div></div>`;
}

export function programEditor(path) {
  const p = getPath(state, path);
  let body = selectField('Method', `${path}.mode`, MODES) + numberField('Repeat period', `${path}.period_s`, {unit: 's', min: 1e-9, max: 1});
  if (p.mode === 'expression') {
    body += textField('Expression x(t)', `${path}.expression`, {textarea: true, rows: 4,
      help: '<code>t</code> time within the period · <code>time</code> absolute seconds · <code>T</code> period · <code>fs</code> sample rate · <code>n</code> sample index · <code>frame</code> capture ID · <code>j</code>, <code>pi</code>. Functions: sin, cos, tan, exp, sqrt, abs, angle, real, imag, conj, log, log10, floor, ceil, sign, tanh, sinc, minimum, maximum, where, clip. Comparisons and & | ~ work element-wise.'})
      + kvEditor('Named parameters', `${path}.parameters`, {help: 'available by name in the expression'});
  } else if (p.mode === 'points') {
    body += pointsTable(path);
  } else if (p.mode === 'iq') {
    body += complexPairs('I/Q samples', `${path}.samples_i`, `${path}.samples_q`, {minLength: 1})
      + numberField('Table rate', `${path}.table_rate`, {unit: 'S/s', min: 1, max: 1e8}) + selectField('Interpolation', `${path}.table_interp`, [['linear', 'Linear'], ['hold', 'Sample and hold']]);
  } else {
    const modules = state.extensions.filter(x => x.waveform).map(x => x.name);
    body += selectField('Extension', `${path}.extension`, [['', modules.length ? 'Choose a module' : 'No waveform extensions installed'], ...modules])
      + kvEditor('Parameters', `${path}.parameters`)
      + '<p class="note">Add a Python module with <code>waveform(context)</code> to the project’s <code>extensions</code> folder and restart the server (docs/PROGRAMMING.md).</p>';
  }
  return section('Arbitrary waveform', body, {id: `program-${path}`, help: 'Every method generates the same kind of signal: complex baseband samples at the transmitter rate, repeating at the period. Keep content below half the sample rate to avoid aliasing.'});
}

export function programActions(after) {
  return {
    'point-add': el => {
      const points = getPath(state, `${el.dataset.path}.points`), last = points.at(-1);
      points.push({...last, time_us: last.time_us + 25});
      after(el.dataset.path);
    },
    'point-remove': el => {
      const points = getPath(state, `${el.dataset.path}.points`);
      if (points.length <= 2) { toast('Keep at least two points'); return; }
      points.splice(Number(el.dataset.index), 1);
      after(el.dataset.path);
    },
  };
}
