/* OFDM numerology, allocation, resource grid painting and shaping. */
import {$, esc} from '../core/dom.js';
import {post, errorText} from '../core/api.js';
import {numberField, section, selectField, toggleField} from '../core/controls.js';
import {fmt} from '../core/format.js';
import {getPath, state} from '../core/state.js';
import {designRequest} from '../core/source.js';
import {numberList, complexPairs} from './structured.js';
import {colors, setup} from '../plots/canvas.js';

const CONSTELLATIONS = ['bpsk', 'qpsk', '8psk', '16qam', '64qam'];
const ROLE_COLORS = ['#10171c', '#2f6f8c', '#c89b43', '#8e61a5'];  // null, data, pilot, explicit value
let brush = {kind: 'pilot', i: 1, q: 0, gain: 1, modulation: 'qpsk'};
let grid = null;

/** ``owner`` is the config holding ofdm_fft/ofdm_cp (``tx`` or ``rx``). */
export function ofdmModulationEditor(owner) {
  const d = `${owner}.ofdm_design`, cfg = getPath(state, owner), design = cfg.ofdm_design;
  const spacing = cfg.sample_rate / cfg.ofdm_fft;
  return section('Numerology', numberField('FFT length', `${owner}.ofdm_fft`, {min: 16, max: 4096, integer: true, step: 16})
      + `<div class="field"><span class="label">Subcarrier spacing</span><b style="font:500 12px var(--mono)">${esc(fmt(spacing, 'Hz'))}</b></div>`
      + numberField('Frame symbols', `${d}.symbols`, {min: 1, max: 128, integer: true})
      + numberField('Default CP', `${owner}.ofdm_cp`, {min: 0, max: 1024, integer: true, help: 'Samples of cyclic prefix; per-symbol values below override it.'})
      + selectField('Payload mapping', `${d}.modulation`, CONSTELLATIONS), {id: `ofdm-num-${owner}`})
    + section('Allocation and pilots', numberField('Left guard', `${d}.guard_left`, {min: 0, integer: true}) + numberField('Right guard', `${d}.guard_right`, {min: 0, integer: true})
      + toggleField('Null the DC carrier', `${d}.dc_null`) + numberField('Pilot I', `${d}.pilot_i`) + numberField('Pilot Q', `${d}.pilot_q`)
      + toggleField('Alternate pilot polarity', `${d}.pilot_alternating`)
      + selectField('Receiver equalization', `${d}.equalizer`, [['pilot-ls', 'Pilot LS + interpolation'], ['off', 'Bypass']])
      + numberList('Active carriers', `${d}.active_carriers`, {defaultValue: 1, help: 'signed indices; empty = from guards'})
      + numberList('Pilot carriers', `${d}.pilot_carriers`, {defaultValue: 1})
      + numberList('Cyclic-prefix sequence', `${d}.cp_lengths`, {defaultValue: 16, help: 'per symbol; empty = default CP'}), {id: `ofdm-alloc-${owner}`})
    + section('Resource grid', `<div class="btn-row">${['data', 'pilot', 'value', 'null'].map(kind => `<button type="button" class="btn small${brush.kind === kind ? ' on' : ''}" data-action="ofdm-brush" data-kind="${kind}">${kind === 'value' ? 'Explicit value' : kind[0].toUpperCase() + kind.slice(1)}</button>`).join('')}`
      + `<button type="button" class="btn small ghost" data-action="ofdm-clear-cells" data-path="${esc(d)}">Clear ${design.cells.length} overrides</button></div>`
      + `<div class="ofdm-grid" id="ofdmGrid" data-path="${esc(owner)}" data-keep-scroll="ofdm-grid"><canvas></canvas></div>`
      + `<p class="note" id="ofdmGridInfo">Click or drag to paint · blue payload · gold pilot · violet value · dark null</p>`,
    {id: `ofdm-grid-${owner}`, help: 'Painting adds per-cell overrides on top of the base allocation. Explicit cells carry the I/Q value of the brush; data cells can use their own constellation.'});
}

export function ofdmShapingEditor(owner) {
  const d = `${owner}.ofdm_design`, design = getPath(state, d);
  return section('Symbol overlap window', numberField('Overlap', `${d}.window_samples`, {min: 0, max: 1024, integer: true, help: 'Weighted overlap-add (WOLA) samples; must fit inside the shortest cyclic prefix.'})
      + selectField('Ramp', `${d}.window_kind`, [['raised-cosine', 'Raised cosine'], ['linear', 'Linear'], ['custom', 'Custom']])
      + (design.window_kind === 'custom' ? numberList('Rising ramp [0…1]', `${d}.window_ramp`, {defaultValue: 1, help: 'one value per overlap sample'}) : ''), {id: `ofdm-window-${owner}`})
    + section('Output filter', selectField('Filter', `${d}.output_filter`, [['off', 'Bypass'], ['lowpass', 'Designed low-pass FIR'], ['custom', 'Custom complex FIR']])
      + (design.output_filter === 'lowpass' ? numberField('Cutoff', `${d}.filter_cutoff_hz`, {unit: 'Hz', min: 1}) + numberField('Taps', `${d}.filter_length`, {min: 3, max: 2047, integer: true, step: 2})
        + selectField('Design window', `${d}.filter_window`, ['hann', 'hamming', 'blackman']) : '')
      + (design.output_filter === 'custom' ? complexPairs('Coefficients', `${d}.filter_i`, `${d}.filter_q`, {minLength: 1}) : '')
      , {id: `ofdm-filter-${owner}`, help: 'A causal L-tap FIR delays the frame by (L−1)/2 samples: set Receiver → OFDM timing offset to match.'});
}

/** Fetch the generated grid and draw it into #ofdmGrid (if present). */
export async function refreshOfdmGrid() {
  const host = $('#ofdmGrid');
  if (!host) return;
  const owner = host.dataset.path;
  const cfg = owner === 'tx' ? designRequest() : {...designRequest(), ...getPath(state, owner), waveform: 'ofdm'};
  try {
    const meta = await post('/api/studio/ofdm', {...cfg, waveform: 'ofdm'});
    grid = {owner, rows: meta.grid_shape[0], cols: meta.grid_shape[1], roles: meta.grid_roles, i: meta.grid_i, q: meta.grid_q, spacing: meta.subcarrier_spacing_hz};
    drawGrid();
    const info = $('#ofdmGridInfo');
    if (info) info.textContent = `${grid.rows} symbols × ${grid.cols} subcarriers · Δf ${fmt(grid.spacing, 'Hz')}. Click or drag to paint (${brush.kind}).`;
  } catch (error) {
    const info = $('#ofdmGridInfo');
    if (info) { info.textContent = errorText(error); info.className = 'note bad'; }
  }
}

function cellSize(host) {
  return Math.max(3, Math.min(16, Math.floor((host.clientWidth - 4) / Math.max(1, grid.cols))));
}

function drawGrid() {
  const host = $('#ofdmGrid');
  if (!host || !grid) return;
  const canvas = host.querySelector('canvas');
  const size = cellSize(host);
  canvas.style.width = `${grid.cols * size}px`;
  canvas.style.height = `${grid.rows * size}px`;
  const s = setup(canvas);
  if (!s) return;
  const {ctx} = s;
  for (let r = 0; r < grid.rows; r++) {
    for (let c = 0; c < grid.cols; c++) {
      ctx.fillStyle = ROLE_COLORS[grid.roles[r * grid.cols + c]] || ROLE_COLORS[0];
      ctx.fillRect(c * size, r * size, size - (size > 4 ? 1 : 0), size - (size > 4 ? 1 : 0));
    }
  }
  ctx.strokeStyle = colors().accent;
  const k0 = Math.floor(grid.cols / 2);
  ctx.beginPath(); ctx.moveTo(k0 * size + .5, 0); ctx.lineTo(k0 * size + .5, grid.rows * size); ctx.stroke();
}

function paintAt(event) {
  const host = $('#ofdmGrid');
  if (!host || !grid) return null;
  const rect = host.querySelector('canvas').getBoundingClientRect();
  const size = cellSize(host);
  const c = Math.floor((event.clientX - rect.left) / size), r = Math.floor((event.clientY - rect.top) / size);
  if (r < 0 || c < 0 || r >= grid.rows || c >= grid.cols) return null;
  const carrier = c - Math.floor(grid.cols / 2);
  const cells = getPath(state, `${grid.owner}.ofdm_design.cells`);
  const kept = cells.filter(cell => cell.symbol !== r || cell.carrier !== carrier);
  if (cells.length >= 8192 && kept.length === cells.length) return null;
  cells.splice(0, cells.length, ...kept, {symbol: r, carrier, ...brush});
  grid.roles[r * grid.cols + c] = {null: 0, data: 1, pilot: 2, value: 3}[brush.kind];
  drawGrid();
  return true;
}

/** Pointer painting; ``after()`` applies the edited cells. */
export function bindOfdmGrid(root, after) {
  if (root.dataset.ofdmBound) return;
  root.dataset.ofdmBound = '1';
  let painting = false;
  root.addEventListener('mousedown', event => {
    if (!event.target.closest('#ofdmGrid canvas')) return;
    painting = true;
    event.preventDefault();
    paintAt(event);
  });
  root.addEventListener('mousemove', event => { if (painting && event.target.closest('#ofdmGrid canvas')) paintAt(event); });
  window.addEventListener('mouseup', () => { if (painting) { painting = false; after(); } });
}

export function ofdmActions(after) {
  return {
    'ofdm-brush': el => {
      brush = {...brush, kind: el.dataset.kind};
      for (const b of document.querySelectorAll('[data-action="ofdm-brush"]')) b.classList.toggle('on', b === el);
    },
    'ofdm-clear-cells': el => { getPath(state, `${el.dataset.path}.cells`).length = 0; after(el.dataset.path); },
  };
}
