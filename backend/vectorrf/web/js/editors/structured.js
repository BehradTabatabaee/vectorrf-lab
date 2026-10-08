/* Row editors for lists, complex pairs, 3-vectors and named parameters.
   Rows bind to ``path.<index>`` like any other control; structural edits go
   through ``structuredActions(after)``. */
import {esc} from '../core/dom.js';
import {formatQuantity} from '../core/format.js';
import {withinAttrs} from '../core/controls.js';
import {getPath, state} from '../core/state.js';
import {toast} from '../core/ui.js';

function cell(path, value, unit = '', label = '', offset = 0, within = null) {
  const range = Object.entries(withinAttrs(within)).filter(([, v]) => v !== undefined && v !== '').map(([k, v]) => ` ${k}="${esc(v)}"`).join('');
  return `<input type="text" inputmode="decimal" spellcheck="false" data-bind="${esc(path)}" data-kind="number" data-unit="${esc(unit)}"${offset ? ` data-offset="${offset}"` : ''}${range} value="${esc(formatQuantity(Number(value) + offset, unit))}" aria-label="${esc(label)}">`;
}

function frame(label, addAction, rows, {help = '', empty = 'No values.'} = {}) {
  return `<div class="structured"><div class="structured-head"><span>${esc(label)}${help ? ` · <span style="color:var(--dim)">${esc(help)}</span>` : ''}</span>${addAction}</div>`
    + `<div class="structured-rows">${rows || `<div class="empty">${esc(empty)}</div>`}</div></div>`;
}

export function numberList(label, path, {unit = '', defaultValue = 0, minLength = 0, help = '', empty, offset = 0, within = null} = {}) {
  const values = getPath(state, path) || [];
  const rows = values.map((v, i) => `<div class="structured-row"><span class="idx">${i + 1}</span>${cell(`${path}.${i}`, v, unit, `${label} ${i + 1}`, offset, within)}`
    + `<button type="button" class="btn small ghost" data-action="list-remove" data-path="${esc(path)}" data-index="${i}" data-min="${minLength}" aria-label="Remove ${esc(label)} ${i + 1}">×</button></div>`).join('');
  return frame(label, `<button type="button" class="btn small" data-action="list-add" data-path="${esc(path)}" data-default="${esc(defaultValue)}">+ Add</button>`, rows, {help, empty});
}

export function complexPairs(label, pathI, pathQ, {help = '', labels = ['I', 'Q'], units = ['', ''], minLength = 0} = {}) {
  const a = getPath(state, pathI) || [], b = getPath(state, pathQ) || [];
  const n = Math.max(a.length, b.length);
  const rows = Array.from({length: n}, (_, i) => `<div class="structured-row cols-2"><span class="idx">${i + 1}</span>${cell(`${pathI}.${i}`, a[i] ?? 0, units[0], `${labels[0]} ${i + 1}`)}${cell(`${pathQ}.${i}`, b[i] ?? 0, units[1], `${labels[1]} ${i + 1}`)}`
    + `<button type="button" class="btn small ghost" data-action="pair-remove" data-a="${esc(pathI)}" data-b="${esc(pathQ)}" data-index="${i}" data-min="${minLength}" aria-label="Remove row ${i + 1}">×</button></div>`).join('');
  const head = `<span style="display:flex;gap:6px;align-items:center"><span class="note" style="margin:0">${esc(labels.join(' · '))}</span><button type="button" class="btn small" data-action="pair-add" data-a="${esc(pathI)}" data-b="${esc(pathQ)}">+ Row</button></span>`;
  return frame(label, head, rows, {help, empty: 'No rows.'});
}

export function vectors3(label, path, {help = '', unit = ''} = {}) {
  const rows = (getPath(state, path) || []).map((r, i) => `<div class="structured-row cols-3"><span class="idx">${i + 1}</span>${['x', 'y', 'z'].map((axis, j) => cell(`${path}.${i}.${j}`, r[j] ?? 0, unit, `${label} ${i + 1} ${axis}`)).join('')}`
    + `<button type="button" class="btn small ghost" data-action="matrix-remove" data-path="${esc(path)}" data-index="${i}" aria-label="Remove element ${i + 1}">×</button></div>`).join('');
  return frame(label, `<button type="button" class="btn small" data-action="matrix-add" data-path="${esc(path)}">+ Element</button>`, rows, {help, empty: 'No elements.'});
}

export function kvEditor(label, path, {help = ''} = {}) {
  const entries = Object.entries(getPath(state, path) || {});
  const rows = entries.map(([key, value], i) => `<div class="structured-row cols-2"><span class="idx">${i + 1}</span>`
    + `<input type="text" spellcheck="false" value="${esc(key)}" data-change-action="kv-rename" data-path="${esc(path)}" data-key="${esc(key)}" aria-label="Parameter name ${i + 1}">`
    + `${cell(`${path}.${key}`, value, '', `Parameter ${key}`)}<button type="button" class="btn small ghost" data-action="kv-remove" data-path="${esc(path)}" data-key="${esc(key)}" aria-label="Remove ${esc(key)}">×</button></div>`).join('');
  return frame(label, `<button type="button" class="btn small" data-action="kv-add" data-path="${esc(path)}">+ Parameter</button>`, rows, {help, empty: 'No named parameters.'});
}

/** Actions for the editors above; ``after(path)`` re-renders and applies the change. */
export function structuredActions(after) {
  return {
    'list-add': el => { const a = getPath(state, el.dataset.path); a.push(Number(el.dataset.default) || 0); after(el.dataset.path); },
    'list-remove': el => {
      const a = getPath(state, el.dataset.path);
      if (a.length <= Number(el.dataset.min || 0)) { toast(`Keep at least ${el.dataset.min} value(s)`); return; }
      a.splice(Number(el.dataset.index), 1);
      after(el.dataset.path);
    },
    'pair-add': el => { getPath(state, el.dataset.a).push(0); getPath(state, el.dataset.b).push(0); after(el.dataset.a); },
    'pair-remove': el => {
      const a = getPath(state, el.dataset.a), b = getPath(state, el.dataset.b);
      if (a.length <= Number(el.dataset.min || 0)) { toast('Keep at least one row'); return; }
      a.splice(Number(el.dataset.index), 1);
      b.splice(Number(el.dataset.index), 1);
      after(el.dataset.a);
    },
    'matrix-add': el => { getPath(state, el.dataset.path).push([0, 0, 0]); after(el.dataset.path); },
    'matrix-remove': el => {
      const a = getPath(state, el.dataset.path);
      if (a.length <= 1) { toast('Keep at least one element'); return; }
      a.splice(Number(el.dataset.index), 1);
      after(el.dataset.path);
    },
    'kv-add': el => {
      const obj = getPath(state, el.dataset.path);
      let i = 1;
      while (Object.hasOwn(obj, `p${i}`)) i += 1;
      obj[`p${i}`] = 0;
      after(el.dataset.path);
    },
    'kv-remove': el => { delete getPath(state, el.dataset.path)[el.dataset.key]; after(el.dataset.path); },
    'kv-rename': el => {
      const obj = getPath(state, el.dataset.path), old = el.dataset.key, key = el.value.trim();
      if (!/^[A-Za-z_]\w*$/.test(key) || (key !== old && Object.hasOwn(obj, key))) {
        toast('Names must be unique identifiers (letters, digits, underscore)');
        el.value = old;
        return;
      }
      const rebuilt = Object.fromEntries(Object.entries(obj).map(([k, v]) => [k === old ? key : k, v]));
      for (const k of Object.keys(obj)) delete obj[k];
      Object.assign(obj, rebuilt);
      after(el.dataset.path);
    },
  };
}
