/* Control builders (HTML strings) and delegated binding.

   Every editable control carries ``data-bind="<state path>"``.  One set of
   listeners per root handles all of them, so re-rendering never needs to
   rebind and never loses handlers.  Number fields accept engineering
   notation and show values with SI prefixes. */
import {$$, esc, foldOpen} from './dom.js';
import {formatQuantity, parseQuantity} from './format.js';
import {getPath, setPath, state} from './state.js';
import {toast} from './ui.js';

let uid = 0;

/** Attributes that give a label its explanation as a hover tooltip (dotted underline). */
export function tip(text) {
  return text ? ` class="tipped" data-tip="${esc(text)}"` : '';
}

function attrs(map) {
  return Object.entries(map).filter(([, v]) => v !== undefined && v !== null && v !== false && v !== '')
    .map(([k, v]) => (v === true ? k : `${k}="${esc(v)}"`)).join(' ');
}

/** Attributes of an open interval ``(lo, hi)`` that a value must lie strictly inside (``within``). */
export function withinAttrs(within) {
  // ``closed``: 'lo', 'hi' or 'both' allows that end itself.
  if (!within) return {};
  return {'data-above': Number.isFinite(within.lo) ? within.lo : undefined, 'data-below': Number.isFinite(within.hi) ? within.hi : undefined,
    'data-within-closed': within.closed || undefined, 'data-within-note': within.note || '',
    // ``base``: the interval is the sampled band ±fs/2 seen from this offset, so a refusal can name the rate that would fit.
    'data-within-base': Number.isFinite(within.base) ? String(within.base) : undefined};
}

/**
 * Numeric field with unit-aware engineering notation and steppers.  ``min``/``max``
 * clamp (with a notice); ``within: {lo, hi, note}`` is an open interval the value
 * must lie strictly inside (e.g. the sampled band): entries outside it are
 * refused with the allowed range instead of being applied.
 */
export function numberField(label, path, opts = {}) {
  const {unit = '', min, max, step, help, offset = 0, placeholder = '', integer = false, disabled = false, nullable = false, within, scale} = opts;
  const raw = opts.value !== undefined ? opts.value : getPath(state, path);
  const text = raw === null || raw === undefined || raw === '' ? '' : formatQuantity(Number(raw) + Number(offset), unit);
  const id = `f${++uid}`;
  return `<div class="field" data-field="${esc(path)}"><label for="${id}"${tip(help)}>${esc(label)}</label>`
    + `<div class="num"><input type="text" inputmode="decimal" autocomplete="off" spellcheck="false" id="${id}" ${attrs({
      'data-bind': path, 'data-kind': 'number', 'data-unit': unit, 'data-offset': offset || '', 'data-min': min, 'data-max': max,
      'data-step': step, 'data-scale': scale, 'data-integer': integer, 'data-nullable': nullable, ...withinAttrs(within), value: text, placeholder, disabled, 'aria-label': label})}>`
    + `<span class="steps"><button type="button" class="up" data-step-dir="1" tabindex="-1" aria-label="Increase ${esc(label)}"${disabled ? ' disabled' : ''}></button>`
    + `<button type="button" class="down" data-step-dir="-1" tabindex="-1" aria-label="Decrease ${esc(label)}"${disabled ? ' disabled' : ''}></button></span></div>`
    + '</div>';
}

export function textField(label, path, opts = {}) {
  const {help, placeholder = '', textarea = false, rows = 3, wide = textarea} = opts;
  const value = opts.value !== undefined ? opts.value : getPath(state, path) ?? '';
  const id = `f${++uid}`;
  const control = textarea
    ? `<textarea id="${id}" rows="${rows}" spellcheck="false" ${attrs({'data-bind': path, 'data-kind': 'text', placeholder, 'aria-label': label})}>${esc(value)}</textarea>`
    : `<input type="text" id="${id}" spellcheck="false" ${attrs({'data-bind': path, 'data-kind': 'text', value, placeholder, 'aria-label': label})}>`;
  return `<div class="field${wide ? ' wide' : ''}"><label for="${id}"${tip(help)}>${esc(label)}</label>${control}</div>`;
}

/** ``items``: values, or [value, label] pairs. */
export function selectField(label, path, items, opts = {}) {
  const {help, disabled = false} = opts;
  const current = opts.value !== undefined ? opts.value : getPath(state, path);
  const numeric = typeof current === 'number';
  const id = `f${++uid}`;
  const options = items.map(item => {
    const [value, text] = Array.isArray(item) ? item : [item, item];
    return `<option value="${esc(value)}"${String(value) === String(current) ? ' selected' : ''}>${esc(text)}</option>`;
  }).join('');
  return `<div class="field"><label for="${id}"${tip(help)}>${esc(label)}</label><select id="${id}" ${attrs({
    'data-bind': path, 'data-kind': numeric ? 'select-number' : 'select', disabled, 'aria-label': label})}>${options}</select></div>`;
}

export function toggleField(label, path, opts = {}) {
  const checked = opts.value !== undefined ? opts.value : getPath(state, path);
  return `<label class="toggle"><input type="checkbox" ${attrs({'data-bind': path, 'data-kind': 'bool', checked: !!checked, disabled: opts.disabled})}><span>${esc(label)}</span></label>`;
}

export function sliderField(label, path, opts = {}) {
  const {min = 0, max = 1, step = 0.01, unit = '', help} = opts;
  const value = opts.value !== undefined ? opts.value : getPath(state, path);
  return `<div class="slider"><div class="slider-top"><span${tip(help)}>${esc(label)}</span><div class="num"><input type="text" inputmode="decimal" ${attrs({
    'data-bind': path, 'data-kind': 'number', 'data-unit': unit, 'data-min': min, 'data-max': max, 'data-step': step,
    value: formatQuantity(value, unit), 'aria-label': `${label} value`})}><span class="steps"><button type="button" class="up" data-step-dir="1" tabindex="-1" aria-label="Increase"></button><button type="button" class="down" data-step-dir="-1" tabindex="-1" aria-label="Decrease"></button></span></div></div>`
    + `<input type="range" ${attrs({'data-bind': path, 'data-kind': 'range', min, max, step, value, 'aria-label': label})}>`
    + '</div>';
}

/** Button group bound to one value. */
export function segmented(path, items, opts = {}) {
  const current = opts.value !== undefined ? opts.value : getPath(state, path);
  return `<div class="segmented" role="group"${opts.label ? ` aria-label="${esc(opts.label)}"` : ''}>${items.map(item => {
    const [value, text] = Array.isArray(item) ? item : [item, item];
    return `<button type="button" data-set-path="${esc(path)}" data-set-value="${esc(JSON.stringify(value))}" class="${String(value) === String(current) ? 'active' : ''}">${esc(text)}</button>`;
  }).join('')}</div>`;
}

/** Collapsible dock/editor section; ``help`` explains it in a tooltip on the title. */
export function section(title, body, opts = {}) {
  const {id = title, open = true, help, tools = ''} = opts;
  return `<details class="section" data-fold="${esc(id)}"${foldOpen(id, open) ? ' open' : ''}><summary><span${tip(help)}>${esc(title)}</span>${tools ? `<span class="summary-tools">${tools}</span>` : ''}</summary><div class="section-body">${body}</div></details>`;
}


export function views(items, current, attr = 'view', right = '') {
  return `<nav class="views">${items.map(([key, label]) => `<button type="button" data-${attr}="${esc(key)}" class="${key === current ? 'active' : ''}">${esc(label)}</button>`).join('')}${right ? `<div class="views-right">${right}</div>` : ''}</nav>`;
}

export function readout(label, id, value = '—', cls = '') {
  return `<div class="readout ${cls}"><span>${esc(label)}</span><b id="${esc(id)}">${esc(value)}</b></div>`;
}

export function pane(title, id, opts = {}) {
  const {cls = '', value = '', body = ''} = opts;
  return `<section class="pane ${cls}"><header class="pane-head"><h3>${esc(title)}</h3><span class="pane-value" id="${esc(id)}Value">${esc(value)}</span></header>`
    + `<div class="pane-body">${body || `<canvas id="${esc(id)}"></canvas>`}</div></section>`;
}

export function textPane(title, id, cls = '') {
  return `<section class="pane text-pane ${cls}"><header class="pane-head"><h3>${esc(title)}</h3><span class="pane-value" id="${esc(id)}Value"></span></header><div class="pane-body panel" id="${esc(id)}" data-keep-scroll="${esc(id)}"></div></section>`;
}

/* ------------------------------------------------------------- binding -- */

const handlers = new WeakMap();

/**
 * Arrow/stepper increment: a tenth of the value's decade, but never smaller
 * than a tenth of the field's typical magnitude (``data-scale``; frequency
 * fields default to a tenth of the transmitter sample rate).  Stepping a
 * frequency up from 0 Hz therefore moves it visibly (10 kHz at 1 MS/s)
 * instead of by 1 Hz, then 0.1 Hz.
 */
function stepFor(input, value) {
  const configured = Number(input.dataset.step);
  if (Number.isFinite(configured) && configured > 0) return configured;
  if (input.dataset.integer) return 1;
  const unit = input.dataset.unit;
  if (/^dB/.test(unit) || unit === '°' || unit === '%') return 1;
  const scale = Number(input.dataset.scale) || (unit === 'Hz' ? Number(state.tx?.sample_rate) / 10 || 0 : 0);
  const a = Math.max(Math.abs(value), scale);
  if (!a) return 1;
  return 10 ** (Math.floor(Math.log10(a)) - 1);
}

function setFieldError(input, message) {
  const field = input.closest('.field, .slider, .structured-row');
  if (!field) return;
  field.classList.toggle('invalid', !!message);
  input.setAttribute('aria-invalid', message ? 'true' : 'false');
  let note = field.querySelector('.field-error');
  if (message) {
    if (!note) { note = Object.assign(document.createElement('div'), {className: 'field-error', role: 'alert'}); field.append(note); }
    note.textContent = message;
  } else note?.remove();
}

/** Why ``value`` (stored units) lies outside the field's open interval, or ''. */
function outsideWithin(input, value) {
  const lo = input.dataset.above !== undefined ? Number(input.dataset.above) : -Infinity;
  const hi = input.dataset.below !== undefined ? Number(input.dataset.below) : Infinity;
  const closed = input.dataset.withinClosed || '';  // 'lo', 'hi' or 'both': that end is allowed
  const aboveLo = /lo|both/.test(closed) ? value >= lo : value > lo;
  const belowHi = /hi|both/.test(closed) ? value <= hi : value < hi;
  if (aboveLo && belowHi) return '';
  const unit = input.dataset.unit || '', offset = Number(input.dataset.offset) || 0;
  const show = v => formatQuantity(v + offset, unit);
  const range = Number.isFinite(lo) && Number.isFinite(hi) ? `between ${show(lo)} and ${show(hi)}`
    : Number.isFinite(hi) ? `below ${show(hi)}` : `above ${show(lo)}`;
  const note = input.dataset.withinNote;
  const base = input.dataset.withinBase;
  // A carrier at baseband offset f needs a sample rate above 2|f|.
  const rate = base !== undefined ? `; ${show(value)} needs a sample rate above ${formatQuantity(2 * Math.abs(value + Number(base)), 'S/s')}` : '';
  return `${show(value)} is not applied: it must be ${range}${note ? ` (${note})` : ''}${rate}`;
}

function commitNumber(input) {
  const unit = input.dataset.unit || '';
  const offset = Number(input.dataset.offset) || 0;
  const text = input.value.trim();
  if (!text) return input.dataset.nullable ? {ok: true, value: null} : {ok: false, message: 'Enter a value'};
  let value = parseQuantity(text, unit);
  if (!Number.isFinite(value)) {
    return {ok: false, message: `Not a ${unit ? `quantity in ${unit}` : 'number'}: try ${unit ? formatQuantity(1e3, unit).replace(' ', '') : '1000'} or 1e3`};
  }
  value -= offset;
  if (input.dataset.integer) value = Math.round(value);
  const min = input.dataset.min !== undefined ? Number(input.dataset.min) : -Infinity;
  const max = input.dataset.max !== undefined ? Number(input.dataset.max) : Infinity;
  if (value < min || value > max) {
    const bounded = Math.min(max, Math.max(min, value));
    toast(`Limited to ${formatQuantity(bounded + offset, unit) || bounded}`);
    value = bounded;
  }
  const outside = outsideWithin(input, value);
  if (outside) return {ok: false, message: outside};
  return {ok: true, value};
}

function displayValue(input, value) {
  if (input.type === 'checkbox') input.checked = !!value;
  else if (input.dataset.kind === 'number') input.value = value === null ? '' : formatQuantity(Number(value) + (Number(input.dataset.offset) || 0), input.dataset.unit || '');
  else input.value = value ?? '';
}

function apply(root, input, value) {
  const path = input.dataset.bind;
  const changed = getPath(state, path) !== value;
  if (changed) setPath(state, path, value);
  for (const peer of $$(`[data-bind="${CSS.escape(path)}"]`, root)) if (peer !== input || peer.dataset.kind === 'number') displayValue(peer, value);
  // A change that leaves the state as it was (a value re-entered, or the
  // browser's own change event when a re-render removes the focused field
  // after Enter already applied it) must not render again: that render would
  // run inside the one in progress.
  if (changed) handlers.get(root)?.onChange?.(path, value, input);
}

function readControl(input) {
  switch (input.dataset.kind) {
    case 'number': return commitNumber(input);
    case 'range': return {ok: true, value: Number(input.value)};
    case 'bool': return {ok: true, value: input.checked};
    case 'select-number': return {ok: true, value: Number(input.value)};
    default: return {ok: true, value: input.value};
  }
}

/**
 * Attach delegated listeners to ``root`` once.  ``onChange(path, value, el)``
 * runs after a bound value changes; ``actions[name](el, event)`` handles
 * ``data-action`` buttons.  Calling again replaces the handlers.
 */
export function bindControls(root, {onChange, actions = {}} = {}) {
  if (!root) return;
  const existing = handlers.get(root);
  handlers.set(root, {onChange, actions});
  if (existing) return;
  root.addEventListener('change', event => {
    const changeAction = event.target.closest('[data-change-action]');
    if (changeAction && root.contains(changeAction)) {
      handlers.get(root)?.actions?.[changeAction.dataset.changeAction]?.(changeAction, event);
      return;
    }
    const input = event.target.closest('[data-bind]');
    if (!input || !root.contains(input) || input.dataset.kind === 'range') return;
    delete input.dataset.draft;
    const result = readControl(input);
    setFieldError(input, result.ok ? '' : result.message);
    if (result.ok) apply(root, input, result.value);
    else toast(result.message, 'bad');
  });
  root.addEventListener('input', event => {
    const input = event.target;
    if (input.dataset?.kind === 'range') apply(root, input, Number(input.value));
    // Typed but not yet committed: a re-render keeps it (see dom.replaceHTML).
    else if (input.dataset?.bind && (input.tagName === 'TEXTAREA' || (input.tagName === 'INPUT' && !['checkbox', 'radio'].includes(input.type)))) input.dataset.draft = '1';
  });
  root.addEventListener('keydown', event => {
    const input = event.target;
    if (input.dataset?.kind !== 'number') return;
    if (event.key === 'Enter') { input.dispatchEvent(new Event('change', {bubbles: true})); input.select(); }
    if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
      event.preventDefault();
      stepInput(root, input, event.key === 'ArrowUp' ? 1 : -1, event.shiftKey ? 10 : 1);
    }
  });
  root.addEventListener('click', event => {
    const step = event.target.closest('[data-step-dir]');
    if (step && root.contains(step)) {
      const input = step.closest('.num')?.querySelector('input');
      if (input) stepInput(root, input, Number(step.dataset.stepDir), event.shiftKey ? 10 : 1);
      return;
    }
    const setter = event.target.closest('[data-set-path]');
    if (setter && root.contains(setter)) {
      const value = JSON.parse(setter.dataset.setValue);
      setPath(state, setter.dataset.setPath, value);
      for (const b of $$(`[data-set-path="${CSS.escape(setter.dataset.setPath)}"]`, root)) b.classList.toggle('active', b === setter);
      handlers.get(root)?.onChange?.(setter.dataset.setPath, value, setter);
      return;
    }
    const action = event.target.closest('[data-action]');
    if (action && root.contains(action) && !action.disabled) {
      const fn = handlers.get(root)?.actions?.[action.dataset.action];
      if (fn) { event.preventDefault(); fn(action, event); }
    }
  });
}

function stepInput(root, input, direction, multiplier = 1) {
  const offset = Number(input.dataset.offset) || 0;
  const current = parseQuantity(input.value, input.dataset.unit || '');
  const base = Number.isFinite(current) ? current - offset : 0;
  const step = stepFor(input, base + offset) * multiplier;
  let value = Number((base + direction * step).toPrecision(12));
  if (input.dataset.min !== undefined) value = Math.max(Number(input.dataset.min), value);
  if (input.dataset.max !== undefined) value = Math.min(Number(input.dataset.max), value);
  if (input.dataset.integer) value = Math.round(value);
  const outside = outsideWithin(input, value);
  if (outside) { setFieldError(input, outside); toast(outside, 'bad'); return; }
  input.value = formatQuantity(value + offset, input.dataset.unit || '');
  setFieldError(input, '');
  apply(root, input, value);
}
