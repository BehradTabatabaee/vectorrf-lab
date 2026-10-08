/* Shell feedback: toasts, status bar, engine error banner and dialogs. */
import {$, esc} from './dom.js';
import {fmt} from './format.js';
import {state} from './state.js';
import {activeRate, sourceLabel} from './source.js';

export function toast(message, kind = '') {
  let host = $('.toast-host');
  if (!host) {
    host = Object.assign(document.createElement('div'), {className: 'toast-host'});
    host.setAttribute('role', 'status');
    document.body.append(host);
  }
  // One notice is enough: the same message already on screen is not repeated.
  if ([...host.children].some(t => t.textContent === String(message) && t.classList.contains('show'))) return;
  const el = Object.assign(document.createElement('div'), {className: `toast ${kind}`, textContent: message});
  host.append(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => { el.classList.remove('show'); setTimeout(() => el.remove(), 200); }, kind === 'bad' ? 5200 : 2600);
}

let statusText = 'ready';
let statusBad = false;

/** Status message on the right of the status bar. */
export function setStatus(message, bad = false) {
  statusText = String(message);
  statusBad = bad;
  renderStatus();
}

/** Recompute every status-bar field from the current state (never stale after a source switch). */
export function renderStatus() {
  if (!state.ready) {
    const message = document.getElementById('statusMessage');
    if (message) message.textContent = statusText;
    return;
  }
  const src = sourceLabel();
  const radar = state.mode === 'radar';
  const fields = {
    statusSource: radar ? 'RADAR SIMULATION' : `${src.kind === 'file' ? 'RECORDING' : 'TRANSMITTER'} · ${src.text}`,
    statusFs: fmt(radar ? state.studio?.sample_rate : activeRate(), 'S/s'),
  };
  for (const [id, text] of Object.entries(fields)) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }
  const message = document.getElementById('statusMessage');
  if (message) {
    message.textContent = statusText;
    message.classList.toggle('bad', statusBad);
  }
  const pill = $('.source-pill');
  if (pill) {
    pill.classList.toggle('file', src.kind === 'file' && !radar);
    // The transmitter editor holds a rejected edit: the instruments still show the applied design.
    const pending = src.kind !== 'file' && !radar && state.txUi?.planError;
    pill.innerHTML = radar ? '<span>RADAR</span><b>simulation</b>'
      : `<span>${src.kind === 'file' ? 'FILE' : 'TX'}</span><b>${esc(src.text)}</b><span>${esc(fmt(activeRate(), 'S/s'))}</span>`
        + (pending ? '<em title="The transmitter editor holds an edit that was not applied">edit not applied</em>' : '');
  }
}

let rateCount = 0;
let rateSince = 0;
/** Count displayed frames; the status bar shows the achieved update rate. */
/** Show which share of each capture real-time processing keeps (hidden at 100 %). */
export function showWork(metrics) {
  const el = document.getElementById('statusWork');
  if (!el) return;
  const share = metrics?.window_samples ? metrics.processed_samples / metrics.window_samples : 1;
  el.hidden = !(share < 0.999) || state.mode === 'radar';
  if (!el.hidden) document.getElementById('statusWorkValue').textContent = `${(100 * share).toFixed(share < 0.1 ? 1 : 0)} %`;
}

export function countFrame() {
  const now = performance.now();
  rateCount += 1;
  if (!rateSince) rateSince = now;
  if (now - rateSince >= 1000) {
    const el = document.getElementById('statusRate');
    if (el) el.textContent = `${(rateCount * 1000 / (now - rateSince)).toFixed(1)} Hz`;
    rateCount = 0;
    rateSince = now;
  }
}

/**
 * One-line bar above the status bar for a rejected live update (click for the
 * full text).  ``scope`` names the rejected part; a rejected receiver or chain
 * only disables that product, anything else leaves every display on its last
 * valid capture.
 */
export function showEngineError(error) {
  const el = $('#engineError');
  if (!el) return;
  const message = error?.message;
  // The transmitter editor explains its own rejected designs in place.
  const shownInPlace = error?.scope === 'tx' && state.mode === 'tx';
  el.hidden = !message || shownInPlace;
  if (!message) { el.classList.remove('open'); return; }
  const product = error.scope === 'rx' || error.scope === 'pipeline';
  const tail = product ? 'That view is paused; the others continue.' : 'The displays keep the last valid capture.';
  el.innerHTML = `<b>Not applied</b><span>${esc(message)}</span><em>${esc(tail)}</em>`;
  el.title = `${message} ${tail}`;
}

/* Hover tooltips for labels carrying ``data-tip`` (their explanation). */
let tipEl = null, tipTarget = null, tipTimer = null;
function hideTip() {
  clearTimeout(tipTimer);
  tipTarget = null;
  tipEl?.remove();
}
function showTip(target) {
  tipEl ||= Object.assign(document.createElement('div'), {className: 'tooltip', role: 'tooltip'});
  tipEl.innerHTML = target.dataset.tip;
  document.body.append(tipEl);
  const r = target.getBoundingClientRect();
  const left = Math.max(8, Math.min(window.innerWidth - tipEl.offsetWidth - 8, r.left));
  let top = r.bottom + 6;
  if (top + tipEl.offsetHeight > window.innerHeight - 8) top = Math.max(8, r.top - tipEl.offsetHeight - 6);
  tipEl.style.left = `${left}px`;
  tipEl.style.top = `${top}px`;
}
document.addEventListener('mouseover', event => {
  const target = event.target.closest?.('[data-tip]');
  if (target === tipTarget) return;
  hideTip();
  if (!target) return;
  tipTarget = target;
  tipTimer = setTimeout(() => showTip(target), 300);
});
document.addEventListener('mousedown', hideTip, true);
document.addEventListener('scroll', hideTip, true);
window.addEventListener('blur', hideTip);

/** Modal dialog; resolves with the value passed to ``close``. */
export function dialog(title, bodyHTML, {actions = '', onMount} = {}) {
  return new Promise(resolve => {
    const backdrop = Object.assign(document.createElement('div'), {className: 'dialog-backdrop'});
    backdrop.innerHTML = `<div class="dialog" role="dialog" aria-modal="true" aria-label="${esc(title)}"><header>${esc(title)}<button class="btn ghost small" data-close>Close</button></header><div class="dialog-body">${bodyHTML}</div>${actions ? `<footer>${actions}</footer>` : ''}</div>`;
    const close = value => { backdrop.remove(); document.removeEventListener('keydown', onKey); resolve(value); };
    const onKey = event => { if (event.key === 'Escape') close(null); };
    backdrop.addEventListener('click', event => { if (event.target === backdrop || event.target.closest('[data-close]')) close(null); });
    document.addEventListener('keydown', onKey);
    document.body.append(backdrop);
    onMount?.(backdrop, close);
  });
}

export function confirmDialog(title, message, confirmLabel = 'Confirm', danger = false) {
  return dialog(title, `<p style="margin:0;line-height:1.6;color:var(--muted)">${esc(message)}</p>`, {
    actions: `<button class="btn ghost" data-close>Cancel</button><button class="btn ${danger ? 'danger' : 'primary'}" data-confirm>${esc(confirmLabel)}</button>`,
    onMount: (root, close) => root.querySelector('[data-confirm]').addEventListener('click', () => close(true)),
  });
}
