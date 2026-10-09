/* DOM helpers shared by every module. */

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const ESCAPES = {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'};
export function esc(value = '') {
  return String(value ?? '').replace(/[&<>"']/g, c => ESCAPES[c]);
}

/* Fold (collapsible section) open states survive re-renders. */
const foldState = new Map();
export function foldOpen(id, fallback = false) {
  return foldState.has(id) ? foldState.get(id) : fallback;
}
document.addEventListener('toggle', event => {
  const el = event.target;
  if (el instanceof HTMLDetailsElement && el.dataset.fold) foldState.set(el.dataset.fold, el.open);
}, true);

/**
 * Replace an element's content while keeping what a user would notice:
 * scroll positions of ``[data-keep-scroll]`` regions and the focused control.
 */
export function replaceHTML(el, html) {
  if (!el) return;
  const scrolls = $$('[data-keep-scroll]', el).map(node => [node.dataset.keepScroll, node.scrollTop, node.scrollLeft]);
  if (el.dataset.keepScroll) scrolls.push([el.dataset.keepScroll, el.scrollTop, el.scrollLeft, true]);
  const active = document.activeElement;
  const focusKey = active && el.contains(active) ? (active.dataset.bind || active.id || null) : null;
  // Text being typed into the focused field when a re-render arrives (a plan
  // answering the previous edit, say) is kept rather than replaced.
  const draft = focusKey && active.dataset.draft ? [active.value, active.selectionStart, active.selectionEnd] : null;
  el.innerHTML = html;
  for (const [key, top, left, self] of scrolls) {
    const node = self ? el : $(`[data-keep-scroll="${CSS.escape(key)}"]`, el);
    if (node) { node.scrollTop = top; node.scrollLeft = left; }
  }
  if (focusKey) {
    const target = $(`[data-bind="${CSS.escape(focusKey)}"]`, el) || document.getElementById(focusKey);
    if (target && target.tagName !== 'BUTTON') {
      if (draft && 'value' in target) {
        target.value = draft[0];
        target.dataset.draft = '1';
      }
      target.focus({preventScroll: true});
      if (draft && draft[1] != null) try { target.setSelectionRange(draft[1], draft[2]); } catch { /* not a text field */ }
    }
  }
}

export function setText(id, text) {
  const el = typeof id === 'string' ? document.getElementById(id) : id;
  if (el && el.textContent !== String(text)) el.textContent = text;
}

export function on(root, type, selector, handler, options) {
  root.addEventListener(type, event => {
    const target = event.target.closest?.(selector);
    if (target && root.contains(target)) handler(event, target);
  }, options);
}

export function download(blob, filename) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.append(a);
  a.click();
  setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 2000);
}

export function pickFile(accept) {
  return new Promise(resolve => {
    const input = Object.assign(document.createElement('input'), {type: 'file', accept});
    input.onchange = () => resolve(input.files?.[0] || null);
    input.click();
  });
}
