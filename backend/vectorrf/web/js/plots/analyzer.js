/* Swept-analyzer display: graticule, traces, markers, band selection,
   channel-power windows and waterfall, driven by mouse and keyboard.

   Frequencies inside the component are complex-baseband Hz; ``axisOffset()``
   adds the RF centre for labels and readouts only. */
import {esc} from '../core/dom.js';
import {fmt, fmtFreq} from '../core/format.js';
import {Waterfall, clear, colors, niceTicks, setup} from './canvas.js';

const MARGIN = {left: 58, right: 12, top: 26, bottom: 30};
const HANDLE_PX = 7;
export const CHANNEL_COLORS = ['#5cc8e6', '#c79bf2', '#7fd99a', '#f29e6b', '#6aa7f0', '#e8d26b', '#f07fb0', '#9ad0c2'];

function lowerBound(a, v) {
  let lo = 0, hi = a.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (a[mid] < v) lo = mid + 1; else hi = mid; }
  return lo;
}

function nearestIndex(freq, hz) {
  if (!freq?.length) return -1;
  let k = lowerBound(freq, hz);
  if (k >= freq.length) k = freq.length - 1;
  else if (k > 0 && Math.abs(freq[k - 1] - hz) < Math.abs(freq[k] - hz)) k -= 1;
  return k;
}

/** Local maxima of ``y`` (strongest first) inside [a, b). */
function peaks(y, a = 0, b = y.length) {
  const out = [];
  for (let i = Math.max(1, a); i < Math.min(y.length - 1, b); i++) {
    if (y[i] >= y[i - 1] && y[i] > y[i + 1] && Number.isFinite(y[i])) out.push(i);
  }
  return out.sort((i, j) => y[j] - y[i]);
}

/** Span rounded to three significant figures (what an instrument keypad shows). */
export function niceSpan(span) {
  const mag = 10 ** (Math.floor(Math.log10(span)) - 2);
  return Math.round(span / mag) * mag;
}

/** Centre rounded to about a thousandth of the span on a 1-2-5 grid. */
export function niceCenter(center, span) {
  const raw = span / 1000, mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map(k => k * mag).find(v => v >= raw);
  return Math.round(center / step) * step;
}

/** The coarsest 1-2-5 step not larger than ``limit``. */
function stepBelow(limit) {
  const mag = 10 ** Math.floor(Math.log10(limit));
  return [5, 2, 1].map(k => k * mag).find(v => v <= limit);
}

export class Analyzer {
  /**
   * @param root element holding the plot, strip and waterfall
   * @param ui   persistent display settings (state.specUi or a private copy)
   * @param opts {getTune, onTune, axisOffset, unit, rbw, waterfall, channels, onChannels,
   *              selectionActions, onSelectionAction, bands, onBandClick, onChange}
   */
  constructor(root, ui, opts = {}) {
    this.root = root;
    this.ui = ui;
    this.opts = opts;
    this.canvas = root.querySelector('.analyzer-plot canvas');
    this.wfCanvas = root.querySelector('.analyzer-waterfall canvas');
    this.strip = root.querySelector('.marker-table');
    this.selectionBar = root.querySelector('.selection-bar');
    this.freq = [];
    this.psd = null;
    this.waterfall = opts.waterfall === false ? null : new Waterfall(640, 180);
    this.signature = '';
    this.hover = null;
    this.drag = null;
    this.bindEvents();
  }

  /* ------------------------------------------------------------ geometry */

  tune() { return this.opts.getTune(); }

  view() {
    const {center, span} = this.tune();
    return {left: center - span / 2, right: center + span / 2, center, span};
  }

  plotRect() {
    const r = this.canvas.getBoundingClientRect();
    return {x: MARGIN.left, y: MARGIN.top, w: Math.max(1, r.width - MARGIN.left - MARGIN.right), h: Math.max(1, r.height - MARGIN.top - MARGIN.bottom), cw: r.width, ch: r.height};
  }

  hzAt(px) {
    const v = this.view(), p = this.plotRect();
    return v.left + (px - p.x) / p.w * v.span;
  }

  /** Frequency the pointer selects: rounded to the coarsest 1-2-5 step within one pixel. */
  pointerHz(px) {
    const step = stepBelow(this.view().span / this.plotRect().w);
    return Number((Math.round(this.hzAt(px) / step) * step).toPrecision(15));
  }

  pxAt(hz) {
    const v = this.view(), p = this.plotRect();
    return p.x + (hz - v.left) / v.span * p.w;
  }

  levels() {
    const ui = this.ui;
    const top = Number(ui.ref), bottom = top - 10 * Number(ui.perDiv);
    return {top, bottom};
  }

  pyAt(db) {
    const {top, bottom} = this.levels(), p = this.plotRect();
    return p.y + (top - db) / (top - bottom) * p.h;
  }

  /* ---------------------------------------------------------------- data */

  /** New analyzer frame: ``freq`` ascending baseband Hz, ``psd`` dB. */
  ingest(freq, psd, signature) {
    if (!freq?.length || !psd?.length || freq.length !== psd.length) return;
    if (signature !== this.signature) {
      this.signature = signature;
      for (const t of this.ui.traces) t.y = null;
      this.waterfall?.reset();
    }
    this.freq = freq;
    this.psd = psd;
    if (this.ui.hold) return;
    for (const t of this.ui.traces) {
      if (t.mode === 'off' || t.mode === 'view') continue;
      if (!t.y || t.y.length !== psd.length || t.mode === 'clear') { t.y = Float32Array.from(psd); continue; }
      const a = 1 / Math.max(1, this.ui.traceAverage || 16);
      for (let i = 0; i < psd.length; i++) {
        if (t.mode === 'max') t.y[i] = Math.max(t.y[i], psd[i]);
        else if (t.mode === 'min') t.y[i] = Math.min(t.y[i], psd[i]);
        else if (t.mode === 'average') t.y[i] = 10 * Math.log10((1 - a) * 10 ** (t.y[i] / 10) + a * 10 ** (psd[i] / 10));
      }
    }
    this.updateAutoRef();
    if (this.waterfall) this.waterfall.push(this.viewRow(psd), this.levels().bottom, this.levels().top);
  }

  resetTraces() {
    for (const t of this.ui.traces) t.y = null;
    this.waterfall?.reset();
  }

  /** Values of ``row`` sampled at 640 evenly spaced view frequencies (NaN outside the data). */
  viewRow(row) {
    const v = this.view(), n = this.waterfall.off.width, out = new Float32Array(n).fill(NaN);
    const f = this.freq;
    if (!f.length) return out;
    for (let i = 0; i < n; i++) {
      const hz = v.left + (i + 0.5) / n * v.span;
      if (hz < f[0] || hz > f.at(-1)) continue;
      out[i] = row[nearestIndex(f, hz)];
    }
    return out;
  }

  /** Auto reference level with hysteresis, so the scale does not jump every frame. */
  updateAutoRef() {
    const ui = this.ui;
    if (!ui.autoRef) return;
    const peak = this.visiblePeak()?.db;
    if (!Number.isFinite(peak)) return;
    const div = Number(ui.perDiv);
    const target = Math.ceil((peak + 0.5 * div) / div) * div;
    if (peak > ui.ref - 0.25 * div || ui.ref - peak > 2.5 * div) ui.ref = target;
  }

  traceData(index = 0) {
    const t = this.ui.traces[index];
    return t && t.mode !== 'off' && t.y ? t.y : this.psd;
  }

  /** Indices of the display points inside the view. */
  visibleRange() {
    const v = this.view();
    return [Math.max(0, lowerBound(this.freq, v.left)), Math.min(this.freq.length, lowerBound(this.freq, v.right + 1e-9))];
  }

  visiblePeak(trace = 0) {
    const y = this.traceData(trace);
    if (!y) return null;
    const [a, b] = this.visibleRange();
    let k = -1;
    for (let i = a; i < b; i++) if (Number.isFinite(y[i]) && (k < 0 || y[i] > y[k])) k = i;
    return k < 0 ? null : {hz: this.freq[k], db: y[k], index: k};
  }

  levelAt(hz, trace = 0) {
    const y = this.traceData(trace);
    const k = nearestIndex(this.freq, hz);
    return k < 0 || !y ? null : {hz: this.freq[k], db: y[k], index: k};
  }

  /* ------------------------------------------------------------- markers */

  get markers() { return this.ui.markers; }

  activeMarker() { return this.ui.markers[this.ui.activeMarker] || null; }

  addMarker(hz) {
    if (this.ui.markers.length >= 8) return null;
    const marker = {hz, trace: 0, delta: false, noise: false, ref: 0};
    this.ui.markers.push(marker);
    this.ui.activeMarker = this.ui.markers.length - 1;
    this.changed();
    return marker;
  }

  setActive(hz) {
    const marker = this.activeMarker() || this.addMarker(hz);
    if (!marker) return;
    marker.hz = hz;
    this.changed();
  }

  removeMarker(index = this.ui.activeMarker) {
    if (index < 0 || index >= this.ui.markers.length) return;
    this.ui.markers.splice(index, 1);
    for (const m of this.ui.markers) if (m.ref >= this.ui.markers.length) m.ref = 0;
    this.ui.activeMarker = Math.min(this.ui.activeMarker, this.ui.markers.length - 1);
    this.changed();
  }

  peakSearch() {
    const p = this.visiblePeak(this.activeMarker()?.trace || 0);
    if (p) this.setActive(p.hz);
  }

  /** Next lower peak (direction 0) or the nearest peak to the left (-1) / right (+1). */
  nextPeak(direction = 0) {
    const m = this.activeMarker();
    const y = this.traceData(m?.trace || 0);
    if (!y) return;
    const [a, b] = this.visibleRange();
    const list = peaks(y, a, b);
    if (!list.length) return;
    if (!m) { this.setActive(this.freq[list[0]]); return; }
    const here = nearestIndex(this.freq, m.hz);
    let pick;
    if (direction === 0) {
      const level = y[here];
      pick = list.find(i => y[i] < level - 1e-9 && i !== here);
    } else {
      const side = list.filter(i => (direction > 0 ? i > here : i < here));
      pick = side.sort((i, j) => Math.abs(i - here) - Math.abs(j - here))[0];
    }
    if (pick !== undefined) this.setActive(this.freq[pick]);
  }

  markerReadout(marker, index) {
    const offset = this.opts.axisOffset?.() || 0;
    const unit = this.opts.unit?.() || 'dBm';
    const at = this.levelAt(marker.hz, marker.trace);
    const resolution = this.view().span / 2000;
    if (!at) return {label: `M${index + 1}`, freq: '—', level: '—'};
    if (marker.delta && this.ui.markers[marker.ref] && marker.ref !== index) {
      const ref = this.levelAt(this.ui.markers[marker.ref].hz, marker.trace);
      return {label: `D${index + 1}`, delta: true, freq: `Δ ${fmt(at.hz - ref.hz, 'Hz')}`, level: `Δ ${(at.db - ref.db).toFixed(2)} dB`, ref: marker.ref + 1};
    }
    if (marker.noise) {
      const rbw = this.opts.rbw?.() || 1;
      return {label: `N${index + 1}`, freq: fmtFreq(at.hz + offset, resolution), level: `${(at.db - 10 * Math.log10(rbw)).toFixed(2)} ${unit}/Hz`};
    }
    return {label: `M${index + 1}`, freq: fmtFreq(at.hz + offset, resolution), level: `${at.db.toFixed(2)} ${unit}`};
  }

  renderStrip() {
    if (!this.strip) return;
    const chips = this.ui.markers.map((m, i) => {
      const r = this.markerReadout(m, i);
      return `<div class="marker-chip${i === this.ui.activeMarker ? ' active' : ''}" data-marker="${i}" title="Click to select · Del removes"><span class="m-tag">${esc(r.label)}</span>`
        + `<b class="${r.delta ? 'm-delta' : ''}">${esc(r.freq)}</b><b class="${r.delta ? 'm-delta' : ''}">${esc(r.level)}</b>`
        + `${r.delta ? `<span>ref M${r.ref}</span>` : ''}<button type="button" data-marker-remove="${i}" aria-label="Remove marker ${i + 1}">×</button></div>`;
    });
    const unit = this.opts.unit?.() || 'dBm';
    const channels = (this.opts.channels?.() || []).map((band, i) => {
      const m = (this.opts.channelResults?.() || [])[i];
      const power = m?.power_db != null ? `${m.power_db.toFixed(2)} ${m.unit || unit}` : m && !m.covered ? 'outside span' : '—';
      return `<div class="channel-chip" title="Channel ${i + 1}: drag the band on the trace to move or resize it"><i style="background:${CHANNEL_COLORS[i % 8]}"></i><span>${esc(band.label || `CH${i + 1}`)}</span><b>${esc(power)}</b>`
        + `${m?.density_db_hz != null ? `<span>${m.density_db_hz.toFixed(1)} ${esc(m.unit || unit)}/Hz</span>` : ''}${m?.obw99_hz ? `<span>OBW ${esc(fmt(m.obw99_hz, 'Hz'))}</span>` : ''}</div>`;
    });
    this.strip.innerHTML = chips.join('') + channels.join('')
      || `<div class="marker-chip" style="cursor:default">${this.opts.hint ?? 'Click the trace to place a marker · drag to select a band · wheel to zoom'}</div>`;
  }

  changed() {
    this.renderStrip();
    this.opts.onChange?.();
    this.render();
  }

  /* ----------------------------------------------------------- selection */

  setSelection(sel) {
    this.ui.selection = sel && sel.hi - sel.lo > 0 ? sel : null;
    this.render();
    this.placeSelectionBar();
  }

  placeSelectionBar() {
    const bar = this.selectionBar;
    if (!bar) return;
    const sel = this.ui.selection;
    if (!sel || !(this.opts.selectionActions?.length)) { bar.hidden = true; return; }
    const offset = this.opts.axisOffset?.() || 0;
    const mid = (this.pxAt(sel.lo) + this.pxAt(sel.hi)) / 2;
    const p = this.plotRect();
    bar.hidden = false;
    bar.style.left = `${Math.max(p.x + 140, Math.min(p.cw - 140, mid))}px`;
    bar.style.top = `${MARGIN.top + 4}px`;
    bar.innerHTML = `<span class="sel-info">${esc(fmtFreq((sel.lo + sel.hi) / 2 + offset, (sel.hi - sel.lo) / 1000))} · ${esc(fmt(sel.hi - sel.lo, 'Hz'))}</span>`
      + this.opts.selectionActions.map(a => `<button type="button" class="btn small${a.primary ? ' primary' : ''}" data-selection-action="${esc(a.id)}" title="${esc(a.title || '')}">${esc(a.label)}</button>`).join('')
      + '<button type="button" class="btn small ghost" data-selection-action="clear" aria-label="Clear selection">×</button>';
  }

  /* --------------------------------------------------------------- input */

  bindEvents() {
    const canvas = this.canvas;
    canvas.tabIndex = 0;
    canvas.addEventListener('mousedown', e => this.onDown(e));
    canvas.addEventListener('mousemove', e => this.onHover(e));
    canvas.addEventListener('mouseleave', () => { this.hover = null; this.render(); });
    canvas.addEventListener('dblclick', e => { e.preventDefault(); this.opts.onTune?.('full'); });
    canvas.addEventListener('wheel', e => this.onWheel(e), {passive: false});
    canvas.addEventListener('contextmenu', e => e.preventDefault());
    canvas.addEventListener('keydown', e => this.onKey(e));
    this.onMove = e => this.onDragMove(e);
    this.onUp = e => this.onDragEnd(e);
    this.strip?.addEventListener('click', e => {
      const remove = e.target.closest('[data-marker-remove]');
      if (remove) { this.removeMarker(Number(remove.dataset.markerRemove)); return; }
      const chip = e.target.closest('[data-marker]');
      if (chip) { this.ui.activeMarker = Number(chip.dataset.marker); this.changed(); }
    });
    this.selectionBar?.addEventListener('click', e => {
      const b = e.target.closest('[data-selection-action]');
      if (!b) return;
      const sel = this.ui.selection;
      if (b.dataset.selectionAction === 'clear') { this.setSelection(null); return; }
      if (sel) this.opts.onSelectionAction?.(b.dataset.selectionAction, sel);
    });
  }

  local(e) {
    const r = this.canvas.getBoundingClientRect();
    return {x: e.clientX - r.left, y: e.clientY - r.top};
  }

  /** What is under the pointer: a marker, a band edge/body, or empty plot. */
  hit(x) {
    const markers = this.ui.markers;
    for (let i = markers.length - 1; i >= 0; i--) if (Math.abs(this.pxAt(markers[i].hz) - x) <= HANDLE_PX) return {kind: 'marker', index: i};
    const bands = [...(this.opts.channels?.() || []).map((b, i) => ({kind: 'channel', index: i, lo: b.center_hz - b.width_hz / 2, hi: b.center_hz + b.width_hz / 2})),
      ...(this.ui.selection ? [{kind: 'selection', lo: this.ui.selection.lo, hi: this.ui.selection.hi}] : [])];
    for (const band of bands.reverse()) {
      const a = this.pxAt(band.lo), b = this.pxAt(band.hi);
      if (Math.abs(x - a) <= HANDLE_PX) return {...band, edge: 'lo'};
      if (Math.abs(x - b) <= HANDLE_PX) return {...band, edge: 'hi'};
      if (x > a && x < b) return {...band, edge: 'body'};
    }
    return {kind: 'empty'};
  }

  onDown(e) {
    const {x, y} = this.local(e);
    const p = this.plotRect();
    if (x < p.x || x > p.x + p.w || y < p.y - 20 || y > p.y + p.h) return;
    this.canvas.focus({preventScroll: true});
    e.preventDefault();
    const hz = this.pointerHz(x);
    if (e.button === 1 || e.button === 2 || e.altKey) {
      this.drag = {kind: 'pan', x0: x, center0: this.tune().center, moved: false};
      this.canvas.classList.add('panning');
    } else {
      const target = this.hit(x);
      if (target.kind === 'marker' && !e.ctrlKey && !e.metaKey) {
        this.ui.activeMarker = target.index;
        this.drag = {kind: 'marker', index: target.index, moved: false};
      } else if ((target.kind === 'channel' || target.kind === 'selection') && !e.shiftKey) {
        this.drag = {...target, x0: x, hz0: hz, lo0: target.lo, hi0: target.hi, moved: false};
      } else {
        this.drag = {kind: 'select', x0: x, hz0: hz, moved: false, add: e.ctrlKey || e.metaKey};
      }
    }
    window.addEventListener('mousemove', this.onMove);
    window.addEventListener('mouseup', this.onUp);
  }

  onDragMove(e) {
    const d = this.drag;
    if (!d) return;
    const {x} = this.local(e);
    const hz = this.pointerHz(x);
    if (Math.abs(x - (d.x0 ?? x)) > 3) d.moved = true;
    if (d.kind === 'pan') {
      if (!d.moved) return;
      const span = this.tune().span;
      const shift = (d.x0 - x) / this.plotRect().w * span;
      this.opts.onTune?.({center: niceCenter(d.center0 + shift, span)}, true);
    } else if (d.kind === 'marker') {
      d.moved = true;
      this.ui.markers[d.index].hz = hz;
      this.changed();
    } else if (d.kind === 'select') {
      if (!d.moved) return;
      this.ui.selection = {lo: Math.min(d.hz0, hz), hi: Math.max(d.hz0, hz)};
      this.render();
    } else if (d.kind === 'channel' || d.kind === 'selection') {
      if (!d.moved) return;
      let lo = d.lo0, hi = d.hi0;
      if (d.edge === 'lo') lo = Math.min(hz, hi - this.tune().span / 2000);
      else if (d.edge === 'hi') hi = Math.max(hz, lo + this.tune().span / 2000);
      else { lo += hz - d.hz0; hi += hz - d.hz0; }
      if (d.kind === 'selection') this.ui.selection = {lo, hi};
      else this.opts.onChannels?.(d.index, {center_hz: (lo + hi) / 2, width_hz: hi - lo}, false);
      this.render();
    }
  }

  onDragEnd(e) {
    window.removeEventListener('mousemove', this.onMove);
    window.removeEventListener('mouseup', this.onUp);
    this.canvas.classList.remove('panning');
    const d = this.drag;
    this.drag = null;
    if (!d) return;
    const {x} = this.local(e);
    const hz = this.pointerHz(x);
    if (d.kind === 'pan') { if (d.moved) this.opts.onTune?.({center: this.tune().center}); return; }
    if (d.kind === 'marker') { this.changed(); return; }
    if (d.kind === 'channel' && d.moved) { this.opts.onChannels?.(d.index, null, true); return; }
    if (d.kind === 'selection' && d.moved) { this.placeSelectionBar(); return; }
    if (d.kind === 'select' && d.moved) { this.setSelection(this.ui.selection); return; }
    // A click without movement places the marker (Ctrl/Cmd adds a new one).
    if (d.add) this.addMarker(hz); else this.setActive(hz);
  }

  onHover(e) {
    if (this.drag) return;
    const {x, y} = this.local(e);
    const p = this.plotRect();
    const inside = x >= p.x && x <= p.x + p.w && y >= p.y && y <= p.y + p.h;
    this.hover = inside ? {x, y} : null;
    const target = inside ? this.hit(x) : {kind: 'empty'};
    this.canvas.classList.toggle('ew', target.kind === 'marker' || ['lo', 'hi'].includes(target.edge));
    this.canvas.classList.toggle('move', target.edge === 'body');
    this.render();
  }

  onWheel(e) {
    e.preventDefault();
    const {x} = this.local(e);
    const t = this.tune();
    if (e.shiftKey) {
      this.opts.onTune?.({center: t.center + Math.sign(e.deltaY || e.deltaX) * t.span * 0.1}, false);
      return;
    }
    const anchor = this.hzAt(x);
    // One wheel notch (deltaY ≈ 100) zooms by 1.25; trackpads send smaller steps.
    const notches = Math.max(-8, Math.min(8, (e.deltaMode === 1 ? e.deltaY * 33 : e.deltaY) / 100));
    const factor = 1.25 ** notches;
    const span = niceSpan(t.span * factor);
    this.opts.onTune?.({span, center: niceCenter(anchor + (t.center - anchor) * span / t.span, span)}, false);
  }

  onKey(e) {
    const step = (this.visibleRange()[1] - this.visibleRange()[0]) > 1 ? this.view().span / Math.max(100, this.freq.length) : 1;
    const m = this.activeMarker();
    const key = e.key;
    const handled = () => e.preventDefault();
    if ((key === 'ArrowLeft' || key === 'ArrowRight') && m) { m.hz += (key === 'ArrowLeft' ? -1 : 1) * step * (e.shiftKey ? 10 : 1); this.changed(); handled(); }
    else if (key === 'p' || key === 'P') { this.peakSearch(); handled(); }
    else if (key === 'n' || key === 'N') { this.nextPeak(0); handled(); }
    else if (key === '[') { this.nextPeak(-1); handled(); }
    else if (key === ']') { this.nextPeak(1); handled(); }
    else if (key === 'm' || key === 'M') { this.addMarker(this.tune().center); handled(); }
    else if (key === 'Delete' || key === 'Backspace') { this.removeMarker(); handled(); }
    else if (key === 'f' || key === 'F') { this.opts.onTune?.('full'); handled(); }
    else if ((key === 'z' || key === 'Z') && this.ui.selection) { this.opts.onSelectionAction?.('zoom', this.ui.selection); handled(); }
    else if (key === 'c' || key === 'C') { if (m) this.opts.onTune?.({center: m.hz}); handled(); }
    else if (key === 'Escape') { this.setSelection(null); handled(); }
    else if (key === '+' || key === '=') { this.opts.onTune?.({span: this.tune().span / 2}); handled(); }
    else if (key === '-' || key === '_') { this.opts.onTune?.({span: this.tune().span * 2}); handled(); }
  }

  /* -------------------------------------------------------------- render */

  render() {
    const s = setup(this.canvas);
    if (!s) return;
    const {ctx, w, h} = s;
    const c = colors();
    clear(ctx, w, h);
    const p = {x: MARGIN.left, y: MARGIN.top, w: Math.max(1, w - MARGIN.left - MARGIN.right), h: Math.max(1, h - MARGIN.top - MARGIN.bottom)};
    const v = this.view();
    const {top, bottom} = this.levels();
    const offset = this.opts.axisOffset?.() || 0;
    const unit = this.opts.unit?.() || 'dBm';
    const X = hz => p.x + (hz - v.left) / v.span * p.w;
    const Y = db => p.y + (top - db) / (top - bottom) * p.h;

    // 10 × 10 graticule.
    ctx.strokeStyle = c.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < 10; i++) {
      const gx = Math.round(p.x + i * p.w / 10) + .5, gy = Math.round(p.y + i * p.h / 10) + .5;
      ctx.moveTo(gx, p.y); ctx.lineTo(gx, p.y + p.h);
      ctx.moveTo(p.x, gy); ctx.lineTo(p.x + p.w, gy);
    }
    ctx.stroke();
    ctx.strokeStyle = c.gridMajor;
    ctx.strokeRect(p.x + .5, p.y + .5, p.w - 1, p.h - 1);
    ctx.font = `10px ${c.mono}`;
    ctx.fillStyle = c.text;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    // Label every division, or every 2nd/5th when the graticule is too short for that.
    const labelEvery = [1, 2, 5, 10].find(n => n * p.h / 10 >= 13) ?? 10;
    for (let i = 0; i <= 10; i += labelEvery) ctx.fillText((top - i * (top - bottom) / 10).toFixed(Number(this.ui.perDiv) < 1 ? 1 : 0), p.x - 6, p.y + i * p.h / 10);

    // Region outside the source's sampled band.
    const data = this.opts.dataBounds?.();
    if (data) {
      ctx.fillStyle = 'rgba(40,46,52,.45)';
      const shade = (a, b) => { const x0 = Math.max(p.x, X(a)), x1 = Math.min(p.x + p.w, X(b)); if (x1 > x0) ctx.fillRect(x0, p.y, x1 - x0, p.h); return x1 - x0; };
      for (const [a, b] of [[v.left, data.left], [data.right, v.right]]) {
        if (shade(a, b) > 120) {
          ctx.fillStyle = '#5b666e'; ctx.textAlign = 'center';
          ctx.fillText('OUTSIDE SOURCE NYQUIST', (Math.max(p.x, X(a)) + Math.min(p.x + p.w, X(b))) / 2, p.y + p.h / 2);
          ctx.fillStyle = 'rgba(40,46,52,.45)';
        }
      }
    }

    ctx.save();
    ctx.beginPath();
    ctx.rect(p.x, p.y, p.w, p.h);
    ctx.clip();
    // Overlay bands (component plans, detected emitters): a faint fill over
    // their width and the label above; a zero-width band (a tone) gets only
    // its label.  Labels that would overlap a previous one are left out.
    const taken = [];
    for (const band of this.opts.bands?.() || []) {
      const a = X(band.lo), b = X(band.hi);
      if (b < p.x || a > p.x + p.w) continue;
      if (b - a >= 2) {
        ctx.fillStyle = band.fill || 'rgba(92,200,230,.06)';
        ctx.fillRect(a, p.y, b - a, p.h);
      }
      if (!band.label) continue;
      const mid = Math.min(p.x + p.w - 4, Math.max(p.x + 4, (a + b) / 2));
      const half = ctx.measureText(band.label).width / 2 + 4;
      if (taken.some(([l, r]) => mid + half > l && mid - half < r)) continue;
      taken.push([mid - half, mid + half]);
      ctx.fillStyle = band.color || c.cyan; ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.fillText(band.label, mid, p.y + 3);
    }
    // Channel-power windows.
    (this.opts.channels?.() || []).forEach((band, i) => {
      const a = X(band.center_hz - band.width_hz / 2), b = X(band.center_hz + band.width_hz / 2);
      const color = CHANNEL_COLORS[i % 8];
      ctx.fillStyle = `${color}1c`;
      ctx.fillRect(a, p.y, b - a, p.h);
      ctx.strokeStyle = color;
      ctx.setLineDash([4, 3]);
      ctx.beginPath(); ctx.moveTo(a, p.y); ctx.lineTo(a, p.y + p.h); ctx.moveTo(b, p.y); ctx.lineTo(b, p.y + p.h); ctx.stroke();
      ctx.setLineDash([]);
      const result = (this.opts.channelResults?.() || [])[i];
      ctx.fillStyle = color; ctx.textAlign = 'left'; ctx.textBaseline = 'top';
      ctx.fillText(`${band.label || `CH${i + 1}`}${result?.power_db != null ? ` ${result.power_db.toFixed(2)} ${result.unit || unit}` : ''}`, Math.max(p.x + 2, a + 3), p.y + 14 + 12 * (i % 3));
    });
    // Selection.
    const sel = this.ui.selection;
    if (sel) {
      const a = X(sel.lo), b = X(sel.hi);
      ctx.fillStyle = 'rgba(233,176,75,.10)';
      ctx.fillRect(a, p.y, b - a, p.h);
      ctx.strokeStyle = c.accent;
      ctx.beginPath(); ctx.moveTo(a, p.y); ctx.lineTo(a, p.y + p.h); ctx.moveTo(b, p.y); ctx.lineTo(b, p.y + p.h); ctx.stroke();
    }
    // Traces, last first so trace 1 sits on top.
    const [ia, ib] = this.visibleRange();
    const from = Math.max(0, ia - 1), to = Math.min(this.freq.length, ib + 1);
    this.ui.traces.map((t, i) => [t, i]).reverse().forEach(([t, i]) => {
      if (t.mode === 'off' || !t.y || t.y.length !== this.freq.length) return;
      ctx.strokeStyle = c.traces[i] || c.accent;
      ctx.lineWidth = i === 0 ? 1.25 : 1.1;
      ctx.beginPath();
      const stepIdx = Math.max(1, Math.floor((to - from) / Math.max(200, p.w * 1.5)));
      let started = false;
      for (let k = from; k < to; k += stepIdx) {
        let best = k;
        for (let j = k + 1; j < Math.min(to, k + stepIdx); j++) if (t.y[j] > t.y[best]) best = j;
        const val = t.y[best];
        if (!Number.isFinite(val)) { started = false; continue; }
        const px = X(this.freq[best]), py = Y(val);
        if (started) ctx.lineTo(px, py); else { ctx.moveTo(px, py); started = true; }
      }
      ctx.stroke();
    });
    for (const mem of this.ui.memories || []) {
      ctx.strokeStyle = mem.color;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      let started = false;
      for (let k = 0; k < mem.x.length; k++) {
        if (mem.x[k] < v.left || mem.x[k] > v.right || !Number.isFinite(mem.y[k])) { started = false; continue; }
        if (started) ctx.lineTo(X(mem.x[k]), Y(mem.y[k])); else { ctx.moveTo(X(mem.x[k]), Y(mem.y[k])); started = true; }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
    // Markers: white diamond-tipped glyphs with a numbered tag (active one filled).
    ctx.font = `bold 10px ${c.mono}`;
    this.ui.markers.forEach((m, i) => {
      const at = this.levelAt(m.hz, m.trace);
      if (!at || m.hz < v.left || m.hz > v.right) return;
      const mx = X(at.hz), my = Math.max(p.y + 24, Math.min(p.y + p.h, Y(at.db)));
      const active = i === this.ui.activeMarker;
      if (active) {
        ctx.strokeStyle = 'rgba(255,255,255,.28)';
        ctx.setLineDash([2, 3]);
        ctx.beginPath(); ctx.moveTo(mx, p.y); ctx.lineTo(mx, p.y + p.h); ctx.stroke();
        ctx.setLineDash([]);
      }
      ctx.beginPath(); ctx.moveTo(mx, my - 1); ctx.lineTo(mx - 6, my - 11); ctx.lineTo(mx + 6, my - 11); ctx.closePath();
      ctx.fillStyle = active ? '#ffffff' : '#9fb0b9';
      ctx.strokeStyle = '#06080a';
      ctx.lineWidth = 1.5;
      ctx.stroke(); ctx.fill();
      const tag = `${m.delta ? 'D' : m.noise ? 'N' : 'M'}${i + 1}`;
      const tw = ctx.measureText(tag).width + 8;
      ctx.fillStyle = active ? '#ffffff' : '#2a3238';
      ctx.fillRect(mx - tw / 2, my - 25, tw, 13);
      ctx.fillStyle = active ? '#06080a' : '#d7dde1';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(tag, mx, my - 18);
      ctx.lineWidth = 1;
    });
    ctx.font = `10px ${c.mono}`;
    // Crosshair readout.
    if (this.hover) {
      const hz = this.hzAt(this.hover.x), db = top - (this.hover.y - p.y) / p.h * (top - bottom);
      ctx.strokeStyle = 'rgba(200,210,215,.22)';
      ctx.beginPath(); ctx.moveTo(this.hover.x, p.y); ctx.lineTo(this.hover.x, p.y + p.h); ctx.moveTo(p.x, this.hover.y); ctx.lineTo(p.x + p.w, this.hover.y); ctx.stroke();
      ctx.restore();
      ctx.save();
      ctx.fillStyle = '#c9d2d8'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
      ctx.fillText(`${fmtFreq(hz + offset, v.span / 1000)}  ${db.toFixed(1)} ${unit}`, p.x + p.w, MARGIN.top / 2);
    }
    ctx.restore();

    // Annotations: reference level, scale, centre/span, RBW.
    ctx.fillStyle = '#c9d2d8';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'left';
    ctx.font = `11px ${c.mono}`;
    ctx.fillText(`Ref ${top.toFixed(1)} ${unit}   ${Number(this.ui.perDiv)} dB/div${this.ui.hold ? '   HOLD' : ''}`, p.x, MARGIN.top / 2);
    const resolution = v.span / 1000;
    const y0 = p.y + p.h + 10;
    ctx.fillStyle = c.text;
    ctx.font = `10px ${c.mono}`;
    ctx.textAlign = 'left'; ctx.fillText(fmtFreq(v.left + offset, resolution), p.x, y0);
    ctx.textAlign = 'center'; ctx.fillText(`Center ${fmtFreq(v.center + offset, resolution)}`, p.x + p.w / 2, y0);
    ctx.textAlign = 'right'; ctx.fillText(fmtFreq(v.right + offset, resolution), p.x + p.w, y0);
    ctx.textAlign = 'center';
    const rbw = this.opts.rbw?.();
    ctx.fillText(`Span ${fmt(v.span, 'Hz')}${rbw ? `   RBW ${fmt(rbw, 'Hz')}` : ''}${this.opts.extraAnnotation?.() || ''}`, p.x + p.w / 2, y0 + 12);
    this.drawWaterfall();
  }

  drawWaterfall() {
    if (!this.waterfall || !this.wfCanvas) return;
    this.waterfall.draw(this.wfCanvas, MARGIN.left, MARGIN.right);
  }
}

/** Markup for an analyzer (plot, marker/channel strip, optional waterfall). */
export function analyzerMarkup({waterfall = true, tools = '', compact = false} = {}) {
  return `<div class="analyzer${waterfall ? '' : ' no-waterfall'}${compact ? ' compact' : ''}"><div class="analyzer-plot"><canvas aria-label="Spectrum display: click for marker, drag to select a band, wheel to zoom"></canvas><div class="selection-bar" hidden></div></div>`
    + `<div class="analyzer-strip"><div class="marker-table"></div><div class="strip-tools">${tools}</div></div>`
    + `${waterfall ? '<div class="analyzer-waterfall"><canvas aria-label="Waterfall, newest at top"></canvas></div>' : ''}</div>`;
}

export {niceTicks};
