/* Canvas plotting primitives: graticules with real tick labels, traces,
   constellations, heatmaps, eye diagrams and a newest-at-top waterfall. */
import {fmt} from '../core/format.js';

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
let palette = null;
export function colors() {
  if (!palette) {
    palette = {
      bg: css('--plot-bg') || '#06080a', grid: css('--grid') || '#161c20', gridMajor: css('--grid-major') || '#232b31',
      text: css('--plot-text') || '#8f9ca5', accent: css('--accent') || '#e9b04b', cyan: css('--cyan') || '#5cc8e6',
      red: css('--red') || '#ec7b72', green: css('--green') || '#6fcf97', violet: css('--violet') || '#b99af2',
      traces: [css('--trace-1'), css('--trace-2'), css('--trace-3'), css('--trace-4')],
      mono: css('--mono') || 'monospace',
    };
  }
  return palette;
}

/** Size the backing store to the element and return a CSS-pixel context. */
export function setup(canvas) {
  if (!canvas || !canvas.isConnected) return null;
  const rect = canvas.getBoundingClientRect();
  if (rect.width < 4 || rect.height < 4) return null;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  const w = Math.round(rect.width * dpr), h = Math.round(rect.height * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const ctx = canvas.getContext('2d', {alpha: false});
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return {ctx, w: rect.width, h: rect.height};
}

export function clear(ctx, w, h) {
  ctx.fillStyle = colors().bg;
  ctx.fillRect(0, 0, w, h);
}

/** 1-2-5 ticks covering [lo, hi]. */
export function niceTicks(lo, hi, target = 6) {
  if (!(hi > lo)) return {ticks: [lo], step: 1};
  const raw = (hi - lo) / Math.max(1, target);
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= raw) || 10 * mag;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return {ticks, step};
}

const PREFIX = [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k'], [1, ''], [1e-3, 'm'], [1e-6, 'µ'], [1e-9, 'n']];
const PREFIXED_UNITS = new Set(['Hz', 's', 'S/s', 'Bd', 'm', '']);

/** Tick label formatter sharing one SI prefix and enough decimals for the step. */
export function tickFormatter(lo, hi, step, unit = '') {
  if (!PREFIXED_UNITS.has(unit)) {
    const decimals = Math.max(0, Math.min(6, -Math.floor(Math.log10(step) + 1e-9)));
    return v => `${v.toFixed(decimals)}${unit ? ` ${unit}` : ''}`;
  }
  const big = Math.max(Math.abs(lo), Math.abs(hi));
  const [scale, prefix] = PREFIX.find(([s]) => big >= s * 0.999) || [1, ''];
  const decimals = Math.max(0, Math.min(9, -Math.floor(Math.log10(step / scale) + 1e-9)));
  return v => `${(v / scale).toFixed(decimals)}${unit || prefix ? ` ${prefix}${unit}` : ''}`;
}

export const MARGIN = {left: 54, right: 10, top: 8, bottom: 22};

/** Graticule with labelled ticks; returns coordinate mappers. */
export function frame(ctx, w, h, {x0, x1, y0, y1, xUnit = '', yUnit = '', margin = MARGIN, xTicks = 7, yTicks = 5, xLabels = true, invertY = false}) {
  const c = colors();
  const m = margin;
  const pw = Math.max(1, w - m.left - m.right), ph = Math.max(1, h - m.top - m.bottom);
  const X = v => m.left + (v - x0) / (x1 - x0 || 1) * pw;
  const Y = invertY ? v => m.top + (v - y0) / (y1 - y0 || 1) * ph : v => m.top + (y1 - v) / (y1 - y0 || 1) * ph;
  ctx.lineWidth = 1;
  ctx.font = `10px ${c.mono}`;
  const xt = niceTicks(x0, x1, xTicks), yt = niceTicks(y0, y1, yTicks);
  const fx = tickFormatter(x0, x1, xt.step, xUnit), fy = tickFormatter(y0, y1, yt.step, yUnit);
  ctx.strokeStyle = c.grid;
  ctx.beginPath();
  for (const v of xt.ticks) { const x = Math.round(X(v)) + .5; ctx.moveTo(x, m.top); ctx.lineTo(x, m.top + ph); }
  for (const v of yt.ticks) { const y = Math.round(Y(v)) + .5; ctx.moveTo(m.left, y); ctx.lineTo(m.left + pw, y); }
  ctx.stroke();
  ctx.strokeStyle = c.gridMajor;
  ctx.strokeRect(m.left + .5, m.top + .5, pw - 1, ph - 1);
  ctx.fillStyle = c.text;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (const v of yt.ticks) { const y = Y(v); if (y > m.top + 5 && y < m.top + ph - 3) ctx.fillText(fy(v), m.left - 5, y); }
  if (xLabels) {
    ctx.textBaseline = 'top';
    ctx.textAlign = 'center';
    let last = -Infinity;
    for (const v of xt.ticks) {
      const x = X(v), label = fx(v), half = ctx.measureText(label).width / 2;
      if (x - half > last + 8 && x - half > 2 && x + half < w - 2) { ctx.fillText(label, x, m.top + ph + 5); last = x + half; }
    }
  }
  return {X, Y, pw, ph, m};
}

/** Draw ``y`` against ``x`` (or index) with per-pixel min/max decimation. */
export function trace(ctx, X, Y, y, x, color, {width = 1.2, from = 0, to = y.length, dash = null} = {}) {
  if (!y || to - from < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = width;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  const n = to - from;
  const px0 = X(x ? x[from] : from), px1 = X(x ? x[to - 1] : to - 1);
  const step = Math.max(1, Math.floor(n / Math.max(200, Math.abs(px1 - px0) * 1.5)));
  let started = false;
  for (let i = from; i < to; i += step) {
    const end = Math.min(to, i + step);
    let lo = Infinity, hi = -Infinity, ilo = -1, ihi = -1;
    for (let j = i; j < end; j++) {
      const v = y[j];
      if (!Number.isFinite(v)) continue;
      if (v < lo) { lo = v; ilo = j; }
      if (v > hi) { hi = v; ihi = j; }
    }
    if (ilo < 0) { started = false; continue; }
    const pts = ilo === ihi ? [ilo] : ilo < ihi ? [ilo, ihi] : [ihi, ilo];
    for (const j of pts) {
      const px = X(x ? x[j] : j), py = Y(y[j]);
      if (started) ctx.lineTo(px, py); else { ctx.moveTo(px, py); started = true; }
    }
  }
  ctx.stroke();
  ctx.setLineDash([]);
}

/**
 * The band between ``lo`` and ``hi`` (one pair per point): a min/max trace, as
 * an oscilloscope's peak detect draws a signal faster than its display.  The
 * outline keeps a band that has collapsed to a line visible.
 */
export function band(ctx, X, Y, lo, hi, x, color) {
  ctx.fillStyle = color;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  const ok = k => Number.isFinite(lo[k]) && Number.isFinite(hi[k]);
  for (let i = 0, n = lo.length; i < n;) {
    while (i < n && !ok(i)) i++;
    let j = i;
    while (j < n && ok(j)) j++;
    if (j > i) {
      ctx.beginPath();
      for (let k = i; k < j; k++) ctx.lineTo(X(x ? x[k] : k), Y(hi[k]));
      for (let k = j - 1; k >= i; k--) ctx.lineTo(X(x ? x[k] : k), Y(lo[k]));
      ctx.closePath();
      ctx.globalAlpha = 0.4;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.stroke();
    }
    i = j;
  }
}

export function finiteRange(...arrays) {
  let lo = Infinity, hi = -Infinity;
  for (const a of arrays) for (const v of a || []) if (Number.isFinite(v)) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!Number.isFinite(lo)) return [-1, 1];
  if (hi - lo < 1e-12) { const pad = Math.abs(hi) * 0.1 || 1; return [lo - pad, hi + pad]; }
  return [lo, hi];
}

/**
 * Line plot.  ``series``: array, or list of {y, y2?, x?, color?, dash?}; with
 * ``y2`` the series is a band from ``y`` (low) to ``y2`` (high) per point.
 * Options: x, xUnit, yUnit, min, max, threshold, cursor (x value), shade [{x0,x1,color}], pad (fraction).
 */
export function linePlot(canvas, series, opts = {}) {
  const s = setup(canvas);
  if (!s) return null;
  const {ctx, w, h} = s;
  clear(ctx, w, h);
  const list = (Array.isArray(series) && series.length && typeof series[0] === 'object' && series[0] !== null && !ArrayBuffer.isView(series[0]) && 'y' in series[0]
    ? series : [{y: series}]).filter(item => item.y?.length);
  const c = colors();
  if (!list.length) return null;
  let [lo, hi] = finiteRange(...list.map(item => item.y), ...list.map(item => item.y2), Number.isFinite(opts.threshold) ? [opts.threshold] : []);
  const pad = opts.pad ?? 0.06;
  const span = hi - lo;
  lo = opts.min ?? lo - span * pad;
  hi = opts.max ?? hi + span * pad;
  const xs = list[0].x || opts.x;
  const n = list[0].y.length;
  const x0 = xs ? xs[0] : 0, x1 = xs ? xs.at(-1) : n - 1;
  const g = frame(ctx, w, h, {x0, x1: x1 === x0 ? x0 + 1 : x1, y0: lo, y1: hi, xUnit: opts.xUnit || '', yUnit: opts.yUnit || ''});
  for (const band of opts.shade || []) {
    ctx.fillStyle = band.color;
    const a = Math.max(g.m.left, g.X(band.x0)), b = Math.min(g.m.left + g.pw, g.X(band.x1));
    if (b > a) ctx.fillRect(a, g.m.top, b - a, g.ph);
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(g.m.left, g.m.top, g.pw, g.ph);
  ctx.clip();
  list.forEach((item, i) => {
    const color = item.color || [c.accent, c.cyan, c.violet, c.green][i % 4];
    if (item.y2) band(ctx, g.X, g.Y, item.y, item.y2, item.x || opts.x, color);
    else trace(ctx, g.X, g.Y, item.y, item.x || opts.x, color, {dash: item.dash});
  });
  if (Number.isFinite(opts.threshold)) {
    ctx.strokeStyle = c.red;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(g.m.left, g.Y(opts.threshold));
    ctx.lineTo(g.m.left + g.pw, g.Y(opts.threshold));
    ctx.stroke();
    ctx.setLineDash([]);
  }
  if (Number.isFinite(opts.cursor)) {
    ctx.strokeStyle = '#f0d49b';
    ctx.beginPath();
    ctx.moveTo(g.X(opts.cursor), g.m.top);
    ctx.lineTo(g.X(opts.cursor), g.m.top + g.ph);
    ctx.stroke();
  }
  ctx.restore();
  return g;
}

/** I/Q scatter; ``ideals`` are reference points drawn as rings. */
export function scatter(canvas, I, Q, opts = {}) {
  const s = setup(canvas);
  if (!s) return;
  const {ctx, w, h} = s;
  clear(ctx, w, h);
  const c = colors();
  if (!I?.length) return;
  let scale = opts.scale || Math.sqrt(I.reduce((acc, v, k) => acc + v * v + (Q[k] || 0) ** 2, 0) / I.length);
  scale = Math.max(scale, 1e-12);
  let lim = 1e-3;
  for (let k = 0; k < I.length; k++) lim = Math.max(lim, Math.abs(I[k] / scale), Math.abs((Q[k] || 0) / scale));
  for (const [i, q] of opts.ideals || []) lim = Math.max(lim, Math.abs(i), Math.abs(q));
  lim = Math.min(lim * 1.12, opts.maxLim || Infinity);
  const size = Math.min(w, h) - 16;
  const cx = w / 2, cy = h / 2;
  const P = (i, q) => [cx + i / lim * size / 2, cy - q / lim * size / 2];
  ctx.strokeStyle = c.grid;
  ctx.strokeRect(cx - size / 2 + .5, cy - size / 2 + .5, size, size);
  ctx.beginPath();
  ctx.moveTo(cx, cy - size / 2); ctx.lineTo(cx, cy + size / 2);
  ctx.moveTo(cx - size / 2, cy); ctx.lineTo(cx + size / 2, cy);
  ctx.strokeStyle = c.gridMajor;
  ctx.stroke();
  if (opts.ideals?.length) {
    ctx.strokeStyle = opts.dimIdeals ? 'rgba(170,185,195,.35)' : 'rgba(246,207,131,.8)';
    for (const [i, q] of opts.ideals) { const [x, y] = P(i, q); ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 2 * Math.PI); ctx.stroke(); }
  }
  ctx.fillStyle = opts.color || 'rgba(233,176,75,.55)';
  const stride = Math.max(1, Math.floor(I.length / 2500));
  for (let k = 0; k < I.length; k += stride) {
    const [x, y] = P(I[k] / scale, (Q[k] || 0) / scale);
    ctx.fillRect(x - 1, y - 1, 2, 2);
  }
}

/* 256-entry instrument colour map: black → blue → cyan → yellow → white-hot. */
const LUT = (() => {
  const stops = [[0, [4, 6, 12]], [0.22, [18, 40, 110]], [0.45, [20, 140, 190]], [0.68, [120, 210, 120]], [0.85, [245, 200, 70]], [1, [255, 250, 230]]];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const q = i / 255;
    const k = stops.findIndex(([p]) => p >= q);
    const [p1, c1] = stops[Math.max(1, k)], [p0, c0] = stops[Math.max(0, k - 1)];
    const t = p1 > p0 ? (q - p0) / (p1 - p0) : 0;
    for (let j = 0; j < 3; j++) lut[i * 3 + j] = c0[j] + (c1[j] - c0[j]) * t;
  }
  return lut;
})();

export function colorAt(q) {
  const i = Math.max(0, Math.min(255, Math.round(q * 255))) * 3;
  return [LUT[i], LUT[i + 1], LUT[i + 2]];
}

/**
 * Heatmap of row-major ``flat`` [rows × cols].  Row 0 is drawn at the top
 * (``y[0]`` labels the top edge) unless ``rowsBottomUp`` puts it at the bottom.
 */
export function heatmap(canvas, flat, shape, opts = {}) {
  const s = setup(canvas);
  if (!s || !flat?.length) return null;
  const {ctx, w, h} = s;
  clear(ctx, w, h);
  const [rows, cols] = shape;
  let lo = opts.lo, hi = opts.hi;
  if (lo === undefined || hi === undefined) { const r = finiteRange(flat); lo ??= r[1] - 80; hi ??= r[1]; }
  const off = document.createElement('canvas');
  off.width = cols; off.height = rows;
  const oc = off.getContext('2d');
  const img = oc.createImageData(cols, rows);
  for (let p = 0; p < flat.length; p++) {
    const [r, g, b] = colorAt((flat[p] - lo) / Math.max(1e-12, hi - lo));
    img.data[p * 4] = r; img.data[p * 4 + 1] = g; img.data[p * 4 + 2] = b; img.data[p * 4 + 3] = 255;
  }
  oc.putImageData(img, 0, 0);
  const x = opts.x || [0, cols - 1], y = opts.y || [0, rows - 1];
  const bottomUp = !!opts.rowsBottomUp;
  const topValue = bottomUp ? y.at(-1) : y[0], bottomValue = bottomUp ? y[0] : y.at(-1);
  const g = frame(ctx, w, h, {x0: x[0], x1: x.at(-1), y0: Math.min(topValue, bottomValue), y1: Math.max(topValue, bottomValue),
    xUnit: opts.xUnit, yUnit: opts.yUnit, invertY: topValue < bottomValue});
  ctx.imageSmoothingEnabled = false;
  ctx.save();
  if (bottomUp) { ctx.translate(0, 2 * g.m.top + g.ph); ctx.scale(1, -1); }
  ctx.drawImage(off, g.m.left, g.m.top, g.pw, g.ph);
  ctx.restore();
  if (Number.isFinite(opts.cursorY)) {
    ctx.strokeStyle = '#f0d49b';
    ctx.beginPath(); ctx.moveTo(g.m.left, g.Y(opts.cursorY)); ctx.lineTo(g.m.left + g.pw, g.Y(opts.cursorY)); ctx.stroke();
  }
  for (const mark of opts.marks || []) {
    ctx.strokeStyle = mark.color || '#fff';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(g.X(mark.x), g.Y(mark.y), mark.r || 6, 0, 2 * Math.PI); ctx.stroke();
  }
  return g;
}

export function eye(canvas, traces) {
  const s = setup(canvas);
  if (!s) return;
  const {ctx, w, h} = s;
  clear(ctx, w, h);
  if (!traces?.length) return;
  let lim = 1e-3;
  for (const t of traces) for (const v of t) lim = Math.max(lim, Math.abs(v));
  lim *= 1.1;
  const g = frame(ctx, w, h, {x0: 0, x1: traces[0].length - 1, y0: -lim, y1: lim, xLabels: false});
  ctx.strokeStyle = 'rgba(233,176,75,.22)';
  for (const t of traces.slice(0, 64)) {
    ctx.beginPath();
    t.forEach((v, i) => (i ? ctx.lineTo(g.X(i), g.Y(v)) : ctx.moveTo(g.X(i), g.Y(v))));
    ctx.stroke();
  }
}

/** Scrolling spectrogram raster; the newest row is drawn at the top. */
export class Waterfall {
  constructor(width = 640, rows = 160) {
    this.off = document.createElement('canvas');
    this.off.width = width;
    this.off.height = rows;
    this.ctx = this.off.getContext('2d', {alpha: false});
    this.reset();
  }

  reset() {
    this.ctx.fillStyle = colors().bg;
    this.ctx.fillRect(0, 0, this.off.width, this.off.height);
    this.rows = 0;
  }

  /** ``row`` values over ``[lo, hi]`` dB; NaN draws as background (no data). */
  push(row, lo, hi) {
    const {off, ctx} = this;
    const W = off.width;
    ctx.drawImage(off, 0, 1);
    const img = ctx.createImageData(W, 1);
    const span = Math.max(1e-9, hi - lo);
    for (let x = 0; x < W; x++) {
      const v = row[Math.min(row.length - 1, Math.floor(x * row.length / W))];
      const p = x * 4;
      if (!Number.isFinite(v)) { img.data[p] = 6; img.data[p + 1] = 8; img.data[p + 2] = 10; img.data[p + 3] = 255; continue; }
      const [r, g, b] = colorAt((v - lo) / span);
      img.data[p] = r; img.data[p + 1] = g; img.data[p + 2] = b; img.data[p + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    this.rows += 1;
  }

  draw(canvas, left = 0, right = 0) {
    const s = setup(canvas);
    if (!s) return;
    const {ctx, w, h} = s;
    clear(ctx, w, h);
    ctx.imageSmoothingEnabled = false;
    // Keep the data columns aligned with the plot axis, while extending the
    // edge colours into the label gutters instead of leaving empty black bars.
    if (left > 0) ctx.drawImage(this.off, 0, 0, 1, this.off.height, 0, 0, left, h);
    if (right > 0) ctx.drawImage(this.off, this.off.width - 1, 0, 1, this.off.height, w - right, 0, right, h);
    ctx.drawImage(this.off, left, 0, Math.max(1, w - left - right), h);
  }
}

export function labelValue(value, unit) {
  return fmt(value, unit);
}
