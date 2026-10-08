/* Design-plan drawings: component bands over frequency and hop/sweep schedules over time. */
import {clear, colors, frame, setup} from './canvas.js';

export const COMPONENT_COLORS = ['#f2c14e', '#5cc8e6', '#c79bf2', '#7fd99a', '#f29e6b', '#6aa7f0', '#e8d26b', '#f07fb0'];

function hatch(ctx, x0, y0, w, h, color) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x0, y0, w, h);
  ctx.clip();
  ctx.strokeStyle = color;
  ctx.globalAlpha = 0.35;
  for (let x = x0 - h; x < x0 + w; x += 7) { ctx.beginPath(); ctx.moveTo(x, y0 + h); ctx.lineTo(x + h, y0); ctx.stroke(); }
  ctx.restore();
}

/**
 * ``items``: [{label, band:[lo,hi]|null, power, extent:[lo,hi]|null, color, active, enabled}] in
 * baseband Hz; ``offset`` adds the RF centre to axis labels.
 */
export function drawBandPlan(canvas, items, {fs, offset = 0, extent = null}) {
  const s = setup(canvas);
  if (!s) return;
  const {ctx, w, h} = s;
  clear(ctx, w, h);
  const c = colors();
  const powers = items.filter(i => i.enabled !== false).map(i => i.power).filter(Number.isFinite);
  const top = (powers.length ? Math.max(...powers) : 0) + 6, bottom = (powers.length ? Math.min(...powers) : -30) - 24;
  const g = frame(ctx, w, h, {x0: -fs / 2 + offset, x1: fs / 2 + offset, y0: bottom, y1: top, xUnit: 'Hz', yUnit: 'dBm', yTicks: 4});
  const X = hz => g.X(hz + offset);
  if (extent) {
    const a = X(extent[0]), b = X(extent[1]);
    hatch(ctx, a, g.m.top, Math.max(2, b - a), g.ph, c.text);
  }
  items.forEach((item, i) => {
    if (item.enabled === false) return;
    const color = item.color || COMPONENT_COLORS[i % 8];
    const y = g.Y(item.power), base = g.m.top + g.ph;
    const span = item.band || item.extent;
    if (item.extent) {
      const a = X(item.extent[0]), b = X(item.extent[1]);
      hatch(ctx, a, y, Math.max(2, b - a), base - y, color);
      ctx.strokeStyle = color;
      ctx.lineWidth = item.active ? 2 : 1;
      ctx.strokeRect(a + .5, y + .5, Math.max(2, b - a), base - y);
      ctx.lineWidth = 1;
    }
    if (item.band) {
      const a = X(item.band[0]), b = X(item.band[1]);
      ctx.fillStyle = `${color}${item.active ? '66' : '33'}`;
      ctx.fillRect(a, y, Math.max(2, b - a), base - y);
      ctx.strokeStyle = color;
      ctx.lineWidth = item.active ? 2 : 1;
      ctx.strokeRect(a + .5, y + .5, Math.max(2, b - a), base - y);
      ctx.lineWidth = 1;
    }
    if (span && item.label) {
      ctx.fillStyle = color;
      ctx.font = `10px ${c.mono}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'bottom';
      ctx.fillText(item.label, (X(span[0]) + X(span[1])) / 2, y - 3);
    }
  });
  // Nyquist edges.
  ctx.strokeStyle = c.red;
  ctx.setLineDash([4, 4]);
  for (const hz of [-fs / 2, fs / 2]) { ctx.beginPath(); ctx.moveTo(X(hz), g.m.top); ctx.lineTo(X(hz), g.m.top + g.ph); ctx.stroke(); }
  ctx.setLineDash([]);
}

/** Hop/sweep schedule over time: ``plan`` is an agility summary (from /api/tx/plan). */
export function drawTimePlan(canvas, summary, {offset = 0, fs, color}) {
  const s = setup(canvas);
  if (!s) return;
  const {ctx, w, h} = s;
  clear(ctx, w, h);
  if (!summary?.plan) return;
  const c = colors();
  const p = summary.plan;
  const n = p.start_s.length;
  if (!n) return;
  const t1 = Math.min(summary.period_s, p.start_s[n - 1] + summary.dwell_s + summary.off_s);
  const lo = Math.min(...p.f_start_hz, ...p.f_stop_hz), hi = Math.max(...p.f_start_hz, ...p.f_stop_hz);
  const pad = Math.max((hi - lo) * 0.08, fs * 0.01);
  const g = frame(ctx, w, h, {x0: 0, x1: t1, y0: lo - pad + offset, y1: hi + pad + offset, xUnit: 's', yUnit: 'Hz', yTicks: 4});
  ctx.strokeStyle = color || c.accent;
  ctx.lineWidth = Math.max(1.5, Math.min(4, g.ph / Math.max(8, n) / 2));
  ctx.beginPath();
  for (let i = 0; i < n; i++) {
    const t0 = p.start_s[i], tEnd = t0 + summary.dwell_s;
    ctx.moveTo(g.X(t0), g.Y(p.f_start_hz[i] + offset));
    ctx.lineTo(g.X(tEnd), g.Y(p.f_stop_hz[i] + offset));
  }
  ctx.stroke();
  ctx.lineWidth = 1;
  if (p.truncated) {
    ctx.fillStyle = c.text;
    ctx.font = `10px ${c.mono}`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'top';
    ctx.fillText(`first ${n} of ${summary.segments} segments`, w - 14, 10);
  }
}
