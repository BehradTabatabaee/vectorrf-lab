/* Top-down radar scene: range rings, beam pattern, targets with velocity vectors and trails.

   Viewed from above with the array at the bottom centre: +X (boresight) points
   up and +Y to the left, so azimuth atan2(y, x) grows counter-clockwise.

   Targets move as p(t) = p0 + v0·t + ½·a·t² (the simulator's model).  The
   scene, its handles and the target editor all work on the state at the
   displayed time t: dragging a target moves where it is now, dragging its
   arrow changes its velocity now, and neither changes the other. */
import {fmt} from '../core/format.js';
import {clear, colors, niceTicks, setup} from './canvas.js';

export const TARGET_COLORS = ['#5cc8e6', '#f2c14e', '#c79bf2', '#7fd99a', '#f29e6b', '#6aa7f0', '#e8d26b', '#f07fb0', '#9ad0c2', '#d7a9f7', '#b5e48c', '#ff9f9f'];

const BODY_RADIUS = 8;        // px: grabbing inside moves the target
const HANDLE_RADIUS = 9;      // px: grabbing the arrow head changes the velocity
const SHAFT_DISTANCE = 5;     // px: grabbing along the arrow changes the velocity
const MIN_ARROW = 24;         // px: shortest drawn arrow, so every velocity can be grabbed
const ARROW_AT_MAX_SPEED = 64;  // px: arrow length of a target at the Doppler search limit

/** Position, velocity and acceleration of ``target`` at time ``t`` (seconds). */
export function stateAt(target, t) {
  const a = target.acceleration || [0, 0, 0];
  return {
    position: [0, 1, 2].map(k => target.position[k] + target.velocity[k] * t + 0.5 * a[k] * t * t),
    velocity: [0, 1, 2].map(k => target.velocity[k] + a[k] * t),
    acceleration: [...a],
  };
}

/**
 * Give ``target`` the state ``{position, velocity, acceleration}`` at time
 * ``t`` (omitted parts keep their current value at ``t``) by re-anchoring the
 * t = 0 terms the simulator uses.  Changing the velocity therefore keeps the
 * current position, and changing the acceleration keeps both.
 */
export function setStateAt(target, t, change) {
  const now = stateAt(target, t);
  const p = change.position ?? now.position, v = change.velocity ?? now.velocity, a = change.acceleration ?? now.acceleration;
  target.acceleration = a.map(Number);
  target.velocity = [0, 1, 2].map(k => Number(v[k]) - target.acceleration[k] * t);
  target.position = [0, 1, 2].map(k => Number(p[k]) - target.velocity[k] * t - 0.5 * target.acceleration[k] * t * t);
}

/** Distance from point p to segment ab (pixels). */
function segmentDistance(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay, len2 = dx * dx + dy * dy;
  const u = len2 ? Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / len2)) : 0;
  return Math.hypot(px - (ax + u * dx), py - (ay + u * dy));
}

export class SceneMap {
  /** Handlers: onMove(i, [x, y] now, done), onVelocity(i, [vx, vy] now, done), onSelect(i), onAdd(x, y). */
  constructor(canvas, {onMove, onVelocity, onSelect, onAdd}) {
    this.canvas = canvas;
    this.handlers = {onMove, onVelocity, onSelect, onAdd};
    this.view = null;
    this.drag = null;
    this.items = [];
    canvas.addEventListener('mousedown', e => this.down(e));
    canvas.addEventListener('dblclick', e => this.dbl(e));
    canvas.addEventListener('mousemove', e => this.hover(e));
    canvas.addEventListener('mouseleave', () => this.canvas.classList.remove('grab', 'aim'));
    this.move = e => this.dragMove(e);
    this.up = () => this.dragEnd();
  }

  /** World metres → canvas pixels for the current scale. */
  project(x, y) {
    const v = this.view;
    return [v.cx - y * v.scale, v.cy - x * v.scale];
  }

  unproject(px, py) {
    const v = this.view;
    return [(v.cy - py) / v.scale, (v.cx - px) / v.scale];
  }

  /** Arrow geometry of one target: body, drawn tip (at least MIN_ARROW long) and direction. */
  arrow(position, velocity) {
    const [bx, by] = this.project(position[0], position[1]);
    // Pixels per m/s along the arrow (the projection keeps lengths, it only rotates).
    const k = this.view.arrowSeconds * this.view.scale;
    let dx = -velocity[1] * k, dy = -velocity[0] * k;
    let length = Math.hypot(dx, dy);
    if (length < MIN_ARROW) {
      // Too slow to see: draw a short arrow in the direction of motion (boresight when at rest).
      const [ux, uy] = length > 1e-9 ? [dx / length, dy / length] : [0, -1];
      dx = ux * MIN_ARROW; dy = uy * MIN_ARROW; length = MIN_ARROW;
    }
    return {bx, by, tx: bx + dx, ty: by + dy, angle: Math.atan2(dy, dx)};
  }

  draw({targets, selected, data, maxRange, maxVelocity = 100, trails, time = 0}) {
    const s = setup(this.canvas);
    if (!s) return;
    const {ctx, w, h} = s;
    const c = colors();
    clear(ctx, w, h);
    const states = targets.map(t => stateAt(t, time));
    const extent = Math.max(maxRange, ...states.map(st => Math.hypot(st.position[0], st.position[1]) * 1.12), 1);
    const scale = Math.min((w - 40) / (2 * extent), (h - 46) / extent);
    this.view = {cx: w / 2, cy: h - 24, scale, extent, arrowSeconds: ARROW_AT_MAX_SPEED / (Math.max(1e-6, maxVelocity) * scale)};
    const [cx, cy] = [w / 2, h - 24];
    // Range rings and bearing spokes.
    ctx.font = `10px ${c.mono}`;
    const {ticks} = niceTicks(0, extent, 5);
    ctx.strokeStyle = c.grid;
    ctx.fillStyle = c.text;
    for (const r of ticks.filter(r => r > 0)) {
      ctx.beginPath(); ctx.arc(cx, cy, r * scale, Math.PI, 2 * Math.PI); ctx.stroke();
      ctx.textAlign = 'left'; ctx.fillText(fmt(r, 'm'), cx + 4, cy - r * scale - 3);
    }
    for (let az = -90; az <= 90; az += 30) {
      const a = az * Math.PI / 180, [x, y] = this.project(extent * Math.cos(a), extent * Math.sin(a));
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(x, y); ctx.stroke();
      if (az % 90) { ctx.textAlign = 'center'; ctx.fillText(`${az}°`, x + (az > 0 ? 10 : -10), y + 4); }
    }
    // Beam pattern lobe (dB → radius over a 40 dB scale).
    if (data?.pattern_db?.length) {
      ctx.beginPath();
      data.angle_az.forEach((az, i) => {
        const r = Math.max(0, (data.pattern_db[i] + 40) / 40) * extent, a = az * Math.PI / 180;
        const [x, y] = this.project(r * Math.cos(a), r * Math.sin(a));
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(233,176,75,.07)';
      ctx.strokeStyle = 'rgba(233,176,75,.45)';
      ctx.fill(); ctx.stroke();
      const steer = (data.steer_az_deg || 0) * Math.PI / 180, [bx, by] = this.project(extent * Math.cos(steer), extent * Math.sin(steer));
      ctx.setLineDash([4, 4]); ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(bx, by); ctx.stroke(); ctx.setLineDash([]);
    }
    // Detected ranges as faint arcs.
    for (const d of data?.detections || []) {
      ctx.strokeStyle = 'rgba(236,123,114,.35)';
      ctx.beginPath(); ctx.arc(cx, cy, d.range_m * scale, Math.PI, 2 * Math.PI); ctx.stroke();
    }
    // Trails, targets and velocity vectors (the selected target last, on top).
    this.items = states.map((st, i) => ({index: i, ...this.arrow(st.position, st.velocity)}));
    const order = targets.map((_, i) => i).filter(i => i !== selected);
    if (selected >= 0 && selected < targets.length) order.push(selected);
    for (const i of order) {
      const t = targets[i], st = states[i], item = this.items[i];
      const color = TARGET_COLORS[i % TARGET_COLORS.length];
      const trail = trails?.[i] || [];
      if (trail.length > 1) {
        ctx.strokeStyle = `${color}66`;
        ctx.beginPath();
        trail.forEach(([x, y], k) => { const [px, py] = this.project(x, y); if (k) ctx.lineTo(px, py); else ctx.moveTo(px, py); });
        ctx.stroke();
      }
      const {bx, by, tx, ty, angle} = item;
      ctx.strokeStyle = color;
      ctx.fillStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(tx - 8 * Math.cos(angle - 0.4), ty - 8 * Math.sin(angle - 0.4)); ctx.lineTo(tx - 8 * Math.cos(angle + 0.4), ty - 8 * Math.sin(angle + 0.4)); ctx.closePath(); ctx.fill();
      ctx.beginPath(); ctx.arc(bx, by, i === selected ? 6 : 5, 0, 2 * Math.PI); ctx.fill();
      if (i === selected) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.arc(bx, by, 9.5, 0, 2 * Math.PI); ctx.stroke();
        ctx.strokeRect(tx - 4.5, ty - 4.5, 9, 9);  // the velocity handle
      }
      ctx.lineWidth = 1;
      const range = Math.hypot(...st.position), radial = range > 0 ? st.velocity.reduce((acc, v, k) => acc + v * st.position[k], 0) / range : 0;
      // Label on a backing box so range-ring labels and lobes never make it unreadable.
      const label = `${t.name} · ${fmt(range, 'm')} · ${radial.toFixed(1)} m/s`;
      const lx = Math.min(bx + 12, w - ctx.measureText(label).width - 6), ly = Math.max(12, by - 10);
      ctx.fillStyle = 'rgba(12,14,16,.82)';
      ctx.fillRect(lx - 3, ly - 7, ctx.measureText(label).width + 6, 14);
      ctx.fillStyle = i === selected ? '#eef1f3' : '#c2c8cc';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, lx, ly);
      ctx.textBaseline = 'alphabetic';
    }
    // Strongest-cell position estimate.
    const est = data?.estimated_position;
    if (est) {
      const [ex, ey] = this.project(est[0], est[1]);
      ctx.strokeStyle = c.accent;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(ex - 8, ey - 8, 16, 16);
      ctx.beginPath(); ctx.moveTo(ex - 12, ey); ctx.lineTo(ex + 12, ey); ctx.moveTo(ex, ey - 12); ctx.lineTo(ex, ey + 12); ctx.stroke();
      ctx.lineWidth = 1;
    }
    // Array at the origin, and the arrow scale.
    ctx.fillStyle = '#e4e8eb';
    ctx.fillRect(cx - 9, cy - 3, 18, 6);
    ctx.fillStyle = c.text;
    ctx.textAlign = 'right';
    ctx.fillText(`arrow = ${fmt(this.view.arrowSeconds, 's', 2)} of motion`, w - 8, 14);
    this.targets = targets;
    this.states = states;
    this.selected = selected;
  }

  local(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }

  /**
   * What lies under the pointer: a target body (move it) or its velocity arrow
   * (head or shaft: change the velocity).  Bodies win over arrows, and the
   * selected target over the others, so overlapping targets stay reachable.
   */
  pick(px, py) {
    if (!this.view || !this.items.length) return null;
    const order = this.items.map((_, i) => i).reverse();
    const sel = order.indexOf(this.selected);
    if (sel > 0) { order.splice(sel, 1); order.unshift(this.selected); }
    for (const i of order) {
      const it = this.items[i];
      if (Math.hypot(px - it.bx, py - it.by) <= BODY_RADIUS) return {index: i, part: 'position'};
    }
    for (const i of order) {
      const it = this.items[i];
      if (Math.hypot(px - it.tx, py - it.ty) <= HANDLE_RADIUS) return {index: i, part: 'velocity'};
      const along = Math.hypot(px - it.bx, py - it.by);
      if (along > BODY_RADIUS && segmentDistance(px, py, it.bx, it.by, it.tx, it.ty) <= SHAFT_DISTANCE) return {index: i, part: 'velocity'};
    }
    return null;
  }

  down(e) {
    if (e.button !== 0) return;
    const [px, py] = this.local(e);
    const hit = this.pick(px, py);
    if (!hit) return;
    e.preventDefault();
    this.handlers.onSelect?.(hit.index);
    // Drags apply as a delta to the state when grabbed, so nothing jumps on
    // grab and the target follows the pointer while the simulation runs.
    const st = this.states[hit.index];
    this.drag = {...hit, start: this.unproject(px, py), position: [...st.position], velocity: [...st.velocity],
      arrowSeconds: this.view.arrowSeconds, moved: false};
    this.canvas.classList.add('grabbing');
    window.addEventListener('mousemove', this.move);
    window.addEventListener('mouseup', this.up);
  }

  dragMove(e) {
    const d = this.drag;
    if (!d || !this.view) return;
    const [px, py] = this.local(e);
    const [x, y] = this.unproject(px, py);
    const dx = x - d.start[0], dy = y - d.start[1];
    d.moved = true;
    if (d.part === 'position') this.handlers.onMove?.(d.index, [d.position[0] + dx, d.position[1] + dy], false);
    else this.handlers.onVelocity?.(d.index, [d.velocity[0] + dx / d.arrowSeconds, d.velocity[1] + dy / d.arrowSeconds], false);
  }

  dragEnd() {
    window.removeEventListener('mousemove', this.move);
    window.removeEventListener('mouseup', this.up);
    this.canvas.classList.remove('grabbing');
    const d = this.drag;
    this.drag = null;
    if (!d || !d.moved) return;
    if (d.part === 'position') this.handlers.onMove?.(d.index, null, true);
    else this.handlers.onVelocity?.(d.index, null, true);
  }

  hover(e) {
    if (this.drag) return;
    const [px, py] = this.local(e);
    const hit = this.pick(px, py);
    this.canvas.classList.toggle('grab', hit?.part === 'position');
    this.canvas.classList.toggle('aim', hit?.part === 'velocity');
    this.canvas.title = hit ? (hit.part === 'position' ? 'Drag to move the target' : 'Drag to change its velocity') : '';
  }

  dbl(e) {
    const [px, py] = this.local(e);
    if (this.pick(px, py) || !this.view) return;
    const [x, y] = this.unproject(px, py);
    if (x > 0) this.handlers.onAdd?.(x, y);
  }
}
