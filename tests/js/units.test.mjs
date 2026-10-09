// Unit tests of the browser's pure modules, run with Node (tests/test_js_units.py).
// A minimal DOM stub lets modules that register document listeners at import load.
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import path from 'node:path';

globalThis.document = {addEventListener() {}, documentElement: {}, querySelector: () => null, querySelectorAll: () => []};
globalThis.getComputedStyle = () => ({getPropertyValue: () => ''});

const web = path.resolve(process.argv[2]);
const load = rel => import(pathToFileURL(path.join(web, rel)).href);
const {parseQuantity, formatQuantity, fmt, fmtFreq} = await load('core/format.js');
const {stateAt, setStateAt} = await load('plots/scene.js');
const {state} = await load('core/state.js');
const {txRequest, liveCaptureSamples, liveSource} = await load('core/source.js');

let passed = 0;
function test(name, fn) {
  try { fn(); passed += 1; } catch (error) { console.error(`FAIL ${name}\n${error.stack}`); process.exitCode = 1; }
}
const close = (a, b, tol, message) => assert.ok(Math.abs(a - b) <= tol, `${message}: ${a} vs ${b} (±${tol})`);

test('real-time captures are bounded in both samples and time', () => {
  state.tx = {waveform: 'tone', sample_rate: 1e6, duration: 0.2};
  state.txApplied = structuredClone(state.tx);
  state.pacing = 'realtime';
  close(txRequest().duration, Math.floor(1e6 / 30) / 1e6, 1e-12, '1 MS/s duration');
  assert.equal(liveCaptureSamples(200_000, 1e6), Math.floor(1e6 / 30));
  state.txApplied = {waveform: 'tone', sample_rate: 5e9, duration: 0.2};
  assert.equal(liveCaptureSamples(1e9, 5e9), 524_288);
  close(txRequest().duration, 524_288 / 5e9, 1e-12, '5 GS/s duration');
  state.pacing = 'contiguous';
  assert.equal(liveCaptureSamples(200_000, 1e6), 200_000);
  assert.equal(txRequest().duration, 0.2);
});

test('low-rate file pacing overlaps windows to keep frame cadence', () => {
  state.txApplied = {waveform: 'tone', sample_rate: 1001, duration: 0.2};
  state.source = {kind: 'file', file: {name: 'slow.cf32', sample_rate: 1001, capture_samples: 1024}, info: null};
  state.files = {selected: null};
  state.pacing = 'realtime';
  assert.deepEqual([liveSource().capture_samples, liveSource().step_samples], [64, 33]);
  state.pacing = 'contiguous';
  assert.equal(liveSource().capture_samples, 1024);
  assert.equal(liveSource().step_samples, undefined);
});

/* ------------------------------------------------------------ format.js */

test('instrument-style entries parse', () => {
  const cases = [
    ['2.4G', 'Hz', 2.4e9], ['2.4 GHz', 'Hz', 2.4e9], ['100k', 'Hz', 1e5], ['100 kHz', 'Hz', 1e5], ['20us', 's', 20e-6], ['20 µs', 's', 20e-6],
    ['20μs', 's', 20e-6], ['1e6', 'S/s', 1e6], ['10 MS/s', 'S/s', 1e7], ['10M', 'S/s', 1e7], ['−3 dBm', 'dBm', -3], ['-3', 'dBm', -3],
    ['5m', 's', 5e-3], ['5M', 'Hz', 5e6], ['5 m', 'm', 5], ['5 km', 'm', 5e3], ['1,000,000', 'Hz', 1e6], ['.5', '', 0.5], ['0', 'Hz', 0],
    ['-12.5 kHz', 'Hz', -12500], ['50 kBd', 'Bd', 5e4], ['3 m/s', 'm/s', 3], ['-60 dBc', 'dBc', -60], ['2 m/s2', 'm/s²', 2],
    ['1e-3', 's', 1e-3], ['+7', '', 7], ['30 °', '°', 30], ['1.5e+3', 'Hz', 1500],
    // Rate fields: lowercase prefixes as people type them; "m" is mega (nobody means millihertz).
    ['10mhz', 'Hz', 10e6], ['10 mhz', 'Hz', 10e6], ['2.4ghz', 'Hz', 2.4e9], ['5m', 'S/s', 5e6], ['1msps', 'S/s', 1e6],
    ['250kbaud', 'Bd', 250e3], ['20ms', 's', 0.02], ['3tHz', 'Hz', 3e12],
  ];
  for (const [text, unit, want] of cases) close(parseQuantity(text, unit), want, Math.abs(want) * 1e-12, `${text} [${unit}]`);
});

test('nonsense is rejected (NaN), never silently guessed', () => {
  for (const [text, unit] of [['', 'Hz'], ['abc', 'Hz'], ['5 kg', 'Hz'], ['5 MHz', 's'], ['5k', 'dBm'], ['1..2', ''], ['--3', ''], ['5 kk', 'Hz'], ['e5', ''], ['NaN', ''], ['Infinity', 'Hz']]) {
    assert.ok(Number.isNaN(parseQuantity(text, unit)), `${JSON.stringify(text)} [${unit}] should be NaN, got ${parseQuantity(text, unit)}`);
  }
});

test('formatQuantity round-trips through parseQuantity exactly', () => {
  const units = ['Hz', 's', 'S/s', 'Bd', 'm', 'W', 'dBm', 'dB', '°', 'm/s', ''];
  const values = [0, 1, -1, 0.5, 1e-12, 3.3e-9, 2e-5, 0.001, 999.999, 1000, 12345.678, 2.4e9, -2.4e9, 5e9, 1e12, 7.77e14, 0.1 + 0.2, 1 / 3];
  for (const unit of units) for (const v of values) {
    const text = formatQuantity(v, unit);
    close(parseQuantity(text, unit), v, Math.abs(v) * 1e-11 + 1e-300, `${v} [${unit}] → ${text}`);
  }
});

test('readout formatting chooses SI prefixes', () => {
  assert.equal(fmt(2.4e9, 'Hz'), '2.400 GHz');
  assert.equal(fmt(-45.314, 'dBm'), '-45.31 dBm');
  assert.equal(fmt(1e7, 'S/s'), '10.00 MS/s');
  assert.equal(fmt(NaN, 'Hz'), '—');
  assert.equal(fmtFreq(2_400_000_123, 1), '2.400000123 GHz');
});

/* ------------------------------------------------------------- scene.js */

function randomTarget(rng) {
  const r = () => rng() * 2 - 1;
  return {position: [3000 + 9000 * rng(), 4000 * r(), 500 * r()], velocity: [80 * r(), 80 * r(), 5 * r()], acceleration: [3 * r(), 3 * r(), 0.2 * r()]};
}
function lcg(seed) { let s = seed >>> 0; return () => ((s = (1664525 * s + 1013904223) >>> 0) / 4294967296); }

test('stateAt is the simulator kinematics', () => {
  const t = {position: [1000, 2000, 300], velocity: [-10, 5, 1], acceleration: [0.5, -0.25, 0]};
  const s = stateAt(t, 4);
  assert.deepEqual(s.position, [1000 - 40 + 4, 2000 + 20 - 2, 304]);
  assert.deepEqual(s.velocity, [-8, 4, 1]);
});

test('editing velocity keeps the current position; editing position keeps the velocity', () => {
  const rng = lcg(7);
  for (let k = 0; k < 500; k++) {
    const target = randomTarget(rng), T = rng() * 3600;
    const before = stateAt(target, T);
    const v = [rng() * 200 - 100, rng() * 200 - 100, rng() * 10 - 5];
    setStateAt(target, T, {velocity: v});
    const after = stateAt(target, T);
    for (let i = 0; i < 3; i++) {
      close(after.position[i], before.position[i], 1e-6 * (1 + Math.abs(before.position[i])), 'position kept');
      close(after.velocity[i], v[i], 1e-9 * (1 + Math.abs(v[i])), 'velocity set');
      close(after.acceleration[i], before.acceleration[i], 0, 'acceleration kept');
    }
    const p = [rng() * 10000, rng() * 10000 - 5000, rng() * 1000];
    setStateAt(target, T, {position: p});
    const moved = stateAt(target, T);
    for (let i = 0; i < 3; i++) {
      close(moved.position[i], p[i], 1e-6 * (1 + Math.abs(p[i])), 'position set');
      close(moved.velocity[i], v[i], 1e-9 * (1 + Math.abs(v[i])), 'velocity kept');
    }
  }
});

test('editing acceleration keeps position and velocity at the edit time', () => {
  const rng = lcg(11);
  for (let k = 0; k < 300; k++) {
    const target = randomTarget(rng), T = rng() * 600;
    const before = stateAt(target, T);
    setStateAt(target, T, {acceleration: [rng() * 4 - 2, rng() * 4 - 2, 0]});
    const after = stateAt(target, T);
    for (let i = 0; i < 3; i++) {
      close(after.position[i], before.position[i], 1e-6 * (1 + Math.abs(before.position[i])), 'position kept');
      close(after.velocity[i], before.velocity[i], 1e-9 * (1 + Math.abs(before.velocity[i])), 'velocity kept');
    }
  }
});

test('re-anchoring is stable over many edits', () => {
  const target = {position: [5000, 0, 0], velocity: [-30, 10, 0], acceleration: [0, 0, 0]};
  for (let k = 1; k <= 2000; k++) {
    const T = k * 0.37, now = stateAt(target, T);
    setStateAt(target, T, {velocity: [now.velocity[0] + (k % 2 ? 0.1 : -0.1), now.velocity[1], now.velocity[2]]});
  }
  // 1000 +0.1 and 1000 −0.1 edits: the velocity returns to where it started.
  close(stateAt(target, 740).velocity[0], -30, 1e-9, 'velocity');
  assert.ok(target.position.every(Number.isFinite));
});

console.log(`${passed} passed`);
