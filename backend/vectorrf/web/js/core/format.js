/* Number formatting and engineering-notation parsing.

   Fields accept instrument-style entries such as "2.4G", "2.4 GHz", "2.4ghz",
   "100k", "20us", "1e6" or "−3 dBm".  SI prefixes apply only to units that
   take them.  "m" is milli and "M" mega, except in frequency, sample-rate and
   symbol-rate fields, where nobody means millihertz: there "10mhz" is 10 MHz
   (those fields never display sub-unit prefixes, so values still round-trip). */

const PREFIXES = [[1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k'], [1, ''], [1e-3, 'm'], [1e-6, 'µ'], [1e-9, 'n'], [1e-12, 'p']];
const PREFIX_VALUE = {T: 1e12, t: 1e12, G: 1e9, g: 1e9, M: 1e6, k: 1e3, K: 1e3, '': 1, m: 1e-3, u: 1e-6, U: 1e-6, n: 1e-9, N: 1e-9, p: 1e-12, P: 1e-12};
// Rates: "m" means mega and no sub-unit prefix is ever shown.
const RATE_UNITS = new Set(['Hz', 'S/s', 'Bd']);
// Units that take SI prefixes, with accepted spellings (compared case-insensitively).
const PREFIXED = {Hz: ['hz'], s: ['s', 'sec'], 'S/s': ['s/s', 'sps'], Bd: ['bd', 'baud'], m: ['m'], W: ['w'], 'm/s': [], V: ['v']};
const PLAIN = {dB: ['db'], dBm: ['dbm'], dBFS: ['dbfs'], dBc: ['dbc'], '°': ['°', 'deg'], '%': ['%'], 'dB/Hz': ['db/hz'], 'm/s': ['m/s', 'mps'],
  'm/s²': ['m/s²', 'm/s2'], 'm²': ['m²', 'm2'], dBi: ['dbi'], λ: ['λ'], '×': ['×', 'x'], '°/s': ['°/s', 'deg/s'], 'Hz/s': ['hz/s']};

const takesPrefix = unit => Object.hasOwn(PREFIXED, unit) && unit !== 'm/s';

function trimNumber(text) {
  return text.includes('e') ? text : text.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '');
}

function mantissa(value, significant = 12) {
  if (value === 0) return '0';
  const text = Number(value).toPrecision(significant);
  return trimNumber(text.includes('e') ? String(Number(text)) : text);
}

function prefixFor(value, unit = '') {
  const a = Math.abs(value);
  if (a === 0) return [1, ''];
  const found = PREFIXES.find(([scale]) => a >= scale * 0.9999999999) || PREFIXES.at(-1);
  return RATE_UNITS.has(unit) && found[0] < 1 ? [1, ''] : found;
}

/** Exact editable representation, e.g. 2.4e9 → "2.4 GHz", 2e-5 → "20 µs". */
export function formatQuantity(value, unit = '') {
  const v = Number(value);
  if (value === null || value === undefined || value === '' || !Number.isFinite(v)) return '';
  if (!unit) return mantissa(v);
  if (!takesPrefix(unit)) return `${mantissa(v)} ${unit}`;
  const [scale, prefix] = prefixFor(v, unit);
  return `${mantissa(v / scale)} ${prefix}${unit}`;
}

/** Parse instrument-style input; returns NaN when the text is not a quantity in ``unit``. */
export function parseQuantity(text, unit = '') {
  let s = String(text ?? '').trim().replace(/[−–]/g, '-').replace(/[μµ]/g, 'u').replace(/[\s_]/g, '');
  if (/^[+-]?\d{1,3}(,\d{3})+(\.\d*)?/.test(s)) s = s.replace(/,/g, '');
  const match = /^([+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?)(.*)$/i.exec(s);
  if (!match) return NaN;
  let rest = match[2];
  const value = Number(match[1]);
  const spellings = [...(PREFIXED[unit] || []), ...(PLAIN[unit] || []), unit.toLowerCase()].filter(Boolean)
    .sort((a, b) => b.length - a.length);
  const lower = rest.toLowerCase();
  for (const spelling of spellings) {
    if (lower.endsWith(spelling.toLowerCase().replace(/[μµ]/g, 'u'))) { rest = rest.slice(0, rest.length - spelling.length); break; }
  }
  if (!rest) return value;
  if ((takesPrefix(unit) || !unit) && rest.length === 1 && Object.hasOwn(PREFIX_VALUE, rest)) {
    return value * (RATE_UNITS.has(unit) && rest === 'm' ? 1e6 : PREFIX_VALUE[rest]);
  }
  return NaN;
}

/** Compact readout with SI prefix, e.g. 2.4e9 → "2.400 GHz", -45.31 dBm → "-45.31 dBm". */
export function fmt(value, unit = '', digits = 3) {
  const v = Number(value);
  if (value === null || value === undefined || !Number.isFinite(v)) return '—';
  if (unit && !takesPrefix(unit)) {
    const a = Math.abs(v);
    return `${v.toFixed(a >= 1000 ? 0 : a >= 100 ? 1 : 2)} ${unit}`;
  }
  const [scale, prefix] = prefixFor(v);
  const m = v / scale;
  const a = Math.abs(m);
  const text = a >= 100 ? m.toFixed(Math.max(0, digits - 2)) : a >= 10 ? m.toFixed(Math.max(0, digits - 1)) : m.toFixed(digits);
  return `${text} ${prefix}${unit}`.trim();
}

/** Plain number with grouping, for sample counts. */
export function fmtCount(value) {
  const v = Number(value);
  return Number.isFinite(v) ? Math.round(v).toLocaleString('en-US') : '—';
}

/** Frequency with enough digits to resolve ``resolution`` Hz (markers near GHz still show kHz). */
export function fmtFreq(hz, resolution = 1) {
  const v = Number(hz);
  if (!Number.isFinite(v)) return '—';
  const res = Math.max(1e-3, Math.abs(Number(resolution) || 1));
  const a = Math.abs(v);
  const [scale, prefix] = a >= 1e9 ? [1e9, 'G'] : a >= 1e6 ? [1e6, 'M'] : a >= 1e3 ? [1e3, 'k'] : [1, ''];
  const decimals = Math.max(0, Math.min(9, Math.ceil(Math.log10(scale / res))));
  return `${(v / scale).toFixed(decimals)} ${prefix}Hz`;
}

export function fmtDb(value, unit = 'dB') {
  const v = Number(value);
  return Number.isFinite(v) ? `${v.toFixed(2)} ${unit}` : '—';
}

export function bytesText(n) {
  let v = Number(n) || 0;
  const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB'];
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(i ? 1 : 0)} ${units[i]}`;
}

export function percent(value, digits = 0) {
  const v = Number(value);
  return Number.isFinite(v) ? `${(100 * v).toFixed(digits)} %` : '—';
}
