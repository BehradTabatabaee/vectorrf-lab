/* Reference constellations drawn under measured symbols (unit average power,
   matching the backend's mappings). */

const BASE = {dbpsk: 'bpsk', dqpsk: 'qpsk', 'pi4-dqpsk': 'qpsk', oqpsk: 'qpsk'};

export function idealConstellation(modulation, format = {}) {
  const mod = BASE[modulation] || modulation;
  if (mod === 'bpsk') return [[-1, 0], [1, 0]];
  if (mod === 'qpsk') return [[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([i, q]) => [i / Math.SQRT2, q / Math.SQRT2]);
  if (mod === '8psk') return Array.from({length: 8}, (_, k) => [Math.cos(Math.PI * k / 4), Math.sin(Math.PI * k / 4)]);
  if (mod === '16qam' || mod === '64qam') {
    const levels = mod === '16qam' ? [-3, -1, 1, 3] : [-7, -5, -3, -1, 1, 3, 5, 7];
    const norm = Math.sqrt(2 * levels.reduce((s, v) => s + v * v, 0) / levels.length);
    return levels.flatMap(i => levels.map(q => [i / norm, q / norm]));
  }
  if (mod === '16apsk' || mod === '32apsk') {
    const counts = mod === '16apsk' ? [4, 12] : [4, 12, 16];
    const radii = [1, format.apsk_ratio2 ?? 2.85, format.apsk_ratio3 ?? 5.27];
    const points = counts.flatMap((n, ring) => Array.from({length: n}, (_, i) => {
      const phase = 2 * Math.PI * i / n + Math.PI / n;
      return [radii[ring] * Math.cos(phase), radii[ring] * Math.sin(phase)];
    }));
    const norm = Math.sqrt(points.reduce((s, [i, q]) => s + i * i + q * q, 0) / points.length);
    return points.map(([i, q]) => [i / norm, q / norm]);
  }
  return [];
}
