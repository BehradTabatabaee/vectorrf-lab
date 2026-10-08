/* Application state.  Model defaults come from the backend (/api/defaults);
   only UI choices that differ from the API defaults are written here. */

export const state = {
  mode: 'tx',
  ready: false,
  defaults: null,
  limits: {max_sample_rate: 5e9, max_span: 5e9, max_capture_samples: 8_388_608, max_export_samples: 33_554_432},
  extensions: [],
  captureId: 0,
  revision: 0,
  rxRevision: 0,
};

export const clone = value => structuredClone(value);

export function getPath(obj, path) {
  return path.split('.').reduce((node, key) => node?.[/^\d+$/.test(key) ? Number(key) : key], obj);
}

export function setPath(obj, path, value) {
  const keys = path.split('.');
  let node = obj;
  for (const key of keys.slice(0, -1)) node = node[/^\d+$/.test(key) ? Number(key) : key];
  const last = keys.at(-1);
  node[/^\d+$/.test(last) ? Number(last) : last] = value;
}

/** Fill the state from backend defaults plus the UI's own starting choices. */
export function initState(defaults) {
  const d = clone(defaults);
  state.defaults = defaults;
  Object.assign(state, {
    // A clean QPSK carrying a full-period PN15 sequence: white data gives the
    // textbook root-raised-cosine spectrum (text repeats every few symbols and
    // shows as spectral lines).
    tx: {...d.tx, waveform: 'qpsk', sample_rate: 1e6, symbol_rate: 50e3, duration: 0.02, amplitude: 1, snr_db: 40,
         data_source: 'pn', pn_order: 15, pn_bits: 32767, rolloff: 0.35, filter_span_symbols: 16},
    txUi: {page: 'signal', noise: true, component: 0, family: 'linear', plan: null, planError: '', planRevision: -1},
    source: {kind: 'transmitter', file: null, info: null},
    pacing: 'realtime',
    files: {list: [], directory: '', selected: null, info: null, overview: null, busy: false, upload: null,
            error: '', filter: '', loaded: false},
    playback: {step: null},  // window step for every-sample recording playback (null = one window)
    exportCfg: {format: 'ci16_le', container: 'sigmf', scaling: 'peak', headroom_db: 1, destination: 'download',
                duration_s: null, excerpt_samples: 65536},
    spectrum: {...d.spectrum, span_hz: null, rbw_hz: 2000, averaging: 16, max_fft: 131072, channels: []},
    specUi: {page: 'sweep', autoRbw: true, rbwRatio: 500, ref: 0, perDiv: 10, autoRef: true, waterfall: true,
             rfAxis: true, hold: false, traceAverage: 16,
             traces: [{mode: 'clear', y: null}, {mode: 'off', y: null}, {mode: 'off', y: null}, {mode: 'off', y: null}],
             markers: [], activeMarker: -1, memories: [], selection: null},
    measurement: {offset_s: 0, duration_s: 0.002, center_hz: 0, bandwidth_hz: null, tuned: false},
    rx: {...d.rx, modulation: 'qpsk', sample_rate: 1e6, symbol_rate: 50e3, timing_recovery: 'mueller-muller',
         coarse_cfo: true},
    rxUi: {page: 'monitor', follow: true, dehop: true, component: 0, output: null, syncSpeed: 1, syncWindow: 180, syncPause: false,
           loopTrace: 'frequency', replay: 0},
    // A gentle default channel: the loops visibly acquire, and the first
    // Demodulation view decodes cleanly instead of looking broken.
    channel: {...d.channel, snr_db: 25, freq_offset_hz: 300, phase_offset_deg: 20, timing_offset_samples: 0.15},
    pipeline: clone(d.pipeline),
    blindUi: {page: 'overview', interval: 1500, threshold: 0.5, paused: false,
              band: {enabled: false, center_hz: 0, width_hz: 100e3}, history: []},
    studio: {...clone(d.radar_experiment), source: 'legacy', samples: 4096, page: 'waveform', running: true,
             speed: 10, fps: 30, file_samples: 4096, selectedItem: 0, selectedTarget: 0,
             plots: ['rx', 'spectrum', 'range', 'rd'], trails: {}},
  });
  state.rx.ofdm_design = clone(state.tx.ofdm_design);
  state.ready = true;
}

export function bump(kind = 'revision') {
  state[kind] = (state[kind] || 0) + 1;
}
