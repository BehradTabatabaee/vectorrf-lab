/* Per-browser conveniences in localStorage.  Every access may fail (private
   windows, blocked storage); the application works without it. */

const DESIGN_KEY = 'vectorrf.design.v1';
let saveTimer = null;

export function saveDesign(tx, noise) {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try { localStorage.setItem(DESIGN_KEY, JSON.stringify({tx, noise})); } catch { /* storage unavailable */ }
  }, 800);
}

export function loadDesign() {
  try { return JSON.parse(localStorage.getItem(DESIGN_KEY) || 'null'); } catch { return null; }
}

export function loadRecordingSettings(name) {
  try { return JSON.parse(localStorage.getItem(`vectorrf.recording.${name}`) || 'null') || {}; } catch { return {}; }
}

const RECORDING_KEYS = ['format', 'sample_rate', 'center_frequency_hz', 'header_bytes', 'power_offset_db', 'power_unit', 'capture_samples'];
export function saveRecordingSettings(settings) {
  try {
    localStorage.setItem(`vectorrf.recording.${settings.name}`, JSON.stringify(Object.fromEntries(RECORDING_KEYS.map(k => [k, settings[k]]))));
  } catch { /* storage unavailable */ }
}
