/* I/Q file export (transmitter captures and recording excerpts). */
import {download} from '../core/dom.js';
import {request, errorText} from '../core/api.js';
import {numberField, selectField} from '../core/controls.js';
import {bytesText, fmtCount} from '../core/format.js';
import {state} from '../core/state.js';
import {setStatus, toast} from '../core/ui.js';
import {publish} from '../core/events.js';

const FORMATS = [['ci16_le', 'ci16 · int16 I/Q'], ['cf32_le', 'cf32 · float32 I/Q'], ['ci8', 'ci8 · int8 I/Q'], ['cu8', 'cu8 · uint8 (RTL-SDR)'], ['cf64_le', 'cf64 · float64 I/Q']];
const CONTAINERS = [['sigmf', 'SigMF archive (.sigmf)'], ['wav', 'WAV / RF64 (2-channel)'], ['raw', 'Raw interleaved']];

export function exportFields(excerpt) {
  const c = state.exportCfg;
  return selectField('Sample format', 'exportCfg.format', FORMATS)
    + selectField('Container', 'exportCfg.container', CONTAINERS)
    + selectField('Scaling', 'exportCfg.scaling', [['peak', 'Peak-normalize'], ['unity', 'Unity (1.0 = full scale)']])
    + (c.scaling === 'peak' ? numberField('Headroom', 'exportCfg.headroom_db', {unit: 'dB', min: 0, max: 60}) : '')
    + (excerpt ? numberField('Samples', 'exportCfg.excerpt_samples', {min: 32, max: state.limits.max_export_samples, integer: true})
      : numberField('Duration', 'exportCfg.duration_s', {unit: 's', min: 1e-9, nullable: true, placeholder: 'one capture',
        help: `Empty exports the current capture exactly as the instruments see it; a duration renders one longer contiguous record (up to ${fmtCount(state.limits.max_export_samples)} samples).`}))
    + selectField('Destination', 'exportCfg.destination', [['download', 'Download to this computer'], ['library', 'Save into the recording library']]);
}

/** POST an export request and deliver the file. */
export async function runExport(body, label) {
  setStatus(`exporting ${label}`);
  try {
    const response = await request('/api/files/export', {method: 'POST', body, raw: true});
    if (body.destination === 'library') {
      const d = await response.json();
      toast(`Saved ${d.saved} to the library · ${fmtCount(d.samples)} samples${d.clipped ? ` · ${d.clipped} clipped` : ''}`, 'good');
      publish('library:changed', d.saved);
    } else {
      const blob = await response.blob();
      const name = (response.headers.get('Content-Disposition') || '').match(/filename="([^"]+)"/)?.[1] || 'capture.bin';
      download(blob, name);
      const clipped = Number(response.headers.get('X-VectorRF-Clipped') || 0);
      toast(`Exported ${name} · ${bytesText(blob.size)}${clipped ? ` · ${clipped} samples clipped` : ''}`, clipped ? 'bad' : 'good');
    }
    setStatus('export complete');
  } catch (error) {
    toast(errorText(error), 'bad');
    setStatus('export failed', true);
  }
}
