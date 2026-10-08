/* Application shell: start-up, workspace routing and the live payload. */
import {$, esc} from './core/dom.js';
import {get, post, errorText} from './core/api.js';
import {connect, disconnect, setPayloadProvider, update as liveUpdate} from './core/live.js';
import {subscribe} from './core/events.js';
import {initState, state} from './core/state.js';
import {applyTx, designRequest, liveSource, txRequest} from './core/source.js';
import {countFrame, renderStatus, setStatus, showEngineError, showWork} from './core/ui.js';
import {loadDesign, saveDesign} from './core/persist.js';
import * as transmitter from './workspaces/transmitter.js';
import * as recordings from './workspaces/recordings.js';
import * as spectrum from './workspaces/spectrum.js';
import * as receiver from './workspaces/receiver.js';
import * as classifier from './workspaces/classifier.js';
import * as radar from './workspaces/radar.js';

const WORKSPACES = [
  ['tx', 'Transmitter', transmitter], ['files', 'Recordings', recordings], ['spectrum', 'Spectrum', spectrum],
  ['rx', 'Receiver', receiver], ['blind', 'Classifier', classifier], ['radar', 'Radar Studio', radar],
];
const byId = Object.fromEntries(WORKSPACES.map(([id, , module]) => [id, module]));
let current = null;

function renderTabs() {
  $('#workspaceTabs').innerHTML = WORKSPACES.map(([id, label]) =>
    `<button type="button" class="workspace-tab${state.mode === id ? ' active' : ''}" data-mode="${id}"${state.mode === id ? ' aria-current="page"' : ''}>${esc(label)}</button>`).join('');
}

/** Live configuration: the shared source plus what the active workspace observes. */
function payload() {
  const extra = current?.payload?.() || {};
  return {tx: txRequest(), source: liveSource(), fps: 30, pacing: state.pacing, pipeline: null, rx: null, channel: state.channel, ...extra};
}

export function switchMode(mode) {
  if (!byId[mode] || !state.ready) return;
  current?.stop?.();
  state.mode = mode;
  current = byId[mode];
  try { localStorage.setItem('vectorrf.mode', mode); } catch { /* storage unavailable */ }
  renderTabs();
  current.render($('#workspace'));
  if (current.usesLive === false) disconnect();
  else { connect(); liveUpdate(true); }
  current.start?.();
  showEngineError(engineError);
  renderStatus();
}

async function init() {
  renderTabs();
  setStatus('connecting');
  try {
    const [health, defaults, extensions] = await Promise.all([
      get('/api/health'), get('/api/defaults'), get('/api/studio/extensions').catch(() => ({extensions: []})),
    ]);
    initState(defaults);
    Object.assign(state.limits, {
      max_sample_rate: health.max_sample_rate, max_span: health.max_spectrum_span,
      max_capture_samples: health.max_capture_samples, max_export_samples: health.max_export_samples,
    });
    state.extensions = extensions.extensions;
    $('#engineLed').className = 'ok';
    $('#engineText').textContent = `engine ${health.version}`;
    await restoreDesign();
    applyTx(designRequest());
  } catch (error) {
    $('#engineLed').className = 'err';
    $('#engineText').textContent = 'backend unavailable';
    setStatus(`backend unavailable: ${errorText(error)}`, true);
    $('#workspace').innerHTML = `<div class="empty-state"><h2>The VectorRF engine is not reachable</h2><p>Start the server (start.bat / start.sh) and reload this page.</p></div>`;
    return;
  }
  setPayloadProvider(payload);
  let mode = 'tx';
  try { mode = localStorage.getItem('vectorrf.mode') || 'tx'; } catch { /* storage unavailable */ }
  switchMode(byId[mode] ? mode : 'tx');
  setStatus('ready');
}

/** Restore the last transmitter design of this browser, if it still validates. */
async function restoreDesign() {
  const saved = loadDesign();
  if (!saved) return;
  try {
    const plan = await post('/api/tx/plan', saved.tx);
    state.tx = plan.config;
    state.txUi.noise = saved.noise !== false;
    if (state.tx.waveform === 'composite') state.txUi.family = 'composite';
  } catch { /* an outdated design is simply not restored */ }
}

$('#workspaceTabs').addEventListener('click', event => {
  const tab = event.target.closest('[data-mode]');
  if (tab) switchMode(tab.dataset.mode);
});

let resizeTimer = null;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => current?.resize?.(), 60);
});

document.addEventListener('visibilitychange', () => {
  if (!state.ready || !current) return;
  if (document.hidden) { current.stop?.(); disconnect(); }
  else { if (current.usesLive !== false) connect(); current.start?.(); }
});

let engineError = null;
$('#engineError').addEventListener('click', event => event.currentTarget.classList.toggle('open'));
subscribe('live:error', error => {
  engineError = error;
  showEngineError(error);
  if (error) setStatus('update not applied', true);
  else setStatus(state.source.kind === 'file' ? 'recording streaming' : 'transmitter streaming');
});
subscribe('live:psd', () => countFrame());
subscribe('live:metrics', metrics => showWork(metrics));
// Only accepted designs are remembered: a reload never starts from a rejected draft.
subscribe('tx:applied', () => saveDesign(state.tx, state.txUi.noise));
subscribe('live:open', () => setStatus(state.source.kind === 'file' ? 'recording streaming' : 'transmitter streaming'));
subscribe('navigate', mode => switchMode(mode));
subscribe('live:close', () => setStatus('stream reconnecting'));

// Inspection handle for debugging and the browser test suite.
window.vectorrf = {state, switchMode};

init();
