/* The one live WebSocket: schedules nothing itself, it requests captures from
   the engine and republishes what arrives on the event bus.

   Events: live:meta, live:metrics, live:psd, live:receiver, live:pipeline,
   live:error, live:open, live:close. */
import {publish} from './events.js';
import {state} from './state.js';

export const live = {
  ws: null, ready: false, metrics: {}, freq: [], psd: null, frames: 0, lastFrameAt: 0, meta: null,
};

let payloadProvider = () => ({});
let reconnectTimer = null;
let attempts = 0;
let updateTimer = null;
let wanted = false;
let revision = 0;

export function setPayloadProvider(fn) {
  payloadProvider = fn;
}

export function connect() {
  wanted = true;
  if (document.hidden) return;
  if (live.ws && [WebSocket.OPEN, WebSocket.CONNECTING].includes(live.ws.readyState)) return;
  clearTimeout(reconnectTimer);
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`);
  ws.binaryType = 'arraybuffer';
  live.ws = ws;
  live.ready = false;
  ws.onopen = () => {
    if (live.ws !== ws) return;
    live.ready = true;
    attempts = 0;
    live.lastFrameAt = performance.now();
    send({...payloadProvider(), action: 'start', acknowledge_frames: true, capture_id: (state.captureId || 0) + 1,
      client_revision: ++revision});
    publish('live:open');
  };
  ws.onmessage = event => {
    if (live.ws !== ws) return;
    if (typeof event.data === 'string') handleText(JSON.parse(event.data));
    else handleBinary(event.data);
  };
  ws.onclose = () => {
    if (live.ws !== ws) return;
    live.ws = null;
    live.ready = false;
    publish('live:close');
    if (wanted) reconnectTimer = setTimeout(connect, Math.min(8000, 500 * 2 ** attempts++));
  };
}

export function disconnect() {
  wanted = false;
  clearTimeout(reconnectTimer);
  clearTimeout(updateTimer);
  const ws = live.ws;
  live.ws = null;
  live.ready = false;
  ws?.close();
}

function send(message) {
  if (live.ws?.readyState === WebSocket.OPEN) live.ws.send(JSON.stringify(message));
}

/** Send the current configuration (coalesced so a burst of edits sends once). */
export function update(immediate = false) {
  clearTimeout(updateTimer);
  const current = ++revision;
  const run = () => {
    if (live.ws?.readyState === WebSocket.OPEN) send({...payloadProvider(), action: 'update', client_revision: current});
    else if (wanted) connect();
  };
  if (immediate) run(); else updateTimer = setTimeout(run, 90);
}

/** Restart the stream at a capture (seeking a recording, resetting a source). */
export function seek(captureId) {
  send({...payloadProvider(), action: 'update', seek_capture: captureId, client_revision: ++revision});
}

function handleText(message) {
  switch (message.type) {
    case 'metrics':
      // An older revision still proves the transport is active while edits
      // are being coalesced; it must not update any current display state.
      live.lastFrameAt = performance.now();
      if (message.client_revision != null && message.client_revision !== revision) break;
      live.metrics = message;
      if (Array.isArray(message.freq_hz) && message.freq_hz.length) live.freq = message.freq_hz;
      state.captureId = message.capture_id ?? state.captureId;
      publish('live:metrics', message);
      break;
    case 'meta':
      if (message.client_revision != null && message.client_revision !== revision) break;
      live.meta = message;
      publish('live:meta', message);
      break;
    case 'receiver': case 'pipeline': case 'measurement':
      if (message.client_revision != null && message.client_revision !== revision) break;
      publish(`live:${message.type}`, message);
      break;
    case 'frame_end':
      send({action: 'ack', seq: message.seq});
      break;
    case 'error':
      if (message.client_revision != null && message.client_revision !== revision) break;
      publish('live:error', {message: message.message, scope: message.scope || 'request'});
      break;
    case 'accepted': case 'product_recovered':
      if (message.client_revision != null && message.client_revision !== revision) break;
      publish('live:error', null);
      break;
    default:
      break;
  }
}

function handleBinary(buffer) {
  if (buffer.byteLength < 16) return;
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== 0x31465256 || view.getUint32(12, true) !== buffer.byteLength - 16) return;
  if (view.getUint16(4, true) !== 1) return;
  if (live.metrics.client_revision != null && live.metrics.client_revision !== revision) return;
  if (view.getUint32(8, true) !== live.metrics.seq) return;
  live.psd = new Float32Array(buffer, 16);
  live.frames += 1;
  live.lastFrameAt = performance.now();
  publish('live:psd', live.psd);
}

// A stalled socket (no frame for 5 s while connected) is replaced.
setInterval(() => {
  if (live.ready && !document.hidden && performance.now() - live.lastFrameAt > 5000) {
    const ws = live.ws;
    live.ws = null;
    ws?.close();
    connect();
  }
}, 2000);
