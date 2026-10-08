/* Minimal publish/subscribe bus between the live stream, workspaces and shell. */

const listeners = new Map();

export function subscribe(event, handler) {
  if (!listeners.has(event)) listeners.set(event, new Set());
  listeners.get(event).add(handler);
  return () => listeners.get(event)?.delete(handler);
}

export function publish(event, payload) {
  for (const handler of listeners.get(event) || []) {
    try { handler(payload); } catch (error) { console.error(`${event} handler failed`, error); }
  }
}
