/* HTTP client.  Validation errors from FastAPI become readable sentences. */

export class ApiError extends Error {
  constructor(message, body = null) {
    super(message);
    this.body = body;  // the parsed error body (``errors``, ``components`` …) when there is one
  }
}

function describe(detail) {
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) {
    return detail.map(e => {
      const where = (e.loc || []).filter(part => part !== 'body').join(' → ');
      const message = String(e.msg || '').replace(/^Value error, /, '');
      return where ? `${where}: ${message}` : message;
    }).join('; ');
  }
  return JSON.stringify(detail);
}

export async function request(path, {method = 'GET', body, raw = false, signal} = {}) {
  const options = {method, signal};
  if (body !== undefined) {
    options.headers = {'Content-Type': 'application/json'};
    options.body = JSON.stringify(body);
  }
  const response = await fetch(path, options);
  if (!response.ok) {
    const text = await response.text();
    let message = text, body = null;
    try { body = JSON.parse(text); message = describe(body.detail ?? text); } catch { /* plain-text error */ }
    throw new ApiError(message || `${response.status} ${response.statusText}`, body);
  }
  if (raw) return response;
  return (response.headers.get('content-type') || '').includes('json') ? response.json() : response.blob();
}

export const get = path => request(path);
export const post = (path, body, options) => request(path, {...options, method: 'POST', body});
export const del = path => request(path, {method: 'DELETE'});

export function errorText(error) {
  const text = String(error?.message || error);
  return text.length > 400 ? `${text.slice(0, 397)}…` : text;
}
