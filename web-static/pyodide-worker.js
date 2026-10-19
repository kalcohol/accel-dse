/* accel-dse static build: Pyodide worker.  Loads Pyodide (pinned, jsdelivr), unpacks the accel_dse package zip
   into the in-memory FS and answers {id, method, path, body} with accel_dse.serve.respond (the server's own
   /api dispatch + JSON serialisation).  Posts {type: 'progress', phase: download | init | package, mb} while loading. */
'use strict';
importScripts('static-config.js');
const C = self.ACCEL_STATIC;
const prog = (phase, extra) => self.postMessage({ type: 'progress', phase, ...extra });
prog('download', { mb: 0 });
importScripts(C.pyodideBase + 'pyodide.js');

// Stream the big files once with a byte counter (the browser HTTP cache then serves loadPyodide's own fetches).
let got = 0, last = 0;
async function pull(url, keep) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(url + ' HTTP ' + r.status);
  if (!r.body || !r.body.getReader) { const b = await r.arrayBuffer(); got += b.byteLength; return keep ? b : null; }
  const rd = r.body.getReader(), parts = [];
  for (;;) {
    const { done, value } = await rd.read();
    if (done) break;
    got += value.byteLength;
    if (keep) parts.push(value);
    const now = performance.now();
    if (now - last > 100) { last = now; prog('download', { mb: got / 1e6 }); }
  }
  if (!keep) return null;
  const out = new Uint8Array(parts.reduce((n, p) => n + p.byteLength, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.byteLength; }
  return out.buffer;
}

let call = null;
const boot = (async () => {
  const t0 = performance.now();
  const [zip] = await Promise.all([pull(C.packageZip, true), pull(C.pyodideBase + 'pyodide.asm.wasm', false),
    pull(C.pyodideBase + 'python_stdlib.zip', false)]);
  prog('init', { mb: got / 1e6 });
  const py = await loadPyodide({ indexURL: C.pyodideBase });
  const t1 = performance.now();
  prog('package', { mb: got / 1e6 });
  py.unpackArchive(zip, 'zip', { extractDir: '/home/pyodide/pkg' });
  py.runPython(`
import sys
sys.path.insert(0, "/home/pyodide/pkg")
from accel_dse import serve as _serve, __version__ as _ver
def _accel_call(method, path, body):
    st, payload = _serve.respond(method, path, body.encode("utf-8"))
    return st, payload.decode("utf-8")
`);
  call = py.globals.get('_accel_call');
  const t2 = performance.now();
  return { pyodide: py.version, python: py.runPython('import sys; sys.version.split()[0]'), accel: py.globals.get('_ver'),
           fetchMs: t1 - t0, importMs: t2 - t1, mb: got / 1e6 };
})();

boot.then((detail) => self.postMessage({ type: 'ready', detail }),
          (e) => self.postMessage({ type: 'fatal', error: String(e && e.message || e) }));

self.onmessage = async (ev) => {
  const { id, method, path, body } = ev.data;
  await boot;
  const t = performance.now();
  let status, text;
  try {
    const r = call(method, path, body || '');
    [status, text] = r.toJs();
    r.destroy();
  } catch (e) {        // respond() never raises; this is an interpreter-level failure
    status = 500; text = JSON.stringify({ error: 'pyodide: ' + String(e && e.message || e) });
  }
  self.postMessage({ id, status, text, ms: performance.now() - t });
};
