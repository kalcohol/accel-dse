/* accel-dse static build: route the UI's /api/* fetches to accel_dse running on Pyodide in Web Workers.
   Loaded before app.js; app.js is unchanged (it still calls fetch('/api/...')).

   Two lanes, each its own worker (= its own Python interpreter):
     fast  — eval / sensitivity / memory / sweep / pareto / catalog …: sub-second to a few seconds;
     slow  — fit / compare / layouts / stability / insight*: searches that can take tens of seconds in WebAssembly.
   so a long capacity search never blocks the single-point evaluation.  Requests in a lane are serialised; a request
   aborted while queued is dropped before it reaches Python (app.js aborts stale requests per channel).  A *running*
   slow-lane request that is aborted terminates its worker (Python cannot be interrupted without cross-origin
   isolation, which GitHub Pages cannot enable) and a fresh one is started; a running fast-lane request finishes and
   its reply is discarded. */
(function () {
  'use strict';
  const CFG = window.ACCEL_STATIC || {};
  const SLOW = new Set(CFG.slowRoutes || ['/api/fit', '/api/compare', '/api/layouts', '/api/stability',
    '/api/insight', '/api/insight_path', '/api/insight_sizing', '/api/insight_slc', '/api/insight_speech',
    '/api/insight_speech_streams', '/api/insight_dispatch']);
  // 0.285: every route the UI calls is classified explicitly (tests/test_static_routes.py: UI calls ⊆ FAST ∪ SLOW);
  // an unclassified /api route still works — it goes to the slow (abortable) lane with a console warning
  const FAST = new Set(CFG.fastRoutes || ['/api/health', '/api/catalog', '/api/models', '/api/memory', '/api/eval',
    '/api/sensitivity', '/api/sweep', '/api/pareto']);
  let nextId = 1, readyResolve, readyReject;
  const ready = new Promise((res, rej) => { readyResolve = res; readyReject = rej; });
  const T = (window.__accelStatic = { log: [], ready, t0: performance.now(), loadMs: null, error: null, restarts: 0, unclassified: new Set(),
    capture: !!window.__accelCapture, busy: () => lanes.reduce((n, l) => n + l.queue.length + (l.running ? 1 : 0), 0) });

  // first-load phases → the app's progress strip (app.js listens; it also reads __accelStatic.progress on start)
  function progress(d) {
    T.progress = d;
    window.dispatchEvent(new CustomEvent('accel-static-progress', { detail: d }));
  }
  progress({ phase: 'download', mb: 0 });

  class Lane {
    constructor(name) { this.name = name; this.queue = []; this.running = null; this.ok = false; }
    start() {
      const t0 = performance.now();
      this.ok = false;
      this.w = new Worker(CFG.worker || 'pyodide-worker.js');
      this.w.onmessage = (ev) => this.onmsg(ev.data, t0);
      this.w.onerror = (e) => { T.error = e.message; if (this.name === 'fast') { progress({ phase: 'error', msg: e.message }); readyReject(e); } };
    }
    onmsg(m, t0) {
      if (m.type === 'progress') { if (this.name === 'fast' && !T.loadMs) progress({ phase: m.phase, mb: m.mb }); return; }
      if (m.type === 'ready') {
        this.ok = true;
        this.loadMs = performance.now() - t0;
        if (this.name === 'fast') {
          T.loadMs = performance.now() - T.t0; T.detail = m.detail;
          progress({ phase: 'ready', mb: m.detail.mb, ms: T.loadMs });
          readyResolve(m.detail);
          if (!slow.w) slow.start();       // second interpreter once the first is up (CDN files are cached by then)
        } else {
          T.slowLoadMs = this.loadMs;
        }
        return this.pump();
      }
      if (m.type === 'fatal') {
        T.error = m.error;
        if (this.name === 'fast') { progress({ phase: 'error', msg: m.error }); readyReject(new Error(m.error)); }
        for (const p of this.queue.splice(0)) p.reject(new Error(m.error));
        return;
      }
      const p = this.running;
      this.running = null;
      if (p && p.id === m.id) {
        const e = { path: p.path, method: p.method, lane: this.name, ms: m.ms, wallMs: performance.now() - p.t,
          status: m.status, bytes: m.text.length, aborted: p.aborted };
        if (T.capture) { e.body = p.body; e.text = m.text; }
        T.log.push(e);
        if (!p.aborted) p.resolve(new Response(m.text, { status: m.status, headers: { 'Content-Type': 'application/json' } }));
      }
      this.pump();
    }
    pump() {
      if (this.running || !this.ok) return;
      while (this.queue.length) {
        const p = this.queue.shift();
        if (p.aborted) { T.log.push({ path: p.path, lane: this.name, dropped: true }); continue; }
        this.running = p;
        this.w.postMessage({ id: p.id, method: p.method, path: p.path, body: p.body });
        return;
      }
    }
    abortRunning(p) {
      if (this.running !== p || this.name !== 'slow') return;      // fast lane: let it finish, discard the reply
      this.w.terminate();
      this.running = null; T.restarts++;
      T.log.push({ path: p.path, lane: this.name, killed: true, wallMs: performance.now() - p.t });
      this.start();
    }
    submit(p) { this.queue.push(p); this.pump(); }
  }
  const fast = new Lane('fast'), slow = new Lane('slow'), lanes = [fast, slow];
  fast.start();

  const realFetch = window.fetch.bind(window);
  window.fetch = function (input, init) {
    const url = typeof input === 'string' ? input : (input && input.url) || '';
    if (!url.startsWith('/api/')) return realFetch(input, init);
    init = init || {};
    const method = (init.method || 'GET').toUpperCase();
    const body = typeof init.body === 'string' ? init.body : '';
    const route = url.split('?', 1)[0];
    if (!SLOW.has(route) && !FAST.has(route) && !T.unclassified.has(route)) { T.unclassified.add(route); console.warn('accel static: unclassified route ' + route + ' → slow lane'); }
    const lane = SLOW.has(route) || !FAST.has(route) ? slow : fast;
    return new Promise((resolve, reject) => {
      const p = { id: nextId++, method, path: url, body, resolve, reject, aborted: false, t: performance.now() };
      const sig = init.signal;
      if (sig) {
        if (sig.aborted) return reject(new DOMException('Aborted', 'AbortError'));
        sig.addEventListener('abort', () => { p.aborted = true; reject(new DOMException('Aborted', 'AbortError')); lane.abortRunning(p); });
      }
      lane.submit(p);
    });
  };
})();
