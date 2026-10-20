"""Stdlib HTTP server: static web UI + JSON API (core v2)."""

from __future__ import annotations

import json
import signal
import threading
from collections import OrderedDict
from concurrent.futures.process import BrokenProcessPool
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__, api

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY = 64 * 1024
GET_ROUTES = {"/api/health": api.api_health, "/api/models": api.api_models, "/api/catalog": api.api_catalog}
POST_ROUTES = {"/api/memory": api.api_memory, "/api/fit": api.api_fit, "/api/eval": api.api_eval, "/api/layouts": api.api_layouts,
               "/api/compare": api.api_compare, "/api/stability": api.api_stability,
               "/api/pareto": api.api_pareto, "/api/sweep": api.api_sweep,
               "/api/sensitivity": api.api_sensitivity, "/api/insight": api.api_insight,
               "/api/insight_path": api.api_insight_path,
               "/api/insight_sizing": api.api_insight_sizing, "/api/insight_slc": api.api_insight_slc,
               "/api/insight_speech": api.api_insight_speech,
               "/api/insight_speech_streams": api.api_insight_speech_streams}
CACHED = {"/api/compare", "/api/stability", "/api/layouts"}
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}


def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name} is not allowed")


def parse_json_strict(raw: bytes) -> object:
    """JSON parser that rejects NaN / Infinity (Python's json accepts them by default)."""
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)
    except RecursionError:      # 0.129: deeply nested JSON was a 500
        raise ValueError("nesting too deep") from None


class _LRU:
    def __init__(self, n: int = 64):
        self.n, self.d, self.lock = n, OrderedDict(), threading.Lock()

    def get(self, k):
        with self.lock:
            if k in self.d:
                self.d.move_to_end(k)
                return self.d[k]
        return None

    def put(self, k, v):
        with self.lock:
            self.d[k] = v
            self.d.move_to_end(k)
            while len(self.d) > self.n:
                self.d.popitem(last=False)


CACHE = _LRU()


def handle(method: str, path: str, raw: bytes = b"") -> tuple[int, dict]:
    """Route one API request; returns (status, payload).  Used by the server and tests."""
    if method == "GET" and path in GET_ROUTES:
        return 200, GET_ROUTES[path]()
    if method == "POST" and path in POST_ROUTES:
        if len(raw) > MAX_BODY:
            return 413, {"error": "request body too large"}
        try:
            body = parse_json_strict(raw or b"{}")
        except (ValueError, UnicodeDecodeError) as e:
            return 400, {"error": f"invalid JSON: {e}"}
        if not isinstance(body, dict):
            return 400, {"error": "body must be a JSON object"}
        key = (path, json.dumps(body, sort_keys=True)) if path in CACHED else None
        if key and (hit := CACHE.get(key)) is not None:
            return 200, hit
        try:
            pool = api._POOL
            try:
                out = POST_ROUTES[path](body)
            except BrokenProcessPool:           # 0.129: replace the dead pool and retry once
                api.reset_pool(pool)
                out = POST_ROUTES[path](body)
        except (api.ApiError, ValueError) as e:     # 0.74: model ValueErrors are input validation → 400, not 500
            return 400, {"error": str(e)}
        if key:
            CACHE.put(key, out)
        return 200, out
    if path in GET_ROUTES or path in POST_ROUTES:
        return 405, {"error": "method not allowed"}
    return 404, {"error": "not found"}


def respond(method: str, path: str, raw: bytes = b"") -> tuple[int, bytes]:
    """One /api call as the server puts it on the wire: (status, UTF-8 JSON bytes); never raises.  Transport-free,
    so the static build (scripts/build_static.py: the UI on Pyodide in the browser) calls it directly."""
    path = path.split("?", 1)[0]
    try:
        status, out = handle(method, path, raw)
        payload = json.dumps(api.clean(out), ensure_ascii=False, allow_nan=False).encode()
    except Exception as e:  # unexpected: report, never crash the server
        status = 500
        payload = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode()
    return status, payload


class Handler(BaseHTTPRequestHandler):
    timeout = 30      # 0.88: socket read / write timeout — a client that never sends its body no longer holds a thread
    server_version = f"accel-dse/{__version__}"

    def log_message(self, fmt, *args):  # quieter log: one line per API call
        if self.path.startswith("/api/"):
            super().log_message(fmt, *args)

    def _send(self, status: int, payload: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _api(self, method: str, raw: bytes = b"") -> None:
        status, payload = respond(method, self.path, raw)
        self._send(status, payload, "application/json")

    def do_GET(self):
        if self.path.startswith("/api/"):
            return self._api("GET")
        rel = self.path.split("?", 1)[0].lstrip("/") or "index.html"
        f = (WEB_DIR / rel).resolve()
        if WEB_DIR not in f.parents or not f.is_file() or f.suffix not in STATIC_TYPES:
            return self._send(404, b"not found", "text/plain")
        self._send(200, f.read_bytes(), STATIC_TYPES[f.suffix])

    def do_POST(self):
        cl = (self.headers.get("Content-Length") or "0").strip()
        if not (cl.isascii() and cl.isdigit()):        # 0.129: "-5" / "abc" crashed the handler (connection dropped, no response)
            self.close_connection = True
            return self._send(400, b'{"error":"invalid Content-Length"}', "application/json")
        n = int(cl)
        if n > MAX_BODY:
            return self._send(413, b'{"error":"request body too large"}', "application/json")
        try:
            body = self.rfile.read(n)
        except TimeoutError:
            self.close_connection = True
            return self._send(408, b'{"error":"request body not received in time"}', "application/json")
        self._api("POST", body)


def _on_sigterm(signum, frame):
    raise KeyboardInterrupt


class Server(ThreadingHTTPServer):
    """0.88: listen backlog 5 (socketserver default) reset 39–55 % of the connections of a 64–128-request burst
    (ECONNRESET); the kernel caps the backlog at net.core.somaxconn.  Handler threads are daemons so a stuck request
    cannot block shutdown."""
    request_queue_size = 1024
    daemon_threads = True


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = Server((host, port), Handler)
    workers = api.enable_pool()
    print(f"accel_dse v{__version__} http://{host}:{port}/ ({workers} search workers)", flush=True)
    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        api.shutdown_pool()
