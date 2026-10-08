"""Stdlib HTTP server: static web UI + JSON API (core v2)."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import __version__, api

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY = 64 * 1024
GET_ROUTES = {"/api/health": api.api_health, "/api/models": api.api_models, "/api/catalog": api.api_catalog}
POST_ROUTES = {"/api/memory": api.api_memory, "/api/eval": api.api_eval, "/api/layouts": api.api_layouts,
               "/api/compare": api.api_compare, "/api/stability": api.api_stability,
               "/api/pareto": api.api_pareto, "/api/sweep": api.api_sweep}
CACHED = {"/api/compare", "/api/stability", "/api/layouts"}
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png"}


def _reject_constant(name: str):
    raise ValueError(f"non-finite JSON constant {name} is not allowed")


def parse_json_strict(raw: bytes) -> object:
    """JSON parser that rejects NaN / Infinity (Python's json accepts them by default)."""
    return json.loads(raw.decode("utf-8"), parse_constant=_reject_constant)


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
            out = POST_ROUTES[path](body)
        except api.ApiError as e:
            return 400, {"error": str(e)}
        if key:
            CACHE.put(key, out)
        return 200, out
    if path in GET_ROUTES or path in POST_ROUTES:
        return 405, {"error": "method not allowed"}
    return 404, {"error": "not found"}


class Handler(BaseHTTPRequestHandler):
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
        path = self.path.split("?", 1)[0]
        try:
            status, out = handle(method, path, raw)
            payload = json.dumps(api.clean(out), ensure_ascii=False, allow_nan=False).encode()
        except Exception as e:  # unexpected: report, never crash the server
            status = 500
            payload = json.dumps({"error": f"{type(e).__name__}: {e}"}, ensure_ascii=False).encode()
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
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY:
            return self._send(413, b'{"error":"request body too large"}', "application/json")
        self._api("POST", self.rfile.read(n))


def run_server(host: str = "127.0.0.1", port: int = 8765) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"accel_dse v{__version__} http://{host}:{port}/", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
