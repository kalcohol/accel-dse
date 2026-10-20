"""0.285 opt-in static-build (GitHub Pages / Pyodide) smoke (needs Playwright + Chrome + the jsdelivr CDN; not part of
run_tests).  Builds dist-static into a temp dir, serves it with http.server, loads the page in headless Chromium and:
  1. enumerates every endpoint app.js calls (req(chan, '/api/…')) and checks it is classified in the shim (fast / slow
     lane) and answered by the Pyodide worker (GET 200; POST with a bad body → 400, i.e. routed, not 404 / 500);
  2. drives the features added since 0.270 through the real UI: eval walls note with the dispatch sensitivity (0.284),
     speech insight with the op-dispatch column (0.283), LLM dispatch panel (0.284), speech streams vs SLC (0.276);
  3. checks the non-blocking progress pill appears while a slow-lane request runs and that stale results are handled:
     a re-click supersedes a running slow-lane request (worker terminated / reply dropped) and a scenario edit during a
     run tags the panel stale.
Usage: python3 scripts/ui_static_smoke.py [--chrome /usr/bin/google-chrome]   exit 0 = clean (prints "N error(s)").
If the CDN is unreachable it prints "SKIP" and exits 0."""
from __future__ import annotations

import functools
import http.server
import importlib.util
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _ws():
    """0.286: the stdlib tests' parser (tests/webstruct.py) — one source of truth for call sites and shim lanes."""
    return _mod("webstruct", ROOT / "tests/webstruct.py")


def ui_calls() -> set[tuple[str, str]]:
    W = _ws()
    return {("POST" if c["body"] else "GET", c["path"]) for c in W.req_calls(W.read(W.APP_JS))}


def shim_lanes() -> tuple[set[str], set[str]]:
    W = _ws()
    shim = W.read(W.SHIM_JS)
    return W.list_values(shim, "fastRoutes"), W.list_values(shim, "slowRoutes")


def main(chrome="/usr/bin/google-chrome") -> int:
    bs = _mod("build_static", ROOT / "scripts/build_static.py")
    try:
        urllib.request.urlopen(bs.CDN.format(v=bs.PYODIDE_VERSION) + "pyodide.js", timeout=15).close()
    except OSError as e:
        print(f"SKIP: Pyodide CDN unreachable ({e})")
        return 0
    from playwright.sync_api import sync_playwright
    errs, steps = [], []
    fast, slow = shim_lanes()
    calls = ui_calls()
    for _, path in sorted(calls):
        if path not in fast | slow:
            errs.append(f"shim: {path} is not classified (fastRoutes / slowRoutes)")
    with tempfile.TemporaryDirectory() as d:
        bs.build(Path(d), bs.PYODIDE_VERSION)
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a, **k):
                pass
        h = functools.partial(Quiet, directory=d)
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), h)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{srv.server_address[1]}/"
        try:
            with sync_playwright() as pw:
                b = pw.chromium.launch(executable_path=chrome, args=["--headless=new", "--no-sandbox"])
                p = b.new_page(viewport={"width": 1600, "height": 1000})
                p.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
                p.on("console", lambda m: m.type == "error" and errs.append(f"console: {m.text}"))
                t0 = time.time()
                p.goto(url)
                p.wait_for_function("window.__accelStatic && __accelStatic.loadMs !== null", timeout=300000)
                p.wait_for_selector("#kpis .kpi", timeout=300000)
                steps.append(("cold load", round(time.time() - t0, 1)))

                def wait_text(sel, pred, timeout=900000):
                    p.wait_for_function(f"(() => {{ const t = (document.querySelector({sel!r}) || {{}}).textContent || '';"
                                        f" return {pred}; }})()", timeout=timeout)
                    return p.text_content(sel) or ""

                # 1. every UI endpoint is answered by the worker
                for method, path in sorted(calls):
                    r = p.evaluate("""async ([m, path]) => {
                        const opt = m === 'POST' ? {method: 'POST', headers: {'Content-Type': 'application/json'}, body: '[1]'} : {};
                        const r = await fetch(path, opt); const txt = await r.text();
                        const e = __accelStatic.log[__accelStatic.log.length - 1];
                        return {status: r.status, lane: e && e.path === path ? e.lane : null, len: txt.length}; }""",
                                   [method, path])
                    ok = r["lane"] in ("fast", "slow") and (r["status"] == 200 if method == "GET" else r["status"] == 400)
                    want = "slow" if path in slow else "fast"
                    if not ok or r["lane"] != want:
                        errs.append(f"static {method} {path}: {r} (want lane {want})")
                steps.append(("endpoints answered by the worker", len(calls)))
                if p.evaluate("__accelStatic.unclassified.size"):
                    errs.append(f"unclassified routes hit: {p.evaluate('[...__accelStatic.unclassified]')}")

                # 2a. eval: walls note carries the dispatch sensitivity
                note = wait_text("#walls-note", "t.includes('派发敏感度')", 120000)
                steps.append(("eval dispatch_sens", "派发敏感度" in note))

                p.click("button[data-tab=insight]")
                # 2b + 3. dispatch panel: a re-click supersedes the running slow-lane request
                p.fill("#ins-dp-models", "qwen3-0.6b, qwen3-30b-a3b")
                p.fill("#ins-dp-batches", "1, 8")
                n0 = p.evaluate("__accelStatic.log.length")
                p.click("#ins-dp-run")
                time.sleep(0.05)
                p.click("#ins-dp-run")
                st = wait_text("#ins-dp-status", "t.startsWith('完成') || t.startsWith('错误')")
                rows = len(p.query_selector_all("#ins-dp-tbl tbody tr"))
                sup = p.evaluate(f"__accelStatic.log.slice({n0}).filter(e => e.path === '/api/insight_dispatch' && (e.killed || e.dropped || e.aborted)).length")
                steps.append(("dispatch panel", rows, st[:30], "superseded", sup))
                if not st.startswith("完成") or rows != 4:
                    errs.append(f"dispatch panel: {st[:200]} rows={rows}")
                if sup < 1:
                    errs.append("dispatch panel: the first (superseded) request was neither killed nor dropped")
                # scenario edit after a run → stale tag (and the eval re-runs on the fast lane meanwhile)
                p.select_option("#model", "qwen3-1.7b")
                st = wait_text("#ins-dp-status", "t.includes('场景已改变')", 60000)
                steps.append(("dispatch panel stale tag", "场景已改变" in st))

                # 2c. speech insight with the op-dispatch column; the progress pill shows while it runs
                p.uncheck("#ins-sp-verify")
                p.click("#ins-sp-run")
                try:
                    p.wait_for_function("document.getElementById('busy').classList.contains('on')"
                                        " && document.getElementById('busy-pill').textContent.includes('语音瓶颈')", timeout=30000)
                    steps.append(("progress pill (speech)", True))
                except Exception:  # noqa: BLE001
                    errs.append("progress pill did not show for the speech insight")
                if p.is_disabled("#ins-dp-run") or p.is_disabled("#model"):
                    errs.append("inputs disabled while a request runs (progress must be non-blocking)")
                st = wait_text("#ins-sp-status", "t.startsWith('完成') || t.startsWith('错误')")
                rows = len(p.query_selector_all("#ins-sp-tbl tbody tr"))
                steps.append(("speech insight", rows, st[:30]))
                if not st.startswith("完成") or rows < 10 or "实时临界" not in (p.text_content("#ins-sp-tbl") or ""):
                    errs.append(f"speech insight: {st[:200]} rows={rows}")

                # 2d. speech streams vs SLC (one SLC size keeps it short)
                p.fill("#ins-sps-gib", "0")
                p.click("#ins-sps-run")
                st = wait_text("#ins-sps-status", "t.startsWith('完成') || t.startsWith('错误')")
                rows = len(p.query_selector_all("#ins-sps-tbl tbody tr"))
                steps.append(("speech streams", rows, st[:40]))
                if not st.startswith("完成") or rows < 8:
                    errs.append(f"speech streams: {st[:200]} rows={rows}")
                b.close()
        finally:
            srv.shutdown()
    for s in steps:
        print("STEP", *s)
    for e in errs:
        print("ERR", e)
    print(f"{len(errs)} error(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:2]))
