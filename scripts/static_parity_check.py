"""Opt-in (needs Playwright + Chrome; not part of run_tests): drive the static (Pyodide) build in headless Chromium,
time cold load / per-endpoint evaluation, then replay every captured /api call on a live `accel_dse serve` and compare
the responses byte for byte and leaf by leaf.
Usage: python3 scripts/static_parity_check.py STATIC_URL SERVER_URL OUT_DIR
       (STATIC_URL = `python3 -m http.server -d dist-static`, SERVER_URL = `python3 -m accel_dse serve`)"""
import json, sys, time, urllib.request
from playwright.sync_api import sync_playwright

STATIC = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18802/"
SERVER = (sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:18801").rstrip("/")
OUT = sys.argv[3] if len(sys.argv) > 3 else "out/static-check"
import os; os.makedirs(OUT, exist_ok=True)
res = {"phases": [], "net": {}, "direct": [], "parity": {}}
net = {}


def idle(p, quiet=2.0, timeout=600):
    t0 = time.monotonic(); last = None; since = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = p.evaluate("[__accelStatic.busy(), __accelStatic.log.length, (document.getElementById('status')||{}).textContent]")
        if st[0] == 0 and st[2] != '计算中…':
            if last != st[1]:
                last, since = st[1], time.monotonic()
            elif time.monotonic() - since >= quiet:
                return time.monotonic() - t0 - quiet
        else:
            last, since = None, time.monotonic()
        time.sleep(0.2)
    raise TimeoutError("not idle")


def phase(p, name, action, shot=None, timeout=600):
    n0 = p.evaluate("__accelStatic.log.length"); t0 = time.monotonic()
    action()
    wall = idle(p, timeout=timeout)
    log = p.evaluate(f"__accelStatic.log.slice({n0}).map(e => ({{path: e.path, ms: e.ms, status: e.status, dropped: e.dropped, aborted: e.aborted}}))")
    if shot:
        p.screenshot(path=f"{OUT}/{shot}.png", full_page=False)
    d = {"phase": name, "wall_s": round(wall, 2), "calls": [{k: (round(v) if k == "ms" and v is not None else v) for k, v in e.items() if v is not None} for e in log]}
    res["phases"].append(d); print(json.dumps(d, ensure_ascii=False), flush=True)


DIRECT = [
    ("/api/eval", {"scenario": {"model": "qwen3-8b"}}),
    ("/api/eval", {"scenario": {"model": "beam-1.0-501b-a23b", "layout": {"tp": 16, "etp": 16}}}),
    ("/api/eval", {"scenario": {"model": "deepseek-v4-pro", "layout": {"tp": 16, "etp": 16}}}),
    ("/api/eval", {"scenario": {"model": "flux-3-image-klein"}}),
    ("/api/eval", {"scenario": {"model": "protenix"}}),
    ("/api/eval", {"scenario": {"model": "alphafold2"}}),
    ("/api/eval", {"scenario": {"model": "esm2-3b"}}),
    ("/api/sweep", {"path": "serving.batch", "values": [1, 2, 4, 8, 16, 32, 64]}),
    ("/api/sweep", {"path": "serving.ctx", "values": [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072],
                    "scenario": {"model": "beam-1.0-501b-a23b", "layout": {"tp": 16, "etp": 16}}}),
    ("/api/pareto", {"scenario": {"model": "qwen3-8b"}}),
    ("/api/fit", {"scenario": {"model": "beam-1.0-501b-a23b"}}),
    ("/api/fit", {"scenario": {"model": "deepseek-v4-pro"}}),
]

with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--headless=new", "--no-sandbox"])
    ctx = b.new_context(viewport={"width": 1600, "height": 1000})
    ctx.add_init_script("window.__accelCapture = true")
    p = ctx.new_page()
    errs = []
    p.on("pageerror", lambda e: errs.append(str(e)))
    p.on("console", lambda m: m.type == "error" and errs.append(m.text))
    cdp = ctx.new_cdp_session(p); cdp.send("Network.enable")
    urls = {}
    cdp.on("Network.responseReceived", lambda e: urls.__setitem__(e["requestId"], e["response"]["url"]))
    cdp.on("Network.loadingFinished", lambda e: net.__setitem__(urls.get(e["requestId"], e["requestId"]), e["encodedDataLength"]))
    # page-level fetches from a dedicated worker show up on the page target too; also listen to playwright for safety
    pw_net = {}
    def on_resp(r):
        try:
            s = r.request.sizes(); pw_net[r.url] = s["responseBodySize"] + s["responseHeadersSize"]
        except Exception:
            pass
    p.on("requestfinished", lambda rq: on_resp(rq.response()) if rq.response() else None)

    t0 = time.monotonic()
    p.goto(STATIC)
    p.wait_for_function("window.__accelStatic && __accelStatic.loadMs !== null", timeout=180000)
    t_ready = time.monotonic() - t0
    p.wait_for_selector("#kpis .kpi", timeout=180000)
    t_kpi = time.monotonic() - t0
    idle(p)
    res["cold"] = {"ready_s": round(t_ready, 2), "first_kpi_s": round(t_kpi, 2),
                   "detail": p.evaluate("__accelStatic.detail"), "shim_loadMs": p.evaluate("__accelStatic.loadMs")}
    print("COLD", res["cold"], flush=True)
    p.screenshot(path=f"{OUT}/01_cold_qwen3-8b_decode.png")

    phase(p, "model beam-1.0-501b-a23b (UI)", lambda: p.select_option("#model", "beam-1.0-501b-a23b"), "02_beam_501b", 900)
    phase(p, "model deepseek-v4-pro (UI)", lambda: p.select_option("#model", "deepseek-v4-pro"), "03_deepseek_v4_pro", 900)
    phase(p, "model flux-3-image-klein (UI)", lambda: p.select_option("#model", "flux-3-image-klein"), "04_flux3_klein", 600)
    phase(p, "model protenix (UI)", lambda: p.select_option("#model", "protenix"), "05_protenix", 600)
    phase(p, "model qwen3-8b (UI)", lambda: p.select_option("#model", "qwen3-8b"), None, 600)
    def tab(t, btn):
        def a():
            p.click(f"#tabs button[data-tab={t}]"); time.sleep(0.3); p.click(btn)
        return a
    phase(p, "compare (8 cards)", tab("compare", "#cmp-run"), "06_compare", 900)
    phase(p, "layouts", tab("layouts", "#lay-run"), "07_layouts", 900)
    phase(p, "stability", tab("layouts", "#lay-stab"), "08_stability", 900)
    phase(p, "sweep (UI default)", tab("sweep", "#sw-run"), "09_sweep", 900)
    phase(p, "pareto", tab("sweep", "#pf-run"), "10_pareto", 900)
    phase(p, "insight", tab("insight", "#ins-run"), "11_insight", 900)
    phase(p, "insight_path", tab("insight", "#ins-path-run"), "12_insight_path", 900)
    phase(p, "insight_sizing", tab("insight", "#ins-size-run"), "13_insight_sizing", 900)

    for path, body in DIRECT:
        r = p.evaluate("""async ([path, body]) => { const t = performance.now();
            const r = await fetch(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
            await r.text(); const e = __accelStatic.log[__accelStatic.log.length - 1];
            return {status: r.status, ms: e.ms, wall: performance.now() - t}; }""", [path, body])
        d = {"path": path, "body": body, **{k: round(v) if isinstance(v, float) else v for k, v in r.items()}}
        res["direct"].append(d); print("DIRECT", json.dumps(d, ensure_ascii=False), flush=True)

    log = p.evaluate("__accelStatic.log.filter(e => !e.dropped)")
    res["errors"] = errs
    b.close()

# ---- parity: replay every captured call on the Python server, compare the bytes and the parsed JSON
def leaves(x, pre=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{pre}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{pre}[{i}]")
    else:
        yield pre, x

tot = {"calls": 0, "byte_identical": 0, "json_equal": 0, "leaves": 0, "leaf_mismatch": 0, "mismatches": []}
srv_times = []
for e in log:
    if "text" not in e:
        continue
    req = urllib.request.Request(SERVER + e["path"], data=e["body"].encode() if e["method"] == "POST" else None,
                                 method=e["method"], headers={"Content-Type": "application/json"})
    t = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            st, txt = r.status, r.read().decode()
    except urllib.error.HTTPError as he:
        st, txt = he.code, he.read().decode()
    srv_times.append((e["path"], round((time.monotonic() - t) * 1e3), round(e["ms"])))
    tot["calls"] += 1
    tot["byte_identical"] += (txt == e["text"] and st == e["status"])
    a, bb = json.loads(e["text"]), json.loads(txt)
    tot["json_equal"] += (a == bb)
    la, lb = dict(leaves(a)), dict(leaves(bb))
    tot["leaves"] += len(lb)
    bad = [k for k in set(la) | set(lb) if la.get(k, "<missing>") != lb.get(k, "<missing>") or type(la.get(k)) is not type(lb.get(k))]
    tot["leaf_mismatch"] += len(bad)
    if bad:
        tot["mismatches"].append({"path": e["path"], "n": len(bad), "examples": [(k, la.get(k), lb.get(k)) for k in sorted(bad)[:5]]})
res["parity"] = tot
res["server_vs_browser_ms"] = srv_times
res["net_cdp"] = net; res["net_pw"] = pw_net
json.dump(res, open(f"{OUT}/results.json", "w"), ensure_ascii=False, indent=1)
print("PARITY", json.dumps({k: v for k, v in tot.items() if k != "mismatches"}), tot["mismatches"][:5])
print("NET cdp total", sum(net.values()), "pw total", sum(pw_net.values()))
print("ERRORS", errs[:10])
