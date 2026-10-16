"""0.185 opt-in headless UI smoke (needs Playwright + Chrome; not part of run_tests).
Starts `accel-dse serve` on a free port, loads the page, drives key controls (model, phase, chip preset, memory
count/type, cards/TP, mapping, density preset, D2D), runs design-insight sizing / joint / upgrade path, and asserts:
no page errors / console errors / failed API calls, KPI cards populate after each change, sizing tables fill.
Usage: python3 scripts/ui_headless_smoke.py [--chrome /usr/bin/google-chrome]   exit 0 = clean."""
from __future__ import annotations

import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main(chrome="/usr/bin/google-chrome") -> int:
    from playwright.sync_api import sync_playwright
    port = _port()
    srv = subprocess.Popen([sys.executable, "-m", "accel_dse", "serve", "--host", "127.0.0.1", "--port", str(port)],
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    errs, steps = [], []
    try:
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.2)
        with sync_playwright() as pw:
            b = pw.chromium.launch(executable_path=chrome, args=["--headless=new", "--no-sandbox"])
            p = b.new_page()
            p.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
            p.on("console", lambda m: m.type == "error" and errs.append(f"console: {m.text}"))
            p.on("response", lambda r: "/api/" in r.url and r.status >= 500 and errs.append(f"HTTP {r.status} {r.url}"))
            p.goto(f"http://127.0.0.1:{port}/")
            p.wait_for_selector("#kpis .kpi", timeout=60000)

            def settle(label, timeout=60.0):
                """KPI cards present, nothing pending, status not an error."""
                t0 = time.time()
                while time.time() - t0 < timeout:
                    st = p.inner_text("#status") if p.is_visible("#status") else ""
                    n = len(p.query_selector_all("#kpis .kpi"))
                    if n and "计算中" not in st:
                        steps.append((label, n, round(time.time() - t0, 1)))
                        if "错误" in st:
                            errs.append(f"{label}: status {st[:200]}")
                        return
                    time.sleep(0.3)
                errs.append(f"{label}: KPI cards did not populate in {timeout}s")

            def act(label, fn, timeout=90000):
                """Run a control change; it must trigger a fresh API POST that answers < 500, then KPIs settle."""
                try:
                    with p.expect_response(lambda r: "/api/" in r.url and r.request.method == "POST",
                                           timeout=timeout) as ri:
                        fn()
                    if ri.value.status >= 400:
                        errs.append(f"{label}: HTTP {ri.value.status} {ri.value.text()[:200]}")
                except Exception as e:                       # noqa: BLE001
                    errs.append(f"{label}: no API request after the change ({type(e).__name__})")
                p.wait_for_load_state("networkidle")
                settle(label)

            def opts(sel):
                return [o.get_attribute("value") for o in p.query_selector_all(f"{sel} option")]

            settle("load")
            for m in ("gpt-oss-120b", "wan2.1-1.3b", "esm2-650m", "qwen3-8b"):
                if m in opts("#model"):
                    act(f"model={m}", lambda m=m: p.select_option("#model", m))
            for ph in ("prefill", "decode"):
                if p.is_visible(f"#phase button[data-v={ph}]"):
                    act(f"phase={ph}", lambda ph=ph: p.click(f"#phase button[data-v={ph}]"))
            chips = p.query_selector_all("#chip-preset button")
            for i in list(range(1, min(4, len(chips)))) + [0]:
                act(f"chip#{i}", lambda i=i: p.query_selector_all("#chip-preset button")[i].click())
            maps = p.query_selector_all("#mapping button")
            for i in list(range(1, min(3, len(maps)))) + [0]:
                act(f"mapping#{i}", lambda i=i: p.query_selector_all("#mapping button")[i].click())
            for sel in ("#m-count", "#m-type", "#b-preset"):
                if p.is_visible(sel):
                    cur = p.input_value(sel)
                    o = [v for v in opts(sel) if v and v != cur]
                    for v in (o[:1] + o[-1:] + [cur]):
                        if v == p.input_value(sel):
                            continue
                        act(f"{sel}={v}", lambda sel=sel, v=v: p.select_option(sel, v))
            if p.is_visible("#l-cards"):
                act("cards=2", lambda: p.fill("#l-cards", "2"))
                if p.is_visible("#l-tp"):
                    act("tp=2", lambda: p.fill("#l-tp", "2"))
            if p.is_visible("#d2d-enabled"):
                act("d2d on", lambda: p.check("#d2d-enabled"))
                act("d2d off", lambda: p.uncheck("#d2d-enabled"))
            if p.is_visible("#l-tp"):
                act("tp=1", lambda: p.fill("#l-tp", "1"))
            if p.is_visible("#l-cards"):
                act("cards=1", lambda: p.fill("#l-cards", "1"))
            # design insight: sizing (+ joint) and upgrade path
            p.click("button[data-tab=insight]")
            p.fill("#ins-models", "qwen3-8b")
            for joint in (False, True):
                p.set_checked("#ins-size-joint", joint)
                p.click("#ins-size-run")
                p.wait_for_function("(() => { const t = document.getElementById('ins-size-status').textContent;"
                                    " return t.startsWith('完成') || t.startsWith('错误'); })()", timeout=600000)
                st = p.inner_text("#ins-size-status")
                rows = len(p.query_selector_all("#ins-size-tbl tbody tr"))
                steps.append((f"sizing joint={joint}", rows, st[:40]))
                if not st.startswith("完成") or rows < 6:
                    errs.append(f"sizing joint={joint}: {st[:200]} rows={rows}")
                if joint and not p.query_selector_all("#ins-size-joint-tbl tbody tr"):
                    errs.append("joint table empty")
            p.fill("#ins-steps", "1")
            p.click("#ins-path-run")
            p.wait_for_function("(() => { const t = document.getElementById('ins-path-status').textContent;"
                                " return t.startsWith('完成') || t.startsWith('错误') || t.includes('无'); })()",
                                timeout=600000)
            steps.append(("path", p.inner_text("#ins-path-status")[:60]))
            if "错误" in p.inner_text("#ins-path-status"):
                errs.append("path: " + p.inner_text("#ins-path-status")[:200])
            b.close()
    finally:
        srv.terminate()
        try:
            srv.wait(10)
        except subprocess.TimeoutExpired:
            srv.kill()
    for s in steps:
        print("STEP", *s)
    for e in errs:
        print("ERROR", e)
    print(f"{len(errs)} error(s)")
    return 1 if errs else 0


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.exit(main(a[a.index("--chrome") + 1] if "--chrome" in a else "/usr/bin/google-chrome"))
