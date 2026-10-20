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
            p.click("#sec-open-all")      # 0.256: budget / energy knob groups start collapsed

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
                    # 0.186: wait for the eval answer itself — a memory change first POSTs /api/memory, and
                    # reading the cards on that answer reads the previous render (false "stale" alarm)
                    with p.expect_response(lambda r: "/api/eval" in r.url and r.request.method == "POST",
                                           timeout=timeout) as ri:
                        fn()
                    if ri.value.status >= 400:
                        errs.append(f"{label}: HTTP {ri.value.status} {ri.value.text()[:200]}")
                except Exception as e:                       # noqa: BLE001
                    errs.append(f"{label}: no API request after the change ({type(e).__name__})")
                p.wait_for_load_state("networkidle")
                time.sleep(0.2)                              # render after the response resolves
                settle(label)

            def opts(sel):
                return [o.get_attribute("value") for o in p.query_selector_all(f"{sel} option")]

            settle("load")
            # 0.273: + the speech-out kinds (Omni: split / SLC / audio-in KPIs; CosyVoice: split / per-chunk)
            for m in ("gpt-oss-120b", "wan2.1-1.3b", "esm2-650m", "qwen2.5-omni-7b", "cosyvoice2-0.5b", "kimi-audio-7b-instruct", "step-audio-2-mini", "qwen3-8b"):
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
            # 0.204: per-step walls bar + dispatch control; rapid toggles must end on the last choice (no race)
            if p.query_selector("#walls-bar") and p.query_selector("#c-dispatch"):
                def wcheck(label, ok, extra=""):
                    steps.append((label, bool(ok)))
                    if not ok:
                        errs.append(f"{label}: {extra[:300]}")
                note = lambda: p.inner_text("#walls-note")
                wcheck("walls bar has a wall", p.query_selector("#walls-bar i.wall") is not None, note())
                wcheck("walls note bilingual", "墙 wall" in note() and "fixed overhead" in note(), note())
                act("dispatch=host", lambda: p.select_option("#c-dispatch", "host"))
                wcheck("dispatch segment", p.query_selector("#walls-bar i[data-k=dispatch]") is not None, note())
                wcheck("note says host", "dispatch：host" in note(), note())
                for v in ("trace", "static", "", "host", "trace"):     # fire without waiting: stale answers must drop
                    p.select_option("#c-dispatch", v)
                act("dispatch=ideal (after burst)", lambda: p.select_option("#c-dispatch", ""))
                p.wait_for_load_state("networkidle")
                time.sleep(0.5)
                wcheck("burst ends on ideal", "dispatch：ideal" in note()
                       and p.query_selector("#walls-bar i[data-k=dispatch]") is None, note())
                wcheck("approx note iff not stage", ("近似" in note()) == ("exec_overlap = stage" not in note()), note())
            # 0.205: presets carry their memory pairing; machine-balance KPI + warning
            names = [b_.inner_text() for b_ in p.query_selector_all("#chip-preset button")]
            if "1P" in names and "1P-lp4x64" in names:
                def bcheck(label, ok, extra=""):
                    steps.append((label, bool(ok)))
                    if not ok:
                        errs.append(f"{label}: {extra[:300]}")
                act("preset 1P", lambda: p.query_selector_all("#chip-preset button")[names.index("1P")].click())
                mf = p.inner_text("#mem-facts")
                k = p.inner_text("#kpis")
                bcheck("1P pairs with HBM (≈4710 GB/s)", "4,710" in mf or "4710" in mf, mf)
                bcheck("balance KPI", "machine balance" in k, k[-300:])
                # 0.206: critical batch in the KPI; balanced-class 1P at the default ctx gets a hint, not a warning
                bcheck("critical batch shown", "critical batch" in k, k[-300:])
                bcheck("no balance warning on 1P", p.query_selector("#warns .balance-warn") is None, p.inner_text("#warns")[:300])
                # 0.208: balance what-if table from the sensitivity response (one lever each)
                try:
                    p.wait_for_selector("#bal-whatif-tbl tbody tr", timeout=60000)
                    wt = p.inner_text("#bal-whatif-tbl")
                except Exception as e_:   # noqa: BLE001
                    wt = f"missing: {e_}"
                bcheck("balance what-if table", "KV fp8" in wt and "SRAM ×2" in wt and "DRAM 带宽 ×2" in wt, wt)
                bcheck("what-if verdicts", "有效" in wt and "无效" in wt, wt)
                act("preset 1P-lp4x64", lambda: p.query_selector_all("#chip-preset button")[names.index("1P-lp4x64")].click())
                mf2 = p.inner_text("#mem-facts")
                bcheck("legacy pairs with LPDDR 273 GB/s", "273" in mf2, mf2)
                bcheck("balance warning shown", p.query_selector("#warns .balance-warn") is not None, p.inner_text("#warns")[:300])
                act("preset 100T", lambda: p.query_selector_all("#chip-preset button")[names.index("100T")].click())
            # 0.186: directional checks — a control that must move a metric has to move the rendered value
            import re

            def kpi(label):
                for k in p.query_selector_all("#kpis .kpi"):
                    if k.query_selector(".l").inner_text().strip().startswith(label):
                        return k.inner_text()
                return ""

            def num(txt, pat):
                m = re.search(pat, txt)
                return float(m.group(1)) if m else None

            def expect(label, before, after, cmp):
                steps.append((f"dir {label}", before, after))
                if before is None or after is None or not cmp(before, after):
                    errs.append(f"stale/wrong direction: {label}: {before} -> {after}")

            def mem_bw():
                return num(p.inner_text("#mem-facts"), r"带宽\s+([\d.]+)\s*GB/s")

            def tpot():
                return num(kpi("TPOT"), r"TPOT\s*([\d.]+)\s*ms")

            def total_tok():
                return num(kpi("吞吐"), r"共\s*([\d.]+)\s*tok/s")

            def dram_cap():
                return num(kpi("DRAM 需求"), r"/\s*([\d.]+)\s*GiB")

            if p.is_visible("#best-batch") and p.is_checked("#best-batch"):
                act("fixed batch", lambda: p.uncheck("#best-batch"))
            act("batch=1", lambda: p.fill("#s-batch", "1"))
            t1 = tpot()
            act("batch=16", lambda: p.fill("#s-batch", "16"))
            expect("batch 1→16 raises TPOT", t1, tpot(), lambda a, b: b > a)
            tot = total_tok()
            act("cards=2 (dir)", lambda: p.fill("#l-cards", "2"))
            act("tp=2 (dir)", lambda: p.fill("#l-tp", "2"))
            t2 = tpot()
            act("tp=1 (dir)", lambda: p.fill("#l-tp", "1"))
            act("cards=1 (dir)", lambda: p.fill("#l-cards", "1"))
            t16 = tpot()
            expect("TP2 (2 cards) lowers TPOT vs TP1 at batch 16", t16, t2, lambda a, b: b < a)
            bw0, cap0 = mem_bw(), dram_cap()
            cur = p.input_value("#m-count")
            fewer = [v for v in opts("#m-count") if v and float(v) < float(cur)]
            if fewer:
                act(f"m-count {cur}→{fewer[-1]} (dir)", lambda: p.select_option("#m-count", fewer[-1]))
                expect("fewer DRAM packages lowers bandwidth", bw0, mem_bw(), lambda a, b: b < a)
                expect("fewer DRAM packages lowers capacity", cap0, dram_cap(), lambda a, b: b < a)
                act("m-count restore", lambda: p.select_option("#m-count", cur))
                expect("restore DRAM count restores bandwidth", bw0, mem_bw(), lambda a, b: abs(a - b) < 1e-9)
            k0 = kpi("TPOT")
            act("phase=prefill (dir)", lambda: p.click("#phase button[data-v=prefill]"))
            if "TTFT" not in p.inner_text("#kpis"):
                errs.append("prefill phase: TTFT card not rendered")
            act("phase=decode (dir)", lambda: p.click("#phase button[data-v=decode]"))
            expect("decode card restored", 1.0, 1.0 if kpi("TPOT") else None, lambda a, b: True)
            act("best batch back", lambda: p.check("#best-batch"))

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
            # 0.187: energy-constrained sizing with an energy preset
            if p.query_selector("#e-preset"):
                p.click("button[data-tab=eval]") if p.query_selector("button[data-tab=eval]") else None
                o = [v for v in opts("#e-preset") if v]
                if "tpuv4i_7nm" in o:
                    p.select_option("#e-preset", "tpuv4i_7nm")
                    time.sleep(1)
                p.click("button[data-tab=insight]")
                p.set_checked("#ins-size-energy", True)
                p.click("#ins-size-run")
                p.wait_for_function("(() => { const t = document.getElementById('ins-size-status').textContent;"
                                    " return t.startsWith('完成') || t.startsWith('错误'); })()", timeout=600000)
                st = p.inner_text("#ins-size-status")
                cells = p.inner_text("#ins-size-tbl")
                steps.append(("sizing energy_tie", st[:40], "能耗 ×" in cells))
                if not st.startswith("完成") or "能耗 ×" not in cells:
                    errs.append(f"energy sizing: {st[:200]} / energy column missing")
                p.set_checked("#ins-size-energy", False)
            # 0.190: energy controls — PHY fields, idle_frac_tdp, tdp_W, static warning, power cap (binding case)
            if p.query_selector("#e-power_cap"):
                if p.query_selector("button[data-tab=eval]"):
                    p.click("button[data-tab=eval]")
                if p.query_selector("#e-preset") and p.input_value("#e-preset") != "tpuv4i_7nm":
                    act("energy preset", lambda: p.select_option("#e-preset", "tpuv4i_7nm"))

                def etbl():
                    return p.inner_text("#energy-tbl") if p.query_selector("#energy-tbl") else ""

                def check(label, ok, extra=""):
                    steps.append((label, bool(ok)))
                    if not ok:
                        errs.append(f"{label}: {extra[:300]}")
                for k in ("pJ_bit_phy_hbm", "pJ_bit_phy_lpddr", "idle_frac_tdp", "tdp_W"):
                    check(f"input e-{k}", p.query_selector(f"#e-{k}") is not None)
                check("preset fills PHY", p.input_value("#e-pJ_bit_phy_lpddr") not in ("",),
                      p.input_value("#e-pJ_bit_phy_lpddr"))
                act("tdp_W 175", lambda: p.fill("#e-tdp_W", "175") or p.press("#e-tdp_W", "Tab"))
                t = etbl()
                check("dram_phy row", "DRAM PHY" in t, t)
                check("static source names H100", "H100" in t, t[-400:])
                sh = num(t, r"静态能耗占[^0-9]*([0-9.]+)%")
                check("static share line", sh is not None, t[-400:])
                if sh is not None and abs(sh - 50) > 0.1:   # warning iff share > 50 % (skip display-rounding tie)
                    check("warning iff > 50%", ("（>" in t) == (sh > 50), t[-400:])
                j0 = num(t, r"合计\s+([0-9.e+-]+)")
                act("idle_frac 0.001", lambda: p.fill("#e-idle_frac_tdp", "0.001") or p.press("#e-idle_frac_tdp", "Tab"))
                t = etbl()
                check("no warning at low static", "静态能耗占" in t and "（>" not in t and "用户填写" in t, t)
                j1 = num(t, r"合计\s+([0-9.e+-]+)")
                check("J falls with idle_frac", j0 is not None and j1 is not None and j1 < j0, f"{j0} -> {j1}")
                act("power_cap on", lambda: p.set_checked("#e-power_cap", True))
                t = etbl()
                check("cap not binding @175W", "未超 TDP" in t, t)
                act("tdp_W 2", lambda: p.fill("#e-tdp_W", "2") or p.press("#e-tdp_W", "Tab"))
                t = etbl()
                f = num(t, r"吞吐 ×([0-9.]+)")
                check("cap binds @2W, throughput falls", f is not None and f < 1, t)
                act("power_cap off", lambda: p.set_checked("#e-power_cap", False))
                check("cap line gone", "功耗封顶" not in etbl(), etbl())
                p.click("button[data-tab=insight]")
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
