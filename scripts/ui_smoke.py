import sys, time
from playwright.sync_api import sync_playwright
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8799/"
errs = []
def kpis(p): return p.inner_text("#kpis")
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/usr/bin/google-chrome", args=["--headless=new","--no-sandbox"]); p = b.new_page()
    p.on("console", lambda m: m.type == "error" and errs.append("console: " + m.text))
    p.on("response", lambda r: "/api/" in r.url and r.request.method=="POST" and print("RESP", r.status, r.url[-15:], "glm" in (r.request.post_data or ""), r.text()[:300] if "eval" in r.url and "glm" in (r.request.post_data or "") else ""))
    p.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))
    p.goto(URL); p.wait_for_selector("#kpis .kpi", timeout=30000)
    p.select_option("#model", "qwen3.8-27b"); time.sleep(2)
    p.click("#phase button[data-v=prefill]"); time.sleep(1.5)
    assert p.is_visible("#s-videos"), "video input hidden on qwen3.8-27b"
    p.fill("#s-videos", "1"); time.sleep(3)
    k = kpis(p); print("VIDEO KPI:", [l for l in k.splitlines() if "视频" in l or "token/段" in l][:3])
    assert "视频编码" in k and "11520 token/段" in k, k
    p.fill("#s-video_frames", "64"); time.sleep(3)
    assert "32 时间片" in kpis(p)
    assert "0 张/请求" not in kpis(p) and "视觉（图像 + 视频） token" in kpis(p), kpis(p)
    p.fill("#s-images", "1"); time.sleep(3)
    assert "1 张/请求" in kpis(p) and "视频编码" in kpis(p)
    p.fill("#s-images", "0"); time.sleep(2)
    p.fill("#h-host_step_us", "500"); time.sleep(3)
    assert "主机开销" in kpis(p), kpis(p)
    p.select_option("#h-host_overlap", "async"); time.sleep(3)
    assert "async" in kpis(p)
    p.fill("#h-host_steps", "4"); time.sleep(2)
    t0=time.time(); p.select_option("#model", "glm-5.3-flash"); time.sleep(3)
    print("GLM KPI head:", kpis(p).splitlines()[:2])
    for _ in range(150):
        if not p.is_visible("#kpis") or "视频编码" not in kpis(p): break
        time.sleep(0.5)
    print("glm wait", round(time.time()-t0,1)); print("vis", p.is_visible("#s-videos"), "视频编码" in kpis(p), errs); assert not p.is_visible("#s-videos") and (not p.is_visible("#kpis") or "视频编码" not in kpis(p))
    p.select_option("#model", "kimi-k2.5"); time.sleep(2)
    assert p.is_visible("#s-videos")
    p.click("#phase button[data-v=prefill]") if p.is_visible("#phase") else None
    p.fill("#s-videos", "1"); p.fill("#s-video_frames", "32")
    for _ in range(150):
        if p.is_visible("#kpis") and "8600 token/段" in kpis(p): break
        time.sleep(0.5)
    print("KIMI fits/kpi:", p.is_visible("#kpis"), "8600 token/段" in kpis(p))
    p.select_option("#model", "wan2.1-14b"); time.sleep(2)      # 0.167 step cache
    p.fill("#w-step_skip", "20")
    for _ in range(150):
        if p.is_visible("#kpis") and "跳步缓存" in kpis(p): break
        time.sleep(0.5)
    assert "20 / 50 步跳过" in kpis(p), kpis(p)
    p.select_option("#w-step_cache", "teacache_ret"); time.sleep(4)
    assert "teacache_ret" in kpis(p)
    p.fill("#w-step_skip", "49"); time.sleep(4)          # over the cap (50 − 5): input flagged, nothing sent
    assert "bad" in (p.get_attribute("#w-step_skip", "class") or "") and "20 / 50 步跳过" in kpis(p)
    p.fill("#w-step_skip", "0"); time.sleep(4)
    print("STEP CACHE ok")
    p.select_option("#model", "qwen3-8b"); time.sleep(3)
    assert not p.is_visible("#s-videos")
    p.screenshot(path="/workspace/smoke/ui166.png", full_page=False)
    b.close()
print("ERRORS:", errs)
