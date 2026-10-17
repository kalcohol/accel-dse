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
    # 0.169 host L2 prefix cache in the PD panel
    p.check("#pd-enabled"); time.sleep(1)
    p.fill("#pd-prefix_len", "2048"); time.sleep(1)
    p.fill("#pd-prefix_host_GB", "200"); time.sleep(1)
    p.select_option("#pd-prefix_host_policy", "write_back")
    for _ in range(240):
        if "主机 L2" in (p.inner_text("#pdpc-basis") if p.is_visible("#pdpc-basis") else ""): break
        time.sleep(0.5)
    t = p.inner_text("#pdpc-basis")
    assert "主机 L2 200 GB" in t and "write_back" in t, t[:400]
    print("HOST L2 ok")
    # 0.231 chunked-prefill engine profile: default vllm (budget − decodes), legacy_512 selectable
    for _ in range(240):
        if "vllm ·" in (p.inner_text("#pdq-tbl") if p.is_visible("#pdq-tbl") else ""): break
        time.sleep(0.5)
    t = p.inner_text("#pdq-tbl")
    assert "分块 prefill（vllm ·" in t and "预算" in t, t[:400]
    assert p.is_disabled("#pd-chunk_tokens") and not p.is_disabled("#pd-chunk_budget")
    p.select_option("#pd-chunk_profile", "legacy_512")
    for _ in range(240):
        if "legacy_512 · 512 tok" in p.inner_text("#pdq-tbl"): break
        time.sleep(0.5)
    assert "legacy_512 · 512 tok" in p.inner_text("#pdq-tbl"), p.inner_text("#pdq-tbl")[:400]
    p.select_option("#pd-chunk_profile", "vllm"); time.sleep(2)
    print("CHUNK PROFILE ok")
    # 0.223 independent draft model picker + spec rollback select
    p.uncheck("#pd-enabled"); time.sleep(2)
    p.select_option("#model", "qwen3-32b"); time.sleep(3)
    p.click("#phase button[data-v=decode]"); time.sleep(2)
    p.fill("#s-spec_k", "4"); time.sleep(3)
    k0 = kpis(p)
    p.select_option("#s-draft_model", "qwen3-0.6b")
    for _ in range(120):
        if kpis(p) != k0: break
        time.sleep(0.5)
    assert kpis(p) != k0, "draft model did not change KPIs"
    assert p.inner_text("#s-draft-warn").strip() == ""
    p.select_option("#s-draft_model", "llama-3.2-1b"); time.sleep(3)
    assert "词表" in p.inner_text("#s-draft-warn"), p.inner_text("#s-draft-warn")
    p.select_option("#s-draft_model", ""); time.sleep(2)
    p.select_option("#model", "qwen3.8-27b"); time.sleep(3)
    p.fill("#s-spec_k", "3"); time.sleep(3)
    k1 = kpis(p)
    p.select_option("#s-spec_rollback", "replay")
    for _ in range(120):
        if kpis(p) != k1: break
        time.sleep(0.5)
    assert kpis(p) != k1, "spec_rollback did not change KPIs"
    print("DRAFT/ROLLBACK ok")
    # 0.225 tree fields (topk / draft tokens / acceptance model / track interval) + chain-vs-tree what-if row
    p.select_option("#s-spec_rollback", "snapshot"); time.sleep(3)
    k2 = kpis(p)
    p.fill("#s-spec_topk", "4")
    for _ in range(120):
        if kpis(p) != k2: break
        time.sleep(0.5)
    assert kpis(p) != k2, "spec_topk did not change KPIs"
    assert "EAGLE-1" in p.inner_text("#s-tree-warn"), p.inner_text("#s-tree-warn")
    p.fill("#s-spec_draft_tokens", "99"); time.sleep(2)
    assert "bad" in (p.get_attribute("#s-spec_draft_tokens", "class") or "")
    p.fill("#s-spec_draft_tokens", "1"); time.sleep(2)
    assert "bad" in (p.get_attribute("#s-spec_draft_tokens", "class") or "")
    p.fill("#s-spec_draft_tokens", "16"); time.sleep(3)
    assert "bad" not in (p.get_attribute("#s-spec_draft_tokens", "class") or "")
    k3 = kpis(p)
    # 0.226: default chain headline + band KPI; switching to indep changes the headline
    for _ in range(60):
        if p.is_visible("#kpi-tree-band"): break
        time.sleep(0.5)
    assert "chain" in p.inner_text("#kpi-tree-band") and "rank" in p.inner_text("#kpi-tree-band"), p.inner_text("#kpi-tree-band")
    p.select_option("#s-spec_tree_accept_model", "indep")
    for _ in range(120):
        if kpis(p) != k3: break
        time.sleep(0.5)
    assert kpis(p) != k3, "spec_tree_accept_model did not change KPIs"
    p.select_option("#s-spec_tree_accept_model", "chain"); time.sleep(3)
    p.fill("#s-mamba_track_interval", "256"); time.sleep(3)
    assert "bad" not in (p.get_attribute("#s-mamba_track_interval", "class") or "")
    for _ in range(400):
        if p.is_visible("#spec-tree-cmp") and "主值 chain" in p.inner_text("#spec-tree-cmp"): break
        time.sleep(0.5)
    t = p.inner_text("#spec-tree-cmp")
    assert "B1" in t and "B32" in t and "crossover" in t and "主值 chain" in t, t[:400]
    print("TREE ok", t[:200].replace("\n", " "))
    # 0.227 power cap: clock / voltage line in the energy panel + per-lever cap binding in the what-if
    p.fill("#s-spec_topk", "1"); p.fill("#s-spec_draft_tokens", "0"); p.fill("#s-mamba_track_interval", "0")
    p.select_option("#model", "qwen3-8b"); time.sleep(3)
    p.fill("#s-spec_k", "0"); time.sleep(2)
    p.select_option("#e-preset", "tpuv4i_7nm"); time.sleep(2)
    p.fill("#e-tdp_W", "8"); time.sleep(1)
    p.check("#e-power_cap")
    for _ in range(240):
        if "仅降频" in p.inner_text("#energy-tbl"): break
        time.sleep(0.5)
    et = p.inner_text("#energy-tbl")
    assert "仅降频" in et and "V–f" in et and "时钟（MAC 槽·周期" in et, et[-600:]      # 0.228: per-cycle row
    for _ in range(600):
        if p.is_visible("#whatif-cap"): break
        time.sleep(0.5)
    wc = p.inner_text("#whatif-cap")
    assert "功耗封顶 8" in wc and "封顶生效" in wc, wc[:300]
    print("POWER CAP ok", wc[:160].replace("\n", " "))
    p.uncheck("#e-power_cap"); time.sleep(2)
    # 0.229: clock-gating lever (chip field) shows in the energy row
    p.fill("#c-clk_ungated_share", "0.25"); p.press("#c-clk_ungated_share", "Tab")
    for _ in range(240):
        if "未门控 0.25" in p.inner_text("#energy-tbl"): break
        time.sleep(0.5)
    et = p.inner_text("#energy-tbl")
    assert "未门控 0.25" in et and "利用率" in et, et[-600:]
    print("CLOCK GATING ok")
    p.screenshot(path="/workspace/smoke/ui166.png", full_page=False)
    b.close()
print("ERRORS:", errs)
