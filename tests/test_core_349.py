"""0.349: adversarial sidecar_bytes; report pkg thermal margin; no visual UI."""
from __future__ import annotations

import importlib.util
import os
from dataclasses import replace
from pathlib import Path

from accel_dse.core.catalog import get_model
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.evaluate import evaluate
from accel_dse.core.parallel import Layout
from accel_dse.core.report import design_report, to_markdown
from accel_dse.core.thermal import COOLING

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()
LP = "lpddr5x_4x64_8533_16g"


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sidecar_bytes_adversarial():
    for mid in ("moss-tts", "moshiko-7b", "moshika-7b"):
        m = get_model(mid)
        assert m.params() == m.release_params and m.sidecar_bytes > 0
        s = Scenario(model=mid, chip=CHIPS["1P"], mem_id=LP, layout=Layout(tp=1, pp=1, dp=1),
                     serving=Serving(batch=1, prompt=128, out_len=64, phase="prefill", ctx=128))
        r = evaluate(s)
        r0 = evaluate(s, model=replace(m, sidecar_bytes=0.0, sidecar_params=0))
        assert abs(r.stages[0].mem.dram_need - r0.stages[0].mem.dram_need - m.sidecar_bytes) < 1.0
    z = get_model("moshiko-7b")
    assert z.sidecar_bytes == z.moshi.stored_bytes == z.moshi.depth_params * 2 + z.moshi.mimi_params * 4
    moss = get_model("moss-tts")
    assert moss.sidecar_bytes == moss.moss_tts.audio_head_params * 2.0
    # PP head-only
    m0 = replace(moss, sidecar_bytes=0.0, sidecar_params=0)
    s = Scenario(model="moss-tts", chip=CHIPS["1P"], mem_id=LP, layout=Layout(tp=1, pp=2, dp=1),
                 serving=Serving(batch=1, prompt=128, out_len=64, phase="prefill"))
    r, r0 = evaluate(s, model=moss), evaluate(s, model=m0)
    assert abs(r.stages[0].mem.dram_need - r0.stages[0].mem.dram_need) < 1.0
    assert abs(r.stages[1].mem.dram_need - r0.stages[1].mem.dram_need - moss.sidecar_bytes) < 1.0
    assert get_model("qwen3-omni-30b-a3b").sidecar_bytes == 0


def test_report_pkg_margin_and_binding():
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None, cooling="air_oam", power_scope="card")
    lim_f = COOLING["air_oam"]["flux_W_mm2"]
    lim_p = COOLING["air_oam"]["pkg_W"]
    assert lim_f == 0.86 and lim_p == 1000.0  # not retuned
    rep = design_report(o, {}, 0)
    th = next(s for s in rep["sections"] if s["id"] == "thermal")
    it = {i["key"]: i["value"] for i in th["items"]}
    assert abs(it["pkg_margin_W"] - (lim_p - it["pkg_W"])) < 1e-9
    assert abs(it["pkg_headroom"] - (lim_p / it["pkg_W"] - 1)) < 1e-9
    assert it["thermal_binding"] in ("热流", "封装功耗")
    # 0.331 flux pins still hold
    assert abs(it["flux_margin_W_mm2"] - (lim_f - it["flux_W_mm2"])) < 1e-12
    ft = th["front_thermal"]
    assert ft["limit_pkg_W"] == lim_p and "n_pkg_within_tie" in ft
    for p in ft["points"]:
        assert "pkg_W" in p and "pkg_margin" in p and "pkg_within_tie" in p
        if p.get("pkg_W") is not None:
            assert abs(p["pkg_margin_W"] - (lim_p - p["pkg_W"])) < 1e-6
    md = to_markdown(rep)
    assert "封装余量" in md and "前沿热敏感性" in md


def test_changelog_349():
    assert "## [0.349.0]" in CL
    sec = CL.split("## [0.349.0]")[1].split("## [")[0]
    assert "sidecar_bytes" in sec and ("pkg" in sec.lower() or "封装" in sec or "thermal" in sec.lower())
    assert "NPU" in sec or "HBM4" in sec
