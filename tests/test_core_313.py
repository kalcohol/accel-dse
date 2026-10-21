"""0.313: adversarial review of 0.310–0.312 (scale-up lane rate by generation, 4-sided memory as an opt-in strict rule,
interposer bounding box, sourced stack size) and buildability from power density / package power (thermal.py)."""
import importlib.util
import os
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import SU_LANE_GENERATIONS, io_need, shoreline_report
from accel_dse.core.energy_dse import energy_design_search
from accel_dse.core.hardware import CHIPS, Link
from accel_dse.core.scenario import Layout, Scenario, Serving
from accel_dse.core.thermal import COOLING, KNOWN_PARTS, dram_dev_W, thermal_report
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H8, H4 = "hbm3e_8s_8h24g_9200", "hbm3e_4s_8h24g_9200"
PF = Serving(phase="prefill", batch=1, prompt=4096, ctx=8192)


def _s(chip="1P", mem=H8, tp=1, link=400.0, **ck):
    return Scenario(model="qwen3-8b", chip=replace(CHIPS[chip], **ck), mem_id=mem, layout=Layout(tp=tp),
                    link=Link(GBps=link))


def _ex():
    spec = importlib.util.spec_from_file_location("chiplet_example", os.path.join(ROOT, "scripts", "chiplet_example.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- (1) review: lane rate
def test_scale_up_lane_rate_by_generation():
    assert CHIPS["1P"].io_su_lane_Gbps == 200.0 and CHIPS["100T"].io_su_lane_Gbps == 200.0
    assert CHIPS["H100-like"].io_su_lane_Gbps == 100.0 and CHIPS["H100-like-lp4x64"].io_su_lane_Gbps == 100.0
    assert set(SU_LANE_GENERATIONS) == {50.0, 100.0, 200.0} and "NVLink5" in SU_LANE_GENERATIONS[200.0]
    # NVLink5-class 900 GB/s per direction = 36 lanes at 200 G (18 links × 2 pairs) — 0.312's 100 G needed 72
    lanes = {g: io_need(_s(tp=8, link=900.0, io_su_lane_Gbps=g))["scaleup_lanes"] for g in (50.0, 100.0, 200.0)}
    assert lanes == {50.0: 144, 100.0: 72, 200.0: 36}
    n = io_need(_s(tp=8, link=900.0))
    # 224G lane ≈ 0.5 mm of edge (OIF 448G workshop: ~4 mm per 8-lane port) → 16 × 0.285 + 36 × 0.5 = 22.56 mm
    assert n["su_lane_Gbps"] == 200.0 and n["su_lane_mm"] == 0.5 and abs(n["need_mm"] - 22.56) < 1e-9
    assert "NVLink5" in n["su_lane_basis"] and CHIPS["1P"].io_su_lane_mm == 0.0
    assert io_need(_s(tp=8, link=900.0, io_su_lane_Gbps=100.0))["su_lane_mm"] == 0.285
    assert io_need(_s(tp=8, link=900.0, io_su_lane_mm=0.3))["su_lane_mm"] == 0.3                 # pinned
    assert io_need(_s("H100-like", tp=8, link=450.0))["scaleup_lanes"] == 36      # NVLink4 18 × 2 (unchanged)
    # the 0.312 verdict "8 × HBM3E monolithic unbuildable at tp8 / 900" holds at 200 G too (22.6 > 16.7 mm free);
    # 400 GB/s fits at 100 G and 200 G, not at 50 G
    for g in (100.0, 200.0):
        assert not shoreline_report(_s(tp=8, link=900.0, io_su_lane_Gbps=g))["feasible"]
        assert shoreline_report(_s(tp=8, link=400.0, io_su_lane_Gbps=g))["feasible"]
    assert not shoreline_report(_s(tp=8, link=400.0, io_su_lane_Gbps=50.0))["feasible"]
    for bad in (0, -1, True):
        try:
            replace(CHIPS["1P"], io_su_lane_Gbps=bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_interposer_bbox_and_sourced_stack():
    r = shoreline_report(_s())                       # 1P + 8 stacks on 4 edges: the corner squares stay empty
    fp = r["footprint"]
    assert abs(fp["sum_mm2"] - 1622.2) < 0.1 and abs(fp["bbox_mm2"] - 2263.6) < 0.1 and fp["stack_depth_mm"] == 11.0
    assert abs(shoreline_report(_s("H100-like"))["footprint"]["bbox_mm2"] - 2053.3) < 0.1
    src = open(os.path.join(ROOT, "accel_dse", "core", "budget.py"), encoding="utf-8").read()
    assert "Micron HBM3E product brief" in src


# ---------------------------------------------------------------- (1) review: strict 4-sided memory (opt-in)
def test_shore_strict_flips_the_8_stack_verdict():
    ex = _ex()
    q = dict(out=lambda *a: None, cost="n5_cowos_l", d0=0.1, macs=(1.0,))
    o = ex.run("qwen3-8b", "decode", **q)
    assert o["chiplet"]["compare"][0]["verdict"] == "mono" and not o["shore_strict"]
    st = ex.run("qwen3-8b", "decode", shore_strict=True, **q)
    assert st["shore_strict"] and st["chiplet"]["compare"][0]["verdict"] == "only_split"
    mono = [x for x in st["rows"] if x["dies"] == 1]
    assert mono and all(x["buildable"] is False and x["shoreline"] == "tight" and "shore_strict" in x["why"] for x in mono)
    assert all(x["buildable"] for x in st["rows"] if x["dies"] > 1)
    t9 = ex.run("qwen3-8b", "decode", tp=8, link=900.0, **q)       # 200 G lanes at 0.5 mm: mono still I/O-limited
    m9 = [x for x in t9["rows"] if x["dies"] == 1 and x.get("clusters", 1) == 1][0]
    assert m9["buildable"] is False and "scale-up 36 通道需 22.6 mm" in m9["why"]
    assert t9["chiplet"]["compare"][0]["verdict"] == "only_split"
    t4 = ex.run("qwen3-8b", "decode", tp=8, link=400.0, **q)
    assert t4["chiplet"]["compare"][0]["verdict"] == "mono"


# ---------------------------------------------------------------- (2) thermal / power
def test_thermal_report_presets_and_known_parts():
    assert set(COOLING) == {"air_oam", "air_hgx", "liquid"}
    # 0.314 (MODEL §259): air_oam 600 → 1000 W (MI350X OAM air), flux air 0.85 → 0.86 (H100 SXM), liquid 2.5 → 1.70 (HIR)
    assert (COOLING["air_oam"]["pkg_W"], COOLING["air_hgx"]["pkg_W"], COOLING["liquid"]["pkg_W"]) == (1000.0, 1000.0, 1400.0)
    assert (COOLING["air_oam"]["flux_W_mm2"], COOLING["liquid"]["flux_W_mm2"]) == (0.86, 1.70)
    h = [p for p in KNOWN_PARTS if p["part"] == "H100 SXM"][0]
    assert abs(h["W"] / h["die_mm2"] - 0.86) < 0.01                # H100 sits at the air flux limit
    t = thermal_report(700.0, 814.0, 1, 60.0, None)
    # 0.314: judged flux = card-level (700 / 814); the die-level figure (DRAM devices out) is reported beside it
    assert t["status"] == "unchecked" and t["feasible"] is None and abs(t["flux_W_mm2"] - 700 / 814) < 1e-12
    assert abs(t["die_flux_W_mm2"] - 640 / 814) < 1e-12
    assert t["limits"]["air_oam"]["pkg_ok"] is True and t["limits"]["air_hgx"]["pkg_ok"] is True
    assert thermal_report(700.0, 814.0, 1, 60.0, "air_hgx")["feasible"] is True
    o = thermal_report(1100.0, 1600.0, 1, 60.0, "air_oam")
    assert o["feasible"] is False and o["reasons"][0].startswith("封装功耗 1100 W > air_oam")
    two = thermal_report(1000.0, 800.0, 2, 200.0, None)            # equal split over the package's dies
    assert two["die_W"] == 400.0 and two["flux_W_mm2"] == 1000.0 / 1600.0 and two["die_flux_W_mm2"] == 0.5
    assert two["pd"]["judged"] is False
    assert thermal_report(None, 800.0)["status"] == "unknown"
    assert dram_dev_W({"J_by_action": {"dram": 1.0, "hbm_static": 1.0, "mac": 2.0}, "avg_W_per_card": 100.0}) == 50.0
    try:
        thermal_report(1.0, 1.0, 1, 0.0, "water")
        raise AssertionError
    except ValueError:
        pass


def test_thermal_in_eval_and_dse():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": H4, "serving": {"phase": "prefill", "batch": 1,
                                                                                         "prompt": 4096, "ctx": 8192}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "sensitivity": False}
    o = api.api_eval(b)
    t = o["thermal"]                  # 1P prefill: 504 W on 571 mm² → 0.88 W/mm² card-level (0.314; air 0.86)
    assert t["status"] == "unchecked" and abs(t["pkg_W"] - 504.25) < 0.1 and abs(t["die_flux_W_mm2"] - 0.8584) < 1e-3
    assert abs(t["flux_W_mm2"] - 0.8825) < 1e-3 and t["static_included"] is False and t["notes"]
    oa = api.api_eval({**b, "cooling": "air_oam"})
    assert oa["thermal"]["feasible"] is False and any(w.startswith("不可建（热 / 功耗）：热流密度 0.88") for w in oa["summary"]["warnings"])
    assert api.api_eval({**b, "cooling": "liquid"})["thermal"]["feasible"] is True
    try:
        api.api_eval({**b, "cooling": "x"})
        raise AssertionError
    except api.ApiError:
        pass
    ex = _ex()
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=H4, serving=PF)
    kw = dict(mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,))
    d0 = energy_design_search(s, ex.table(), ex.budget(), **kw)
    assert d0["cooling"] is None and d0["n_thermal_over"] == 0 and d0["front"] == [0]
    assert d0["rows"][0]["thermal"]["status"] == "unchecked" and d0["rows"][0]["thermal"]["A_per_mm2"] > 0
    da = energy_design_search(s, ex.table(), ex.budget(), cooling="air_oam", **kw)
    assert da["n_thermal_over"] == 1 and da["front"] == [] and "热流密度" in da["rows"][0]["why"]
    di = energy_design_search(s, ex.table(), ex.budget(), cooling="air_oam", include_unbuildable=True, **kw)
    assert di["front"] == [0] and "why_thermal" in di["rows"][0]
    assert energy_design_search(s, ex.table(), ex.budget(), cooling="liquid", **kw)["front"] == [0]
    o2 = api.api_insight_dse({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": H4},
                              "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived"},
                              "mac_scales": [1], "sram_scales": [1], "sides": [128], "cooling": "liquid",
                              "shore_strict": True})
    assert o2["cooling"] == "liquid" and o2["shore_strict"]


def test_ui_and_example_cli():
    html, js = W.read(W.INDEX_HTML), W.read()
    for k in ('id="c-io_su_lane_Gbps"', 'id="c-cooling"', 'id="chip-thermal"', 'id="ins-dse-strict"', 'id="ins-dse-cooling"'):
        assert k in html, k
    assert "renderThermal" in js and "ex.shore_strict = true" in js and "cooling: S.cooling" in js
    assert "0.313: buildability switches apply to every DSE" in js
    src = open(os.path.join(ROOT, "scripts", "chiplet_example.py"), encoding="utf-8").read()
    for k in ('"--lane"', '"--strict"', '"--cooling"'):
        assert k in src
