"""0.328: the HBM4 / HBM4E memory-controller area defaults on in n5_derived at 7.15 mm² per stack (MODEL §273) —
the 0.327 double-count question resolved from TPU floor plans (TPUv1 Jouppi ISCA 2017 Fig. 2, TPUv2 Norrie IEEE
Micro 2021 Fig. 4, TPUv4i Jouppi ISCA 2021 Fig. 6, NeuroMeter Tang et al. HPCA 2021): the DRAM / HBM port is drawn
outside the ~21 % unknown share the uncore_frac comes from.  Scope HBM4 / HBM4E (0.327: any HBM); mc_preset now
overrides the density default; NeuroMeter citation fix."""
import json

from accel_dse import api
from accel_dse.core import budget as bmod
from tests import webstruct as W  # noqa: E402

HBM4, HBM4E, HBM3E = "hbm4_8s_12h24g_10000", "hbm4e_8s_12h24g_12000", "hbm3e_8s_8h24g_9200"
LP = "lpddr5x_4x64_8533_16g"


def _approx(x, rel=1e-9):
    class A:
        def __eq__(self, y):
            return abs(y - x) <= rel * max(abs(x), abs(y), 1e-300)
    return A()


def _body(base="std", mem=HBM4, bud=None):
    ch = {"hbm_base": base} if base != "std" else {}
    return {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": mem, "chip": ch,
            "serving": {"phase": "decode", "batch": 64, "prompt": 2048, "ctx": 4096}},
            "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived", **(bud or {})},
            "sensitivity": False, "tco": {"preset": "accel_2026", "cost": "n5_cowos_s", "d0_per_cm2": 0.1}}


def test_default_value_and_sources():
    v, tag, src = bmod.DENSITY_PRESETS["n5_derived"]["fields"]["mm2_per_hbm_mc"]
    assert v == _approx(0.05 * 26 * 33 / 6) and v == bmod.MC_PRESETS["samsung_hc2026_lo"]["fields"]["mm2_per_hbm_mc"][0]
    assert tag == "二手/推导"
    for s in ("Hot Chips 2026", "ISCA 2017 Fig. 2", "HPCA 2021", "IEEE Micro 2021 Fig. 4", "ISCA 2021 Fig. 6", "假设"):
        assert s in src, s
    for k in ("n7_tpuv4i", "n28_tpuv1"):                  # no node scaling invented for the other presets
        assert "mm2_per_hbm_mc" not in bmod.DENSITY_PRESETS[k]["fields"]
    assert bmod.auto_area_budget().mm2_per_hbm_mc == _approx(v)     # the auto memory-edge proxy is the same die


def test_scope_hbm4_only():
    b = bmod.Budget(mm2_per_hbm_mc=7.15)
    # 0.329 re-pin (MODEL §274): scope HBM4 / HBM4E → HBM3 / 3E / 4 / 4E, HBM3 / 3E scaled by JEDEC channels 16 / 32
    assert bmod.MC_TYPES == ("HBM3", "HBM3E", "HBM4", "HBM4E")
    assert bmod.mc_mm2_per_unit(b, "HBM4") == 7.15 and bmod.mc_mm2_per_unit(b, "HBM4E") == 7.15
    assert bmod.mc_mm2_per_unit(b, "HBM3E") == _approx(3.575) and bmod.mc_mm2_per_unit(b, "HBM3") == _approx(3.575)
    assert bmod.mc_mm2_per_unit(b, "LPDDR5X") is None
    assert bmod.mc_mm2_per_unit(bmod.Budget(mm2_per_hbm_mc=7.15, mm2_per_unit_mc=2.0), "HBM3E") == 2.0


def test_eval_default_numbers_and_decision():
    on, off = api.api_eval(_body()), api.api_eval(_body(bud={"mm2_per_hbm_mc": 0}))
    assert off["shoreline"]["die_mm2"] == _approx(820.436, rel=1e-5)                  # = 0.327 default
    assert on["shoreline"]["die_mm2"] == _approx(off["shoreline"]["die_mm2"] + 8 * 7.15 / (1 - 0.23), rel=1e-6)
    assert on["shoreline"]["die_mm2"] == _approx(894.72, rel=1e-5)
    assert on["energy"]["J_per_unit"] == off["energy"]["J_per_unit"]
    cu = api.api_eval(_body("custom"))
    # the decision flips (old → new): std 9.7658 m$ < custom 9.8678 → std 9.8969 m$ > custom 9.8678
    # 0.333 re-pin (HBM 5.80 pJ/bit + per-channel standby, MODEL §278): on 0.009896946997665858 → 0.01085371734447578
    # (off 0.009766 → 0.010723, custom 0.009868 → 0.010825); the decision order is unchanged
    assert on["tco"]["usd_per_Munit"] == _approx(0.01085371734447578, rel=1e-9)
    assert off["tco"]["usd_per_Munit"] < cu["tco"]["usd_per_Munit"] < on["tco"]["usd_per_Munit"]
    assert cu["hbm_base"]["mc_mm2_per_stack"] == _approx(7.15) and cu["hbm_base"]["mc_reclaim_mm2"] == _approx(57.2)
    e4e = api.api_eval(_body(mem=HBM4E))
    assert e4e["budget"]["area"]["terms"]["mem_ctrl"] == _approx(8 * 7.15)


def test_hbm3e_and_lpddr_unchanged():
    # 0.329 re-pin: HBM3E is no longer unchanged — it gets 16 / 32 of the per-HBM4-stack controller (3.575 mm²) and
    # the HBM3 PHY (24 mm² per stack); die 654.2 → 774.5 mm² (737.3 with the controller off); LPDDR still untouched
    a, b = api.api_eval(_body(mem=LP)), api.api_eval(_body(mem=LP, bud={"mm2_per_hbm_mc": 0}))
    assert a["shoreline"]["die_mm2"] == b["shoreline"]["die_mm2"] and a["tco"]["usd_per_Munit"] == b["tco"]["usd_per_Munit"]
    assert "mem_ctrl" not in a["budget"]["area"]["terms"]
    assert not any("HBM" in n for n in a["budget"]["area"]["phy_notes"])
    a, b = api.api_eval(_body(mem=HBM3E)), api.api_eval(_body(mem=HBM3E, bud={"mm2_per_hbm_mc": 0}))
    assert a["budget"]["area"]["terms"]["mem_ctrl"] == _approx(8 * 3.575) and b["budget"]["area"]["terms"].get("mem_ctrl", 0) == 0
    assert a["shoreline"]["die_mm2"] == _approx(b["shoreline"]["die_mm2"] + 8 * 3.575 / (1 - 0.23), rel=1e-9)
    assert a["shoreline"]["die_mm2"] == _approx(774.462, rel=1e-5) and b["shoreline"]["die_mm2"] == _approx(737.319, rel=1e-5)
    assert a["tco"]["usd_per_Munit"] > b["tco"]["usd_per_Munit"]
    assert not any("只用于 HBM4 / HBM4E" in n for n in a["budget"]["area"]["phy_notes"])     # 0.328 note retired


def test_mc_preset_overrides_density_default():
    hi = api.api_eval(_body(bud={"mc_preset": "samsung_hc2026_hi"}))
    assert hi["budget"]["area"]["terms"]["mem_ctrl"] == _approx(8 * 14.3)
    usr = api.api_eval(_body(bud={"mc_preset": "samsung_hc2026_hi", "mm2_per_hbm_mc": 3.0}))   # user value still wins
    assert usr["budget"]["area"]["terms"]["mem_ctrl"] == _approx(24.0)
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "chiplet_example", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "chiplet_example.py"))
    ce = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ce)
    assert ce.budget().mm2_per_hbm_mc == _approx(7.15) and ce.budget("samsung_hc2026_hi").mm2_per_hbm_mc == _approx(14.3)


def test_eval_equals_dse_row_default():
    for base in ("std", "custom"):
        bod = _body(base)
        e = api.api_eval(bod)
        d = {**bod, "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026", "d0_per_cm2": 0.1}
        d.pop("sensitivity")
        r = api.api_insight_dse(d)
        row = r["rows"][r["base"]]
        assert row["tco"]["usd_per_Munit"] == _approx(e["tco"]["usd_per_Munit"])
        assert row["mm2"] == _approx(e["shoreline"]["die_mm2"])
    assert row["mm2"] < 858 < api.api_eval(_body())["shoreline"]["die_mm2"]          # std now above the reticle


def test_report_item_source():
    def rep(mem):
        b = {**_body(mem=mem), "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026"}
        b.pop("sensitivity")
        return json.dumps(api.api_report(b), ensure_ascii=False)
    t4 = rep(HBM4)
    assert "HBM 内存控制器面积" in t4 and "面积密度预设 n5_derived" in t4
    t3 = rep(HBM3E)                    # 0.329: HBM3E now carries the item, with the channel scaling disclosed
    assert "HBM 内存控制器面积" in t3 and "× 16 / 32 通道" in t3 and "× 16 / 32 通道" not in t4


def test_citation_fix_and_ui():
    src = bmod.DENSITY_PRESETS["n5_derived"]["fields"]["mm2_host_io"][2]
    assert src.startswith("NeuroMeter (Tang, Li, Nai, Jouppi, Xie, HPCA 2021") and "was 'Shao et al., MICRO 2021'" in src and "MODELLED" in src and "3.0 %" in src
    assert bmod.DENSITY_PRESETS["n5_derived"]["fields"]["mm2_host_io"][0] == _approx(0.018 * 331)   # value unchanged
    html, js = W.read(W.INDEX_HTML), W.read()
    assert "n5_derived 密度预设默认 7.15" in html and W.has(js, "'（每 HBM4 栈；HBM3 / 3E × 16/32 通道；悬停看出处）'")   # 0.329: was '（只用于 HBM4 / HBM4E；悬停看出处）'
    assert W.has(js, "catKey === 'mc_presets' && (S.presetFilled || {})[f] === cur")
