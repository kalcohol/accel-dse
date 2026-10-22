"""0.327: HBM memory-controller area (budget.mm2_per_hbm_mc / mc_preset, MODEL §272) — Samsung Hot Chips 2026 "about
5-10% of the SoC area" 「二手」 × its 26 × 33 mm, 6-stack reference xPU 「推导」; opt-in (no preset → byte-identical);
std hosts carry it, a custom base die takes it off the host; HBM only (0.328: HBM4 / 4E, n5_derived default — re-pinned); eval = DSE row; report / UI wiring.  Plus the
0.326 consistency fixes: hbm_base no longer lists organic SPHBM4 as unmodelled."""
import json

from accel_dse import api
from accel_dse.core import budget as bmod, hbm_base
from tests import webstruct as W  # noqa: E402

HBM4 = "hbm4_8s_12h24g_10000"


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


def test_sourced_constants():
    assert bmod.MC_SOC_FRAC == (0.05, 0.10) and bmod.MC_XPU_MM2 == 26 * 33 and bmod.MC_XPU_STACKS == 6
    lo = bmod.MC_PRESETS["samsung_hc2026_lo"]["fields"]["mm2_per_hbm_mc"]
    hi = bmod.MC_PRESETS["samsung_hc2026_hi"]["fields"]["mm2_per_hbm_mc"]
    assert lo[0] == _approx(7.15) and hi[0] == _approx(14.3) and lo[1] == hi[1] == "二手/推导"
    assert "Hot Chips 2026" in lo[2] and "5-10%" in lo[2] and "假设" in lo[2]
    # 0.328 (old → new): opt-in only → n5_derived carries the low end for HBM4 / HBM4E (MODEL §273); n7 / n28 do not
    for k, p in bmod.DENSITY_PRESETS.items():
        assert ("mm2_per_hbm_mc" in p["fields"]) == (k == "n5_derived") and "mm2_per_unit_mc" not in p["fields"]
    assert bmod.apply_mc_preset({"mc_preset": "samsung_hc2026_lo", "mm2_per_hbm_mc": 3.0})["mm2_per_hbm_mc"] == 3.0
    try:
        bmod.apply_mc_preset({"mc_preset": "x"})
        raise AssertionError("bad mc_preset accepted")
    except ValueError:
        pass


def test_off_is_unchanged():
    # 0.328: "off" is now mm2_per_hbm_mc = 0 on n5_derived (0.327 default); a density preset without it still reports the range
    a = api.api_eval(_body(bud={"mm2_per_hbm_mc": 0}))
    assert "mc_mm2_per_stack" not in (a.get("hbm_base") or {}) and a["shoreline"]["die_mm2"] == _approx(820.39, rel=1e-4)
    c = api.api_eval(_body("custom", bud={"density_preset": "n7_tpuv4i"}))
    assert c["hbm_base"]["mc_mm2_per_stack"] is None and c["hbm_base"]["mc_reclaim_mm2"] is None
    lo, hi = c["hbm_base"]["mc_reclaim_range_mm2"]
    assert lo == _approx(8 * 7.15) and hi == _approx(8 * 14.3)
    assert any(n.startswith("内存控制器面积未计") for n in c["hbm_base"]["notes"])


def test_std_carries_custom_moves():
    b = bmod.Budget(**{k: v for k, (v, _, _) in bmod.DENSITY_PRESETS["n5_derived"]["fields"].items()
                       if k in bmod.Budget.__dataclass_fields__})            # 0.328: carries 7.15 itself
    st = api.scenario_from_body(_body())
    cu = api.scenario_from_body(_body("custom"))
    ts, tc = bmod.full_area(st, b)["terms"], bmod.full_area(cu, b)["terms"]
    assert ts["mem_ctrl"] == _approx(8 * 7.15) and "mem_ctrl" not in tc
    s0, s1 = (api.api_eval(_body(bud=x)) for x in ({"mm2_per_hbm_mc": 0}, {"mc_preset": "samsung_hc2026_lo"}))
    assert s1["shoreline"]["die_mm2"] == _approx(s0["shoreline"]["die_mm2"] + 8 * 7.15 / (1 - 0.23), rel=1e-6)
    c0, c1 = (api.api_eval(_body("custom", bud=x)) for x in ({"mm2_per_hbm_mc": 0}, {"mc_preset": "samsung_hc2026_lo"}))
    assert c1["shoreline"]["die_mm2"] == c0["shoreline"]["die_mm2"] and c1["tco"]["usd_per_Munit"] == c0["tco"]["usd_per_Munit"]
    assert c1["hbm_base"]["mc_reclaim_mm2"] == _approx(8 * 7.15)
    # the decision: without the controller std is cheaper per Mtok, with Samsung's 5 % the custom die is
    assert s0["tco"]["usd_per_Munit"] < c0["tco"]["usd_per_Munit"] and s1["tco"]["usd_per_Munit"] > c1["tco"]["usd_per_Munit"]
    # generic user field wins over the HBM one
    b2 = bmod.Budget(**{**b.__dict__, "mm2_per_unit_mc": 1.0})
    assert bmod.full_area(st, b2)["terms"]["mem_ctrl"] == _approx(8.0)


def test_hbm_only():
    mem = "lpddr5x_4x64_8533_16g"
    e0, e1 = (api.api_eval(_body(mem=mem, bud=x)) for x in ({"mm2_per_hbm_mc": 0}, {"mc_preset": "samsung_hc2026_hi"}))
    assert e1["shoreline"]["die_mm2"] == e0["shoreline"]["die_mm2"] and e1["energy"]["J_per_unit"] == e0["energy"]["J_per_unit"]
    s = api.scenario_from_body(_body(mem=mem))
    b = bmod.Budget(**{**{k: v for k, (v, _, _) in bmod.DENSITY_PRESETS["n5_derived"]["fields"].items()
                          if k in bmod.Budget.__dataclass_fields__}, "mm2_per_hbm_mc": 14.3})
    fa = bmod.full_area(s, b)
    # 0.328 (old → new): no note on non-HBM memory any more (n5_derived fills the field by default); HBM3 / 3E get it
    assert "mem_ctrl" not in fa["terms"] and not any("只用于 HBM" in n for n in fa["phy_notes"])
    assert bmod.full_area(s, bmod.Budget(**{**b.__dict__, "mm2_per_unit_mc": 2.0}))["terms"]["mem_ctrl"] > 0


def test_eval_equals_dse_row_with_preset():
    for base in ("std", "custom"):
        bod = _body(base, bud={"mc_preset": "samsung_hc2026_hi"})
        e = api.api_eval(bod)
        d = {**bod, "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026", "d0_per_cm2": 0.1}
        d.pop("sensitivity")
        r = api.api_insight_dse(d)
        row = r["rows"][r["base"]]
        assert row["tco"]["usd_per_Munit"] == _approx(e["tco"]["usd_per_Munit"])
        assert row["usd_good_pkg"] == _approx(e["tco"]["pkg_usd"])


def test_report_item():
    # 0.328: n5_derived carries the controller → item shown by default; n7_tpuv4i (no default) → the "未计" note
    for bud, want in (({"density_preset": "n7_tpuv4i"}, False), ({}, True), ({"mc_preset": "samsung_hc2026_lo"}, True)):
        b = {**_body("custom", bud=bud), "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026"}
        b.pop("sensitivity")
        txt = json.dumps(api.api_report(b), ensure_ascii=False)
        assert ("HBM 内存控制器面积" in txt) == want
        assert ("内存控制器面积未计" in txt) == (not want)
    b = {**_body(), "mac_scales": [1], "sram_scales": [1], "cost": "n5_cowos_s", "tco": "accel_2026"}
    b.pop("sensitivity")
    txt = json.dumps(api.api_report(b), ensure_ascii=False)
    assert "HBM 内存控制器面积" in txt and "内存控制器面积未计" not in txt          # 0.328: std n5_derived shows it


def test_catalog_ui():
    cat = api.api_catalog()
    assert set(cat["mc_presets"]) == {"samsung_hc2026_lo", "samsung_hc2026_hi"}
    html, js = W.read(W.INDEX_HTML), W.read()
    ids = W.html_ids(html)
    assert "b-mc-preset" in ids and "b-mm2_per_hbm_mc" in ids and "b-mc-preset-note" in ids
    assert W.has(js, "initElectricityPresets('b-mc-preset', 'mc_presets', 'mcPresetFilled')")
    assert W.has(js, "'mm2_per_unit_mc', 'mm2_per_hbm_mc'])")


def test_stale_sphbm4_notes_fixed():
    assert not any("SPHBM4" in x and "未建模" in x for x in hbm_base.NOT_MODELLED)
    c = api.api_eval(_body("custom"))
    assert not any("另一种封装，未建模" in n for n in c["hbm_base"]["notes"])
    assert any("hbm_pkg=sphbm4" in n for n in c["hbm_base"]["notes"])
