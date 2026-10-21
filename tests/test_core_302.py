"""0.302: sourced HBM4 / LPDDR6 pJ/bit, LPDDR background (standby + refresh) power, DVFS operating point as a DSE axis."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import Budget, apply_preset
from accel_dse.core.energy import (ENERGY_PRESETS, EnergyTable, apply_energy_preset, dram_pJ_field, energy_report,
                                   hbm_static_W)
from accel_dse.core.energy_dse import MEM_PROXY, energy_design_search
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.powercap import volt
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.mem_catalog import parse_mem_id
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
NEW = ("pJ_bit_dram_hbm4", "pJ_bit_dram_lpddr6", "mW_lpddr_ch_standby", "mW_lpddr_Gb_refresh")


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _r(mem, chip="1P", phase="decode", batch=8):
    return evaluate(Scenario(model=M, chip=CHIPS[chip], mem_id=mem,
                             serving=Serving(phase=phase, batch=batch, prompt=2048, ctx=4096)))


def test_preset_values_and_tags():
    f = ENERGY_PRESETS["tpuv4i_7nm"]["fields"]
    h3, l5 = f["pJ_bit_dram_hbm"][0], f["pJ_bit_dram_lpddr"][0]
    # vendor / ISSCC efficiency ratios applied to the previous generation's preset 「推导」
    assert abs(f["pJ_bit_dram_hbm4"][0] - h3 / 1.4) < 1e-12 and abs(f["pJ_bit_dram_lpddr6"][0] - l5 / 1.21) < 1e-12
    # Micron Y52Q Table 16: IDD2N on all four rails at typical voltages; refresh (IDD5 − IDD3N) × tRFCab / tREFI / 8 Gb
    assert abs(f["mW_lpddr_ch_standby"][0] - 32.22) < 1e-9
    assert abs(f["mW_lpddr_Gb_refresh"][0] - (11.3 * 1.8 + 89 * 1.05) * 0.210 / 3.904 / 8) < 1e-12
    ok = {"来源", "推导", "二手", "假设"}
    for k in NEW:
        assert set(f[k][1].split("/")) <= ok and f[k][2], k
    assert "ISSCC 2026" in f["pJ_bit_dram_lpddr6"][2] and "Y52Q" in f["mW_lpddr_ch_standby"][2]
    assert "HBM4E" in MEM_PROXY["HBM4E"] and "pJ_bit_dram_hbm4" in MEM_PROXY["HBM4"]


def test_dram_field_by_generation():
    t = _t()
    assert dram_pJ_field(_r("hbm4_4s_12h24g_8000"), t) == "pJ_bit_dram_hbm4"
    assert dram_pJ_field(_r("hbm3e_4s_8h24g_9200"), t) == "pJ_bit_dram_hbm"
    assert dram_pJ_field(_r("lpddr6_4x96_10667_16g", "100T"), t) == "pJ_bit_dram_lpddr6"
    assert dram_pJ_field(_r("lpddr5x_4x64_8533_16g", "100T"), t) == "pJ_bit_dram_lpddr"
    # unset generation field → the per-kind one (≤ 0.301 behaviour); user value wins over the preset
    t0 = replace(t, pJ_bit_dram_hbm4=None)
    assert dram_pJ_field(_r("hbm4_4s_12h24g_8000"), t0) == "pJ_bit_dram_hbm"
    assert _t(pJ_bit_dram_hbm4=1.0).pJ_bit_dram_hbm4 == 1.0
    # DRAM energy ratio is exactly the per-bit ratio on the same run (same bytes)
    r = _r("hbm4_4s_12h24g_8000")
    a, b = energy_report(r, t), energy_report(r, t0)
    assert abs(a["J_by_action"]["dram"] / b["J_by_action"]["dram"] - 1 / 1.4) < 1e-9


def test_lpddr_background_power():
    t = _t()
    sp = parse_mem_id("lpddr5x_4x64_8533_16g")
    hs = hbm_static_W(sp.id, t)
    gb = sp.nominal_capacity_GB * 8
    assert hs["kind"] == "LPDDR" and abs(hs["channel_slices"] - gb / 8) < 1e-12
    assert abs(hs["standby_W"] - gb / 8 * 32.22e-3) < 1e-12
    assert abs(hs["refresh_W"] - gb * t.mW_lpddr_Gb_refresh * 1e-3) < 1e-12
    # scales with capacity, not with bandwidth: 2× capacity per package → 2×
    assert abs(hbm_static_W("lpddr5x_4x64_8533_32g", t)["W"] / hs["W"] - 2) < 1e-9
    # HBM unchanged (0.216 terms), LPDDR terms absent → None (≤ 0.301)
    assert hbm_static_W("hbm3e_4s_8h24g_9200", t)["kind"] == "HBM"
    assert hbm_static_W(sp.id, replace(t, mW_lpddr_ch_standby=None, mW_lpddr_Gb_refresh=None)) is None
    # charged in the report as static (no idle figure); a measured idle already contains it
    r = _r(sp.id, "100T")
    e = energy_report(r, t)
    assert abs(e["hbm_static_W_card"] - hs["W"]) < 1e-12 and e["J_by_action"]["hbm_static"] > 0
    e2 = energy_report(r, replace(t, idle_W=5.0))
    assert "hbm_static" not in e2["J_by_action"]
    assert not {"mW_lpddr_ch_standby", "pJ_bit_dram_lpddr6"} & set(energy_report(r, EnergyTable(pJ_mac=1))["missing"])


def _dse(**kw):
    s = Scenario(model=M, chip=CHIPS["1P"], serving=Serving(phase="prefill", batch=1, prompt=2048, ctx=2048))
    b = Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i", "die_mm2": 1200, **kw.pop("bud", {})}))
    return energy_design_search(s, _t(), b, mac_scales=kw.pop("mac_scales", (0.5, 1.0)), sram_scales=(1.0,),
                                sides=(128,), **kw)


def test_dvfs_axis_v2_mechanism():
    o = _dse(f_scales=(1.0, 0.7), curve="rx5700xt_n7")
    assert o["f_scales"] == [1.0, 0.7] and len(o["rows"]) == 4 and any("频率轴" in n for n in o["notes"])
    rows = {(x["mac_x"], x["f_op"]): x for x in o["rows"]}
    a, b = rows[(1.0, 1.0)], rows[(1.0, 0.7)]
    assert a["base"] and not b["base"] and abs(b["f_ghz"] - 0.7 * a["f_ghz"]) < 1e-12
    # prefill is compute-bound: tok/s ∝ f; MAC energy share falls by V(0.7)² per token
    v = volt("rx5700xt_n7", 0.7)
    assert abs(b["thr"] / a["thr"] - 0.7) < 0.02
    ja, jb = a["J_per_unit"] * a["share"]["mac"], b["J_per_unit"] * b["share"]["mac"]
    assert abs(jb / ja - v * v) < 1e-6
    # "dfs" keeps V: the per-token MAC energy is unchanged
    d = {(x["mac_x"], x["f_op"]): x for x in _dse(f_scales=(1.0, 0.7), curve="dfs")["rows"]}
    assert abs(d[(1.0, 0.7)]["J_per_unit"] * d[(1.0, 0.7)]["share"]["mac"] / ja - 1) < 1e-6
    for bad in ((0.1,), (1.2,)):
        try:
            _dse(f_scales=bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_dvfs_axis_trades_macs_vs_clock_under_cap():
    # compute-bound prefill under a 300 W cap: 2× MACs at 0.6 f_max (lower V) beats 1× MACs at f_max (cap-throttled)
    o = _dse(mac_scales=(1.0, 2.0), f_scales=(1.0, 0.6), curve="rx5700xt_n7", bud={"power_W_card": 300, "die_mm2": 3000})
    rows = {(x["mac_x"], x["f_op"]): x for x in o["rows"]}
    for k, x in rows.items():
        assert x["feasible"] and x["W_card"] <= 300 * (1 + 1e-6), k
        assert x["f_scale"] <= x["f_op"] + 1e-12           # running clock never above the operating point
        if x["capped"]:                                      # a capped point equals its f_op = 1 twin
            tw = rows[(x["mac_x"], 1.0)]
            assert abs(x["J_per_unit"] - tw["J_per_unit"]) < 1e-9 * tw["J_per_unit"]
    a, w = rows[(1.0, 1.0)], rows[(2.0, 0.6)]
    assert a["capped"] and not w["capped"] and a["W_nominal"] > 300 > w["W_nominal"]
    assert w["thr"] > 1.3 * a["thr"] and w["J_per_unit"] < a["J_per_unit"]
    assert w.get("front") and not a.get("front")


def test_api_and_ui_wiring():
    body = {"chip_preset": "1P", "scenario": {"model": M, "serving": {"phase": "prefill", "batch": 1, "prompt": 1024}},
            "energy": {"energy_preset": "tpuv4i_7nm"}, "mac_scales": [1], "sram_scales": [1], "f_scales": [1, 0.5],
            "curve": "rx5700xt_n7"}
    o = api.api_insight_dse(body)
    assert [x["f_op"] for x in o["rows"]] == [1.0, 0.5] * (len(o["rows"]) // 2) and o["f_scales"] == [1.0, 0.5]
    for bad in ([0.1], [2], "x", []):
        try:
            api.api_insight_dse({**body, "f_scales": bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass
    html, js = W.read(W.INDEX_HTML), W.read()
    assert {"ins-dse-freq", "ins-dse-curve", "e-pJ_bit_dram_hbm4", "e-pJ_bit_dram_lpddr6"} <= W.html_ids(html)
    src = W.function_source(js, "runInsightDse")
    for k in ("f_scales", "ins-dse-curve", "f_op"):
        assert k in src, k
    assert W.has(js, "'pJ_bit_dram_hbm4', 'pJ_bit_dram_lpddr6'")
