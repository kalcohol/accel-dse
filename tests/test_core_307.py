"""0.307: NoC router static power priced before the power cap (energy table fields, DVFS-scaled); router area counted
once (after the uncore share, not with mm2_fixed); engine local-port model with the DRAM stream sharing the ports /
ring memory-stop links (Chip.noc_eject, opt-in)."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import Budget, apply_preset, full_area, noc_router_mm2
from accel_dse.core.energy import (NOC_ROUTER_CLK_PJ_45, NOC_ROUTER_LEAK_MW_45, EnergyTable, apply_energy_preset,
                                   energy_report, noc_router_scale, noc_router_W)
from accel_dse.core.energy_dse import energy_design_search, noc_router_cost
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, NOC_EJECT
from accel_dse.core.powercap import _point, dvfs_cap, static_W, volt
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
H = "hbm3e_4s_8h24g_9200"
CLK, LEAK = NOC_ROUTER_CLK_PJ_45 / 2.6, NOC_ROUTER_LEAK_MW_45 / 2.6


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _chip(e, side, topo="mesh", link=64.0, **kw):
    return replace(CHIPS["1P"], engines=e, rows=side, cols=side, noc_topology=topo, noc_link_Bpc=link, **kw)


def _scn(c, phase="decode", batch=64):
    return Scenario(model=M, chip=c, mem_id=H, serving=Serving(phase=phase, batch=batch, prompt=2048, ctx=4096))


# ---------------------------------------------------------------- (1a) router static power in the energy table
def test_router_power_in_report():
    c = _chip(512, 32, link=32.0)
    r = evaluate(_scn(c))
    base, t = energy_report(r, _t()), _t(pJ_noc_router_cycle=CLK, mW_noc_router_leak=LEAK)
    e = energy_report(r, t)
    sc = noc_router_scale(32.0, "mesh")
    clk = 512 * sc["lin"] * CLK * c.freq_ghz * 1e-3
    leak = 512 * sc["area_x"] * LEAK * 1e-3
    nr = e["noc_router"]
    assert abs(nr["clk_W"] - clk) < 1e-9 and abs(nr["leak_W"] - leak) < 1e-9 and e["noc_leak_W_card"] == nr["leak_W"]
    assert abs(e["avg_W_per_card"] - base["avg_W_per_card"] - clk - leak) < 1e-6 * e["avg_W_per_card"]
    assert e["breakdown"]["J"]["noc"] > base["breakdown"]["J"]["noc"]
    assert e["breakdown"]["J"]["static"] - base["breakdown"]["J"]["static"] > 0
    assert abs(static_W(e) - static_W(base) - leak) < 1e-9
    # the 0.306 DSE cost function and the table path agree (same reference router, same scaling)
    assert abs(noc_router_cost(32.0, "mesh", c.freq_ghz, None)["W"] * 512 - nr["W"]) < 1e-9
    # nothing without a link width, on a crossbar, on one engine, or with the fields unset
    for x in (replace(c, noc_link_Bpc=0.0), replace(c, noc_topology="crossbar"), _chip(1, 724)):
        assert noc_router_W(t, x) is None
    assert noc_router_W(_t(), c) is None and "noc_router" not in base


def test_dvfs_scales_router_terms():
    c = _chip(512, 32, link=32.0)
    r = evaluate(_scn(c))
    t = _t(pJ_noc_router_cycle=CLK, mW_noc_router_leak=LEAK)
    s = 0.6
    q, e = _point(r, t, "rx5700xt_n7", s)
    v = volt("rx5700xt_n7", s)
    nr0 = energy_report(r, t)["noc_router"]
    assert abs(e["noc_router"]["clk_W"] - nr0["clk_W"] * s * v * v) < 1e-9 * nr0["clk_W"]
    assert abs(e["noc_router"]["leak_W"] - nr0["leak_W"] * v) < 1e-9 * nr0["leak_W"]
    # priced before the cap: the capped clock is lower with the router terms than without
    P = energy_report(r, t)["avg_W_per_card"]
    cap = 0.8 * P
    with_r, without = dvfs_cap(r, t, cap, "rx5700xt_n7"), dvfs_cap(r, _t(), cap, "rx5700xt_n7")
    assert with_r["binds"] and with_r["W_card"] <= cap * (1 + 1e-9) and with_r["s"] < without["s"]


def _dse(cap, **kw):
    b = Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i", "die_mm2": 1300, "power_W_card": cap}))
    s = Scenario(model=M, chip=CHIPS["1P"], serving=Serving(phase="decode", batch=64, prompt=2048, ctx=4096))
    return energy_design_search(s, _t(), b, mac_scales=(1.0,), sram_scales=(1.0,), sides=(16, 32, 64, 128, 256),
                                phase="decode", noc_links=(8, 32, 128), **kw)


def test_dse_recaps_with_router_power():
    """old → new: at 250 W, 2048 × 16² @ 32 B/cycle needs 268 W incl. 38 W of routers — 0.306 dropped it
    ('功耗 268 > 250 W（含 NoC 静态）'); 0.307 runs it DVFS-capped (f × 0.93, 3.19 k tok/s, 78.4 mJ)."""
    r = _dse(250, power_scope="card")       # 0.316: card-W cap; package scope runs it at f × 0.52, 1.80 k tok/s
    x = next(x for x in r["rows"] if x["engines"] == 2048 and x["noc_link_Bpc"] == 32.0)
    # 0.333: HBM 5.80 pJ/bit + per-channel standby → 268 → 345 W nominal: capped deeper, f × 0.926 → 0.682, 3.19 k →
    # 2.35 k tok/s; router W unchanged
    assert x["feasible"] and x["capped"] and x["W_card"] <= 250 * (1 + 1e-9) and 0.65 < x["f_scale"] < 0.72
    assert 2.2e3 < x["thr"] < 2.5e3 and abs(x["noc_W"] - 38.1) < 0.2
    assert all("NoC 静态" not in x.get("why", "") for x in r["rows"])
    # uncapped points are unchanged from 0.306 (the router W was already in W / J there)
    y = next(x for x in r["rows"] if x["engines"] == 128 and x["noc_link_Bpc"] == 128.0)
    # 0.333: 128 × 128 B now 282 W nominal > 250 → capped too (J/token 0.0613 → 0.0854; ≤ 0.332 uncapped at 206 W)
    assert y["capped"] and y["W_card"] <= 250 * (1 + 1e-9) and abs(y["J_per_unit"] - 0.0854) < 2e-4 and abs(y["noc_W"] - 14.9) < 0.1
    assert [r["rows"][i]["engines"] for i in r["front"]] == [128]
    assert any("功耗封顶 / DVFS 之前" in n for n in r["notes"])
    # user values win over the axis defaults
    t = _t(pJ_noc_router_cycle=0.0, mW_noc_router_leak=0.0)
    b = Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i", "die_mm2": 1300, "power_W_card": 250}))
    s = Scenario(model=M, chip=CHIPS["1P"], serving=Serving(phase="decode", batch=64, prompt=2048, ctx=4096))
    z = energy_design_search(s, t, b, mac_scales=(1.0,), sram_scales=(1.0,), sides=(16,), phase="decode", noc_links=(32,))
    assert z["rows"][0]["noc_W"] == 0.0


# ---------------------------------------------------------------- (1b) router area counted once
def test_router_area_once():
    c = _chip(512, 32, link=32.0)
    s = _scn(c)
    b = Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i"}))
    a0, a1 = full_area(s, b), full_area(s, b, noc_routers=True)
    nr = noc_router_mm2(c, b)
    assert abs(a1["terms"]["noc_routers"] - nr) < 1e-9 and abs(a1["mm2"] - a0["mm2"] - nr) < 1e-9
    assert a1["terms"]["uncore"] == a0["terms"]["uncore"]          # not scaled by the uncore share
    assert abs(nr - 512 * noc_router_cost(32.0, "mesh", 1.0, b.mm2_per_kmac)["mm2"]) < 1e-9
    bf = replace(b, mm2_fixed=100.0)                                 # "everything else, incl. NoC"
    f0, f1 = full_area(s, bf), full_area(s, bf, noc_routers=True)
    assert f1["mm2"] == f0["mm2"] and "noc_routers" not in f1["terms"] and any("NoC" in n for n in f1["phy_notes"])
    assert noc_router_mm2(replace(c, noc_link_Bpc=0.0), b) is None
    assert noc_router_mm2(c, replace(b, mm2_per_kmac=None)) is None


# ---------------------------------------------------------------- (2) local ports shared with the DRAM stream
def test_eject_geometry_and_validation():
    assert NOC_EJECT == ("off", "link") and CHIPS["1P"].noc_eject == "off"
    try:
        replace(CHIPS["1P"], noc_eject="wide")
        raise AssertionError
    except ValueError:
        pass
    c = _chip(16, 128, noc_eject="link")
    assert c.noc_eject_load(1600.0, 16) == 1500.0 and c.noc_eject_load(1600.0, 4, 4) == 1200.0 / 4
    assert c.noc_eject_load(1600.0, 4, 4, "reduce") == 400.0 and c.noc_eject_load(1600.0, 1) == 0.0
    # mesh 4 × 4 all-gather: column link 3/4 · P, local port 15/16 · P → the port binds
    assert c.noc_traffic_cycles(1600.0, 16) == 1500.0 / 64 and replace(c, noc_eject="off").noc_traffic_cycles(1600.0, 16) == 1200.0 / 64
    assert c.noc_port_cycles(1600.0, 16) == (1200.0 / 64, 1500.0 / 64)
    # crossbar: the output port is the local port already (no change)
    x = replace(c, noc_topology="crossbar")
    assert x.noc_traffic_cycles(1600.0, 16) == replace(x, noc_eject="off").noc_traffic_cycles(1600.0, 16)
    # off / no link / bisection: no port model
    for y in (replace(c, noc_eject="off"), replace(c, noc_link_Bpc=0.0), replace(c, noc_routing="bisection")):
        assert y.noc_port_cycles(1600.0, 16) == (0.0, 0.0) and y.noc_share_seconds(1.0, 1.0, 1e9) == 0.0
    # shared resources: port Σ + D / E; ring also the memory-stop link Σ + D / 4
    bw = 64 * c.freq_ghz * 1e9
    assert abs(c.noc_share_seconds(1e-3, 0.0, 1.6e9) - (1e-3 + 1e8 / bw)) < 1e-15
    rg = replace(c, noc_topology="ring")
    assert abs(rg.noc_share_seconds(0.0, 1e-3, 1.6e9) - (1e-3 + 4e8 / bw)) < 1e-15
    assert replace(c, noc_mem_attach="none").noc_share_seconds(1e-3, 0.0, 1.6e9) == 0.0


def _thr(c, phase="decode", batch=64):
    return evaluate(_scn(c, phase, batch)).throughput


def test_eject_effect_recorded():
    """old → new (noc_eject "link" vs "off"), 1P qwen3-8b HBM3E, L = 64 B/cycle; decode b64 / prefill b1 p2048."""
    for e, s, topo, ph, b, lo, hi in ((2, 512, "mesh", "decode", 64, -0.03, -0.005),
                                      (2, 512, "mesh", "prefill", 1, -0.10, -0.05),
                                      (32, 128, "mesh", "prefill", 1, -0.18, -0.13),
                                      (32, 128, "ring", "prefill", 1, -0.36, -0.30),
                                      (2048, 16, "ring", "prefill", 1, -0.37, -0.30),
                                      (512, 32, "crossbar", "prefill", 1, -1e-9, 1e-9),
                                      (2048, 16, "mesh", "decode", 64, -1e-9, 1e-9)):
        c = _chip(e, s, topo)
        d = _thr(replace(c, noc_eject="link"), ph, b) / _thr(c, ph, b) - 1
        assert lo <= d <= hi, (e, topo, ph, d)
    # default off: bit-identical to ≤ 0.306, and the presets carry no new field
    c = _chip(32, 128)
    assert _thr(c) == _thr(replace(c, noc_eject="off"))
    assert "noc_eject" not in api.api_eval({"chip_preset": "1P", "scenario": {"model": M}, "sensitivity": False}).get(
        "scenario", {}).get("chip", {})


def test_stage_dram_time_covers_shared_ports():
    c = _chip(2, 512, noc_eject="link")
    r = evaluate(_scn(c, "prefill", 1))
    st = r.stages[0]
    assert st.time.t_dram >= st.dram["total"] / c.engines / (64 * c.freq_ghz * 1e9) * (1 - 1e-12)
    assert st.time.t_dram > evaluate(_scn(replace(c, noc_eject="off"), "prefill", 1)).stages[0].time.t_dram


def test_ui_wiring():
    html, js = W.read(W.INDEX_HTML), W.read()
    ids = W.html_ids(html)
    assert {"c-noc_eject", "e-pJ_noc_router_cycle", "e-mW_noc_router_leak"} <= ids
    assert W.select_options(html, "c-noc_eject") == ["", "link"]
    assert W.count(js, "for (const k of ['noc_topology', 'noc_routing', 'noc_mem_attach', 'noc_eject'])") == 2
    assert W.has(js, "'noc_mm2_per_kmac', 'pJ_noc_router_cycle', 'mW_noc_router_leak'])")
    assert "noc_router" in W.function_source(js, "renderEnergy")
    o = api.api_eval({"chip_preset": "1P", "scenario": {"model": M, "chip": {"noc_link_Bpc": 32, "noc_eject": "link"}},
                      "energy": {"energy_preset": "tpuv4i_7nm", "pJ_noc_router_cycle": CLK, "mW_noc_router_leak": LEAK},
                      "sensitivity": False})
    assert o["scenario"]["chip"]["noc_eject"] == "link" and o["energy"]["noc_router"]["routers"] == 32
