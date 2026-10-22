"""0.306: NoC hop energy of DRAM traffic (memory columns → engines); chip-panel NoC controls; NoC link width as an
energy-DSE axis with a sourced router area / static-power cost."""
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import Budget, apply_preset
from accel_dse.core.energy import EnergyTable, action_counts, apply_energy_preset, energy_report
from accel_dse.core.energy_dse import (NOC_MAC_UM2_45, NOC_ROUTER_CLK_PJ_45, NOC_ROUTER_LEAK_MW_45, NOC_ROUTER_UM2_45,
                                       NOC_XBAR_UM2_45, energy_design_search, noc_router_cost)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _chip(e, side, topo="mesh", **kw):
    return replace(CHIPS["1P"], engines=e, rows=side, cols=side, noc_topology=topo, **kw)


def _res(c, phase="decode", batch=64):
    return evaluate(Scenario(model=M, chip=c, serving=Serving(phase=phase, batch=batch, prompt=2048, ctx=4096)))


# ---------------------------------------------------------------- DRAM bytes cross the NoC
def test_mem_hops_geometry():
    assert _chip(2, 512).noc_mem_hops() == (1.0, 1.0)                     # 1 × 2: each engine beside a memory column
    assert _chip(16, 128).noc_mem_hops() == (1.5, 1.5)                    # 4 × 4: columns 1, 2, 2, 1
    assert _chip(16, 128, "ring").noc_mem_hops() == (2.5, 2.5)            # two stops, arcs of 8: 1..4..1
    assert _chip(16, 128, "crossbar").noc_mem_hops() == (1.0, 1.5)        # one switch, same floor-plan wire
    assert _chip(16, 128, noc_mem_attach="none").noc_mem_hops() == (0.0, 0.0)
    assert _chip(1, 724).noc_mem_hops() == (0.0, 0.0)
    h = [_chip(e, s).noc_mem_hops()[0] for e, s in ((32, 128), (512, 32), (2048, 16))]
    assert h == sorted(h) and h[-1] == 12.0                               # 45 × 46 → ≈ C / 4


def test_dram_noc_energy_booked_and_timing_unchanged():
    t = _t()
    for topo in ("mesh", "ring", "crossbar"):
        e, n = _chip(512, 32, topo), _chip(512, 32, topo, noc_mem_attach="none")
        re_, rn = _res(e), _res(n)
        assert abs(re_.throughput - rn.throughput) < 1e-9 * rn.throughput     # link unset: energy only
        ae, an = action_counts(re_), action_counts(rn)
        h, p = e.noc_mem_hops()
        assert abs(ae["counts"]["noc"] - an["counts"]["noc"] - ae["counts"]["dram"] * h) < 1e-6 * ae["counts"]["noc"]
        assert abs(ae["counts"]["noc_wire"] - an["counts"]["noc_wire"] - ae["counts"]["dram"] * p) \
            < 1e-6 * ae["counts"]["noc_wire"]
        ee, en = energy_report(re_, t), energy_report(rn, t)
        assert "dram" not in en["noc"] and ee["noc"]["dram"]["hops_per_byte"] == h
        assert abs(ee["J_per_unit"] - en["J_per_unit"] - ee["noc"]["dram"]["J"]) < 1e-6 * ee["J_per_unit"]
    # decode (weight streaming) pays more than prefill, relatively
    d, pf = (energy_report(_res(_chip(512, 32), ph, b), t) for ph, b in (("decode", 64), ("prefill", 1)))
    assert d["noc"]["dram"]["J"] / d["J_per_unit"] > 5 * pf["noc"]["dram"]["J"] / pf["J_per_unit"]


def test_recorded_old_new():
    """old → new (0.306 default: DRAM bytes cross the NoC under noc_mem_attach "edge") on the tpuv4i_7nm preset."""
    t = _t()
    for chip, ph, b, lo, hi in (("1P", "decode", 8, 0.10, 0.20), ("1P", "prefill", 1, 0.002, 0.03),
                                ("100T", "decode", 64, 0.02, 0.08)):
        c = CHIPS[chip]
        new, old = (energy_report(_res(x, ph, b), t)["J_per_unit"] for x in (c, replace(c, noc_mem_attach="none")))
        assert lo < new / old - 1 < hi, (chip, ph, new / old)
    # design conclusion: wan2.1-1.3b energy-tied SRAM sizing is J/token-limited again (16 → ≈ 19 MiB)
    w = api.api_insight_sizing({"models": ["wan2.1-1.3b"], "energy": {"energy_preset": "tpuv4i_7nm"},
                                "resources": ["sram"], "energy_tie": True})["resources"][0]
    assert w["limited_by"]["metric"] == "J_tok" and 18 < w["min_ok"] < 20


# ---------------------------------------------------------------- router cost per link width
def test_router_cost_scaling():
    kd = NOC_MAC_UM2_45 * 1024 * 1e-6                     # a 45 nm MAC density → node factor 1
    r = noc_router_cost(8, "mesh", 1.0, kd)                 # 64-bit, 5 ports = the Park DAC 2012 router
    assert abs(r["mm2"] - NOC_ROUTER_UM2_45 * 1e-6) < 1e-12 and abs(r["area_x"] - 1) < 1e-12
    assert abs(r["W"] - (NOC_ROUTER_CLK_PJ_45 + NOC_ROUTER_LEAK_MW_45) * 1e-3 / 2.6) < 1e-12
    r2 = noc_router_cost(16, "mesh", 1.0, kd)
    assert abs(r2["mm2"] * 1e6 - (4 * NOC_XBAR_UM2_45 + 2 * (NOC_ROUTER_UM2_45 - NOC_XBAR_UM2_45))) < 1e-6
    big = noc_router_cost(1024, "mesh", 1.0, kd)            # crossbar (quadratic) dominates at wide links
    assert big["mm2"] / r["mm2"] > 128 ** 2 * NOC_XBAR_UM2_45 / NOC_ROUTER_UM2_45
    assert noc_router_cost(8, "ring", 1.0, kd)["mm2"] < r["mm2"]
    assert noc_router_cost(8, "crossbar", 1.0, kd) is None and noc_router_cost(0, "mesh", 1.0, kd) is None
    assert noc_router_cost(8, "mesh", 1.0, None)["mm2"] is None
    assert noc_router_cost(8, "mesh", 2.0, kd)["W"] > r["W"]


def _dse(power_W=300, **kw):
    b = Budget.from_dict(apply_preset({"density_preset": "n7_tpuv4i", "die_mm2": 1300, "power_W_card": power_W}))
    s = Scenario(model=M, chip=CHIPS["1P"], serving=Serving(phase="decode", batch=64, prompt=2048, ctx=4096))
    return energy_design_search(s, _t(), b, mac_scales=(1.0,), sram_scales=(1.0,), sides=(16, 32, 64, 128, 256),
                                phase="decode", **kw)


def test_link_width_axis_front():
    r = _dse(noc_links=(32, 8, 128), power_scope="card")     # 0.316: card-W cap (package scope: front = 128 × 128 B only)
    rows = r["rows"]
    assert r["noc_links"] == [8.0, 32.0, 128.0] and r["noc_link_Bpc"] is None and len(rows) == 15
    for e in {x["engines"] for x in rows}:
        g = sorted((x for x in rows if x["engines"] == e), key=lambda x: x["noc_link_Bpc"])
        assert [x["noc_link_Bpc"] for x in g] == [8.0, 32.0, 128.0]
        assert g[0]["noc_W"] < g[1]["noc_W"] < g[2]["noc_W"] and g[0]["noc_mm2"] < g[2]["noc_mm2"]
    big = max(rows, key=lambda x: x["noc_W"])
    # 0.307: router static power is priced before the cap → the point runs DVFS-capped (was: dropped over budget)
    assert big["engines"] == 2048 and not big["feasible"] and big["capped"] and big["W_card"] <= 300 * (1 + 1e-9)
    assert "面积" in big["why"] and "NoC 静态" not in big["why"]
    fr = [rows[i] for i in r["front"]]
    # 0.333: HBM 5.80 pJ/bit + per-channel standby add ≈ 77 W to every point: at 300 W the 2048-engine × 32 B point
    # (268 → 345 W nominal) is capped and dominated, so the card-scope front is 128 × 128 B only (as package scope)
    assert [(x["engines"], x["noc_link_Bpc"]) for x in fr] == [(128, 128.0)]
    m = next(x for x in rows if (x["engines"], x["noc_link_Bpc"]) == (2048, 32.0))
    assert m["capped"] and m["thr"] < fr[0]["thr"]
    # the trade-off with the same headroom as ≤ 0.332's 300 W (380 W): the narrowest link on the front sits on more
    # (smaller) engines than the widest one
    r3 = _dse(noc_links=(32, 8, 128), power_scope="card", power_W=380)
    fr = [r3["rows"][i] for i in r3["front"]]
    lo, hi = min(fr, key=lambda x: x["noc_link_Bpc"]), max(fr, key=lambda x: x["noc_link_Bpc"])
    assert lo["noc_link_Bpc"] < hi["noc_link_Bpc"] and lo["engines"] > hi["engines"]
    assert any("NoC 链路宽度轴" in n and "Park" in n for n in r["notes"])
    # without the axis nothing changes: no NoC cost terms
    r0 = _dse(noc_link_Bpc=32)
    assert all("noc_W" not in x for x in r0["rows"]) and r0["noc_links"] is None
    for bad in ({"noc_links": ()}, {"noc_links": (0,)}, {"noc_links": (8,), "noc_link_Bpc": 8}):
        try:
            _dse(**bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_api_and_ui_wiring():
    body = {"chip_preset": "1P", "scenario": {"model": M, "serving": {"phase": "decode", "batch": 8}},
            "energy": {"energy_preset": "tpuv4i_7nm"}, "mac_scales": [1], "sram_scales": [1], "noc_links": [16, 64]}
    o = api.api_insight_dse(body)
    assert o["noc_links"] == [16.0, 64.0] and {x["noc_link_Bpc"] for x in o["rows"]} == {16.0, 64.0}
    for bad in ([0], [-1], "x", [], [1] * 7, [True]):
        try:
            api.api_insight_dse({**body, "noc_links": bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass
    try:
        api.api_insight_dse({**body, "noc_link_Bpc": 32})
        raise AssertionError
    except api.ApiError:
        pass
    html, js = W.read(W.INDEX_HTML), W.read()
    ids = W.html_ids(html)
    assert {"c-noc_topology", "c-noc_link_Bpc", "c-noc_routing", "c-noc_mem_attach", "ins-dse-links"} <= ids
    assert W.select_options(html, "c-noc_routing") == ["", "bisection"]
    assert W.select_options(html, "c-noc_mem_attach") == ["", "none"]
    assert W.select_options(html, "c-noc_topology") == ["", "mesh", "ring", "crossbar"]
    assert "Blackhole" in W.label_text(html, "c-noc_mem_attach")
    assert W.has(js, "'clk_ungated_share', 'noc_link_Bpc']") and W.has(js, "k === 'noc_link_Bpc'")
    # init + change (0.307: + noc_eject)
    assert W.count(js, "for (const k of ['noc_topology', 'noc_routing', 'noc_mem_attach', 'noc_eject'])") == 2
    src = W.function_source(js, "runInsightDse")
    for k in ("noc_links", "ins-dse-links", "noc_W", "noc_mm2"):
        assert k in src, k
