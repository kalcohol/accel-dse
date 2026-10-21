"""0.308: chiplet partitioning in the energy-aware DSE (dies per package × D2D grade as axes, TP / PP / EP split over
the D2D tier, reticle / interposer / yield limits, monolithic-vs-split verdicts); per-grade D2D PHY footprint; D2D
consistent with the NoC (edge attach: bandwidth cap, NoC hop energy, two D2D edges for ≥ 3 dies)."""
import math
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import Budget, apply_preset, d2d_shoreline, full_area, phy_area, shoreline_check
from accel_dse.core.d2d_catalog import D2D_PHY_GEOM, D2D_STANDARDS, d2d_edge_mm, d2d_edges, d2d_phy_mm2
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.energy_dse import (PKG_LIMITS, RETICLE_MM2, _mem_split, _split_layout, energy_design_search)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
H8 = "hbm3e_8s_8h24g_9200"


def _t(**kw):
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm", **kw}))


def _b(**kw):
    return Budget(**apply_preset({"density_preset": "n5_derived", **kw}))


def _scn(c=None, mem=H8, phase="decode", batch=64, **kw):
    return Scenario(model=M, chip=c or CHIPS["1P"], mem_id=mem,
                    serving=Serving(phase=phase, batch=batch, prompt=2048, ctx=4096), **kw)


def _two(std="ucie-a-48", n=2, link=0.0):
    c = replace(CHIPS["1P"], engines=32 // n, sram_mib=256.0 / n, noc_link_Bpc=link)
    return _scn(c, mem=f"hbm3e_{8 // n}s_8h24g_9200", layout=Layout(tp=n), package_cards=n, d2d_enabled=True, d2d_std=std)


# ---------------------------------------------------------------- catalog: PHY footprint per grade
def test_phy_geometry_and_area():
    assert abs(d2d_phy_mm2("ucie-a-48") - 0.3888 * 1.043) < 1e-12 and abs(d2d_phy_mm2("ucie-s-32") - 0.5715 * 1.54) < 1e-12
    assert abs(d2d_edge_mm("bow-256") - 0.65) < 1e-12 and d2d_phy_mm2("nvlink-c2c") is None
    assert set(D2D_PHY_GEOM) <= set(D2D_STANDARDS)
    # shoreline bandwidth density: UCIe-A @ 32 GT/s would be 256 / 0.3888 = 658 GB/s/mm (the UCIe published figure)
    assert abs(256 / d2d_edge_mm("ucie-a-48") - 658.4) < 0.1
    b = _b()
    a = phy_area(_two("ucie-a-48"), b)["terms"]["d2d_phy"]
    s = phy_area(_two("ucie-s-32"), b)
    assert abs(a - 4 * b.mm2_per_d2d_unit) < 1e-12               # UCIe-A: the density field (preset 「来源」)
    assert abs(s["terms"]["d2d_phy"] - 4 * d2d_phy_mm2("ucie-s-32")) < 1e-12 and any("ucie-s-32" in n for n in s["notes"])
    assert "d2d_phy" not in phy_area(_two("nvlink-c2c"), b)["terms"]


def test_d2d_edges_and_shoreline():
    assert [d2d_edges(n) for n in (1, 2, 3, 4, 8)] == [0, 1, 2, 2, 2]
    s4 = _two(n=4)
    sh = shoreline_check(s4, 300.0)
    assert sh["d2d_edges"] == 2 and sh["sides_max"] == 2
    assert shoreline_check(_two(n=2), 300.0)["d2d_edges"] == 1
    d = d2d_shoreline(s4, 300.0)
    assert d["edges"] == 2 and abs(d["need_mm"] - 4 * 0.3888) < 1e-12 and d["status"] == "ok"
    assert d2d_shoreline(replace(s4, d2d_units=64), 4.0)["status"] == "over"
    assert d2d_shoreline(_scn(), 300.0) is None


# ---------------------------------------------------------------- (2) D2D ↔ NoC edge attach
def test_d2d_bandwidth_capped_by_noc_edge():
    s = _two(link=32.0)                       # 16 engines / die: 4 × 4 mesh, 4 attach links on the D2D edge
    assert s.chip.noc_d2d_links(1) == 4 and abs(s.d2d_link.GBps - 4 * 32 * s.chip.freq_ghz) < 1e-9
    assert _two(n=4, link=32.0).d2d_link.GBps == 2 * 4 * 32 * s.chip.freq_ghz       # 8 engines: R 2, C 4; 2 edges
    assert _two().d2d_link.GBps == 1536.0                     # no link width: catalog figure (bit-identical)
    assert replace(s, chip=replace(s.chip, noc_mem_attach="none")).d2d_link.GBps == 1536.0
    ring = replace(s, chip=replace(s.chip, noc_topology="ring"))
    assert ring.chip.noc_d2d_links(1) == 2
    assert evaluate(s).per_card < evaluate(_two()).per_card    # the cap binds the TP collectives


def test_d2d_bytes_cross_the_noc():
    r = evaluate(_two())
    e = energy_report(r, _t())
    nd = e["noc"]["d2d"]
    R = math.isqrt(16)
    assert abs(nd["hops_per_byte"] - 2 * (R + 1) / 2) < 1e-12 and nd["J"] > 0
    mono = energy_report(evaluate(_scn()), _t())
    assert "d2d" not in mono["noc"]


# ---------------------------------------------------------------- (1) chiplet axes in the energy DSE
def _dse(**kw):
    a = dict(mac_scales=(1.0, 2.0), sram_scales=(1.0,), sides=(128,), dies=(1, 2, 4),
             d2d_stds=("ucie-a-48", "ucie-s-32"))
    a.update(kw)
    return energy_design_search(_scn(phase=a.pop("phase", "decode")), _t(), _b(power_W_card=700), **a)


def test_dse_split_package_figures():
    o = _dse(mono_clusters=False, mem_edges_shoreline=False)      # the 0.308 view (0.309 rows: test_core_309)
    rows = o["rows"]
    assert len(rows) == 2 * (1 + 2 * 2) and o["chiplet"]["reticle_mm2"] == RETICLE_MM2
    x = next(r for r in rows if r["mac_x"] == 1.0 and r["dies"] == 2 and r["d2d_std"] == "ucie-a-48")
    s = _two()
    r = evaluate(s)
    assert abs(x["thr"] - 2 * r.per_card) < 1e-6 * x["thr"]                   # package = per die × dies
    assert abs(x["J_per_unit"] - energy_report(r, _t())["J_per_unit"]) < 1e-9 * x["J_per_unit"]
    assert abs(x["si_mm2"] - 2 * x["mm2"]) < 1e-9 and x["d2d_phy_mm2"] > 0 and x["d2d_GBps"] == 1536.0
    u = next(r for r in rows if r["mac_x"] == 1.0 and r["dies"] == 2 and r["d2d_std"] == "ucie-s-32")
    assert u["mm2"] > x["mm2"]                                           # UCIe-S modules are larger
    big = next(r for r in rows if r["mac_x"] == 2.0 and r["dies"] == 1)
    assert not big["feasible"] and "光罩" in big["why"]
    v = {(e["mac_x"]): e for e in o["chiplet"]["compare"]}
    assert v[2.0]["verdict"] == "only_split" and v[2.0]["best_thr"] is not None
    assert v[1.0]["verdict"] in ("mono", "split_J", "split_thr", "split_both")
    assert any(rows[i].get("dies", 1) > 1 for i in o["front"])


def test_dse_noc_edge_split_beats_monolithic():
    # narrow NoC links: DRAM ingress per die ∝ its edge (2·⌊√E⌋ links) — two half dies have more edge than one
    # 0.308 view: two memory columns on the monolithic die (0.309: HBM attach on every occupied edge → mono)
    o = _dse(mac_scales=(1.0,), d2d_stds=("ucie-a-48",), noc_link_Bpc=32.0, noc_routing="xy",
             mono_clusters=False, mem_edges_shoreline=False)
    e = o["chiplet"]["compare"][0]
    assert e["verdict"] in ("split_thr", "split_both") and e["d_thr"] > 0.3
    sp = o["rows"][e["best_thr"]]
    assert sp["d2d_noc_capped"] and sp["d2d_GBps"] < 1536.0


def test_dse_yield_package_and_power():
    o = _dse(mac_scales=(1.0,), d2d_stds=("ucie-a-48",), d0_per_cm2=0.1, pkg_mm2="cowos-s")
    for x in o["rows"]:
        y = math.exp(-x["mm2"] * 0.1 / 100)
        assert abs(x["die_yield"] - y) < 1e-12 and abs(x["kgd_mm2"] - x["dies"] * x["mm2"] / y) < 1e-6
        assert x["pkg_mm2"] > x["si_mm2"]                    # + 8 HBM3E stack footprints
    mono = next(x for x in o["rows"] if x["dies"] == 1)
    four = next(x for x in o["rows"] if x["dies"] == 4)
    assert four["die_yield"] > mono["die_yield"] and four["kgd_mm2"] < mono["kgd_mm2"]
    assert any(x.get("front_kgd") for x in o["rows"]) and PKG_LIMITS["cowos-s"][0] == 2700.0
    o2 = _dse(mac_scales=(1.0,), d2d_stds=("ucie-a-48",), pkg_mm2=1000.0)
    assert all("中介层" in x["why"] for x in o2["rows"])
    # the package power budget is split over the dies (each die under cap / n)
    o3 = energy_design_search(_scn(), _t(), _b(power_W_card=200), mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,),
                              dies=(1, 2), d2d_stds=("ucie-a-48",))
    for x in o3["rows"]:
        assert x["W_card"] <= 200 * (1 + 1e-6)


def test_split_layouts_and_memory():
    assert _split_layout(Layout(), 2, "tp", False) == Layout(tp=2)
    assert _split_layout(Layout(), 2, "tp", True) == Layout(tp=2, etp=2)
    assert _split_layout(Layout(), 4, "pp", False) == Layout(pp=4)
    assert _split_layout(Layout(), 2, "ep", True) == Layout(dp=2, ep=2)
    try:
        _split_layout(Layout(), 2, "ep", False)
        raise AssertionError
    except ValueError:
        pass
    assert _mem_split(H8, 4) == "hbm3e_2s_8h24g_9200" and _mem_split("hbm3e_3s_8h24g_9200", 2) is None
    o = energy_design_search(_scn(mem="hbm3e_6s_8h24g_9200"), _t(), _b(), mac_scales=(1.0,), sram_scales=(1.0,),
                             sides=(128,), dies=(1, 4))
    assert "不能平分" in next(x for x in o["rows"] if x["dies"] == 4)["why"]
    m = energy_design_search(Scenario(model="qwen3-30b-a3b", chip=CHIPS["1P"], mem_id=H8,
                                      serving=Serving(phase="decode", batch=64, ctx=4096)), _t(), _b(),
                             mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,), dies=(1, 2), die_split="ep")
    assert all(x["feasible"] for x in m["rows"]) and m["rows"][1]["die_split"] == "ep"


def test_dse_validation_and_default_unchanged():
    o = energy_design_search(_scn(), _t(), _b(), mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,))
    assert o["chiplet"] is None and "dies" not in o["rows"][0]
    for kw in ({"d2d_stds": ("ucie-a-48",)}, {"dies": (0,)}, {"dies": (2,), "d2d_stds": ("x",)},
               {"dies": (2,), "die_split": "dp"}, {"dies": (2,), "pkg_mm2": "foo"}, {"dies": (2,), "d0_per_cm2": 0}):
        try:
            energy_design_search(_scn(), _t(), _b(), mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,), **kw)
            raise AssertionError(kw)
        except ValueError:
            pass
    try:
        energy_design_search(_two(), _t(), _b(), dies=(2,))
        raise AssertionError
    except ValueError:
        pass


# ---------------------------------------------------------------- API / UI
def test_api_and_ui():
    body = {"chip_preset": "1P", "scenario": {"model": M, "mem_id": H8}, "energy": {"energy_preset": "tpuv4i_7nm"},
            "budget": {"density_preset": "n5_derived"}, "mac_scales": [2], "sram_scales": [1], "sides": [128],
            "dies": [1, 2], "d2d_stds": ["bow-256"], "d2d_units": 8, "die_split": "tp", "pkg_mm2": "cowos-l",
            "d0_per_cm2": 0.1}
    o = api.api_insight_dse(body)
    assert o["chiplet"]["d2d_units"] == 8 and o["chiplet"]["compare"][0]["verdict"] == "only_split"
    for bad in ({"dies": [0]}, {"d2d_stds": ["x"]}, {"die_split": "dp"}, {"pkg_mm2": "x"}, {"d0_per_cm2": 20}):
        try:
            api.api_insight_dse({**body, **bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass
    html, js = W.read(W.INDEX_HTML), W.read()
    ids = W.html_ids(html)
    assert {"ins-dse-dies", "ins-dse-d2d", "ins-dse-d2dunits", "ins-dse-split", "ins-dse-reticle", "ins-dse-pkg",
            "ins-dse-d0"} <= ids
    assert W.select_options(html, "ins-dse-split") == ["tp", "pp", "ep"]
    assert W.select_options(html, "ins-dse-pkg") == ["", "cowos-s", "cowos-l"]
    src = W.function_source(js, "runInsightDse")
    for k in ("ex.dies", "ex.d2d_stds", "ex.d2d_units", "ex.die_split", "ex.reticle_mm2", "ex.pkg_mm2", "ex.d0_per_cm2",
              "r.chiplet", "only_split"):
        assert k in src, k
