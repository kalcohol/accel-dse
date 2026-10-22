"""0.312: shoreline feasibility as a first-class result (per-edge report on every evaluation / DSE row, unbuildable
points kept off the fronts unless include_unbuildable), the host + scale-up I/O edge sized from lane counts × a
sourced PHY shoreline per lane (noc_mem_io_edges = 1 reproduces 0.311, 0 reproduces 0.310), the per-part scale-up link
of split / clustered packages, buildable-only chiplet verdicts, and test_core_238 on CPU time."""
import importlib.util
import os
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import (DIE_ASPECT_MAX, RETICLE_FIELD_MM, discrete_fit, io_need, mem_sides_max,
                                   shoreline_report)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, Chip, Link
from accel_dse.core.scenario import Layout, Scenario, Serving
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H8 = "hbm3e_8s_8h24g_9200"


def _s(chip="1P", mem=H8, tp=1, link=400.0, model="qwen3-8b", **ck):
    ck.setdefault("io_su_lane_Gbps", 100.0)       # 0.313: these tests pin the 0.312 scale-up lane rate (100 G)
    return Scenario(model=model, chip=replace(CHIPS[chip], **ck), mem_id=mem, layout=Layout(tp=tp),
                    link=Link(GBps=link), serving=Serving(phase="decode", batch=8, prompt=2048, ctx=4096))


def _ex():
    spec = importlib.util.spec_from_file_location("chiplet_example", os.path.join(ROOT, "scripts", "chiplet_example.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- (2) sized I/O edge
def test_io_need_lanes_times_sourced_lane_edge():
    c = CHIPS["1P"]
    assert (c.noc_mem_io_edges, c.io_host_lanes, c.io_lane_mm, c.io_su_lane_mm) == (-1, 16, 0.285, 0.0)   # 0.313: 0 = by rate
    assert c.io_su_lane_Gbps == 200.0 and CHIPS["H100-like"].io_su_lane_Gbps == 100.0     # 0.313 defaults
    n = io_need(_s("H100-like", tp=8, link=450.0))
    # PCIe x16 + NVLink4-like 450 GB/s / dir ÷ 100 Gb/s = 36 lanes (18 links × 2 「二手」) × 0.285 mm (Intel Gen5 SerDes)
    assert (n["host_lanes"], n["scaleup_lanes"]) == (16, 36) and abs(n["need_mm"] - 14.82) < 1e-9
    one = io_need(_s())                                            # one card: no scale-up SerDes on the die
    assert (one["host_lanes"], one["scaleup_lanes"]) == (16, 0) and abs(one["need_mm"] - 4.56) < 1e-9
    assert io_need(_s(tp=8, link=900.0))["scaleup_lanes"] == 72
    assert all(abs(m - 4 * 0.285) < 1e-12 for m in one["macros_mm"]) and len(one["macros_mm"]) == 4   # x4 macros


def test_sized_default_vs_reproduce_switches():
    s = _s()
    assert discrete_fit(s) and not discrete_fit(_s(noc_mem_io_edges=1)) and not discrete_fit(_s(noc_mem_io_edges=0))
    assert mem_sides_max(s) == 4 and mem_sides_max(_s(noc_mem_io_edges=1)) == 3 and mem_sides_max(_s(noc_mem_io_edges=0)) == 4
    r = shoreline_report(s)                                        # 1P + 8 × HBM3E: 2 whole stacks per 25.6 mm edge
    assert r["io_mode"] == "sized" and r["feasible"] and r["status"] == "tight"
    assert [e["mem_units"] for e in r["edges"]] == [2, 2, 2, 2] and r["io"]["status"] == "ok"
    assert sum(e["io_mm"] for e in r["edges"]) == r["io"]["need_mm"] and r["io"]["unplaced_macros"] == 0
    for e in r["edges"]:
        assert abs(e["used_mm"] + e["free_mm"] - e["len_mm"]) < 1e-9 and e["used_mm"] <= e["len_mm"] + 1e-9
    r1 = shoreline_report(_s(noc_mem_io_edges=1))                  # 0.311: whole I/O edge → 8 stacks over
    assert r1["io_mode"] == "edge" and not r1["feasible"] and r1["io"]["status"] == "reserved" and r1["io"]["fits_edge"]
    assert r1["reasons"][0].startswith("HBM 岸线：8 堆栈需 88 mm")
    r0 = shoreline_report(_s(noc_mem_io_edges=0))
    assert r0["io_mode"] == "none" and r0["feasible"] and r0["io"]["status"] == "off"
    hot = shoreline_report(_s(tp=8, link=900.0))                   # 16 + 72 lanes = 25.1 mm > 16.7 mm left
    assert not hot["feasible"] and hot["io"]["status"] == "over" and hot["io"]["unplaced_macros"] == 10
    assert hot["reasons"][0].startswith("I/O 岸线：主机 16 + scale-up 72 通道需 25.1 mm")
    assert shoreline_report(_s(tp=8, link=400.0))["feasible"]
    assert shoreline_report(_s("100T"))["status"] == "over"        # 100T die cannot hold 8 stacks
    assert RETICLE_FIELD_MM == (26.0, 33.0) and DIE_ASPECT_MAX == 2.0


def test_chip_fields_serialise_and_validate():
    d = CHIPS["1P"].to_dict() if hasattr(Chip, "to_dict") else None
    if d is not None:
        assert "noc_mem_io_edges" not in d and "io_host_lanes" not in d          # omit-default
        d1 = replace(CHIPS["1P"], noc_mem_io_edges=1, io_host_lanes=32).to_dict()
        assert d1["noc_mem_io_edges"] == 1 and d1["io_host_lanes"] == 32
    for bad in ({"noc_mem_io_edges": 2}, {"io_host_lanes": -1}, {"io_lane_mm": 0}, {"io_su_lane_Gbps": 0}, {"io_su_lane_mm": -1}):
        try:
            replace(CHIPS["1P"], **bad)
            raise AssertionError(bad)
        except ValueError:
            pass


# ---------------------------------------------------------------- (1) first-class result: eval / API / DSE
def test_result_and_api_carry_shoreline():
    ev = evaluate(_s(tp=8, link=900.0))
    assert ev.shoreline["status"] == "over"
    o = api.api_eval({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": H8, "layout": {"tp": 8},
                                                         "link": {"GBps": 900}, "chip": {"io_su_lane_Gbps": 100}},
                      "sensitivity": False})
    sh = o["shoreline"]
    assert sh["feasible"] is False and len(sh["edges"]) == 4
    assert any(w.startswith("不可建（岸线）：I/O 岸线") for w in o["summary"]["warnings"])


def test_dse_rows_buildable_and_front_switch():
    ex = _ex()
    s = _s(tp=8, link=900.0)
    s = s.replace("serving", Serving(phase="prefill", batch=1, prompt=2048, ctx=4096))   # 100 G lanes (0.312)
    from accel_dse.core.energy_dse import energy_design_search
    kw = dict(mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,), clusters=[1, 2, 4], noc_link_Bpc=64, noc_routing="xy")
    o = energy_design_search(s, ex.table(), ex.budget(), **kw)
    assert o["n_unbuildable"] == 3 and o["front"] == [] and not o["include_unbuildable"]
    assert all(x["buildable"] is False and x["why"].startswith("I/O 岸线") for x in o["rows"])
    # the package keeps its scale-up bandwidth: K clusters share 900 GB/s → 72 lanes for every K (≤ 0.311: K × 72)
    assert all("scale-up 72 通道" in x["why"] for x in o["rows"])
    oi = energy_design_search(s, ex.table(), ex.budget(), include_unbuildable=True, **kw)
    assert oi["include_unbuildable"] and oi["front"] and all("why_shore" in x for x in oi["rows"])
    ok = energy_design_search(s.replace("link", Link(GBps=400.0)), ex.table(), ex.budget(), **kw)
    assert ok["n_unbuildable"] == 0 and ok["front"] and all(x["buildable"] for x in ok["rows"])
    assert all(x["shore"]["io_mm"] > 0 and x["shoreline"] in ("ok", "tight") for x in ok["rows"])
    o2 = api.api_insight_dse({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": H8, "layout": {"tp": 8},
                                                                "link": {"GBps": 900}, "chip": {"io_su_lane_Gbps": 100}},
                              "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived"},
                              "mac_scales": [1], "sram_scales": [1], "sides": [128], "include_unbuildable": True})
    assert o2["include_unbuildable"] and o2["n_unbuildable"] >= 1


def test_chiplet_verdicts_buildable_only():
    ex = _ex()
    lines = []
    if True:
        fl = {"preset": "n5_cowos_l", "asm_model": "flat"}    # 0.319: pinned $ are the 0.312 (flat) rule
        o = ex.run("qwen3-8b", "decode", out=lambda *a: None, cost=fl, d0=0.1, macs=(1.0,))
        o9 = ex.run("qwen3-8b", "decode", out=lambda *a: None, cost="n5_cowos_l", d0=0.1, macs=(1.0,), tp=8, link=900.0,
                    lane=100)
        o1 = ex.run("qwen3-8b", "decode", out=lambda *a: None, cost="n5_cowos_l", d0=0.1, macs=(1.0,), io_edges=1)
    mono = [x for x in o["rows"] if x["dies"] == 1 and x.get("clusters", 1) == 1][0]
    assert mono["buildable"] and mono["mem_edges"] == 4 and round(mono["usd_good_pkg"]) == 2013
    best4 = min(x["usd_good_pkg"] for x in o["rows"] if x["dies"] == 4 and x["buildable"])
    assert round(best4) == 2059 and o["chiplet"]["compare"][0]["verdict"] == "mono"
    assert o1["chiplet"]["compare"][0]["verdict"] == "only_split"           # 0.311 whole I/O edge
    m9 = [x for x in o9["rows"] if x["dies"] == 1 and x.get("clusters", 1) == 1][0]
    assert m9["buildable"] is False and "I/O 岸线" in m9["why"]
    assert any(x["dies"] == 4 and x["buildable"] for x in o9["rows"])      # 4 dies × 18 lanes fit
    assert o9["chiplet"]["compare"][0]["verdict"] == "only_split"
    ex.run("qwen3-8b", "decode", out=lines.append, cost="n5_cowos_l", d0=0.1, macs=(1.0,), tp=8, link=900.0, lane=100)
    assert any(ln.rstrip().endswith("I/O 岸线：主机 16 + scale-up 72 通道需 25.1 mm，外存 / D2D 之外剩余 16.7 mm 放不下 10 个 x4 宏")
               or " b I/O 岸线" in ln for ln in lines)


# ---------------------------------------------------------------- UI / (3) 238 on CPU time
def test_ui_and_238_cpu_time():
    html, js = W.read(W.INDEX_HTML), W.read()
    for k in ('id="c-noc_mem_io_edges"', 'id="c-io_host_lanes"', 'id="chip-shore"', 'id="ins-dse-unbuild"'):
        assert k in html, k
    assert "renderShore" in js and "include_unbuildable" in js
    t = open(os.path.join(ROOT, "tests", "test_core_238.py"), encoding="utf-8").read()
    assert "process_time" in t and "60" in t
