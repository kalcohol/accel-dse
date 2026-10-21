"""0.309: fair monolithic-vs-split comparison — a monolithic die split into K on-die NoC clusters (hierarchical NoC:
cluster tier = on-die global links, wire / router energy, no D2D PHY), memory attach on every die edge the HBM stacks
occupy, verdicts against the best monolithic die (verdict_flat = 0.308), and the self-contained example script."""
import math
import os
import subprocess
import sys
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import (Budget, apply_preset, d2d_shoreline, full_area, hbm_mem_edges, phy_area,
                                   shoreline_check, whole_die)
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.energy_dse import energy_design_search
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.parallel import Layout
from accel_dse.core.scenario import CLUSTER_UNLIMITED_GBps, Scenario, Serving
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
H8 = "hbm3e_8s_8h24g_9200"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _t():
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm"}))


def _b(**kw):
    return Budget(**apply_preset({"density_preset": "n5_derived", **kw}))


def _scn(c=None, mem=H8, phase="decode", batch=64, **kw):
    return Scenario(model=M, chip=c or CHIPS["1P"], mem_id=mem,
                    serving=Serving(phase=phase, batch=batch, prompt=2048, ctx=4096), **kw)


def _clu(k=2, link=0.0, clusters=True):
    c = replace(CHIPS["1P"], engines=32 // k, sram_mib=256.0 / k, noc_link_Bpc=link,
                noc_clusters=k if clusters else 1)
    return _scn(c, mem=f"hbm3e_{8 // k}s_8h24g_9200", layout=Layout(tp=k), package_cards=k, d2d_enabled=True)


# ---------------------------------------------------------------- hardware: clusters and memory edges
def test_chip_fields_validate():
    c = CHIPS["1P"]
    assert c.noc_clusters == 1 and c.noc_mem_edges == 2
    for kw in ({"noc_clusters": 0}, {"noc_clusters": 65}, {"noc_clusters": True}, {"noc_mem_edges": 1},
               {"noc_mem_edges": 5}):
        try:
            replace(c, **kw)
            raise AssertionError(kw)
        except ValueError:
            pass
    try:                                        # a cluster chip needs the K-card on-die tier
        _scn(replace(c, engines=16, noc_clusters=2))
        raise AssertionError
    except ValueError:
        pass


def test_cluster_memory_geometry_is_the_whole_die():
    flat = replace(CHIPS["1P"], noc_link_Bpc=32.0)
    for k in (2, 4):
        ck = replace(flat, engines=32 // k, noc_clusters=k)
        assert ck.noc_mem_hops() == flat.noc_mem_hops()
        assert abs(ck.noc_mem_links() * k - flat.noc_mem_links()) < 1e-12
    assert flat.noc_mem_links() == 2 * math.isqrt(32)               # 0.305 (two memory columns) unchanged


def test_mem_edges_geometry():
    c = replace(CHIPS["1P"], noc_link_Bpc=32.0)                       # 32 engines: R 5, C 7
    assert c.noc_mem_links() == 10
    assert replace(c, noc_mem_edges=3).noc_mem_links() == 10 + 7
    assert replace(c, noc_mem_edges=4).noc_mem_links() == 10 + 14
    h2, h3, h4 = (replace(c, noc_mem_edges=m).noc_mem_hops()[0] for m in (2, 3, 4))
    assert h2 > h3 > h4 > 0
    exp4 = sum(min(cc, 8 - cc, r, 6 - r) for r in range(1, 6) for cc in range(1, 8)) / 35
    assert abs(h4 - exp4) < 1e-12
    ring = replace(c, noc_topology="ring")
    assert replace(ring, noc_mem_edges=4).noc_mem_links() == 8 and ring.noc_mem_links() == 4
    xb = replace(c, noc_topology="crossbar", noc_mem_edges=4).noc_mem_hops()
    assert xb[0] == 1.0 and abs(xb[1] - exp4) < 1e-12


def test_hbm_mem_edges_from_shoreline():
    s = _scn()                                                       # 8 stacks × 11 mm = 88 mm
    assert hbm_mem_edges(s, 654.0) == 4 and hbm_mem_edges(s, 908.0) == 3 and hbm_mem_edges(s, 3000.0) == 2
    assert hbm_mem_edges(s, None) == 2 and hbm_mem_edges(s, 100.0) == 2      # "over": unchanged
    two = _clu(2, clusters=False)                                     # 4 stacks, one D2D edge → ≤ 3
    assert hbm_mem_edges(two, 333.0) == 3


# ---------------------------------------------------------------- scenario / area / energy for clusters
def test_cluster_tier_link_and_area():
    s = _clu(2)
    assert s.d2d_link.GBps == CLUSTER_UNLIMITED_GBps                  # no link width: unlimited
    s32 = _clu(2, link=32.0)
    assert abs(s32.d2d_link.GBps - s32.chip.noc_d2d_links(1) * 32 * s32.chip.freq_ghz) < 1e-9
    b = _b()
    flat = full_area(_scn(), b)
    fa = full_area(s, b)
    assert abs(fa["mm2"] - flat["mm2"]) < 1e-9 and fa.get("clusters") == 2 and "d2d_phy" not in fa["terms"]
    assert "d2d_phy" not in phy_area(s, b)["terms"]
    assert d2d_shoreline(s, fa["mm2"]) is None
    assert shoreline_check(s, fa["mm2"])["d2d_edges"] == 0
    w = whole_die(s)
    assert w.chip.engines == 32 and w.package_cards == 1 and not w.d2d_enabled and w.mem_id == H8


def test_cluster_energy_on_die():
    e = energy_report(evaluate(_clu(2)), _t())
    n = e["noc"]["d2d"]
    assert n["on_die"] and n["J"] > 0 and not e["J_by_action"].get("d2d")
    sp = energy_report(evaluate(_clu(2, clusters=False)), _t())
    assert sp["J_by_action"]["d2d"] > 0 and not sp["noc"]["d2d"].get("on_die")


# ---------------------------------------------------------------- DSE: cluster rows and re-derived verdicts
def _dse(**kw):
    a = dict(mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,), dies=(1, 2, 4), d2d_stds=("ucie-a-48",))
    a.update(kw)
    ph = a.pop("phase", "decode")
    return energy_design_search(_scn(phase=ph, batch=64 if ph == "decode" else 1), _t(), _b(power_W_card=700), **a)


def test_dse_cluster_rows():
    o = _dse()
    rows = o["rows"]
    assert len(rows) == 1 + 2 + 2
    cl = [x for x in rows if x["dies"] == 1 and x["clusters"] > 1]
    assert [x["clusters"] for x in cl] == [2, 4] and all(x["feasible"] for x in cl)
    flat = rows[0]
    assert flat["clusters"] == 1 and flat["mem_edges"] == 4 and flat["base"]
    for x in cl:
        assert abs(x["mm2"] - flat["mm2"]) < 1e-9 and x["si_mm2"] == x["mm2"] and "d2d_phy_mm2" not in x
        assert x.get("d2d_std") is None and not x["base"] and x["W_card"] <= 700 * (1 + 1e-6)
    assert any("0.309" in n for n in o["notes"])
    o0 = _dse(mono_clusters=False, mem_edges_shoreline=False)
    assert len(o0["rows"]) == 3 and all("mem_edges" not in x for x in o0["rows"])


def test_dse_verdicts_old_to_new():
    # decode, link width 32 B/cycle: 0.308 split_both (two memory columns on the mono die) → 0.309 mono
    old = _dse(noc_link_Bpc=32.0, noc_routing="xy", mono_clusters=False, mem_edges_shoreline=False)
    new = _dse(noc_link_Bpc=32.0, noc_routing="xy")
    assert old["chiplet"]["compare"][0]["verdict"] == "split_both"
    e = new["chiplet"]["compare"][0]
    assert e["verdict"] == "mono" and e["verdict_flat"] == "mono"
    # prefill, 64 B/cycle: the edge rule alone leaves split_both; the 4-cluster mono die closes it
    p = _dse(phase="prefill", noc_link_Bpc=64.0, noc_routing="xy")
    e = p["chiplet"]["compare"][0]
    assert e["verdict_flat"] == "split_both" and e["verdict"] == "mono"
    assert p["rows"][e["mono_thr"]]["clusters"] == 4 and e["d_thr"] <= 0.01
    # MAC ×2 stays split-only (908 mm² > reticle, clusters do not help)
    q = _dse(mac_scales=(2.0,))
    assert q["chiplet"]["compare"][0]["verdict"] == "only_split"
    assert all(not x["feasible"] for x in q["rows"] if x["dies"] == 1)


# ---------------------------------------------------------------- API / UI / script
def test_api_and_ui():
    body = {"chip_preset": "1P", "scenario": {"model": M, "mem_id": H8}, "energy": {"energy_preset": "tpuv4i_7nm"},
            "budget": {"density_preset": "n5_derived"}, "mac_scales": [1], "sram_scales": [1], "sides": [128],
            "dies": [1, 2], "d2d_stds": ["ucie-a-48"]}
    o = api.api_insight_dse(body)
    assert any(x.get("clusters") == 2 for x in o["rows"])
    o0 = api.api_insight_dse({**body, "mono_clusters": False, "mem_edges_shoreline": False})
    assert not any(x.get("clusters", 1) > 1 for x in o0["rows"])
    for bad in ({"mono_clusters": 1}, {"mem_edges_shoreline": "yes"}):
        try:
            api.api_insight_dse({**body, **bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass
    html, js = W.read(W.INDEX_HTML), W.read()
    assert "ins-dse-clusters" in W.html_ids(html)
    src = W.function_source(js, "runInsightDse")
    for k in ("ins-dse-clusters", "ex.mono_clusters", "ex.mem_edges_shoreline", "verdict_flat", "簇"):
        assert k in src, k


def test_example_script_runs():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "chiplet_example.py"), "--case", "0"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    assert "verdict=mono" in r.stdout and "mono×2簇" in r.stdout
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "chiplet_example.py"), "--flat", "--case", "0"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0 and "verdict=split_J" in r.stdout and "簇" not in r.stdout
