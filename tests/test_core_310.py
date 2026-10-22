"""0.310: memory-attach edges by shoreline occupancy as the mechanism default on every path (Chip.noc_mem_edges = 0
= auto, resolved per Scenario; 2 reproduces 0.309), LPDDR beachfront, the NoC-cluster axis in the energy DSE, and
chiplet economics (cost per good package, cost front)."""
import math
import os
import subprocess
import sys
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import (LPDDR_PINS_PER_MM, Budget, apply_preset, full_area, hbm_mem_edges, mem_edge_need_mm,
                                   mem_edges_auto)
from accel_dse.core.energy import EnergyTable, apply_energy_preset, energy_report
from accel_dse.core.energy_dse import (COST_PRESETS, _cost_params, _hbm_footprint, dies_per_wafer, energy_design_search,
                                       hbm_stacks, package_cost)

FLAT_L = {"preset": "n5_cowos_l", "asm_model": "flat"}       # 0.319: the 0.310–0.318 cost rule (asm_yield 1)
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS, preset_die_mm2
from accel_dse.core.scenario import Scenario, Serving
from accel_dse import mem_catalog as mc
from tests import webstruct as W  # noqa: E402

M = "qwen3-8b"
H8 = "hbm3e_8s_8h24g_9200"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _t():
    return EnergyTable(**apply_energy_preset({"energy_preset": "tpuv4i_7nm"}))


def _b(**kw):
    return Budget(**apply_preset({"density_preset": "n5_derived", **kw}))


# 0.311: these tests pin the 0.310 rule (no host-I/O edge kept, pooled attach links) — the 0.311 switch values
V310 = {"noc_mem_io_edges": 0, "noc_mem_balance": "pooled"}


def _scn(chip="1P", mem=H8, phase="decode", batch=64, **ck):
    c = replace(CHIPS[chip], **{**V310, **ck})
    return Scenario(model=M, chip=c, mem_id=mem, serving=Serving(phase=phase, batch=batch, prompt=2048, ctx=4096))


# ---------------------------------------------------------------- (1) auto memory edges everywhere
def test_chip_field_auto_default_and_serialisation():
    c = replace(CHIPS["1P"], **V310)
    assert c.noc_mem_edges == 0 and c.mem_edges == 2              # a bare chip: two columns
    for v in (1, 5, True):
        try:
            replace(c, noc_mem_edges=v)
            raise AssertionError(v)
        except ValueError:
            pass
    s = _scn()
    assert s.chip.noc_mem_edges_auto == 4 and s.chip.mem_edges == 4
    d = s.to_dict()
    assert "noc_mem_edges" not in d["chip"] and "noc_mem_edges_auto" not in d["chip"] and "noc_mem_load" not in d["chip"]
    assert Scenario.from_dict(d).chip.mem_edges == 4
    try:                                                          # internal: never accepted
        Scenario.from_dict({**d, "chip": {**d["chip"], "noc_mem_edges_auto": 3}})
        raise AssertionError
    except (ValueError, TypeError, KeyError):
        pass


def test_auto_rule_cases():
    assert _scn(mem="hbm3e_4s_8h24g_9200").chip.mem_edges == 2   # 1P default memory: unchanged
    assert _scn("H100-like", "hbm3_5s_8h16g_5234").chip.mem_edges == 2   # 814 mm² published die (proxy says 283)
    assert preset_die_mm2(CHIPS["H100-like"]) == 814.0
    assert preset_die_mm2(replace(CHIPS["H100-like"], engines=32)) is None
    assert _scn("100T", "lpddr5x_8x64_8533_16g").chip.mem_edges == 3     # 512 bits / 18.75 = 27 mm on a 98 mm² die
    assert _scn("100T", "lpddr5x_4x64_8533_16g").chip.mem_edges == 2
    assert _scn(noc_mem_edges=2).chip.mem_edges == 2              # fixed = 0.309
    assert _scn(noc_mem_attach="none").chip.mem_edges == 2
    s = _scn()
    assert s.replace("mem_id", "hbm3e_4s_8h24g_9200").chip.mem_edges == 2     # re-resolved on replace
    assert mem_edges_auto(s) == hbm_mem_edges(s, full_area(s, _b())["mm2"]) == 4


def test_lpddr_beachfront():
    assert abs(LPDDR_PINS_PER_MM - 18.75) < 1e-12
    sp = mc.parse_mem_id("lpddr5x_8x64_8533_16g")
    assert abs(mem_edge_need_mm(sp) - 512 / 18.75) < 1e-12
    assert mem_edge_need_mm(mc.parse_mem_id(H8)) == 88.0
    s = _scn("100T", "lpddr5x_8x64_8533_16g")
    assert hbm_mem_edges(s, 98.0, lpddr=False) == 2 and hbm_mem_edges(s, 98.0) == 3
    assert hbm_mem_edges(s, 1000.0) == 2 and hbm_mem_edges(s, 20.0) == 2      # fits two / beyond four sides


def test_old_to_new_single_eval():
    t = _t()
    out = {}
    for me in (2, 0):
        for L in (0.0, 32.0):
            r = evaluate(_scn(noc_mem_edges=me, noc_link_Bpc=L))
            e = energy_report(r, t)
            out[me, L] = (r.per_card, e["J_per_unit"], e["noc"]["dram"])
    assert abs(out[0, 0.0][1] / out[2, 0.0][1] - 1 + 0.048) < 0.002            # decode J −4.8 %
    assert out[0, 0.0][0] == out[2, 0.0][0]
    assert out[0, 32.0][0] / out[2, 32.0][0] > 2.3                           # DRAM ingress 10 → 24 links
    assert out[0, 0.0][2]["mem_edges"] == 4 and out[0, 0.0][2]["mem_edges_auto"]
    assert out[2, 0.0][2]["mem_edges"] == 2 and not out[2, 0.0][2]["mem_edges_auto"]


def test_pd_pool_and_api_paths_resolve():
    s = _scn(mem="hbm3e_4s_8h24g_9200")
    ps = s.colocated().replace("mem_id", H8)                       # the PD prefill pool's own memory
    assert ps.chip.mem_edges == 4
    body = {"chip_preset": "1P", "scenario": {"model": M, "mem_id": H8, "chip": dict(V310)},
            "energy": {"energy_preset": "tpuv4i_7nm"}}
    o = api.api_eval(body)
    assert o["energy"]["noc"]["dram"]["mem_edges"] == 4
    o2 = api.api_eval({**body, "scenario": {"model": M, "mem_id": H8, "chip": {**V310, "noc_mem_edges": 2}}})
    assert o2["energy"]["noc"]["dram"]["mem_edges"] == 2
    assert o2["energy"]["J_per_unit"] > o["energy"]["J_per_unit"]


def test_dse_rows_resolve_from_their_die():
    o = energy_design_search(_scn(), _t(), _b(power_W_card=700), mac_scales=(1.0, 2.0), sram_scales=(1.0,),
                             sides=(128,))
    assert [x["mem_edges"] for x in o["rows"]] == [4, 3]           # 654 / 908 mm²
    assert any("0.310" in n for n in o["notes"])
    o2 = energy_design_search(_scn(noc_mem_edges=2), _t(), _b(power_W_card=700), mac_scales=(1.0,),
                              sram_scales=(1.0,), sides=(128,))
    assert o2["rows"][0]["mem_edges"] == 2 and o2["rows"][0]["J_per_unit"] > o["rows"][0]["J_per_unit"]


# ---------------------------------------------------------------- clusters axis in the energy DSE
def test_dse_clusters_axis():
    o = energy_design_search(_scn(phase="prefill", batch=1), _t(), _b(power_W_card=700), mac_scales=(1.0,),
                             sram_scales=(1.0,), sides=(128,), clusters=(1, 2, 4), noc_link_Bpc=64.0)
    rows = o["rows"]
    assert [x["clusters"] for x in rows] == [1, 2, 4] and o["clusters"] == [2, 4] and o["chiplet"] is None
    assert rows[2]["thr"] > 1.4 * rows[0]["thr"]                   # narrow links: 4 clusters +54 %
    assert all(abs(x["mm2"] - rows[0]["mm2"]) < 1e-9 for x in rows)
    for bad in ((0,), (1, 2, 3, 4, 5, 6), (True,)):
        try:
            energy_design_search(_scn(), _t(), _b(), mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,), clusters=bad)
            raise AssertionError(bad)
        except ValueError:
            pass
    c = energy_design_search(_scn(), _t(), _b(), mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,), dies=(1, 2),
                             d2d_stds=("ucie-a-48",), clusters=(4,))
    assert sorted(x["clusters"] for x in c["rows"] if x["dies"] == 1) == [1, 4]


# ---------------------------------------------------------------- (2) chiplet economics
def test_cost_terms():
    assert abs(dies_per_wafer(654.0) - (math.pi * 22500 / 654 - math.pi * 300 / math.sqrt(1308))) < 1e-9
    cp = _cost_params(FLAT_L)
    assert cp["wafer_usd"] == 16988.0 and abs(cp["pkg_usd_per_mm2"] - 1100 / 2568) < 1e-12
    assert abs(COST_PRESETS["n5_cowos_s"]["pkg_usd_per_mm2"] - 750 / 1540) < 1e-12
    m = package_cost(cp, 654.0, 1, 0.52, 654 + 968)
    s4 = package_cost(cp, 171.0, 4, 0.84, 686 + 968)
    assert m["test_usd"] == 920.0 and abs(s4["test_usd"] - 920 * 1.225) < 1e-9
    assert abs(m["usd_good_pkg"] - (m["si_usd"] + m["pkg_usd"] + m["test_usd"])) < 1e-9
    assert abs(m["good_die_usd"] - 16988 / dies_per_wafer(654.0) / 0.52) < 1e-9
    assert _cost_params({"preset": "n5_cowos_s", "wafer_usd": 20000})["wafer_usd"] == 20000.0
    for bad in ("x", {"preset": "x"}, {"foo": 1}, {"asm_yield": 0}, {"wafer_usd": 0}, 5):
        try:
            _cost_params(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def _cdse(d0):
    return energy_design_search(_scn(), _t(), _b(power_W_card=700), mac_scales=(1.0, 2.0), sram_scales=(1.0,),
                                sides=(128,), dies=(1, 2, 4), d2d_stds=("ucie-a-48",), d0_per_cm2=d0,
                                pkg_mm2="cowos-l", cost="n5_cowos_l")


def test_dse_cost_rows_and_front():
    o = _cdse(0.1)
    rows = o["rows"]
    assert o["costed"] and o["cost"]["preset"] == "n5_cowos_l" and any("成本" in n for n in o["notes"])
    for x in rows:
        cc = package_cost(_cost_params("n5_cowos_l"), x["mm2"], x["dies"], x["die_yield"],
                          x["si_mm2"] + _hbm_footprint(H8), hbm_stacks(H8))
        assert abs(x["usd_good_pkg"] - cc["usd_good_pkg"]) < 1e-9
    mono = next(x for x in rows if x["mac_x"] == 1 and x["dies"] == 1 and x["clusters"] == 1)
    clu = next(x for x in rows if x["mac_x"] == 1 and x["dies"] == 1 and x["clusters"] == 2)
    four = next(x for x in rows if x["mac_x"] == 1 and x["dies"] == 4)
    assert clu["usd_good_pkg"] == mono["usd_good_pkg"]                # clusters: same die, same cost
    # 0.329 (MODEL §274): the per-generation HBM3E PHY + controller grows the mono die 654 → 774 mm², moving the
    # mono / 4-die cost crossover from D0 ∈ (0.1, 0.2) to ≈ 0.07 /cm²: at 0.1 the 4-die package is now cheaper
    # ($2232.9 vs $2354.5); the "mono cheaper" leg moves to D0 0.05 ($2133.6 vs $2199.8)
    assert four["usd_good_pkg"] < mono["usd_good_pkg"]
    lo = _cdse(0.05)["rows"]
    m05 = next(x for x in lo if x["mac_x"] == 1 and x["dies"] == 1 and x["clusters"] == 1)
    f05 = next(x for x in lo if x["mac_x"] == 1 and x["dies"] == 4)
    assert m05["usd_good_pkg"] < f05["usd_good_pkg"] and m05.get("front_cost")       # D0 0.05: mono cheaper
    hi = _cdse(0.2)["rows"]
    m2 = next(x for x in hi if x["mac_x"] == 1 and x["dies"] == 1 and x["clusters"] == 1)
    f2 = next(x for x in hi if x["mac_x"] == 1 and x["dies"] == 4)
    assert f2["usd_good_pkg"] < m2["usd_good_pkg"] and f2.get("front_cost")         # D0 0.2: 4 dies cheaper
    # MAC ×2: split-only by reticle, and cheaper than the infeasible monolith would be
    big = next(x for x in rows if x["mac_x"] == 2 and x["dies"] == 1 and x["clusters"] == 1)
    assert not big["feasible"] and big["usd_good_pkg"] > min(x["usd_good_pkg"] for x in rows if x["mac_x"] == 2 and x["dies"] > 1)
    none = energy_design_search(_scn(), _t(), _b(), mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,))
    assert none["cost"] is None and "usd_good_pkg" not in none["rows"][0]


def test_cost_crossover_d0():
    cp = _cost_params(FLAT_L)                                       # the 0.310 crossover (0.319: component ~0.10)
    H = _hbm_footprint(H8)

    def c(n, a, d0):
        return package_cost(cp, a, n, math.exp(-a * d0 / 100), n * a + H)["usd_good_pkg"]
    lo, hi = 0.0, 1.0
    for _ in range(50):
        m = (lo + hi) / 2
        lo, hi = (lo, m) if c(4, 171, m) < c(1, 654, m) else (m, hi)
    assert 0.11 < hi < 0.13                                         # MODEL §255.3


# ---------------------------------------------------------------- API / UI / script
def test_api_and_ui():
    body = {"chip_preset": "1P", "scenario": {"model": M, "mem_id": H8, "chip": dict(V310)},
            "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived"}, "mac_scales": [1], "sram_scales": [1], "sides": [128],
            "clusters": [1, 2], "cost": "n5_cowos_s", "d0_per_cm2": 0.1}
    o = api.api_insight_dse(body)
    assert o["costed"] and [x["clusters"] for x in o["rows"]] == [1, 2]
    for bad in ({"clusters": [0]}, {"cost": "x"}, {"cost": {"foo": 1}}):
        try:
            api.api_insight_dse({**body, **bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass
    html, js = W.read(W.INDEX_HTML), W.read()
    ids = W.html_ids(html)
    assert {"c-noc_mem_edges", "ins-dse-kclu", "ins-dse-cost"} <= ids
    assert W.select_options(html, "c-noc_mem_edges") == ["", "2", "3", "4"]
    assert W.select_options(html, "ins-dse-cost") == ["", "n5_cowos_s", "n5_cowos_l"]
    src = W.function_source(js, "runInsightDse")
    for k in ("ex.clusters", "ex.cost", "usd_good_pkg", "front_cost", "r.costed"):
        assert k in src, k
    assert "S.chipOver.noc_mem_edges = parseInt" in js


def test_example_script_cost():
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "chiplet_example.py"), "--case", "0",
                        "--cost", "n5_cowos_l", "--d0", "0.2", "--io-edges", "0"], capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    import re
    assert re.search(r"\$\s*\d+C", r.stdout)                         # a cost-front row
