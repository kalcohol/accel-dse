"""0.326: SPHBM4 on an organic substrate (Chip.hbm_pkg, core/hbm_pkg.py; MODEL §271) — opt-in, si byte-identical to
0.325; sourced constants; shoreline per stack 11 → 25/3 mm; stacks off the interposer (DSE area limit); everything
without a public number unchanged and listed; HBM4 in the JESD330-4 rate range only.  Plus the exact speed-ups of the
custom-HBM DSE (shore_floor memo, sized_layout prefix cache, pruned shapes, water-fill scan): same results."""
import copy
import json
import math
import os
import random
import subprocess
import sys
from dataclasses import replace

from accel_dse import api
from accel_dse.core import budget as bmod, hbm_base, hbm_pkg
from accel_dse.core.hardware import CHIPS
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HBM4 = "hbm4_8s_12h24g_10000"


def _body(pkg="si", mem=HBM4, edges=None, tco=False):
    ch = {"hbm_pkg": pkg} if pkg != "si" else {}
    if edges:
        ch["noc_mem_edges"] = edges
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": mem, "chip": ch,
         "serving": {"phase": "decode", "batch": 64, "prompt": 2048, "ctx": 4096}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived"}, "sensitivity": False}
    if tco:
        b["tco"] = {"preset": "accel_2026", "cost": "n5_cowos_s", "d0_per_cm2": 0.1}
    return b


def test_sourced_constants():
    assert hbm_pkg.SPHBM4_DQ == 512 and hbm_pkg.SPHBM4_SERDES == 4                       # JEDEC PR 「来源」
    assert hbm_pkg.SPHBM4_GTPS == (22.4, 46.0)                                            # 「二手」
    assert hbm_pkg.SPHBM4_EDGE_MM == 25.0 / 3 and hbm_pkg.SI_STACKS_PER_25MM == 2         # Eliyan 「二手」
    assert hbm_pkg.rate_range_MTps() == (5600.0, 11500.0)                                 # 「推导」
    assert hbm_pkg.SPHBM4_BUMP_PITCH_UM_MIN == 90 and hbm_pkg.SPHBM4_REACH_MM == 20
    # 46 GT/s × 512 / 8 = 2.944 TB/s per stack (the published peak) = HBM4 2048 bits at 11500 MT/s
    assert 512 * 46.0 / 8 == 2944.0 == 2048 * 11500 / 8 / 1000
    assert len(hbm_pkg.NOT_MODELLED) == 6 and all(("无公开" in x or "无可靠" in x) and "——" in x
                                                    for x in hbm_pkg.NOT_MODELLED)       # each: no public number → kept


def test_shoreline_edge_and_si_unchanged():
    a, c = api.api_eval(_body()), api.api_eval(_body("sphbm4"))
    assert "stack_edge_mm" not in a["shoreline"] and "hbm_pkg" not in a                   # si output unchanged
    assert a["shoreline"]["mem"]["unit_mm"] == 11.0 and c["shoreline"]["mem"]["unit_mm"] == 25.0 / 3
    assert c["shoreline"]["mem"]["need_mm"] == 8 * 25.0 / 3
    assert (a["shoreline"]["mem"]["edges"], c["shoreline"]["mem"]["edges"]) == (4, 3)
    assert c["budget"]["shoreline"]["stack_edge_mm"] == 25.0 / 3 and "stack_edge_mm" not in a["budget"]["shoreline"]
    hp = c["hbm_pkg"]
    assert hp["applies"] and hp["stack_GBps"] == 2560.0 and hp["pin_GTps"] == 40.0
    assert (hp["si_mem_edges"], hp["mem_edges"]) == (4, 3) and any("外存接入边 4 → 3" in n for n in hp["notes"])
    assert hp["not_modelled"] == list(hbm_pkg.NOT_MODELLED)
    # 25 mm edge holds exactly 3 stacks (Eliyan) under the discrete-fit rule
    assert bmod._fit_units([25.0], hbm_pkg.SPHBM4_EDGE_MM) == 3 and bmod._fit_units([25.0], 11.0) == 2


def test_energy_unchanged_with_edges_pinned():
    """The package changes geometry only: with the memory edges pinned, energy and throughput are identical — the
    J difference under the auto rule is the fewest-edges placement (fewer NoC attach edges), not SPHBM4."""
    a, c = api.api_eval(_body(edges=4)), api.api_eval(_body("sphbm4", edges=4))
    assert json.dumps(a["energy"]["J_by_action"], sort_keys=True) == json.dumps(c["energy"]["J_by_action"], sort_keys=True)
    assert a["energy"]["J_per_unit"] == c["energy"]["J_per_unit"]


def test_ignored_out_of_scope():
    for mem, frag in (("hbm4_8s_12h24g_11700", "46.8 GT/s 超出 JESD330-4"), ("hbm3e_8s_12h24g_9200", "只对 HBM4")):
        a, c = api.api_eval(_body(mem=mem)), api.api_eval(_body("sphbm4", mem=mem))
        assert not c["hbm_pkg"]["applies"] and any(frag in w for w in c["summary"]["warnings"])
        assert c["shoreline"] == a["shoreline"] and c["energy"]["J_per_unit"] == a["energy"]["J_per_unit"]


def test_exclusive_with_custom_base_die_and_validation():
    for bad in ("organic", "SPHBM4"):
        try:
            replace(CHIPS["1P"], hbm_pkg=bad)
            raise AssertionError(f"hbm_pkg={bad!r} accepted")
        except ValueError:
            pass
    try:
        replace(CHIPS["1P"], hbm_pkg="sphbm4", hbm_base="custom")
        raise AssertionError("sphbm4 + custom accepted")
    except ValueError as e:
        assert "exclusive" in str(e)


def test_eval_package_bound_and_tco_unchanged():
    a, c = api.api_eval(_body(tco=True)), api.api_eval(_body("sphbm4", tco=True))
    from accel_dse.core.energy_dse import _cost_params
    pm = _cost_params("n5_cowos_s")["pkg_usd_per_mm2"]
    assert c["hbm_pkg"]["interposer_freed_mm2"] == 8 * 11.0 * 12.8
    assert math.isclose(c["hbm_pkg"]["pkg_usd_bound"], 8 * 11.0 * 12.8 * pm)
    assert c["tco"]["pkg_usd"] == a["tco"]["pkg_usd"] and c["tco"]["mem_usd"] == a["tco"]["mem_usd"]   # not modelled


def _dse(pkg, mem_scales=(1,)):
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "chip": ({"hbm_pkg": pkg} if pkg != "si" else {}),
         "serving": {"phase": "decode", "batch": 64, "prompt": 2048, "ctx": 4096}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived", "power_W_card": 700},
         "mac_scales": [1], "sram_scales": [1], "topologies": ["mesh"], "mem_types": ["HBM4"],
         "mem_count_scales": list(mem_scales), "slc_mibs": [0], "f_scales": [1], "dies": [1, 2], "d2d_stds": ["ucie-a-48"],
         "pkg_mm2": "cowos-l", "include_unbuildable": True}
    return api.api_insight_dse(b)


def test_dse_interposer_and_shoreline():
    a, c = _dse("si", (1, 1.5)), _dse("sphbm4", (1, 1.5))
    assert "hbm_pkg" not in a and c["hbm_pkg"] == "sphbm4"
    assert any("SPHBM4 有机基板" in n for n in c["notes"])
    for x, y in zip(a["rows"], c["rows"]):
        assert y["hbm_sphbm4"] and y["pkg_mm2"] == y["si_mm2"] < x["pkg_mm2"]
    m12 = [(x, y) for x, y in zip(a["rows"], c["rows"]) if x["dies"] == 1 and "12s" in x["mem_id"]]
    assert m12 and all(x["shoreline"] == "over" and y["shoreline"] == "tight" for x, y in m12)


def _place_ref(ed, dpos, units, links4, macros, d2d_need):
    """The 0.312 – 0.325 water-fill (min with a lambda key) — reference for the 0.326 scan."""
    non = [i for i in range(4) if i not in dpos]
    n, umm = units
    cap = {i: math.floor(ed[i] / umm + 1e-9) for i in non}
    lo = min(2, len(non))
    k = next((j for j in range(lo, len(non) + 1) if sum(cap[i] for i in non[:j]) >= n), len(non))
    mst = "ok" if n <= sum(cap[i] for i in non[:lo]) else ("tight" if n <= sum(cap[i] for i in non[:k]) else "over")
    med, ld = non[:k], [0] * k
    for _ in range(n):
        ok = [j for j in range(k) if mst == "over" or ld[j] < cap[med[j]]]
        j = min(ok, key=lambda j: ((ld[j] + 1) / links4[j], j))
        ld[j] += 1
    return tuple(ld)


def test_place_scan_and_pruning_exact():
    rnd = random.Random(326)
    for _ in range(3000):
        L, S = rnd.uniform(8, 34), rnd.uniform(8, 26)
        ed = [L, L, S, S]
        dp = rnd.choice([(), (3,), (1,), (2, 3), (0, 1)])
        units = (rnd.randint(1, 14), rnd.choice([11.0, 25.0 / 3, 0.85]))
        links4 = [rnd.choice([1, 2, 3, 4, 6, 8]) for _ in range(4)]
        need = rnd.choice([None, rnd.uniform(0, 40)])
        macros = [rnd.choice([1.14, 2.0]) for _ in range(rnd.randint(0, 8))]
        p = bmod._place(ed, dp, units, links4, macros, need)
        assert p["load"] == _place_ref(ed, dp, units, links4, macros, need)
        if bmod._surely_bad(ed, dp, units, need):              # pruning is sound: such a candidate is never chosen
            assert p["bad"] >= 1 and (p["mem_status"] == "over" or p["d2d_status"] == "over")
        else:
            assert p["mem_status"] != "over" and p["d2d_status"] != "over"


def test_floor_memo_exact():
    from accel_dse.core.scenario import Scenario
    from accel_dse.core.parallel import Layout
    s = api.scenario_from_body({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": HBM4,
                                "chip": {"hbm_base": "custom"}, "serving": {"phase": "decode", "batch": 64}}})
    assert isinstance(s, Scenario) and Layout
    for d in (300.0, 450.0, 517.5):
        hbm_base._FLOOR_MEMO.clear()
        r1 = hbm_base.shore_floor_mm2(s, d)
        assert hbm_base.shore_floor_mm2(s, d) == r1 == hbm_base._shore_floor_mm2(s, d)


def test_ui_cli_report_wiring():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert "c-hbm_pkg" in W.html_ids(html) and W.select_options(html, "c-hbm_pkg") == ["", "sphbm4"]
    assert W.has(js, "S.chipOver.hbm_pkg = e.target.value") and W.has(js, "$('c-hbm_pkg').value = S.chipOver.hbm_pkg || ''")
    assert W.has(js, "b.shoreline.stack_edge_mm || 11")
    assert W.count(js, "for (const k of ['noc_topology', 'noc_routing', 'noc_mem_attach', 'noc_eject'])") == 2   # 0.306 pin
    out = subprocess.run([sys.executable, "-m", "accel_dse.cli", "eval", "--chip", "1P", "--mem", HBM4,
                          "--hbm-pkg", "sphbm4", "--json"], capture_output=True, text=True, cwd=ROOT, check=True)
    d = json.loads(out.stdout)
    assert d["hbm_pkg"]["applies"] and d["shoreline"]["mem"]["unit_mm"] == 25.0 / 3
    from accel_dse.core import report
    src = open(report.__file__, encoding="utf-8").read()
    assert 'item("hbm_pkg", "HBM 封装"' in src
