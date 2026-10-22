"""0.319: component package / bonding yield (carrier × HBM attach × die bond, KGD + HBM scrap on a failed assembly,
flat switch = 0.318) and the joint-corner check of the TCO sensitivity."""
import importlib.util
import math
import os

from accel_dse import api
from accel_dse.core.energy_dse import (ASM_ATTACH, ASM_CARRIER, ASM_HYBRID, ASM_SRC, _cost_params, _tco_ok, asm_note,
                                       asm_yield, hbm_stacks, package_cost)
from accel_dse.core.tco import tco_params
from accel_dse.core.tco_sens import (CARRIER_RANGES, INPUTS, LOW_IS_ADVERSE, RANGES, applicable, corner, row_usd,
                                     scan_flip)
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = ("来源", "二手", "推导", "假设", "近似")
_C = {}


def _mod(name):
    if name not in _C:
        spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _C[name] = m
    return _C[name]


def _run(asm, **kw):
    key = (asm, tuple(sorted(kw.items())))
    if key not in _C:
        kw = {"tco": "accel_2026", "d0": 0.1, **kw}
        _C[key] = _mod("chiplet_example").run("qwen3-8b", "decode", "tp", 0, None, out=lambda *_: None,
                                              cost={"preset": "n5_cowos_l", "asm_model": asm}, **kw)
    return _C[key]


def test_calibration_and_sources():
    l, s = _cost_params("n5_cowos_l"), _cost_params("n5_cowos_s")
    assert (l["carrier"], s["carrier"], l["asm_model"], l["flow"], l["bond"], l["rework"]) == \
        ("cowos_l", "cowos_s", "component", "chip_first", "microbump", 0.0)
    assert (l["carrier_yield"], s["carrier_yield"], l["attach_yield"], l["hybrid_yield"]) == (0.81, 0.985, 0.998, 0.97)
    b200 = asm_yield(l, 2, 8)                       # Epoch: B200 yield loss $1,000 on ~$3,800 of dies + HBM → 0.79
    assert abs(b200["Y"] - 0.794) < 1e-3 and abs(b200["Y"] - (1 - 1000 / 3800 * 0.8)) < 0.01
    h100 = asm_yield(s, 1, 6)                       # Epoch CoWoS-S 95–99 % (mid 0.97)
    assert abs(h100["Y"] - 0.97) < 2e-3 and 0.95 <= h100["Y"] <= 0.99
    for k in ("cowos_s", "cowos_l", "attach", "hybrid", "scrap"):
        assert any(t in ASM_SRC[k] for t in TAGS), k
    for c, (lo, hi, tl, th, src) in CARRIER_RANGES.items():
        assert lo <= ASM_CARRIER[c] <= hi <= 1 and tl in TAGS and th in TAGS and src, c
    assert RANGES["attach_yield"][0] <= ASM_ATTACH <= RANGES["attach_yield"][1] <= 1
    assert 0.95 < ASM_HYBRID < 0.99


def test_scrap_factors_flow_rework_bond():
    cp = _cost_params("n5_cowos_l")
    a = asm_yield(cp, 2, 8)
    assert math.isclose(a["k_kgd"], 1 / a["Y"]) and a["k_mem"] == a["k_kgd"] and a["k_all"] == 1.0
    assert asm_yield(cp, 2, 0)["k_mem"] == 1.0                                      # LPDDR is off the carrier
    last = asm_yield({**cp, "flow": "chip_last"}, 2, 8)
    assert math.isclose(last["k_kgd"], 1 / 0.998 ** 10) and last["Y"] == a["Y"]     # carrier tested first
    rw = asm_yield({**cp, "rework": 0.5}, 2, 8)
    assert math.isclose(rw["k_kgd"], (1 - 0.5 * (1 - a["Y"])) / a["Y"]) and 1 < rw["k_kgd"] < a["k_kgd"]
    hy = asm_yield({**cp, "bond": "hybrid"}, 2, 8)
    assert math.isclose(hy["Y"], 0.81 * 0.998 ** 8 * 0.97 ** 2)
    f = asm_yield(_cost_params({"preset": "n5_cowos_l", "asm_model": "flat"}), 2, 8)
    assert (f["model"], f["Y"], f["k_mem"]) == ("flat", 1.0, 1.0)


def test_package_cost_component_vs_flat():
    cp = _cost_params("n5_cowos_l")
    fl = _cost_params({"preset": "n5_cowos_l", "asm_model": "flat"})
    c, f = package_cost(cp, 400, 2, 0.8, 2500, 8), package_cost(fl, 400, 2, 0.8, 2500, 8)
    assert math.isclose(f["usd_good_pkg"], f["si_usd"] + f["pkg_usd"] + f["test_usd"]) and f["kgd_scrap_usd"] == 0
    k = c["asm"]["k_kgd"]
    assert math.isclose(c["kgd_scrap_usd"], c["si_usd"] * (k - 1))
    assert math.isclose(c["usd_good_pkg"], c["si_usd"] * k + c["pkg_usd"] + c["test_usd"])   # pkg / test per shipped part
    f2 = package_cost({**fl, "asm_yield": 0.9}, 400, 2, 0.8, 2500, 8)
    assert math.isclose(f2["usd_good_pkg"], f["usd_good_pkg"] / 0.9)                 # 0.318 rule untouched
    assert "0.319" in asm_note(cp) and "0.318" in asm_note(fl)


def test_cost_validation():
    for bad in ({"asm_model": "x"}, {"flow": "x"}, {"bond": "x"}, {"carrier_yield": 0}, {"carrier_yield": 1.2},
                {"attach_yield": True}, {"rework": 1.0}, {"rework": -0.1}, {"hybrid_yield": "a"}):
        try:
            _cost_params({"preset": "n5_cowos_l", **bad})
            raise AssertionError(bad)
        except ValueError:
            pass
    assert _cost_params({"preset": "n5_cowos_s", "rework": 0.3, "flow": "chip_last"})["rework"] == 0.3
    assert hbm_stacks("hbm3e_8s_8h24g_9200") == 8 and hbm_stacks("hbm3e_4s_8h24g_9200") == 4


def test_flat_reproduces_0318_and_component_case0():
    f, c = _run("flat"), _run("component")
    bf, bc = f["rows"][f["tco"]["best"]], c["rows"][c["tco"]["best"]]
    # 0.329 re-pin (HBM3E PHY 16 → 24 mm² + controller 3.575 mm² per stack): the best row is now the 4-die split;
    # flat $2013 / 0.006594 → $2161 / 0.006799, component $2116 / 0.007372 → $2233 / 0.007577
    assert bf["dies"] == 4 and bc["dies"] == 4
    assert round(bf["usd_good_pkg"]) == 2161 and math.isclose(bf["tco"]["usd_per_Munit"], 0.006799, rel_tol=1e-3)
    assert bf["mem_scrap_k"] == 1.0 and bf["kgd_scrap_usd"] == 0
    assert round(bc["usd_good_pkg"]) == 2233 and math.isclose(bc["tco"]["usd_per_Munit"], 0.007577, rel_tol=1e-3)
    assert 1.2 < bc["mem_scrap_k"] < 1.3 and bc["asm_Y"] < 0.8
    assert math.isclose(bc["tco"]["mem_usd"], bc["tco"]["mem_dev_usd"] * bc["mem_scrap_k"])
    assert c["cost"]["asm_model"] == "component" and c["cost"]["carrier"] == "cowos_l"


def test_joint_corners_bracket_base():
    o = _run("component")
    r = o["rows"][o["tco"]["best"]]
    s = o["tco"]["sens"]
    b = {"tco": tco_params("accel_2026"), "cost": _cost_params("n5_cowos_l"), "d0": 0.1, "prices": None}
    adv, fav = corner(b, "adverse"), corner(b, "favourable")
    assert adv["mem_end"] == "hi" and fav["mem_end"] == "lo" and adv["life_years"] == RANGES["life_years"][0]
    assert adv["carrier_yield"] == CARRIER_RANGES["cowos_l"][0] and fav["carrier_yield"] == CARRIER_RANGES["cowos_l"][1]
    assert adv["wafer_usd"] == RANGES["wafer_usd"][1] and set(LOW_IS_ADVERSE) <= set(INPUTS)
    t0 = r["tco"]["usd_per_Munit"]
    assert row_usd(r, b, adv) > t0 > row_usd(r, b, fav)
    assert math.isclose(s["corners"]["adverse"]["usd"], row_usd(r, b, adv), rel_tol=1e-12)
    for x in s["bars"]:               # the joint corner is at least as adverse as every single input's adverse end
        assert s["corners"]["adverse"]["d"] >= max(x["d_lo"], x["d_hi"]) - 1e-12
        assert s["corners"]["favourable"]["d"] <= min(x["d_lo"], x["d_hi"]) + 1e-12
    assert "d0" not in corner(b, "adverse", skip=("d0",))
    fl = {**b, "cost": _cost_params({"preset": "n5_cowos_l", "asm_model": "flat"})}
    assert not applicable("carrier_yield", fl) and "carrier_yield" not in corner(fl, "adverse")
    for e in o["chiplet"]["compare"]:
        if e.get("verdict_tco") in ("mono", "split"):
            assert set(e["tco_corners"]) == {"adverse", "favourable"}


def test_scan_outside_range_base():
    # base D0 0.3 lies above the sourced range; the label changes between the range end and the base
    r = scan_flip(lambda ov: "a" if ov.get("d0", 0.3) < 0.25 else "b", "d0", (0.05, 0.2, "二手", "二手", "x"), 0.3, n=8)
    assert r["base"] == "b" and r["flips"]
    th = [t for t in r["thresholds"] if t.get("outside")]
    assert th and abs(th[0]["at"] - 0.25) < 1e-6 and r["nearest"] is th[0]
    r2 = scan_flip(lambda ov: "a", "d0", (0.05, 0.2, "二手", "二手", "x"), 0.3, n=8)
    assert not r2["flips"] and "nearest" not in r2


def test_server_d0_03_runs():
    # 0.318 raised ValueError (min of an empty threshold list) on this cell
    o = _run("component", tco="server_2026", d0=0.3, mem="hbm3e_4s_8h24g_9200", tp=1, link=400.0)
    assert any(_tco_ok(x) for x in o["rows"])


def test_study_subset_corners():
    T = _mod("tco_sens")
    res = T.study(out=lambda *_: None, cases=[0], sweep_cells=[(8, 0.2)])
    fd = next(f for f in res["cases"][0]["flips"] if f["input"] == "d0")
    th = fd["thresholds"][0]
    assert 0.10 < th["at"] < 0.12                                           # 0.329: ≈ 0.150 → ≈ 0.110
    assert set(res["corner_flips"]) == {"adverse", "favourable"}
    m = [x for x in res["mem"] if x["tp"] == 8 and x["link"] == 400 and x["mac_x"] == 2.0]
    assert m and m[0]["by_tco"] == "hbm3e_8s_8h24g_9200" and not m[0]["flips"]
    assert set(m[0]["corners"].values()) == {"hbm3e_8s_8h24g_9200"}        # 0.329: 4-stack → 8-stack at both corners


def test_api_carries_corners_and_flat_switch():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "serving": {"phase": "decode", "batch": 64,
                                                                            "prompt": 2048, "ctx": 4096}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived"}, "mac_scales": [1],
         "sram_scales": [1], "dies": [1, 2], "cost": "n5_cowos_s", "tco": "accel_2026", "d0_per_cm2": 0.2}
    r = api.api_tco_sens(b)
    assert set(r["sens"]["corners"]) == {"adverse", "favourable"} and all("tco_corners" in e for e in r["compare"])
    d = api.api_insight_dse({**b, "cost": {"preset": "n5_cowos_s", "asm_model": "flat"}})
    ps = [x for x in d["rows"] if x.get("tco")]
    assert ps and all(x["mem_scrap_k"] == 1.0 for x in ps)
    try:
        api.api_insight_dse({**b, "cost": {"preset": "n5_cowos_s", "rework": 2}})
        raise AssertionError
    except api.ApiError:
        pass
    ev = api.api_eval({"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": "lpddr5x_8x64_8533_16g",
                                                         "serving": {"phase": "decode", "batch": 8, "prompt": 512, "ctx": 1024}},
                       "energy": {"energy_preset": "tpuv4i_7nm"}, "sensitivity": False, "tco": {"cost": "n5_cowos_s"}})
    assert ev["tco"]["mem_scrap_k"] == 1.0 and ev["tco"]["mem_usd"] == ev["tco"]["mem_dev_usd"]   # LPDDR off-carrier


def test_ui_strings():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert 'id="ins-dse-asm"' in html and "Epoch B200" in html and "asm_model: 'flat'" in js
    assert "${id}-corners" in js and "ins-dse-sens-corners" in js and "e.tco_corners" in js
