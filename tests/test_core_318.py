"""0.318: TCO sensitivity — each input across its sourced range (one at a time), the tornado of a chosen design, and
whether the 0.317 $/M verdicts flip inside those ranges (with the threshold); newly priced memory types."""
import importlib.util
import math
import os

from accel_dse import api, serve
from accel_dse.core.energy_dse import _cost_params, _tco_ok
from accel_dse.core.tco import MEM_RANGES, MEM_USD_PER_GB, TCO_PRESETS, tco_params
from accel_dse.core.tco_sens import (COST_INPUTS, INPUTS, RANGES, base_values, row_usd, scan_flip, tornado)
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAGS = {"来源", "二手", "推导", "假设", "近似"}
_C = {}


def _mod(name):
    if name not in _C:
        spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _C[name] = m
    return _C[name]


def _case0():
    if "o" not in _C:
        _C["o"] = _mod("chiplet_example").run("qwen3-8b", "decode", "tp", 0, None, out=lambda *_: None,
                                              cost="n5_cowos_l", d0=0.1, tco="accel_2026")
    return _C["o"]


def _base(tco="accel_2026", d0=0.1):
    return {"tco": tco_params(tco), "cost": _cost_params("n5_cowos_l"), "d0": d0, "prices": None}


def test_ranges_sourced_and_contain_defaults():
    for k, (lo, hi, tl, th, src) in RANGES.items():
        if k in ("mem_usd_per_GB", "carrier_yield"):                 # per memory type / per carrier (0.319)
            continue
        assert lo < hi and tl in TAGS and th in TAGS and len(src) > 20, k
    for t, (lo, hi, tl, th, src) in MEM_RANGES.items():
        assert lo < hi and tl in TAGS and th in TAGS and src, t
        assert lo <= MEM_USD_PER_GB[t][0] <= hi, t                  # every priced default sits inside its range
    p = TCO_PRESETS["accel_2026"]
    for k in ("usd_per_kWh", "pue", "life_years"):
        assert RANGES[k][0] <= p[k] <= RANGES[k][1], k
    cp = _cost_params("n5_cowos_l")
    assert RANGES["wafer_usd"][0] <= cp["wafer_usd"] <= RANGES["wafer_usd"][1]
    assert RANGES["d0"][0] <= 0.1 <= RANGES["d0"][1] and RANGES["pkg_scale"][0] < 1 < RANGES["pkg_scale"][1]
    assert RANGES["d0"][2] == "假设"                                  # no public mature N5 D0 → the low end is labelled


def test_new_memory_prices_labelled():
    assert MEM_USD_PER_GB["HBM3"][:2] == (13.0, "二手") and "Silicon Analysts" in MEM_USD_PER_GB["HBM3"][2]
    assert MEM_USD_PER_GB["HBM4E"][:2] == (28.0, "二手") and "预测" in MEM_USD_PER_GB["HBM4E"][2]
    assert MEM_USD_PER_GB["LPDDR5"][:2] == (10.0, "二手") and "SemiAnalysis" in MEM_USD_PER_GB["LPDDR5"][2]
    assert "LPDDR6" not in MEM_USD_PER_GB and "LPDDR6" not in MEM_RANGES   # only a rumoured premium → unpriced
    import re
    for t in MEM_USD_PER_GB:
        assert re.search(r"20\d\d|\dQ\d\d", MEM_USD_PER_GB[t][2]), t     # dated


def test_row_usd_exact_and_directions():
    o = _case0()
    b = _base()
    r = o["rows"][o["tco"]["best"]]
    assert math.isclose(row_usd(r, b), r["tco"]["usd_per_Munit"], rel_tol=1e-12)
    t = r["tco"]
    # life: capex share scales 5 / L exactly (straight line, capital rate 0)
    cap = t["capex_usd_per_yr"] / (t["capex_usd_per_yr"] + t["energy_usd_per_yr"])
    for L in (4.0, 6.0):
        assert math.isclose(row_usd(r, b, {"life_years": L}) / t["usd_per_Munit"], cap * 5 / L + (1 - cap), rel_tol=1e-9)
    # $/kWh and PUE scale only the energy part
    en = 1 - cap
    assert math.isclose(row_usd(r, b, {"usd_per_kWh": 0.0891 * 2}) / t["usd_per_Munit"], 1 + en, rel_tol=1e-9)
    assert math.isclose(row_usd(r, b, {"pue": 1.54 * 1.5}) / t["usd_per_Munit"], 1 + 0.5 * en, rel_tol=1e-9)
    # memory $ only through GB × $/GB (× the 0.319 HBM scrap factor)
    d = row_usd(r, b, {"mem_usd_per_GB": 18.0}) - row_usd(r, b, {"mem_usd_per_GB": 17.0})
    assert math.isclose(d, r["mem_GB"] * r["mem_scrap_k"] / 5 / (r["thr"] * 3600 * 8760) * 1e6, rel_tol=1e-9)
    for k, up in (("wafer_usd", True), ("d0", True), ("pkg_scale", True), ("usd_per_kWh", True), ("pue", True),
                  ("life_years", False), ("host", True), ("capital_rate", True)):
        lo, hi = RANGES[k][:2]
        assert (row_usd(r, b, {k: hi}) > row_usd(r, b, {k: lo})) == up, k


def test_tornado_sorted_and_complete():
    o = _case0()
    s = o["tco"]["sens"]
    assert s["row"] == o["tco"]["best"] and len(s["bars"]) == len(INPUTS) and not s["skipped"]
    mx = [max(abs(x["d_lo"]), abs(x["d_hi"])) for x in s["bars"]]
    assert mx == sorted(mx, reverse=True)
    assert s["bars"][0]["input"] == "host"                    # host share is the biggest swing for an accelerator-only base
    bv = base_values(_base(), "HBM3E")
    assert bv["host"] == 0 and bv["mem_usd_per_GB"] == 17.5 and bv["pkg_scale"] == 1.0
    r = dict(o["rows"][0])
    r["tco"] = {**r["tco"], "usd_per_Munit": r["tco"]["usd_per_Munit"] * 1.01}
    try:
        tornado(r, _base())
        raise AssertionError("mismatch not caught")
    except ValueError:
        pass


def test_d0_flip_threshold_case0():
    o = _case0()
    e = next(x for x in o["chiplet"]["compare"] if x.get("verdict_tco") == "mono")
    fl = {f["input"]: f for f in e["tco_flips"]}
    assert set(fl) == {"d0"}                                   # only D0 flips mono → split inside the sourced ranges
    th = fl["d0"]["thresholds"][0]
    assert 0.14 < th["at"] < 0.17 and (th["from"], th["to"]) == ("mono", "split")   # 0.319 component: ≈ 0.150
    # bisection is exact: the verdict on either side of the threshold
    from accel_dse.core.tco_sens import compare_key, mono_split_verdict
    rows = o["rows"]
    g = [i for i, x in enumerate(rows) if compare_key(x) == compare_key(e) and _tco_ok(x)]
    f = mono_split_verdict(rows, [i for i in g if rows[i]["dies"] == 1], [i for i in g if rows[i]["dies"] > 1], _base())
    assert f({"d0": th["at"] - 1e-6}) == "mono" and f({"d0": th["at"] + 1e-6}) == "split"
    s = scan_flip(f, "pue", RANGES["pue"], 1.54)
    assert not s["flips"] and s["thresholds"] == []


def test_study_subset():
    T = _mod("tco_sens")
    res = T.study(out=lambda *_: None, cases=[0], sweep_cells=[(8, 0.2)])
    assert res["cases"][0]["flips"][0]["input"] == "d0"
    m = [x for x in res["mem"] if x["tp"] == 8 and x["link"] == 400 and x["mac_x"] == 2.0]
    assert m and m[0]["by_tco"] == "hbm3e_4s_8h24g_9200" and not m[0]["flips"]   # 0.317 4-stack choice holds
    assert all(s["input"] != "d0" for x in res["sweep"] + res["mem"] for s in x["flips"])   # D0 is a grid axis there


def test_eval_sens_and_cost_inputs_skipped_for_given_pkg_usd():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": "hbm3e_4s_8h24g_9200",
                                           "serving": {"phase": "decode", "batch": 64, "prompt": 2048, "ctx": 4096}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "sensitivity": False}
    t = api.api_eval({**b, "tco": {"cost": "n5_cowos_s"}})["tco"]
    assert {x["input"] for x in t["sens"]["bars"]} == set(INPUTS)
    u = api.api_eval({**b, "tco": {"pkg_usd": 3000, "preset": "server_2026"}})["tco"]
    assert not {x["input"] for x in u["sens"]["bars"]} & set(COST_INPUTS) and len(u["sens"]["skipped"]) == len(COST_INPUTS)
    h = next(x for x in u["sens"]["bars"] if x["input"] == "host")
    assert h["base"] == 1.0 and h["d_hi"] == 0 and h["d_lo"] < 0


def test_api_route():
    assert "/api/tco_sens" in serve.POST_ROUTES
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "serving": {"phase": "decode", "batch": 64,
                                                                            "prompt": 2048, "ctx": 4096}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "budget": {"density_preset": "n5_derived"}, "mac_scales": [1, 2],
         "sram_scales": [1], "dies": [1, 2], "cost": "n5_cowos_s", "tco": "accel_2026", "d0_per_cm2": 0.2}
    r = api.api_tco_sens(b)
    assert r["sens"]["row"] == r["row"] and r["design"]["dies"] >= 1 and "HBM4E" in r["ranges"]["mem_usd_per_GB"]
    assert r["compare"] and all(e["verdict_tco"] for e in r["compare"])
    j = next(i for i, e in enumerate(api.api_insight_dse(b)["rows"]) if _tco_ok(e) and i != r["row"])
    assert api.api_tco_sens({**b, "tco_sens_row": j})["row"] == j
    no_tco = {k: v for k, v in b.items() if k != "tco"}
    for fn, bad in ((api.api_tco_sens, no_tco), (api.api_tco_sens, {**b, "tco_sens_row": -1}),
                    (api.api_tco_sens, {**b, "tco_sens_row": 10 ** 6}),
                    (api.api_insight_dse, {**no_tco, "tco_sens_row": 0})):
        try:
            fn(bad)
            raise AssertionError(bad)
        except api.ApiError:
            pass


def test_ui_strings():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert 'id="ins-dse-sens"' in html and "function sensBars(" in js
    assert "sensBars(t.sens, 'chip-tco-sens')" in js and "sensBars(sn, 'ins-dse-sens-bars')" in js
    assert "e.tco_flips" in js and "HBM4E $28 UBS 2027 预测「二手」" in html
