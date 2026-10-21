"""0.317: cost per unit of work (TCO, opt-in) — $/M unit and unit/s/W per design from package cost per good unit, memory
$/GB, the complete package power × PUE × $/kWh, amortisation life and throughput; adversarial no-double-count and
buildability checks; verdicts under $/Mtok."""
import importlib.util
import os

from accel_dse import api
from accel_dse.core.energy_dse import energy_design_search
from accel_dse.core.tco import HOURS_PER_YEAR, MEM_USD_PER_GB, TCO_PRESETS, tco_params, tco_point
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_C = {}


def _ex():
    if "m" not in _C:
        spec = importlib.util.spec_from_file_location("chiplet_example",
                                                      os.path.join(ROOT, "scripts", "chiplet_example.py"))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _C["m"] = m
    return _C["m"]


def _run(**kw):
    k = tuple(sorted((a, str(b)) for a, b in kw.items()))
    if k not in _C:
        a = dict(ph="decode", cost="n5_cowos_l", tco="accel_2026")
        a.update(kw)
        ph = a.pop("ph")
        mp = a.pop("mem_usd_per_GB", None)
        m = _ex()
        if mp is not None:                         # pass a DSE memory price through chiplet_example.run
            orig = m.energy_design_search
            m.energy_design_search = lambda *x, **y: orig(*x, **{**y, "mem_usd_per_GB": mp})
            try:
                _C[k] = m.run("qwen3-8b", ph, "tp", 0, None, out=lambda *_: None, **a)
            finally:
                m.energy_design_search = orig
        else:
            _C[k] = m.run("qwen3-8b", ph, "tp", 0, None, out=lambda *_: None, **a)
    return _C[k]


def test_sources_and_defaults():
    p = TCO_PRESETS["accel_2026"]
    assert p["pue"] == 1.54 and "Uptime" in p["src"]["pue"] and "「来源」" in p["src"]["pue"]
    assert p["usd_per_kWh"] == 0.0891 and "EIA" in p["src"]["usd_per_kWh"]
    assert p["life_years"] == 5.0 and "10-K" in p["src"]["life_years"]
    assert p["system_usd_per_card"] == 0 and p["system_W_per_card"] == 0
    s = TCO_PRESETS["server_2026"]
    assert s["system_usd_per_card"] == (250e3 - 190e3) / 8 and s["system_W_per_card"] == (10.2e3 - 8 * 700) / 8
    assert MEM_USD_PER_GB["HBM3E"][0] == 17.5 and MEM_USD_PER_GB["HBM3E"][1] == "二手"
    assert "LPDDR6" not in MEM_USD_PER_GB                    # 0.318: still no public $/GB → never a silent default


def test_formula_exact():
    p = tco_params({"preset": "accel_2026", "capital_rate": 0.0})
    t = tco_point(p, pkg_usd=2000.0, mem_usd=1000.0, thr=5000.0, pkg_W=500.0, pkg_W_idle=100.0)
    en = 500.0 * 1.54 * HOURS_PER_YEAR / 1000 * 0.0891
    assert abs(t["energy_usd_per_yr"] - en) < 1e-9 and t["capex_usd_per_yr"] == 600.0
    assert abs(t["usd_per_Munit"] - (600.0 + en) / (5000.0 * 3600 * HOURS_PER_YEAR) * 1e6) < 1e-15
    assert abs(sum(t["share"].values()) - 1) < 1e-12 and t["units_per_s_per_W"] == 10.0
    q = tco_params({"util": 0.5, "capital_rate": 0.1})
    u = tco_point(q, pkg_usd=2000.0, mem_usd=1000.0, thr=5000.0, pkg_W=500.0, pkg_W_idle=100.0)
    assert u["it_W"] == 300.0 and abs(u["capex_usd_per_yr"] - 3000 * 0.1 / (1 - 1.1 ** -5)) < 1e-9
    for bad in ({"pue": 0.9}, {"util": 0}, {"life_years": 0}, {"foo": 1}, {"preset": "x"}, {"mem_usd_per_GB": {"XDR": 1}}):
        try:
            tco_params(bad)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_memory_counted_once_and_not_in_package():
    a = _run()
    b = _run(mem_usd_per_GB={"HBM3E": 30.0})                   # DSE memory price → used by TCO too, not added twice
    for x, y in zip(a["rows"], b["rows"]):
        assert x["usd_good_pkg"] == y["usd_good_pkg"]           # package $ never sees the memory price
        assert abs(y["tco"]["mem_usd"] - y["mem_usd"]) < 1e-9
        assert y["tco"]["mem_price_src"] == "用户输入" and x["tco"]["mem_price_src"] == "二手"
        assert abs(x["tco"]["capex_usd"] - (x["usd_good_pkg"] + x["mem_GB"] * 17.5)) < 1e-6
        assert abs(y["tco"]["capex_usd"] - x["tco"]["capex_usd"] - x["mem_GB"] * (30.0 - 17.5)) < 1e-6
        assert x["tco"]["pkg_usd"] == x["usd_good_pkg"]
    from accel_dse.core.energy_dse import COST_PRESETS
    for cp in COST_PRESETS.values():                            # no memory-device term exists in package cost
        assert not any("mem" in k or "hbm" in k.lower() for k in cp)


def test_power_is_the_complete_package_power_once():
    o = _run(ph="prefill")                                     # 700 W budget caps several prefill rows (0.316)
    assert any(x.get("capped") for x in o["rows"])
    for x in o["rows"]:
        t = x["tco"]
        assert t["pkg_W"] == x["thermal"]["pkg_W"] and t["pkg_W"] > x["W_card"]       # complete, not card W
        if x.get("capped"):
            assert abs(t["pkg_W"] - x["cap_pkg_W"]) <= 1e-9 * x["cap_pkg_W"]        # same power as the cap check
        en = t["it_W"] * 1.54 * HOURS_PER_YEAR / 1000 * 0.0891
        assert abs(t["energy_usd_per_yr"] - en) < 1e-6 and t["it_W"] == t["pkg_W"]   # PUE once, nothing else added
        assert abs(t["units_per_s_per_W"] - x["thr"] / t["pkg_W"]) < 1e-9


def test_buildability_and_front():
    for kw in ({}, {"cooling": "air_oam"}, {"include_unbuildable": True}, {"include_unbuildable": True, "cooling": "air_oam"}):
        o = _run(ph="prefill", **kw)
        fr = [x for x in o["rows"] if x.get("front_tco")]
        assert fr and all(x["feasible"] and x.get("buildable") is not False for x in fr), kw
        b = o["rows"][o["tco"]["best"]]
        assert b["feasible"] and b.get("buildable") is not False and b.get("best_tco")
        ok = [x for x in o["rows"] if x["feasible"] and x.get("buildable") is not False]
        assert b["tco"]["usd_per_Munit"] == min(x["tco"]["usd_per_Munit"] for x in ok)
    o = _run(include_unbuildable=True, shore_strict=True)      # decode, 8 × HBM3E: monos tight → unbuildable
    kept = [i for i, x in enumerate(o["rows"]) if x.get("front") and x.get("buildable") is False]
    assert kept == [0, 1, 2]                                    # main front keeps them (include_unbuildable) …
    assert not any(o["rows"][i].get("front_tco") for i in kept)   # … the $/M front never
    assert o["rows"][o["tco"]["best"]].get("buildable") is not False
    th = _run(ph="prefill", cooling="air_oam", power_scope="card")
    out = [x for x in th["rows"] if x.get("front_thermal_out")]
    assert out and not any(x.get("front_tco") for x in out)


def test_opt_in_and_needs_cost():
    o = _run(tco=None)
    assert o["tco"] is None and not any("tco" in x or "front_tco" in x for x in o["rows"])
    assert not any("$/M" in n for n in o["notes"])
    try:
        energy_design_search(None, None, None, tco="accel_2026")
        raise AssertionError
    except ValueError as e:
        assert "tco needs cost" in str(e)


def test_verdicts_under_tco():
    r = _ex().tco_study(out=lambda *_: None)
    assert all(c["verdict"] == c["verdict_tco"] for c in r["cases"])          # mono / only_split unchanged
    assert sum(1 for c in r["cases"] if c["verdict_tco"] == "mono") == 7
    diff = [x for x in r["sweep"] if x["cheaper_pkg"] != x["cheaper_tco"]]
    assert len(diff) == 16 and all(x["tp"] == 8 and (x["cheaper_pkg"], x["cheaper_tco"]) == ("split", "mono") for x in diff)
    m = {(x["tp"], x["link"], x["d0"], x["mac_x"]): x for x in r["mem"]}
    assert m[(8, 400.0, 0.1, 2.0)]["by_thr"].startswith("hbm3e_8s") and m[(8, 400.0, 0.1, 2.0)]["by_tco"].startswith("hbm3e_4s")
    assert all(x["by_capex"].startswith("hbm3e_4s") for x in r["mem"])
    assert all(x["by_tco"] == x["by_thr"] for x in r["mem"] if x["tp"] == 1)


def test_eval_and_api():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": "hbm3e_4s_8h24g_9200",
                                           "serving": {"phase": "decode", "batch": 64, "prompt": 2048, "ctx": 4096}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "sensitivity": False}
    o = api.api_eval({**b, "tco": {"cost": "n5_cowos_s"}})
    t = o["tco"]
    assert abs(t["pkg_W"] - o["thermal"]["pkg_W"]) < 1e-9 and t["mem_usd"] == 96 * 17.5 and t["unit"] == "token"
    c = api.api_eval({**b, "energy": {"energy_preset": "tpuv4i_7nm", "tdp_W": 150}, "power_cap": True,
                      "tco": {"pkg_usd": 3000}})["tco"]
    assert abs(c["pkg_W"] - 150.0) < 1e-3 and c["pkg_usd"] == 3000 and c["thr"] < t["thr"]     # capped point
    assert "tco" not in api.api_eval(b)
    for bad in ({}, {"cost": "x"}, {"pkg_usd": -1}, {"pkg_usd": 1, "pue": 0.5}):
        try:
            api.api_eval({**b, "tco": bad})
            raise AssertionError(bad)
        except api.ApiError:
            pass


def test_ui_strings():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert 'id="ins-dse-tco"' in html and 'id="c-tco"' in html and 'id="chip-tco"' in html
    assert "ex.tco = $('ins-dse-tco').value" in js and "tco: { preset: S.tco, cost: 'n5_cowos_s' }" in js
    assert "function renderTco(" in js and "e.verdict_tco" in js and "x.best_tco ? ' ◆'" in js
    assert "PUE 1.54（Uptime 2025「来源」）" in html
