"""0.315: complete package power for thermal / power buildability (static floor always, VR / PDN loss, HBM / PHY
placement, per-definition flux limits, hot-leakage margin flag), suite-time check, Flint release."""
import importlib.util
import os
from dataclasses import replace

from accel_dse import api
from accel_dse.core.energy import energy_report
from accel_dse.core.energy_dse import energy_design_search
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.powercap import static_W
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.thermal import (COOLING, STATIC_FLOOR_W_MM2, VR_EFF, _hot_k, dram_dev_W, package_power,
                                    thermal_report)
from accel_dse.core.budget import shoreline_report
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PF = Serving(phase="prefill", batch=1, prompt=4096, ctx=8192)


def _ex():
    spec = importlib.util.spec_from_file_location("chiplet_example", os.path.join(ROOT, "scripts", "chiplet_example.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sources_of_the_new_terms():
    # VR efficiency inside the sourced full-load range (ECCE 2024 77.2 / 84.4 %, two-stage ≤ 0.95 × second stage)
    assert 0.772 <= VR_EFF <= 0.90
    # static floor = two independent idle measurements per mm² of logic footprint (H100 board, MI300X socket)
    assert abs(71.8 / 814 - STATIC_FLOOR_W_MM2) / STATIC_FLOOR_W_MM2 < 0.01
    assert abs(132.0 / (4 * 370) - STATIC_FLOOR_W_MM2) / STATIC_FLOOR_W_MM2 < 0.02
    assert 3.0 < _hot_k() < 3.4           # TSMC 3 nm leakage ratio, 40 → 85 °C


def test_package_power_composition():
    p = package_power(500.0, 600.0, 1, static_W=10.0)
    floor = STATIC_FLOOR_W_MM2 * 600 * VR_EFF
    assert p["static_basis"] == "floor" and abs(p["static_add_W"] - (floor - 10.0)) < 1e-9
    assert abs(p["vr_loss_W"] - (500 + floor - 10) * (1 / VR_EFF - 1)) < 1e-9
    assert abs(p["pkg_W"] - (500 + floor - 10) / VR_EFF) < 1e-9 and p["pkg_W_hot"] > p["pkg_W"]
    # already-charged static above the floor: nothing added (no double count)
    assert package_power(500.0, 600.0, 1, static_W=80.0)["static_add_W"] == 0.0
    # a board-level idle figure: no floor, no second VR loss on it
    q = package_power(500.0, 600.0, 1, static_W=60.0, idle_W=50.0)
    assert q["static_basis"] == "idle" and q["static_add_W"] == 0.0
    assert abs(q["pkg_W"] - (50.0 + 450.0 / VR_EFF)) < 1e-9
    # no area: no floor, noted
    t = thermal_report(500.0, None, 1, 0.0, None)
    assert t["static_basis"] == "none" and any("晶粒面积未知" in n for n in t["notes"])


def test_floor_and_measured_idle_agree_on_h100():
    """The mechanism check: H100-like with no idle figure (floor) and with the measured H100 idle fraction give the
    same package power — the floor is the same measurement, placed on the same side of the VR."""
    ex = _ex()
    t = ex.table()
    ts = replace(t, idle_frac_tdp=71.8 / 700, tdp_W=700.0)
    for ph, b in (("decode", 64), ("prefill", 1)):
        s = Scenario(model="qwen3-8b", chip=CHIPS["H100-like"], mem_id="hbm3_5s_8h16g_5234",
                     serving=Serving(phase=ph, batch=b, prompt=4096, ctx=8192))
        r = evaluate(s)
        a = shoreline_report(s)["die_mm2"]
        out = []
        for tab in (t, ts):
            er = energy_report(r, tab)
            out.append(thermal_report(er["avg_W_per_card"], a, 1, dram_dev_W(er), "air_hgx", static_W(er),
                                      er.get("idle_W_used")))
        assert out[0]["static_basis"] == "floor" and out[1]["static_basis"] == "idle"
        assert abs(out[0]["pkg_W"] - out[1]["pkg_W"]) / out[1]["pkg_W"] < 0.01, ph
        assert out[0]["pkg_W"] < 700.0        # under H100's 700 W TDP (TDP is the rated maximum)


def test_each_precedent_judged_on_its_own_definition():
    # liquid: card-level over the shipped 0.96 but die heat under HIR 1.70 → buildable (0.314 judged 1.70 card-level)
    t = thermal_report(1100.0, 1000.0, 1, 50.0, "liquid", basis="package")
    assert t["flux_W_mm2"] > COOLING["liquid"]["flux_W_mm2"] and t["die_flux_W_mm2"] < 1.70 and t["feasible"] is True
    assert abs(t["flux_ratio"] - t["die_flux_W_mm2"] / 1.70) < 1e-12
    # die heat excludes the module VR loss and the DRAM devices
    m = thermal_report(500.0, 600.0, 1, 30.0, None)
    assert abs(m["die_W"] - (m["power"]["delivered_W"] - 30.0)) < 1e-9 and m["die_W"] < m["pkg_W"] - m["vr_loss_W"]


def test_verdict_change_and_hot_margin():
    ex = _ex()
    kw = dict(mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,))
    # 1P + 8 × HBM3E prefill: 0.314 ok on air (0.78 W/mm²), 0.315 over (≈ 0.99 with floor + VR)
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id="hbm3e_8s_8h24g_9200", serving=PF)
    r = evaluate(s)
    er = energy_report(r, ex.table())
    a = shoreline_report(s)["die_mm2"]
    th = thermal_report(er["avg_W_per_card"], a, 1, dram_dev_W(er), "air_oam", static_W(er), None)
    assert er["avg_W_per_card"] / a < 0.86 and th["feasible"] is False and 0.95 < th["flux_W_mm2"] < 1.05
    d = energy_design_search(s, ex.table(), ex.budget(), cooling="liquid", **kw)
    row = d["rows"][0]["thermal"]
    for k in ("card_W", "pkg_W", "static_add_W", "vr_loss_W", "pkg_W_hot", "hot_ok", "static_basis"):
        assert k in row
    assert row["pkg_W"] > row["card_W"] and row["static_basis"] == "floor"
    assert any("稳压器 / PDN 损耗" in n for n in d["notes"])
    # hot margin: ok at the idle-temperature floor, over at the 85 °C upper bound → flagged, still buildable
    h = thermal_report(600.0, 1000.0, 1, 0.0, "air_hgx", static_W=0.0)     # 794 W pkg; 988 W hot
    assert h["feasible"] is True and h["hot_ok"] is False and any("余量不足" in n for n in h["notes"])
    assert thermal_report(300.0, 1000.0, 1, 0.0, "air_hgx")["hot_ok"] is True


def test_api_reports_the_breakdown():
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": "hbm3e_4s_8h24g_9200",
                                           "serving": {"phase": "prefill", "batch": 1, "prompt": 4096, "ctx": 8192}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "sensitivity": False}
    t = api.api_eval(b)["thermal"]
    assert abs(t["pkg_W"] - (t["card_W"] + t["static_add_W"] + t["vr_loss_W"])) < 1e-6


def test_flint_release():
    from accel_dse.core.catalog import list_models
    L = {x["id"]: x for x in list_models()}
    f = L["flint"]
    assert f["provider"] == "recursive" and f["param_err"] == 0.0 and abs(f["params_B"] - 39.785) < 0.01
    assert f["is_moe"] and f["n_layers"] == 49
    r = evaluate(Scenario(model="flint", chip=CHIPS["H100-like"], mem_id="hbm3_5s_8h16g_5234",
                          serving=Serving(phase="decode", batch=8, ctx=4096)))
    assert r.fits and r.per_card > 0


def test_ui_strings():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert "稳压器损耗" in js and "x.hot_ok" in js and "静态底功耗" in js
    assert "0.088 W/mm²" in html and "液冷 0.96 W/mm²" in html
