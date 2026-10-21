"""0.314: thermal model audited against shipped parts (one card-level flux definition, re-sourced limits, static-power
flag), shore_strict evidence (tight → precedent note), OvisOCR3 release."""
import importlib.util
import os
from dataclasses import replace

from accel_dse import api
from accel_dse.core.budget import TIGHT_PRECEDENT, shoreline_report
from accel_dse.core.catalog import list_models
from accel_dse.core.energy_dse import energy_design_search
from accel_dse.core.evaluate import evaluate
from accel_dse.core.hardware import CHIPS
from accel_dse.core.scenario import Scenario, Serving
from accel_dse.core.thermal import COOLING, KNOWN_PARTS, thermal_report
from tests import webstruct as W  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
H8, H4 = "hbm3e_8s_8h24g_9200", "hbm3e_4s_8h24g_9200"
PF = Serving(phase="prefill", batch=1, prompt=4096, ctx=8192)


def _ex():
    spec = importlib.util.spec_from_file_location("chiplet_example", os.path.join(ROOT, "scripts", "chiplet_example.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _flux(p):
    return p["W"] / (p["die_mm2"] * p["dies"])


# ---------------------------------------------------------------- (1) thermal audit
def test_limits_follow_the_shipped_envelope_rule():
    parts = {p["part"]: p for p in KNOWN_PARTS}
    air = [_flux(p) for p in KNOWN_PARTS if p["cooling"] == "air" and p["W"] and p["die_mm2"]]
    liq = [_flux(p) for p in KNOWN_PARTS if p["cooling"] == "liquid" and p["W"] and p["die_mm2"]]
    # air flux = the highest shipped air part (H100 / H200 SXM), above HIR's 0.55 thermal-budget figure
    assert abs(max(air) - 700 / 814) < 1e-12 and COOLING["air_hgx"]["flux_W_mm2"] == COOLING["air_oam"]["flux_W_mm2"] == 0.86
    assert max(air) <= 0.86 + 1e-9 and 0.86 > 0.55
    # liquid: card-level = the highest shipped liquid part (MI355X 0.96, 0.315); HIR 1.70 is the die-heat limit
    assert abs(max(liq) - 1400 / (2 * 732.2)) < 1e-12 and COOLING["liquid"]["flux_W_mm2"] == round(max(liq), 2)
    assert COOLING["liquid"]["die_flux_W_mm2"] == 1.70
    # package power: the largest shipped part per cooling class
    assert COOLING["air_oam"]["pkg_W"] == parts["MI350X"]["W"] == 1000.0 > parts["Gaudi 3 HL-325L"]["W"]
    assert COOLING["liquid"]["pkg_W"] == parts["MI355X"]["W"] == 1400.0
    assert parts["TPU v5p / v6e"]["W"] is None and parts["Gaudi 3 HL-325L"]["die_mm2"] is None
    # every shipped air part passes its own preset (the limits are not below any shipped part)
    for p in KNOWN_PARTS:
        if p["cooling"] == "air" and p["W"] and p["die_mm2"]:
            assert thermal_report(p["W"], p["die_mm2"], p["dies"], 0.0, "air_oam", basis="package")["feasible"] is True, p["part"]


def test_card_level_flux_and_static_flag():
    # 0.315: as-is module power (basis "package") keeps the 0.314 card-level arithmetic
    t = thermal_report(504.25, 571.4, 1, 14.0, "air_hgx", basis="package")
    assert t["feasible"] is False and abs(t["flux_W_mm2"] - 504.25 / 571.4) < 1e-12
    assert abs(t["die_flux_W_mm2"] - 490.25 / 571.4) < 1e-12
    assert abs(t["flux_ratio"] - min(504.25 / 571.4 / 0.86, 490.25 / 571.4 / 0.55)) < 1e-12
    assert t["flux_basis"].startswith("封装功耗（含整卡静态与稳压器损耗")
    two = thermal_report(1200.0, 858.0, 2, 0.0, "liquid", basis="package")
    assert abs(two["flux_W_mm2"] - 1200 / 1716) < 1e-12 and two["feasible"] is True


def test_static_floor_reaches_the_report():
    ex = _ex()
    t = ex.table()
    b = {"chip_preset": "1P", "scenario": {"model": "qwen3-8b", "mem_id": H4, "serving": {"phase": "prefill", "batch": 1,
                                                                                         "prompt": 4096, "ctx": 8192}},
         "energy": {"energy_preset": "tpuv4i_7nm"}, "sensitivity": False}
    o = api.api_eval(b)["thermal"]
    assert o["static_included"] is True and o["static_basis"] == "floor" and o["notes"] == []     # 0.315: always on
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=H4, serving=PF)
    kw = dict(mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,))
    d = energy_design_search(s, t, ex.budget(), **kw)
    assert d["rows"][0]["thermal"]["static_basis"] == "floor" and d["rows"][0]["thermal"]["static_add_W"] > 0
    assert any("整卡静态底功耗" in n for n in d["notes"])
    ds = energy_design_search(s, replace(t, idle_frac_tdp=71.8 / 700, tdp_W=700.0), ex.budget(), **kw)
    th = ds["rows"][0]["thermal"]
    assert th["static_basis"] == "idle" and th["static_add_W"] == 0.0
    assert not any("整卡静态底功耗" in n for n in ds["notes"])


# ---------------------------------------------------------------- (2) shore_strict evidence
def test_tight_carries_the_precedent_note():
    s = Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=H8, serving=Serving(phase="decode", batch=64))
    r = shoreline_report(s)
    assert r["status"] == "tight" and r["precedent"] == TIGHT_PRECEDENT and "US 12,315,860" in TIGHT_PRECEDENT
    assert "precedent" not in shoreline_report(Scenario(model="qwen3-8b", chip=CHIPS["1P"], mem_id=H4))
    ex = _ex()
    d = energy_design_search(s, ex.table(), ex.budget(), mac_scales=(1.0,), sram_scales=(1.0,), sides=(128,),
                             shore_strict=True)
    assert any("US 12,315,860" in n for n in d["notes"])


# ---------------------------------------------------------------- (3) freshness
def test_ovisocr3_release():
    from accel_dse.core.catalog import MERGED, entries, get_model, unlisted_models
    from accel_dse.core.model import load_release
    L = {x["id"]: x for x in list_models()}
    assert "ovisocr3" not in L and MERGED["ovisocr3"] == "startlux-decision-v1-0.8b"
    assert any(u["id"] == "ovisocr3" for u in unlisted_models())
    idx = {e["id"]: e for e in entries()}
    a, b = load_release(idx["ovisocr3"]["hf_id"]), load_release(idx["startlux-decision-v1-0.8b"]["hf_id"])
    assert (a["params_llm"], a["params_vision"], a["params_mtp"]) == (b["params_llm"], b["params_vision"], 0)
    m = get_model("ovisocr3")
    assert m.vision is not None and m.vision.params == a["params_vision"] and m.vision.depth == 12
    r = evaluate(Scenario(model="ovisocr3", chip=CHIPS["1P"], mem_id=H4, serving=Serving(phase="decode", batch=8, ctx=4096)))
    assert r.fits and r.per_card > 0


def test_ui_strings():
    html, js = W.read(W.INDEX_HTML), W.read()
    assert "风冷 OAM 1000 W" in html and "风冷 OAM 600 W" not in html and "1.70 W/mm²" in html
    assert "die_flux_W_mm2" in js and "x.notes" in js and "flux_ratio" in js
