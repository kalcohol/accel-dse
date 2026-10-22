"""0.351: adversarial card-axis across report/DSE/UI; card_flux_ratio persistence; no visual UI."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from accel_dse.core.report import design_report, DIRECTION
from accel_dse.core.thermal import COOLING, thermal_report
from tests import webstruct as W

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_cooling_limits_not_retuned():
    assert COOLING["air_oam"]["pkg_W"] == 1000.0 and COOLING["air_oam"]["flux_W_mm2"] == 0.86
    assert COOLING["air_oam"]["die_flux_W_mm2"] == 0.55
    assert COOLING["liquid"]["pkg_W"] == 1400.0 and COOLING["liquid"]["flux_W_mm2"] == 0.96
    assert COOLING["liquid"]["die_flux_W_mm2"] == 1.70


def test_thermal_report_exposes_card_die_ratios():
    # synthetic: card flux near liquid limit, die flux far below (classic liquid skew)
    th = thermal_report(700.0, 800.0, dies=1, dram_dev_W=150.0, cooling="liquid",
                        static_W=20.0, basis="package")
    lim_f = COOLING["liquid"]["flux_W_mm2"]
    lim_d = COOLING["liquid"]["die_flux_W_mm2"]
    assert th["card_flux_ratio"] is not None and th["die_flux_ratio"] is not None
    assert abs(th["card_flux_ratio"] - th["flux_W_mm2"] / lim_f) < 1e-12
    assert abs(th["die_flux_ratio"] - th["die_flux_W_mm2"] / lim_d) < 1e-12
    assert th["flux_ratio"] == min(th["card_flux_ratio"], th["die_flux_ratio"])
    assert th["die_flux_ratio"] < th["card_flux_ratio"] - 0.2


def test_report_and_dse_card_axis_not_feas_headroom():
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None,
               cooling="liquid", power_scope="card")
    lim_f = COOLING["liquid"]["flux_W_mm2"]
    rep = design_report(o, {}, 0)
    thsec = next(s for s in rep["sections"] if s["id"] == "thermal")
    it = {i["key"]: i for i in thsec["items"]}
    assert "可行性" in it["flux_ratio"]["label"]
    card_r = it["flux_W_mm2"]["value"] / lim_f
    assert abs(it["card_flux_ratio"]["value"] - card_r) < 1e-12
    assert abs(it["flux_headroom"]["value"] - (1 / card_r - 1)) < 1e-9
    assert it["flux_headroom"]["value"] < 0.05  # ~1.5%, not ~140% from feas
    assert it["flux_ratio"]["value"] < card_r - 0.2
    # DSE row persists design ratios
    row = o["rows"][0]
    tr = row["thermal"]
    assert "card_flux_ratio" in tr and "die_flux_ratio" in tr
    assert abs(tr["card_flux_ratio"] - tr["flux_W_mm2"] / lim_f) < 1e-12
    assert tr["flux_ratio"] == min(tr["card_flux_ratio"], tr["die_flux_ratio"])
    # old feas-min headroom would be ~1.4 — must not match report headroom
    feas_hr = 1 / tr["flux_ratio"] - 1
    assert feas_hr > 1.0
    assert abs(it["flux_headroom"]["value"] - feas_hr) > 0.5


def test_direction_pkg_and_card_ratio():
    assert DIRECTION["pkg_margin_W"] == 1 and DIRECTION["pkg_headroom"] == 1
    assert DIRECTION["card_flux_ratio"] == -1
    assert DIRECTION["flux_headroom"] == 1


def test_ui_card_axis_strings():
    js = W.read()
    assert "card_flux_ratio" in W.idents(js)
    assert W.any_string_contains(js, "卡级") and W.any_string_contains(js, "可行性取 min")
    # old feas-as-design chip-panel display gone (must not be sole ×上限 without 卡级)
    assert not W.has(js, "· ${x.flux_ratio.toFixed(2)}× 上限")
    assert W.has(js, "x.card_flux_ratio != null ? x.card_flux_ratio : x.flux_ratio")


def test_changelog_351():
    assert "## [0.351.0]" in CL
    sec = CL.split("## [0.351.0]")[1].split("## [")[0]
    assert "card" in sec.lower() or "卡级" in sec
    assert "DSE" in sec or "dse" in sec or "UI" in sec or "界面" in sec
    assert "visual" in sec.lower() or "UI" in sec or "视觉" in sec
    assert "NPU" in sec or "npu" in sec
