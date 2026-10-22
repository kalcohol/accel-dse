"""0.352: adversarial card/die ratio persist+UI; power-cap three-way binding; no visual UI."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from accel_dse.core.report import DIRECTION, design_report
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
    assert COOLING["liquid"]["pkg_W"] == 1400.0 and COOLING["liquid"]["flux_W_mm2"] == 0.96
    assert COOLING["liquid"]["die_flux_W_mm2"] == 1.70


def test_adversarial_ratio_persist_report_dse_ui():
    """0.351 persistence still holds: thermal/DSE/report/UI card axis; liquid headroom ~1.5%."""
    th = thermal_report(700.0, 800.0, dies=1, dram_dev_W=150.0, cooling="liquid", basis="package")
    assert th["card_flux_ratio"] is not None and th["die_flux_ratio"] is not None
    assert th["flux_ratio"] == min(th["card_flux_ratio"], th["die_flux_ratio"])
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None,
               cooling="liquid", power_scope="card")
    lim_f = COOLING["liquid"]["flux_W_mm2"]
    tr = o["rows"][0]["thermal"]
    assert abs(tr["card_flux_ratio"] - tr["flux_W_mm2"] / lim_f) < 1e-12
    rep = design_report(o, {}, 0)
    thsec = next(s for s in rep["sections"] if s["id"] == "thermal")
    it = {i["key"]: i for i in thsec["items"]}
    assert "可行性" in it["flux_ratio"]["label"]
    assert it["flux_headroom"]["value"] < 0.05
    js = W.read()
    assert "card_flux_ratio" in W.idents(js)
    assert W.any_string_contains(js, "卡级") and W.any_string_contains(js, "可行性取 min")
    assert not W.has(js, "· ${x.flux_ratio.toFixed(2)}× 上限")


def test_static_build_copies_card_axis_ui():
    """Static path = live web app.js (build_static copies WEB/app.js). Pages @ 0.259 may lag."""
    js = W.read()
    assert "card_flux_ratio" in W.idents(js)
    assert W.any_string_contains(js, "卡级") and W.any_string_contains(js, "可行性取 min")
    # build_static recipe still copies from WEB (not a vendored stale blob)
    body = (ROOT / "scripts" / "build_static.py").read_text()
    assert "copy2" in body and "app.js" in body and "WEB" in body


def test_three_way_binding_power_cap_changes_decision():
    """liquid+package: power_cap binds (was 热流-only); liquid+card: still 热流."""
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None,
               cooling="liquid", power_scope="package")
    assert o.get("power_cap_W") and o.get("power_scope") == "package"
    rep = design_report(o, {}, 0)
    th = next(s for s in rep["sections"] if s["id"] == "thermal")
    it = {i["key"]: i["value"] for i in th["items"]}
    assert it["thermal_binding"] == "功耗预算"
    assert it["cap_ratio"] is not None and it["cap_ratio"] >= it["card_flux_ratio"] - 1e-9
    assert any(m["what"] == "功耗预算" for m in (rep.get("margins") or []))
    # card scope: flux still tighter than W_card/cap
    o2 = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None,
                cooling="liquid", power_scope="card")
    rep2 = design_report(o2, {}, 0)
    th2 = next(s for s in rep2["sections"] if s["id"] == "thermal")
    it2 = {i["key"]: i["value"] for i in th2["items"]}
    assert it2["thermal_binding"] == "热流"
    assert it2["flux_headroom"] < 0.05


def test_direction_cap_fields():
    assert DIRECTION["cap_margin_W"] == 1 and DIRECTION["cap_headroom"] == 1
    assert DIRECTION["cap_ratio"] == -1


def test_changelog_352():
    assert "## [0.352.0]" in CL
    sec = CL.split("## [0.352.0]")[1].split("## [")[0]
    assert "binding" in sec.lower() or "约束" in sec or "功耗预算" in sec
    assert "card" in sec.lower() or "卡级" in sec
    assert "visual" in sec.lower() or "UI" in sec or "视觉" in sec
