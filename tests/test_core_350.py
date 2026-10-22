"""0.350: adversarial pkg/flux margins; card-axis binding/headroom; no visual UI."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from accel_dse.core.report import design_report, to_markdown
from accel_dse.core.thermal import COOLING

ROOT = Path(__file__).resolve().parents[1]
CL = (ROOT / "CHANGELOG.md").read_text()


def _script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_card_axis_headroom_and_binding():
    ex = _script("chiplet_example")
    for cool in ("air_oam", "liquid"):
        o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None, cooling=cool, power_scope="card")
        lim_f = COOLING[cool]["flux_W_mm2"]
        lim_p = COOLING[cool]["pkg_W"]
        assert lim_f == COOLING[cool]["flux_W_mm2"] and lim_p == COOLING[cool]["pkg_W"]  # not retuned
        rep = design_report(o, {}, 0)
        th = next(s for s in rep["sections"] if s["id"] == "thermal")
        it = {i["key"]: i["value"] for i in th["items"]}
        card_r = it["flux_W_mm2"] / lim_f
        pr = it["pkg_W"] / lim_p
        assert abs(it["flux_margin_W_mm2"] - (lim_f - it["flux_W_mm2"])) < 1e-12
        assert abs(it["flux_headroom"] - (1 / card_r - 1)) < 1e-9
        assert abs(it["card_flux_ratio"] - card_r) < 1e-12
        assert abs(it["pkg_margin_W"] - (lim_p - it["pkg_W"])) < 1e-9
        assert it["thermal_binding"] == ("热流" if card_r >= pr else "封装功耗")
        # liquid: feas min(card,die) << card → must not drive headroom/binding
        if cool == "liquid":
            assert it.get("feas_flux_ratio") is not None
            assert it["feas_flux_ratio"] < card_r - 0.2
            assert it["flux_headroom"] < 0.05  # ~1.5%, not ~140%
            assert it["thermal_binding"] == "热流"
        ft = th["front_thermal"]
        tie = o.get("tie") or 0.01
        for p in ft["points"]:
            cr = p["flux_W_mm2"] / lim_f
            assert abs(p["margin"] - (1 - cr)) < 1e-12
            assert abs(p["headroom"] - (1 / cr - 1)) < 1e-12
            assert p["within_tie"] is ((1 - cr) <= tie)
            assert abs(p["pkg_margin_W"] - (lim_p - p["pkg_W"])) < 1e-6


def test_no_cooling_omits_limit_margins():
    ex = _script("chiplet_example")
    o = ex.run("qwen3-8b", "prefill", "tp", 0, None, out=lambda *_: None, cooling=None, power_scope="card")
    rep = design_report(o, {}, 0)
    th = next(s for s in rep["sections"] if s["id"] == "thermal")
    keys = {i["key"] for i in th["items"] if i["value"] is not None}
    assert "pkg_margin_W" not in keys and "thermal_binding" not in keys
    assert th.get("front_thermal") is None


def test_cooling_limits_not_retuned():
    assert COOLING["air_oam"]["pkg_W"] == 1000.0 and COOLING["air_oam"]["flux_W_mm2"] == 0.86
    assert COOLING["liquid"]["pkg_W"] == 1400.0 and COOLING["liquid"]["flux_W_mm2"] == 0.96


def test_changelog_350():
    assert "## [0.350.0]" in CL
    sec = CL.split("## [0.350.0]")[1].split("## [")[0]
    assert "card" in sec.lower() or "卡级" in sec
    assert "binding" in sec.lower() or "紧侧" in sec or "thermal_binding" in sec
    assert "visual" in sec.lower() or "UI" in sec or "视觉" in sec
